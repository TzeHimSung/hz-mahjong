"""Recorder adapters: validation, durable events, and final publication."""
from __future__ import annotations

import copy
import json
import sys
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from heuristic_ai.artifacts.cleanup import CleanupResult, cleanup_after_export
from heuristic_ai.artifacts.export import RoundNames, merge_store, validate_round_payload
from heuristic_ai.artifacts.files import OwnedArtifactFile
from heuristic_ai.artifacts.projection import ParticipantProjection, Projection, projection_type
from heuristic_ai.artifacts.store import ArtifactStore, PARTICIPANT_MODES, WorkDirectoryCollision, legacy_line
from heuristic_ai.artifacts.models import ArtifactStatus, CleanupStatus, DecisionTrace, RunResult, RunStatus
from heuristic_ai.artifacts.recovery import RECOVERY_FIELDS, plain_json, positive_generation


class RecorderBase:
    def bind_before_publish(self, callback: Callable[[], None]) -> None:
        """Flush a run-owned evidence companion before sealing its main archive."""
        self._before_publish = callback
    _poll_fields: frozenset[str] = frozenset()

    def _initialize_storage(self, output_root: Path, mode: str, manifest: dict[str, Any],
                            *, record_subdir: str | None = None) -> None:
        self.mode = mode
        projector = projection_type(mode)
        self._lock = threading.RLock()
        self._activity = threading.Condition()
        self._active = 0
        self._closing = False
        self._final_result: RunResult | None = None
        self._names = RoundNames()
        while True:
            try:
                self._store = ArtifactStore(output_root, mode, self._serializer.redact_tree(manifest),
                                            record_subdir=record_subdir)
                break
            except WorkDirectoryCollision:
                self.run_id = self.run_id.rsplit("_", 1)[0] + "_" + uuid.uuid4().hex[:8]
                manifest["run_id"] = self.run_id
        self._projection: Projection = projector(manifest, now=self._now)
        self.run_dir = self._store.default_output
        self.manifest_path = self.run_dir / "manifest.json"
        self.decisions_path = self.run_dir / "decisions.jsonl"
        if mode == "train":
            self.rounds_dir = self.run_dir / "rounds"
            self.state_polls_path = self.run_dir / "state_polls.jsonl"
            try:
                self._store.prepare("replay")
            except BaseException:
                self._store.close()
                raise

    @property
    def _manifest(self) -> dict[str, Any]:
        return self._projection._manifest

    @property
    def _rounds_per_game(self) -> int | None:
        return self._projection._rounds_per_game

    @property
    def work_dir(self) -> Path | None:
        return None if self._final_result and self._final_result.cleanup_status is CleanupStatus.CLEANED else self._store.work_dir

    @property
    def audit_failed(self) -> bool:
        return self._store.failed.is_set()

    def bind_stop(self, stop: threading.Event) -> None:
        self._store.bind_stop(stop)

    def bind_health_progress(self, progress: Callable[[str], None]) -> None:
        """Attach optional execution checkpoints without importing orchestration."""
        self._health_progress = progress
        self._store.bind_health_progress(progress)

    def _health_checkpoint(self, label: str) -> None:
        progress = getattr(self, "_health_progress", None)
        if progress is not None:
            progress(label)

    def ensure_audit_healthy(self) -> None:
        if self.audit_failed:
            raise self._safe_storage_error(self._store.first_error or RuntimeError("audit persistence failed")) from None

    @contextmanager
    def _recording(self) -> Iterator[None]:
        with self._activity:
            if self._closing:
                raise RuntimeError("cannot record an event after run completion")
            self._active += 1
        try:
            yield
        finally:
            with self._activity:
                self._active -= 1
                self._activity.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._serializer.redact_tree(copy.deepcopy(self._manifest))

    def progress(self) -> dict[str, Any]:
        with self._lock:
            counts = self._manifest["decision_counts"].copy()
            games = len(self._manifest["game_ids"])
            status = self._manifest["status"]
            rounds = len(self._manifest.get("rounds", []))
            recoveries = [{"game_id": game["game_id"], **copy.deepcopy(game["recovery"])}
                          for game in self._manifest["games"] if "recovery" in game]
        return {"run_id": self.run_id, "status": status, "games": games,
                "archived_rounds": rounds, "decisions": counts,
                "recoveries": self._serializer.redact_tree(recoveries),
                "recovering_games": sum(row["state"] in {"failure", "restart_scheduled", "restarted"} for row in recoveries),
                **self._store.metrics()}

    def _safe_storage_error(self, error: BaseException) -> BaseException:
        return error

    def _error_text(self, error: BaseException) -> str:
        if self.mode in PARTICIPANT_MODES:
            detached = self._safe_storage_error(error)
            return str(detached)
        return self._serializer.sanitize(str(error))

    def _failure_arguments(self, error: BaseException) -> dict[str, Any]:
        return {"error": self._error_descriptor(error) if self.mode in PARTICIPANT_MODES else self._error_text(error)}

    def _arguments(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        arguments = dict(arguments)
        if arguments.get("participant_ranking") is not None:
            arguments["participant_ranking"] = ParticipantProjection._ranking(arguments["participant_ranking"])
        if "final_scores" in arguments:
            arguments["final_scores"] = ParticipantProjection._scores(arguments["final_scores"])
        if "game_ids" in arguments:
            arguments["game_ids"] = list(arguments["game_ids"])
        return arguments

    def _control(self, operation: str, **arguments: Any) -> None:
        with self._recording():
            with self._lock:
                at = self._now().isoformat()
                arguments = self._arguments(operation, arguments)
                # A repeated seat notification must still validate identities,
                # but has no durable event and need not copy the growing manifest.
                if (self.mode == "train" and operation == "identify_player"
                        and not self._projection._player_identity_changes(**arguments)):
                    return
                prepared = self._projection.clone()
                prepared.control(operation, arguments, at)
                if prepared.revision == self._projection.revision:
                    return
                data = {"operation": operation, "arguments": self._serializer.redact_tree(arguments)}
                failure = None
                try:
                    self._store.append("control", (), "control", data, at)
                except BaseException as error:
                    failure = self._safe_storage_error(error)
                if failure is not None:
                    raise failure from None
                self._projection = prepared
            if operation in {"register_games", "register_game", "identify_participant"}:
                self._prepare_workers()

    def _prepare_workers(self) -> None:
        # Train discovers game IDs after play starts. Let append() create each
        # shard on its first event instead of delaying every initial state read.
        if self.mode == "train":
            return
        with self._lock:
            manifest = self._serializer.redact_tree(self._manifest)
            user_id = manifest["participant"]["user_id"]
            owners = [(game_id, user_id) for game_id in manifest["game_ids"] if isinstance(user_id, str)]
        for owner in owners:
            self._store.prepare("decision", owner)

    def record_decision(self, trace: DecisionTrace) -> None:
        with self._recording():
            data = self._trace_json(trace)
            encoded = legacy_line(data)
            # The live validator uses actual identities; only redacted payloads
            # cross the persistence boundary, as in the legacy serializers.
            live = dict(data)
            live.update(game_id=trace.game_id, player_name=trace.player_name,
                        user_id=trace.user_id, fallback_reason=trace.fallback_reason)
            live["decision"] = {**data["decision"], "details": trace.decision.details}
            with self._lock:
                self._projection.validate_decision(live)
            failure = None
            try:
                self._store.append("decision", (data["game_id"], data["user_id"]),
                                   "decision", data, data["recorded_at"], encoded_payload=encoded[:-1])
            except BaseException as error:
                failure = self._safe_storage_error(error)
            if failure is not None:
                raise failure from None
            with self._lock:
                self._projection.apply_decision(live, len(encoded))

    def record_state_poll(self, event: Mapping[str, Any]) -> None:
        if self.mode != "train":
            raise ValueError("race artifacts do not contain state polls")
        if not isinstance(event, Mapping):
            raise ValueError("state poll must be a mapping")
        projected = {key: event[key] for key in self._poll_fields if key in event}
        if "generation" in projected:
            positive_generation(projected["generation"])
        for key, value in projected.items():
            if key == "responding_seats" and isinstance(value, (tuple, list)):
                if any(type(seat) is not int or seat not in range(4) for seat in value):
                    raise ValueError("state poll responding seats are invalid")
            elif value is not None and type(value) not in {str, int, float, bool}:
                raise ValueError("state poll values must be scalar projections")
        with self._recording():
            at = self._now().isoformat()
            payload = self._serializer.redact_tree({"schema_version": 1, "recorded_at": at, **projected})
            owner = (payload.get("game_id"), payload.get("user_id"))
            if any(not isinstance(value, str) or not value for value in owner):
                owner = ()
            self._store.append("poll", owner, "poll", payload, at, sync=False)
            with self._lock:
                self._projection.record_poll(payload)

    def record_game_recovery(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping) or set(event) - RECOVERY_FIELDS:
            raise ValueError("recovery event has unsupported fields")
        payload = plain_json(event)
        payload.update(schema_version=1, recorded_at=self._now().isoformat(), run_id=self.run_id)
        if isinstance(payload.get("reason"), str):
            # Summary clipping must happen after redaction, so live and offline
            # projections agree even when replacing a token changes its length.
            payload["reason"] = self._serializer.sanitize(payload["reason"])
        # _control validates a detached projection before its synchronous append.
        # Only redacted evidence crosses the durable persistence boundary.
        self._control("record_game_recovery", event=payload)

    def record_round_bundle(self, bundle: Any) -> None:
        if self.mode != "train":
            raise ValueError("race artifacts cannot contain full round replays")
        with self._recording(), self._lock:
            names = copy.deepcopy(self._names)
            relative, fresh = names.assign(bundle.directory_name, bundle.record, bundle.events)
            if not fresh:
                return
            payload = {"relative": relative, "record": bundle.record, "events": bundle.events}
            validate_round_payload(payload)
            prepared = self._projection.clone()
            prepared.apply_round(relative, bundle.record)
            self._store.append("replay", (), "round_bundle", self._serializer.redact_tree(payload), self._now().isoformat())
            self._projection = prepared
            self._names = names

    def record_archive_failure(self, game_id: str, failure: dict[str, Any], raw: Any = None) -> None:
        from heuristic_ai.artifacts.replay import _validate_identifier
        _validate_identifier(game_id)
        with self._recording():
            payload = self._serializer.redact_tree({"relative": f"rounds/failed/{game_id}", "failure": failure, "raw": raw})
            self._store.append("replay", (), "archive_failure", payload, self._now().isoformat())

    def add_round(self, round_dir: Path) -> None:
        """Read-only compatibility import; the live archiver sends RoundBundle."""
        from heuristic_ai.artifacts.replay import RoundBundle
        resolved = round_dir.resolve()
        if not resolved.is_relative_to(self.run_dir.resolve()):
            raise ValueError("round artifact must be inside the run directory")
        try:
            with OwnedArtifactFile(resolved / "record.json", existing=True,
                                   ownership_root=self.run_dir, shared_read=True) as source:
                with source.reader() as stream:
                    record = json.load(stream)
            events = []
            if (resolved / "events.jsonl").exists():
                with OwnedArtifactFile(resolved / "events.jsonl", existing=True,
                                       ownership_root=self.run_dir, shared_read=True) as source:
                    with source.reader() as stream:
                        events = [json.loads(line) for line in stream]
        except (OSError, ValueError) as error:
            raise ValueError("round artifact must contain valid record.json") from error
        self.record_round_bundle(RoundBundle(resolved.name, record, events))

    def _finish(self, operation: str, **arguments: Any) -> RunResult:
        before_publish = getattr(self, "_before_publish", None)
        if before_publish is not None:
            self._before_publish = None
            before_publish()
        if self._final_result is not None:
            return self._final_result
        at = self._now().isoformat()
        self._health_checkpoint("artifact.prepare_terminal")
        arguments = self._arguments(operation, arguments)
        # Invalid completion requests remain correctable; do not seal the store.
        with self._lock:
            prepared = self._projection.clone()
            prepared.control(operation, arguments, at)
        with self._activity:
            self._closing = True
            while self._active:
                self._activity.wait()
        self._health_checkpoint("artifact.synchronize")
        terminal_recorded = False
        try:
            self._store.synchronize()
        except BaseException as error:
            self._store._fail(error)
        if self._store.failed.is_set() and operation == "complete":
            operation = "fail"
            arguments = self._failure_arguments(self._store.first_error or RuntimeError("audit persistence failed"))
        with self._lock:
            prepared = self._projection.clone()
            prepared.control(operation, arguments, at)
            try:
                data = {"operation": operation, "arguments": self._serializer.redact_tree(arguments),
                        "inventory": self._store.inventory()}
                self._store.append("control", (), "control", data, at)
                terminal_recorded = True
            except BaseException as error:
                self._store._fail(error)
                prepared = self._projection.clone()
                prepared.control("fail", self._failure_arguments(error), at)
            self._projection = prepared
        self._store.seal()
        self._health_checkpoint("artifact.merge")
        artifact_error = None
        cleanup = CleanupResult("retained", (self._store.work_dir,))
        artifact_status = ArtifactStatus.INCOMPLETE
        status = RunStatus(self._manifest["status"])
        try:
            health_options = ({"on_progress": self._health_progress}
                              if getattr(self, "_health_progress", None) is not None else {})
            exported = merge_store(self._store, expected_manifest=self.snapshot() if terminal_recorded else None,
                                   **health_options)
            self.run_dir = exported.artifact_dir
            self.manifest_path = self.run_dir / "manifest.json"
            self.decisions_path = self.run_dir / "decisions.jsonl"
            if self.mode == "train":
                self.rounds_dir = self.run_dir / "rounds"
                self.state_polls_path = self.run_dir / "state_polls.jsonl"
            status = RunStatus(exported.manifest["status"])
            artifact_status = ArtifactStatus.READY
            self._health_checkpoint("artifact.cleanup")
            try:
                cleanup = cleanup_after_export(self._store, exported)
            except Exception as error:
                cleanup = CleanupResult("pending", (self._store.work_dir,), self._error_text(error))
        except Exception as error:
            artifact_error = self._error_text(error)
        finally:
            self._health_checkpoint("artifact.close")
            try:
                self._store.close()
            except Exception as error:
                if artifact_error is None and cleanup.status != "cleaned":
                    cleanup = CleanupResult("pending", (self._store.work_dir,), self._error_text(error))
        self._final_result = RunResult(
            status=status, run_id=self.run_id, artifact_dir=self.run_dir,
            game_ids=tuple(self._manifest["game_ids"]),
            work_dir=None if cleanup.status == "cleaned" else self._store.work_dir,
            artifact_status=artifact_status, artifact_error=artifact_error,
            cleanup_status=CleanupStatus(cleanup.status), remaining_temp_paths=cleanup.remaining,
            cleanup_error=cleanup.error,
            mode="match" if self.mode == "match" else None,
        )
        self._health_checkpoint("artifact.complete")
        return self._final_result

    def close(self) -> None:
        """Release handles without deleting evidence or manufacturing a finish."""
        primary = sys.exception()
        try:
            self._store.close()
        except BaseException:
            if primary is None:
                raise

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
