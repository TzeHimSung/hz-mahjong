"""Merge immutable shard prefixes into a fresh legacy-compatible export."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from heuristic_ai.artifacts.files import OwnedArtifactFile, checked_path
from heuristic_ai.artifacts.projection import MatchProjection, TERMINAL_OPERATIONS, projection_type
from heuristic_ai.artifacts.recovery import recovery_log_path, verify_recovery_log
from heuristic_ai.artifacts.store import ArtifactStore, FrameLocation, PARTICIPANT_MODES, encode_frame, json_bytes, legacy_line

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def relative_artifact_path(value: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("invalid relative artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} or not _SAFE_COMPONENT.fullmatch(part) for part in path.parts):
        raise ValueError("invalid relative artifact path")
    if path.as_posix() != value:
        raise ValueError("artifact paths must use their normalized spelling")
    return value


class RoundNames:
    def __init__(self) -> None:
        self.by_identity: dict[tuple[str, int], tuple[str, str]] = {}
        self.reserved: set[str] = set()

    def assign(self, name: str, record: dict[str, Any], events: list[dict[str, Any]]) -> tuple[str, bool]:
        relative_artifact_path(name)
        if "/" in name:
            raise ValueError("round name must be one path component")
        identity = (record["game_id"], record["round_no"])
        signature = hashlib.sha256(json.dumps([record, events], ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if identity in self.by_identity:
            previous_name, previous_signature = self.by_identity[identity]
            if previous_signature != signature:
                raise ValueError("round replay verification failed: conflicting round data")
            return previous_name, False
        suffix = f"{name}_round_{record['round_no']}"
        choices = [name, suffix]
        number = 2
        while True:
            for candidate in choices:
                if candidate not in self.reserved:
                    relative = f"rounds/{candidate}"
                    self.reserved.add(candidate)
                    self.by_identity[identity] = (relative, signature)
                    return relative, True
            choices = [f"{suffix}_{number}"]
            number += 1


def validate_round_payload(payload: dict[str, Any]) -> None:
    relative = relative_artifact_path(payload.get("relative"))
    if len(PurePosixPath(relative).parts) != 2 or not relative.startswith("rounds/"):
        raise ValueError("round bundle must belong to the rounds directory")
    record, events = payload.get("record"), payload.get("events")
    if not isinstance(record, dict) or record.get("schema_version") != 3:
        raise ValueError("round bundle must use record schema 3")
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("round bundle events must be objects")
    if "blocks" in record:
        blocks = record["blocks"]
        if not isinstance(blocks, list) or any(not isinstance(b, dict) or not isinstance(b.get("events"), list) for b in blocks):
            raise ValueError("round replay blocks are invalid")
        if [event for block in blocks for event in block["events"]] != events:
            raise ValueError("round replay events disagree with its blocks")
        from heuristic_ai.protocol.compatibility import round_ended_metadata
        for event in events:
            if event.get("type") == "round_ended":
                round_ended_metadata(event, expected_round_no=record["round_no"])
        from heuristic_ai.artifacts.replay import _draw_to_action, _round_outcome
        outcome = _round_outcome(events, record["result"], result_source=record["result_source"], players=record["players"])
        if outcome != record["outcome"]:
            raise ValueError("round replay outcome verification failed")
        timings = _draw_to_action(events, record["players"],
                                 terminal_winner_seat=outcome["winner"]["seat"] if outcome["kind"] == "win" else None,
                                 terminal_hu_absent=not any(event.get("type") == "hu" for event in events))
        if timings != record["draw_to_action"]:
            raise ValueError("round replay timing verification failed")


@dataclass(frozen=True, slots=True)
class ExportResult:
    artifact_dir: Path
    manifest: dict[str, Any]
    files: dict[str, dict[str, Any]]
    warnings: tuple[str, ...]
    cleanup_eligible: bool


class ExportWriter:
    def __init__(self, store: ArtifactStore, output_dir: Path | None = None, *,
                 on_progress: Callable[[str], None] | None = None) -> None:
        self.store = store
        self._on_progress = on_progress
        self.files: dict[str, OwnedArtifactFile] = {}
        self.directories: dict[str, OwnedArtifactFile] = {}
        self.summaries: dict[str, dict[str, Any]] = {}
        self._hashes: dict[str, Any] = {}
        self._lengths: dict[str, int] = {}
        self.job: OwnedArtifactFile | None = None
        self._job_sequence = 0
        self.published = False
        parent = store.final_parent if output_dir is None else Path(os.path.abspath(output_dir)).parent
        checked_path(parent, parent, allow_root=True)
        parent.mkdir(parents=True, exist_ok=True)
        if output_dir is None:
            target = store.default_output
            while target.exists():
                target = parent / f"{store.run_id}_export_{uuid.uuid4().hex[:12]}"
        else:
            target = Path(os.path.abspath(output_dir))
        if target == store.work_dir or target.is_relative_to(store.work_dir) or store.work_dir.is_relative_to(target):
            raise ValueError("final export and working directory must be disjoint")
        self.path = target
        try:
            self.job = OwnedArtifactFile(store.work_dir / f"attempt-{uuid.uuid4().hex}.jsonl",
                                         temporary=True, ownership_root=store.work_dir)
            self.job.bind_health_progress(on_progress)
            store.auxiliary.append(self.job)
            self._job("export_attempt", {"run_id": store.run_id, "mode": store.mode,
                       "output_dir": str(target), "work_dir": str(store.work_dir),
                       "file_identity": self.job.file_identity(),
                       "directory_identity": None})
            # Durable intent precedes directory creation. A crash in the gap
            # before its identity is recorded remains visible as pending work.
            try:
                self.directories[""] = OwnedArtifactFile.create_directory(target, ownership_root=parent)
            except FileExistsError:
                self._job("rejected", {"reason": "directory_exists"})
                raise
            self._job("directory_owned", {"file_identity": self.directories[""].file_identity()})
        except BaseException:
            self.close()
            raise

    def _job(self, kind: str, payload: dict[str, Any]) -> None:
        assert self.job is not None
        self.job.append(encode_frame({"format_version": 1, "seq": self._job_sequence,
                                      "order": self._job_sequence, "kind": kind,
                                      "payload": payload}))
        self._job_sequence += 1

    def _ensure_directory(self, relative: str) -> None:
        if not relative or relative in self.directories:
            return
        parent = PurePosixPath(relative).parent.as_posix()
        self._ensure_directory("" if parent == "." else parent)
        path = self.path / relative
        directory = OwnedArtifactFile.create_directory(path, ownership_root=self.path)
        self.directories[relative] = directory
        self._job("directory_created", {"path": relative, "file_identity": directory.file_identity()})

    def create(self, relative: str) -> OwnedArtifactFile:
        relative_artifact_path(relative)
        if relative in self.files or relative in self.summaries:
            raise ValueError("export file would be overwritten")
        parent = PurePosixPath(relative).parent.as_posix()
        self._ensure_directory("" if parent == "." else parent)
        file = OwnedArtifactFile(self.path / relative, ownership_root=self.path)
        file.bind_health_progress(self._on_progress)
        self.files[relative] = file
        self._hashes[relative] = hashlib.sha256()
        self._lengths[relative] = 0
        self._job("file_created", {"path": relative, "file_identity": file.file_identity()})
        return file

    def append(self, relative: str, data: bytes) -> None:
        file = self.files.get(relative)
        if file is None:
            file = self.create(relative)
        file.append(data, sync=False)
        self._hashes[relative].update(data)
        self._lengths[relative] += len(data)

    def finish_file(self, relative: str, *, retain: bool = False) -> None:
        file = self.files[relative]
        file.seal()
        expected = self._hashes[relative].hexdigest()
        if file.size() != self._lengths[relative] or file.digest() != expected:
            raise ValueError("export readback verification failed")
        self.summaries[relative] = {"size": file.size(), "sha256": expected,
                                    "file_identity": file.file_identity()}
        if not retain:
            file.close()
            del self.files[relative]

    def write_json(self, relative: str, value: Any) -> None:
        self.append(relative, json_bytes(value, pretty=True) + b"\n")
        self.finish_file(relative, retain=True)

    def write_jsonl(self, relative: str, values: list[Any]) -> None:
        self.create(relative)
        for value in values:
            self.append(relative, json_bytes(value) + b"\n")
        self.finish_file(relative, retain=True)

    def publish(self, manifest: dict[str, Any]) -> None:
        for name in tuple(self.files):
            self.finish_file(name, retain=True)
        self.append("manifest.json", json_bytes(manifest, pretty=True) + b"\n")
        self.finish_file("manifest.json", retain=True)
        # Persist the positive publication certificate before releasing manifest.
        self._job("committed", {"files": self.summaries})
        assert self.job is not None
        self.job.seal()
        for name in tuple(self.files):
            if name != "manifest.json":
                self.files[name].close()
                del self.files[name]
        self.files["manifest.json"].close()
        del self.files["manifest.json"]
        for relative in sorted(self.directories, key=lambda value: value.count("/"), reverse=True):
            self.directories[relative].temporary = False
            self.directories[relative].close()
        self.published = True

    def close(self) -> None:
        for file in [*self.files.values(), *reversed(tuple(self.directories.values()))]:
            try:
                file.close()
            except BaseException:
                pass


def _normalized_manifest(value: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(value)
    value["degradations"] = sorted(value["degradations"], key=lambda item: json.dumps(item, sort_keys=True))
    return value


def merge_store(store: ArtifactStore, *, output_dir: Path | None = None,
                expected_manifest: dict[str, Any] | None = None,
                recovery: bool = False,
                on_progress: Callable[[str], None] | None = None) -> ExportResult:
    def checkpoint(label: str) -> None:
        if on_progress is not None:
            on_progress(label)

    checkpoint("artifact.read_initialization")
    locations = iter(store.locations(recovery=recovery))
    first = next(locations, None)
    if first is None:
        raise ValueError("artifact store has no initialization event")
    first_body = first.read()
    if first.stream_kind != "control" or first_body.get("kind") != "init":
        raise ValueError("artifact initialization must be the first event")
    initial = first_body.get("payload", {}).get("manifest")
    if not isinstance(initial, dict) or initial.get("run_id") != store.run_id or initial.get("status") != "running":
        raise ValueError("invalid artifact initialization manifest")
    if store.mode in PARTICIPANT_MODES:
        if initial.get("schema_version") != 1 or initial.get("artifact_type") != f"{store.mode}_run":
            raise ValueError(f"{store.mode} initialization schema mismatch")
        if store.mode == "match" and initial.get("mode") != "match":
            raise ValueError("match initialization mode mismatch")
    else:
        if initial.get("schema_version") != 2 or "artifact_type" in initial:
            raise ValueError("train initialization schema mismatch")
    projection = projection_type(store.mode)(copy.deepcopy(initial), public=True)
    writer = ExportWriter(store, output_dir, **({"on_progress": on_progress} if on_progress else {}))
    terminal: tuple[FrameLocation, dict[str, Any]] | None = None
    round_digests: dict[str, str] = {}
    failures: list[tuple[str, FrameLocation]] = []
    failure_names: dict[str, int] = {}
    try:
        writer.create("decisions.jsonl")
        if store.mode == "train":
            writer.create("state_polls.jsonl")
            writer._ensure_directory("rounds")
        for location in locations:
            body = location.read()
            checkpoint("artifact.read_event")
            payload = body.get("payload")
            kind = body.get("kind")
            if not isinstance(payload, dict):
                raise ValueError("artifact event payload is invalid")
            if kind == "control" and location.stream_kind == "control":
                operation = payload.get("operation")
                if operation in TERMINAL_OPERATIONS:
                    if terminal is not None:
                        raise ValueError("artifact has multiple terminal events")
                    terminal = (location, payload)
                projection.control(operation, payload["arguments"], body["recorded_at"])
                if operation == "record_game_recovery":
                    writer.append("recovery.jsonl", legacy_line(payload["arguments"]["event"]))
            elif kind == "decision" and location.stream_kind == "decision":
                if location.owner != (payload.get("game_id"), payload.get("user_id")):
                    raise ValueError("decision does not match its shard owner")
                encoded = legacy_line(payload)
                projection.apply_decision(payload, len(encoded))
                writer.append("decisions.jsonl", encoded)
            elif kind == "round_bundle" and location.stream_kind == "replay" and store.mode == "train":
                validate_round_payload(payload)
                relative = payload["relative"]
                digest = hashlib.sha256(json_bytes(payload)).hexdigest()
                if relative in round_digests:
                    if round_digests[relative] != digest:
                        raise ValueError("round replay verification failed: duplicate path conflict")
                    continue
                projection.apply_round(relative, payload["record"])
                writer.write_json(relative + "/record.json", payload["record"])
                writer.write_jsonl(relative + "/events.jsonl", payload["events"])
                round_digests[relative] = digest
            elif kind == "archive_failure" and location.stream_kind == "replay" and store.mode == "train":
                relative = relative_artifact_path(payload["relative"])
                if not relative.startswith("rounds/failed/"):
                    raise ValueError("archive failure has an invalid path")
                failure_names[relative] = failure_names.get(relative, 0) + 1
                occurrence = failure_names[relative]
                failures.append((relative if occurrence == 1 else f"{relative}_attempt_{occurrence}", location))
            else:
                raise ValueError("unsupported artifact event or stream")
        for location in store.locations(polls=True, recovery=recovery):
            body = location.read()
            checkpoint("artifact.read_poll")
            if body.get("kind") != "poll" or location.stream_kind != "poll":
                raise ValueError("invalid state poll event")
            payload = body["payload"]
            if location.owner and location.owner != (payload.get("game_id"), payload.get("user_id")):
                raise ValueError("state poll does not match its shard owner")
            projection.record_poll(payload)
            writer.append("state_polls.jsonl", legacy_line(payload))
        for relative, location in failures:
            payload = location.read()["payload"]
            checkpoint("artifact.read_failure")
            if payload.get("raw") is not None:
                writer.write_json(relative + "/raw.json", payload["raw"])
            writer.write_json(relative + "/failure.json", payload["failure"])
        complete = projection._manifest["status"] in {"completed", "degraded"}
        if terminal is None:
            projection._now = lambda: datetime.now().astimezone()
            if isinstance(projection, MatchProjection):
                projection.interrupt("interrupted_before_finalization")
            else:
                projection.fail({"type": "RuntimeError"} if store.mode == "race" else "interrupted_before_finalization")
            store.warnings.append("run has no durable terminal event; exported as interrupted")
        else:
            store.verify_inventory(terminal[1].get("inventory"), terminal[0], completed=complete)
        manifest = projection._manifest
        if expected_manifest is not None and _normalized_manifest(manifest) != _normalized_manifest(expected_manifest):
            raise ValueError("replayed artifact projection disagrees with the live result")
        if complete and store.warnings:
            raise ValueError("a completed run contains incomplete audit evidence")
        if store.mode in PARTICIPANT_MODES:
            if manifest["decision_log"]["committed_bytes"] != writer._lengths["decisions.jsonl"]:
                raise ValueError("race decision byte checkpoint verification failed")
            if manifest["decision_log"]["committed_entries"] != manifest["decision_counts"]["total"]:
                raise ValueError("race decision entry checkpoint verification failed")
        recovery_path = recovery_log_path(manifest)
        if recovery_path is not None:
            with writer.files[recovery_path].reader() as reader:
                verify_recovery_log(reader, manifest)
        checkpoint("artifact.publish")
        writer.publish(manifest)
        checkpoint("artifact.published")
        return ExportResult(writer.path, manifest, writer.summaries.copy(),
                            tuple(dict.fromkeys(store.warnings)), complete and terminal is not None)
    finally:
        writer.close()
