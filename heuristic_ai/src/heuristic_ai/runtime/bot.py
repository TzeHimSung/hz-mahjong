from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import time
import re
from dataclasses import dataclass, replace
from datetime import datetime
from threading import Event
from typing import Any, Callable, Protocol

from heuristic_ai.protocol.client import ApiError, RequestDeadlineExceeded, StateChangedBeforeSend, redact
from heuristic_ai.game.errors import SnapshotError
from heuristic_ai.game.serialization import action_to_json, decision_key
from heuristic_ai.protocol.snapshots import normalize_snapshot
from heuristic_ai.game.models import ActionKind, GameOutcome, GameOutcomeStatus, Observation, Phase
from heuristic_ai.strategy.models import DecisionContext, GameContext, DecisionResult
from heuristic_ai.artifacts.models import DecisionTrace, SubmissionStatus
from heuristic_ai.game.tiles import is_tile_code
from heuristic_ai.game.rules import legal_actions, validate_actionable_observation
from heuristic_ai.game.public_inventory import PublicInventoryTracker, StaleInventorySnapshot
from heuristic_ai.runtime.response_polling import EventProgress, action_window_key, response_window_key
from heuristic_ai.runtime.recovery_context import ActionMemory, failure_context
from heuristic_ai.runtime.managed_state import ManagedState



def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def safe_submission_error(error: Exception) -> dict[str, Any] | None:
    """Leave final token replacement and the 16-KiB cap to the artifact writer."""

    if not isinstance(error, ApiError):
        return None
    code = error.code
    if not isinstance(code, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) is None:
        code = None
    return {"http_status": error.status, "code": code, "body": redact(error.body)}


def _poll_fields(response: Any) -> dict[str, Any]:
    """Only bounded protocol metadata may enter the training polling journal."""

    if not isinstance(response, dict):
        return {}
    fields = {
        key: response.get(key) is True for key in ("pending", "gap", "finished")
    }
    seq = response.get("seq")
    fields["response_seq"] = seq if type(seq) is int and seq >= 0 else None
    snapshot = response.get("snapshot")
    if not isinstance(snapshot, dict):
        return fields
    round_no = snapshot.get("round_no")
    fields["round_no"] = round_no if type(round_no) is int and round_no > 0 else None
    phase = snapshot.get("phase")
    fields["raw_phase"] = phase if isinstance(phase, str) and phase in {item.value for item in Phase} else None
    turn = snapshot.get("turn")
    fields["raw_turn"] = turn if type(turn) is int and -1 <= turn <= 3 else None
    for key in ("responding_seats", "waited_seat"):
        fields[f"{key}_present"] = key in snapshot
        value = snapshot.get(key)
        if key == "responding_seats":
            valid = isinstance(value, (list, tuple)) and len(value) <= 4 and all(
                type(seat) is int and 0 <= seat <= 3 for seat in value
            )
            fields[key] = list(value) if valid else None
        else:
            fields[key] = value if type(value) is int and -1 <= value <= 3 else None
    return fields


class JsonClient(Protocol):
    def request(self, method: str, path: str, body: Any = None, **options: Any) -> Any:
        raise NotImplementedError


class DecisionController(Protocol):
    degraded: bool
    degradation_reason: str | None

    def observe(self, observation: Observation) -> None:
        raise NotImplementedError

    def decide(self, context: DecisionContext) -> DecisionResult:
        raise NotImplementedError

    def finish(self, outcome: GameOutcome) -> None:
        raise NotImplementedError


class GameStopped(RuntimeError):
    """Raised when a coordinator stop event ends an otherwise live game loop."""


@dataclass(frozen=True, slots=True)
class GameResult:
    game_id: str
    scores: tuple[int, ...]
    degraded: bool
    degradation_reason: str | None = None


def _finished_scores(response: Any) -> tuple[int, ...]:
    if not isinstance(response, dict):
        raise RuntimeError("finished game response must be an object")
    snapshot = response.get("snapshot") or {}
    scores = snapshot.get("scores")
    if not isinstance(scores, list) or len(scores) != 4:
        raise RuntimeError("finished game snapshot must contain four scores")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in scores):
        raise RuntimeError("finished game scores must be integers")
    return tuple(scores)


def _trace(
    game: GameContext,
    context: DecisionContext,
    supervised: DecisionResult,
    *,
    server_seq: int,
    submission: SubmissionStatus,
    draw_to_action_ms: float | None = None,
    error: str | None = None,
    action_request_started_at: str | None = None,
    action_request_finished_at: str | None = None,
    submission_error: dict[str, Any] | None = None,
    generation: int | None = None,
) -> DecisionTrace:
    return DecisionTrace(
        game_id=game.game_id,
        round_no=context.observation.round_no,
        server_seq=server_seq,
        player_name=game.player_name,
        user_id=game.user_id,
        seat=context.observation.seat,
        decision_key=context.decision_key,
        context_hash=context.decision_key,
        candidates=context.candidates,
        execution_source=supervised.execution_source,
        decision=supervised.decision,
        elapsed_ms=supervised.elapsed_ms,
        fallback_used=supervised.fallback_used,
        fallback_reason=supervised.fallback_reason,
        submission=submission,
        draw_to_action_ms=draw_to_action_ms,
        error=error,
        observation_phase=context.observation.phase,
        action_request_started_at=action_request_started_at,
        action_request_finished_at=action_request_finished_at,
        submission_error=submission_error,
        generation=generation,
    )


def _ingest_public_events(
    response: dict[str, Any],
    tracker: PublicInventoryTracker,
    progress: EventProgress,
    *,
    coverage_start_seq: int | None = None,
    own_seat: int | None = None,
    covers_watermark: bool = False,
    audit_record: dict[str, Any] | None = None,
) -> bool:
    covers_watermark = (
        covers_watermark and type(coverage_start_seq) is int and coverage_start_seq > 0
        and not response.get("gap") and not response.get("pending")
    )
    events = response.get("events")
    if not isinstance(events, list):
        return False
    if any(
        not isinstance(event, dict) or type(event.get("seq")) is not int or event["seq"] <= 0
        or not isinstance(event.get("type"), str) or not event["type"].strip()
        for event in events
    ):
        return False
    maximum = max((event["seq"] for event in events), default=0)
    supplied_watermark = response.get("seq", maximum)
    if type(supplied_watermark) is not int or supplied_watermark < maximum:
        return False
    if coverage_start_seq is not None and (
        type(coverage_start_seq) is not int or coverage_start_seq < 0 or supplied_watermark < coverage_start_seq
    ):
        return False
    public_types = {
        "tile_discarded", "chi", "peng", "gang", "gang_ming", "gang_an",
        "gang_bu", "round_ended", "game_ended", "timeout", "pass",
    }
    # Hidden draws may occupy sequence positions. Never demand integer-adjacent
    # visible events, and never send private hand payloads into this ledger.
    public = [event for event in events if isinstance(event.get("type"), str) and event["type"] in public_types]
    public.extend(
        {"seq": event["seq"], "type": "tile_drawn", "seat": own_seat, "tile": event["tile"], "data": None}
        for event in events
        if event.get("type") == "tile_drawn" and own_seat is not None
        and event.get("seat") == own_seat and is_tile_code(event.get("tile"))
    )
    public.sort(key=lambda event: event["seq"])
    tracker.ingest(
        public, coverage_start_seq=coverage_start_seq,
        watermark=supplied_watermark, covers_watermark=covers_watermark,
    )
    if covers_watermark:
        progress.consumed(supplied_watermark)
    if audit_record is not None:
        audit_record["covers_watermark"] = covers_watermark
    return True


def _event_poll_fields(response: Any, request_seq: int) -> dict[str, Any]:
    events = response.get("events") if isinstance(response, dict) else None
    sequences = [event["seq"] for event in events
                 if isinstance(event, dict) and type(event.get("seq")) is int and event["seq"] >= 0] if isinstance(events, list) else []
    return {
        "coverage_start_seq": request_seq if request_seq > 0 else None,
        "covers_watermark": False,
        "event_count": len(events) if isinstance(events, list) else 0,
        "first_event_seq": min(sequences) if sequences else None,
        "last_event_seq": max(sequences) if sequences else None,
    }


@dataclass(slots=True)
class _PendingStateResponse:
    response: dict[str, Any]
    poll: dict[str, Any]
    request_seq: int
    received_at: float
    observed_at: float
    request_started_at: float
    transport_started_at: float | None = None


@dataclass(slots=True)
class _FailureProbe:
    stage: str = "state_request"
    request_seq: int | None = None
    response: Any = None
    observation: Observation | None = None


def _poll_game(
    client: JsonClient,
    game: GameContext,
    controller: DecisionController,
    stop: Event,
    on_decision: Callable[[DecisionTrace], None] | None,
    on_seat: Callable[[str, str, str, int], None] | None,
    clock: Callable[[], float],
    on_state_poll: Callable[[dict[str, Any]], None] | None,
    recovery_memory: ActionMemory,
    generation: int | None,
    on_ready: Callable[[], None] | None,
    failure_probe: _FailureProbe,
    on_state_response: Callable[..., None] | None,
    health_operation: Callable[[str, float | None], AbstractContextManager[Any]] | None = None,
    on_health_checkpoint: Callable[[str], None] | None = None,
) -> tuple[int, ...]:
    def operation(label: str, seconds: float | None = None) -> AbstractContextManager[Any]:
        return health_operation(label, seconds) if health_operation else nullcontext()

    seq = 0
    observed_keys: set[str] = set()
    accepted_keys = recovery_memory.accepted_keys
    decisions: dict[str, DecisionResult] = {}
    first_observed_at = recovery_memory.first_observed_at
    window_first_observed_at = recovery_memory.window_first_observed_at
    expired_keys: set[str] = set()
    normalization_retry = False
    progress = EventProgress()
    inventory = PublicInventoryTracker(game.game_id, certify_history=True)
    if recovery_memory.inventory_checkpoint is not None:
        inventory.restore(recovery_memory.inventory_checkpoint)
    if recovery_memory.event_progress is not None:
        progress = EventProgress(**recovery_memory.event_progress)
    accepted_response_windows = recovery_memory.accepted_response_windows
    current_round: int | None = recovery_memory.round_no
    current_seat: int | None = None
    previous_phase: Phase | None = None
    previous_snapshot_seq: int | None = None
    pending_gap = False
    immediate_full = True
    pending_response: _PendingStateResponse | None = None
    next_request_purpose: str | None = None
    ready_reported = False
    managed = getattr(client, "managed_state", None) or ManagedState(game, recovery_memory)
    rejected_keys = recovery_memory.rejected_keys
    backfill_pending = None

    def save_inventory():
        recovery_memory.inventory_checkpoint = inventory.checkpoint()
        recovery_memory.event_progress = {name: getattr(progress, name) for name in
            ('snapshot_seq', 'event_consumed_seq', 'bootstrap_anchor', 'round_bootstrap_floor')}

    def ready() -> None:
        nonlocal ready_reported
        if not ready_reported and not stop.is_set():
            if on_ready is not None:
                on_ready()
            ready_reported = True

    def poll_record(request_seq: int, purpose: str) -> dict[str, Any]:
        return {
            "game_id": game.game_id, "player_name": game.player_name,
            "user_id": game.user_id, "request_seq": request_seq,
            "request_purpose": purpose,
            "event_poll_mode": "snapshot" if request_seq == 0 else "events",
            "started_at": _timestamp(),
            **({"generation": generation} if generation is not None else {}),
        }

    def emit(record: dict[str, Any], reason: str) -> None:
        if on_health_checkpoint is not None:
            on_health_checkpoint("state_poll:" + reason)
        if on_state_poll is not None:
            record.update(progress.details())
            record.update(managed.details())
            diagnostics = inventory.diagnostics()
            record.update(
                accepted_round_no=current_round,
                inventory_round_no=diagnostics.get("round_no"),
                **{key: diagnostics.get(key) for key in (
                    "round_anchor_seq", "verified_round_start_seq", "round_boundary_status",
                    "quarantined_event_count", "reassigned_event_count",
                )},
            )
            record.setdefault("snapshot_disposition", "absent")
            record.setdefault("snapshot_reason", None)
            record["skip_reason"] = reason
            on_state_poll(record)

    def backfill(*, completed_delta_only=False) -> None:
        nonlocal pending_gap, pending_response, backfill_pending
        if completed_delta_only:
            # A foreground GET can outlast a background delta. Consume that
            # ready proof before validating its snapshot, without waiting or
            # starting another request. Private/gap responses retain the usual
            # loop ordering, so they cannot replace this snapshot implicitly.
            if backfill_pending is None or not backfill_pending[0].done():
                return
            try:
                batch = backfill_pending[0].result()[0]
            except Exception:
                return  # The normal backfill path records the failure next loop.
            if (not isinstance(batch, dict) or batch.get('snapshot') is not None
                    or any(batch.get(flag) for flag in ('pending', 'gap', 'finished'))):
                return
        # Full-snapshot tails do not certify event coverage. A chi with no
        # claimed tile needs a covered delta, even when all its visible
        # events happened to arrive in those tails. Also retain enough
        # cursor headroom to obtain those facts before the server evicts them.
        if backfill_pending is None and not managed.needs_inventory_catchup(progress, inventory.diagnostics()['inventory_reason']):
            return
        if stop.is_set() or pending_response is not None:
            return
        future = None
        if backfill_pending is not None:
            future, record, cursor, started, read_deadline = backfill_pending
            if not future.done():
                return
            backfill_pending = None
        else:
            if not progress.needs_backfill:
                return
            cursor = progress.fetch_cursor
            assert cursor is not None and cursor > 0
            record = poll_record(cursor, "inventory_backfill")
            started = time.monotonic()
            submit = getattr(client, 'submit_backfill', None)
            asynchronous = managed.notifications is not None and callable(submit)
            read_deadline = managed.inventory_catchup_deadline(started, background=asynchronous)
            if read_deadline is None:
                return
            if asynchronous:
                backfill_pending = (submit(cursor, read_deadline), record, cursor, started, read_deadline)
                return
        request_once = getattr(client, "request_once", None)
        if not callable(request_once):
            record.update(finished_at=_timestamp(), elapsed_ms=0.0)
            emit(record, "event_backfill_unavailable")
            return
        try:
            failure_probe.stage = "inventory_backfill_request"
            failure_probe.request_seq = cursor
            with operation("inventory_backfill", read_deadline - started):
                if future is not None:
                    batch, timing, received_at = future.result()
                    request_origin = timing.get('request_started_monotonic', started)
                else:
                    batch = request_once(
                        "GET", f"/api/games/{game.game_id}/state?seq={cursor}",
                        deadline_monotonic=read_deadline,
                        purpose="inventory_backfill",
                    )
                    received_at = time.monotonic()
                    request_origin = (managed.last_state_request_started_at
                        if managed.last_state_request_started_at is not None else started)
        except Exception as error:
            record.update(
                finished_at=_timestamp(), elapsed_ms=(time.monotonic() - started) * 1000,
                error_type=type(error).__name__,
            )
            if isinstance(error, ApiError):
                record["http_status"] = error.status
            emit(record, "event_backfill_error")
            managed.inventory_catchup_failed()
            return
        if on_state_response is not None:
            on_state_response(batch, request_seq=cursor)
        record.update(_poll_fields(batch))
        record.update(_event_poll_fields(batch, cursor))
        record.update(finished_at=_timestamp(), elapsed_ms=(received_at - started) * 1000)
        if not isinstance(batch, dict):
            emit(record, "event_backfill_unavailable")
        elif batch.get("snapshot") is not None or batch.get("finished"):
            # The authoritative snapshot belongs to this same request. Process it
            # in the action loop, retaining its original receipt time and one log.
            pending_response = _PendingStateResponse(batch, record, cursor, received_at, clock(), request_origin,
                timing.get('transport_started_monotonic') if future is not None else managed.last_state_transport_started_at)
            managed.inventory_catchup_completed()
        elif batch.get("pending"):
            managed.inventory_catchup_failed()
            emit(record, "pending")
        elif batch.get("gap"):
            managed.inventory_catchup_completed()
            progress.gap()
            pending_gap = True
            _ingest_public_events(
                batch, inventory, progress, coverage_start_seq=cursor,
                own_seat=current_seat, audit_record=record,
            )
            emit(record, "gap_refetch")
        elif _ingest_public_events(
            batch, inventory, progress, coverage_start_seq=cursor,
            own_seat=current_seat, covers_watermark=True, audit_record=record,
        ):
            save_inventory()
            if (progress.event_consumed_seq or 0) >= (progress.snapshot_seq or 0):
                managed.inventory_catchup_completed()
            emit(record, "event_backfill_complete")
        else:
            emit(record, "event_backfill_unavailable")

    while pending_response is not None or not stop.is_set():
        if managed.notifications is not None:
            backfill()
        if on_health_checkpoint is not None:
            on_health_checkpoint("state_poll")
        observed_at = None
        if pending_response is not None:
            response, poll = pending_response.response, pending_response.poll
            request_seq, received_at = pending_response.request_seq, pending_response.received_at
            observed_at = pending_response.observed_at
            request_origin = pending_response.request_started_at
            managed.last_state_transport_started_at = pending_response.transport_started_at
            pending_response = None
            immediate_full = False
        else:
            request_seq = seq
            cursor = progress.fetch_cursor
            request_seq, wait_seconds = managed.request(
                cursor if cursor is not None else seq,
                time.monotonic(), immediate_full=immediate_full or pending_gap,
            )
            if wait_seconds:
                with operation("response_poll_wait", wait_seconds):
                    if managed.wait_for_read(stop, wait_seconds):
                        raise GameStopped("game stopped while waiting for response poll")
            immediate_full = False
            purpose = next_request_purpose or (
                "event_poll" if request_seq > 0 else
                "response_snapshot" if managed.active else "state_snapshot"
            )
            next_request_purpose = None
            if stop.is_set():
                raise GameStopped("game stopped before state request")
            poll = poll_record(request_seq, purpose)
            request_started = time.monotonic()
            failure_probe.stage = "state_request"
            failure_probe.request_seq = request_seq
            try:
                with operation("state_request"):
                    response = client.request(
                        "GET",
                        f"/api/games/{game.game_id}/state?seq={request_seq}",
                    )
            except Exception as error:
                poll.update(finished_at=_timestamp(), elapsed_ms=(time.monotonic() - request_started) * 1000, error_type=type(error).__name__)
                if isinstance(error, ApiError):
                    poll["http_status"] = error.status
                emit(poll, "state_request_error")
                raise
            received_at = time.monotonic()
            request_origin = (managed.last_state_request_started_at
                if managed.last_state_request_started_at is not None else request_started)
            if not isinstance(response, dict):
                failure_probe.response = response
                failure_probe.observation = None
                failure_probe.stage = "snapshot_validation"
                poll.update(finished_at=_timestamp(), elapsed_ms=(received_at - request_started) * 1000,
                            snapshot_disposition="invalid", snapshot_reason="invalid_response")
                emit(poll, "normalization_refetch")
                if normalization_retry:
                    raise SnapshotError("state response must be an object")
                normalization_retry = True
                seq = 0
                immediate_full = True
                continue
            if on_state_response is not None:
                on_state_response(response, request_seq=request_seq)
            managed.note_read(received_at)
            poll.update(_poll_fields(response))
            poll.update(_event_poll_fields(response, request_seq))
            poll.update(finished_at=_timestamp(), elapsed_ms=(received_at - request_started) * 1000)
        backfill(completed_delta_only=True)
        managed.events(response, request_seq, request_origin, received_at)
        failure_probe.response = response
        failure_probe.request_seq = request_seq
        failure_probe.observation = None
        failure_probe.stage = "snapshot_validation"
        # Legacy clients may receive only the final score tuple. A complete
        # terminal snapshot must still pass the epoch and inventory checks.
        final_snapshot = response.get("snapshot")
        if response.get("finished") and isinstance(final_snapshot, dict) and set(final_snapshot) == {"scores"}:
            poll.update(snapshot_disposition="finished", snapshot_reason="legacy_scores_only")
            scores = _finished_scores(response)
            emit(poll, "finished")
            ready()
            return scores
        if stop.is_set() and not response.get("finished"):
            poll.update(snapshot_disposition="stopped", snapshot_reason="stop_requested")
            emit(poll, "stopped")
            raise GameStopped("game stopped before completion")
        if response.get("pending"):
            emit(poll, "pending")
            continue

        snapshot = response.get("snapshot")
        if snapshot is None:
            if response.get("gap"):
                progress.gap()
                pending_gap = True
                normalization_retry = False
            events_ingested = _ingest_public_events(
                response, inventory, progress,
                coverage_start_seq=request_seq if request_seq > 0 else None,
                own_seat=current_seat, covers_watermark=request_seq > 0,
                audit_record=poll,
            )
            emit(poll, "gap_refetch" if response.get("gap") else "incremental_refetch")
            if events_ingested and managed.skip_delta_snapshot(response):
                seq = progress.fetch_cursor or seq
                continue
            seq = 0
            immediate_full = True
            next_request_purpose = "event_snapshot"
            continue

        if observed_at is None:
            observed_at = clock()
        try:
            server_seq = response.get("seq", seq)
            if type(server_seq) is not int or server_seq < 0:
                raise SnapshotError("state sequence must be a nonnegative integer")
            if snapshot.get("game_id") is not None and snapshot["game_id"] != game.game_id:
                raise SnapshotError("state snapshot belongs to another game")
            observation = normalize_snapshot(snapshot)
            failure_probe.observation = observation
            recovery_memory.check_snapshot(observation.round_no, server_seq)
            validate_actionable_observation(observation)
            # Register the authoritative round before any attached batch is
            # assigned to an epoch. Global event sequences can span many rounds.
            inventory.begin_snapshot(observation, snapshot_seq=server_seq)
        except StaleInventorySnapshot as error:
            poll.update(snapshot_disposition="stale", snapshot_reason=error.reason)
            emit(poll, "stale_snapshot")
            seq = 0
            immediate_full = True
            continue
        except SnapshotError:
            poll.update(snapshot_disposition="invalid", snapshot_reason="invalid_snapshot")
            if response.get("gap"):
                progress.gap()
                pending_gap = True
            emit(poll, "normalization_refetch")
            if normalization_retry:
                raise
            normalization_retry = True
            seq = 0
            immediate_full = True
            continue
        if response.get("gap"):
            progress.gap()
            pending_gap = True
        if ((managed.uses_public_wakeups or managed.notifications is not None)
                and current_round != observation.round_no):
            # An old-round cursor always yields gap in the new round. Keep a
            # new fetch origin without claiming receipt of its missing prefix.
            progress.begin_round(server_seq, has_public_action=any(observation.discards) or any(observation.melds))
        _ingest_public_events(
            response, inventory, progress,
            coverage_start_seq=request_seq if request_seq > 0 else None,
            own_seat=observation.seat, covers_watermark=request_seq > 0,
            audit_record=poll,
        )
        progress.snapshot(server_seq)
        observation = inventory.observe(observation, snapshot_seq=server_seq, gap=pending_gap)
        candidates = legal_actions(observation, game.rules)
        recovery_memory.accept_snapshot(observation.round_no, server_seq)
        save_inventory()
        seq = server_seq
        normalization_retry = False
        pending_gap = False
        if current_round is not None and current_round != observation.round_no:
            observed_keys.clear()
            decisions.clear()
            expired_keys.clear()
        current_round = observation.round_no
        current_seat = observation.seat
        managed.observe(observation, server_seq, received_at)
        info = observation.public_inventory
        poll.update(
            seat=observation.seat, observed_phase=observation.phase.value,
            candidate_count=len(candidates),
            same_seq_phase_changed=(previous_snapshot_seq == server_seq and previous_phase is not None and previous_phase != observation.phase),
            inventory_known=info is not None and info.status == "exact",
            inventory_reason=info.reason if info is not None else None,
            snapshot_disposition="accepted",
            snapshot_reason="gap_snapshot" if response.get("gap") else
            "backfill_snapshot" if poll["request_purpose"] == "inventory_backfill" else "current_snapshot",
        )
        previous_phase, previous_snapshot_seq = observation.phase, server_seq
        if response.get("finished"):
            poll["snapshot_reason"] = "game_finished"
            scores = _finished_scores(response)
            emit(poll, "finished")
            ready()
            return scores
        if stop.is_set():
            poll.update(snapshot_disposition="stopped", snapshot_reason="stop_requested")
            emit(poll, "stopped")
            raise GameStopped("game stopped before strategy observation")
        ready()
        failure_probe.stage = "strategy_observe"
        key = decision_key(game.game_id, observation, candidates)
        poll["decision_key"] = key
        window = response_window_key(observation)
        opportunity = action_window_key(observation)
        response_already_accepted = window is not None and window in accepted_response_windows
        action_already_accepted = opportunity in recovery_memory.accepted_action_windows
        action_uncertain = opportunity in recovery_memory.uncertain_action_windows
        already_submitted = (key in accepted_keys or key in rejected_keys or response_already_accepted or action_already_accepted or action_uncertain
                             or opportunity in recovery_memory.unsubmitted_action_windows)
        if candidates and not already_submitted:
            first_observed_at.setdefault(key, observed_at)
            # Preserve the first receipt even when observe()/decide() itself
            # fails, or inventory enrichment changes the next decision key.
            window_first_observed_at.setdefault(opportunity, first_observed_at[key])
            deadline = managed.action_deadline
        if key not in observed_keys:
            if on_seat is not None:
                on_seat(
                    game.game_id,
                    game.player_name,
                    game.user_id,
                    observation.seat,
                )
            observe_budget = (max(0.0, deadline - clock())
                              if health_operation and candidates and not already_submitted else None)
            with operation("strategy_observe", observe_budget):
                controller.observe(observation)
            observed_keys.add(key)
        if not candidates or already_submitted:
            emit(poll, "no_candidates" if not candidates else "submission_unconfirmed" if action_uncertain else "already_accepted")
            if poll["request_purpose"] != "inventory_backfill":
                backfill()
            continue

        from .managed_state import INVENTORY_REPAIR_REASONS
        # Optional history repair has one budget from this window's first
        # observation. Slow repeated snapshots must not renew it and consume
        # the discard deadline; incomplete history keeps its existing fallback.
        repair_deadline = managed.inventory_catchup_deadline(window_first_observed_at[opportunity])
        if ((managed.uses_public_wakeups or managed.notifications is not None)
                and observation.phase is Phase.DRAW and observation.turn == observation.seat
                and info is not None and info.reason in INVENTORY_REPAIR_REASONS
                and repair_deadline is not None and clock() < repair_deadline
                and managed.action_deadline - clock() > 1.0):
            # Use the long draw window to recover public provenance before
            # asking the configured strategy to fall back. Refresh the private
            # snapshot after the delta; never synthesize a hand or advance the
            # snapshot watermark from public events.
            before_cursor = progress.fetch_cursor
            backfill()
            # An unfinished background delta leaves this snapshot usable.
            # Refresh only after receiving new coverage, a snapshot or a gap.
            if (progress.fetch_cursor != before_cursor
                    or pending_response is not None or pending_gap):
                emit(poll, "inventory_repair_refetch")
                seq, immediate_full = 0, True
                continue
        emit(poll, "decide")
        context = DecisionContext(
            game=game,
            observation=observation,
            candidates=candidates,
            decision_key=key,
            budget_ms=game.decision_budget_ms,
            deadline_monotonic=managed.search_deadline,
        )
        if clock() >= deadline:
            recovery_memory.unsubmitted_action_windows.add(opportunity)
            cached = decisions.pop(key, None)
            if cached is not None and on_decision is not None and key not in expired_keys:
                on_decision(_trace(game, context, cached, server_seq=server_seq,
                                   submission=SubmissionStatus.ERROR, error="decision_deadline_expired", generation=generation))
                expired_keys.add(key)
            emit(dict(poll), "decision_deadline_expired")
            seq = 0
            immediate_full = not managed.active
            continue
        supervised = decisions.get(key)
        if supervised is None:
            failure_probe.stage = "strategy_decide"
            with operation("strategy_decide", max(0.0, deadline - clock()) if health_operation else None):
                supervised = controller.decide(context)
            decisions[key] = supervised
        if stop.is_set():
            raise GameStopped("game stopped before completion")
        if clock() >= deadline:
            # The strategy can overrun during a non-preemptible primitive or
            # audit construction. Never submit that result, including a cached
            # decision from a previous 409 attempt in this same window.
            if on_decision is not None and key not in expired_keys:
                on_decision(_trace(
                    game, context, supervised, server_seq=server_seq,
                    submission=SubmissionStatus.ERROR, error="decision_deadline_expired",
                    generation=generation,
                ))
            expired_keys.add(key)
            decisions.pop(key, None)
            seq = 0
            immediate_full = not managed.active
            continue
        validate = getattr(controller, 'validate_decision', None)
        if validate is not None and not validate(context, supervised):
            decisions.pop(key, None)
            seq = 0
            continue
        action_started_at = _timestamp()
        if stop.is_set():
            raise GameStopped("game stopped before action submission")
        failure_probe.stage = "action_submission"
        recovery_memory.uncertain_action_windows.add(opportunity)
        try:
            with operation("action_submission"):
                client.request(
                    "POST",
                    f"/api/games/{game.game_id}/action",
                    action_to_json(supervised.decision.action),
                    deadline_monotonic=deadline, purpose="action_submit", retry_policy="never",
                )
        except Exception as error:
            action_finished_at = _timestamp()
            clock()
            if isinstance(error, ApiError) and error.status == 409:
                recovery_memory.uncertain_action_windows.discard(opportunity)
                rejected_keys.add(key)
                decisions.pop(key, None)
                first_observed_at.pop(key, None)
                if on_decision is not None:
                    on_decision(
                        _trace(
                            game,
                            context,
                            supervised,
                            server_seq=server_seq,
                            submission=SubmissionStatus.CONFLICT,
                            action_request_started_at=action_started_at,
                            action_request_finished_at=action_finished_at,
                            submission_error=safe_submission_error(error),
                            generation=generation,
                        )
                    )
                seq = 0
                immediate_full = True
                del supervised
                continue
            if on_decision is not None:
                on_decision(
                    _trace(
                        game,
                        context,
                        supervised,
                        server_seq=server_seq,
                        submission=SubmissionStatus.ERROR,
                        error=type(error).__name__,
                        action_request_started_at=None if isinstance(error, StateChangedBeforeSend) else action_started_at,
                        action_request_finished_at=None if isinstance(error, StateChangedBeforeSend) else action_finished_at,
                        submission_error=safe_submission_error(error),
                        generation=generation,
                    )
                )
            if isinstance(error, StateChangedBeforeSend):
                recovery_memory.uncertain_action_windows.discard(opportunity)
                decisions.pop(key, None)
                seq, immediate_full = 0, True
                continue
            if isinstance(error, RequestDeadlineExceeded) and not error.sent:
                recovery_memory.uncertain_action_windows.discard(opportunity)
                recovery_memory.unsubmitted_action_windows.add(opportunity)
                decisions.pop(key, None)
                seq = 0
                continue
            if isinstance(error, RequestDeadlineExceeded) and error.sent:
                # Preserve the uncertain POST and quarantine the first fresh
                # window, just as a recovery would. The independent state pool
                # can resynchronize without stopping the other seats. A late
                # transport completion never acknowledges this application
                # attempt; ordinary state failures still enter recovery.
                managed.fence = True
                decisions.pop(key, None)
                seq, immediate_full = 0, True
                continue
            raise

        accepted_keys.add(key)
        managed.submitted(supervised.decision.action)
        recovery_memory.accepted_action_windows.add(opportunity)
        recovery_memory.uncertain_action_windows.discard(opportunity)
        if window is not None:
            accepted_response_windows.add(window)
        action_finished_at = _timestamp()
        submitted_at = clock()
        failure_probe.stage = "action_recording"
        inventory.record_action(observation, supervised.decision.action, snapshot_seq=server_seq)
        save_inventory()
        accepted_hook = getattr(controller, 'accepted_action', None)
        if accepted_hook is not None:
            accepted_hook(context, supervised.decision.action)
        draw_to_action_ms = None
        if (
            observation.phase.value == "draw"
            and observation.drawn_tile is not None
            and supervised.decision.action.kind
            in {ActionKind.DISCARD, ActionKind.GANG, ActionKind.HU}
        ):
            draw_to_action_ms = (
                submitted_at - window_first_observed_at[opportunity]
            ) * 1000
        if on_decision is not None:
            on_decision(
                _trace(
                    game,
                    context,
                    supervised,
                    server_seq=server_seq,
                    submission=SubmissionStatus.ACCEPTED,
                    draw_to_action_ms=draw_to_action_ms,
                    action_request_started_at=action_started_at,
                    action_request_finished_at=action_finished_at,
                    generation=generation,
                )
            )
        decisions.pop(key, None)
        first_observed_at.pop(key, None)
        # Let connected SSE coalesce our known normal-discard notification.
        # Other actions and all later forced recovery/gap reads keep snapshots.
        defer_discard_read = (managed.submitted_discard_seq == managed.snapshot_seq
            and managed.notifications is not None and not managed.uses_public_wakeups
            and managed.notifications.snapshot()['notification_connected'])
        immediate_full = supervised.decision.action.kind is not ActionKind.PASS and not defer_discard_read
        del supervised
        seq = 0
        if poll["request_purpose"] != "inventory_backfill":
            backfill()

    raise GameStopped("game stopped before completion")


def play_game(
    client: JsonClient,
    game: GameContext,
    controller: DecisionController,
    stop: Event,
    on_decision: Callable[[DecisionTrace], None] | None = None,
    on_seat: Callable[[str, str, str, int], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    *,
    on_state_poll: Callable[[dict[str, Any]], None] | None = None,
    on_state_response: Callable[..., None] | None = None,
    recovery_memory: ActionMemory | None = None,
    generation: int | None = None,
    on_failure: Callable[[BaseException, dict[str, Any]], None] | None = None,
    on_ready: Callable[[], None] | None = None,
    failure_sanitizer: Callable[[str], str] | None = None,
    health_operation: Callable[[str, float | None], AbstractContextManager[Any]] | None = None,
    on_health_checkpoint: Callable[[str], None] | None = None,
) -> GameResult:
    """Run one normalized player view through a supervised strategy session."""

    delegate = getattr(client, "delegate", client)
    network = getattr(delegate, "managed_network", None)
    if network is not None:
        return network.service.play_game(delegate.managed_token, game, controller, stop,
            on_decision=on_decision, on_seat=on_seat, on_state_poll=on_state_poll,
            on_state_response=on_state_response, recovery_memory=recovery_memory,
            generation=generation, on_failure=on_failure, on_ready=on_ready,
            failure_sanitizer=failure_sanitizer, health_operation=health_operation,
            on_health_checkpoint=on_health_checkpoint)

    outcome = GameOutcome(game.game_id, GameOutcomeStatus.ABORTED)
    probe = _FailureProbe()
    pending_error: BaseException | None = None
    pending_priority = 0

    def retain(error: BaseException, *, callback: bool = False) -> None:
        nonlocal pending_error, pending_priority
        priority = 3 if isinstance(error, (KeyboardInterrupt, SystemExit)) else 2 if callback else 1
        if priority > pending_priority:
            pending_error, pending_priority = error, priority

    def report(error: BaseException, *, cleanup: bool = False) -> None:
        if on_failure is None:
            return
        try:
            context = failure_context(
                error, game_id=game.game_id, generation=generation,
                request_seq=probe.request_seq, stage="strategy_cleanup" if cleanup else probe.stage,
                response=probe.response, observation=probe.observation,
                sanitize=failure_sanitizer,
            )
            action_request = probe.stage in {"action_submission", "action_recording"}
            context["last_request"] = {
                "method": "POST" if action_request else "GET",
                "path": f"/api/games/{game.game_id}/action" if action_request else
                        f"/api/games/{game.game_id}/state?seq={probe.request_seq or 0}",
            }
            on_failure(error, context)
        except BaseException as recording_error:
            if recording_error is not error and recording_error.__cause__ is None:
                recording_error.__cause__ = error
            retain(recording_error, callback=True)

    try:
        scores = _poll_game(
            client,
            game,
            controller,
            stop,
            on_decision,
            on_seat,
            clock,
            on_state_poll,
            recovery_memory if recovery_memory is not None else ActionMemory(),
            generation,
            on_ready,
            probe,
            on_state_response,
            health_operation,
            on_health_checkpoint,
        )
        outcome = GameOutcome(game.game_id, GameOutcomeStatus.COMPLETED, scores)
    except BaseException as error:
        outcome = GameOutcome(
            game.game_id,
            GameOutcomeStatus.ABORTED,
            error=type(error).__name__,
        )
        retain(error)
        report(error)
    try:
        operation = health_operation("strategy_finish", None) if health_operation else nullcontext()
        with operation:
            controller.finish(outcome)
    except BaseException as error:
        retain(error)
        report(error, cleanup=True)
    if pending_error is not None:
        raise pending_error.with_traceback(pending_error.__traceback__)
    return GameResult(
        game.game_id,
        scores,
        controller.degraded,
        controller.degradation_reason,
    )
