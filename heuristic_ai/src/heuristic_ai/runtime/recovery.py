"""Per-game generations and durable recovery, independent of room lifecycle."""
from __future__ import annotations

import inspect
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
from typing import Any

from .bot import GameResult, GameStopped, play_game
from .recovery_context import ActionMemory, failure_context, safe_exception

LOGGER = logging.getLogger("heuristic_ai.recovery")


class RecorderCallbackFailure(RuntimeError):
    """Preserve callback provenance even when its exception resembles a game error."""

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__("recorder callback failed")


def guard_callback(callback: Callable[..., Any], parent_stop: Any = None) -> Callable[..., Any]:
    @wraps(callback)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return callback(*args, **kwargs)
        except RecorderCallbackFailure:
            if parent_stop is not None:
                parent_stop.set()
            raise
        except Exception as error:
            if parent_stop is not None:
                parent_stop.set()
            raise RecorderCallbackFailure(error) from None
        except BaseException:
            if parent_stop is not None:
                parent_stop.set()
            raise
    return guarded


def process_control(error: BaseException) -> BaseException | None:
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return error
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            found = process_control(child)
            if found is not None:
                return found
    return None


def recording_failure(error: BaseException, seen: set[int] | None = None) -> RecorderCallbackFailure | None:
    seen = set() if seen is None else seen
    if id(error) in seen:
        return None
    seen.add(id(error))
    if isinstance(error, RecorderCallbackFailure):
        return error
    children = list(error.exceptions) if isinstance(error, BaseExceptionGroup) else []
    cause = BaseException.__cause__.__get__(error, type(error))
    context = BaseException.__context__.__get__(error, type(error))
    children.extend(child for child in (cause, context) if child is not None)
    for child in children:
        found = recording_failure(child, seen)
        if found is not None:
            return found
    return None


class LinkedStop:
    """An Event-like local stop which also observes one run-wide parent Event."""

    def __init__(self, parent: Any) -> None:
        self.parent = parent
        self.local = threading.Event()

    def is_set(self) -> bool:
        return self.local.is_set() or self.parent.is_set()

    def set(self) -> None:
        self.local.set()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return self.is_set()
            self.local.wait(0.05 if remaining is None else min(0.05, remaining))
        return True


def retry_delay(restarts: int, error: BaseException | None = None) -> float:
    delay = float((1, 2, 4, 8, 16, 30, 60)[min(max(restarts - 1, 0), 6)])
    value = getattr(error, "retry_after", None)
    if not isinstance(value, str):
        return delay
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                from datetime import timezone
                date = date.replace(tzinfo=timezone.utc)
            seconds = date.timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return delay
    return max(delay, seconds) if math.isfinite(seconds) else delay


@dataclass
class RecoveryGroup:
    game_id: str
    members: Mapping[str, str]
    parent_stop: Any
    recorder: Any
    clock: Callable[[], float] = time.monotonic
    stage_no: int | None = None
    generation: int = field(default=1, init=False)
    restarts: int = field(default=0, init=False)
    next_retry_at: float | None = field(default=None, init=False)
    terminal_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.stop = LinkedStop(self.parent_stop)
        sanitize = getattr(getattr(self.recorder, "_serializer", None), "sanitize", None)
        self.sanitize = guard_callback(sanitize, self.parent_stop) if callable(sanitize) else lambda value: value
        self.memories = {user_id: ActionMemory() for user_id in self.members}
        self._lock = threading.RLock()
        self._reported: set[str] = set()
        self._exception_ids: set[tuple[str, int]] = set()
        self._ready: set[str] = set()
        self._retry_delay = 0.0
        self._backoff_failures = 0
        self._healthy_since: float | None = None

    @property
    def pending(self) -> bool:
        return self.next_retry_at is not None and self.terminal_reason is None

    def reported(self, user_id: str) -> bool:
        with self._lock:
            return user_id in self._reported

    def _emit(self, event: str, **details: Any) -> None:
        callback = getattr(self.recorder, "record_game_recovery", None)
        if callback is not None:
            guard_callback(callback, self.parent_stop)({
                'at_utc': datetime.now(timezone.utc).isoformat(), 'monotonic': self.clock(),
                "game_id": self.game_id, "generation": self.generation,
                "event": event, "restart_count": self.restarts,
                **({"stage_no": self.stage_no} if self.stage_no is not None else {}),
                **details,
            })
        LOGGER.info("game=%s generation=%s recovery=%s restart_count=%s next_retry=%s",
                    self.sanitize(self.game_id), self.generation, event, self.restarts, details.get("next_retry_at"))

    def note_failure(self, user_id: str, error: BaseException,
                     context: Mapping[str, Any] | None = None) -> None:
        control = process_control(error)
        if control is not None:
            self.parent_stop.set()
            raise control
        recording_error = recording_failure(error)
        if recording_error is not None:
            self.parent_stop.set()
            raise recording_error
        if not isinstance(error, Exception):
            self.parent_stop.set()
            raise error
        if type(error) is GameStopped and self.stop.is_set():
            return
        # Block additional attempts before entering the persistence boundary.
        self.stop.set()
        with self._lock:
            error_key = (user_id, id(error))
            if error_key in self._exception_ids:
                return
            self._exception_ids.add(error_key)
            self._reported.add(user_id)
            first = not self.pending and self.terminal_reason is None
            if first:
                self.restarts += 1
                # A recovered game that stayed healthy for 30 seconds begins
                # a new failure streak. Lifetime counts/generations and action
                # fences remain intact. Downtime and partial member readiness
                # never count toward this stable interval.
                if self._healthy_since is not None and self.clock() - self._healthy_since >= 30:
                    self._backoff_failures = 0
                self._backoff_failures += 1
                self._healthy_since = None
            self._emit("failure", user_id=user_id, player_name=self.members[user_id],
                       error=safe_exception(error, sanitize=self.sanitize), context=dict(context or failure_context(
                           error, game_id=self.game_id, generation=self.generation, sanitize=self.sanitize)))
            if self.terminal_reason is not None or self.parent_stop.is_set():
                return
            delay = retry_delay(self._backoff_failures, error)
            due = self.clock() + delay
            if first or delay > self._retry_delay:
                self._retry_delay = delay
                self.next_retry_at = due
                now = datetime.now().astimezone()
                try:
                    scheduled_at = now + timedelta(seconds=delay)
                except OverflowError:
                    # The monotonic deadline still respects Retry-After; the
                    # diagnostic wall timestamp is capped at ISO's range.
                    scheduled_at = datetime.max.replace(tzinfo=now.tzinfo)
                self._emit("restart_scheduled", retry_delay_seconds=delay,
                           next_retry_at=scheduled_at.isoformat())

    def begin_restart(self) -> None:
        with self._lock:
            if not self.pending or self.parent_stop.is_set():
                raise GameStopped("game is not eligible for restart")
            self.generation += 1
            self.next_retry_at = None
            self.stop = LinkedStop(self.parent_stop)
            self._reported.clear()
            self._exception_ids.clear()
            self._ready.clear()
            self._retry_delay = 0.0
            self._emit("restarted")

    def ready(self, user_id: str) -> None:
        with self._lock:
            if user_id in self._ready or self.stop.is_set():
                return
            self._ready.add(user_id)
            if self._ready == set(self.members):
                self._healthy_since = self.clock()
                if self.generation > 1:
                    self._emit("recovered")

    def terminal(self, reason: str) -> None:
        with self._lock:
            if self.terminal_reason is not None:
                return
            self.terminal_reason = reason
            self.next_retry_at = None
            self.stop.set()
            # Successful untouched games preserve their historical file layout.
            if self.restarts or reason not in {"completed", "finished"}:
                self._emit("terminal", reason=reason)


def validate_game_result(result: Any, game_id: str) -> GameResult:
    if not isinstance(result, GameResult) or result.game_id != game_id:
        raise RuntimeError("player worker returned an invalid game result")
    if (not isinstance(result.scores, Sequence) or isinstance(result.scores, (str, bytes))
            or len(result.scores) != 4 or any(type(score) is not int for score in result.scores)):
        raise RuntimeError("player worker returned invalid final scores")
    if type(result.degraded) is not bool:
        raise RuntimeError("player worker returned an invalid degradation state")
    if result.degraded and result.degradation_reason is not None and (
        not isinstance(result.degradation_reason, str)
        or re.fullmatch(r"[a-z][a-z0-9_]*", result.degradation_reason) is None
    ):
        raise RuntimeError("player worker returned an invalid stable degradation reason")
    return result


def handle_start_failure(group: RecoveryGroup, user_id: str, error: BaseException,
                         controller: Any = None) -> None:
    """Journal a startup failure and always release a constructed controller."""
    fatal: BaseException | None = None
    try:
        group.note_failure(user_id, error, {"stage": "worker_start"})
    except BaseException as failure:
        fatal = failure
    finally:
        finish = getattr(controller, "finish", None)
        if callable(finish):
            from heuristic_ai.game.models import GameOutcome, GameOutcomeStatus
            try:
                finish(GameOutcome(group.game_id, GameOutcomeStatus.ABORTED))
            except BaseException as cleanup:
                try:
                    group.note_failure(user_id, cleanup, {"stage": "worker_start_cleanup"})
                except BaseException as failure:
                    if fatal is None or process_control(fatal) is None and process_control(failure) is not None:
                        fatal = failure
    if fatal is not None:
        raise fatal


def invoke_recoverable_worker(group: RecoveryGroup, user_id: str, worker: Callable[..., GameResult],
                              client: Any, game: Any, controller: Any, recorder: Any,
                              **bot_options: Any) -> GameResult:
    """Run one generation, retaining callback provenance and fault evidence."""
    options = {
        **bot_options, "generation": group.generation,
        "recovery_memory": group.memories[user_id],
        "failure_sanitizer": group.sanitize,
        "on_failure": lambda error, context: group.note_failure(user_id, error, context),
        "on_ready": lambda: group.ready(user_id),
    }
    for callback_name in ("on_state_poll", "on_state_response"):
        if options.get(callback_name) is not None:
            options[callback_name] = guard_callback(options[callback_name], group.parent_stop)
    result: GameResult | None = None
    primary: BaseException | None = None
    try:
        # Preserve the existing injectable six-argument worker interface.
        parameters = inspect.signature(worker).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            options = {key: value for key, value in options.items() if key in parameters}
        result = worker(client, game, controller, group.stop,
                        guard_callback(recorder.record_decision, group.parent_stop),
                        guard_callback(recorder.identify_player, group.parent_stop), **options)
        result = validate_game_result(result, group.game_id)
        group.ready(user_id)
    except BaseException as error:
        primary = error
        try:
            group.note_failure(user_id, error)
        except BaseException as recording_error:
            primary = recording_error
    finally:
        # Production play_game owns its finally block. Injected workers still
        # need the same cleanup guarantee; real supervisors finish idempotently.
        finish = getattr(controller, "finish", None)
        if worker is not play_game and callable(finish) and not getattr(controller, "_finished", False):
            from heuristic_ai.game.models import GameOutcome, GameOutcomeStatus
            try:
                finish(GameOutcome(group.game_id,
                    GameOutcomeStatus.COMPLETED if primary is None and result is not None else GameOutcomeStatus.ABORTED,
                    result.scores if primary is None and result is not None else None))
            except BaseException as cleanup:
                try:
                    group.note_failure(user_id, cleanup, {"stage": "worker_cleanup"})
                except BaseException as recording_error:
                    cleanup = recording_error
                if primary is None:
                    primary = cleanup
                else:
                    control = process_control(primary) or process_control(cleanup)
                    if control is not None:
                        primary = control
                    elif isinstance(cleanup, RecorderCallbackFailure):
                        primary = cleanup
                    elif not isinstance(primary, RecorderCallbackFailure) and isinstance(primary, Exception) and isinstance(cleanup, Exception):
                        primary = ExceptionGroup("player execution and cleanup failures", [primary, cleanup])
    if primary is not None:
        raise primary
    assert result is not None
    return result
