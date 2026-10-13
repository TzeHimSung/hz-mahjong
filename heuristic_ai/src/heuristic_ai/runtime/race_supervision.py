"""Foreground race process supervision with explicit, acknowledged lifecycle IPC."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any, Callable
import uuid

from heuristic_ai.artifacts.models import RunResult
from heuristic_ai.config.models import RaceConfig
from ._race_worker import race_worker_main
from ._race_index import RunsIndex
from .process_guard import DuplicateRaceError, ProcessTreeGuard, RaceIdentityLock, identity_key, default_lock_dir
from .supervision_contracts import (
    ChildEvent, ControlCommand, FATAL_FAILURE_CATEGORIES, HEALTHY_RESET_SECONDS,
    RaceSupervisionResult, RESTART_DELAYS, STARTUP_GRACE_SECONDS, STOP_GRACE_SECONDS,
    StopSignal, TERMINAL_REASONS, race_config_to_payload, run_result_from_payload,
)


@dataclass(frozen=True)
class _Timing:
    startup_grace: float = STARTUP_GRACE_SECONDS
    stop_grace: float = STOP_GRACE_SECONDS
    healthy_reset: float = HEALTHY_RESET_SECONDS
    restart_delays: tuple[float, ...] = RESTART_DELAYS
    poll_interval: float = 0.05
    control_drain_timeout: float = 1.0


@dataclass
class _Generation:
    number: int
    process: Any
    guard: ProcessTreeGuard
    control: Connection
    telemetry: Connection
    logs: Connection
    started_at: float
    pid: int
    events: queue.Queue[Any] = field(default_factory=queue.Queue)
    control_eof: threading.Event = field(default_factory=threading.Event)
    telemetry_lock: threading.Lock = field(default_factory=threading.Lock)
    heartbeat_at: float | None = None
    health: dict[str, Any] | None = None
    telemetry_error: str | None = None
    connected_at: float | None = None
    healthy_since: float | None = None
    result: RunResult | None = None
    result_at: float | None = None
    failure: dict[str, Any] | None = None
    stop_reason: str | None = None
    identity_approved: bool = False
    finished: bool = False
    exit_code: int | None = None
    run_id: str | None = None
    work_dir: str | None = None
    dropped_logs: int = 0
    reported_dropped_logs: int = 0
    control_drain_incomplete: bool = False
    command_kinds: set[str] = field(default_factory=set)
    readers: list[threading.Thread] = field(default_factory=list)


class RaceSupervisor:
    """Own exactly one generation at a time; test injection never changes the CLI."""

    def __init__(self, config: RaceConfig, *, stop: StopSignal | None = None,
                 timing: _Timing | None = None, worker_target: Callable[..., None] = race_worker_main,
                 log_manager: Any = None, lock_dir: Path | None = None) -> None:
        self.config = config
        # Resolve and freeze once; subsequent attempts never reopen a YAML file.
        self.config_payload = race_config_to_payload(config)
        self.stop = stop if stop is not None else threading.Event()
        self._signal_stop_requested = False
        self.timing = timing if timing is not None else _Timing()
        self.worker_target = worker_target
        self.context = multiprocessing.get_context("spawn")
        self.supervision_id = uuid.uuid4().hex
        self.log_dir = config.competition.output_dir / "runtime_logs" / "race" / self.supervision_id
        self.log_manager = log_manager
        self.lock_dir = lock_dir
        self.restart_count = 0
        self.failure_streak = 0
        self.last_result: RunResult | None = None
        self.terminal_reason: str | None = None
        self.fatal_reason: str | None = None
        self._identity: str | None = None
        self._identity_lock: RaceIdentityLock | None = None
        self._current: _Generation | None = None
        self._previous_pid: int | None = None
        self._index: RunsIndex | None = None

    def _on_stop_signal(self, _signum: int, _frame: Any) -> None:
        # A signal can interrupt Event.wait while its condition lock is held.
        # Acquiring that lock again via Event.set inside the handler deadlocks.
        self._signal_stop_requested = True

    def _stopping(self) -> bool:
        if self._signal_stop_requested:
            self._signal_stop_requested = False
            self.stop.set()
        return self.stop.is_set()

    def _wait_for_stop(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self._stopping():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            # Signals only set a flag, so long restart backoffs must periodically
            # return to normal control flow to apply it without taking locks in
            # the handler. External Event.set still wakes this wait immediately.
            self.stop.wait(min(remaining, 0.05))
        return True

    def _log(self, kind: str, payload: dict[str, Any], *, level: int = logging.INFO,
             generation: _Generation | None = None) -> None:
        current = generation if generation is not None else self._current
        self.log_manager.submit(kind, payload, generation=current.number if current else 0,
                                pid=current.pid if current else None,
                                restart_count=self.restart_count, level=level)

    def _control_reader(self, generation: _Generation) -> None:
        last_sequence = 0
        try:
            while True:
                event = ChildEvent.from_payload(generation.control.recv())
                if event.generation != generation.number or event.sequence <= last_sequence:
                    continue
                last_sequence = event.sequence
                generation.events.put(event)
        except (EOFError, OSError):
            pass
        except (ValueError, TypeError):
            generation.events.put("invalid_control_event")
        finally:
            generation.control_eof.set()

    def _telemetry_reader(self, generation: _Generation) -> None:
        last_sequence = 0
        try:
            while True:
                event = ChildEvent.from_payload(generation.telemetry.recv())
                if event.generation != generation.number or event.sequence <= last_sequence:
                    continue
                last_sequence = event.sequence
                with generation.telemetry_lock:
                    if event.kind == "heartbeat":
                        sent_at = event.payload.get("sent_at")
                        if type(sent_at) not in (int, float):
                            generation.telemetry_error = "invalid_heartbeat"
                        else:
                            generation.heartbeat_at = min(float(sent_at), time.monotonic())
                        dropped = event.payload.get("dropped_logs", 0)
                        if type(dropped) is int and dropped >= 0:
                            generation.dropped_logs = dropped
                    elif event.kind == "health":
                        generation.health = event.payload
        except (EOFError, OSError):
            pass
        except (ValueError, TypeError):
            with generation.telemetry_lock:
                generation.telemetry_error = "invalid_health_event"

    def _log_reader(self, generation: _Generation) -> None:
        try:
            while True:
                value = generation.logs.recv()
                if not isinstance(value, dict) or type(value.get("kind")) is not str or type(value.get("payload")) is not dict:
                    continue
                payload = value["payload"]
                level = payload.get("level", logging.INFO)
                self._log(value["kind"], payload, level=level if type(level) is int else logging.INFO,
                          generation=generation)
        except (EOFError, OSError):
            pass

    def _spawn(self) -> _Generation:
        guard = ProcessTreeGuard()
        parent_control, child_control = self.context.Pipe(duplex=True)
        parent_telemetry, child_telemetry = self.context.Pipe(duplex=False)
        parent_logs, child_logs = self.context.Pipe(duplex=False)
        number = self.restart_count + 1
        process = self.context.Process(
            target=self.worker_target,
            args=(self.config_payload, number, child_control, child_telemetry, child_logs, self.supervision_id),
            name=f"mahjong-race-{number}",
        )
        try:
            process.start()
            guard.attach(process)
        except BaseException:
            try:
                guard.kill()
            finally:
                if process.pid is not None and process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
                guard.close()
                for connection in (parent_control, parent_telemetry, parent_logs):
                    connection.close()
            raise
        finally:
            for connection in (child_control, child_telemetry, child_logs):
                connection.close()
        generation = _Generation(number, process, guard, parent_control, parent_telemetry,
                                 parent_logs, time.monotonic(), process.pid)
        self._current = generation
        for target, name in ((self._control_reader, "control"), (self._telemetry_reader, "health"),
                             (self._log_reader, "logs")):
            reader = threading.Thread(target=target, args=(generation,), name=f"race-{name}-{number}", daemon=True)
            generation.readers.append(reader)
            reader.start()
        self._log("child_started", {"old_pid": self._previous_pid, "new_pid": process.pid})
        self._index.submit("child_started", generation.number, generation.pid)
        return generation

    def _may_proceed(self, generation: _Generation) -> bool:
        return (not self._stopping() and generation.stop_reason is None
                and self.terminal_reason is None and self.fatal_reason is None
                and not generation.finished)

    def _command(self, generation: _Generation, kind: str) -> bool:
        requested_kind = kind
        if kind == "proceed" and not self._may_proceed(generation):
            kind = "stop"
        if kind in generation.command_kinds:
            return kind == requested_kind
        try:
            # There are at most four small parent commands per generation.
            # Only this thread writes this pipe direction.
            payload = {}
            if kind == "proceed" and self._identity is not None:
                directory = self.lock_dir if self.lock_dir is not None else default_lock_dir()
                path = directory.parent / "race-state" / self._identity / "statistics.json"
                payload["statistics_path"] = str(path.resolve())
            generation.control.send(ControlCommand(generation.number, kind, payload).to_payload())
            generation.command_kinds.add(kind)
            return kind == requested_kind
        except (EOFError, OSError):
            return False

    def _identity_validated(self, generation: _Generation, payload: dict[str, Any]) -> None:
        if not self._may_proceed(generation):
            self._command(generation, "stop")
            return
        user_id, tournament_id = payload.get("user_id"), payload.get("tournament_id")
        if not isinstance(user_id, str) or not user_id or tournament_id != self.config.competition.room_id:
            self.fatal_reason = generation.stop_reason = "identity"
            self._log("identity_rejected", {"reason": "invalid_verified_identity"}, level=logging.ERROR)
            return
        key = identity_key(self.config.server.url, tournament_id, user_id)
        if self._identity is not None and self._identity != key:
            self.fatal_reason = generation.stop_reason = "identity"
            self._log("identity_rejected", {"reason": "participant_changed_after_restart"}, level=logging.ERROR)
            return
        if self._identity_lock is None:
            lock = RaceIdentityLock(key, directory=self.lock_dir)
            try:
                lock.acquire()
            except DuplicateRaceError:
                self.fatal_reason = generation.stop_reason = "duplicate_instance"
                self._log("duplicate_instance", {"reason": "participant_already_managed"}, level=logging.ERROR)
                return
            self._identity_lock, self._identity = lock, key
        # Acquiring/creating the OS lock can be interrupted. Recheck directly
        # before authorization, and again at the control-channel send boundary.
        if not self._may_proceed(generation):
            self._command(generation, "stop")
            return
        self._log("identity_validated", {"user_id": user_id, "tournament_id": tournament_id})
        generation.identity_approved = self._command(generation, "proceed")

    def _drain_events(self, generation: _Generation) -> None:
        while True:
            try:
                event = generation.events.get_nowait()
            except queue.Empty:
                return
            if not isinstance(event, ChildEvent):
                generation.stop_reason = "invalid_control_event"
                continue
            if event.kind == "run_started":
                generation.run_id = event.payload.get("run_id")
                generation.work_dir = event.payload.get("work_dir")
                self._index.submit("run_started", generation.number, generation.pid,
                                   run_id=generation.run_id, work_dir=generation.work_dir)
                self._log(event.kind, event.payload)
            elif event.kind == "identity_validated":
                self._identity_validated(generation, event.payload)
            elif event.kind == "terminal":
                reason = event.payload.get("reason")
                if reason not in TERMINAL_REASONS:
                    generation.stop_reason = "invalid_terminal_event"
                    continue
                # Latch before acknowledging: finalization failures can never
                # erase authoritative knowledge that playing has ended.
                self.terminal_reason = reason
                self._log("terminal", event.payload)
                self._command(generation, "ack_terminal")
            elif event.kind == "coordinator_active":
                if not generation.identity_approved:
                    generation.stop_reason = "coordinator_before_identity"
                    continue
                generation.connected_at = time.monotonic()
                self._log("reconnected" if self.restart_count else "connected", {
                    "old_pid": self._previous_pid, "new_pid": generation.pid,
                })
            elif event.kind == "failure":
                generation.failure = event.payload
                if event.payload.get("category") in FATAL_FAILURE_CATEGORIES:
                    self.fatal_reason = event.payload["category"]
                self._log("run_failure", event.payload, level=logging.ERROR)
            elif event.kind == "logs_incomplete":
                self._log("logs_incomplete", event.payload, level=logging.WARNING)
            elif event.kind == "result":
                try:
                    generation.result = run_result_from_payload(event.payload)
                    generation.result_at = time.monotonic()
                    self.last_result = generation.result
                except (KeyError, TypeError, ValueError):
                    generation.stop_reason = "invalid_result_event"
            elif event.kind == "final_state":
                if self.terminal_reason is not None:
                    self._log("race_state", event.payload)
            else:
                self._log(event.kind, event.payload)

    def _publish_health(self, generation: _Generation, *, state: str | None = None) -> None:
        update = getattr(self.log_manager, "update_health", None)
        if not callable(update):
            return
        with generation.telemetry_lock:
            payload = {"state": state or ("connected" if generation.connected_at is not None else "starting"),
                       "started_at": generation.started_at, "observed_at": time.monotonic(),
                       "heartbeat_at": generation.heartbeat_at, "snapshot": generation.health,
                       "telemetry_error": generation.telemetry_error,
                       "heartbeat_timeout_seconds": self.config.supervision.heartbeat_timeout_seconds,
                       "stall_timeout_seconds": self.config.supervision.stall_timeout_seconds}
        update(payload, generation=generation.number)

    def _health_fault(self, generation: _Generation, now: float) -> dict[str, Any] | None:
        from .race_health import evaluate_health
        with generation.telemetry_lock:
            heartbeat_at = generation.heartbeat_at
            snapshot = generation.health
            telemetry_error = generation.telemetry_error
            dropped_logs = generation.dropped_logs
        if dropped_logs > generation.reported_dropped_logs:
            self._log("worker_logs_dropped", {"dropped_logs": dropped_logs}, level=logging.WARNING)
            generation.reported_dropped_logs = dropped_logs
        if telemetry_error is not None:
            return {"reason": telemetry_error}
        startup_over = now - generation.started_at >= self.timing.startup_grace
        heartbeat_required = generation.connected_at is not None or startup_over
        heartbeat_reference = heartbeat_at if heartbeat_at is not None else generation.started_at
        if heartbeat_required and now - heartbeat_reference > self.config.supervision.heartbeat_timeout_seconds:
            return {"reason": "heartbeat_timeout", "last_heartbeat_at": heartbeat_at}
        if snapshot is None:
            # The control and health pipes are intentionally independent. A
            # fast identity handshake can overtake the first health snapshot.
            return {"reason": "startup_progress_missing"} if startup_over else None
        try:
            fault = evaluate_health(snapshot, now, self.config.supervision.stall_timeout_seconds)
        except (KeyError, TypeError, ValueError):
            return {"reason": "invalid_health_snapshot"}
        if fault is not None:
            return fault
        if generation.connected_at is not None:
            if generation.healthy_since is None:
                generation.healthy_since = now
            if self.failure_streak and now - generation.healthy_since >= self.timing.healthy_reset:
                self.failure_streak = 0
                self._log("restart_backoff_reset", {"healthy_seconds": self.timing.healthy_reset})
        return None

    def _stop_generation(self, generation: _Generation, reason: str) -> None:
        generation.stop_reason = reason
        self._publish_health(generation, state="stopping")
        self._log("child_stop_requested", {"reason": reason, "grace_seconds": self.timing.stop_grace},
                  level=logging.WARNING)
        if reason not in {"interrupted", "duplicate_instance", "identity", "supervisor_stopping"}:
            self._command(generation, "dump_stacks")
        self._command(generation, "stop")
        deadline = time.monotonic() + self.timing.stop_grace
        while generation.process.is_alive() and time.monotonic() < deadline:
            self._drain_events(generation)
            self._publish_health(generation, state="stopping")
            generation.process.join(timeout=self.timing.poll_interval)
        if generation.process.is_alive():
            self._log("child_force_killed", {"reason": reason}, level=logging.ERROR)
            generation.guard.kill()
            generation.process.join(timeout=5.0)
            if generation.process.is_alive():
                generation.process.kill()
                generation.process.join(timeout=5.0)
            if generation.process.is_alive():
                raise RuntimeError("old race process did not terminate; replacement is prohibited")

    def _finish_generation(self, generation: _Generation) -> None:
        # Kill any remaining descendants before releasing the generation/job.
        generation.guard.kill()
        generation.process.join(timeout=0.2)
        if not generation.guard.wait_empty():
            raise RuntimeError("race process tree did not terminate; replacement is prohibited")
        deadline = time.monotonic() + self.timing.control_drain_timeout
        while not generation.control_eof.is_set() and time.monotonic() < deadline:
            self._drain_events(generation)
            generation.control_eof.wait(timeout=min(self.timing.poll_interval, max(0.0, deadline - time.monotonic())))
        self._drain_events(generation)
        for connection in (generation.control, generation.telemetry, generation.logs):
            connection.close()
        for thread in generation.readers:
            thread.join(timeout=0.2)
        # A decoder may enqueue a result while close/join is taking place.
        # EOF is published only after the reader's final queue insertion.
        self._drain_events(generation)
        if not generation.control_eof.is_set():
            generation.control_drain_incomplete = True
            self._log("control_drain_incomplete", {"reason": "control_reader_did_not_finish"}, level=logging.ERROR)
        generation.guard.close()
        self._previous_pid = generation.pid
        generation.exit_code = generation.process.exitcode
        generation.process.close()
        generation.finished = True

    def _monitor(self, generation: _Generation) -> str:
        reason = "unexpected_exit"
        while generation.process.is_alive():
            self._drain_events(generation)
            self._publish_health(generation)
            if self._stopping():
                reason = "interrupted"
                self._stop_generation(generation, reason)
                break
            if generation.stop_reason is not None:
                reason = generation.stop_reason
                self._stop_generation(generation, reason)
                break
            if generation.control_eof.is_set() and generation.result_at is None:
                reason = "control_channel_closed"
                self._stop_generation(generation, reason)
                break
            now = time.monotonic()
            # The result is the final business event. Its IPC threads may now
            # stop normally; only the bounded process-exit deadline applies.
            fault = self._health_fault(generation, now) if generation.result_at is None else None
            if fault is not None:
                reason = fault.get("reason", "stalled")
                self._log("child_unhealthy", fault, level=logging.ERROR)
                self._stop_generation(generation, reason)
                break
            if generation.result_at is not None and now - generation.result_at >= self.timing.stop_grace:
                reason = "shutdown_stalled"
                self._stop_generation(generation, reason)
                break
            self._wait_for_stop(self.timing.poll_interval)
        self._finish_generation(generation)
        if reason == "unexpected_exit" and generation.exit_code == 0 and generation.result is not None:
            reason = self.terminal_reason or "run_returned"
        elif generation.failure is not None and reason == "unexpected_exit":
            reason = generation.failure.get("failure_code", "run_failed")
        self._log("child_exited", {"exit_code": generation.exit_code, "reason": reason},
                  level=logging.INFO if self.terminal_reason or self._stopping() else logging.ERROR)
        self._index.submit("child_exited", generation.number, generation.pid,
                           run_id=generation.run_id, work_dir=generation.work_dir,
                           artifact_dir=str(generation.result.artifact_dir) if generation.result else None,
                           exit_code=generation.result.exit_code if generation.result else generation.exit_code,
                           process_exit_code=generation.exit_code, reason=self.terminal_reason or reason)
        return reason

    def _result(self, exit_code: int, reason: str) -> RaceSupervisionResult:
        self._log("supervisor_stopped", {"reason": reason, "exit_code": exit_code})
        return RaceSupervisionResult(self.supervision_id, self.last_result, exit_code,
                                     reason, self.restart_count, self.log_dir)

    def _run(self) -> RaceSupervisionResult:
        while not self._stopping():
            generation = self._spawn()
            reason = self._monitor(generation)
            if self._stopping():
                return self._result(130, "interrupted")
            if self.terminal_reason is not None:
                code = generation.result.exit_code if generation.result is not None else 3
                if generation.control_drain_incomplete:
                    code = 3
                if code == 0 and self.restart_count:
                    code = 2
                return self._result(code, self.terminal_reason)
            if self.fatal_reason is not None:
                return self._result(1, self.fatal_reason)
            delay = self.timing.restart_delays[min(self.failure_streak, len(self.timing.restart_delays) - 1)]
            self.failure_streak += 1
            self._log("restart_scheduled", {"reason": reason, "delay_seconds": delay,
                                            "next_generation": generation.number + 1,
                                            "old_pid": generation.pid}, level=logging.WARNING)
            if self._wait_for_stop(delay):
                return self._result(130, "interrupted")
            self.restart_count += 1
        return self._result(130, "interrupted")

    def run(self) -> RaceSupervisionResult:
        from .race_logging import RaceLogManager
        if self.log_manager is None:
            self.log_manager = RaceLogManager(self.log_dir, (self.config.participant.token,), self.supervision_id)
        self._index = RunsIndex(self.log_dir / "runs.jsonl", self.supervision_id,
                                (self.config.participant.token,), self._log)
        previous_handlers: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            signals = (signal.SIGINT, signal.SIGTERM)
            if hasattr(signal, "SIGBREAK"):
                signals += (signal.SIGBREAK,)
            for signum in signals:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._on_stop_signal)
        try:
            self._log("supervisor_started", {"room_id": self.config.competition.room_id})
            return self._run()
        except KeyboardInterrupt:
            self.stop.set()
            return self._result(130, "interrupted")
        except Exception as error:
            self._log("supervisor_failed", {"error_type": type(error).__name__}, level=logging.ERROR)
            return self._result(1, "supervisor_failed")
        finally:
            generation = self._current
            try:
                if generation is not None and not generation.finished and generation.process.is_alive():
                    self._stop_generation(generation, "supervisor_stopping")
                    self._finish_generation(generation)
            finally:
                if generation is not None:
                    generation.guard.close()
                if self._identity_lock is not None:
                    self._identity_lock.close()
                for signum, previous in previous_handlers.items():
                    signal.signal(signum, previous)
                self._index.close(timeout=2.0)
                self.log_manager.close(timeout=2.0)


def run_supervised_race(config: RaceConfig, *, stop: StopSignal | None = None) -> RaceSupervisionResult:
    return RaceSupervisor(config, stop=stop).run()
