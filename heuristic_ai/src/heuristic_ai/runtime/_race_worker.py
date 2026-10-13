"""Private spawn target and nonblocking runtime-observer IPC adapter."""
from __future__ import annotations

from collections.abc import Mapping
import codecs
import faulthandler
import io
import logging
import multiprocessing
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from typing import Any

from .process_guard import install_child_process_guard
from .supervision_contracts import (
    ChildEvent, ControlCommand, HEARTBEAT_INTERVAL_SECONDS, RaceRuntimeStatus,
    STARTUP_GRACE_SECONDS, STOP_GRACE_SECONDS, plain_value,
    race_config_from_payload, run_result_to_payload,
)


class WorkerObserver:
    """Callbacks enqueue messages; separate threads own each IPC transport."""

    def __init__(self, generation: int, control: Connection, telemetry: Connection,
                 logs: Connection, stop: threading.Event, heartbeat_interval: float,
                 secrets: tuple[str, ...] = ()) -> None:
        self.generation = generation
        self.control, self.telemetry, self.logs = control, telemetry, logs
        self.stop = stop
        self.secrets = secrets
        self.statistics_path: Path | None = None
        self._closed = threading.Event()
        self._proceed, self._terminal_ack = threading.Event(), threading.Event()
        self._outgoing: queue.Queue[Any] = queue.Queue(maxsize=256)
        self._log_queue: queue.Queue[Any] = queue.Queue(maxsize=1024)
        self._health_lock = threading.Lock()
        self._health: dict[str, Any] | None = None
        self._health_revision = 0
        self._dropped_logs = 0
        self._log_state_lock = threading.Lock()
        self._logs_closing = threading.Event()
        self._log_sender_done = threading.Event()
        self._pending_logs = 0
        self._log_in_flight = 0
        self._log_send_failed = False
        self._heartbeat_interval = heartbeat_interval
        self._threads = [
            threading.Thread(target=self._send_control, name="race-control-send", daemon=True),
            threading.Thread(target=self._receive_commands, name="race-control-read", daemon=True),
            threading.Thread(target=self._send_telemetry, name="race-heartbeat", daemon=True),
            threading.Thread(target=self._send_logs, name="race-log-send", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._outgoing.put_nowait((kind, plain_value(payload), None))
        except queue.Full:
            self.stop.set()
            raise RuntimeError("race control event buffer exhausted") from None

    def _send_control(self) -> None:
        sequence = 0
        while not self._closed.is_set():
            try:
                kind, payload, flushed = self._outgoing.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if kind is not None:
                    sequence += 1
                    self.control.send(ChildEvent(self.generation, sequence, kind, payload).to_payload())
                if flushed is not None:
                    flushed.set()
            except (EOFError, OSError):
                self.stop.set()
                return

    def _receive_commands(self) -> None:
        while not self._closed.is_set():
            try:
                command = ControlCommand.from_payload(self.control.recv())
                if command.generation != self.generation:
                    continue
                if command.kind == "proceed":
                    path = command.payload.get("statistics_path")
                    if path is not None:
                        if not isinstance(path, str) or not Path(path).is_absolute():
                            raise ValueError("statistics path must be absolute")
                        self.statistics_path = Path(path)
                    self._proceed.set()
                elif command.kind == "ack_terminal":
                    self._terminal_ack.set()
                elif command.kind == "stop":
                    self.stop.set()
                elif command.kind == "dump_stacks":
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            except (EOFError, OSError):
                self.stop.set()
                return
            except (TypeError, ValueError):
                self.stop.set()
                return

    def _send_telemetry(self) -> None:
        sequence = 0
        revision = -1
        next_heartbeat = 0.0
        while not self._closed.is_set():
            now = time.monotonic()
            with self._health_lock:
                snapshot, current_revision = self._health, self._health_revision
            try:
                if now >= next_heartbeat:
                    sequence += 1
                    self.telemetry.send(ChildEvent(self.generation, sequence, "heartbeat", {
                        "sent_at": now, "dropped_logs": self._dropped_logs,
                    }).to_payload())
                    next_heartbeat = now + self._heartbeat_interval
                if snapshot is not None and revision != current_revision:
                    sequence += 1
                    self.telemetry.send(ChildEvent(self.generation, sequence, "health", snapshot).to_payload())
                    revision = current_revision
            except (EOFError, OSError):
                self.stop.set()
                return
            self._closed.wait(min(0.1, self._heartbeat_interval))

    def send_log(self, kind: str, payload: dict[str, Any]) -> None:
        with self._log_state_lock:
            if self._logs_closing.is_set() or self._closed.is_set():
                self._dropped_logs += 1
                return
            try:
                self._log_queue.put_nowait((kind, payload))
                self._pending_logs += 1
            except queue.Full:
                self._dropped_logs += 1

    def _send_logs(self) -> None:
        try:
            while (not self._closed.is_set() and not self._logs_closing.is_set()) or not self._log_queue.empty():
                try:
                    kind, payload = self._log_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._log_state_lock:
                    self._log_in_flight = 1
                try:
                    self.logs.send({"kind": kind, "payload": payload})
                except (EOFError, OSError):
                    with self._log_state_lock:
                        self._log_send_failed = True
                        self._log_in_flight = 0
                    return
                with self._log_state_lock:
                    self._log_in_flight = 0
                    self._pending_logs -= 1
        finally:
            self._log_sender_done.set()

    def _await(self, event: threading.Event, timeout: float, label: str) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if self.stop.is_set():
                raise RuntimeError(f"race stopped while awaiting {label}")
            if event.wait(0.05):
                if self.stop.is_set():
                    raise RuntimeError(f"race stopped while awaiting {label}")
                return
            if time.monotonic() >= deadline:
                self.stop.set()
                raise RuntimeError(f"supervisor acknowledgement timed out: {label}")

    def run_started(self, run_id: str, work_dir: Path | None) -> None:
        self._emit("run_started", {"run_id": run_id, "work_dir": str(work_dir) if work_dir else None})

    def identity_validated(self, user_id: str, tournament_id: str) -> None:
        self._emit("identity_validated", {"user_id": user_id, "tournament_id": tournament_id})
        self._await(self._proceed, STARTUP_GRACE_SECONDS, "identity")

    def coordinator_active(self) -> None:
        self._emit("coordinator_active", {})

    def health(self, snapshot: dict[str, Any]) -> None:
        # B constructs independent plain-value snapshots. No recorder locks or
        # transport operations are acquired from this business-thread hook.
        with self._health_lock:
            self._health = snapshot
            self._health_revision += 1

    def race_state(self, status: RaceRuntimeStatus) -> None:
        self.send_log("race_state", status.to_payload())

    def final_state(self, status: RaceRuntimeStatus) -> None:
        self._emit("final_state", status.to_payload())

    def business_terminal(self, reason: str, own_ranking: Mapping[str, int] | None) -> None:
        self._emit("terminal", {"reason": reason, "own_ranking": plain_value(own_ranking)})
        self._await(self._terminal_ack, STOP_GRACE_SECONDS, "terminal")

    def run_failed(self, category: str, error_type: str, failure_code: str) -> None:
        self._emit("failure", {"category": category, "error_type": error_type, "failure_code": failure_code})

    def result(self, result: Any) -> None:
        from .race_logging import sanitize_text
        # The recorder already sanitizes its evidence. Keep exception detail in
        # the result channel under the same defense as other worker output.
        def redact_strings(value: Any) -> Any:
            if isinstance(value, str):
                return sanitize_text(value, self.secrets)
            if isinstance(value, dict):
                return {key: redact_strings(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact_strings(item) for item in value]
            return value
        self._emit("result", redact_strings(run_result_to_payload(result)))

    def close(self, *, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        # Keep heartbeat/control alive while the independent log sender drains.
        # No producer can add to the bounded queue after this point.
        with self._log_state_lock:
            self._logs_closing.set()
        self._log_sender_done.wait(timeout=min(0.5, max(0.0, timeout / 2)))
        with self._log_state_lock:
            losses = {"unsent_logs": self._pending_logs, "queued_logs": self._log_queue.qsize(),
                      "in_flight_logs": self._log_in_flight, "dropped_logs": self._dropped_logs,
                      "send_failed": self._log_send_failed}
        if losses["unsent_logs"] or losses["dropped_logs"] or losses["send_failed"]:
            self._emit("logs_incomplete", losses)
        flushed = threading.Event()
        try:
            self._outgoing.put((None, {}, flushed), timeout=max(0.0, deadline - time.monotonic()))
            flushed.wait(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Full:
            pass
        self._closed.set()
        for connection in (self.logs, self.telemetry, self.control):
            connection.close()
        for thread in self._threads:
            thread.join(timeout=min(0.2, max(0.0, deadline - time.monotonic())))


class _StreamRedactor:
    """Hold only a possible token prefix between native read boundaries."""

    def __init__(self, secrets: tuple[str, ...]) -> None:
        self.secrets = tuple(sorted((value for value in secrets if value), key=len, reverse=True))
        self.pending = ""

    def feed(self, text: str, *, final: bool = False) -> str:
        value = self.pending + text
        output: list[str] = []
        index = 0
        while index < len(value):
            match = next((secret for secret in self.secrets if value.startswith(secret, index)), None)
            if match is not None:
                output.append("<redacted>")
                index += len(match)
            elif not final and any(secret.startswith(value[index:]) for secret in self.secrets):
                break
            else:
                output.append(value[index])
                index += 1
        self.pending = value[index:]
        return "".join(output)


def _capture_native_output(observer: WorkerObserver, secrets: tuple[str, ...]) -> None:
    """Both Python prints and native fd writes enter the same redacted log sink."""
    from .race_logging import sanitize_text

    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(1), "wb", buffering=0), encoding="utf-8", write_through=True)
    sys.stderr = io.TextIOWrapper(os.fdopen(os.dup(2), "wb", buffering=0), encoding="utf-8", write_through=True)

    def read_output() -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        redactor = _StreamRedactor(secrets)
        pending = ""
        try:
            while True:
                block = os.read(read_fd, 4096)
                pending += redactor.feed(decoder.decode(block, final=not block), final=not block)
                while "\n" in pending or len(pending) > 16384 or (not block and pending):
                    if "\n" in pending:
                        line, pending = pending.split("\n", 1)
                    else:
                        line, pending = pending[:16384], pending[16384:]
                    observer.send_log("log", {"message": sanitize_text(line.rstrip("\r"), secrets),
                                             "level": logging.ERROR, "logger": "native.output"})
                if not block:
                    return
        except OSError:
            return
        finally:
            os.close(read_fd)

    threading.Thread(target=read_output, name="race-native-log", daemon=True).start()
    faulthandler.enable(file=sys.stderr, all_threads=True)


def _watch_parent() -> None:
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    wait([parent.sentinel])
    # This path must not wait for Python worker/recorder thread shutdown.
    os._exit(1)


def race_worker_main(config_payload: dict[str, Any], generation: int,
                     control: Connection, telemetry: Connection, logs: Connection,
                     supervision_id: str) -> None:
    """A module-level target so both CLI entrypoints work with spawn."""
    del supervision_id  # The parent attaches this ID to every persisted event.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    install_child_process_guard()
    threading.Thread(target=_watch_parent, name="race-parent-watch", daemon=True).start()
    stop = threading.Event()
    configured_timeout = config_payload["supervision"]["heartbeat_timeout_seconds"]
    secrets = (config_payload["participant"]["token"],)
    observer = WorkerObserver(generation, control, telemetry, logs, stop,
                              min(HEARTBEAT_INTERVAL_SECONDS, configured_timeout / 3.0), secrets)
    try:
        from .race_logging import make_worker_log_handler
        handler = make_worker_log_handler(observer.send_log, secrets)
        root_logger = logging.getLogger()
        root_logger.handlers[:] = [handler]
        root_logger.setLevel(logging.INFO)
        logging.raiseExceptions = False
        _capture_native_output(observer, secrets)
        config = race_config_from_payload(config_payload)
        from .race import run_race
        result = run_race(config, observer=observer, stop=stop)
        observer.result(result)
    except BaseException as error:
        from heuristic_ai.config.loader import ConfigError
        category = "configuration" if isinstance(error, ConfigError) else "runtime"
        observer.run_failed(category, type(error).__name__, "worker_unhandled_exception")
    finally:
        observer.close()
