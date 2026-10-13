"""Bounded, independently drained console/file logs for foreground race supervision.

No submitter performs sink I/O. A stalled terminal cannot stop disk logging or
the watchdog, and a stalled disk cannot stop terminal output or the watchdog.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TextIO

from .progress import RaceProgress, RaceProgressMessage
from .supervision_contracts import RaceRuntimeStatus


_SENSITIVE_PARTS = ("authorization", "credential", "password", "secret", "token")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:authorization|(?:access_|global_|participant_)?token|password|secret|credential)"
    r"\b[\"']?\s*[:=]\s*)(?:Bearer\s+)?(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}\]]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;\"'}\]]+")
_MAX_TEXT = 64 * 1024


def sanitize_text(value: str, secrets: Sequence[str]) -> str:
    """Redact full text before clipping, including traceback and native output."""
    if type(value) is not str:
        return "<unprintable>"
    for token in sorted({secret for secret in secrets if type(secret) is str and secret},
                        key=len, reverse=True):
        value = value.replace(token, "<redacted>")
    value = _CREDENTIAL_ASSIGNMENT.sub(r"\1<redacted>", value)
    value = _BEARER.sub("Bearer <redacted>", value)
    return value


def sanitize_payload(value: Any, secrets: Sequence[str], *, _depth: int = 0) -> Any:
    """Copy safe data; unknown object reprs never enter the logging channel."""
    if _depth > 12:
        return "<depth limit>"
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        safe = sanitize_text(value, secrets)
        return safe if len(safe) <= _MAX_TEXT else safe[:_MAX_TEXT] + " <truncated>"
    if isinstance(value, Mapping):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 128:
                result["<truncated>"] = True
                break
            if type(key) is not str:
                continue
            safe_key = sanitize_text(key, secrets)
            result[safe_key] = ("<redacted>" if any(part in key.casefold() for part in _SENSITIVE_PARTS)
                                else sanitize_payload(item, secrets, _depth=_depth + 1))
        return result
    if type(value) in (tuple, list):
        return [sanitize_payload(item, secrets, _depth=_depth + 1) for item in value[:128]]
    return "<unprintable>"


class _WorkerLogHandler(logging.Handler):
    def __init__(self, send: Callable[[str, dict[str, Any]], None], secrets: Sequence[str]) -> None:
        super().__init__()
        self.send = send
        self.secrets = tuple(secrets)
        self.dropped_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Formatting a copy also prevents a cached, unsanitized exc_text
            # from being added to the caller's record by Formatter.format().
            message = self.format(copy.copy(record))
            self.send("log", {
                "message": sanitize_payload(message, self.secrets),
                "level": record.levelno,
                "logger": sanitize_text(record.name, self.secrets),
            })
        except Exception:
            # logging.Handler.handleError prints raw record.args and exceptions
            # to stderr; never take that path for secret-bearing worker logs.
            self.dropped_count += 1

    def handleError(self, record: logging.LogRecord) -> None:
        self.dropped_count += 1


def make_worker_log_handler(send: Callable[[str, dict[str, Any]], None],
                            secrets: Sequence[str]) -> logging.Handler:
    """Forward fully formatted safe strings; send must itself be nonblocking."""
    return _WorkerLogHandler(send, secrets)


@dataclass(frozen=True, slots=True)
class _Line:
    text: str
    level: int
    important: bool


class _RaisingRotatingHandler(RotatingFileHandler):
    def _open(self) -> TextIO:
        # Keep byte accounting identical on Windows and POSIX. TextIO's
        # default newline translation would otherwise add an uncounted CR.
        return open(self.baseFilename, self.mode, encoding="utf-8", errors=self.errors, newline="\n")

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes <= 0:
            return False
        self.stream.seek(0, 2)
        # The standard handler adds a character count to a byte file offset.
        # Count encoded bytes so Chinese progress lines obey the same limit.
        size = len((self.format(record) + "\n").encode("utf-8", errors="backslashreplace"))
        return self.stream.tell() > 0 and self.stream.tell() + size >= self.maxBytes

    def handleError(self, record: logging.LogRecord) -> None:
        raise OSError("runtime log write failed") from None


class _Sink:
    def __init__(self, name: str, writer: Callable[[_Line], None], capacity: int,
                 failed: Callable[[str], None], finished: Callable[[], None] | None = None) -> None:
        self.name, self.writer, self.capacity = name, writer, capacity
        self.failed, self.finished = failed, finished
        self.condition = threading.Condition()
        self.pending: deque[_Line] = deque()
        self.closing = False
        self.deadline: float | None = None
        self.dropped_count = 0
        self.unreported_drops = 0
        self.failure_reported = False
        self.thread = threading.Thread(target=self._run, name=f"race-log-{name}", daemon=True)
        self.thread.start()

    def put(self, line: _Line) -> None:
        with self.condition:
            if self.closing:
                return
            if len(self.pending) >= self.capacity:
                if line.important:
                    ordinary = next((i for i, row in enumerate(self.pending) if not row.important), 0)
                    del self.pending[ordinary]
                else:
                    self.dropped_count += 1
                    self.unreported_drops += 1
                    return
                self.dropped_count += 1
                self.unreported_drops += 1
            self.pending.append(line)
            self.condition.notify()

    def stop(self, deadline: float) -> None:
        with self.condition:
            self.closing, self.deadline = True, deadline
            self.condition.notify_all()

    def _run(self) -> None:
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.closing)
                    if (not self.pending or
                            self.closing and self.deadline is not None and time.monotonic() >= self.deadline):
                        break
                    line = self.pending.popleft()
                    dropped, self.unreported_drops = self.unreported_drops, 0
                try:
                    if dropped:
                        self.writer(_Line(f"WARNING event=log_overflow sink={self.name} dropped={dropped}",
                                          logging.WARNING, True))
                    self.writer(line)
                    self.failure_reported = False
                except Exception:
                    if not self.failure_reported:
                        self.failure_reported = True
                        self.failed(self.name)
        finally:
            if self.finished is not None:
                try:
                    self.finished()
                except Exception:
                    pass


class RaceLogManager:
    """Parent-owned log sinks and race summary timer; submit never writes I/O."""
    def __init__(self, log_dir: Path, secrets: Sequence[str], supervision_id: str, *,
                 console: TextIO | None = None, queue_capacity: int = 256,
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5,
                 clock: Callable[[], float] = time.monotonic,
                 active_interval: float = 5.0, idle_interval: float = 5.0) -> None:
        if queue_capacity < 1 or max_bytes < 1 or backup_count < 0:
            raise ValueError("invalid race log limits")
        self.log_dir = Path(log_dir)
        self.secrets = tuple(secrets)
        self.supervision_id = sanitize_text(supervision_id, self.secrets)
        self.console = sys.stderr if console is None else console
        self._file_handler: _RaisingRotatingHandler | None = None
        self._max_bytes, self._backup_count = max_bytes, backup_count
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._finished = False
        self._exited_generation = -1
        self._terminal_generation = -1
        self._stopping_generation = -1
        self.progress = RaceProgress(clock=clock, active_interval=active_interval, idle_interval=idle_interval)
        self._sinks: dict[str, _Sink] = {}
        self._sinks["console"] = _Sink("console", self._write_console, queue_capacity, self._sink_failed)
        self._sinks["file"] = _Sink("file", self._write_file, queue_capacity, self._sink_failed, self._close_file)
        self._timer = threading.Thread(target=self._run, name="race-log-progress", daemon=True)
        self._timer.start()

    @property
    def dropped_count(self) -> int:
        return sum(sink.dropped_count for sink in self._sinks.values())

    def _write_console(self, line: _Line) -> None:
        self.console.write(line.text + "\n")
        self.console.flush()

    def _write_file(self, line: _Line) -> None:
        if self._file_handler is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._file_handler = _RaisingRotatingHandler(
                self.log_dir / "race.log", maxBytes=self._max_bytes,
                backupCount=self._backup_count, encoding="utf-8", errors="backslashreplace",
            )
            self._file_handler.setFormatter(logging.Formatter("%(message)s"))
        record = logging.LogRecord("race", line.level, "", 0, line.text, (), None)
        self._file_handler.emit(record)
        self._file_handler.flush()

    def _close_file(self) -> None:
        if self._file_handler is not None:
            self._file_handler.close()

    def _sink_failed(self, source: str) -> None:
        line = self._line("log_write_failed", f"sink={source} 日志写入失败", 0, None, 0,
                          logging.ERROR, True)
        # Do not feed failure diagnostics back into the failing sink or invoke
        # logging's own stderr error handler. At most one notice per outage.
        for name, sink in self._sinks.items():
            if name != source:
                sink.put(line)

    def _line(self, kind: str, message: str, generation: int, pid: int | None,
              restart_count: int, level: int, important: bool) -> _Line:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        text = (f"{timestamp} {logging.getLevelName(level)} supervision={self.supervision_id} "
                f"generation={generation} pid={pid if pid is not None else '未知'} "
                f"restarts={restart_count} event={kind} {message}")
        safe = sanitize_text(text, self.secrets)
        if len(safe) > _MAX_TEXT:
            safe = safe[:_MAX_TEXT] + " <truncated>"
        return _Line(safe, level, important)

    def _enqueue(self, line: _Line) -> None:
        for sink in self._sinks.values():
            sink.put(line)

    def _progress_line(self, message: RaceProgressMessage) -> _Line:
        return self._line("progress", message.message, message.generation, message.pid,
                          message.restart_count, logging.INFO, message.important)

    def update_health(self, payload: dict[str, Any], *, generation: int) -> None:
        # This snapshot comes from the parent's independent watchdog. Updating
        # presentation neither acknowledges a heartbeat nor refreshes a deadline.
        with self._lock:
            if not self._finished and generation == self.progress.generation:
                self.progress.health = payload

    def _lifecycle_progress(self, kind: str, payload: dict[str, Any], generation: int,
                            pid: int | None, restart_count: int) -> None:
        if generation < self.progress.generation or self._finished:
            return
        phase = {"supervisor_started": "starting", "child_started": "starting",
                 "terminal": "finalizing", "child_stop_requested": "stopping",
                 "child_exited": "reconnecting", "restart_scheduled": "reconnecting",
                 "supervisor_stopped": "stopped"}.get(kind)
        status = self.progress.status or RaceRuntimeStatus()
        if kind == "run_started":
            status = replace(status, run_id=payload.get("run_id"))
        elif phase is None:
            return
        else:
            if kind in {"child_exited", "child_stop_requested"} and generation == self._terminal_generation:
                phase = "finalizing"
            status = replace(status, phase=phase)
        if phase in {"reconnecting", "stopping", "stopped"} and self.progress.health is not None:
            self.progress.health = {**self.progress.health, "state": phase}
        message = self.progress.update(status, generation=generation, pid=pid, restart_count=restart_count)
        if message is not None:
            self._enqueue(self._progress_line(message))

    def submit(self, kind: str, payload: Mapping[str, Any], *, generation: int = 0,
               pid: int | None = None, restart_count: int = 0, level: int = logging.INFO) -> None:
        if self._stop.is_set():
            return
        safe = sanitize_payload(payload, self.secrets)
        if not isinstance(safe, dict):
            safe = {"message": "invalid log payload"}
        kind = sanitize_text(kind, self.secrets)
        with self._lock:
            self.progress.advance_generation(generation)
            if kind == "race_state":
                if self._finished or generation <= self._exited_generation:
                    return
                try:
                    status = RaceRuntimeStatus.from_payload(safe)
                    if (generation == self._terminal_generation
                            and status.phase not in {"finalizing", "exporting", "stopped"}):
                        return
                    if generation == self._stopping_generation and status.phase not in {"finalizing", "exporting", "stopped"}:
                        return
                    message = self.progress.update(status, generation=generation, pid=pid,
                                                   restart_count=restart_count)
                except (TypeError, ValueError):
                    line = self._line("invalid_progress", "invalid race state", generation, pid,
                                      restart_count, logging.WARNING, True)
                else:
                    line = self._progress_line(message) if message is not None else None
            else:
                if kind == "child_exited":
                    self._exited_generation = max(self._exited_generation, generation)
                if kind == "terminal":
                    self._terminal_generation = max(self._terminal_generation, generation)
                if kind == "child_stop_requested":
                    self._stopping_generation = max(self._stopping_generation, generation)
                if kind == "log":
                    message = safe.get("message", "")
                    if type(message) is not str:
                        message = "invalid log message"
                    if type(safe.get("level")) is int:
                        level = safe["level"]
                    prefix = safe.get("logger")
                    if type(prefix) is str:
                        message = f"logger={prefix} {message}"
                else:
                    try:
                        message = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    except (TypeError, ValueError):
                        message = "invalid log payload"
                line = self._line(kind, message, generation, pid, restart_count, level,
                                  kind != "log" or level >= logging.WARNING)
            # Keep the presentation state and the order observed by both
            # sinks atomic. Sink.put only takes its short queue lock; it never
            # writes I/O or calls back while holding that lock.
            if line is not None:
                self._enqueue(line)
            if kind != "race_state":
                self._lifecycle_progress(kind, safe, generation, pid, restart_count)
            if kind == "supervisor_stopped":
                self._finished = True

    def tick(self) -> None:
        """Advance presentation time; public for deterministic local tests."""
        if self._stop.is_set():
            return
        with self._lock:
            message = self.progress.tick()
            if message is not None:
                self._enqueue(self._progress_line(message))

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            self.tick()

    def close(self, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        self._stop.set()
        for sink in self._sinks.values():
            sink.stop(deadline)
        self._timer.join(max(0.0, deadline - time.monotonic()))
        for sink in self._sinks.values():
            sink.thread.join(max(0.0, deadline - time.monotonic()))

    def __enter__(self) -> RaceLogManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
