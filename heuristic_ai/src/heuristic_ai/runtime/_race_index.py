"""Bounded asynchronous append-only links between a supervisor and its runs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import queue
import threading
from typing import Any, Callable


class RunsIndex:
    def __init__(self, path: Path, supervision_id: str, secrets: tuple[str, ...],
                 report: Callable[..., None], *, capacity: int = 128,
                 open_file: Callable[..., Any] | None = None) -> None:
        self.path, self.supervision_id = path, supervision_id
        self.secrets, self.report = secrets, report
        self._open = open_file if open_file is not None else path.open
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=capacity)
        self._closed = threading.Event()
        self._failed = threading.Event()
        self._failure_lock = threading.Lock()
        self._thread = threading.Thread(target=self._write, name="race-run-index", daemon=True)
        self._thread.start()

    def _fail(self, reason: str, **details: Any) -> None:
        with self._failure_lock:
            if self._failed.is_set():
                return
            self._failed.set()
        self.report("run_index_incomplete", {"reason": reason, "path": str(self.path), **details},
                    level=logging.ERROR)

    def submit(self, event: str, generation: int, pid: int, **fields: Any) -> None:
        from .race_logging import sanitize_payload
        allowed = {"run_id", "work_dir", "artifact_dir", "exit_code", "process_exit_code", "reason"}
        if set(fields) - allowed:
            raise ValueError("invalid run index fields")
        if self._closed.is_set() or self._failed.is_set():
            return
        record = {"at": datetime.now(timezone.utc).isoformat(), "event": event,
                  "supervision_id": self.supervision_id, "generation": generation, "pid": pid, **fields}
        record = sanitize_payload(record, self.secrets)
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._fail("queue_full")

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._open("a", encoding="utf-8", newline="\n") as stream:
                while not self._closed.is_set() or not self._queue.empty():
                    try:
                        record = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        except Exception as error:
            self._fail("write_failed", error_type=type(error).__name__)

    def close(self, timeout: float = 2.0) -> bool:
        self._closed.set()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            self._fail("close_timeout")
        return not self._thread.is_alive() and not self._failed.is_set()
