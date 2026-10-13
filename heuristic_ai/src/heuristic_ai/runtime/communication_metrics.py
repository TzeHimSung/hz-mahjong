"""Bounded, asynchronous communication evidence and host-load samples."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable

from heuristic_ai.protocol.client import redact


class HostLoad:
    def __init__(self) -> None:
        self.previous = None

    def sample(self) -> dict[str, Any]:
        row = {"kind": "host_load", "at_utc": datetime.now(timezone.utc).isoformat(),
               "monotonic": time.monotonic(), "logical_cpus": os.cpu_count(),
               "cpu_percent": None, "available_memory_bytes": None}
        if sys.platform == "win32":
            idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                current = (idle.value, kernel.value + user.value)
                if self.previous is not None and current[1] > self.previous[1]:
                    row["cpu_percent"] = 100 * (1 - (current[0] - self.previous[0]) / (current[1] - self.previous[1]))
                self.previous = current
            class Memory(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                          "virtual_total", "virtual_available", "extended")]
            memory = Memory()
            memory.length = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                row["available_memory_bytes"] = memory.available
        elif Path("/proc/stat").exists():
            fields = tuple(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
            current = (fields[3] + (fields[4] if len(fields) > 4 else 0), sum(fields[:8]))
            if self.previous is not None and current[1] > self.previous[1]:
                row["cpu_percent"] = 100 * (1 - (current[0] - self.previous[0]) / (current[1] - self.previous[1]))
            self.previous = current
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    row["available_memory_bytes"] = int(line.split()[1]) * 1024
        return row


@dataclass
class _Barrier:
    ready: threading.Event


class CommunicationJournal:
    """Telemetry never waits behind search audit serialization; overflow is visible."""

    def __init__(self, path: Path, *, sanitize: Callable[[Any], Any] = redact, capacity: int = 8192) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("x", encoding="utf-8", newline="\n")
        self._sanitize = sanitize
        self._queue: queue.Queue = queue.Queue(capacity)
        self._failure: BaseException | None = None
        self._closed = False
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._write, name="communication-journal", daemon=True)
        self._writer.start()
        self._sampler = threading.Thread(target=self._sample, name="communication-load", daemon=True)
        self._sampler.start()

    def check(self) -> None:
        if self._failure is not None:
            raise RuntimeError("communication evidence writer failed") from self._failure

    def record(self, row: dict[str, Any]) -> None:
        self.check()
        if self._closed:
            raise RuntimeError("communication journal is closed")
        payload = self._sanitize(dict(row))
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._failure = RuntimeError("communication evidence queue overflow")
            self.check()

    def _sample(self) -> None:
        sampler = HostLoad()
        while not self._stop.is_set():
            try:
                self.record(sampler.sample())
            except BaseException as error:
                self._failure = error
                return
            self._stop.wait(5)

    def _write(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if isinstance(item, _Barrier):
                    self._file.flush()
                    item.ready.set()
                elif self._failure is None:
                    self._file.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            except BaseException as error:
                self._failure = error
                if isinstance(item, _Barrier):
                    item.ready.set()
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        self.check()
        barrier = _Barrier(threading.Event())
        self._queue.put(barrier, timeout=5)
        if not barrier.ready.wait(10):
            raise RuntimeError("communication evidence flush timed out")
        self.check()

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._sampler.join(6)
        try:
            self.flush()
        finally:
            self._closed = True
            self._queue.put(None, timeout=5)
            self._writer.join(10)
            self._file.close()
        self.check()
