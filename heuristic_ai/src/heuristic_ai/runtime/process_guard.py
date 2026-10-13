"""OS-backed race instance/identity locks and child process lifetime protection."""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit


class DuplicateRaceError(RuntimeError):
    """Another process owns the machine race slot or participant identity."""


_MACHINE_RACE_MUTEX = r"Global\hz-mahjong-heuristic-ai.race"
LOGGER = logging.getLogger("heuristic_ai.process_guard")


class MachineRaceLock:
    """One Windows CLI race per machine, owned by the startup thread.

    The handle is not inherited by workers. Keep this context around the entire
    supervisor run, including generation restarts and artifact finalization.
    Other platforms retain their existing identity-lock behavior.
    """

    def __init__(self) -> None:
        self._kernel: Any = None
        self._handle: Any = None
        self._owned = False

    @staticmethod
    def _error(operation: str) -> OSError:
        error = ctypes.WinError(ctypes.get_last_error())
        return OSError(f"race 全机互斥锁 {operation} 失败: {error}")

    def acquire(self) -> None:
        if os.name != "nt":
            return
        if self._handle is not None:
            raise RuntimeError("machine race lock is already acquired")
        self._kernel = _windows_api()
        # NULL security attributes create a non-inheritable handle. FALSE avoids
        # ambiguous initial ownership when multiple launchers create it at once.
        self._handle = self._kernel.CreateMutexW(None, False, _MACHINE_RACE_MUTEX)
        if not self._handle:
            self._handle = None
            raise self._error("CreateMutexW")
        try:
            result = self._kernel.WaitForSingleObject(self._handle, 0)
            if result in (0, 0x80):  # WAIT_OBJECT_0 / WAIT_ABANDONED grant ownership.
                self._owned = True
            elif result == 0x102:  # WAIT_TIMEOUT: leave the existing race alone.
                raise DuplicateRaceError("本机已有 race 模式实例运行，当前启动已取消")
            elif result == 0xFFFFFFFF:  # WAIT_FAILED
                raise self._error("WaitForSingleObject")
            else:
                raise RuntimeError(f"race 全机互斥锁返回未知等待状态: {result}")
        except BaseException as error:
            self._close_preserving_error(error)
            raise

    @staticmethod
    def _record_cleanup_error(error: BaseException, cleanup_error: OSError) -> None:
        error.add_note(str(cleanup_error))
        LOGGER.error("%s", cleanup_error)

    def _close_preserving_error(self, error: BaseException | None) -> None:
        try:
            self.close()
        except OSError as cleanup_error:
            if error is None:
                raise
            self._record_cleanup_error(error, cleanup_error)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        owned, self._owned = self._owned, False
        if handle is not None:
            release_error = None
            try:
                if owned and not self._kernel.ReleaseMutex(handle):
                    release_error = self._error("ReleaseMutex")
            finally:
                if not self._kernel.CloseHandle(handle):
                    close_error = self._error("CloseHandle")
                    if release_error is None:
                        raise close_error
                    self._record_cleanup_error(release_error, close_error)
            if release_error is not None:
                raise release_error

    def __enter__(self) -> MachineRaceLock:
        self.acquire()
        return self

    def __exit__(self, _exc_type: object, error: BaseException | None, _traceback: object) -> None:
        self._close_preserving_error(error)


def identity_key(server_url: str, tournament_id: str, user_id: str) -> str:
    parsed = urlsplit(server_url)
    hostname = (parsed.hostname or "").lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    port = parsed.port
    if port is not None and (parsed.scheme.lower(), port) not in {("http", 80), ("https", 443)}:
        hostname += f":{port}"
    server = urlunsplit((parsed.scheme.lower(), hostname, parsed.path.rstrip("/"), parsed.query, ""))
    value = json.dumps([server, tournament_id, user_id], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def default_lock_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "hz-mahjong-heuristic-ai" / "race-locks"


class RaceIdentityLock:
    """An OS lock, held across generations and independent of output/worktree."""

    def __init__(self, key: str, *, directory: Path | None = None) -> None:
        self.path = (directory if directory is not None else default_lock_dir()) / f"{key}.lock"
        self._file: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise DuplicateRaceError("another managed race already owns this participant") from None
        self._file = stream

    def close(self) -> None:
        stream, self._file = self._file, None
        if stream is not None:
            # Closing releases the kernel lock even after an unclean owner exit.
            stream.close()


def _windows_api() -> Any:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int
    specifications = {
        "CreateMutexW": ([ctypes.c_void_p, boolean, ctypes.c_wchar_p], handle),
        "WaitForSingleObject": ([handle, dword], dword),
        "ReleaseMutex": ([handle], boolean),
        "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], handle),
        "SetInformationJobObject": ([handle, ctypes.c_int, ctypes.c_void_p, dword], boolean),
        "AssignProcessToJobObject": ([handle, handle], boolean),
        "TerminateJobObject": ([handle, dword], boolean),
        "QueryInformationJobObject": ([handle, ctypes.c_int, ctypes.c_void_p, dword, ctypes.c_void_p], boolean),
        "OpenProcess": ([dword, boolean, dword], handle),
        "CloseHandle": ([handle], boolean),
    }
    for name, (arguments, result) in specifications.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result
    return kernel


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in (
        "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime",
    )] + [(name, ctypes.c_uint32) for name in (
        "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses",
    )]


class ProcessTreeGuard:
    """One fresh process group/job per generation, installed before play is allowed."""

    def __init__(self) -> None:
        self.process: Any = None
        self._kernel: Any = None
        self._job: Any = None
        if os.name == "nt":
            self._kernel = _windows_api()
            self._job = self._kernel.CreateJobObjectW(None, None)
            if not self._job:
                raise ctypes.WinError(ctypes.get_last_error())
            limits = _ExtendedLimit()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
            if not self._kernel.SetInformationJobObject(self._job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                error = ctypes.WinError(ctypes.get_last_error())
                self.close()
                raise error

    def attach(self, process: Any) -> None:
        self.process = process
        if self._job is not None:
            # Do not assume multiprocessing's sentinel handle has assignment rights.
            handle = self._kernel.OpenProcess(0x0100 | 0x0001 | 0x0400, False, process.pid)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not self._kernel.AssignProcessToJobObject(self._job, handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                self._kernel.CloseHandle(handle)

    def kill(self) -> None:
        if self._job is not None:
            if not self._kernel.TerminateJobObject(self._job, 1):
                raise ctypes.WinError(ctypes.get_last_error())
        elif self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if self.process.is_alive():
                    self.process.kill()

    def wait_empty(self, timeout: float = 5.0) -> bool:
        if self._job is None:
            return self.process is None or not self.process.is_alive()
        deadline = time.monotonic() + timeout
        while True:
            accounting = _BasicAccounting()
            if not self._kernel.QueryInformationJobObject(
                self._job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not accounting.ActiveProcesses:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

    def close(self) -> None:
        job, self._job = self._job, None
        if job is not None:
            self._kernel.CloseHandle(job)


def install_child_process_guard() -> None:
    """POSIX group setup; Windows containment is assigned by the parent."""
    if os.name != "nt":
        os.setsid()
        if sys.platform.startswith("linux"):
            # The parent sentinel watchdog also covers the death/setup race.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
                raise OSError(ctypes.get_errno(), "could not install parent-death signal")
