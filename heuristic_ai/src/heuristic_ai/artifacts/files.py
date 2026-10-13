"""One-owner artifact files. Published files are never reopened for writing."""
from __future__ import annotations

import ctypes
import hashlib
import io
import os
import stat
import threading
from pathlib import Path
from typing import Callable, Iterator


def checked_path(path: Path, root: Path, *, allow_root: bool = False) -> Path:
    """Reject escapes and reparse points before destructive or exclusive I/O."""
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    if path == root:
        if not allow_root:
            raise ValueError("artifact path must be inside its ownership root")
    elif not path.is_relative_to(root):
        raise ValueError("artifact path escapes its ownership root")
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("artifact paths must not contain reparse points")
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and not resolved.is_relative_to(resolved_root):
        raise ValueError("artifact path escapes its resolved ownership root")
    return path


if os.name == "nt":
    from ctypes import wintypes as W

    _K = ctypes.WinDLL("kernel32", use_last_error=True)
    _K.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, W.LPVOID, W.DWORD, W.DWORD, W.HANDLE]
    _K.CreateFileW.restype = W.HANDLE
    _K.ReadFile.argtypes = [W.HANDLE, W.LPVOID, W.DWORD, ctypes.POINTER(W.DWORD), W.LPVOID]
    _K.ReadFile.restype = W.BOOL
    _K.WriteFile.argtypes = [W.HANDLE, W.LPCVOID, W.DWORD, ctypes.POINTER(W.DWORD), W.LPVOID]
    _K.WriteFile.restype = W.BOOL
    _K.SetFilePointerEx.argtypes = [W.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), W.DWORD]
    _K.SetFilePointerEx.restype = W.BOOL
    _K.GetFileSizeEx.argtypes = [W.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
    _K.GetFileSizeEx.restype = W.BOOL
    _K.FlushFileBuffers.argtypes = [W.HANDLE]
    _K.FlushFileBuffers.restype = W.BOOL
    _K.CloseHandle.argtypes = [W.HANDLE]
    _K.CloseHandle.restype = W.BOOL
    _K.SetFileInformationByHandle.argtypes = [W.HANDLE, ctypes.c_int, W.LPVOID, W.DWORD]
    _K.SetFileInformationByHandle.restype = W.BOOL
    _K.GetFinalPathNameByHandleW.argtypes = [W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD]
    _K.GetFinalPathNameByHandleW.restype = W.DWORD
    class _FileInformation(ctypes.Structure):
        _fields_ = [("attributes", W.DWORD), ("created", W.FILETIME),
                    ("accessed", W.FILETIME), ("written", W.FILETIME),
                    ("volume", W.DWORD), ("size_high", W.DWORD),
                    ("size_low", W.DWORD), ("links", W.DWORD),
                    ("index_high", W.DWORD), ("index_low", W.DWORD)]

    _K.GetFileInformationByHandle.argtypes = [W.HANDLE, ctypes.POINTER(_FileInformation)]
    _K.GetFileInformationByHandle.restype = W.BOOL

    class _Disposition(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    def _checked(success: int) -> None:
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())

    def _windows_path(path: Path) -> str:
        value = str(path)
        if value.startswith("\\\\?\\"):
            return value
        return "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value


class OwnedArtifactFile:
    """A non-inheritable file handle with OPEN/SEALED/FAULTED/CLOSED states.

    Windows holds share_mode=0 for its entire lifetime. Temporary handles have
    DELETE access but are never delete-on-close until cleanup is authorized.
    """

    def bind_health_progress(self, progress: Callable[[str], None] | None) -> None:
        self._health_progress = progress

    def _health_checkpoint(self, label: str) -> None:
        progress = getattr(self, "_health_progress", None)
        if progress is not None:
            progress(label)

    def __init__(self, path: Path, *, temporary: bool = False,
                 existing: bool = False, directory: bool = False,
                 ownership_root: Path | None = None, shared_read: bool = False) -> None:
        self.path = Path(os.path.abspath(path))
        self.root = Path(os.path.abspath(ownership_root or self.path.parent))
        checked_path(self.path, self.root, allow_root=directory)
        self.temporary = temporary
        self.directory = directory
        self.state = "SEALED" if existing or directory else "OPEN"
        self._lock = threading.RLock()
        self._handle: int | None = None
        self._delete_pending = False
        self._written = 0
        self.synced_size = 0
        if shared_read and (not existing or temporary or directory):
            raise ValueError("shared reading is only for published immutable files")
        if os.name == "nt":
            access = 0x80000000 | (0 if existing else 0x40000000)
            if temporary:
                access |= 0x00010000
            handle = _K.CreateFileW(
                # Directory enumeration uses a separate metadata handle. Permit
                # listing, but keep other directory writers/deleters out. Data
                # files always remain share_mode=0.
                _windows_path(self.path), access, 1 if directory or shared_read else 0, None,
                3 if existing or directory else 1,
                0x02000000 if directory else 0x80, None,
            )
            if handle == W.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = handle
        else:
            flags = os.O_RDONLY if existing or directory else os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            if directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            self._handle = os.open(self.path, flags, 0o600)
            os.set_inheritable(self._handle, False)
        try:
            self._identity = self._actual_path()
            if os.path.normcase(str(self._identity)) != os.path.normcase(str(self.path.resolve())):
                raise ValueError("opened artifact does not match its requested path")
            if not directory:
                self._written = self.size()
                self.synced_size = self._written
        except BaseException:
            self.close()
            raise

    @classmethod
    def create_directory(cls, path: Path, *, ownership_root: Path,
                         temporary: bool = True) -> OwnedArtifactFile:
        checked_path(path, ownership_root, allow_root=True)
        path.mkdir(exist_ok=False)
        return cls(path, temporary=temporary, existing=True, directory=True,
                   ownership_root=ownership_root)

    def _require_handle(self) -> int:
        if self._handle is None:
            raise ValueError("artifact handle is closed")
        return self._handle

    def _actual_path(self) -> Path:
        handle = self._require_handle()
        if os.name != "nt":
            current = os.fstat(handle)
            requested = self.path.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (requested.st_dev, requested.st_ino):
                raise ValueError("artifact identity changed")
            return self.path.resolve()
        buffer = ctypes.create_unicode_buffer(32768)
        size = _K.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if not size or size >= len(buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        value = buffer.value
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return Path(value)

    def size(self) -> int:
        handle = self._require_handle()
        if self.directory:
            raise ValueError("directory does not have an artifact byte length")
        if os.name == "nt":
            value = ctypes.c_longlong()
            _checked(_K.GetFileSizeEx(handle, ctypes.byref(value)))
            return value.value
        return os.fstat(handle).st_size

    def file_identity(self) -> str:
        if os.name == "nt":
            information = _FileInformation()
            _checked(_K.GetFileInformationByHandle(self._require_handle(), ctypes.byref(information)))
            return f"nt:{information.volume:x}:{information.index_high:x}:{information.index_low:x}"
        information = os.fstat(self._require_handle())
        return f"posix:{information.st_dev:x}:{information.st_ino:x}"

    def _seek(self, offset: int) -> None:
        handle = self._require_handle()
        if os.name == "nt":
            _checked(_K.SetFilePointerEx(handle, offset, None, 0))
        else:
            os.lseek(handle, offset, os.SEEK_SET)

    def _write(self, data: bytes) -> int:
        handle = self._require_handle()
        if os.name == "nt":
            written = W.DWORD()
            _checked(_K.WriteFile(handle, data, len(data), ctypes.byref(written), None))
            return written.value
        return os.write(handle, data)

    def append(self, data: bytes, *, sync: bool = True) -> int:
        if not isinstance(data, bytes):
            raise TypeError("artifact writes require immutable bytes")
        with self._lock:
            if self.state != "OPEN" or self.directory:
                raise RuntimeError("artifact is not open for append")
            start = self._written
            try:
                self._seek(start)
                remaining = memoryview(data)
                while remaining:
                    chunk = remaining[:1024 * 1024].tobytes()
                    count = self._write(chunk)
                    if count <= 0 or count > len(chunk):
                        raise OSError("artifact write made no valid progress")
                    self._written += count
                    remaining = remaining[count:]
                    self._health_checkpoint("artifact.write_chunk")
                if sync:
                    self.sync()
            except BaseException:
                self.state = "FAULTED"
                raise
            return start

    def sync(self) -> None:
        with self._lock:
            if self.state != "OPEN":
                raise RuntimeError("only an open artifact can be synced")
            try:
                if os.name == "nt":
                    _checked(_K.FlushFileBuffers(self._require_handle()))
                else:
                    os.fsync(self._require_handle())
                self.synced_size = self._written
                self._health_checkpoint("artifact.synced")
            except BaseException:
                self.state = "FAULTED"
                raise

    def seal(self) -> None:
        with self._lock:
            if self.state == "OPEN":
                if self.synced_size != self._written:
                    self.sync()
                self.state = "SEALED"
            elif self.state not in {"SEALED", "FAULTED"}:
                raise RuntimeError("cannot seal a closed artifact")

    def read_at(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0:
            raise ValueError("negative artifact read range")
        with self._lock:
            if self.directory:
                raise ValueError("cannot read a directory as a file")
            self._seek(offset)
            if os.name == "nt":
                buffer = ctypes.create_string_buffer(length)
                read = W.DWORD()
                _checked(_K.ReadFile(self._require_handle(), buffer, length, ctypes.byref(read), None))
                return buffer.raw[:read.value]
            return os.read(self._require_handle(), length)

    def chunks(self, *, limit: int | None = None, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        end = self.size() if limit is None else limit
        position = 0
        while position < end:
            block = self.read_at(position, min(chunk_size, end - position))
            if not block:
                raise ValueError("artifact ended before its declared length")
            position += len(block)
            yield block

    def digest(self, *, limit: int | None = None) -> str:
        digest = hashlib.sha256()
        for block in self.chunks(limit=limit):
            digest.update(block)
            self._health_checkpoint("artifact.hash_chunk")
        return digest.hexdigest()

    def reader(self, *, limit: int | None = None) -> io.BufferedReader:
        return io.BufferedReader(_HandleReader(self, self.size() if limit is None else limit))

    def mark_delete(self, *, ownership_root: Path) -> None:
        with self._lock:
            if not self.temporary or self.state not in {"SEALED", "FAULTED"}:
                raise RuntimeError("only sealed temporary artifacts can be deleted")
            checked_path(self.path, ownership_root, allow_root=self.directory)
            actual = self._actual_path()
            if os.path.normcase(str(actual)) != os.path.normcase(str(self._identity)):
                raise ValueError("temporary artifact identity changed")
            if os.name == "nt":
                flag = _Disposition(1)
                _checked(_K.SetFileInformationByHandle(self._require_handle(), 4,
                                                     ctypes.byref(flag), ctypes.sizeof(flag)))
            self._delete_pending = True

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            if os.name == "nt":
                _checked(_K.CloseHandle(handle))
            else:
                os.close(handle)
            self._handle = None
            self.state = "CLOSED"
            if os.name != "nt" and self._delete_pending:
                checked_path(self.path, self.root, allow_root=self.directory)
                if self.directory:
                    self.path.rmdir()
                else:
                    self.path.unlink()

    def __enter__(self) -> OwnedArtifactFile:
        return self

    def __exit__(self, kind: object, error: object, traceback: object) -> None:
        try:
            self.close()
        except BaseException:
            if error is None:
                raise

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


class _HandleReader(io.RawIOBase):
    """Independent read cursor; closing it never closes the owning handle."""

    def __init__(self, owner: OwnedArtifactFile, limit: int) -> None:
        self.owner = owner
        self.limit = limit
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.limit)
        if whence not in {0, 1, 2} or target < 0:
            raise ValueError("invalid artifact seek")
        self.position = target
        return target

    def readinto(self, buffer: bytearray) -> int:
        length = min(len(buffer), max(0, self.limit - self.position))
        if not length:
            return 0
        data = self.owner.read_at(self.position, length)
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)
