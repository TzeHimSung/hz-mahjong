"""Immutable, bounded audit receipts owned by the One-Ply Expectation precompute process."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from pathlib import Path
import gzip
import hashlib
import io
import json
import pickle
import sys
import uuid

from .errors import AuditIntegrityError
from .models import NATIVE_ENABLED
try:
    from .native import _one_ply_native
except ImportError:
    _one_ply_native = None

MAX_BLOB_BYTES = 32 * 1024 * 1024
RUN_AUDIT_BYTES = 256 * 1024 * 1024
_PLAIN_CONTAINERS = (dict, MappingProxyType, tuple, list)


class AuditWriteError(RuntimeError):
    pass


def canonical(value):
    """Stable structural representation, independent of aliases and dict order."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(map(canonical, value))
    return value


def semantic_digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                             allow_nan=False,
                             default=lambda item: dict(item) if isinstance(item, Mapping) else repr(item))
    except (ValueError, TypeError) as error:
        raise AuditIntegrityError('invalid handoff semantic data') from error
    return hashlib.sha256(encoded.encode('utf8')).hexdigest()


def decision_value_digest(decision):
    return action_values_digest(decision.action, decision.details)


def action_values_digest(action, details):
    fields = {"action", "numerator", "denominator", "expected", "static_key"}
    roots = [{key: value for key, value in row.items() if key in fields}
             for row in details.get("candidate_evaluations", ())]
    return semantic_digest((action, details.get("selected_expected_value"), roots,
                            details.get("search", {}).get("completed_depth", 1)))


def plain(value, memo=None):
    if not isinstance(value, _PLAIN_CONTAINERS):
        return value
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, (dict, MappingProxyType)):
        result = {}
        memo[id(value)] = result
        result.update((k, plain(v, memo) if isinstance(v, _PLAIN_CONTAINERS) else v) for k, v in value.items())
    else:
        result = tuple(plain(v, memo) if isinstance(v, _PLAIN_CONTAINERS) else v for v in value)
        memo[id(value)] = result
    return result


def retained_size(value: object) -> int:
    """Conservative retained bytes: shared children are deliberately overcounted.

    All callers supply acyclic, internally constructed trees. Counting shared
    strings/integers again avoids a global ownership registry which itself
    would retain evicted objects. This is not a process RSS measurement.
    """
    if NATIVE_ENABLED.get() and _one_ply_native is not None:
        return _one_ply_native.retained_size(value)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(retained_size(k) + retained_size(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(retained_size(item) for item in value)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return size
    return size + 1024  # Match the native allowance for tiny frozen state records.


@dataclass(frozen=True)
class BlobRef:
    kind: str
    key: tuple
    path: str
    sha256: str
    size: int
    raw_size: int
    semantic_sha256: str

    def read(self, *, check=lambda: None):
        check()
        if not 0 < self.size <= MAX_BLOB_BYTES or not 0 < self.raw_size <= MAX_BLOB_BYTES:
            raise AuditIntegrityError('handoff blob size out of bounds')
        with Path(self.path).open('rb') as file:
            payload = file.read(self.size + 1)
        if len(payload) != self.size or hashlib.sha256(payload).hexdigest() != self.sha256:
            raise AuditIntegrityError('handoff blob checksum mismatch')
        check()
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as file:
                raw = file.read(self.raw_size + 1)
            if len(raw) != self.raw_size:
                raise AuditIntegrityError('handoff decompressed size mismatch')
            result = pickle.loads(raw)
        except (OSError, EOFError, pickle.UnpicklingError) as error:
            raise AuditIntegrityError('invalid handoff encoding') from error
        check()
        return result


class AuditStore:
    """Run-owned files: published refs survive until the table's safe detach."""
    def __init__(self, directory, limit=RUN_AUDIT_BYTES):
        self.directory = Path(directory)
        self.limit = limit
        self.size = 0
        self.peak_bytes = 0
        self.peak_files = 0
        self.files = {}
        self.refused = 0

    def write(self, handle, kind, key, value, *, digest=None, check=lambda: None):
        check()
        # Reserve for the serializer, gzip buffer and immutable reconstructed copy.
        if retained_size(value) * 3 > MAX_BLOB_BYTES:
            self.refused += 1
            return None
        try:
            raw = pickle.dumps(value, protocol=5)
        except TypeError:
            # Wire payloads are already plain. Only frozen audit fragments need
            # thawing; avoid walking every complete decision a second time.
            value = plain(value)
            if retained_size(value) * 3 > MAX_BLOB_BYTES:
                self.refused += 1
                return None
            raw = pickle.dumps(value, protocol=5)
        if len(raw) > MAX_BLOB_BYTES:
            self.refused += 1
            return None
        check()
        payload = gzip.compress(raw, compresslevel=1, mtime=0)
        if self.size + len(payload) > self.limit or len(payload) > MAX_BLOB_BYTES:
            self.refused += 1
            return None
        folder = self.directory / hashlib.sha256(handle.encode()).hexdigest()
        path = folder / (uuid.uuid4().hex + '.pickle.gz')
        try:
            folder.mkdir(exist_ok=True)
            check()
            path.write_bytes(payload)
        except OSError as error:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise AuditWriteError(type(error).__name__) from error
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self.size += len(payload)
        self.files.setdefault(handle, {})[path] = len(payload)
        self.peak_bytes = max(self.peak_bytes, self.size)
        self.peak_files = max(self.peak_files, sum(map(len, self.files.values())))
        return BlobRef(kind, key, str(path), hashlib.sha256(payload).hexdigest(),
                       len(payload), len(raw), digest or semantic_digest(value))

    def detach(self, handle):
        for path, size in self.files.pop(handle, {}).items():
            if path.resolve().parent.parent != self.directory.resolve():
                raise AuditIntegrityError('audit file outside run directory')
            path.unlink(missing_ok=True)
            self.size -= size


class PrefetchVersions:
    """Ordered activation/revocation; receiving a result never activates a task."""
    def __init__(self):
        self.control_seq = 0
        self.highest_version = 0
        self.active = {}

    def apply(self, payload):
        seq, kind, entries = payload
        if seq <= self.control_seq:
            return ()
        if kind not in {'activate', 'invalidate'} or len(entries) > 192:
            raise AuditIntegrityError('invalid prefetch control')
        if any(type(version) is not int or version < 1 for _, version in entries):
            raise AuditIntegrityError('invalid task version')
        removed = []
        watermark = self.highest_version
        for signature, version in entries:
            if kind == 'activate':
                if version > watermark or self.active.get(signature) == version:
                    if signature in self.active and self.active[signature] != version:
                        removed.append(signature)
                    self.active[signature] = version
            elif self.active.get(signature) == version:
                self.active.pop(signature)
                removed.append(signature)
            self.highest_version = max(self.highest_version, version)
        self.control_seq = seq
        return tuple(removed)

    def accepts(self, signature, version):
        return self.active.get(signature) == version
