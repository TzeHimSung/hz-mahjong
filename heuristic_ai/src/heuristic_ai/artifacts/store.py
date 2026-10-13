"""Durable per-owner shards and bounded-memory, handle-based iteration."""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heuristic_ai.artifacts.files import OwnedArtifactFile, checked_path

_PREFIX = b'{"body":'
_MIDDLE = b',"sha256":"'
_SUFFIX = b'"}\n'
_MAX_FRAME_BYTES = 64 * 1024 * 1024
_STREAM_KINDS = frozenset({"control", "decision", "poll", "replay"})
ARTIFACT_MODES = frozenset({"train", "race", "match"})
PARTICIPANT_MODES = frozenset({"race", "match"})


def default_record_subdir(mode: str) -> str:
    if mode not in ARTIFACT_MODES:
        raise ValueError("invalid artifact mode")
    return {"train": "test_records", "race": "official_records", "match": "test_records"}[mode]


def validate_record_subdir(mode: str, value: object) -> str:
    default_record_subdir(mode)
    if not isinstance(value, str) or value not in {"test_records", "official_records", "match_records"}:
        raise ValueError("invalid artifact record subdirectory")
    if mode == "race" and value != "official_records":
        raise ValueError("race artifacts require official_records")
    # Keep persisted destinations readable for older match recovery and cleanup.
    if mode == "match" and value not in {"test_records", "match_records"}:
        raise ValueError("match artifacts require test_records or legacy match_records")
    if mode == "train" and value == "match_records":
        raise ValueError("train artifacts cannot use match_records")
    return value


class WorkDirectoryCollision(FileExistsError):
    """Only the atomic creation of this run's work directory collided."""


def json_bytes(value: object, *, pretty: bool = False) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      indent=2 if pretty else None,
                      separators=None if pretty else (",", ":")).encode("utf-8")


def legacy_line(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def encode_frame(body: Mapping[str, Any], payload_bytes: bytes | None = None) -> bytes:
    if payload_bytes is None:
        raw = json_bytes(body)
    else:
        raw = json_bytes(body)[:-1] + b',"payload":' + payload_bytes + b'}'
    if len(raw) > _MAX_FRAME_BYTES:
        raise ValueError("artifact event exceeds the 64 MiB frame limit")
    return _PREFIX + raw + _MIDDLE + hashlib.sha256(raw).hexdigest().encode("ascii") + _SUFFIX


def decode_frame(line: bytes) -> dict[str, Any]:
    tail = len(_MIDDLE) + 64 + len(_SUFFIX)
    if (not line.startswith(_PREFIX) or not line.endswith(_SUFFIX)
            or len(line) <= len(_PREFIX) + tail
            or line[-tail:-64-len(_SUFFIX)] != _MIDDLE):
        raise ValueError("invalid artifact frame envelope")
    raw = line[len(_PREFIX):-tail]
    expected = line[-64-len(_SUFFIX):-len(_SUFFIX)]
    if hashlib.sha256(raw).hexdigest().encode("ascii") != expected:
        raise ValueError("artifact frame checksum mismatch")
    try:
        body = json.loads(raw)
    except (UnicodeError, ValueError):
        raise ValueError("artifact frame contains invalid JSON") from None
    if not isinstance(body, dict) or type(body.get("format_version")) is not int or body["format_version"] != 1:
        raise ValueError("unsupported artifact frame version")
    return body


@dataclass(frozen=True, slots=True)
class FrameLocation:
    source: OwnedArtifactFile
    offset: int
    length: int
    order: int
    sequence: int
    stream_kind: str
    owner: tuple[str, ...]

    def read(self) -> dict[str, Any]:
        raw = self.source.read_at(self.offset, self.length)
        if len(raw) != self.length:
            raise ValueError("artifact frame became incomplete")
        return decode_frame(raw)


class Shard:
    def __init__(self, file: OwnedArtifactFile, *, run_id: str, mode: str,
                 kind: str, owner: tuple[str, ...], recorded_at: str,
                 create: bool = True) -> None:
        self.file = file
        self.kind = kind
        self.owner = owner
        self.run_id = run_id
        self.mode = mode
        self.sequence = 0
        self.validated_entries: int | None = None
        self.lock = threading.RLock()
        self.header = {"run_id": run_id, "mode": mode, "stream_kind": kind,
                       "owner": list(owner), "file_identity": file.file_identity()}
        if create:
            file.append(encode_frame({"format_version": 1, "seq": 0, "order": 0,
                                      "kind": "header", "recorded_at": recorded_at,
                                      "payload": self.header}))

    def append(self, kind: str, payload_bytes: bytes, recorded_at: str,
               next_order: Callable[[], int], *, sync: bool) -> int:
        with self.lock:
            order = next_order()
            frame = encode_frame({"format_version": 1, "seq": self.sequence + 1,
                                  "order": order, "kind": kind,
                                  "recorded_at": recorded_at}, payload_bytes)
            self.file.append(frame, sync=sync)
            self.sequence += 1
            return order

    def locations(self, *, recovery: bool = False,
                  warnings: list[str] | None = None) -> Iterator[FrameLocation]:
        limit = self.file.size() if recovery else self.file.synced_size
        expected_seq = 0
        last_order = 0
        with self.file.reader(limit=limit) as reader:
            while True:
                offset = reader.tell()
                line = reader.readline(_MAX_FRAME_BYTES + 256)
                if not line:
                    break
                if not line.endswith(b"\n"):
                    if reader.tell() != limit or len(line) > _MAX_FRAME_BYTES + 128:
                        raise ValueError("artifact frame exceeds its length limit")
                    if warnings is not None:
                        warnings.append(f"{self.file.path.name}: incomplete tail at byte {offset}")
                    break
                body = decode_frame(line)
                if type(body.get("seq")) is not int or body["seq"] != expected_seq:
                    raise ValueError("artifact shard sequence is not continuous")
                if type(body.get("order")) is not int:
                    raise ValueError("artifact order is invalid")
                if expected_seq == 0:
                    payload = body.get("payload")
                    if body.get("kind") != "header" or body["order"] != 0 or not isinstance(payload, dict):
                        raise ValueError("artifact shard has no valid header")
                    for key in ("run_id", "mode", "stream_kind", "owner"):
                        if payload.get(key) != self.header[key]:
                            raise ValueError("artifact shard ownership mismatch")
                    if payload.get("file_identity") != self.file.file_identity():
                        raise ValueError("artifact shard file identity changed")
                else:
                    if body["order"] <= last_order:
                        raise ValueError("artifact shard order is not increasing")
                    if not isinstance(body.get("recorded_at"), str):
                        raise ValueError("artifact timestamp is missing")
                    last_order = body["order"]
                    location = FrameLocation(self.file, offset, len(line), body["order"],
                                             expected_seq, self.kind, self.owner)
                    # A suspended iterator must not retain the decoded search tree.
                    del body, line
                    yield location
                expected_seq += 1
        if not expected_seq:
            raise ValueError("artifact shard is missing its header")
        self.validated_entries = expected_seq - 1


class ArtifactStore:
    def __init__(self, output_root: Path, mode: str, manifest: dict[str, Any],
                 *, record_subdir: str | None = None) -> None:
        self.record_subdir = validate_record_subdir(
            mode, default_record_subdir(mode) if record_subdir is None else record_subdir)
        self.output_root = Path(os.path.abspath(output_root))
        self.mode = mode
        self.run_id = manifest["run_id"]
        if not isinstance(self.run_id, str) or not self.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.run_id):
            raise ValueError("invalid artifact run identifier")
        self.final_parent = self.output_root / self.record_subdir
        self.default_output = self.final_parent / self.run_id
        self.work_parent = self.output_root / ".record_work" / mode
        checked_path(self.work_parent, self.output_root)
        self.work_parent.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.work_parent / self.run_id
        try:
            self.directory = OwnedArtifactFile.create_directory(self.work_dir, ownership_root=self.work_parent)
        except FileExistsError as error:
            raise WorkDirectoryCollision(str(self.work_dir)) from error
        self._registry_lock = threading.RLock()
        self._order_lock = threading.Lock()
        self._order = 0
        self._poll_order = 0
        self._state = "OPEN"
        self._shards: dict[tuple[str, tuple[str, ...]], Shard] = {}
        self.auxiliary: list[OwnedArtifactFile] = []
        self.failed = threading.Event()
        self._failure_listeners: list[Callable[[], None]] = []
        self.first_error: BaseException | None = None
        self.started_at = manifest["started_at"]
        self.warnings: list[str] = []
        self._metric_lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=4096)
        self._metric_counts = {"decision": 0, "poll": 0, "control": 0, "replay": 0}
        try:
            self.append("control", (), "init", {"manifest": manifest,
                        "output_root": str(self.output_root),
                        "record_subdir": self.record_subdir}, self.started_at)
        except BaseException:
            self.close()
            raise

    def bind_stop(self, stop: threading.Event) -> None:
        with self._registry_lock:
            listener = stop.set
            if listener not in self._failure_listeners:
                self._failure_listeners.append(listener)
            if self.failed.is_set():
                stop.set()

    def bind_health_progress(self, progress: Callable[[str], None]) -> None:
        self._health_progress = progress
        with self._registry_lock:
            for shard in self._shards.values():
                shard.file.bind_health_progress(progress)

    def _health_checkpoint(self, label: str) -> None:
        progress = getattr(self, "_health_progress", None)
        if progress is not None:
            progress(label)

    def _fail(self, error: BaseException) -> None:
        with self._registry_lock:
            if self.first_error is None:
                self.first_error = error
            self.failed.set()
            listeners = tuple(self._failure_listeners)
        for listener in listeners:
            listener()

    def _next_order(self, poll: bool = False) -> int:
        with self._order_lock:
            if poll:
                self._poll_order += 1
                return self._poll_order
            self._order += 1
            return self._order

    def prepare(self, kind: str, owner: tuple[str, ...] = ()) -> Shard:
        if kind not in _STREAM_KINDS or (self.mode in PARTICIPANT_MODES and kind in {"poll", "replay"}):
            raise ValueError("unsupported artifact stream")
        key = (kind, owner)
        with self._registry_lock:
            if self._state != "OPEN":
                raise RuntimeError("artifact store is sealed")
            if key in self._shards:
                return self._shards[key]
            suffix = "" if not owner else "-" + hashlib.sha256(json_bytes(list(owner))).hexdigest()[:24]
            path = self.work_dir / f"{kind}{suffix}.jsonl"
            file = OwnedArtifactFile(path, temporary=True, ownership_root=self.work_dir)
            file.bind_health_progress(getattr(self, "_health_progress", None))
            try:
                shard = Shard(file, run_id=self.run_id, mode=self.mode, kind=kind,
                              owner=owner, recorded_at=self.started_at)
            except BaseException:
                self.auxiliary.append(file)
                self._fail(RuntimeError("artifact shard initialization failed"))
                raise
            self._shards[key] = shard
            return shard

    def append(self, stream: str, owner: tuple[str, ...], kind: str,
               payload: Any, recorded_at: str, *, sync: bool = True,
               encoded_payload: bytes | None = None) -> int:
        # Render before assigning an order or acquiring an I/O lock.
        encoded = json_bytes(payload) if encoded_payload is None else encoded_payload
        start = time.perf_counter()
        try:
            shard = self.prepare(stream, owner)
            order = shard.append(kind, encoded, recorded_at,
                                 lambda: self._next_order(stream == "poll"), sync=sync)
        except BaseException as error:
            self._fail(error)
            raise
        with self._metric_lock:
            self._latencies.append((time.perf_counter() - start) * 1000)
            self._metric_counts[stream] += 1
        return order

    def synchronize(self) -> None:
        with self._registry_lock:
            shards = tuple(self._shards.values())
        errors = []
        for shard in shards:
            with shard.lock:
                try:
                    if shard.file.state == "OPEN" and shard.file.synced_size != shard.file._written:
                        shard.file.sync()
                except BaseException as error:
                    self._fail(error)
                    errors.append(error)
            self._health_checkpoint("artifact.shard_synchronized")
        if errors:
            raise errors[0]

    def seal(self) -> None:
        with self._registry_lock:
            self._state = "SEALED"
            shards = tuple(self._shards.values())
        for shard in shards:
            with shard.lock:
                try:
                    shard.file.seal()
                except BaseException as error:
                    self._fail(error)
            self._health_checkpoint("artifact.shard_sealed")

    def locations(self, *, polls: bool = False, recovery: bool = False) -> Iterator[FrameLocation]:
        if self._state != "SEALED":
            raise RuntimeError("seal all writers before merging")
        sources = [shard for shard in self._shards.values() if (shard.kind == "poll") == polls]
        streams = [iter(shard.locations(recovery=recovery, warnings=self.warnings)) for shard in sources]
        heap: list[tuple[int, int, FrameLocation]] = []
        for index, stream in enumerate(streams):
            item = next(stream, None)
            if item is not None:
                heapq.heappush(heap, (item.order, index, item))
            self._health_checkpoint("artifact.shard_opened")
        last_order = 0
        while heap:
            order, index, item = heapq.heappop(heap)
            if order <= last_order:
                raise ValueError("artifact order is duplicated across shards")
            if order != last_order + 1:
                self.warnings.append(f"{'poll' if polls else 'audit'} order gap before {order}")
            last_order = order
            yield item
            item = next(streams[index], None)
            if item is not None:
                heapq.heappush(heap, (item.order, index, item))

    def metrics(self) -> dict[str, Any]:
        with self._metric_lock:
            samples = sorted(self._latencies)
            counts = self._metric_counts.copy()
        with self._registry_lock:
            shards = tuple(self._shards.values())
        return {"counts": counts, "open_shards": len(shards),
                "synced_bytes": sum(s.file.synced_size for s in shards),
                "record_p99_ms": samples[min(len(samples)-1, (len(samples)*99+99)//100-1)] if samples else 0,
                "audit_failed": self.failed.is_set()}

    def inventory(self) -> list[dict[str, Any]]:
        """Describe confirmed prefixes before appending a terminal control event."""
        with self._registry_lock:
            shards = tuple(self._shards.values())
        return [{"path": s.file.path.name, "kind": s.kind, "owner": list(s.owner),
                 "entries": s.sequence, "bytes": s.file.synced_size,
                 "sha256": s.file.digest(limit=s.file.synced_size),
                 "file_identity": s.file.file_identity()}
                for s in sorted(shards, key=lambda item: item.file.path.name)]

    def verify_inventory(self, inventory: object, terminal: FrameLocation,
                         *, completed: bool) -> None:
        if not isinstance(inventory, list):
            raise ValueError("terminal event is missing its shard inventory")
        actual = {s.file.path.name: s for s in self._shards.values()}
        if len(inventory) != len(actual) or {i.get("path") for i in inventory if isinstance(i, dict)} != set(actual):
            raise ValueError("terminal shard inventory does not match the working directory")
        for item in inventory:
            shard = actual[item["path"]]
            size = item.get("bytes")
            entries = item.get("entries")
            if type(size) is not int or size < 1 or type(entries) is not int or entries < 0:
                raise ValueError("terminal shard checkpoint is invalid")
            if item.get("file_identity") != shard.file.file_identity() or item.get("owner") != list(shard.owner) or item.get("kind") != shard.kind:
                raise ValueError("terminal shard ownership is invalid")
            expected_entries = entries + (1 if shard.file is terminal.source else 0)
            if completed and shard.validated_entries != expected_entries:
                raise ValueError("terminal shard entry checkpoint verification failed")
            if shard.file is terminal.source:
                if size != terminal.offset or entries + 1 != terminal.sequence:
                    raise ValueError("terminal control prefix is inconsistent")
                if completed and shard.file.size() != terminal.offset + terminal.length:
                    raise ValueError("completed control shard has an unexpected tail")
            elif completed and size != shard.file.size():
                raise ValueError("completed shard has an unexpected tail")
            if size > shard.file.size() or shard.file.digest(limit=size) != item.get("sha256"):
                raise ValueError("terminal shard checkpoint verification failed")
            self._health_checkpoint("artifact.inventory_verified")

    def temporary_files(self) -> list[OwnedArtifactFile]:
        return [shard.file for shard in self._shards.values()] + list(self.auxiliary)

    def close(self) -> None:
        errors = []
        for file in [*self.temporary_files(), self.directory]:
            try:
                file.close()
                self._health_checkpoint("artifact.handle_closed")
            except BaseException as error:
                errors.append(error)
        self._state = "CLOSED"
        if errors and self.first_error is None:
            raise errors[0]

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    @classmethod
    def open_recovery(cls, work_dir: Path) -> ArtifactStore:
        work_dir = Path(os.path.abspath(work_dir))
        if work_dir.parent.name not in ARTIFACT_MODES or work_dir.parent.parent.name != ".record_work":
            raise ValueError("recovery requires a managed run working directory")
        checked_path(work_dir, work_dir.parent, allow_root=False)
        result = cls.__new__(cls)
        result.mode = work_dir.parent.name
        result.run_id = work_dir.name
        result.work_dir = work_dir
        result.work_parent = work_dir.parent
        result.output_root = work_dir.parent.parent.parent
        result._registry_lock = threading.RLock()
        result._order_lock = threading.Lock()
        result._order = result._poll_order = 0
        result._state = "SEALED"
        result._shards = {}
        result.auxiliary = []
        result.failed = threading.Event()
        result._failure_listeners = []
        result.first_error = None
        result.warnings = []
        result._metric_lock = threading.Lock()
        result._latencies = deque(maxlen=4096)
        result._metric_counts = {kind: 0 for kind in _STREAM_KINDS}
        result.directory = OwnedArtifactFile(work_dir, temporary=True, existing=True,
                                             directory=True, ownership_root=result.work_parent)
        try:
            paths = sorted(work_dir.iterdir(), key=lambda p: (p.name != "control.jsonl", p.name))
            if not paths or paths[0].name != "control.jsonl":
                raise ValueError("working directory has no control shard")
            for path in paths:
                if not path.name.endswith(".jsonl"):
                    continue
                file = OwnedArtifactFile(path, temporary=True, existing=True, ownership_root=work_dir)
                try:
                    with file.reader() as reader:
                        header = decode_frame(reader.readline(_MAX_FRAME_BYTES + 256))
                    payload = header.get("payload")
                    if path.name.startswith("attempt-"):
                        if (header.get("kind") != "export_attempt" or not isinstance(payload, dict)
                                or payload.get("run_id") != result.run_id or payload.get("mode") != result.mode
                                or payload.get("file_identity") != file.file_identity()):
                            raise ValueError("invalid export attempt ownership")
                        result.auxiliary.append(file)
                        continue
                    if (header.get("kind") != "header" or header.get("seq") != 0
                            or not isinstance(payload, dict) or payload.get("run_id") != result.run_id
                            or payload.get("mode") != result.mode or payload.get("stream_kind") not in _STREAM_KINDS):
                        raise ValueError("invalid recovery shard header")
                    owner = payload.get("owner")
                    if result.mode == "match" and payload["stream_kind"] in {"poll", "replay"}:
                        raise ValueError("unsupported participant artifact stream")
                    if not isinstance(owner, list) or any(not isinstance(x, str) for x in owner):
                        raise ValueError("invalid recovery shard owner")
                    key = (payload["stream_kind"], tuple(owner))
                    if key in result._shards:
                        raise ValueError("duplicate recovery shard ownership")
                    shard = Shard(file, run_id=result.run_id, mode=result.mode,
                                  kind=key[0], owner=key[1], recorded_at=header["recorded_at"], create=False)
                    # Retain the recorded identity, never silently accept a replaced file.
                    if payload.get("file_identity") != file.file_identity():
                        raise ValueError("recovery shard file identity changed")
                    result._shards[key] = shard
                except BaseException:
                    file.close()
                    raise
            if ("control", ()) not in result._shards:
                raise ValueError("working directory has no control stream")
            # Restore the destination from the durable init event, independently of mode.
            control = result._shards[("control", ())]
            with closing(control.locations(recovery=True)) as locations:
                first = next(locations, None)
                body = first.read() if first is not None else {}
            payload = body.get("payload")
            if body.get("kind") != "init" or body.get("order") != 1 or not isinstance(payload, dict):
                raise ValueError("artifact store has no valid initialization event")
            result.record_subdir = validate_record_subdir(
                result.mode, payload.get("record_subdir", default_record_subdir(result.mode)))
            result.final_parent = result.output_root / result.record_subdir
            result.default_output = result.final_parent / result.run_id
        except BaseException:
            result.close()
            raise
        return result
