"""Conditional cleanup: published output is verified before any deletion."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from heuristic_ai.artifacts.export import ExportResult, relative_artifact_path
from heuristic_ai.artifacts.files import OwnedArtifactFile, checked_path
from heuristic_ai.artifacts.projection import TERMINAL_OPERATIONS
from heuristic_ai.artifacts.recovery import recovery_log_path, verify_recovery_log
from heuristic_ai.artifacts.store import ARTIFACT_MODES, PARTICIPANT_MODES, ArtifactStore, decode_frame, default_record_subdir, json_bytes, validate_record_subdir


@dataclass(frozen=True, slots=True)
class CleanupResult:
    status: str
    remaining: tuple[Path, ...] = ()
    error: str | None = None


def _reason(error: BaseException) -> str:
    code = getattr(error, "winerror", None) if isinstance(error, OSError) else None
    return f"{type(error).__name__}" + (f" (winerror={code})" if code is not None else "")


def _managed_work_dir(work_dir: Path) -> Path:
    work_dir = Path(os.path.abspath(work_dir))
    if work_dir.parent.name not in ARTIFACT_MODES or work_dir.parent.parent.name != ".record_work":
        raise ValueError("cleanup requires a managed run working directory")
    checked_path(work_dir, work_dir.parent)
    return work_dir


def _receipt_paths(work_dir: Path) -> list[Path]:
    return sorted(work_dir.parent.glob(f"{work_dir.name}.cleanup-*.json"))


def _read_json(file: OwnedArtifactFile, *, maximum: int = 16 * 1024 * 1024) -> Any:
    if file.size() > maximum:
        raise ValueError("artifact metadata exceeds its size limit")
    try:
        with file.reader() as reader:
            return json.load(reader)
    except (ValueError, UnicodeError):
        raise ValueError("invalid artifact metadata JSON") from None


def _verify_export(work_dir: Path, proof: dict[str, Any]) -> list[OwnedArtifactFile]:
    path = Path(proof["artifact_dir"])
    if not path.is_absolute() or path == work_dir or path.is_relative_to(work_dir) or work_dir.is_relative_to(path):
        raise ValueError("cleanup proof overlaps its working directory")
    checked_path(path, path, allow_root=True)
    files = proof.get("files")
    if not isinstance(files, dict) or not {"manifest.json", "decisions.jsonl"}.issubset(files):
        raise ValueError("cleanup proof is missing final files")
    held: list[OwnedArtifactFile] = []
    manifest = None
    try:
        for relative, expected in files.items():
            relative_artifact_path(relative)
            file = OwnedArtifactFile(path / relative, existing=True, ownership_root=path, shared_read=True)
            held.append(file)
            if (file.file_identity() != expected.get("file_identity") or file.size() != expected.get("size")
                    or file.digest() != expected.get("sha256")):
                raise ValueError("final artifact no longer matches the cleanup proof")
            if relative == "manifest.json":
                manifest = _read_json(file)
        if not isinstance(manifest, dict) or manifest.get("run_id") != work_dir.name or manifest.get("status") not in {"completed", "degraded"}:
            raise ValueError("run is not eligible for automatic cleanup")
        recovery_path = recovery_log_path(manifest)
        if ("recovery.jsonl" in files) != (recovery_path is not None):
            raise ValueError("recovery log does not match its manifest declaration")
        if recovery_path is not None:
            recovery_file = next(file for file in held if file.path == path / recovery_path)
            with recovery_file.reader() as reader:
                verify_recovery_log(reader, manifest)
        if work_dir.parent.name in PARTICIPANT_MODES:
            required = {"manifest.json", "decisions.jsonl"}
            if recovery_path is not None:
                required.add(recovery_path)
            if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != f"{work_dir.parent.name}_run" or set(files) != required:
                raise ValueError(f"{work_dir.parent.name} export is not complete")
        else:
            required = {"state_polls.jsonl"}
            for relative in manifest.get("rounds", []):
                relative_artifact_path(relative)
                required.update({relative + "/record.json", relative + "/events.jsonl"})
            if manifest.get("schema_version") != 2 or not required.issubset(files):
                raise ValueError("train export is not complete")
        return held
    except BaseException:
        for file in held:
            file.close()
        raise


def _descriptor(file: OwnedArtifactFile, scope: str) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(file.path), "directory": file.directory,
                              "file_identity": file.file_identity(), "scope": scope}
    if not file.directory:
        result.update(size=file.size(), sha256=file.digest())
    return result


def _attempts(store: ArtifactStore) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[Path]]:
    targets: list[dict[str, Any]] = []
    roots: dict[str, dict[str, Any]] = {}
    unresolved: list[Path] = []
    for file in store.auxiliary:
        if not file.path.name.startswith("attempt-"):
            continue
        rows = []
        try:
            with file.reader() as reader:
                for line in reader:
                    if not line.endswith(b"\n"):
                        break
                    rows.append(decode_frame(line))
            if not rows or rows[0].get("kind") != "export_attempt":
                raise ValueError("invalid export attempt")
            header = rows[0]["payload"]
            if header["run_id"] != store.run_id or header["file_identity"] != file.file_identity():
                raise ValueError("export attempt ownership mismatch")
            target = Path(header["output_dir"])
            if any(row.get("kind") in {"committed", "rejected"} for row in rows):
                continue
            if not target.exists():
                continue
            directory_identity = header.get("directory_identity") or next(
                (row["payload"].get("file_identity") for row in rows if row.get("kind") == "directory_owned"), None)
            if not directory_identity:
                unresolved.append(target)
                continue
            # Any manifest is protected: an interrupted publication or a user
            # modification must never be mistaken for disposable scratch data.
            if (target / "manifest.json").exists():
                unresolved.append(target)
                continue
            if target.parent != store.final_parent or not (target.name == store.run_id or target.name.startswith(store.run_id + "_export_")):
                unresolved.append(target)
                continue
            checked_path(target, store.final_parent)
            roots[str(target)] = {"file_identity": directory_identity}
            targets.append({"path": str(target), "directory": True,
                            "file_identity": directory_identity, "scope": "attempt"})
            for row in rows[1:]:
                if row.get("kind") not in {"file_created", "directory_created"}:
                    continue
                value = row["payload"]
                relative_artifact_path(value["path"])
                targets.append({"path": str(target / value["path"]),
                                "directory": row["kind"] == "directory_created",
                                "file_identity": value["file_identity"], "scope": "attempt"})
        except (KeyError, ValueError, OSError):
            unresolved.append(file.path)
    return targets, roots, unresolved


def _validate_target(item: dict[str, Any], work_dir: Path, attempts: dict[str, Any],
                     record_subdir: str) -> Path:
    path = Path(item["path"])
    if not path.is_absolute() or type(item.get("directory")) is not bool or not isinstance(item.get("file_identity"), str):
        raise ValueError("invalid cleanup target")
    scope = item.get("scope")
    if scope == "work":
        checked_path(path, work_dir, allow_root=item["directory"])
    elif scope == "receipt":
        if path.parent != work_dir.parent or not path.name.startswith(work_dir.name + ".cleanup-") or path.suffix != ".json":
            raise ValueError("cleanup receipt is outside its run scope")
        checked_path(path, work_dir.parent)
    elif scope == "attempt":
        parent = work_dir.parent.parent.parent / record_subdir
        matching = [Path(root) for root in attempts if path == Path(root) or path.is_relative_to(Path(root))]
        if len(matching) != 1:
            raise ValueError("cleanup attempt scope is ambiguous")
        root = matching[0]
        if root.parent != parent or not (root.name == work_dir.name or root.name.startswith(work_dir.name + "_export_")):
            raise ValueError("cleanup attempt is outside its run scope")
        if (root / "manifest.json").exists():
            raise ValueError("cleanup must not touch published exports")
        checked_path(path, root, allow_root=item["directory"])
    else:
        raise ValueError("unknown cleanup target scope")
    return path


def _read_receipt(file: OwnedArtifactFile, work_dir: Path) -> dict[str, Any]:
    data = _read_json(file)
    if (not isinstance(data, dict) or data.get("format_version") != 1
            or data.get("type") != "artifact_cleanup" or data.get("run_id") != work_dir.name
            or data.get("work_dir") != str(work_dir) or data.get("file_identity") != file.file_identity()
            or not isinstance(data.get("targets"), list) or not isinstance(data.get("proof"), dict)
            or not isinstance(data.get("attempt_roots"), dict)):
        raise ValueError("invalid cleanup receipt")
    record_subdir = validate_record_subdir(
        work_dir.parent.name, data.get("record_subdir", default_record_subdir(work_dir.parent.name)))
    data["record_subdir"] = record_subdir
    seen = set()
    for item in data["targets"]:
        path = _validate_target(item, work_dir, data["attempt_roots"], record_subdir)
        if path in seen:
            raise ValueError("cleanup receipt contains duplicate targets")
        seen.add(path)
    return data


def _execute(receipt: OwnedArtifactFile, data: dict[str, Any],
             held: dict[Path, OwnedArtifactFile]) -> CleanupResult:
    work_dir = Path(data["work_dir"])
    remaining: set[Path] = set()
    problems: list[str] = []
    opened: list[OwnedArtifactFile] = []
    expected_paths = {Path(item["path"]) for item in data["targets"]}
    # Unknown files are never swept up by a recursive delete.
    for root in [work_dir, *(Path(value) for value in data["attempt_roots"])]:
        if root.exists():
            stack = [root]
            while stack:
                parent = stack.pop()
                try:
                    entries = list(parent.iterdir())
                except OSError:
                    remaining.add(parent)
                    continue
                for entry in entries:
                    try:
                        checked_path(entry, root)
                    except (OSError, ValueError):
                        remaining.add(entry)
                        continue
                    if entry not in expected_paths:
                        remaining.add(entry)
                    if entry.is_dir():
                        stack.append(entry)
    ordered = sorted(data["targets"], key=lambda item: (item["scope"] == "receipt", item["directory"], -len(Path(item["path"]).parts)))
    for item in ordered:
        path = Path(item["path"])
        if item["scope"] == "receipt" and remaining:
            remaining.add(path)
            continue
        if not path.exists():
            continue
        try:
            _validate_target(item, work_dir, data["attempt_roots"], data["record_subdir"])
            file = held.get(path)
            if file is None:
                root = work_dir if item["scope"] == "work" else path.parent
                file = OwnedArtifactFile(path, temporary=True, existing=True,
                                         directory=item["directory"], ownership_root=root)
                opened.append(file)
            if file.file_identity() != item["file_identity"]:
                raise ValueError("cleanup target identity changed")
            if not file.directory and "sha256" in item:
                if file.size() != item["size"] or file.digest() != item["sha256"]:
                    raise ValueError("cleanup target contents changed")
            if file.state == "OPEN":
                file.seal()
            root = work_dir if item["scope"] == "work" else path.parent
            file.mark_delete(ownership_root=root)
            file.close()
            if path.exists():
                remaining.add(path)
        except (OSError, ValueError, RuntimeError) as error:
            remaining.add(path)
            problems.append(f"{path}: {_reason(error)}")
    for path in data.get("unresolved", []):
        if Path(path).exists():
            remaining.add(Path(path))
    for file in opened:
        try:
            file.close()
        except OSError:
            remaining.add(file.path)
    if work_dir.exists():
        remaining.add(work_dir)
    if remaining:
        remaining.add(receipt.path)
        return CleanupResult("pending", tuple(sorted(remaining)), "; ".join(problems) or "temporary objects remain")
    receipt.mark_delete(ownership_root=work_dir.parent)
    receipt.close()
    if receipt.path.exists():
        return CleanupResult("pending", (receipt.path,), "cleanup receipt deletion is pending")
    return CleanupResult("cleaned")


def cleanup_after_export(store: ArtifactStore, result: ExportResult) -> CleanupResult:
    if not result.cleanup_eligible:
        return CleanupResult("retained", (store.work_dir,))
    work_dir = _managed_work_dir(store.work_dir)
    proof = {"artifact_dir": str(result.artifact_dir), "files": result.files}
    final_handles: list[OwnedArtifactFile] = []
    prior: list[OwnedArtifactFile] = []
    receipt: OwnedArtifactFile | None = None
    try:
        final_handles = _verify_export(work_dir, proof)
        for path in _receipt_paths(work_dir):
            file = OwnedArtifactFile(path, temporary=True, existing=True, ownership_root=work_dir.parent)
            prior.append(file)
            _read_receipt(file, work_dir)
        held_files = [*store.temporary_files(), store.directory]
        targets = [_descriptor(file, "work") for file in held_files]
        targets.extend(_descriptor(file, "receipt") for file in prior)
        attempts, roots, unresolved = _attempts(store)
        targets.extend(attempts)
        receipt = OwnedArtifactFile(work_dir.parent / f"{store.run_id}.cleanup-{uuid.uuid4().hex}.json",
                                    temporary=True, ownership_root=work_dir.parent)
        data = {"format_version": 1, "type": "artifact_cleanup", "run_id": store.run_id,
                "record_subdir": store.record_subdir,
                "work_dir": str(work_dir), "created_at": datetime.now().astimezone().isoformat(),
                "file_identity": receipt.file_identity(), "proof": proof,
                "targets": targets, "attempt_roots": roots, "unresolved": [str(p) for p in unresolved]}
        receipt.append(json_bytes(data, pretty=True) + b"\n")
        receipt.seal()
        # No deletion is attempted until the complete receipt itself is durable.
        _read_receipt(receipt, work_dir)
        held = {file.path: file for file in [*held_files, *prior]}
        return _execute(receipt, data, held)
    except (OSError, ValueError, RuntimeError) as error:
        paths = [work_dir]
        if receipt is not None:
            paths.append(receipt.path)
        return CleanupResult("pending", tuple(paths), _reason(error))
    finally:
        for file in [*final_handles, *prior, *([receipt] if receipt is not None else [])]:
            try:
                file.close()
            except OSError:
                pass


def cleanup_work_dir(work_dir: Path) -> CleanupResult:
    work_dir = _managed_work_dir(work_dir)
    receipts = _receipt_paths(work_dir)
    if not receipts:
        if not work_dir.exists():
            return CleanupResult("cleaned")
        # A receipt failure before deletion leaves the complete source. Recover
        # the positive export certificate without generating another export.
        store = ArtifactStore.open_recovery(work_dir)
        try:
            terminal = None
            for location in store.locations(recovery=True):
                body = location.read()
                if body.get("kind") == "control" and body["payload"].get("operation") in TERMINAL_OPERATIONS:
                    terminal = (location, body["payload"])
            if terminal is None or terminal[1]["operation"] != "complete":
                return CleanupResult("retained", (work_dir,), "run did not complete normally")
            store.verify_inventory(terminal[1].get("inventory"), terminal[0], completed=True)
            candidates = []
            for file in store.auxiliary:
                with file.reader() as reader:
                    rows = [decode_frame(line) for line in reader if line.endswith(b"\n")]
                if rows and rows[-1].get("kind") == "committed":
                    candidates.append((file.path.stat().st_mtime_ns, rows[0]["payload"], rows[-1]["payload"]))
            if not candidates:
                return CleanupResult("retained", (work_dir,), "no verified final export exists")
            _, header, committed = max(candidates, key=lambda item: item[0])
            result = ExportResult(Path(header["output_dir"]), {}, committed["files"], (), True)
            return cleanup_after_export(store, result)
        finally:
            store.close()
    loaded: list[tuple[dict[str, Any], OwnedArtifactFile]] = []
    final_handles: list[OwnedArtifactFile] = []
    try:
        for path in receipts:
            file = OwnedArtifactFile(path, temporary=True, existing=True, ownership_root=work_dir.parent)
            try:
                data = _read_receipt(file, work_dir)
            except BaseException:
                file.close()
                raise
            loaded.append((data, file))
        data, receipt = max(loaded, key=lambda item: item[0]["created_at"])
        final_handles = _verify_export(work_dir, data["proof"])
        held = {file.path: file for _, file in loaded}
        return _execute(receipt, data, held)
    except (OSError, ValueError, RuntimeError) as error:
        return CleanupResult("pending", tuple([work_dir, *receipts]), _reason(error))
    finally:
        for file in [*final_handles, *(file for _, file in loaded)]:
            try:
                file.close()
            except OSError:
                pass
