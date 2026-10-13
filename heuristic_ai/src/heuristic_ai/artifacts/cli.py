"""Offline recovery commands; no configuration, credentials, or HTTP clients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from heuristic_ai.artifacts.cleanup import cleanup_after_export, cleanup_work_dir
from heuristic_ai.artifacts.export import merge_store
from heuristic_ai.artifacts.store import ArtifactStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge", help="verify retained shards and publish a fresh export")
    merge.add_argument("--work-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path)
    cleanup = commands.add_parser("cleanup", help="retry eligible cleanup using verified evidence")
    cleanup.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "cleanup":
            cleaned = cleanup_work_dir(args.work_dir)
            report = {"cleanup_status": cleaned.status, "remaining_temp_paths": cleaned.remaining,
                      "cleanup_error": cleaned.error}
            code = 0 if cleaned.status == "cleaned" else 1
        else:
            store = ArtifactStore.open_recovery(args.work_dir)
            try:
                exported = merge_store(store, output_dir=args.output_dir, recovery=True)
                cleaned = cleanup_after_export(store, exported)
                report = {"artifact_status": "ready", "artifact_dir": exported.artifact_dir,
                          "status": exported.manifest["status"], "warnings": exported.warnings,
                          "cleanup_status": cleaned.status, "remaining_temp_paths": cleaned.remaining,
                          "cleanup_error": cleaned.error}
                code = {"completed": 0, "degraded": 2, "failed": 1, "interrupted": 130}[exported.manifest["status"]]
            finally:
                store.close()
    except (OSError, ValueError, RuntimeError) as error:
        report = {"error": str(error), "work_dir": args.work_dir}
        code = 3 if args.command == "merge" else 1
    print(json.dumps(report, ensure_ascii=False, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
