"""Plain JSON fault evidence and bounded recovery summaries."""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

RECOVERY_EVENTS = frozenset({"failure", "restart_scheduled", "restarted", "recovered", "terminal"})
RECOVERY_FIELDS = frozenset({
    "game_id", "generation", "event", "restart_count", "user_id", "player_name", "stage_no",
    "retry_delay_seconds", "next_retry_at", "reason", "error", "context",
    "at_utc", "monotonic",
})
RECOVERY_METADATA = frozenset({"schema_version", "recorded_at", "run_id"})


def plain_json(value: Any, depth: int = 0) -> Any:
    """Detach evidence without calling exception renderers or object encoders."""
    if depth > 32:
        raise ValueError("recovery evidence is nested too deeply")
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("recovery evidence keys must be strings")
        return {key: plain_json(item, depth + 1) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [plain_json(item, depth + 1) for item in value]
    raise ValueError("recovery evidence must contain plain JSON values")


def positive_generation(value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("generation must be a positive integer")


def validate_recovery_event(event: Mapping[str, Any]) -> None:
    if not isinstance(event, Mapping) or set(event) - RECOVERY_FIELDS - RECOVERY_METADATA:
        raise ValueError("recovery event has unsupported fields")
    if type(event.get("schema_version")) is not int or event["schema_version"] != 1:
        raise ValueError("recovery event must use schema 1")
    positive_generation(event.get("generation"))
    if type(event.get("event")) is not str or event["event"] not in RECOVERY_EVENTS:
        raise ValueError("invalid recovery event kind")
    if type(event.get("restart_count")) is not int or event["restart_count"] < 0:
        raise ValueError("restart_count must be a non-negative integer")
    for field in ("game_id", "run_id", "recorded_at"):
        if type(event.get(field)) is not str or not event[field]:
            raise ValueError("recovery identity or timestamp is incomplete")
    for field in ("user_id", "player_name", "reason"):
        if field in event and event[field] is not None and (type(event[field]) is not str or not event[field]):
            raise ValueError("recovery identity and reason must be non-empty strings")
    if event.get("stage_no") is not None and (type(event["stage_no"]) is not int or event["stage_no"] < 0):
        raise ValueError("recovery stage_no must be a non-negative integer")
    delay = event.get("retry_delay_seconds")
    if delay is not None and (type(delay) not in {int, float} or delay < 0
                              or type(delay) is float and not math.isfinite(delay)):
        raise ValueError("recovery retry delay must be finite and non-negative")
    stamp = event.get('monotonic')
    if stamp is not None and (type(stamp) not in {int, float} or not math.isfinite(stamp)):
        raise ValueError('recovery monotonic timestamp must be finite')
    for field in ("recorded_at", "next_retry_at", "at_utc"):
        value = event.get(field)
        if value is not None:
            try:
                if type(value) is not str:
                    raise ValueError
                parsed = datetime.fromisoformat(value)
                if field == 'at_utc' and parsed.tzinfo is None:
                    raise ValueError
            except ValueError:
                raise ValueError("recovery timestamps must use ISO 8601") from None
    if event.get("context") is not None and not isinstance(event["context"], Mapping):
        raise ValueError("recovery context must be a mapping")
    plain_json(event)


def recovery_log_path(manifest: Mapping[str, Any]) -> str | None:
    """Only the declared fixed-name, nonempty schema 1 log is publishable."""
    checkpoint = manifest.get("recovery_log")
    if checkpoint is None:
        return None
    if (not isinstance(checkpoint, Mapping) or set(checkpoint) != {"schema_version", "path", "entries"}
            or type(checkpoint.get("schema_version")) is not int or checkpoint["schema_version"] != 1
            or checkpoint.get("path") != "recovery.jsonl"
            or type(checkpoint.get("entries")) is not int or checkpoint["entries"] < 1):
        raise ValueError("invalid recovery log checkpoint")
    return checkpoint["path"]


def verify_recovery_log(lines: Iterable[bytes], manifest: Mapping[str, Any]) -> None:
    if recovery_log_path(manifest) is None:
        raise ValueError("recovery log is not declared by the manifest")
    count = 0
    for line in lines:
        if not line.endswith(b"\n"):
            raise ValueError("recovery log contains an incomplete event")
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            raise ValueError("recovery log contains invalid JSON") from None
        validate_recovery_event(event)
        if event["run_id"] != manifest["run_id"] or event["game_id"] not in manifest["game_ids"]:
            raise ValueError("recovery log identity does not match its manifest")
        count += 1
    if count != manifest["recovery_log"]["entries"]:
        raise ValueError("recovery log entry count does not match its checkpoint")
