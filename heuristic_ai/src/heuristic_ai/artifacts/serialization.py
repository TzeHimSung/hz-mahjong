"""Shared JSON serialization and credential redaction for match records."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from functools import lru_cache
from typing import Any

from heuristic_ai.game.models import Action, CandidateAction
from heuristic_ai.strategy.models import policy_metadata
from heuristic_ai.artifacts.models import DecisionTrace
from heuristic_ai.artifacts.recovery import positive_generation


_SUBMISSION_BODY_LIMIT = 16 * 1024


_SAFE_API_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()




def _action(value: Action) -> dict[str, object]:
    return {
        "kind": value.kind.value,
        "tile": value.tile,
        "tiles": list(value.tiles),
    }


def _candidate(value: CandidateAction) -> dict[str, object]:
    return {"action": _action(value.action), "copies": value.copies}


def _json_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_tree(item) for item in value]
    return value


_SENSITIVE_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def _is_sensitive_key(value: str) -> bool:
    normalized = str(value).casefold()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


class _ArtifactSerializer:
    """Shared secure JSON rendering for train and race audit artifacts."""

    def __init__(
        self,
        tokens: tuple[str, ...],
        now: Callable[[], datetime],
    ) -> None:
        self._tokens = tuple(str(token) for token in tokens if token)
        self._now = now
        # Search nodes repeat a small vocabulary of keys thousands of times.
        # Bound the cache and keep token-dependent projections per recorder.
        self._key_projection = lru_cache(maxsize=2048)(self._project_key)

    def _project_key(self, key: str) -> tuple[str, bool]:
        return self.sanitize(key), _is_sensitive_key(key)

    def sanitize(self, value: str) -> str:
        result = str(value)
        for token in self._tokens:
            result = result.replace(token, "<redacted>")
        return result

    def redact_tree(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.sanitize(value)
        if isinstance(value, Mapping):
            safe: dict[Any, Any] = {}
            for key, item in value.items():
                if isinstance(key, str):
                    plain_key = str(key)
                    safe_key, sensitive = (self._key_projection(plain_key) if len(plain_key) <= 256
                                           else self._project_key(plain_key))
                else:
                    safe_key, sensitive = key, False
                safe[safe_key] = (
                    "<redacted>"
                    if sensitive
                    else self.redact_tree(item)
                )
            return safe
        if isinstance(value, (list, tuple)):
            return [self.redact_tree(item) for item in value]
        return value

    def trace_json(self, trace: DecisionTrace) -> dict[str, Any]:
        result = self.redact_tree(
            {
                "schema_version": 4,
                "recorded_at": self._now().isoformat(),
                "game_id": trace.game_id,
                "round_no": trace.round_no,
                "server_seq": trace.server_seq,
                "player_name": trace.player_name,
                "user_id": trace.user_id,
                "seat": trace.seat,
                "decision_key": trace.decision_key,
                "context_hash": trace.context_hash,
                "candidates": [_candidate(item) for item in trace.candidates],
                "strategy": policy_metadata(),
                "execution_source": trace.execution_source,
                # redact_tree already converts immutable mappings and tuples;
                # walking a separate intermediate JSON tree doubles the work.
                "decision": {"action": _action(trace.decision.action),
                             "reason": trace.decision.reason,
                             "details": trace.decision.details},
                "elapsed_ms": trace.elapsed_ms,
                "draw_to_action_ms": trace.draw_to_action_ms,
                "fallback_used": trace.fallback_used,
                "fallback_reason": trace.fallback_reason,
                "submission": trace.submission.value,
                "error": trace.error,
            }
        )
        phase = getattr(trace, "observation_phase", None)
        generation = getattr(trace, "generation", None)
        if generation is not None:
            positive_generation(generation)
            result["generation"] = generation
        if phase is not None:
            result["observation_phase"] = phase.value
        for field in ("action_request_started_at", "action_request_finished_at"):
            value = getattr(trace, field, None)
            if value is not None:
                result[field] = self.sanitize(value)
        submission_error = getattr(trace, "submission_error", None)
        if submission_error is not None:
            safe = self.redact_tree(submission_error)
            body = safe.get("body")
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            truncated = len(encoded) > _SUBMISSION_BODY_LIMIT
            if truncated:
                body = encoded[:_SUBMISSION_BODY_LIMIT].decode("utf-8", errors="ignore")
            code = safe.get("code")
            result["submission_error"] = {
                "http_status": safe.get("http_status"),
                "code": code if isinstance(code, str) and _SAFE_API_CODE.fullmatch(code) else None,
                "body": body,
                "body_truncated": truncated,
            }
        return result
