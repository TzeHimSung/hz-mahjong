"""Inputs, immutable decisions and execution records for the fixed One-Ply Expectation policy."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias
from typing import Callable

from heuristic_ai.game.models import Action, CandidateAction, Observation, RuleConfig

try:
    from heuristic_ai.strategy.native._one_ply_native import freeze_json as _native_freeze_json
except ImportError:
    _native_freeze_json = None


JsonScalar: TypeAlias = str | int | float | bool | None


JsonValue: TypeAlias = (
    JsonScalar
    | Mapping[str, "JsonValue"]
    | list["JsonValue"]
    | tuple["JsonValue", ...]
)


_REASON = re.compile(r"^[a-z][a-z0-9_]*$")

# Session-local policy also covers shared allocation and immutable-wire helpers.
# ContextVar isolates both threads and nested calls; a child cannot re-enable C.
NATIVE_ENABLED = ContextVar("strategy_native_enabled", default=True)

POLICY_NAME = "one_ply_expectation"
POLICY_VERSION = 3
MAX_SEARCH_NODES = 20000
CACHE_ENABLED = True


def policy_metadata():
    """Fixed audit identity, never a configurable strategy selector."""
    return {"name": POLICY_NAME, "version": POLICY_VERSION}


def session_backend(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        enabled = getattr(self, "native_enabled", True)
        token = NATIVE_ENABLED.set(NATIVE_ENABLED.get() and enabled)
        try:
            return method(self, *args, **kwargs)
        finally:
            NATIVE_ENABLED.reset(token)
    return call


@dataclass(frozen=True, slots=True)
class GameContext:
    game_id: str
    player_name: str
    user_id: str
    seed: int
    rules: RuleConfig
    decision_budget_ms: int = 2000


@dataclass(frozen=True, slots=True)
class DecisionContext:
    game: GameContext
    observation: Observation
    candidates: tuple[CandidateAction, ...]
    decision_key: str
    budget_ms: int
    # Absolute time.monotonic() deadline for this action window. Repeated snapshots
    # and 409 retries must not create a new window budget. Offline callers may
    # omit it and use the existing relative budget.
    deadline_monotonic: float | None = None
    publish_candidate: Callable[["CandidatePublication"], None] | None = field(default=None, compare=False, repr=False)
    cancelled: Callable[[], bool] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        deadline = self.deadline_monotonic
        if deadline is None:
            return
        try:
            valid = not isinstance(deadline, bool) and isinstance(deadline, (int, float)) and math.isfinite(deadline)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("decision deadline must be a finite monotonic timestamp or None")


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: str
    details: Mapping[str, JsonValue] = field(default_factory=dict)
    native_enabled: bool = field(default_factory=NATIVE_ENABLED.get, compare=False, repr=False)

    def __post_init__(self) -> None:
        if _REASON.fullmatch(self.reason) is None:
            raise ValueError("decision reason must be a stable snake_case identifier")
        if not isinstance(self.details, Mapping):
            raise ValueError("decision details must be a mapping")
        if self.native_enabled and NATIVE_ENABLED.get() and _native_freeze_json is not None:
            frozen = _native_freeze_json(self.details)
            if frozen is not NotImplemented:
                object.__setattr__(self, "details", frozen)
                return
        normalized: dict[str, JsonValue] = {}
        for key, value in self.details.items():
            normalized_key = _normalize_json_key(key)
            normalized[normalized_key] = _freeze_json_value(value)
        object.__setattr__(self, "details", MappingProxyType(normalized))

    def with_details(self, updates: Mapping[str, JsonValue], *, reason: str | None = None) -> "Decision":
        """Freeze new fields and share this decision's already-owned immutable tree.

        Arbitrary mapping proxies are not trusted: only fields previously
        validated and copied by this Decision constructor are reused.
        """
        updated = Decision(self.action, self.reason if reason is None else reason, updates, self.native_enabled)
        merged = dict(self.details)
        merged.update(updated.details)
        object.__setattr__(updated, "details", MappingProxyType(merged))
        return updated


def _normalize_json_scalar(value: JsonScalar) -> JsonScalar:
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
        serialized.encode("utf-8")
        return json.loads(serialized)
    except (OverflowError, TypeError, UnicodeError, ValueError):
        raise ValueError("decision details must be JSON-serializable") from None


def _normalize_json_key(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("decision detail keys must be non-empty strings")
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ValueError("decision details must be JSON-serializable") from None
        normalized = value
    else:
        normalized = _normalize_json_scalar(value)
    if type(normalized) is not str or not normalized:
        raise ValueError("decision detail keys must be non-empty strings")
    return normalized


def _freeze_json_value(value: object) -> JsonValue:
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ValueError("decision details must be JSON-serializable") from None
        return value
    if type(value) is int:
        try:
            str(value)
        except (OverflowError, ValueError):
            raise ValueError("decision details must be JSON-serializable") from None
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("decision detail floats must be finite")
        return value
    if isinstance(value, (str, int, bool)):
        return _normalize_json_scalar(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("decision detail floats must be finite")
        return _normalize_json_scalar(value)
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized_key = _normalize_json_key(key)
            normalized[normalized_key] = _freeze_json_value(item)
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    raise ValueError("decision details must contain JSON values")


@dataclass(frozen=True, slots=True)
class CandidatePublication:
    """One complete One-Ply Expectation comparison, with a lazily materialized audit."""
    action: Action
    build_decision: Callable[[], Decision] = field(compare=False, repr=False)
    native_enabled: bool = field(default_factory=NATIVE_ENABLED.get, compare=False)


@dataclass(frozen=True, slots=True)
class DecisionResult:
    decision: Decision
    execution_source: str
    elapsed_ms: float
    fallback_used: bool
    fallback_reason: str | None = None
