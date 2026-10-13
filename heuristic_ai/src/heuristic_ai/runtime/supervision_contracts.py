"""Pure-value contracts shared by race supervision, health and presentation.

Secret-bearing configuration snapshots travel only over private IPC.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from heuristic_ai.artifacts.models import RunResult, RunStatus
from heuristic_ai.config.models import (
    CompetitionConfig, RaceConfig, RaceParticipantConfig, RaceSupervisionConfig, ServerConfig,
)


PROTOCOL_VERSION = 2
HEARTBEAT_INTERVAL_SECONDS = 5.0
STARTUP_GRACE_SECONDS = 60.0
STOP_GRACE_SECONDS = 10.0
OPERATION_GRACE_SECONDS = 10.0
HEALTHY_RESET_SECONDS = 300.0
RESTART_DELAYS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0)
TERMINAL_REASONS = frozenset({"finished", "eliminated", "closed", "void"})
FATAL_FAILURE_CATEGORIES = frozenset({"configuration", "authentication", "identity"})


def plain_value(value: Any) -> Any:
    """Copy a JSON-shaped value without invoking arbitrary string conversion."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("supervision object keys must be strings")
        return {key: plain_value(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [plain_value(item) for item in value]
    raise ValueError("supervision values must be finite JSON data")


@dataclass(frozen=True, slots=True)
class RaceRuntimeStatus:
    run_id: str | None = None
    phase: str = "starting"
    tournament_status: str | None = None
    stage_no: int | None = None
    stage_name: str | None = None
    stage_status: str | None = None
    qualified: bool | None = None
    qualify_role: str | None = None
    rank: int | None = None
    total_score: int | None = None
    stage_settled_rounds: int | None = None
    active_games: int | None = None
    decisions: dict[str, int] = field(default_factory=dict)
    audit_synced_bytes: int | None = None
    record_p99_ms: float | None = None
    waiting_reason: str | None = None
    ready: bool | None = None
    ranking_scope: str = "current"
    stage_stats: dict[str, Any] | None = None
    last_server_poll_at: float | None = None
    server_poll_in_flight: bool = False
    server_poll_error: str | None = None
    revision: int = 0

    def to_payload(self) -> dict[str, Any]:
        return plain_value(asdict(self))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RaceRuntimeStatus:
        value = plain_value(payload)
        allowed = {item.name for item in fields(cls)}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("invalid race progress fields")
        if type(value.get("revision", 0)) is not int or value.get("revision", 0) < 0:
            raise ValueError("invalid race progress revision")
        if value.get("ranking_scope", "current") not in {"current", "previous", "void"}:
            raise ValueError("invalid ranking scope")
        for key in ("waiting_reason", "server_poll_error"):
            if value.get(key) is not None and type(value[key]) is not str:
                raise ValueError("invalid race status label")
        if value.get("ready") is not None and type(value["ready"]) is not bool:
            raise ValueError("invalid ready status")
        if type(value.get("server_poll_in_flight", False)) is not bool:
            raise ValueError("invalid server query status")
        if value.get("last_server_poll_at") is not None and (type(value["last_server_poll_at"]) not in (int, float)
                                                             or value["last_server_poll_at"] < 0):
            raise ValueError("invalid server query timestamp")
        stats = value.get("stage_stats")
        if stats is not None:
            if not isinstance(stats, dict):
                raise ValueError("invalid stage statistics")
            for key in ("total_rounds", "won_rounds", "unknown_rounds"):
                if type(stats.get(key, 0)) is not int or stats.get(key, 0) < 0:
                    raise ValueError("invalid stage round count")
            total, won, unknown = (stats.get(key, 0) for key in ("total_rounds", "won_rounds", "unknown_rounds"))
            complete, rate = stats.get("statistics_complete", False), stats.get("win_rate")
            if won + unknown > total or type(complete) is not bool:
                raise ValueError("inconsistent stage statistics")
            expected_rate = won / total if total and complete and not unknown else None
            if rate is not None and (type(rate) not in (int, float) or expected_rate is None
                                     or not math.isclose(rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("inconsistent stage win rate")
            if stats.get("incomplete_reason") is not None and type(stats["incomplete_reason"]) is not str:
                raise ValueError("invalid statistics completeness reason")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ChildEvent:
    generation: int
    sequence: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION

    def to_payload(self) -> dict[str, Any]:
        return plain_value(asdict(self))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ChildEvent:
        value = plain_value(payload)
        if not isinstance(value, dict) or set(value) != {"generation", "sequence", "kind", "payload", "version"}:
            raise ValueError("invalid child event")
        if (type(value["generation"]) is not int or value["generation"] < 1
                or type(value["sequence"]) is not int or value["sequence"] < 1
                or type(value["version"]) is not int or value["version"] != PROTOCOL_VERSION
                or type(value["kind"]) is not str or not value["kind"]
                or type(value["payload"]) is not dict):
            raise ValueError("invalid child event")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ControlCommand:
    generation: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return plain_value(asdict(self))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ControlCommand:
        value = plain_value(payload)
        if (not isinstance(value, dict) or set(value) != {"generation", "kind", "payload"}
                or type(value["generation"]) is not int or value["generation"] < 1
                or type(value["kind"]) is not str
                or value["kind"] not in {"proceed", "stop", "dump_stacks", "ack_terminal"}
                or type(value["payload"]) is not dict):
            raise ValueError("invalid supervisor command")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class RaceSupervisionResult:
    supervision_id: str
    last_result: RunResult | None
    exit_code: int
    stop_reason: str
    restart_count: int
    log_dir: Path


class StopSignal(Protocol):
    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def wait(self, timeout: float | None = None) -> bool: ...


class RaceRuntimeObserver(Protocol):
    """Synchronous hooks; implementations must not perform sink I/O.

Identity and terminal hooks may wait for bounded parent acknowledgements.
Health snapshots use real monotonic time and contain only plain dictionaries.
"""
    def run_started(self, run_id: str, work_dir: Path | None) -> None: ...
    def identity_validated(self, user_id: str, tournament_id: str) -> None: ...
    def coordinator_active(self) -> None: ...
    def health(self, snapshot: dict[str, Any]) -> None: ...
    def race_state(self, status: RaceRuntimeStatus) -> None: ...
    def business_terminal(self, reason: str, own_ranking: Mapping[str, int] | None) -> None: ...
    def run_failed(self, category: str, error_type: str, failure_code: str) -> None: ...


def race_config_to_payload(config: RaceConfig) -> dict[str, Any]:
    return {
        "server": asdict(config.server),
        "competition": {
            "room_id": config.competition.room_id,
            "seed": config.competition.seed,
            "poll_interval_seconds": config.competition.poll_interval_seconds,
            "decision_budget_ms": config.competition.decision_budget_ms,
            "output_dir": str(config.competition.output_dir.resolve()),
        },
        "participant": {
            "token": config.participant.token,
        },
        "supervision": asdict(config.supervision),
    }


def race_config_from_payload(payload: Mapping[str, Any]) -> RaceConfig:
    value = plain_value(payload)
    competition = dict(value["competition"])
    competition["output_dir"] = Path(competition["output_dir"])
    if not competition["output_dir"].is_absolute():
        raise ValueError("worker output directory must be absolute")
    participant = value["participant"]
    server = dict(value["server"])
    return RaceConfig(
        server=ServerConfig(**server),
        competition=CompetitionConfig(**competition),
        participant=RaceParticipantConfig(participant["token"]),
        supervision=RaceSupervisionConfig(**value["supervision"]),
    )


def run_result_to_payload(result: RunResult) -> dict[str, Any]:
    return {
        "status": result.status.value, "run_id": result.run_id,
        "artifact_dir": str(result.artifact_dir), "game_ids": list(result.game_ids),
        "work_dir": str(result.work_dir) if result.work_dir is not None else None,
        "artifact_status": result.artifact_status.value, "artifact_error": result.artifact_error,
        "cleanup_status": result.cleanup_status.value, "cleanup_error": result.cleanup_error,
        "remaining_temp_paths": [str(path) for path in result.remaining_temp_paths], "mode": result.mode,
    }


def run_result_from_payload(payload: Mapping[str, Any]) -> RunResult:
    value = plain_value(payload)
    value["status"] = RunStatus(value["status"])
    value["artifact_dir"] = Path(value["artifact_dir"])
    value["game_ids"] = tuple(value["game_ids"])
    value["work_dir"] = Path(value["work_dir"]) if value["work_dir"] is not None else None
    value["remaining_temp_paths"] = tuple(Path(path) for path in value["remaining_temp_paths"])
    return RunResult(**value)
