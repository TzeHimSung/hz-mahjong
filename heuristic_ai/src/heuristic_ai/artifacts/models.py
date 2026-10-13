"""Decision audit traces and run publication results."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from heuristic_ai.game.models import CandidateAction, Phase
from heuristic_ai.strategy.models import Decision, JsonValue


class SubmissionStatus(StrEnum):
    ACCEPTED = "accepted"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    game_id: str
    round_no: int
    server_seq: int
    player_name: str
    user_id: str
    seat: int
    decision_key: str
    context_hash: str
    candidates: tuple[CandidateAction, ...]
    execution_source: str
    decision: Decision
    elapsed_ms: float
    fallback_used: bool
    fallback_reason: str | None
    submission: SubmissionStatus
    draw_to_action_ms: float | None = None
    error: str | None = None
    observation_phase: Phase | None = None
    action_request_started_at: str | None = None
    action_request_finished_at: str | None = None
    submission_error: Mapping[str, JsonValue] | None = None
    generation: int | None = None


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ArtifactStatus(StrEnum):
    READY = "ready"
    INCOMPLETE = "incomplete"


class CleanupStatus(StrEnum):
    RETAINED = "retained"
    CLEANED = "cleaned"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class MatchWinStats:
    """Completed small rounds, including unresolved outcomes in the denominator."""

    won_rounds: int = 0
    total_rounds: int = 0
    unknown_rounds: int = 0

    def __post_init__(self) -> None:
        for field in ("won_rounds", "total_rounds", "unknown_rounds"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.won_rounds + self.unknown_rounds > self.total_rounds:
            raise ValueError("won and unknown rounds cannot exceed total rounds")

    @property
    def statistics_complete(self) -> bool:
        return self.unknown_rounds == 0

    @property
    def win_rate(self) -> float | None:
        if not self.total_rounds or not self.statistics_complete:
            return None
        return self.won_rounds / self.total_rounds

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {"won_rounds": self.won_rounds, "total_rounds": self.total_rounds,
                "unknown_rounds": self.unknown_rounds, "win_rate": self.win_rate,
                "statistics_complete": self.statistics_complete}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MatchWinStats:
        if not isinstance(value, Mapping):
            raise ValueError("match statistics must be a mapping")
        try:
            return cls(won_rounds=value["won_rounds"], total_rounds=value["total_rounds"],
                       unknown_rounds=value["unknown_rounds"])
        except KeyError:
            raise ValueError("match statistics are missing round counts") from None

    def __add__(self, other: object) -> MatchWinStats:
        if not isinstance(other, MatchWinStats):
            return NotImplemented
        return MatchWinStats(self.won_rounds + other.won_rounds,
                             self.total_rounds + other.total_rounds,
                             self.unknown_rounds + other.unknown_rounds)


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    run_id: str
    artifact_dir: Path
    game_id: str | None = None
    game_ids: tuple[str, ...] = ()
    work_dir: Path | None = None
    artifact_status: ArtifactStatus = ArtifactStatus.READY
    artifact_error: str | None = None
    cleanup_status: CleanupStatus = CleanupStatus.RETAINED
    remaining_temp_paths: tuple[Path, ...] = ()
    cleanup_error: str | None = None
    mode: str | None = None

    def __post_init__(self) -> None:
        identifiers = list(self.game_ids)
        if self.game_id is not None:
            identifiers.append(self.game_id)
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("game ids must be non-empty strings")
        normalized = tuple(sorted(set(identifiers)))
        object.__setattr__(self, "game_ids", normalized)
        object.__setattr__(self, "game_id", normalized[0] if normalized else None)
        object.__setattr__(self, "artifact_status", ArtifactStatus(self.artifact_status))
        object.__setattr__(self, "cleanup_status", CleanupStatus(self.cleanup_status))
        object.__setattr__(self, "remaining_temp_paths", tuple(Path(path) for path in self.remaining_temp_paths))

    @property
    def exit_code(self) -> int:
        if (self.mode == "match" or self.status is RunStatus.INTERRUPTED) and self.artifact_status is ArtifactStatus.INCOMPLETE:
            return 3
        if self.status is RunStatus.INTERRUPTED:
            return 130
        if self.status not in {RunStatus.COMPLETED, RunStatus.DEGRADED}:
            return 1
        if self.artifact_status is ArtifactStatus.INCOMPLETE:
            return 3
        if self.status is RunStatus.COMPLETED:
            return 0
        if self.status is RunStatus.DEGRADED:
            return 2
        return 1
