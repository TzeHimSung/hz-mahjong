from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal



ACTION_RESERVE_MS = 300.0


@dataclass(frozen=True, slots=True)
class ActionReserves:
    response_ms: float = ACTION_RESERVE_MS
    discard_ms: float = ACTION_RESERVE_MS

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for v in (self.response_ms, self.discard_ms)):
            raise ValueError("action reserves must be finite nonnegative milliseconds")


DEFAULT_ACTION_RESERVES = ActionReserves()
DEFAULT_MATCH_ACTION_RESERVES = ActionReserves(300.0, 300.0)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    url: str
    insecure: bool
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class CompetitionConfig:
    room_id: str
    seed: int
    poll_interval_seconds: float
    decision_budget_ms: int
    output_dir: Path


@dataclass(frozen=True, slots=True)
class PlayerConfig:
    name: str
    token: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    server: ServerConfig
    competition: CompetitionConfig
    players: tuple[PlayerConfig, ...]
    save_to_official: bool = False

    @property
    def record_output_dir(self) -> Path:
        return self.competition.output_dir / ("official_records" if self.save_to_official else "test_records")


@dataclass(frozen=True, slots=True)
class RaceParticipantConfig:
    token: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class RaceSupervisionConfig:
    enabled: bool = True
    heartbeat_timeout_seconds: float = 30.0
    stall_timeout_seconds: float = 180.0


@dataclass(frozen=True, slots=True)
class RaceConfig:
    server: ServerConfig
    competition: CompetitionConfig
    participant: RaceParticipantConfig
    supervision: RaceSupervisionConfig = field(default_factory=RaceSupervisionConfig)

    @property
    def record_output_dir(self) -> Path:
        return self.competition.output_dir / "official_records"


@dataclass(frozen=True, slots=True)
class MatchmakingConfig:
    max_parallel_games: int = 10
    max_rounds_per_game: int = 8


@dataclass(frozen=True, slots=True)
class MatchCompetitionConfig:
    seed: int
    poll_interval_seconds: float
    decision_budget_ms: int
    output_dir: Path


@dataclass(frozen=True, slots=True)
class MatchConfig:
    server: ServerConfig
    competition: MatchCompetitionConfig
    matchmaking: MatchmakingConfig = field(default_factory=MatchmakingConfig)
    continuous: bool = False

    @property
    def record_output_dir(self) -> Path:
        return self.competition.output_dir / "test_records"


@dataclass(frozen=True, slots=True)
class AppConfig:
    schema_version: int
    mode: Literal["train", "race", "match"]
    train: TrainConfig
    race: RaceConfig
    global_token: str | None = field(default=None, repr=False)
    match: MatchConfig | None = None
