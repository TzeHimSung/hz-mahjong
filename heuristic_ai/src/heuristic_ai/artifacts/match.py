"""Participant decisions and results for one automatic matching session."""
from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from heuristic_ai.artifacts.models import MatchWinStats, RunResult, RunStatus
from heuristic_ai.artifacts.participant import ParticipantRunRecorder
from heuristic_ai.artifacts.projection import MatchProjection
from heuristic_ai.artifacts.serialization import _ArtifactSerializer, _canonical_hash
from heuristic_ai.protocol.compatibility import guide_compatibility_profile
from heuristic_ai.strategy.models import policy_metadata

if TYPE_CHECKING:
    from heuristic_ai.config.models import MatchConfig


class MatchRunRecorder(ParticipantRunRecorder):
    def __init__(
        self,
        config: MatchConfig,
        *,
        global_token: str,
        package_version: str,
        session_group_id: str,
        session_index: int,
        guide_version: int | None = None,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if not isinstance(session_group_id, str) or not session_group_id:
            raise ValueError("session_group_id must be a non-empty string")
        if type(session_index) is not int or session_index < 1:
            raise ValueError("session_index must be a positive integer")
        self._config = config
        self._now = now
        self._serializer = _ArtifactSerializer((global_token,), now)
        started = now()
        self.run_id = f"{started.strftime('%Y_%m_%d_%H_%M_%S')}_{id_factory()[:8]}"
        metadata = policy_metadata()
        request_config = {
            "server": {"url": config.server.url, "insecure": config.server.insecure,
                       "timeout_seconds": config.server.timeout_seconds},
            "seed": config.competition.seed,
            "poll_interval_seconds": config.competition.poll_interval_seconds,
            "decision_budget_ms": config.competition.decision_budget_ms,
            "matchmaking": {"M": config.matchmaking.max_parallel_games,
                            "Rounds": config.matchmaking.max_rounds_per_game},
            "strategy": metadata,
            "continuous": config.continuous,
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "match_run",
            "mode": "match",
            "run_id": self.run_id,
            "session_group_id": session_group_id,
            "session_index": session_index,
            "status": RunStatus.RUNNING.value,
            "package_version": package_version,
            "guide_version": guide_version,
            "client_compatibility_profile": guide_compatibility_profile("match"),
            "config_hash": _canonical_hash(request_config),
            "request_config": request_config,
            "room_id": None,
            "match_config": None,
            "round_no": None,
            "matched_at": None,
            "seed": config.competition.seed,
            "started_at": started.isoformat(),
            "finished_at": None,
            "participant": {"user_id": None, "strategy": metadata},
            "lifecycle": [],
            "game_ids": [],
            "games": [],
            "statistics_base": MatchWinStats().to_dict(),
            "win_stats": MatchWinStats().to_dict(),
            "session_win_stats": MatchWinStats().to_dict(),
            "decision_log": {"schema_version": 4, "committed_bytes": 0, "committed_entries": 0},
            "execution_counts": {},
            "decision_counts": {"total": 0, "accepted": 0, "conflict": 0, "error": 0, "fallback": 0},
            "degradations": [],
            "summary": None,
            "error": None,
        }
        self._initialize_storage(config.competition.output_dir, "match", manifest)

    def set_statistics_base(self, stats: MatchWinStats) -> None:
        if not isinstance(stats, MatchWinStats):
            raise ValueError("statistics base must be MatchWinStats")
        self._control("set_statistics_base", stats=stats.to_dict())

    def win_stats(self) -> MatchWinStats:
        with self._lock:
            return self._projection.win_stats()

    def session_win_stats(self) -> MatchWinStats:
        with self._lock:
            return self._projection.session_win_stats()

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return {**super().progress(), "win_stats": self.win_stats().to_dict()}

    def record_round_progress(self, game_id: str, completed_rounds: int) -> None:
        self._control("record_round_progress", game_id=game_id, completed_rounds=completed_rounds)

    def record_round_outcome(self, game_id: str, round_no: int, *, outcome: str, source: str,
                             conflicted: bool = False) -> None:
        self._control("record_round_outcome", game_id=game_id, round_no=round_no,
                      outcome=outcome, source=source, conflicted=conflicted)

    def bind_match(self, room_id: str, *, match_config: dict[str, Any], round_no: int) -> None:
        # Project before the control event crosses the persistence boundary.
        self._control("bind_match", room_id=room_id,
                      match_config=MatchProjection.public_match_config(match_config), round_no=round_no)

    def complete(self, terminal_reason: str = "finished",
                 participant_ranking: Mapping[str, object] | None = None) -> RunResult:
        self._require_unfinished()
        return self._finish("complete", terminal_reason=terminal_reason, participant_ranking=participant_ranking)

    def fail(self, error: BaseException, *, terminal_reason: str = "runtime_error",
             participant_ranking: Mapping[str, object] | None = None) -> RunResult:
        self._require_unfinished()
        return self._finish("fail", **self._failure_arguments(error), terminal_reason=terminal_reason,
                            participant_ranking=participant_ranking)

    def interrupt(self, terminal_reason: str = "user_interrupt") -> RunResult:
        self._require_unfinished()
        return self._finish("interrupt", terminal_reason=terminal_reason)

    def _require_unfinished(self) -> None:
        if self._final_result is not None:
            raise RuntimeError("match run has already finished")
