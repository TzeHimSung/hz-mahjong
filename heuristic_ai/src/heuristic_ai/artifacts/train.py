"""Training match recording and final artifact publication."""
from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from heuristic_ai.config.models import TrainConfig
from heuristic_ai.protocol.compatibility import guide_compatibility_profile
from heuristic_ai.strategy.models import policy_metadata
from .models import DecisionTrace, RunResult, RunStatus
from .recorder import RecorderBase
from .serialization import _ArtifactSerializer, _canonical_hash


_STATE_POLL_FIELDS = frozenset({
    "game_id", "player_name", "user_id", "seat", "generation", "request_purpose", "event_poll_mode",
    "request_seq", "response_seq", "snapshot_seq", "event_consumed_seq", "bootstrap_anchor",
    "started_at", "finished_at", "elapsed_ms", "pending", "gap", "finished", "raw_phase",
    "raw_turn", "responding_seats_present", "responding_seats", "waited_seat_present",
    "waited_seat", "observed_phase", "candidate_count", "decision_key", "skip_reason",
    "inventory_known", "inventory_reason", "same_seq_phase_changed", "error_type", "http_status",
    "round_no", "accepted_round_no", "snapshot_disposition", "snapshot_reason",
    "inventory_round_no", "round_anchor_seq", "verified_round_start_seq",
    "round_boundary_status", "coverage_start_seq", "covers_watermark", "event_count",
    "first_event_seq", "last_event_seq", "quarantined_event_count", "reassigned_event_count",
})


class RunRecorder(RecorderBase):
    """Train recorder with per-player durable shards and a final legacy export."""

    _poll_fields = _STATE_POLL_FIELDS

    def __init__(
        self,
        config: TrainConfig,
        *,
        package_version: str,
        guide_version: int | None = None,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._config = config
        self._tokens = tuple(player.token for player in config.players)
        self._now = now
        self._serializer = _ArtifactSerializer(self._tokens, now)
        started = now()
        suffix = id_factory()[:8]
        self.run_id = f"{started.strftime('%Y_%m_%d_%H_%M_%S')}_{suffix}"
        self.run_dir = config.record_output_dir / self.run_id
        self.rounds_dir = self.run_dir / "rounds"
        self.manifest_path = self.run_dir / "manifest.json"
        self.decisions_path = self.run_dir / "decisions.jsonl"
        self.state_polls_path = self.run_dir / "state_polls.jsonl"

        players: list[dict[str, Any]] = []
        public_players: list[dict[str, Any]] = []
        for player in config.players:
            strategy_meta = policy_metadata()
            players.append(
                {
                    "name": player.name,
                    "user_id": None,
                    "strategy": strategy_meta,
                }
            )
            public_players.append(
                {"name": player.name, "strategy": strategy_meta}
            )

        public_config = {
            "schema_version": 3,
            "server": {
                "url": config.server.url,
                "insecure": config.server.insecure,
                "timeout_seconds": config.server.timeout_seconds,
            },
            "room_id": config.competition.room_id,
            "seed": config.competition.seed,
            "poll_interval_seconds": config.competition.poll_interval_seconds,
            "decision_budget_ms": config.competition.decision_budget_ms,
            "players": public_players,
        }
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "run_id": self.run_id,
            "status": RunStatus.RUNNING.value,
            "package_version": package_version,
            "guide_version": guide_version,
            "client_compatibility_profile": guide_compatibility_profile("train"),
            "config_hash": _canonical_hash(public_config),
            "room_id": config.competition.room_id,
            "seed": config.competition.seed,
            "started_at": started.isoformat(),
            "finished_at": None,
            "game_ids": [],
            "games": [],
            "players": players,
            "execution_counts": {},
            "decision_counts": {
                "total": 0,
                "accepted": 0,
                "conflict": 0,
                "error": 0,
                "fallback": 0,
            },
            "degradations": [],
            "inventory_decision_counts": [],
            "state_poll_log": {"schema_version": 1, "path": "state_polls.jsonl", "entries": 0},
            "rounds": [],
            "summary": None,
            "error": None,
        }
        self._initialize_storage(config.competition.output_dir, "train", manifest,
                                 record_subdir=config.record_output_dir.name)

    def _trace_json(self, trace: DecisionTrace) -> dict[str, Any]:
        return self._serializer.trace_json(trace)

    def _sanitize(self, value: str) -> str:
        return self._serializer.sanitize(value)

    def record_guide_version(self, version: int | None) -> None:
        self._control("record_guide_version", version=version)

    def register_games(self, game_ids: tuple[str, ...], *, rounds_per_game: int) -> None:
        self._control("register_games", game_ids=game_ids, rounds_per_game=rounds_per_game)

    def identify_player(self, game_id: str | None, player_name: str, user_id: str, seat: int | None = None) -> None:
        self._control("identify_player", game_id=game_id, player_name=player_name, user_id=user_id, seat=seat)

    def mark_degraded(self, game_id: str, player_name: str, reason: str) -> None:
        self._control("mark_degraded", game_id=game_id, player_name=player_name, reason=reason)

    def _build_summary(self) -> dict[str, Any]:
        return self._projection._build_summary()

    def complete(self) -> RunResult:
        return self._finish("complete")

    def fail(self, error: BaseException) -> RunResult:
        return self._finish("fail", **self._failure_arguments(error))
