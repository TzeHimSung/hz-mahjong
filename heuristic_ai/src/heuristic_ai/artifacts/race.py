from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from heuristic_ai.artifacts.serialization import _ArtifactSerializer, _canonical_hash
from heuristic_ai.artifacts.participant import ParticipantRunRecorder
from heuristic_ai.config.models import RaceConfig
from heuristic_ai.artifacts.models import RunResult, RunStatus
from heuristic_ai.strategy.models import policy_metadata
from heuristic_ai.protocol.compatibility import guide_compatibility_profile


class RaceRunRecorder(ParticipantRunRecorder):
    """Participant-only race shards with a minimal, immutable final export."""

    def __init__(
        self,
        config: RaceConfig,
        *,
        package_version: str,
        guide_version: int | None = None,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._config = config
        self._now = now
        self._serializer = _ArtifactSerializer((config.participant.token,), now)

        started = now()
        self.run_id = (
            f"{started.strftime('%Y_%m_%d_%H_%M_%S')}_{id_factory()[:8]}"
        )
        strategy_metadata = policy_metadata()
        public_config = {
            "server": {
                "url": config.server.url,
                "insecure": config.server.insecure,
                "timeout_seconds": config.server.timeout_seconds,
            },
            "room_id": config.competition.room_id,
            "seed": config.competition.seed,
            "poll_interval_seconds": config.competition.poll_interval_seconds,
            "decision_budget_ms": config.competition.decision_budget_ms,
            "strategy": strategy_metadata,
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "race_run",
            "mode": "race",
            "run_id": self.run_id,
            "status": RunStatus.RUNNING.value,
            "package_version": package_version,
            "guide_version": guide_version,
            "client_compatibility_profile": guide_compatibility_profile("race"),
            "config_hash": _canonical_hash(public_config),
            "room_id": config.competition.room_id,
            "seed": config.competition.seed,
            "started_at": started.isoformat(),
            "finished_at": None,
            "participant": {
                "user_id": None,
                "strategy": strategy_metadata,
            },
            "lifecycle": [],
            "game_ids": [],
            "games": [],
            "decision_log": {
                "schema_version": 4,
                "committed_bytes": 0,
                "committed_entries": 0,
            },
            "execution_counts": {},
            "decision_counts": {
                "total": 0,
                "accepted": 0,
                "conflict": 0,
                "error": 0,
                "fallback": 0,
            },
            "degradations": [],
            "summary": None,
            "error": None,
        }

        self.run_dir = config.record_output_dir / self.run_id
        self.manifest_path = self.run_dir / "manifest.json"
        self.decisions_path = self.run_dir / "decisions.jsonl"
        self._initialize_storage(config.competition.output_dir, "race", manifest)

    def complete(self, terminal_reason: str, participant_ranking: Mapping[str, object] | None = None) -> RunResult:
        if self._final_result is not None:
            raise RuntimeError("race run has already finished")
        return self._finish("complete", terminal_reason=terminal_reason, participant_ranking=participant_ranking)

    def fail(self, error: BaseException, *, terminal_reason: str | None = None,
             participant_ranking: Mapping[str, object] | None = None) -> RunResult:
        if self._final_result is not None:
            raise RuntimeError("race run has already finished")
        return self._finish("fail", **self._failure_arguments(error), terminal_reason=terminal_reason,
                            participant_ranking=participant_ranking)
