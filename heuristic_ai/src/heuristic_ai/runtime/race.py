from __future__ import annotations

import copy
import logging
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal, Protocol

from heuristic_ai import __version__
from heuristic_ai.protocol.client import ApiClient, ApiError
from heuristic_ai.runtime.bot import GameResult, GameStopped, _finished_scores, play_game
from heuristic_ai.config.models import RaceConfig
from heuristic_ai.strategy.models import GameContext
from heuristic_ai.game.models import RuleConfig
from heuristic_ai.artifacts.models import RunResult
from heuristic_ai.artifacts.race import RaceRunRecorder
from .rate_limit import ParticipantStateRequestGate, StateLimitedClient
from .network import RunNetwork
from .recovery import (
    RecoveryGroup,
    RecorderCallbackFailure as _RecorderCallbackFailure,
    guard_callback,
    invoke_recoverable_worker,
    process_control, recording_failure,
)
from heuristic_ai.strategy.one_ply import OnePlyController
from heuristic_ai.protocol.compatibility import normalize_rules, required_int
from .race_health import HealthComponent, RaceHealth
from .race_statistics import RaceStageStatistics
from .supervision_contracts import RaceRuntimeObserver, RaceRuntimeStatus, StopSignal


_TOURNAMENT_STATUSES = frozenset(
    {
        "registering",
        "running",
        "stage_open",
        "stage_done",
        "finished",
        "void",
        "closed",
    }
)
_RANKING_FIELDS = (
    "rank",
    "total_score",
    "place_points",
    "god_count",
    "games_played",
)
LOGGER = logging.getLogger("heuristic_ai.race")
_ConfirmationWindowIdentity = tuple[
    str,
    int | None,
    str | None,
    str | None,
    bool | None,
    bool | None,
    str | None,
]


class JsonClient(Protocol):
    def request(self, method: str, path: str, body: Any = None) -> Any:
        raise NotImplementedError


ClientFactory = Callable[[str], JsonClient]


@dataclass(frozen=True, slots=True)
class ValidatedRaceParticipant:
    token: str = field(repr=False)
    user_id: str
    tournament_id: str
    match_count: int
    rules: RuleConfig
    config: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class _TournamentSnapshot:
    status: str
    stage_no: int | None = None
    stage_name: str | None = None
    stage_status: str | None = None
    stage_crashed: bool | None = None
    qualified: bool | None = None
    qualify_role: str | None = None
    own_ranking: Mapping[str, int] | None = None
    game_ids: tuple[str, ...] | None = None
    observed_at: float | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class RaceOutcome:
    terminal_reason: Literal["finished", "eliminated", "void", "closed"]
    own_ranking: Mapping[str, int] | None = None
    result_unavailable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.terminal_reason not in {"finished", "eliminated", "void", "closed"}:
            raise ValueError("invalid race terminal reason")
        if self.own_ranking is not None:
            object.__setattr__(
                self,
                "own_ranking",
                MappingProxyType(dict(self.own_ranking)),
            )


@dataclass(slots=True)
class _WorkerState:
    game_id: str
    stage_no: int | None
    stop: threading.Event
    group: RecoveryGroup
    future: Future[GameResult] | None = None
    abort_reason: str | None = None
    health_component: HealthComponent | None = None


def _identity_failure(message: str) -> RuntimeError:
    error = RuntimeError(message)
    error.race_failure_category = "identity"
    return error


def _failure_category(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        categories = [_failure_category(item) for item in error.exceptions]
        return next((category for category in categories
                     if category in {"configuration", "authentication", "identity"}), "runtime")
    if isinstance(error, ApiError) and error.status in {401, 403}:
        return "authentication"
    category = getattr(error, "race_failure_category", None)
    return category if category in {"configuration", "identity"} else "runtime"


def _report_failure(observer: RaceRuntimeObserver | None, error: BaseException) -> None:
    if observer is not None and process_control(error) is None:
        observer.run_failed(_failure_category(error), _safe_error_type(error), _safe_failure_code(error))


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} response must be an object")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"tournament {field_name} must be a non-empty string")
    return value


def _optional_qualify_role(value: Any) -> str | None:
    # The live API uses an empty string until a qualification role is assigned.
    if value == "":
        return None
    return _optional_string(value, "qualify_role")


def _own_ranking(value: Any, user_id: str) -> Mapping[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeError("tournament ranking must be a list")
    own: dict[str, int] | None = None
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError("tournament ranking entry must be an object")
        entry_user_id = item.get("user_id")
        if not isinstance(entry_user_id, str) or not entry_user_id:
            raise RuntimeError("tournament ranking entry is missing user_id")
        if entry_user_id != user_id:
            continue
        if own is not None:
            raise RuntimeError("tournament ranking contains duplicate participant entries")
        own = {}
        for key in _RANKING_FIELDS:
            if key not in item:
                continue
            rank_value = item[key]
            if isinstance(rank_value, bool) or not isinstance(rank_value, int):
                raise RuntimeError(f"tournament ranking.{key} must be an integer")
            own[key] = rank_value
    return MappingProxyType(own) if own is not None else None


def _parse_tournament_snapshot(
    value: Any,
    *,
    room_id: str,
    user_id: str,
) -> _TournamentSnapshot:
    tournament = _object(value, "tournament")
    if tournament.get("tournament_id") != room_id:
        raise _identity_failure("tournament response does not match configured room id")
    status = tournament.get("status")
    if not isinstance(status, str) or status not in _TOURNAMENT_STATUSES:
        raise RuntimeError("tournament response has an invalid status")

    stage_no: int | None = None
    stage_name: str | None = None
    if "stage" in tournament and tournament["stage"] is not None:
        stage = tournament["stage"]
        if not isinstance(stage, dict):
            raise RuntimeError("tournament stage must be an object")
        stage_no = stage.get("no")
        if (
            isinstance(stage_no, bool)
            or not isinstance(stage_no, int)
            or stage_no < 0
        ):
            raise RuntimeError("tournament stage.no must be a non-negative integer")
        stage_name = stage.get("name")
        if not isinstance(stage_name, str) or not stage_name:
            raise RuntimeError("tournament stage.name must be a non-empty string")

    optional_bools: dict[str, bool | None] = {}
    for key in ("stage_crashed", "qualified"):
        item = tournament.get(key)
        if item is not None and not isinstance(item, bool):
            raise RuntimeError(f"tournament {key} must be a boolean")
        optional_bools[key] = item

    game_ids = None
    if isinstance(tournament.get("my_games"), list):
        identifiers = []
        for item in tournament["my_games"]:
            game_id = item.get("game_id") if isinstance(item, dict) else item
            if isinstance(item, dict) and item.get("stage_no", stage_no) != stage_no:
                continue
            if not isinstance(game_id, str) or not game_id:
                identifiers = None
                break
            identifiers.append(game_id)
        if identifiers is not None:
            game_ids = tuple(dict.fromkeys(identifiers))
    return _TournamentSnapshot(
        status=status,
        stage_no=stage_no,
        stage_name=stage_name,
        stage_status=_optional_string(tournament.get("stage_status"), "stage_status"),
        stage_crashed=optional_bools["stage_crashed"],
        qualified=optional_bools["qualified"],
        qualify_role=_optional_qualify_role(tournament.get("qualify_role")),
        own_ranking=_own_ranking(tournament.get("ranking"), user_id),
        game_ids=game_ids, observed_at=time.monotonic(),
    )


def validate_race_participant(
    config: RaceConfig,
    client_factory: ClientFactory,
) -> tuple[ValidatedRaceParticipant, _TournamentSnapshot]:
    """Validate all participant and tournament contracts without mutating the race."""

    client = client_factory(config.participant.token)
    me_value = client.request("GET", "/api/me")
    rules_value = client.request("GET", "/api/tournaments/me/rules")
    tournament_value = client.request(
        "GET",
        f"/api/tournaments/{config.competition.room_id}",
    )

    me = _object(me_value, "participant identity")
    rules_response = _object(rules_value, "tournament rules")
    tournament = _object(tournament_value, "tournament")

    user_id = me.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise RuntimeError("participant identity is missing user_id")
    tournament_id = me.get("tournament_id")
    if not isinstance(tournament_id, str) or not tournament_id:
        raise _identity_failure("participant token is not scoped to a tournament")

    room_id = config.competition.room_id
    if tournament_id != room_id:
        raise _identity_failure("participant tournament does not match configured room id")
    if rules_response.get("tournament_id") != room_id:
        raise _identity_failure("rules tournament does not match configured room id")
    if tournament.get("tournament_id") != room_id:
        raise _identity_failure("tournament response does not match configured room id")

    raw_config = rules_response.get("config")
    if not isinstance(raw_config, dict) or not all(
        isinstance(key, str) for key in raw_config
    ):
        raise RuntimeError("tournament rules must contain a config object")
    if raw_config.get("Kind") != "":
        error = RuntimeError("tournament config Kind must be the empty string")
        error.race_failure_category = "configuration"
        raise error
    match_count = required_int(raw_config, "M", minimum=1, maximum=16)
    normalized_rules = normalize_rules(raw_config)

    participant = ValidatedRaceParticipant(
        token=config.participant.token,
        user_id=user_id,
        tournament_id=tournament_id,
        match_count=match_count,

        rules=normalized_rules,
        config=copy.deepcopy(raw_config),
    )
    snapshot = _parse_tournament_snapshot(
        tournament,
        room_id=room_id,
        user_id=user_id,
    )
    return participant, snapshot


class RaceCoordinator:
    """Follow one participant through every authoritative tournament stage."""

    def __init__(
        self,
        *,
        participant: ValidatedRaceParticipant,
        control_client: JsonClient,
        client_factory: ClientFactory,
        recorder: Any,
        seed: int,
        decision_budget_ms: int,
        poll_interval: float,
        player_worker: Callable[..., GameResult] = play_game,
        sleep: Callable[[float], None] = time.sleep,
        max_control_polls: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        request_timeout: float = 35.0,
        observer: RaceRuntimeObserver | None = None,
        health: RaceHealth | None = None,
        stop: StopSignal | None = None,
        statistics: RaceStageStatistics | None = None,
    ) -> None:
        if decision_budget_ms < 1:
            raise ValueError("decision_budget_ms must be positive")
        if poll_interval < 0:
            raise ValueError("poll_interval cannot be negative")
        if max_control_polls is not None and (
            isinstance(max_control_polls, bool)
            or not isinstance(max_control_polls, int)
            or max_control_polls < 1
        ):
            raise ValueError("max_control_polls must be a positive integer")
        self.participant = participant
        self.control_client = control_client
        self.client_factory = client_factory
        self.recorder = recorder
        self.seed = seed
        self.decision_budget_ms = decision_budget_ms
        # v13+ OnlineConfirm races require authenticated traffic within 90s
        # of the freeze, including idle stages with no game notification stream.
        self.poll_interval = (
            min(poll_interval, 60.0)
            if participant.config.get("OnlineConfirm") is True
            else poll_interval
        )
        self.player_worker = player_worker
        self.sleep = sleep
        self.max_control_polls = max_control_polls
        self.clock = clock
        self.request_timeout = request_timeout
        self.stop = stop if stop is not None else threading.Event()
        self.observer = observer
        self.health = health
        self.failure_reported = False
        self._terminal_notified = False
        bind_stop = getattr(self.recorder, "bind_stop", None)
        if bind_stop is not None:
            bind_stop(self.stop)
        self._workers: dict[str, _WorkerState] = {}
        self._seen_game_ids: set[str] = set()
        self._groups: dict[str, RecoveryGroup] = {}
        self._game_lifecycles: dict[str, tuple[Any, ...]] = {}
        self._completed_game_ids: set[str] = set()
        self._aborted_game_ids: set[str] = set()
        self._result_unavailable: set[str] = set()
        self._probe_needed: set[str] = set()
        self._probe_attempted: set[str] = set()
        self._probes: dict[str, Future[GameResult]] = {}
        self._probe_health: dict[str, HealthComponent] = {}
        self.statistics = statistics if statistics is not None else RaceStageStatistics(participant.rules.rounds)
        self._round_observers: dict[str, Callable[..., None]] = {}
        self._round_evidence: dict[str, Any] = {}
        self._last_snapshot: _TournamentSnapshot | None = None
        self._last_server_poll_at: float | None = None
        self._server_poll_in_flight = False
        self._server_poll_error: str | None = None
        self._confirmed_window: _ConfirmationWindowIdentity | None = None
        self._state_lock = threading.RLock()
        self._status_revision = 0
        self._probe_executor: ThreadPoolExecutor | None = None
        self._executor: ThreadPoolExecutor | None = None
        self.state_request_gate = ParticipantStateRequestGate()

    def _operation(self, label: str, seconds: float | None = None) -> AbstractContextManager[Any]:
        return self.health.operation(label, seconds) if self.health else nullcontext()

    def _checkpoint(self, label: str) -> None:
        if self.health:
            self.health.checkpoint(label)

    def _notify_terminal(self, snapshot: _TournamentSnapshot, reason: str | None = None) -> None:
        if self._terminal_notified or self.observer is None:
            return
        if reason is None:
            if snapshot.status in {"finished", "closed", "void"}:
                reason = snapshot.status
            elif snapshot.status == "stage_open" and snapshot.qualified is False:
                reason = "eliminated"
        if reason is not None:
            # Commit the business fact before waiting for workers or artifacts.
            # Cancellation is nonblocking; do not keep submitting actions while
            # the parent durably acknowledges the terminal fact.
            for state in tuple(self._workers.values()):
                state.stop.set()
            with self._operation("terminal_acknowledgement", 10.0):
                self.observer.business_terminal(reason, snapshot.own_ranking)
            self._terminal_notified = True

    def _emit_state(self, snapshot: _TournamentSnapshot) -> None:
        with self._state_lock:
            if self._last_snapshot is not None and snapshot is not self._last_snapshot:
                return
            self._emit_current_state(snapshot)

    def _emit_current_state(self, snapshot: _TournamentSnapshot) -> None:
        if self.observer is None:
            return
        with self._operation("progress_projection"):
            progress_method = getattr(self.recorder, "progress", None)
            progress = progress_method() if callable(progress_method) else {}
            ranking = snapshot.own_ranking or {}
            if snapshot.stage_crashed:
                waiting_reason = "stage_replay"
            elif snapshot.status == "stage_done":
                waiting_reason = "between_stages"
            elif snapshot.status == "running":
                waiting_reason = "assignment"
            else:
                waiting_reason = "ready" if self._confirmed_window == self._confirmation_window_identity(snapshot) else "not_started"
            self._status_revision += 1
            status = RaceRuntimeStatus(
                run_id=progress.get("run_id", getattr(self.recorder, "run_id", None)),
                phase="finalizing" if self._terminal_notified else "playing" if self._workers and snapshot.status == "running" else "waiting",
                tournament_status=snapshot.status, stage_no=snapshot.stage_no,
                stage_name=snapshot.stage_name, stage_status=snapshot.stage_status,
                qualified=snapshot.qualified, qualify_role=snapshot.qualify_role,
                rank=ranking.get("rank"), total_score=ranking.get("total_score"),
                stage_settled_rounds=ranking.get("games_played"), active_games=len(self._workers),
                decisions=dict(progress.get("decisions", {})),
                audit_synced_bytes=progress.get("synced_bytes"),
                record_p99_ms=progress.get("record_p99_ms"),
                waiting_reason=waiting_reason,
                ready=self._confirmed_window is not None and self._confirmed_window == self._confirmation_window_identity(snapshot),
                ranking_scope="void" if snapshot.stage_crashed else "previous" if snapshot.status == "stage_open" else "current",
                stage_stats=self.statistics.snapshot(), last_server_poll_at=self._last_server_poll_at,
                server_poll_in_flight=self._server_poll_in_flight, server_poll_error=self._server_poll_error,
                revision=self._status_revision,
            )
            final_state = getattr(self.observer, "final_state", None)
            if self._terminal_notified and callable(final_state):
                final_state(status)
            else:
                self.observer.race_state(status)

    def _accept_snapshot(self, snapshot: _TournamentSnapshot) -> None:
        with self._state_lock:
            self._accept_current_snapshot(snapshot)

    def _accept_current_snapshot(self, snapshot: _TournamentSnapshot) -> None:
        self._last_snapshot = snapshot
        self._last_server_poll_at = snapshot.observed_at if snapshot.observed_at is not None else self.clock()
        self._server_poll_error = None
        self._server_poll_in_flight = False
        with self._operation("stage_statistics"):
            self.statistics.observe_tournament(stage_no=snapshot.stage_no, status=snapshot.status,
                                               crashed=snapshot.stage_crashed is True, game_ids=snapshot.game_ids)

    def _await_future(self, future: Future[GameResult]) -> GameResult:
        if self.health is None:
            return future.result()
        while True:
            try:
                return future.result(timeout=0.5)
            except FutureTimeoutError:
                if future.done():
                    # A task itself may have raised TimeoutError.
                    return future.result()
                # This is an actual join of another independently monitored
                # component, not a heartbeat pretending that work progressed.
                components = [state.health_component for state in self._workers.values()
                              if state.future is future]
                components.extend(self._probe_health.get(game_id)
                                  for game_id, item in self._probes.items() if item is future)
                if any(self.health.is_active(component) for component in components):
                    self._checkpoint("waiting_for_worker_cleanup")

    def _fetch_tournament(self) -> _TournamentSnapshot:
        self._server_poll_in_flight = True
        if self._last_snapshot is not None:
            self._emit_state(self._last_snapshot)
        try:
            value = self.control_client.request("GET", f"/api/tournaments/{self.participant.tournament_id}")
            snapshot = _parse_tournament_snapshot(value, room_id=self.participant.tournament_id,
                                                  user_id=self.participant.user_id)
        except Exception as error:
            self._server_poll_in_flight = False
            self._server_poll_error = _safe_failure_code(error)
            if self._last_snapshot is not None:
                self._emit_state(self._last_snapshot)
            raise
        self._notify_terminal(snapshot)
        self._accept_snapshot(snapshot)
        self._checkpoint("tournament_fetched")
        return snapshot

    def _record_lifecycle(self, snapshot: _TournamentSnapshot) -> None:
        self._accept_snapshot(snapshot)
        optional = {
            "stage_no": snapshot.stage_no,
            "stage_name": snapshot.stage_name,
            "stage_status": snapshot.stage_status,
            "stage_crashed": snapshot.stage_crashed,
            "qualified": snapshot.qualified,
            "qualify_role": snapshot.qualify_role,
        }
        self.recorder.record_lifecycle(
            snapshot.status,
            **{key: value for key, value in optional.items() if value is not None},
        )
        self._emit_state(snapshot)

    def _active_game_ids(self, response: Any) -> tuple[str, ...]:
        identity = _object(response, "participant identity")
        if identity.get("user_id") != self.participant.user_id:
            raise _identity_failure("participant identity changed during race")
        if identity.get("tournament_id") != self.participant.tournament_id:
            raise _identity_failure("participant tournament changed during race")
        active_games = identity.get("active_games")
        if not isinstance(active_games, list):
            raise RuntimeError("active_games must be a list")
        game_ids: list[str] = []
        for item in active_games:
            if not isinstance(item, dict):
                raise RuntimeError("active game must be an object")
            game_id = item.get("game_id")
            if not isinstance(game_id, str) or not game_id:
                raise RuntimeError("active game is missing game_id")
            game_ids.append(game_id)
        if len(set(game_ids)) != len(game_ids):
            raise RuntimeError("active games contain duplicate game ids")
        if len(game_ids) > self.participant.match_count:
            raise RuntimeError("active games exceed tournament config M")
        return tuple(game_ids)

    @staticmethod
    def _lifecycle_identity(
        snapshot: _TournamentSnapshot,
    ) -> tuple[str, int | None, str | None, bool | None]:
        return (
            snapshot.status,
            snapshot.stage_no,
            snapshot.stage_status,
            snapshot.stage_crashed,
        )

    @staticmethod
    def _confirmation_window_identity(
        snapshot: _TournamentSnapshot,
    ) -> _ConfirmationWindowIdentity | None:
        if snapshot.status not in {"registering", "stage_open"}:
            return None
        return (
            snapshot.status,
            snapshot.stage_no,
            snapshot.stage_name,
            snapshot.stage_status,
            snapshot.stage_crashed,
            snapshot.qualified,
            snapshot.qualify_role,
        )

    @staticmethod
    def _future(state: _WorkerState) -> Future[GameResult]:
        if state.future is None:
            raise RuntimeError("race worker was not submitted")
        return state.future

    @staticmethod
    def _stage_matches(
        worker_stage_no: int | None,
        snapshot_stage_no: int | None,
    ) -> bool:
        return (
            worker_stage_no is None
            or snapshot_stage_no is None
            or worker_stage_no == snapshot_stage_no
        )

    @staticmethod
    def _is_expected_game_not_found(
        error: BaseException,
        state: _WorkerState,
        snapshot: _TournamentSnapshot,
    ) -> bool:
        return (
            RaceCoordinator._is_exact_game_not_found(error)
            and snapshot.stage_crashed is True
            and RaceCoordinator._stage_matches(state.stage_no, snapshot.stage_no)
        )

    @staticmethod
    def _is_exact_game_not_found(error: BaseException) -> bool:
        return (
            type(error) is ApiError
            and error.status == 404
            and error.code == "GAME_NOT_FOUND"
        )

    def _mark_completed(self, result: GameResult) -> None:
        if result.game_id in self._aborted_game_ids:
            return
        guard_callback(self.recorder.mark_game_completed, self.stop)(result.game_id, result.scores)
        evidence = self._round_evidence.get(result.game_id)
        if evidence is not None:
            evidence.recorder.record_round_progress(result.game_id, self.participant.rules.rounds)
        self._completed_game_ids.add(result.game_id)
        self._probe_needed.discard(result.game_id)
        group = self._groups[result.game_id]
        group.terminal("completed")
        if result.degraded:
            guard_callback(self.recorder.mark_degraded, self.stop)(
                result.game_id, self.participant.user_id,
                result.degradation_reason or "strategy_session_degraded",
            )

    def _mark_aborted(self, game_id: str, reason: str) -> None:
        if game_id in self._completed_game_ids or game_id in self._aborted_game_ids:
            return
        guard_callback(self.recorder.mark_game_aborted, self.stop)(game_id, reason)
        self._aborted_game_ids.add(game_id)
        self._probe_needed.discard(game_id)
        if reason == "result_unavailable":
            self._result_unavailable.add(game_id)

    def _consume_worker(
        self,
        state: _WorkerState,
        snapshot: _TournamentSnapshot,
    ) -> _TournamentSnapshot:
        latest = snapshot
        try:
            result = self._await_future(self._future(state))
        except BaseException as error:
            if type(error) is _RecorderCallbackFailure:
                raise error.original from None
            if not isinstance(error, Exception):
                raise
            if self.observer is not None and _failure_category(error) in {"configuration", "authentication", "identity"}:
                raise
            if (
                self._is_exact_game_not_found(error)
                and not self.stop.is_set()
                and state.abort_reason is None
                and state.group.terminal_reason != "stage_crashed"
                and not self._is_expected_game_not_found(error, state, latest)
            ):
                latest = self._fetch_tournament()
                self._record_lifecycle(latest)
            if self._is_expected_game_not_found(error, state, latest):
                state.group.terminal("stage_crashed")
                self._mark_aborted(state.game_id, "stage_crashed")
            elif state.abort_reason is not None:
                # The authoritative lifecycle cancels this game even if cleanup
                # reports another ordinary error. Keep its evidence separately.
                if type(error) is not GameStopped:
                    state.group.note_failure(self.participant.user_id, error)
                self._mark_aborted(state.game_id, state.abort_reason)
            elif type(error) is GameStopped and state.stop.is_set():
                pass
            else:
                state.group.note_failure(self.participant.user_id, error)
        else:
            self._mark_completed(result)
        del self._workers[state.game_id]
        return latest

    def _reap_completed(
        self,
        snapshot: _TournamentSnapshot,
    ) -> _TournamentSnapshot:
        latest = snapshot
        for state in tuple(self._workers.values()):
            if self._future(state).done():
                try:
                    latest = self._consume_worker(state, latest)
                finally:
                    self._workers.pop(state.game_id, None)
        self._reap_probes()
        self._start_probes()
        self._release_retired_groups()
        return latest

    def _drain_workers(
        self,
        snapshot: _TournamentSnapshot,
    ) -> _TournamentSnapshot:
        """Final shutdown only; stage confirmations never join player tasks."""
        latest = snapshot
        primary: BaseException | None = None
        for state in tuple(self._workers.values()):
            try:
                latest = self._consume_worker(state, latest)
            except BaseException as error:
                if primary is None or process_control(primary) is None and process_control(error) is not None:
                    primary = process_control(error) or error
            finally:
                self._workers.pop(state.game_id, None)
        try:
            if not self.stop.is_set():
                self._start_probes()
            self._reap_probes(wait=True)
        except BaseException as error:
            if primary is None or process_control(primary) is None and process_control(error) is not None:
                primary = process_control(error) or error
        self._release_retired_groups()
        if primary is not None:
            raise primary
        return latest

    def _release_retired_groups(self) -> None:
        # Historical identities remain in scalar sets; action memory and stop
        # tokens are needed only until the last action/probe task has exited.
        for game_id in tuple(self._groups):
            if (
                game_id not in self._workers and game_id not in self._probes
                and (game_id in self._completed_game_ids or game_id in self._aborted_game_ids)
            ):
                del self._groups[game_id]
                self._game_lifecycles.pop(game_id, None)
                self._probe_attempted.discard(game_id)

    def _retire_games(
        self,
        snapshot: _TournamentSnapshot,
        reason: str = "result_unavailable",
        *,
        stage_only: bool = False,
        inactive_game_ids: set[str] | None = None,
    ) -> _TournamentSnapshot:
        for game_id, group in self._groups.items():
            if game_id in self._completed_game_ids or game_id in self._aborted_game_ids:
                continue
            if stage_only and not self._stage_matches(group.stage_no, snapshot.stage_no):
                continue
            if inactive_game_ids is not None and game_id in inactive_game_ids:
                continue
            game_reason = reason
            if (
                reason == "result_unavailable" and snapshot.stage_crashed is True
                and self._stage_matches(group.stage_no, snapshot.stage_no)
            ):
                game_reason = "stage_crashed"
            group.terminal("finished" if game_reason == "result_unavailable" else game_reason)
            state = self._workers.get(game_id)
            if state is not None:
                state.stop.set()
                if game_reason != "result_unavailable":
                    state.abort_reason = game_reason
            if game_reason == "result_unavailable":
                self._probe_needed.add(game_id)
            elif state is None:
                self._mark_aborted(game_id, game_reason)
        return self._reap_completed(snapshot)

    def _abort_workers(
        self,
        snapshot: _TournamentSnapshot,
        reason: str,
        *,
        stage_only: bool = False,
        inactive_game_ids: set[str] | None = None,
    ) -> _TournamentSnapshot:
        return self._retire_games(
            snapshot, reason, stage_only=stage_only,
            inactive_game_ids=inactive_game_ids,
        )

    _guard_recorder_callback = staticmethod(guard_callback)

    def _invoke_worker(self, state: _WorkerState, game: GameContext) -> GameResult:
        context = self.health.bind(state.health_component) if self.health and state.health_component else nullcontext()
        with context:
            return self._invoke_monitored_worker(state, game)

    def _invoke_monitored_worker(self, state: _WorkerState, game: GameContext) -> GameResult:
        try:
            # Construction belongs to this game generation, so strategy/client
            # initialization failures have exactly the same isolation boundary.
            client = StateLimitedClient(
                self.client_factory(self.participant.token),
                self.state_request_gate, state.stop,
            )
            with self._operation("strategy_initialization"):
                controller = OnePlyController(game)
            result = invoke_recoverable_worker(
                state.group, self.participant.user_id, self.player_worker,
                client, game, controller, self.recorder,
                health_operation=self.health.operation if self.health else None,
                on_health_checkpoint=self.health.checkpoint if self.health else None,
                on_state_response=self._round_observers.get(game.game_id),
            )
        except BaseException as error:
            if self.observer is not None and _failure_category(error) in {"configuration", "authentication", "identity"}:
                _report_failure(self.observer, error)
                self.failure_reported = True
                self.stop.set()
            if isinstance(error, Exception) and type(error) is not _RecorderCallbackFailure:
                if not (type(error) is GameStopped and state.stop.is_set()):
                    state.group.note_failure(self.participant.user_id, error)
            raise
        return result

    def _start_game(self, game_id: str, snapshot: _TournamentSnapshot) -> None:
        if game_id in self._workers:
            return
        if self._executor is None:
            raise RuntimeError("race worker pool is not running")
        group = self._groups.get(game_id)
        if group is None:
            if game_id in self._seen_game_ids:
                return
            self.recorder.register_game(
                game_id, stage_no=snapshot.stage_no, stage_name=snapshot.stage_name,
            )
            self._round_evidence[game_id] = self.statistics.evidence(game_id)
            observe = self.statistics.observe(game_id)

            def on_state(response: Any, *, request_seq: int | None = None) -> None:
                before = self.statistics.snapshot()
                observe(response, request_seq=request_seq)
                if before != self.statistics.snapshot() and self._last_snapshot is not None:
                    self._emit_state(self._last_snapshot)

            self._round_observers[game_id] = on_state
            group = RecoveryGroup(
                game_id, {self.participant.user_id: self.participant.user_id},
                self.stop, self.recorder, clock=self.clock, stage_no=snapshot.stage_no,
            )
            self._groups[game_id] = group
            self._game_lifecycles[game_id] = self._lifecycle_identity(snapshot)
            self._seen_game_ids.add(game_id)
        else:
            if group.terminal_reason is not None or not group.pending:
                return
            if self._game_lifecycles[game_id] != self._lifecycle_identity(snapshot):
                return
            if group.next_retry_at is None or self.clock() < group.next_retry_at:
                return
            group.begin_restart()
        game = GameContext(
            game_id=game_id, player_name=self.participant.user_id,
            user_id=self.participant.user_id, seed=self.seed,
            rules=self.participant.rules, decision_budget_ms=self.decision_budget_ms,
        )
        state = _WorkerState(
            game_id=game_id, stage_no=snapshot.stage_no,
            stop=group.stop, group=group,
        )
        self._workers[game_id] = state
        if self.health:
            state.health_component = self.health.register("game:" + game_id, group.generation)
        try:
            state.future = self._executor.submit(self._invoke_worker, state, game)
        except BaseException:
            if state.health_component:
                state.health_component.close()
            self._workers.pop(game_id, None)
            raise

    def _invoke_probe(self, game_id: str, component: HealthComponent | None) -> GameResult:
        context = self.health.bind(component) if self.health and component else nullcontext()
        with context:
            return self._terminal_probe(game_id)

    def _terminal_probe(self, game_id: str) -> GameResult:
        # A different client and the run stop token permit read-only collection
        # after the action generation has been stopped. The participant gate is
        # shared, and the transport gets one deadline with no HTTP retries.
        client = StateLimitedClient(
            self.client_factory(self.participant.token), self.state_request_gate,
            self.stop,
        )
        with self._operation("terminal_result_request", self.request_timeout):
            response = client.request_once(
                "GET", f"/api/games/{game_id}/state?seq=0",
                deadline_monotonic=time.monotonic() + self.request_timeout,
            )
        if not isinstance(response, dict) or response.get("finished") is not True:
            raise RuntimeError("terminal result is not available")
        on_state = self._round_observers.get(game_id)
        if on_state is not None:
            on_state(response, request_seq=0)
        return GameResult(game_id, _finished_scores(response), False)

    def _start_probes(self) -> None:
        if self._probe_executor is None:
            return
        for game_id in tuple(self._probe_needed):
            if game_id in self._workers or game_id in self._probe_attempted:
                continue
            self._probe_attempted.add(game_id)
            component = self.health.register("result:" + game_id) if self.health else None
            if component is not None:
                self._probe_health[game_id] = component
            try:
                self._probes[game_id] = self._probe_executor.submit(self._invoke_probe, game_id, component)
            except BaseException:
                if component:
                    component.close()
                self._probe_health.pop(game_id, None)
                raise

    def _reap_probes(self, *, wait: bool = False) -> None:
        primary: BaseException | None = None
        for game_id, future in tuple(self._probes.items()):
            if not wait and not future.done():
                continue
            try:
                try:
                    result = self._await_future(future)
                except BaseException as error:
                    if not isinstance(error, Exception) or self.stop.is_set():
                        raise
                    if game_id not in self._aborted_game_ids:
                        self._groups[game_id].note_failure(
                            self.participant.user_id, error,
                            context={"operation": "terminal_result_probe"},
                        )
                        self._mark_aborted(game_id, "result_unavailable")
                else:
                    if game_id not in self._aborted_game_ids:
                        self._mark_completed(result)
                        if self._groups[game_id].reported(self.participant.user_id):
                            guard_callback(self.recorder.mark_degraded, self.stop)(
                                game_id, self.participant.user_id, "terminal_result_recovered",
                            )
            except BaseException as error:
                if not (type(error) is GameStopped and self.stop.is_set()):
                    if primary is None or process_control(primary) is None and process_control(error) is not None:
                        primary = process_control(error) or error
            finally:
                del self._probes[game_id]
                self._probe_health.pop(game_id, None)
        if primary is not None:
            raise primary

    def _outcome(self, reason: Any, snapshot: _TournamentSnapshot) -> RaceOutcome:
        self._notify_terminal(snapshot, reason)
        self._emit_state(snapshot)
        return RaceOutcome(reason, snapshot.own_ranking, tuple(sorted(self._result_unavailable)))

    def _ready(
        self,
        snapshot: _TournamentSnapshot,
    ) -> tuple[RaceOutcome | None, _TournamentSnapshot | None]:
        if self.stop.is_set():
            raise GameStopped("race stopped before confirmation")
        ensure_audit = getattr(self.recorder, "ensure_audit_healthy", None)
        if ensure_audit is not None:
            ensure_audit()
        try:
            self.control_client.request("POST", "/api/tournaments/me/ready")
        except ApiError as error:
            if error.status != 409:
                raise
            if error.code == "TOURNAMENT_STARTED":
                latest = self._fetch_tournament()
                self._record_lifecycle(latest)
                return None, latest
            if error.code == "TOURNAMENT_CLOSED":
                self._notify_terminal(snapshot, "closed")
                self._abort_workers(snapshot, "tournament_closed")
                self._drain_workers(snapshot)
                return self._outcome("closed", snapshot), None
            if error.code == "NOT_QUALIFIED":
                latest = self._fetch_tournament()
                self._record_lifecycle(latest)
                if latest.status == "stage_open" and latest.qualified is False:
                    self._retire_games(latest)
                    drained = self._drain_workers(latest)
                    if drained is not latest:
                        return None, drained
                    return self._outcome("eliminated", latest), None
                raise RuntimeError(
                    "race ready was rejected without an authoritative elimination"
                ) from None
            raise
        self._confirmed_window = self._confirmation_window_identity(snapshot)
        self._emit_state(snapshot)
        return None, None

    def _stop_after_failure(self) -> BaseException | None:
        self.stop.set()
        states = tuple(self._workers.values())
        primary: BaseException | None = None
        for state in states:
            state.stop.set()
        for state in states:
            try:
                result = self._await_future(self._future(state))
            except BaseException as error:
                control = process_control(error)
                callback = recording_failure(error)
                if control is not None:
                    primary = control if primary is None or process_control(primary) is None else primary
                elif callback is not None and (primary is None or process_control(primary) is None):
                    primary = callback
            else:
                try:
                    self._mark_completed(result)
                except BaseException as error:
                    if primary is None or process_control(primary) is None and process_control(error) is not None:
                        primary = process_control(error) or error
            finally:
                self._workers.pop(state.game_id, None)
        try:
            # Existing reads may already hold valid final results. Drain them
            # without scheduling any new state/control request during shutdown.
            self._reap_probes(wait=True)
        except BaseException as error:
            if primary is None or process_control(primary) is None and process_control(error) is not None:
                primary = process_control(error) or error
        self._release_retired_groups()
        if primary is not None and process_control(primary) is None:
            raise primary
        return primary

    def run(
        self,
        initial_snapshot: _TournamentSnapshot | None = None,
    ) -> RaceOutcome:
        context = self.health.component("coordinator") if self.health else nullcontext()
        with context:
            return self._run(initial_snapshot)

    def _run(self, initial_snapshot: _TournamentSnapshot | None = None) -> RaceOutcome:
        confirmation_window_identity: _ConfirmationWindowIdentity | None = None
        confirmation_attempted = False
        snapshot_already_recorded = False
        polls = 0
        primary_error: BaseException | None = None
        process_control_error: BaseException | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=self.participant.match_count,
            thread_name_prefix="mahjong-race",
        )
        self._probe_executor = ThreadPoolExecutor(
            max_workers=self.participant.match_count,
            thread_name_prefix="mahjong-race-results",
        )
        try:
            snapshot = initial_snapshot or self._fetch_tournament()
            if self.observer is not None:
                self.observer.coordinator_active()
            while True:
                self._notify_terminal(snapshot)
                self._checkpoint("control_poll")
                ensure_audit = getattr(self.recorder, "ensure_audit_healthy", None)
                if ensure_audit is not None:
                    ensure_audit()
                polls += 1
                if self.max_control_polls is not None and polls > self.max_control_polls:
                    raise TimeoutError("race control poll limit reached")
                if not snapshot_already_recorded:
                    self._record_lifecycle(snapshot)
                snapshot_already_recorded = False
                snapshot = self._reap_completed(snapshot)
                if self.stop.is_set():
                    # Await the originating callback exception before reporting
                    # the parent stop; its type may itself be GameStopped/404.
                    self._drain_workers(snapshot)
                    raise GameStopped("race stopped")
                observed_window_identity = self._confirmation_window_identity(snapshot)
                if observed_window_identity != confirmation_window_identity:
                    confirmation_window_identity = observed_window_identity
                    confirmation_attempted = False

                if snapshot.status in {"void", "closed"}:
                    reason = f"tournament_{snapshot.status}"
                    aborted = self._abort_workers(snapshot, reason)
                    if aborted is not snapshot:
                        snapshot = aborted
                        snapshot_already_recorded = True
                        continue
                    self._drain_workers(snapshot)
                    return self._outcome(snapshot.status, snapshot)

                if snapshot.status == "finished":
                    retired = self._retire_games(snapshot)
                    if retired is not snapshot:
                        snapshot = retired
                        snapshot_already_recorded = True
                        continue
                    drained = self._drain_workers(snapshot)
                    if drained is not snapshot:
                        snapshot = drained
                        snapshot_already_recorded = True
                        continue
                    return self._outcome("finished", snapshot)

                if snapshot.status == "registering":
                    if not confirmation_attempted:
                        confirmation_attempted = True
                        outcome, transition = self._ready(snapshot)
                        if outcome is not None:
                            return outcome
                        if transition is not None:
                            snapshot = transition
                            snapshot_already_recorded = True
                            continue
                elif snapshot.status == "running":
                    assignment_response = self.control_client.request("GET", "/api/me")
                    latest = self._fetch_tournament()
                    if (
                        latest.status != "running"
                        or self._lifecycle_identity(latest)
                        != self._lifecycle_identity(snapshot)
                    ):
                        snapshot = latest
                        continue
                    snapshot = latest
                    self._record_lifecycle(snapshot)
                    active_game_ids = self._active_game_ids(assignment_response)
                    if snapshot.stage_crashed is True:
                        aborted = self._abort_workers(
                            snapshot,
                            "stage_crashed",
                            stage_only=True,
                            inactive_game_ids=set(active_game_ids),
                        )
                        if aborted is not snapshot:
                            snapshot = aborted
                            snapshot_already_recorded = True
                            continue
                    # A disappeared assignment or an older stage must never
                    # restart actions, even if its local retry deadline is due.
                    for game_id, group in self._groups.items():
                        if (
                            group.pending and group.terminal_reason is None
                            and self._game_lifecycles[game_id] != self._lifecycle_identity(snapshot)
                        ):
                            if (
                                snapshot.stage_crashed is True
                                and self._stage_matches(group.stage_no, snapshot.stage_no)
                            ):
                                group.terminal("stage_crashed")
                                state = self._workers.get(game_id)
                                if state is None:
                                    self._mark_aborted(game_id, "stage_crashed")
                                else:
                                    state.abort_reason = "stage_crashed"
                            else:
                                group.terminal("finished")
                                self._probe_needed.add(game_id)
                    retired = self._retire_games(
                        snapshot, inactive_game_ids=set(active_game_ids),
                    )
                    if retired is not snapshot:
                        snapshot = retired
                        snapshot_already_recorded = True
                        continue
                    for game_id in active_game_ids:
                        self._start_game(game_id, snapshot)
                elif snapshot.status == "stage_done":
                    if snapshot.stage_crashed is True:
                        aborted = self._abort_workers(
                            snapshot,
                            "stage_crashed",
                            stage_only=True,
                        )
                        if aborted is not snapshot:
                            snapshot = aborted
                            snapshot_already_recorded = True
                            continue
                    else:
                        retired = self._retire_games(snapshot)
                        if retired is not snapshot:
                            snapshot = retired
                            snapshot_already_recorded = True
                            continue
                elif snapshot.status == "stage_open":
                    if not isinstance(snapshot.qualified, bool):
                        raise RuntimeError(
                            "tournament stage_open response requires qualified boolean"
                        )
                    retired = self._retire_games(snapshot)
                    if retired is not snapshot:
                        snapshot = retired
                        snapshot_already_recorded = True
                        continue
                    if snapshot.qualified is False:
                        drained = self._drain_workers(snapshot)
                        if drained is not snapshot:
                            snapshot = drained
                            snapshot_already_recorded = True
                            continue
                        return self._outcome("eliminated", snapshot)
                    if not confirmation_attempted:
                        confirmation_attempted = True
                        outcome, transition = self._ready(snapshot)
                        if outcome is not None:
                            return outcome
                        if transition is not None:
                            snapshot = transition
                            snapshot_already_recorded = True
                            continue

                delay = min(self.poll_interval, 5.0) if snapshot.status != "running" else self.poll_interval
                deadlines = [
                    group.next_retry_at for group in self._groups.values()
                    if group.pending and group.next_retry_at is not None
                    and group.game_id not in self._workers
                ]
                if deadlines:
                    delay = min(delay, max(0.0, min(deadlines) - self.clock()))
                self._emit_state(snapshot)
                with self._operation("control_poll_wait", delay):
                    if self.sleep is time.sleep:
                        self.stop.wait(delay)
                    else:
                        self.sleep(delay)
                if self.stop.is_set():
                    self._drain_workers(snapshot)
                    raise GameStopped("race stopped")
                snapshot = self._fetch_tournament()
        except BaseException as error:
            if type(error) is _RecorderCallbackFailure:
                error = error.original
            primary_error = error
            if self.observer is not None and not self.failure_reported:
                _report_failure(self.observer, error)
                self.failure_reported = True
            self._checkpoint("failure_cleanup")
            if type(error) in {KeyboardInterrupt, SystemExit}:
                process_control_error = error
            try:
                cleanup_process_control = self._stop_after_failure()
            except BaseException as cleanup_error:
                if process_control_error is not None:
                    raise error
                if type(cleanup_error) in {KeyboardInterrupt, SystemExit}:
                    process_control_error = cleanup_error
                    primary_error = cleanup_error
                    raise cleanup_error
                if isinstance(error, Exception) and isinstance(
                    cleanup_error, Exception
                ):
                    raise ExceptionGroup(
                        "race execution and cleanup failures",
                        (error, cleanup_error),
                    ) from None
                raise cleanup_error
            if process_control_error is None and cleanup_process_control is not None:
                process_control_error = cleanup_process_control
                primary_error = cleanup_process_control
                raise cleanup_process_control
            raise error
        finally:
            executor = self._executor
            self._executor = None
            probe_executor = self._probe_executor
            self._probe_executor = None
            shutdown_errors: list[BaseException] = []
            shutdown_control: BaseException | None = None
            for pool in (probe_executor, executor):
                if pool is None:
                    continue
                try:
                    with self._operation("executor_shutdown"):
                        pool.shutdown(wait=True)
                except BaseException as cleanup_error:
                    if type(cleanup_error) in {KeyboardInterrupt, SystemExit}:
                        shutdown_control = shutdown_control or cleanup_error
                    else:
                        shutdown_errors.append(cleanup_error)
            if process_control_error is None:
                if shutdown_control is not None:
                    raise shutdown_control
                if shutdown_errors:
                    errors = ([primary_error] if primary_error is not None else []) + shutdown_errors
                    if len(errors) == 1:
                        raise errors[0]
                    raise BaseExceptionGroup(
                        "race execution and executor shutdown failures", errors,
                    ) from None


def _record_failed_run(
    recorder: RaceRunRecorder,
    primary: BaseException,
    *,
    terminal_reason: Literal["void", "closed"] | None = None,
    own_ranking: Mapping[str, int] | None = None,
) -> RunResult:
    recorder_error: BaseException | None = None
    try:
        return recorder.fail(
            primary,
            terminal_reason=terminal_reason,
            participant_ranking=own_ranking,
        )
    except Exception as error:
        recorder_error = error
    raise BaseExceptionGroup(
        "race execution and artifact finalization failed",
        (
            RuntimeError(f"primary failure: {_safe_error_type(primary)}"),
            RuntimeError(
                f"artifact finalization failure: {_safe_error_type(recorder_error)}"
            ),
        ),
    ) from None


def _safe_error_type(error: BaseException) -> str:
    return {
        ApiError: "ApiError",
        RuntimeError: "RuntimeError",
        ValueError: "ValueError",
        OSError: "OSError",
        KeyboardInterrupt: "KeyboardInterrupt",
        SystemExit: "SystemExit",
    }.get(type(error), "unexpected_error")


def _safe_failure_code(error: BaseException) -> str:
    if type(error) is not RuntimeError or len(error.args) != 1:
        return "unclassified"
    message = error.args[0]
    if type(message) is not str:
        return "unclassified"
    if message.startswith("transport failed after "):
        return "transport_exhausted"
    if message in {
        "finished game response must be an object",
        "finished game snapshot must contain four scores",
        "finished game scores must be integers",
    }:
        return "finished_response_shape"
    if message.startswith("tournament "):
        return "tournament_response_shape"
    if message.startswith("active game") or message.startswith("active_games"):
        return "assignment_response_shape"
    return "unclassified"


def run_race(config: RaceConfig, *, observer: RaceRuntimeObserver | None = None,
             stop: StopSignal | None = None) -> RunResult:
    """Run one pre-registered participant through a complete tournament."""
    health = RaceHealth(observer.health) if observer is not None else None
    startup = health.register("startup") if health else None
    recorder: RaceRunRecorder | None = None
    from heuristic_ai.runtime.progress import RunProgress
    try:
        with health.bind(startup) if health and startup else nullcontext():
            if stop is not None and stop.is_set():
                raise KeyboardInterrupt
            try:
                recorder = RaceRunRecorder(
                    config, package_version=__version__, guide_version=None,
                )
            except Exception as error:
                if isinstance(error, (ValueError, KeyError)):
                    error.race_failure_category = "configuration"
                _report_failure(observer, error)
                raise
            if observer is not None:
                observer.run_started(recorder.run_id, recorder.work_dir)
                recorder.bind_health_progress(health.checkpoint)
            # The supervised parent owns presentation; the old progress thread
            # remains available to direct Python callers with no observer.
            with RunProgress(recorder, LOGGER) if observer is None else nullcontext():
                return _run_race(config, recorder, observer=observer, health=health,
                                 startup=startup, stop=stop)
    finally:
        if startup:
            startup.close()
        if recorder is not None:
            with health.component("recorder_close") if health else nullcontext():
                recorder.close()


def _run_race(config: RaceConfig, recorder: RaceRunRecorder, **options) -> RunResult:
    with RunNetwork(config.server, config.competition.output_dir) as network:
        network.bind_recorder(recorder)
        return _run_race_with_network(config, recorder, network, **options)


def _run_race_with_network(config: RaceConfig, recorder: RaceRunRecorder, network: RunNetwork, *,
              observer: RaceRuntimeObserver | None = None, health: RaceHealth | None = None,
              startup: HealthComponent | None = None, stop: StopSignal | None = None) -> RunResult:

    def client_factory(token: str) -> ApiClient:
        return network.client(
            ApiClient,
            token=token,
            insecure=config.server.insecure,
            timeout=config.server.timeout_seconds,
            **({"health_operation": health.operation} if health else {}),
        )

    control_clients: list[JsonClient] = []

    def preflight_client_factory(token: str) -> JsonClient:
        client = client_factory(token)
        control_clients.append(client)
        return client

    run_error: Exception | None = None
    outcome: RaceOutcome | None = None
    coordinator: RaceCoordinator | None = None
    try:
        participant, initial_snapshot = validate_race_participant(
            config,
            preflight_client_factory,
        )
        if len(control_clients) != 1:
            raise RuntimeError("race preflight constructed an invalid control client set")
        if stop is not None and stop.is_set():
            raise KeyboardInterrupt
        if observer is not None:
            with health.operation("identity_acknowledgement", 60.0) if health else nullcontext():
                observer.identity_validated(participant.user_id, participant.tournament_id)
        if stop is not None and stop.is_set():
            raise KeyboardInterrupt
        recorder.identify_participant(participant.user_id)
        network.bind_identity(config.participant.token, participant.user_id)
        if network.uses_managed_io:
            network.service.prepare((config.participant.token,))
        statistics = RaceStageStatistics(participant.rules.rounds,
            path=getattr(observer, "statistics_path", None) if observer is not None else None,
            warning=lambda reason: LOGGER.warning("阶段统计不完整：%s", reason))
        coordinator = RaceCoordinator(
            participant=participant,
            control_client=control_clients[0],
            client_factory=client_factory,
            recorder=recorder,
            seed=config.competition.seed,
            decision_budget_ms=config.competition.decision_budget_ms,
            poll_interval=config.competition.poll_interval_seconds,
            player_worker=play_game,
            request_timeout=config.server.timeout_seconds,
            statistics=statistics,
            **({"observer": observer, "health": health, "stop": stop}
               if observer is not None or stop is not None else {}),
        )
        if startup:
            startup.close()
        outcome = coordinator.run(initial_snapshot)
    except Exception as error:
        run_error = error
        if coordinator is None or not coordinator.failure_reported:
            _report_failure(observer, error)
    finally:
        if startup:
            startup.close()

    with health.component("finalization") if health else nullcontext():
        return _finalize_race(recorder, outcome, run_error, observer)


def _finalize_race(recorder: RaceRunRecorder, outcome: RaceOutcome | None,
                   run_error: Exception | None, observer: RaceRuntimeObserver | None) -> RunResult:

    if run_error is not None:
        LOGGER.error(
            "race run failed: %s failure_code=%s",
            _safe_error_type(run_error),
            _safe_failure_code(run_error),
        )
        return _record_failed_run(recorder, run_error)
    if outcome is None:
        missing_outcome = RuntimeError("race coordinator returned no outcome")
        _report_failure(observer, missing_outcome)
        LOGGER.error("race run failed: RuntimeError")
        return _record_failed_run(recorder, missing_outcome)

    if outcome.terminal_reason in {"finished", "eliminated"}:
        if outcome.result_unavailable:
            return _record_failed_run(
                recorder, RuntimeError("race finished with unavailable game results"),
                own_ranking=outcome.own_ranking,
            )
        completion_error: Exception | None = None
        try:
            return recorder.complete(
                outcome.terminal_reason,
                outcome.own_ranking,
            )
        except Exception as error:
            completion_error = error
            _report_failure(observer, error)
        LOGGER.error(
            "race artifact completion failed: %s",
            _safe_error_type(completion_error),
        )
        return _record_failed_run(recorder, completion_error)

    terminal_error = RuntimeError(
        "race tournament was voided"
        if outcome.terminal_reason == "void"
        else "race tournament was closed"
    )
    return _record_failed_run(
        recorder,
        terminal_error,
        terminal_reason=outcome.terminal_reason,
        own_ranking=outcome.own_ranking,
    )
