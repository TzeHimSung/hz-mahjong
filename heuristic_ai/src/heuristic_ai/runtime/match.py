"""Global-token automatic matchmaking, with one durable run per room."""
from __future__ import annotations

import copy
import logging
import math
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any

from heuristic_ai import __version__
from heuristic_ai.artifacts.models import ArtifactStatus, MatchWinStats, RunResult, RunStatus
from heuristic_ai.config.models import MatchConfig
from heuristic_ai.game.models import RuleConfig
from heuristic_ai.protocol.client import ApiClient, ApiError
from heuristic_ai.protocol.compatibility import normalize_rules, required_int
from heuristic_ai.strategy.models import GameContext
from heuristic_ai.strategy.one_ply import OnePlyController
from .bot import GameResult, GameStopped, JsonClient, play_game
from .progress import RunProgress
from .rate_limit import ParticipantStateRequestGate, StateLimitedClient
from .network import RunNetwork
from heuristic_ai.config.models import DEFAULT_MATCH_ACTION_RESERVES
from .match_statistics import MatchStatisticsReader, RoundEvidence, StatisticsCancelled
from .recovery import (
    LinkedStop, RecoveryGroup, RecorderCallbackFailure as _RecorderCallbackFailure,
    guard_callback, handle_start_failure, invoke_recoverable_worker, process_control,
)


LOGGER = logging.getLogger("heuristic_ai.match")
_STATUSES = frozenset({"registering", "running", "finished", "void", "closed"})
_RANKING_FIELDS = ("rank", "total_score", "place_points", "god_count", "games_played")


@dataclass(frozen=True, slots=True)
class MatchLoopResult:
    session_group_id: str
    session_count: int
    completed_sessions: int
    degraded_sessions: int
    last_result: RunResult | None
    stop_reason: str
    exit_code: int
    win_stats: MatchWinStats = field(default_factory=MatchWinStats)


@dataclass(slots=True)
class _LoopState:
    session_group_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_count: int = 0
    completed_sessions: int = 0
    degraded_sessions: int = 0
    last_result: RunResult | None = None
    win_stats: MatchWinStats = field(default_factory=MatchWinStats)

    def result(self, reason: str, code: int) -> MatchLoopResult:
        return MatchLoopResult(self.session_group_id, self.session_count, self.completed_sessions,
                               self.degraded_sessions, self.last_result, reason, code, self.win_stats)


class MatchFailure(RuntimeError):
    """A fixed diagnostic code, safe to expose without server response text."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _MatchInterrupted(KeyboardInterrupt):
    pass


def _check_stop(stop: threading.Event) -> None:
    if stop.is_set():
        raise _MatchInterrupted()


class _StopAttemptGate:
    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop

    def wait(self, method: str, path: str) -> None:
        _check_stop(self.stop)

    def wait_until(self, method: str, path: str, deadline_monotonic: float) -> None:
        _check_stop(self.stop)


def _object(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MatchFailure(code)
    return value


def _identifier(value: Any, code: str) -> str:
    if (not isinstance(value, str) or not value
            or any(character.isspace() or character in "/?#" for character in value)):
        raise MatchFailure(code)
    return value


def _validate_identity(value: Any, expected_user: str | None = None) -> str:
    identity = _object(value, "invalid_global_identity")
    user_id = _identifier(identity.get("user_id"), "invalid_global_identity")
    if identity.get("tournament_id") != "":
        raise MatchFailure("global_token_required")
    if expected_user is not None and user_id != expected_user:
        raise MatchFailure("global_identity_changed")
    active_games = identity.get("active_games")
    if not isinstance(active_games, list):
        raise MatchFailure("invalid_active_games")
    if active_games:
        raise MatchFailure("active_games_present")
    return user_id


@dataclass(frozen=True, slots=True)
class MatchAssignment:
    room_id: str
    round_no: int
    match_count: int
    rules: RuleConfig
    config: dict[str, Any] = field(repr=False)


def _binding(value: Any) -> tuple[str, int, dict[str, Any]]:
    response = _object(value, "invalid_match_response")
    room_id = _identifier(response.get("room_id"), "invalid_match_room")
    round_no = response.get("round_no")
    if type(round_no) is not int or round_no < 0:
        raise MatchFailure("invalid_match_round")
    raw_config = _object(response.get("config"), "invalid_match_config")
    return room_id, round_no, raw_config


def _assignment(value: Any, config: MatchConfig, previous_room: str | None) -> MatchAssignment:
    room_id, round_no, raw_config = _binding(value)
    if room_id == previous_room:
        raise MatchFailure("repeated_match_room")
    if raw_config.get("Kind") != "auto":
        raise MatchFailure("auto_room_required")
    count = required_int(raw_config, "M", minimum=1,
                         maximum=config.matchmaking.max_parallel_games)
    required_int(raw_config, "Rounds", minimum=1,
                 maximum=config.matchmaking.max_rounds_per_game)
    return MatchAssignment(room_id, round_no, count, normalize_rules(raw_config), copy.deepcopy(raw_config))


def _retry_delay(error: ApiError, default: float, wall_clock: Callable[[], float]) -> float:
    value = error.retry_after
    if not isinstance(value, str):
        return default
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = date.timestamp() - wall_clock()
        except (ValueError, TypeError, OverflowError):
            return default
    return max(default, seconds) if math.isfinite(seconds) else default


class _MatchEvents:
    """Small, redacted console events scoped to one matching session."""

    def __init__(self, recorder: Any, session: int) -> None:
        self.run_id = recorder.run_id
        self.session = session
        self.sanitize = recorder._serializer.sanitize
        self.room_id: str | None = None

    def emit(self, event: str, **details: object) -> None:
        if not LOGGER.isEnabledFor(logging.INFO):
            return
        fields: dict[str, object] = {"run": self.run_id, "session": self.session, "event": event}
        if self.room_id is not None:
            fields["room"] = self.room_id
        fields.update(details)
        LOGGER.info("%s", " ".join(f"{key}={self.sanitize(str(value))}" for key, value in fields.items()))


class _MatchAdmission:
    """Keep attempt spacing across sessions as well as explicit busy retries."""

    def __init__(self, client: JsonClient, wait: Callable[[float], None],
                 clock: Callable[[], float], wall_clock: Callable[[], float]) -> None:
        self.client, self.wait, self.clock, self.wall_clock = client, wait, clock, wall_clock
        self.last_attempt: float | None = None

    def request(self, config: MatchConfig, events: _MatchEvents) -> Any:
        next_allowed = self.clock()
        for attempt in range(4):
            if self.last_attempt is not None:
                next_allowed = max(next_allowed, self.last_attempt + 6.0)
            self.wait(max(0.0, next_allowed - self.clock()))
            events.emit("match_request", attempt=attempt + 1)
            self.last_attempt = self.clock()
            try:
                return self.client.request("POST", "/api/match", {
                    "M": config.matchmaking.max_parallel_games,
                    "Rounds": config.matchmaking.max_rounds_per_game,
                })
            except ApiError as error:
                if (error.code in {"MATCH_LIMIT_REACHED", "NO_ROOM_AVAILABLE"}
                        or (error.status != 429 and error.code != "MATCH_BUSY") or attempt == 3):
                    raise
                delay = _retry_delay(error, float(6 * 2**attempt), self.wall_clock)
                next_allowed = self.clock() + delay
                events.emit("match_retry", attempt=attempt + 1, retry_seconds=delay)
        raise AssertionError("unreachable matchmaking retry")


@dataclass(slots=True)
class _Worker:
    game_id: str
    future: Future[GameResult]


class MatchCoordinator:
    """Discover only the authenticated participant's games in one auto room."""

    def __init__(self, *, config: MatchConfig, assignment: MatchAssignment,
                 user_id: str, control_client: JsonClient,
                 worker_client_factory: Callable[[], JsonClient], recorder: Any,
                 events: _MatchEvents,
                 stop: threading.Event, state_gate: ParticipantStateRequestGate,
                 wait: Callable[[float], None], player_worker: Callable[..., GameResult] = play_game,

                 max_control_polls: int | None = None,
                 statistics_client_factory: Callable[[], JsonClient] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config, self.assignment, self.user_id = config, assignment, user_id
        self.control_client, self.worker_client_factory = control_client, worker_client_factory
        self.recorder, self.stop, self.state_gate, self.wait = recorder, stop, state_gate, wait
        self.player_worker = player_worker
        self.max_control_polls = max_control_polls
        self.events = events
        self.last_status: str | None = None
        self.clock = clock
        self.workers: dict[str, _Worker] = {}
        self.groups: dict[str, RecoveryGroup] = {}
        self.unavailable: set[str] = set()
        self.seen: set[str] = set()
        self.completed: set[str] = set()
        self.executor: ThreadPoolExecutor | None = None
        self.own_ranking: dict[str, int] | None = None
        self.room_finished = False
        self.terminal_executor: ThreadPoolExecutor | None = None
        self.terminal_reads: dict[str, Future[tuple[int, ...]]] = {}
        self.terminal_attempted: set[str] = set()
        self.evidence: dict[str, RoundEvidence] = {}
        self.statistics_stop = LinkedStop(stop)
        self.statistics_reader = (
            MatchStatisticsReader(room_id=assignment.room_id, rounds=assignment.rules.rounds,
                user_id=user_id, client_factory=statistics_client_factory,
                stop=self.statistics_stop, recorder=recorder)
            if statistics_client_factory is not None and callable(getattr(recorder, "record_round_outcome", None))
            else None
        )
        self.statistics_executor: ThreadPoolExecutor | None = None
        self.statistics_reads: dict[str, Future[None]] = {}

    def _check(self) -> None:
        self.recorder.ensure_audit_healthy()
        _check_stop(self.stop)

    def _record_call(self, callback: Callable[..., Any], *args: Any,
                     unwrap_callback: bool = True) -> Any:
        try:
            return guard_callback(callback, self.stop)(*args)
        except _RecorderCallbackFailure as error:
            if unwrap_callback:
                raise error.original from None
            raise

    def _start(self, game_id: str) -> None:
        if game_id in self.seen:
            return
        self._check()
        if len(self.seen) >= self.assignment.match_count:
            raise MatchFailure("too_many_match_games")
        self.recorder.register_game(game_id)
        if callable(getattr(self.recorder, "record_round_outcome", None)):
            self.evidence[game_id] = RoundEvidence(game_id, self.assignment.rules.rounds, self.recorder)
        self.seen.add(game_id)
        self.events.emit("game_discovered", game=game_id)
        self.groups[game_id] = RecoveryGroup(game_id, {self.user_id: self.user_id}, self.stop,
                                              self.recorder, clock=self.clock)
        self._launch(game_id)

    def _launch(self, game_id: str) -> None:
        group = self.groups[game_id]
        game = GameContext(game_id=game_id, player_name=self.user_id, user_id=self.user_id,
                           seed=self.config.competition.seed, rules=self.assignment.rules,
                           decision_budget_ms=self.config.competition.decision_budget_ms)
        controller = None
        try:
            controller = OnePlyController(game)
            client = StateLimitedClient(self.worker_client_factory(), self.state_gate, group.stop)
            self._check()
            assert self.executor is not None
            future = self.executor.submit(invoke_recoverable_worker, group, self.user_id,
                                          self.player_worker, client, game, controller, self.recorder,
                                          on_state_response=self.evidence[game_id].observe if game_id in self.evidence else None)
            self.workers[game_id] = _Worker(game_id, future)
        except BaseException as error:
            handle_start_failure(group, self.user_id, error, controller)

    def _restart_due(self) -> None:
        if self.room_finished:
            return
        for game_id, group in self.groups.items():
            if (group.pending and game_id not in self.workers
                    and self.clock() >= group.next_retry_at):
                self._check()
                group.begin_restart()
                self._launch(game_id)

    def _consume(self, game_id: str, *, unwrap_callback: bool = True) -> None:
        worker = self.workers[game_id]
        try:
            result = worker.future.result()
        except _RecorderCallbackFailure as error:
            if not unwrap_callback:
                raise
            raise error.original from None
        except BaseException as error:
            if not unwrap_callback or process_control(error) is not None:
                raise
            self._check()
            group = self.groups[game_id]
            group.note_failure(self.user_id, error)
            del self.workers[game_id]
            return
        if (not isinstance(result, GameResult) or result.game_id != game_id
                or not isinstance(result.scores, Sequence) or isinstance(result.scores, (str, bytes))
                or len(result.scores) != 4 or any(type(score) is not int for score in result.scores)
                or type(result.degraded) is not bool):
            raise MatchFailure("invalid_game_result")
        reason = None
        if result.degraded:
            reason = result.degradation_reason or "strategy_session_degraded"
            if (not isinstance(reason, str) or not reason.isascii() or not reason
                    or not reason[0].islower() or any(not (c.islower() or c.isdigit() or c == "_") for c in reason)):
                raise MatchFailure("invalid_degradation_reason")
        self._record_call(self.recorder.mark_game_completed, game_id, result.scores,
                          unwrap_callback=unwrap_callback)
        if reason is not None:
            self._record_call(self.recorder.mark_degraded, game_id, self.user_id, reason,
                              unwrap_callback=unwrap_callback)
        self._note_completed(game_id)
        del self.workers[game_id]
        self.groups[game_id].terminal("completed")

    def _note_completed(self, game_id: str) -> None:
        if game_id not in self.completed:
            self.completed.add(game_id)
            self.events.emit("game_completed", game=game_id,
                             completed=f"{len(self.completed)}/{self.assignment.match_count}")
            if (self.statistics_reader is not None and self.statistics_executor is not None
                    and not self.statistics_stop.is_set()):
                evidence = self.evidence.get(game_id)
                self.statistics_reads[game_id] = self.statistics_executor.submit(
                    self.statistics_reader.read, game_id, evidence.own_seat if evidence else None)

    def _reap(self) -> None:
        for game_id, worker in tuple(self.workers.items()):
            if worker.future.done():
                self._consume(game_id)
        self._reap_statistics()

    def _reap_statistics(self, *, draining: bool = False) -> None:
        for game_id, future in tuple(self.statistics_reads.items()):
            if not draining and not future.done():
                continue
            try:
                future.result()
            except (CancelledError, StatisticsCancelled):
                pass
            except _RecorderCallbackFailure as error:
                raise error.original from None
            except Exception:
                self.events.emit("statistics_unavailable", game=game_id, code="replay_result_unavailable")
            finally:
                del self.statistics_reads[game_id]

    def _drain(self) -> None:
        self.room_finished = True
        for group in self.groups.values():
            group.terminal("finished")
        while self.workers or self.terminal_reads or self.seen - self.completed - self.unavailable:
            self._check()
            self._reap()
            for game_id in self.seen - self.completed - self.unavailable - set(self.workers):
                if game_id not in self.terminal_attempted:
                    self.terminal_attempted.add(game_id)
                    assert self.terminal_executor is not None
                    self.terminal_reads[game_id] = self.terminal_executor.submit(self._read_terminal, game_id)
            self._reap_terminal()
            if self.workers or self.terminal_reads:
                self.wait(0.05)

    def _read_terminal(self, game_id: str) -> tuple[int, ...]:
        from .bot import _finished_scores
        delegate = self.worker_client_factory()
        client = StateLimitedClient(delegate, self.state_gate, LinkedStop(self.stop))
        path = f"/api/games/{game_id}/state?seq=0"
        if callable(getattr(delegate, "request_once", None)):
            response = client.request_once("GET", path,
                deadline_monotonic=time.monotonic() + self.config.server.timeout_seconds)
        else:
            response = client.request("GET", path)
        if not isinstance(response, dict) or response.get("finished") is not True:
            raise MatchFailure("terminal_game_not_finished")
        if game_id in self.evidence:
            guard_callback(self.evidence[game_id].observe, self.stop)(response, request_seq=0)
        return _finished_scores(response)

    def _reap_terminal(self, *, draining: bool = False) -> None:
        primary: BaseException | None = None
        for game_id, future in tuple(self.terminal_reads.items()):
            if not draining and not future.done():
                continue
            try:
                try:
                    scores = future.result()
                except BaseException as error:
                    if isinstance(error, _RecorderCallbackFailure):
                        raise error.original from None
                    if process_control(error) is not None:
                        raise
                    if not draining or not self.stop.is_set():
                        self.groups[game_id].note_failure(self.user_id, error, {"stage": "terminal_result"})
                        self._record_call(self.recorder.mark_game_aborted, game_id, "result_unavailable")
                        self.unavailable.add(game_id)
                else:
                    self._record_call(self.recorder.mark_game_completed, game_id, scores)
                    if self.groups[game_id].reported(self.user_id):
                        self._record_call(self.recorder.mark_degraded, game_id, self.user_id, "terminal_result_recovered")
                    self._note_completed(game_id)
            except BaseException as error:
                if primary is None or process_control(primary) is None and process_control(error) is not None:
                    primary = process_control(error) or error
            finally:
                del self.terminal_reads[game_id]
        if primary is not None:
            raise primary

    def _cleanup(self, reason: str) -> None:
        self.stop.set()
        self.statistics_stop.set()
        for future in self.statistics_reads.values():
            future.cancel()
        for group in self.groups.values():
            group.stop.set()
        for worker in self.workers.values():
            worker.future.cancel()
        error: BaseException | None = None
        aborted_here: set[str] = set()

        def remember(failure: BaseException) -> None:
            nonlocal error
            failure = failure.original if isinstance(failure, _RecorderCallbackFailure) else failure
            if error is None or process_control(error) is None and process_control(failure) is not None:
                error = process_control(failure) or failure

        for game_id in tuple(self.workers):
            try:
                self._consume(game_id, unwrap_callback=False)
            except (CancelledError, GameStopped, _MatchInterrupted):
                try:
                    if reason != "user_interrupt":
                        self.recorder.mark_game_aborted(game_id, reason)
                        aborted_here.add(game_id)
                except BaseException as failure:
                    remember(failure)
            except BaseException as failure:
                remember(failure)
                try:
                    if reason != "user_interrupt":
                        self.recorder.mark_game_aborted(game_id, reason)
                        aborted_here.add(game_id)
                except BaseException:
                    pass
        self.workers.clear()
        try:
            self._reap_terminal(draining=True)
        except BaseException as failure:
            remember(failure)
        for game_id in self.seen - self.completed - self.unavailable - aborted_here:
            if reason != "user_interrupt":
                try:
                    self.recorder.mark_game_aborted(game_id, reason)
                except BaseException as failure:
                    remember(failure)
        # A callback/storage error remains a failure even when shutdown raced it.
        if error is not None:
            raise error

    def _snapshot(self, value: Any) -> tuple[str, tuple[str, ...]]:
        room = _object(value, "invalid_match_room_response")
        if room.get("tournament_id") != self.assignment.room_id:
            raise MatchFailure("match_room_changed")
        if room.get("config") != self.assignment.config:
            raise MatchFailure("match_config_changed")
        status = room.get("status")
        if status not in _STATUSES:
            raise MatchFailure("invalid_match_room_status")
        games = room.get("my_games")
        if not isinstance(games, list):
            raise MatchFailure("invalid_match_games")
        identifiers = tuple(dict.fromkeys(_identifier(item, "invalid_match_game_id") for item in games))
        if len(self.seen.union(identifiers)) > self.assignment.match_count:
            raise MatchFailure("too_many_match_games")
        ranking = room.get("ranking")
        if ranking is not None:
            if not isinstance(ranking, list):
                raise MatchFailure("invalid_match_ranking")
            own = [item for item in ranking if isinstance(item, dict) and item.get("user_id") == self.user_id]
            if len(own) > 1:
                raise MatchFailure("invalid_match_ranking")
            if own:
                projected = {key: own[0][key] for key in _RANKING_FIELDS if key in own[0]}
                if any(type(value) is not int for value in projected.values()):
                    raise MatchFailure("invalid_match_ranking")
                self.own_ranking = projected
        return status, identifiers

    def run(self) -> Mapping[str, int] | None:
        self.executor = ThreadPoolExecutor(max_workers=self.assignment.match_count, thread_name_prefix="mahjong-match")
        self.terminal_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mahjong-match-final")
        if self.statistics_reader is not None:
            self.statistics_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mahjong-match-stats")
        polls = 0
        try:
            while True:
                self._check()
                self._reap()
                polls += 1
                if self.max_control_polls is not None and polls > self.max_control_polls:
                    raise MatchFailure("match_control_poll_limit")
                try:
                    response = self.control_client.request("GET", f"/api/tournaments/{self.assignment.room_id}")
                except ApiError as error:
                    if error.status == 404 and error.code == "TOURNAMENT_NOT_FOUND":
                        self._reap()
                        if len(self.completed) == self.assignment.match_count:
                            return self.own_ranking
                    raise
                self._check()
                status, games = self._snapshot(response)
                self.recorder.record_lifecycle(status)
                if status != self.last_status:
                    self.last_status = status
                    self.events.emit("room_status", status=status)
                if status in {"void", "closed"}:
                    raise MatchFailure(f"match_room_{status}")
                for game_id in games:
                    self._start(game_id)
                if status == "finished":
                    if len(self.seen) != self.assignment.match_count:
                        raise MatchFailure("missing_match_games")
                    self._drain()
                    if len(self.completed) != self.assignment.match_count:
                        raise MatchFailure("missing_match_results")
                    return self.own_ranking
                self._restart_due()
                interval = self.config.competition.poll_interval_seconds
                due = [group.next_retry_at for group in self.groups.values() if group.pending]
                if due:
                    interval = min(interval, max(0.05, min(due) - self.clock()))
                self.wait(interval)
        except BaseException as primary:
            reason = "user_interrupt" if isinstance(primary, KeyboardInterrupt) else "runtime_error"
            try:
                self._cleanup(reason)
            except BaseException as cleanup:
                if isinstance(primary, KeyboardInterrupt) and isinstance(cleanup, Exception):
                    raise cleanup from None
                if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
                    raise cleanup from None
            raise
        finally:
            primary = sys.exception()
            shutdown_error: BaseException | None = None
            for executor in (self.executor, self.terminal_executor, self.statistics_executor):
                try:
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=(
                            executor is not self.statistics_executor or primary is not None or self.stop.is_set()))
                except BaseException as error:
                    if shutdown_error is None or process_control(error) is not None:
                        shutdown_error = error
            self.executor = self.terminal_executor = None
            self.statistics_executor = None
            statistics_error: BaseException | None = None
            try:
                self._reap_statistics(draining=True)
            except BaseException as error:
                statistics_error = error
                if shutdown_error is None or process_control(error) is not None:
                    shutdown_error = error
            if statistics_error is not None:
                raise statistics_error
            if shutdown_error is not None:
                if primary is not None and process_control(primary) is not None:
                    pass
                elif process_control(shutdown_error) is not None:
                    raise process_control(shutdown_error)
                elif primary is None:
                    raise shutdown_error
                elif isinstance(primary, Exception) and isinstance(shutdown_error, Exception):
                    raise ExceptionGroup("match execution and executor shutdown failures", [primary, shutdown_error])


def _run_match(config: MatchConfig, *, action_reserves=DEFAULT_MATCH_ACTION_RESERVES, **options) -> MatchLoopResult:
    with RunNetwork(config.server, config.competition.output_dir, action_reserves=action_reserves) as network:
        return _run_match_with_network(config, network=network, **options)


def _run_match_with_network(config: MatchConfig, *, network: RunNetwork, global_token: str, _state: _LoopState,
              stop_event: threading.Event | None = None,
              client_factory: Callable[..., Any] = ApiClient,
              recorder_factory: Callable[..., Any] | None = None,
              player_worker: Callable[..., GameResult] = play_game,

              clock: Callable[[], float] = time.monotonic,
              wall_clock: Callable[[], float] = time.time,
              wait: Callable[[float], bool] | None = None,
              state_gate: ParticipantStateRequestGate | None = None,
              max_control_polls: int | None = None) -> MatchLoopResult:
    """Publish each session before admitting another; retain only its latest result."""
    if not isinstance(global_token, str) or not global_token.strip():
        raise ValueError("match requires global_token")
    if recorder_factory is None:
        from heuristic_ai.artifacts.match import MatchRunRecorder
        recorder_factory = MatchRunRecorder
    stop = stop_event if stop_event is not None else threading.Event()
    gate = state_gate if state_gate is not None else ParticipantStateRequestGate()
    wait_function = wait if wait is not None else stop.wait

    def pause(seconds: float) -> None:
        _check_stop(stop)
        if wait_function(seconds):
            raise _MatchInterrupted()
        _check_stop(stop)

    def make_client(*, token: str | None, max_attempts: int = 4, worker: bool = False) -> Any:
        return network.client(client_factory, token=token, insecure=config.server.insecure,
                              timeout=config.server.timeout_seconds, max_attempts=max_attempts,
                              sleep=pause, request_gate=None if worker else _StopAttemptGate(stop))

    group_id = _state.session_group_id
    sessions = completed = degraded = 0
    last_result: RunResult | None = None
    previous_room: str | None = None
    expected_user: str | None = None
    admission: _MatchAdmission | None = None
    stop_reason, exit_code = "failed", 1
    while True:
        # Between sessions Ctrl+C must not create a spurious second run.
        if stop.is_set() and last_result is not None:
            stop_reason, exit_code = "interrupted", 130
            break
        recorder = recorder_factory(config, global_token=global_token, package_version=__version__,
                                    session_group_id=group_id, session_index=sessions + 1, guide_version=None)
        network.bind_recorder(recorder)
        sessions += 1
        _state.session_count = sessions
        result: RunResult | None = None
        session_failure = False
        interrupted = False
        artifact_failed = False
        statistics_base_initialized = False
        try:
            set_base = getattr(recorder, "set_statistics_base", None)
            if callable(set_base):
                set_base(_state.win_stats)
            statistics_base_initialized = True
            with RunProgress(recorder, LOGGER):
                try:
                    events = _MatchEvents(recorder, sessions)
                    recorder.bind_stop(stop)
                    recorder.ensure_audit_healthy()
                    _check_stop(stop)
                    events.emit("preflight")
                    control = make_client(token=global_token)
                    user_id = _validate_identity(control.request("GET", "/api/me"), expected_user)
                    expected_user = user_id
                    recorder.identify_participant(user_id)
                    network.bind_identity(global_token, user_id)
                    if network.uses_managed_io:
                        network.service.prepare((global_token,))
                    recorder.ensure_audit_healthy()
                    if admission is None:
                        admission = _MatchAdmission(make_client(token=global_token, max_attempts=1), pause, clock, wall_clock)
                    match_response = admission.request(config, events)
                    room_id, round_no, raw_config = _binding(match_response)
                    recorder.bind_match(room_id, match_config=raw_config, round_no=round_no)
                    assignment = _assignment(match_response, config, previous_room)
                    previous_room = assignment.room_id
                    events.room_id = assignment.room_id
                    events.emit("matched", games=assignment.match_count, rounds_per_game=assignment.rules.rounds)
                    coordinator = MatchCoordinator(
                        config=config, assignment=assignment, user_id=user_id, control_client=control,
                        worker_client_factory=lambda: make_client(token=global_token, worker=True), recorder=recorder,
                        events=events, stop=stop, state_gate=gate, wait=pause, player_worker=player_worker,
                        max_control_polls=max_control_polls,
                        statistics_client_factory=lambda: make_client(token=None, worker=True),
                        clock=clock,
                    )
                    ranking = coordinator.run()
                    recorder.ensure_audit_healthy()
                    _check_stop(stop)
                    try:
                        result = recorder.complete(participant_ranking=ranking)
                    except Exception:
                        artifact_failed = True
                        raise
                except KeyboardInterrupt:
                    stop.set()
                    interrupted = True
                    try:
                        result = recorder.interrupt()
                    except Exception:
                        session_failure = True
                        artifact_failed = True
                        LOGGER.error("match interruption artifact finalization failed")
                except Exception as error:
                    session_failure = True
                    LOGGER.error("match session failed")
                    try:
                        result = recorder.fail(error, terminal_reason=error.code if isinstance(error, MatchFailure) else "runtime_error")
                    except Exception:
                        artifact_failed = True
                        LOGGER.error("match failure artifact finalization failed")
        except Exception as error:
            session_failure = True
            try:
                result = recorder.fail(error, terminal_reason="runtime_error")
            except Exception:
                artifact_failed = True
                LOGGER.error("match initialization artifact finalization failed")
        finally:
            if result is not None:
                _state.last_result = result
            read_stats = getattr(recorder, "win_stats", None)
            if statistics_base_initialized and callable(read_stats):
                try:
                    _state.win_stats = read_stats()
                except Exception:
                    session_failure = True
                    artifact_failed = True
                    LOGGER.error("match statistics snapshot failed")
            try:
                recorder.close()
            except Exception:
                session_failure = True
                artifact_failed = True
                LOGGER.error("match recorder close failed")
        if result is not None:
            last_result = result
            LOGGER.info("session=%d run=%s status=%s artifacts=%s output=%s", sessions, result.run_id,
                        result.status, result.artifact_status, result.artifact_dir)
            if result.status in {RunStatus.COMPLETED, RunStatus.DEGRADED}:
                completed += 1
                degraded += result.status is RunStatus.DEGRADED
                _state.completed_sessions, _state.degraded_sessions = completed, degraded
        interrupted = interrupted or (stop.is_set() and not session_failure and result is not None
                                      and result.status in {RunStatus.COMPLETED, RunStatus.DEGRADED})
        if artifact_failed or result is not None and result.artifact_status is ArtifactStatus.INCOMPLETE:
            stop_reason, exit_code = "interrupted" if interrupted else "failed", 3
            break
        if result is not None and result.status is RunStatus.FAILED:
            stop_reason, exit_code = "failed", 1
            break
        if interrupted:
            stop_reason, exit_code = "interrupted", 130
            break
        if session_failure or result is None or result.status not in {RunStatus.COMPLETED, RunStatus.DEGRADED}:
            stop_reason, exit_code = "failed", 1
            break
        if not config.continuous:
            stop_reason, exit_code = "once_completed", 2 if degraded else 0
            break
    return MatchLoopResult(group_id, sessions, completed, degraded, last_result, stop_reason, exit_code, _state.win_stats)


def run_match(config: MatchConfig, *, global_token: str,
              stop_event: threading.Event | None = None, **kwargs: Any) -> MatchLoopResult:
    """Stop local scheduling on SIGINT while preserving publication and loop totals.

    In-flight HTTP calls finish within the configured transport timeout. Signal
    ownership is temporary and restricted to the main thread; embedding callers
    and tests can provide an Event without modifying process signal handlers.
    """
    if not isinstance(global_token, str) or not global_token.strip():
        raise ValueError("match requires global_token")
    stop = stop_event if stop_event is not None else threading.Event()
    state = _LoopState()
    previous_handler: Any = None
    installed = threading.current_thread() is threading.main_thread()
    if installed:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    try:
        return _run_match(config, global_token=global_token, stop_event=stop, _state=state, **kwargs)
    except KeyboardInterrupt:
        stop.set()
        return state.result("interrupted", 130)
    except Exception:
        LOGGER.error("match loop setup or publication failed")
        return state.result("failed", 1)
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous_handler)
