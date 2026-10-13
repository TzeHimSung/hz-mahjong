"""Training validation and four-participant match orchestration."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from heuristic_ai import __version__
from heuristic_ai.artifacts.models import RunResult
from heuristic_ai.artifacts.replay import ReplayArchiver
from heuristic_ai.artifacts.train import RunRecorder
from heuristic_ai.config.models import PlayerConfig, TrainConfig
from heuristic_ai.game.models import RuleConfig
from heuristic_ai.protocol.client import ApiClient
from heuristic_ai.protocol.compatibility import (
    normalize_rules,
    required_int,
)
from heuristic_ai.strategy.models import GameContext, policy_metadata
from heuristic_ai.strategy.one_ply import OnePlyController
from .bot import GameResult, play_game
from .progress import RunProgress
from .rate_limit import ParticipantStateRequestGate, StateLimitedClient
from .network import RunNetwork
from .recovery import (
    RecoveryGroup, RecorderCallbackFailure, guard_callback, handle_start_failure,
    invoke_recoverable_worker, process_control, recording_failure,
)


LOGGER = logging.getLogger("heuristic_ai.runner")


@dataclass(frozen=True, slots=True)
class ValidatedPlayer:
    name: str
    token: str = field(repr=False)
    user_id: str
    tournament_id: str
    match_count: int
    rules: RuleConfig
    config: dict[str, Any]


class JsonClient(Protocol):
    def request(self, method: str, path: str, body: Any = None) -> Any:
        raise NotImplementedError


class Recorder(Protocol):
    def record_decision(self, trace: Any) -> None:
        raise NotImplementedError

    def record_state_poll(self, event: Any) -> None:
        raise NotImplementedError

    def mark_degraded(
        self,
        game_id: str,
        player_name: str,
        reason: str,
    ) -> None:
        raise NotImplementedError

    def register_games(
        self,
        game_ids: tuple[str, ...],
        *,
        rounds_per_game: int,
    ) -> None:
        raise NotImplementedError

    def identify_player(
        self,
        game_id: str | None,
        player_name: str,
        user_id: str,
        seat: int | None = None,
    ) -> None:
        raise NotImplementedError


ClientFactory = Callable[[str], JsonClient]


PlayerWorker = Callable[..., GameResult]


def _sanitize_error(error: BaseException, config: TrainConfig) -> str:
    message = str(error)
    for player in config.players:
        message = message.replace(player.token, "<redacted>")
    return message


def validate_players(
    players: tuple[PlayerConfig, ...],
    client_factory: ClientFactory,
    public_client: JsonClient,
    room_id: str,
) -> tuple[ValidatedPlayer, ...]:
    """Complete every read-only contract check before any player becomes ready."""

    room = public_client.request("GET", f"/api/test-rooms/{room_id}/games")
    if not isinstance(room, dict):
        raise RuntimeError("test-room response must be an object")
    if room.get("room_id") != room_id:
        raise RuntimeError("test-room response returned a different room id")
    if room.get("status") not in {"registering", "finished"}:
        raise RuntimeError(f"test room is not ready to start: {room.get('status')}")

    validated: list[ValidatedPlayer] = []
    for player in players:
        client = client_factory(player.token)
        me = client.request("GET", "/api/me")
        rules_response = client.request("GET", "/api/tournaments/me/rules")
        if not isinstance(me, dict) or not isinstance(rules_response, dict):
            raise RuntimeError("player identity and rules responses must be objects")
        if me.get("tournament_id") != rules_response.get("tournament_id"):
            raise RuntimeError("token scope mismatch between identity and rules")
        raw_config = rules_response.get("config")
        if not isinstance(raw_config, dict):
            raise RuntimeError("tournament rules must contain a config object")
        if raw_config.get("Kind") != "test":
            raise RuntimeError("train tournament config Kind must be test")
        user_id = me.get("user_id")
        tournament_id = me.get("tournament_id")
        if not isinstance(user_id, str) or not user_id:
            raise RuntimeError("player identity is missing user_id")
        if not isinstance(tournament_id, str) or not tournament_id:
            raise RuntimeError("participant token is not scoped to a tournament")
        validated.append(
            ValidatedPlayer(
                name=player.name,
                token=player.token,
                user_id=user_id,
                tournament_id=tournament_id,
                match_count=required_int(
                    raw_config,
                    "M",
                    minimum=1,
                    maximum=16,
                ),

                rules=normalize_rules(raw_config),
                config=dict(raw_config),
            )
        )

    if len({player.user_id for player in validated}) != 4:
        raise RuntimeError("players must map to four unique users")
    tournament_ids = {player.tournament_id for player in validated}
    if len(tournament_ids) != 1:
        raise RuntimeError("players must share one tournament")
    if next(iter(tournament_ids)) != room_id:
        raise RuntimeError("participant tournament does not match room id")
    if any(player.config != validated[0].config for player in validated[1:]):
        raise RuntimeError("players returned different tournament configs")
    return tuple(validated)


class Coordinator:
    """Ready four participants and run one supervised strategy session per player."""

    def __init__(
        self,
        *,
        players: tuple[ValidatedPlayer, ...],
        client_factory: ClientFactory,
        archiver: Any,
        recorder: Recorder,
        seed: int,
        decision_budget_ms: int,
        poll_interval: float = 1.0,
        max_start_polls: int = 7200,
        player_worker: PlayerWorker = play_game,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger = LOGGER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if len(players) != 4:
            raise ValueError("coordinator requires exactly four players")
        if decision_budget_ms < 1:
            raise ValueError("decision_budget_ms must be positive")
        if poll_interval < 0:
            raise ValueError("poll_interval cannot be negative")
        if max_start_polls < 1:
            raise ValueError("max_start_polls must be at least one")
        self.players = players
        self.client_factory = client_factory
        self.archiver = archiver
        self.recorder = recorder
        self.seed = seed
        self.decision_budget_ms = decision_budget_ms
        self.poll_interval = poll_interval
        self.max_start_polls = max_start_polls
        self.player_worker = player_worker
        self.sleep = sleep
        self.logger = logger
        self.clock = clock
        self.state_request_gates = {
            player.user_id: ParticipantStateRequestGate() for player in players
        }

    def run(self) -> None:
        stop = threading.Event()
        bind_stop = getattr(self.recorder, "bind_stop", None)
        if bind_stop is not None:
            bind_stop(stop)
        archive_callback_errors: list[BaseException] = []

        def archive_guard(callback: Callable[..., Any]) -> Callable[..., Any]:
            guarded = guard_callback(callback, stop)

            def invoke(*args: Any, **kwargs: Any) -> Any:
                try:
                    return guarded(*args, **kwargs)
                except BaseException as error:
                    archive_callback_errors.append(error.original if isinstance(error, RecorderCallbackFailure) else error)
                    raise
            return invoke

        for name in ("on_round_bundle", "on_archive_failure", "on_round_published"):
            callback = getattr(self.archiver, name, None)
            if callable(callback):
                setattr(self.archiver, name, archive_guard(callback))
        clients = [self.client_factory(player.token) for player in self.players]
        primary_error: BaseException | None = None
        unavailable: set[str] = set()
        self.archiver.start()

        try:
            for player, client in zip(self.players, clients, strict=True):
                ensure_audit = getattr(self.recorder, "ensure_audit_healthy", None)
                if ensure_audit is not None:
                    ensure_audit()
                client.request("POST", "/api/tournaments/me/ready")
                self.logger.info("player=%s user=%s ready", player.name, player.user_id)

            tournament_id = self.players[0].tournament_id
            self._wait_until_running(clients[0], tournament_id)
            game_ids = self._wait_for_common_games(clients)
            self.logger.info(
                "tournament=%s games=%s started",
                tournament_id,
                ",".join(game_ids),
            )
            self.recorder.register_games(
                game_ids,
                rounds_per_game=self.players[0].rules.rounds,
            )
            for game_id in game_ids:
                self.archiver.submit(game_id)

            unavailable = self._run_games(game_ids, stop)
        except BaseException as error:
            primary_error = error
        finally:
            stop.set()
            try:
                if primary_error is None:
                    self.archiver.close_and_wait()
                else:
                    self.archiver.cancel_and_wait()
            except BaseException as archive_error:
                if type(primary_error) in {KeyboardInterrupt, SystemExit}:
                    pass
                elif type(archive_error) in {KeyboardInterrupt, SystemExit}:
                    primary_error = archive_error
                elif primary_error is None:
                    primary_error = archive_error
                else:
                    primary_error = ExceptionGroup(
                        "game and archive failures",
                        [primary_error, archive_error],
                    )

        for error in archive_callback_errors:
            if primary_error is None or process_control(primary_error) is None:
                primary_error = process_control(error) or error
        if primary_error is not None:
            raise primary_error
        if unavailable:
            raise RuntimeError("training games ended without results: " + ",".join(sorted(unavailable)))

    def _run_games(self, game_ids: tuple[str, ...], stop: threading.Event) -> set[str]:
        groups = {game_id: RecoveryGroup(
            game_id, {player.user_id: player.name for player in self.players},
            stop, self.recorder, clock=self.clock,
        ) for game_id in game_ids}
        completed: dict[str, set[str]] = {game_id: set() for game_id in game_ids}
        remaining = set(game_ids)
        unavailable: set[str] = set()
        futures: dict[Any, tuple[str, ValidatedPlayer]] = {}
        executor = ThreadPoolExecutor(max_workers=4 * len(game_ids), thread_name_prefix="mahjong-player")

        def launch(group: RecoveryGroup) -> None:
            completed[group.game_id].clear()
            for player in self.players:
                if group.stop.is_set():
                    break
                controller = None
                try:
                    client = StateLimitedClient(self.client_factory(player.token),
                                                self.state_request_gates[player.user_id], group.stop)
                    game = GameContext(game_id=group.game_id, player_name=player.name,
                                       user_id=player.user_id, seed=self.seed, rules=player.rules,
                                       decision_budget_ms=self.decision_budget_ms)
                    controller = OnePlyController(game)
                    future = executor.submit(invoke_recoverable_worker, group, player.user_id,
                                             self.player_worker, client, game, controller, self.recorder,
                                             on_state_poll=getattr(self.recorder, "record_state_poll", None))
                    futures[future] = (group.game_id, player)
                except BaseException as error:
                    handle_start_failure(group, player.user_id, error, controller)
                    break

        primary: BaseException | None = None
        try:
            for group in groups.values():
                launch(group)
            while remaining:
                for future in tuple(futures):
                    if not future.done():
                        continue
                    game_id, player = futures.pop(future)
                    group = groups[game_id]
                    try:
                        result = future.result()
                    except RecorderCallbackFailure as error:
                        raise error.original from None
                    except BaseException as error:
                        group.note_failure(player.user_id, error)
                    else:
                        completed[game_id].add(player.user_id)
                        if result.degraded:
                            guard_callback(self.recorder.mark_degraded, stop)(
                                game_id, player.name, result.degradation_reason or "strategy_session_degraded")
                ensure_audit = getattr(self.recorder, "ensure_audit_healthy", None)
                if ensure_audit is not None:
                    ensure_audit()
                if stop.is_set():
                    # A process-control error may still be finishing in a worker;
                    # the finally block drains it and preserves its precedence.
                    raise RuntimeError("training run was stopped")
                busy = {game_id for game_id, _ in futures.values()}
                for game_id in tuple(remaining):
                    group = groups[game_id]
                    status_for = getattr(self.archiver, "status_for", None)
                    status = status_for(game_id) if status_for is not None else None
                    if status in {"abandoned", "void", "closed"} or status == "finished" and group.pending:
                        group.terminal(status)
                        if status != "finished":
                            unavailable.add(game_id)
                    # Healthy players must observe their own terminal snapshot;
                    # the archive listing only retires an already-failed group.
                    if game_id in busy:
                        continue
                    if group.terminal_reason is not None:
                        remaining.remove(game_id)
                    elif group.pending:
                        if self.clock() >= group.next_retry_at:
                            group.begin_restart()
                            launch(group)
                    elif len(completed[game_id]) == 4:
                        group.terminal("completed")
                        remaining.remove(game_id)
                if remaining:
                    if futures:
                        wait(tuple(futures), timeout=0.05, return_when=FIRST_COMPLETED)
                    elif self.sleep is time.sleep:
                        stop.wait(0.05)
                    else:
                        self.sleep(0.05)
        except RecorderCallbackFailure as error:
            primary = error.original
        except BaseException as error:
            primary = error
        finally:
            if primary is not None:
                stop.set()
            for future in futures:
                try:
                    future.result()
                except BaseException as error:
                    control = process_control(error)
                    if control is not None and (primary is None or process_control(primary) is None):
                        primary = control
                    elif recording_failure(error) is not None and (primary is None or process_control(primary) is None):
                        primary = recording_failure(error).original
            try:
                executor.shutdown(wait=True)
            except BaseException as error:
                if primary is None or process_control(primary) is None:
                    primary = process_control(error) or error
        if primary is not None:
            raise primary
        return unavailable

    def _wait_until_running(self, client: JsonClient, tournament_id: str) -> None:
        for _ in range(self.max_start_polls):
            tournament = client.request("GET", f"/api/tournaments/{tournament_id}")
            status = tournament.get("status")
            if status == "running":
                return
            if status in {"finished", "void", "closed"}:
                raise RuntimeError(f"tournament terminated before start: {status}")
            self.sleep(self.poll_interval)
        raise TimeoutError("tournament did not start within the configured poll budget")

    def _wait_for_common_games(
        self,
        clients: list[JsonClient],
    ) -> tuple[str, ...]:
        expected_count = self.players[0].match_count
        for _ in range(self.max_start_polls):
            active_sets: list[set[str]] = []
            complete = True
            for client in clients:
                response = client.request("GET", "/api/me")
                if not isinstance(response, dict):
                    raise RuntimeError("player identity response must be an object")
                active = response.get("active_games")
                if active is None:
                    active = []
                if not isinstance(active, list):
                    raise RuntimeError("active_games must be an array")
                identifiers = [
                    item.get("game_id") if isinstance(item, dict) else None
                    for item in active
                ]
                if not all(
                    isinstance(game_id, str) and game_id
                    for game_id in identifiers
                ):
                    raise RuntimeError("active game is missing game_id")
                if len(set(identifiers)) != len(identifiers):
                    raise RuntimeError("active games contain duplicate game ids")
                if len(identifiers) > expected_count:
                    raise RuntimeError("active games exceed tournament config M")
                complete = complete and len(identifiers) == expected_count
                active_sets.append(set(identifiers))
            if complete and all(
                active == active_sets[0] for active in active_sets[1:]
            ):
                return tuple(sorted(active_sets[0]))
            tournament = clients[0].request(
                "GET",
                f"/api/tournaments/{self.players[0].tournament_id}",
            )
            if not isinstance(tournament, dict):
                raise RuntimeError("tournament response must be an object")
            status = tournament.get("status")
            if status in {"finished", "void", "closed"}:
                raise RuntimeError(
                    f"tournament terminated before games were assigned: {status}"
                )
            self.sleep(self.poll_interval)
        raise TimeoutError("players did not receive the same games within the poll budget")


def run_train(config: TrainConfig) -> RunResult:
    """Run one configured four-player competition and return its bundle status."""

    recorder = RunRecorder(
        config,
        package_version=__version__,
    )
    try:
        with RunProgress(recorder, LOGGER):
            return _run_train(config, recorder)
    finally:
        recorder.close()


def _run_train(config: TrainConfig, recorder: RunRecorder) -> RunResult:
    with RunNetwork(config.server, config.competition.output_dir) as network:
        network.bind_recorder(recorder)
        return _run_train_with_network(config, recorder, network)


def _run_train_with_network(config: TrainConfig, recorder: RunRecorder, network: RunNetwork) -> RunResult:

    def client_factory(token: str) -> ApiClient:
        return network.client(
            ApiClient,
            token=token,
            insecure=config.server.insecure,
            timeout=config.server.timeout_seconds,
        )

    public_client = network.client(
        ApiClient,
        insecure=config.server.insecure,
        timeout=config.server.timeout_seconds,
    )
    try:
        players = validate_players(
            config.players,
            client_factory,
            public_client,
            config.competition.room_id,
        )
        for player in players:
            network.bind_identity(player.token, player.user_id, player.name)
            recorder.identify_player(None, player.name, player.user_id)
        if network.uses_managed_io:
            network.service.prepare(player.token for player in players)
        replay_players = []
        for player in players:
            replay_players.append(
                {
                    "name": player.name,
                    "user_id": player.user_id,
                    "strategy": policy_metadata(),
                }
            )
        archiver = ReplayArchiver(
            public_client,
            room_id=config.competition.room_id,
            output_root=recorder.rounds_dir,
            players=replay_players,
            config=players[0].config,
            seed=config.competition.seed,
            run_id=recorder.run_id,
            poll_interval=config.competition.poll_interval_seconds,
            on_round_bundle=recorder.record_round_bundle,
            on_archive_failure=recorder.record_archive_failure,
        )
        Coordinator(
            players=players,
            client_factory=client_factory,
            archiver=archiver,
            recorder=recorder,
            seed=config.competition.seed,
            decision_budget_ms=config.competition.decision_budget_ms,
            poll_interval=config.competition.poll_interval_seconds,
        ).run()
        return recorder.complete()
    except Exception as error:
        LOGGER.error("run failed: %s", _sanitize_error(error, config))
        return recorder.fail(error)
