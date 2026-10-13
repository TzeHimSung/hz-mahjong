"""Pure legacy-format projections shared by live recording and recovery."""
from __future__ import annotations

import copy
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from heuristic_ai.artifacts.models import DecisionTrace, MatchWinStats, RunStatus
from heuristic_ai.strategy.models import policy_metadata
from heuristic_ai.artifacts.recovery import positive_generation, validate_recovery_event

_STABLE_REASON = re.compile(r"^[a-z][a-z0-9_]*$")
_RANKING_FIELDS = ("rank", "total_score", "place_points", "god_count", "games_played")
TRAIN_OPERATIONS = frozenset({"record_guide_version", "register_games", "identify_player", "mark_degraded", "record_game_recovery", "complete", "fail"})
RACE_OPERATIONS = frozenset({"record_guide_version", "identify_participant", "record_lifecycle", "register_game", "identify_player", "mark_game_completed", "mark_game_aborted", "mark_degraded", "record_game_recovery", "complete", "fail"})
MATCH_OPERATIONS = RACE_OPERATIONS | {"bind_match", "interrupt", "set_statistics_base",
                                      "record_round_progress", "record_round_outcome"}
TERMINAL_OPERATIONS = frozenset({"complete", "fail", "interrupt"})


def validate_decision(payload: Mapping[str, Any]) -> None:
    schema = payload.get("schema_version")
    if schema not in {3, 4}:
        raise ValueError("decision must use schema 3 or 4")
    if schema == 4:
        if payload.get("strategy") != policy_metadata():
            raise ValueError("decision must identify the fixed One-Ply Expectation policy")
        source = payload.get("execution_source")
        if source not in {"completed_search", "prefetch_hit", "forced_action", "static_fallback", "legal_fallback"}:
            raise ValueError("invalid One-Ply Expectation execution source")
        if payload.get("fallback_used") != (source in {"static_fallback", "legal_fallback"}):
            raise ValueError("One-Ply Expectation execution source disagrees with fallback status")
        if "configured_strategy" in payload or "effective_strategy" in payload:
            raise ValueError("decision schema 4 does not contain strategy selection fields")
    if "generation" in payload:
        positive_generation(payload["generation"])
    for key in ("game_id", "user_id", "player_name"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError("decision identity is incomplete")
    for key in ("round_no", "server_seq"):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError("decision sequence is invalid")
    if type(payload.get("seat")) is not int or payload["seat"] not in range(4):
        raise ValueError("decision seat is invalid")
    if payload.get("submission") not in {"accepted", "conflict", "error"}:
        raise ValueError("decision submission is invalid")
    if type(payload.get("fallback_used")) is not bool:
        raise ValueError("decision fallback flag is invalid")
    if not isinstance(payload.get("candidates"), list) or not isinstance(payload.get("decision"), dict):
        raise ValueError("decision payload is incomplete")
    if not isinstance(payload["decision"].get("details"), Mapping):
        raise ValueError("decision details are invalid")


class Projection:
    operations = TRAIN_OPERATIONS

    def __init__(self, manifest: dict[str, Any], rounds_per_game: int | None = None,
                 now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
                 *, public: bool = False) -> None:
        self._manifest = manifest
        self._rounds_per_game = rounds_per_game
        self._now = now
        self.public = public
        self.run_id = manifest["run_id"]
        self._lock = threading.RLock()
        self.revision = 0
        self._inventory_decisions = {
            (row["game_id"], row["user_id"], row["round_no"]): row
            for row in manifest.get("inventory_decision_counts", [])
        }

    def clone(self) -> Projection:
        result = type(self)(copy.deepcopy(self._manifest), self._rounds_per_game, self._now, public=self.public)
        result.revision = self.revision
        return result

    def _changed(self) -> None:
        self.revision += 1

    def control(self, operation: str, arguments: dict[str, Any], recorded_at: str) -> None:
        if operation not in self.operations:
            raise ValueError("unsupported artifact control operation")
        self._now = lambda: datetime.fromisoformat(recorded_at)
        getattr(self, operation)(**arguments)

    def record_poll(self, payload: dict[str, Any]) -> None:
        if isinstance(self, ParticipantProjection):
            raise ValueError(f"{self._manifest['mode']} artifacts cannot contain state polls")
        if payload.get("schema_version") != 1:
            raise ValueError("invalid state poll schema")
        if "generation" in payload:
            positive_generation(payload["generation"])
        self._manifest["state_poll_log"]["entries"] += 1

    def record_game_recovery(self, event: Mapping[str, Any]) -> None:
        validate_recovery_event(event)
        if self._manifest["status"] != "running":
            raise RuntimeError("cannot record recovery after run completion")
        if event["run_id"] != self.run_id:
            raise ValueError("recovery event references another run")
        game = next((game for game in self._manifest["games"] if game["game_id"] == event["game_id"]), None)
        if game is None:
            raise ValueError("recovery event references an unknown game")
        user_id, player_name = event.get("user_id"), event.get("player_name")
        if isinstance(self, ParticipantProjection):
            participant_id = self._manifest["participant"]["user_id"]
            if participant_id is None or any(value is not None and value != participant_id for value in (user_id, player_name)):
                raise ValueError("recovery event references an unknown participant")
        elif user_id is not None or player_name is not None:
            matches = [player for player in game["players"]
                       if (player_name is None or player["name"] == player_name)
                       and (user_id is None or player["user_id"] == user_id)]
            if len(matches) != 1:
                raise ValueError("recovery event references an unknown player")
        if event.get("stage_no") is not None and event["stage_no"] != game.get("stage_no"):
            raise ValueError("recovery event references another stage")
        previous = game.get("recovery", {})
        # Keep one bounded row per game; late errors from cancelled workers are
        # retained in the log without reviving the terminal recovery state.
        terminal = previous.get("state") == "terminal"
        scheduled = (previous.get("state") == "restart_scheduled" and event["event"] == "failure"
                     and event["generation"] <= previous["generation"])
        summary = {
            "generation": max(event["generation"], previous.get("generation", 1)),
            "restart_count": max(event["restart_count"], previous.get("restart_count", 0)),
            "state": "terminal" if terminal else "restart_scheduled" if scheduled else event["event"],
            "last_event": event["event"],
            "updated_at": event["recorded_at"],
            "failure_count": previous.get("failure_count", 0) + (event["event"] == "failure"),
            "restart_attempted": previous.get("restart_attempted", False) or event["event"] in {"restarted", "recovered"},
        }
        if summary["state"] == "restart_scheduled":
            for field in ("next_retry_at", "retry_delay_seconds"):
                value = previous.get(field) if scheduled else event.get(field)
                if value is not None:
                    summary[field] = value
        reason = previous.get("reason") if terminal else event.get("reason")
        if reason is not None:
            summary["reason"] = reason[:256]
        game["recovery"] = summary
        checkpoint = self._manifest.setdefault("recovery_log", {"schema_version": 1, "path": "recovery.jsonl", "entries": 0})
        checkpoint["entries"] += 1
        self._changed()

    def _completion_status(self) -> str:
        recovered = any(game.get("recovery", {}).get("restart_attempted", False) for game in self._manifest["games"])
        return "degraded" if self._manifest["degradations"] or recovered else "completed"


class TrainProjection(Projection):
    def record_guide_version(self, version: int | None) -> None:
        if version is not None and type(version) is not int:
            raise ValueError("observed guide version must be an integer or None")
        with self._lock:
            if self._manifest["status"] != RunStatus.RUNNING.value:
                raise RuntimeError("cannot record guide version after completion")
            self._manifest["guide_version"] = version
            self._changed()

    @staticmethod
    def _game_player(player: dict[str, Any]) -> dict[str, Any]:
        strategy = player["strategy"]
        return {
            "name": player["name"],
            "user_id": player["user_id"],
            "server_name": None,
            "seat": None,
            "strategy": {key: strategy[key] for key in ("mode", "name", "version") if key in strategy},
        }

    def register_games(
        self,
        game_ids: tuple[str, ...],
        *,
        rounds_per_game: int,
    ) -> None:
        with self._lock:
            if self._manifest["status"] != RunStatus.RUNNING.value:
                raise RuntimeError("cannot register games after run completion")
            if (
                isinstance(rounds_per_game, bool)
                or not isinstance(rounds_per_game, int)
                or rounds_per_game < 1
            ):
                raise ValueError("rounds_per_game must be a positive integer")
            if (
                not game_ids
                or any(not isinstance(game_id, str) or not game_id for game_id in game_ids)
                or len(set(game_ids)) != len(game_ids)
            ):
                raise ValueError("game_ids must contain unique non-empty strings")
            normalized = sorted(game_ids)
            if self._manifest["game_ids"]:
                if (
                    self._manifest["game_ids"] == normalized
                    and self._rounds_per_game == rounds_per_game
                ):
                    return
                raise ValueError("games were already registered for this run")
            self._rounds_per_game = rounds_per_game
            self._manifest["game_ids"] = normalized
            self._manifest["games"] = [
                {
                    "game_id": game_id,
                    "batch": None,
                    "players": [
                        self._game_player(player)
                        for player in self._manifest["players"]
                    ],
                    "rounds": [],
                }
                for game_id in normalized
            ]
            self._changed()

    def identify_player(
        self,
        game_id: str | None,
        player_name: str,
        user_id: str,
        seat: int | None = None,
    ) -> None:
        with self._lock:
            changes = self._player_identity_changes(game_id, player_name, user_id, seat)
            for player, field, value in changes:
                player[field] = value
            if changes:
                self._changed()

    def _player_identity_changes(
        self,
        game_id: str | None,
        player_name: str,
        user_id: str,
        seat: int | None = None,
    ) -> list[tuple[dict[str, Any], str, str | int]]:
        with self._lock:
            if not isinstance(user_id, str) or not user_id:
                raise ValueError("player user_id must be a non-empty string")
            if seat is not None and (
                isinstance(seat, bool) or not isinstance(seat, int) or seat not in range(4)
            ):
                raise ValueError("player seat must be between zero and three")
            target_game = None
            if game_id is not None:
                target_game = next(
                    (
                        item
                        for item in self._manifest["games"]
                        if item["game_id"] == game_id
                    ),
                    None,
                )
                if target_game is None:
                    raise ValueError("player identity references an unknown game")
            for player in self._manifest["players"]:
                if player["name"] != player_name:
                    continue
                if player["user_id"] not in {None, user_id}:
                    raise ValueError("player identity changed during a run")
                changes = []
                if player["user_id"] != user_id:
                    changes.append((player, "user_id", user_id))
                for registered_game in self._manifest["games"]:
                    game_player = next(
                        item
                        for item in registered_game["players"]
                        if item["name"] == player_name
                    )
                    if game_player["user_id"] not in {None, user_id}:
                        raise ValueError("player identity changed during a run")
                    if game_player["user_id"] != user_id:
                        changes.append((game_player, "user_id", user_id))
                if game_id is not None:
                    game_player = next(
                        item
                        for item in target_game["players"]
                        if item["name"] == player_name
                    )
                    if seat is not None and game_player["seat"] not in {None, seat}:
                        raise ValueError("player seat changed during a game")
                    if seat is not None and any(
                        item["name"] != player_name and item["seat"] == seat
                        for item in target_game["players"]
                    ):
                        raise ValueError("two players cannot occupy the same game seat")
                    if seat is not None and game_player["seat"] != seat:
                        changes.append((game_player, "seat", seat))
                return changes
            raise ValueError("player identity references an unknown player")

    def _record_inventory_decision(self, trace: DecisionTrace) -> None:
        """Count the recorded evaluation, independently of POST and fallback status.

        Legacy decisions without inventory metadata remain unreported. A disabled
        weight is counted only when the strategy explicitly records ``False``;
        forced actions are additionally separated from actual candidate comparisons.
        """
        key = (trace.game_id, trace.user_id, trace.round_no)
        row = self._inventory_decisions.setdefault(key, {
            "game_id": trace.game_id, "player_name": trace.player_name,
            "user_id": trace.user_id, "seat": trace.seat, "round_no": trace.round_no,
            "decisions": 0, "exact_decisions": 0, "unknown_decisions": 0,
            "unreported_decisions": 0, "effective_tile_weight_disabled": 0,
            "multi_candidate_weight_disabled": 0, "unknown_reasons": {},
        })
        details = trace.decision.details
        inventory = details.get("public_inventory")
        if inventory is None and isinstance(details.get("search"), Mapping):
            inventory = details["search"].get("public_inventory")
        status = inventory.get("status") if isinstance(inventory, Mapping) else None
        if not isinstance(status, str) or status not in {"exact", "unknown"}:
            status = "unreported"
        row["decisions"] += 1
        row[f"{status}_decisions"] += 1
        if status == "unknown":
            reason = inventory.get("reason")
            reason = reason if isinstance(reason, str) else "unspecified"
            row["unknown_reasons"][reason] = row["unknown_reasons"].get(reason, 0) + 1
        if details.get("effective_tile_weight_used") is False:
            row["effective_tile_weight_disabled"] += 1
            if len(trace.candidates) > 1:
                row["multi_candidate_weight_disabled"] += 1
        self._manifest["inventory_decision_counts"] = [
            self._inventory_decisions[item] for item in sorted(self._inventory_decisions)
        ]

    def _inventory_summary(self) -> dict[str, Any]:
        rows = self._manifest["inventory_decision_counts"]
        fields = (
            "decisions", "exact_decisions", "unknown_decisions", "unreported_decisions",
            "effective_tile_weight_disabled", "multi_candidate_weight_disabled",
        )
        players = []
        for player in self._manifest["players"]:
            matching = [row for row in rows if row["user_id"] == player["user_id"]]
            players.append({
                "player_name": player["name"], "user_id": player["user_id"],
                **{field: sum(row[field] for row in matching) for field in fields},
            })
        return {
            "scope": "all_recorded_decisions_including_conflicts_and_forced_actions",
            "weight_disabled_definition": "explicit_effective_tile_weight_used_false",
            **{field: sum(row[field] for row in rows) for field in fields},
            "by_player": players,
            "by_player_round": rows,
        }

    def mark_degraded(self, game_id: str, player_name: str, reason: str) -> None:
        with self._lock:
            if game_id not in self._manifest["game_ids"]:
                raise ValueError("degradation references an unknown game")
            degradation = {
                "game_id": game_id,
                "player_name": player_name,
                "reason": reason,
            }
            if degradation not in self._manifest["degradations"]:
                self._manifest["degradations"].append(degradation)
                self._changed()

    def _build_summary(self) -> dict[str, Any]:
        if self._rounds_per_game is None or not self._manifest["games"]:
            raise RuntimeError("cannot complete run without a complete replay set")
        expected_rounds = set(range(1, self._rounds_per_game + 1))
        player_user_ids = [player["user_id"] for player in self._manifest["players"]]
        if (
            any(not isinstance(user_id, str) or not user_id for user_id in player_user_ids)
            or len(set(player_user_ids)) != 4
        ):
            raise RuntimeError("cannot complete run without four player identities")

        wins = {user_id: 0 for user_id in player_user_ids}
        draw_count = 0
        for game in self._manifest["games"]:
            if game["batch"] is None:
                raise RuntimeError("cannot complete run without a complete replay set")
            if {item["round_no"] for item in game["rounds"]} != expected_rounds:
                raise RuntimeError("cannot complete run without a complete replay set")
            players = game["players"]
            if (
                len(players) != 4
                or {player.get("seat") for player in players} != set(range(4))
                or {player.get("user_id") for player in players} != set(player_user_ids)
                or any(not isinstance(player.get("server_name"), str) for player in players)
            ):
                raise RuntimeError("cannot complete run with invalid game players")
            for item in game["rounds"]:
                outcome = item["outcome"]
                if outcome["kind"] == "draw":
                    draw_count += 1
                    continue
                winner = outcome["winner"]
                user_id = winner.get("user_id") if isinstance(winner, dict) else None
                if user_id not in wins:
                    raise RuntimeError("cannot complete run with an unknown winner")
                wins[user_id] += 1

        game_count = len(self._manifest["games"])
        round_count = game_count * self._rounds_per_game
        indexed_players = list(enumerate(self._manifest["players"]))
        indexed_players.sort(
            key=lambda item: (-wins[item[1]["user_id"]], item[0])
        )
        leaderboard = []
        for _, player in indexed_players:
            strategy = player["strategy"]
            win_count = wins[player["user_id"]]
            leaderboard.append(
                {
                    "name": player["name"],
                    "user_id": player["user_id"],
                    **({"strategy_mode": strategy["mode"]} if "mode" in strategy else {}),
                    "strategy_name": strategy["name"],
                    "strategy_version": strategy["version"],
                    "wins": win_count,
                    "rounds_played": round_count,
                    "win_rate": win_count / round_count,
                }
            )
        if sum(wins.values()) + draw_count != round_count:
            raise RuntimeError("replay outcomes do not cover every round")
        return {
            "game_count": game_count,
            "round_count": round_count,
            "draw_count": draw_count,
            "leaderboard": leaderboard,
            "inventory": self._inventory_summary(),
        }

    def apply_round(self, relative: str, record: dict[str, Any]) -> None:
        with self._lock:
            if not isinstance(record, dict) or record.get("schema_version") != 3:
                raise ValueError("round artifact must use record schema 3")
            if record.get("run_id") != self.run_id:
                raise ValueError("round artifact belongs to a different run")
            game_id = record.get("game_id")
            game = next(
                (
                    item
                    for item in self._manifest["games"]
                    if item["game_id"] == game_id
                ),
                None,
            )
            if game is None:
                raise ValueError("round artifact belongs to an unknown game")
            round_no = record.get("round_no")
            if (
                isinstance(round_no, bool)
                or not isinstance(round_no, int)
                or self._rounds_per_game is None
                or round_no not in range(1, self._rounds_per_game + 1)
            ):
                raise ValueError("round artifact has an invalid round_no")
            batch = record.get("batch")
            if isinstance(batch, bool) or not isinstance(batch, int):
                raise ValueError("round artifact has an invalid batch")
            if game["batch"] not in {None, batch}:
                raise ValueError("game batch changed across round artifacts")
            players = record.get("players")
            if (
                not isinstance(players, list)
                or len(players) != 4
                or {player.get("seat") for player in players if isinstance(player, dict)}
                != set(range(4))
            ):
                raise ValueError("round artifact must contain four seated players")
            configured_names = {player["name"] for player in self._manifest["players"]}
            if {player.get("name") for player in players} != configured_names:
                raise ValueError("round artifact players do not match configured players")
            configured_by_name = {
                player["name"]: player for player in self._manifest["players"]
            }
            observed_by_name = {
                player["name"]: player for player in game["players"]
            }
            for replay_player in players:
                configured = configured_by_name[replay_player["name"]]
                observed = observed_by_name[replay_player["name"]]
                expected_strategy = {
                    key: configured["strategy"][key]
                    for key in ("mode", "name", "version") if key in configured["strategy"]
                }
                if (
                    not isinstance(replay_player.get("user_id"), str)
                    or replay_player["user_id"] != configured["user_id"]
                    or replay_player.get("strategy") != expected_strategy
                    or not isinstance(replay_player.get("server_name"), str)
                    or not replay_player["server_name"]
                    or observed["seat"] not in {None, replay_player["seat"]}
                    or observed["server_name"]
                    not in {None, replay_player["server_name"]}
                ):
                    raise ValueError(
                        "round artifact conflicts with configured player identities"
                    )
            outcome = record.get("outcome")
            if not isinstance(outcome, dict) or outcome.get("kind") not in {"win", "draw"}:
                raise ValueError("round artifact has an invalid outcome")
            if outcome["kind"] == "draw" and outcome.get("winner") is not None:
                raise ValueError("draw round artifact cannot contain a winner")
            if outcome["kind"] == "win" and not isinstance(outcome.get("winner"), dict):
                raise ValueError("winning round artifact must contain a winner")

            existing = next(
                (item for item in game["rounds"] if item["round_no"] == round_no),
                None,
            )
            summary = {
                "round_no": round_no,
                "record_path": relative,
                "outcome": outcome,
            }
            if existing is not None:
                if existing != summary:
                    raise ValueError("round artifact conflicts with an existing round")
                return
            game["batch"] = batch
            game["players"] = sorted(players, key=lambda player: player["seat"])
            game["rounds"].append(summary)
            game["rounds"].sort(key=lambda item: item["round_no"])
            self._manifest["rounds"].append(relative)
            self._manifest["rounds"] = sorted(set(self._manifest["rounds"]))
            self._changed()

    def validate_decision(self, payload: dict[str, Any]) -> None:
        validate_decision(payload)
        if self._manifest["status"] != "running":
            raise RuntimeError("cannot record a decision after run completion")
        game = next((g for g in self._manifest["games"] if g["game_id"] == payload["game_id"]), None)
        if game is None:
            raise ValueError("player identity references an unknown game")
        player = next((p for p in self._manifest["players"] if p["name"] == payload["player_name"]), None)
        if player is None or player["user_id"] not in {None, payload["user_id"]}:
            raise ValueError("player identity changed during a run")
        for item in game["players"]:
            if item["name"] == payload["player_name"]:
                if item["seat"] not in {None, payload["seat"]}:
                    raise ValueError("player seat changed during a game")
            elif item["seat"] == payload["seat"]:
                raise ValueError("two players cannot occupy the same game seat")

    def apply_decision(self, payload: dict[str, Any], encoded_size: int) -> None:
        self.validate_decision(payload)
        self.identify_player(payload["game_id"], payload["player_name"], payload["user_id"], payload["seat"])
        counts = self._manifest["decision_counts"]
        counts["total"] += 1
        counts[payload["submission"]] += 1
        if payload["schema_version"] == 4:
            executions = self._manifest.setdefault("execution_counts", {})
            source = payload["execution_source"]
            executions[source] = executions.get(source, 0) + 1
        if payload["fallback_used"]:
            counts["fallback"] += 1
            self.mark_degraded(payload["game_id"], payload["player_name"], payload["fallback_reason"])
        trace = SimpleNamespace(**{key: payload[key] for key in ("game_id", "user_id", "round_no", "player_name", "seat", "candidates")})
        trace.decision = SimpleNamespace(details=payload["decision"]["details"])
        self._record_inventory_decision(trace)

    def complete(self) -> None:
        if self._manifest["status"] != "running":
            return
        summary = self._build_summary()
        self._manifest["summary"] = summary
        self._manifest["status"] = self._completion_status()
        self._manifest["finished_at"] = self._now().isoformat()
        self._changed()

    def fail(self, error: str) -> None:
        if self._manifest["status"] != "running":
            return
        self._manifest.update(status="failed", finished_at=self._now().isoformat(), error=error)
        self._changed()


class ParticipantProjection(Projection):
    def _require_running(self, action: str) -> None:
        if self._manifest["status"] != RunStatus.RUNNING.value:
            raise RuntimeError(f"cannot {action} after {self._manifest['mode']} run completion")

    def record_guide_version(self, version: int | None) -> None:
        if version is not None and type(version) is not int:
            raise ValueError("observed guide version must be an integer or None")
        with self._lock:
            self._require_running("record guide version")
            previous = self._manifest["guide_version"]
            self._manifest["guide_version"] = version
            try:
                self._changed()
            except BaseException:
                self._manifest["guide_version"] = previous
                raise

    @staticmethod
    def _non_empty_string(value: object, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
        return str(value)

    @staticmethod
    def _optional_stage_no(value: int | None) -> int | None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError("stage_no must be a non-negative integer")
        return value

    def _game(self, game_id: str) -> dict[str, Any]:
        game = next(
            (item for item in self._manifest["games"] if item["game_id"] == game_id),
            None,
        )
        if game is None:
            raise ValueError("operation references an unknown game")
        return game

    def identify_participant(self, user_id: str) -> None:
        user_id = self._non_empty_string(user_id, "participant user_id")
        with self._lock:
            self._require_running("identify participant")
            current = self._manifest["participant"]["user_id"]
            if current not in {None, user_id}:
                raise ValueError("participant identity changed during a race run")
            if current is None:
                self._manifest["participant"]["user_id"] = user_id
                try:
                    self._changed()
                except BaseException:
                    self._manifest["participant"]["user_id"] = None
                    raise

    def _validate_identity(
        self,
        game_id: str,
        player_name: str,
        user_id: str,
        seat: int,
    ) -> tuple[dict[str, Any], bool, bool]:
        game = self._game(game_id)
        player_name = self._non_empty_string(player_name, "player_name")
        user_id = self._non_empty_string(user_id, "user_id")
        if player_name != user_id:
            raise ValueError("race player_name must match the participant user_id")
        if isinstance(seat, bool) or not isinstance(seat, int) or seat not in range(4):
            raise ValueError("participant seat must be between zero and three")
        participant_user_id = self._manifest["participant"]["user_id"]
        if participant_user_id not in {None, user_id}:
            raise ValueError("participant identity changed during a race run")
        if game["seat"] not in {None, seat}:
            raise ValueError("participant seat changed during a game")
        return game, participant_user_id is None, game["seat"] is None

    def identify_player(
        self,
        game_id: str,
        player_name: str,
        user_id: str,
        seat: int,
    ) -> None:
        with self._lock:
            self._require_running("identify participant seat")
            game, set_identity, set_seat = self._validate_identity(
                game_id, player_name, user_id, seat
            )
            if not set_identity and not set_seat:
                return
            if set_identity:
                self._manifest["participant"]["user_id"] = user_id
            if set_seat:
                game["seat"] = seat
            try:
                self._changed()
            except BaseException:
                if set_identity:
                    self._manifest["participant"]["user_id"] = None
                if set_seat:
                    game["seat"] = None
                raise

    @staticmethod
    def _scores(final_scores: Sequence[int]) -> list[int]:
        if (
            isinstance(final_scores, (str, bytes))
            or len(final_scores) != 4
            or any(isinstance(score, bool) or not isinstance(score, int) for score in final_scores)
        ):
            raise ValueError("final_scores must contain exactly four integers")
        return list(final_scores)

    def mark_game_completed(
        self,
        game_id: str,
        final_scores: Sequence[int],
    ) -> None:
        scores = self._scores(final_scores)
        with self._lock:
            self._require_running("complete a game")
            game = self._game(game_id)
            if game["status"] == "completed":
                if game["final_scores"] == scores:
                    return
                raise ValueError("completed game has conflicting final scores")
            if game["status"] != "running":
                raise ValueError("cannot complete an aborted game")
            game["status"] = "completed"
            game["final_scores"] = scores
            try:
                self._changed()
            except BaseException:
                game["status"] = "running"
                game["final_scores"] = None
                raise

    def _reason(self, value: str, field: str) -> str:
        if self.public and value == "<redacted>":
            return value
        if not isinstance(value, str) or _STABLE_REASON.fullmatch(value) is None:
            raise ValueError(f"{field} must be a stable snake_case reason")
        return str(value)

    def mark_game_aborted(self, game_id: str, reason: str) -> None:
        reason = self._reason(reason, "aborted reason")
        with self._lock:
            self._require_running("abort a game")
            game = self._game(game_id)
            if game["status"] == "aborted":
                if game["aborted_reason"] == reason:
                    return
                raise ValueError("aborted game has a conflicting reason")
            if game["status"] != "running":
                raise ValueError("cannot abort a completed game")
            game["status"] = "aborted"
            game["aborted_reason"] = reason
            try:
                self._changed()
            except BaseException:
                game["status"] = "running"
                game["aborted_reason"] = None
                raise

    def _append_degradation(
        self,
        game_id: str,
        participant_user_id: str,
        reason: str,
    ) -> bool:
        degradation = {
            "game_id": game_id,
            "participant_user_id": participant_user_id,
            "reason": reason,
        }
        if degradation in self._manifest["degradations"]:
            return False
        self._manifest["degradations"].append(degradation)
        return True

    def mark_degraded(self, game_id: str, player_name: str, reason: str) -> None:
        reason = self._reason(reason, "degradation reason")
        with self._lock:
            self._require_running("mark strategy degradation")
            self._game(game_id)
            participant_user_id = self._manifest["participant"]["user_id"]
            if participant_user_id is None or player_name != participant_user_id:
                raise ValueError("degradation references an unknown participant")
            if self._append_degradation(game_id, participant_user_id, reason):
                try:
                    self._changed()
                except BaseException:
                    self._manifest["degradations"].pop()
                    raise

    @staticmethod
    def _ranking(value: Mapping[str, object] | None) -> dict[str, int] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("participant_ranking must be a mapping")
        ranking: dict[str, int] = {}
        for key in _RANKING_FIELDS:
            if key not in value:
                continue
            item = value[key]
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"participant_ranking.{key} must be an integer")
            ranking[key] = item
        return ranking

    def _summary(
        self,
        terminal_reason: str,
        participant_ranking: Mapping[str, object] | None,
    ) -> dict[str, Any]:
        games = self._manifest["games"]
        summary: dict[str, Any] = {
            "terminal_reason": terminal_reason,
            "game_count": len(games),
            "completed_game_count": sum(
                game["status"] == "completed" for game in games
            ),
            "aborted_game_count": sum(
                game["status"] == "aborted" for game in games
            ),
        }
        ranking = self._ranking(participant_ranking)
        if ranking is not None:
            summary["participant_ranking"] = ranking
        return summary

    def validate_decision(self, payload: dict[str, Any]) -> None:
        validate_decision(payload)
        self._require_running("record a decision")
        game, _, _ = self._validate_identity(payload["game_id"], payload["player_name"], payload["user_id"], payload["seat"])
        if game["status"] != "running":
            raise ValueError("cannot record a decision for a terminal game")
        if payload["fallback_used"]:
            self._reason(payload["fallback_reason"] or "fallback", "fallback reason")

    def apply_decision(self, payload: dict[str, Any], encoded_size: int) -> None:
        self.validate_decision(payload)
        game, set_identity, set_seat = self._validate_identity(payload["game_id"], payload["player_name"], payload["user_id"], payload["seat"])
        if set_identity:
            self._manifest["participant"]["user_id"] = payload["user_id"]
        if set_seat:
            game["seat"] = payload["seat"]
        counts = self._manifest["decision_counts"]
        counts["total"] += 1
        counts[payload["submission"]] += 1
        if payload["schema_version"] == 4:
            executions = self._manifest.setdefault("execution_counts", {})
            source = payload["execution_source"]
            executions[source] = executions.get(source, 0) + 1
        self._manifest["decision_log"]["committed_bytes"] += encoded_size
        self._manifest["decision_log"]["committed_entries"] += 1
        if payload["fallback_used"]:
            counts["fallback"] += 1
            self._append_degradation(payload["game_id"], payload["user_id"], payload["fallback_reason"] or "fallback")


class RaceProjection(ParticipantProjection):
    operations = RACE_OPERATIONS

    def record_lifecycle(
        self,
        tournament_status: str,
        *,
        stage_no: int | None = None,
        stage_name: str | None = None,
        stage_status: str | None = None,
        stage_crashed: bool | None = None,
        qualified: bool | None = None,
        qualify_role: str | None = None,
    ) -> None:
        tournament_status = self._non_empty_string(
            tournament_status, "tournament_status"
        )
        stage_no = self._optional_stage_no(stage_no)
        optional_strings = {
            "stage_name": stage_name,
            "stage_status": stage_status,
            "qualify_role": qualify_role,
        }
        for field, value in optional_strings.items():
            if value is not None:
                optional_strings[field] = self._non_empty_string(value, field)
        for field, value in (("stage_crashed", stage_crashed), ("qualified", qualified)):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field} must be a boolean")

        observation: dict[str, Any] = {"tournament_status": tournament_status}
        optional = {
            "stage_no": stage_no,
            **optional_strings,
            "stage_crashed": stage_crashed,
            "qualified": qualified,
        }
        observation.update({key: value for key, value in optional.items() if value is not None})
        with self._lock:
            self._require_running("record lifecycle")
            lifecycle = self._manifest["lifecycle"]
            if lifecycle:
                previous = {
                    key: value
                    for key, value in lifecycle[-1].items()
                    if key != "observed_at"
                }
                if previous == observation:
                    return
            observation["observed_at"] = self._now().isoformat()
            lifecycle.append(observation)
            try:
                self._changed()
            except BaseException:
                lifecycle.pop()
                raise

    def register_game(
        self,
        game_id: str,
        *,
        stage_no: int | None = None,
        stage_name: str | None = None,
    ) -> None:
        game_id = self._non_empty_string(game_id, "game_id")
        stage_no = self._optional_stage_no(stage_no)
        if stage_name is not None:
            stage_name = self._non_empty_string(stage_name, "stage_name")
        candidate = {
            "game_id": game_id,
            "stage_no": stage_no,
            "stage_name": stage_name,
            "status": "running",
            "seat": None,
            "final_scores": None,
            "aborted_reason": None,
        }
        with self._lock:
            self._require_running("register games")
            existing = next(
                (item for item in self._manifest["games"] if item["game_id"] == game_id),
                None,
            )
            if existing is not None:
                if existing["stage_no"] != stage_no or existing["stage_name"] != stage_name:
                    raise ValueError("game re-registration conflicts with existing metadata")
                return
            self._manifest["game_ids"].append(game_id)
            self._manifest["games"].append(candidate)
            try:
                self._changed()
            except BaseException:
                self._manifest["game_ids"].pop()
                self._manifest["games"].pop()
                raise

    def complete(self, terminal_reason: str, participant_ranking: Mapping[str, object] | None = None) -> None:
        if terminal_reason not in {"finished", "eliminated"}:
            raise ValueError("terminal_reason must be finished or eliminated")
        self._require_running("complete")
        if any(game["status"] == "running" for game in self._manifest["games"]):
            raise RuntimeError("cannot complete race run with a running game")
        summary = self._summary(terminal_reason, participant_ranking)
        self._manifest.update(status=self._completion_status(),
                              finished_at=self._now().isoformat(), summary=summary)
        self._changed()

    def fail(self, error: dict[str, Any], terminal_reason: str | None = None,
             participant_ranking: Mapping[str, object] | None = None) -> None:
        if terminal_reason not in {None, "void", "closed"}:
            raise ValueError("failed terminal_reason must be void or closed")
        self._require_running("fail")
        summary = self._summary(terminal_reason, participant_ranking) if terminal_reason is not None or participant_ranking is not None else None
        self._manifest.update(status="failed", finished_at=self._now().isoformat(), summary=summary, error=error)
        self._changed()


class MatchProjection(ParticipantProjection):
    """One automatic matching session, including failures before assignment."""

    operations = MATCH_OPERATIONS

    def session_win_stats(self) -> MatchWinStats:
        total = won = known = 0
        for game in self._manifest["games"]:
            total += game.get("completed_rounds", 0)
            for row in game.get("round_outcomes", []):
                won += row["outcome"] == "win"
                known += row["outcome"] != "unknown"
        config = self._manifest.get("match_config") or {}
        tables, rounds = config.get("M"), config.get("Rounds")
        if ("statistics_base" in self._manifest
                and type(tables) is int and tables > 0
                and type(rounds) is int and rounds > 0
                and any(row["tournament_status"] == "finished" for row in self._manifest["lifecycle"])):
            # A finished room confirms every configured small round ended,
            # including tables missed by discovery or missing terminal scores.
            total = max(total, tables * rounds)
        return MatchWinStats(won, total, total - known)

    def win_stats(self) -> MatchWinStats:
        base = MatchWinStats.from_dict(self._manifest["statistics_base"])
        return base + self.session_win_stats()

    def _changed(self) -> None:
        # Old records did not collect outcomes. Replaying them must not invent
        # zero wins or reinterpret their completed games as new statistics.
        if "statistics_base" in self._manifest:
            self._manifest["session_win_stats"] = self.session_win_stats().to_dict()
            self._manifest["win_stats"] = self.win_stats().to_dict()
        super()._changed()

    def set_statistics_base(self, stats: Mapping[str, object]) -> None:
        self._require_running("set statistics base")
        stats_value = MatchWinStats.from_dict(stats)
        previous = MatchWinStats.from_dict(self._manifest["statistics_base"])
        if previous == stats_value:
            return
        if self._manifest["games"] or previous != MatchWinStats():
            raise ValueError("statistics base cannot change after collection starts")
        self._manifest["statistics_base"] = stats_value.to_dict()
        self._changed()

    def _completed_rounds(self, value: int, *, allow_zero: bool = False) -> int:
        rounds = (self._manifest.get("match_config") or {}).get("Rounds")
        if type(rounds) is not int or rounds < 1:
            raise ValueError("statistics require a positive actual match Rounds")
        if type(value) is not int or not (0 if allow_zero else 1) <= value <= rounds:
            raise ValueError("completed rounds must be within actual match Rounds")
        return value

    def record_round_progress(self, game_id: str, completed_rounds: int) -> None:
        self._require_running("record round progress")
        number = self._completed_rounds(completed_rounds, allow_zero=True)
        game = self._game(game_id)
        if number <= game.get("completed_rounds", 0):
            return
        game["completed_rounds"] = number
        self._changed()

    def record_round_outcome(self, game_id: str, round_no: int, *, outcome: str, source: str,
                             conflicted: bool = False) -> None:
        self._require_running("record round outcome")
        number = self._completed_rounds(round_no)
        if outcome not in {"win", "loss", "draw", "unknown"}:
            raise ValueError("invalid round outcome")
        if source not in {"state_event", "score_delta", "replay"}:
            raise ValueError("invalid round outcome source")
        if type(conflicted) is not bool or (conflicted and outcome != "unknown"):
            raise ValueError("conflicted must be a boolean and only marks unknown outcomes")
        game = self._game(game_id)
        rows = game.setdefault("round_outcomes", [])
        previous = next((row for row in rows if row["round_no"] == number), None)
        incoming = {"round_no": number, "outcome": outcome, "source": source}
        if conflicted:
            incoming["conflicted"] = True
        if previous is not None:
            if previous["source"] == "replay" and source != "replay":
                return
            if previous.get("conflicted") and (previous["source"] == "replay" or source != "replay"):
                return
            if outcome == "unknown" and previous["outcome"] != "unknown" and not conflicted:
                return
            if conflicted:
                pass  # Proven contradictory evidence must invalidate live counts.
            elif source == "replay" and previous["source"] != "replay":
                pass  # An atomic validated replay supersedes live evidence.
            elif previous["outcome"] not in {"unknown", outcome}:
                incoming.update(outcome="unknown", conflicted=True)
            elif previous["outcome"] == outcome:
                return
            rows[rows.index(previous)] = incoming
        else:
            rows.append(incoming)
            rows.sort(key=lambda row: row["round_no"])
        game["completed_rounds"] = max(game.get("completed_rounds", 0), number)
        self._changed()

    def mark_game_completed(self, game_id: str, final_scores: Sequence[int]) -> None:
        super().mark_game_completed(game_id, final_scores)
        if "statistics_base" in self._manifest:
            self.record_round_progress(game_id, (self._manifest.get("match_config") or {}).get("Rounds"))

    @staticmethod
    def public_match_config(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("match_config must be a mapping")
        fields = {"M", "Rounds", "BaseScore", "PengTimeoutSec", "ChiTimeoutSec", "DiscardTimeoutSec",
                  "StartAt", "RegisterDeadlineAt", "TimeoutMin", "YouCaiBiKao", "OnlineConfirm",
                  "Kind", "Name", "Description"}
        projected = {key: value[key] for key in sorted(fields) if key in value}
        # Assignment evidence must survive invalid business values. The runtime
        # validates Kind, required rules and requested limits after durable bind.
        try:
            return json.loads(json.dumps(projected, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError):
            raise ValueError("match_config must contain JSON values") from None

    def bind_match(self, room_id: str, *, match_config: Mapping[str, Any], round_no: int) -> None:
        self._require_running("bind a match")
        room_id = self._non_empty_string(room_id, "room_id")
        config = self.public_match_config(match_config)
        if type(round_no) is not int or round_no < 0:
            raise ValueError("match round_no must be a non-negative integer")
        if self._manifest["room_id"] is not None:
            if (self._manifest["room_id"], self._manifest["match_config"], self._manifest["round_no"]) != (room_id, config, round_no):
                raise ValueError("match binding conflicts with the existing session")
            return
        self._manifest.update(room_id=room_id, match_config=config, round_no=round_no,
                              matched_at=self._now().isoformat())
        self._changed()

    def record_lifecycle(self, tournament_status: str, *, stage_no: int | None = None,
                         stage_name: str | None = None, stage_status: str | None = None,
                         stage_crashed: bool | None = None, qualified: bool | None = None,
                         qualify_role: str | None = None) -> None:
        self._require_running("record lifecycle")
        if any(value is not None for value in (stage_no, stage_name, stage_status, stage_crashed, qualified, qualify_role)):
            raise ValueError("match sessions do not have tournament stages")
        status = self._non_empty_string(tournament_status, "tournament_status")
        lifecycle = self._manifest["lifecycle"]
        if lifecycle and lifecycle[-1]["tournament_status"] == status:
            return
        lifecycle.append({"tournament_status": status, "observed_at": self._now().isoformat()})
        self._changed()

    def register_game(self, game_id: str, *, stage_no: int | None = None, stage_name: str | None = None) -> None:
        self._require_running("register games")
        game_id = self._non_empty_string(game_id, "game_id")
        if self._manifest["room_id"] is None:
            raise ValueError("cannot register a game before matching a room")
        if stage_no is not None or stage_name is not None:
            raise ValueError("match games do not have tournament stages")
        if game_id in self._manifest["game_ids"]:
            return
        self._manifest["game_ids"].append(game_id)
        self._manifest["games"].append({"game_id": game_id, "status": "running", "seat": None,
                                       "final_scores": None, "aborted_reason": None,
                                       "interrupted_reason": None})
        if "statistics_base" in self._manifest:
            self._manifest["games"][-1].update(completed_rounds=0, round_outcomes=[])
        self._changed()

    def _summary(self, terminal_reason: str, participant_ranking: Mapping[str, object] | None) -> dict[str, Any]:
        summary = super()._summary(terminal_reason, participant_ranking)
        summary["interrupted_game_count"] = sum(game["status"] == "interrupted" for game in self._manifest["games"])
        summary["unfinished_game_count"] = sum(game["status"] in {"running", "interrupted"} for game in self._manifest["games"])
        if "statistics_base" in self._manifest:
            summary["session_win_stats"] = self.session_win_stats().to_dict()
            summary["win_stats"] = self.win_stats().to_dict()
        return summary

    def complete(self, terminal_reason: str = "finished", participant_ranking: Mapping[str, object] | None = None) -> None:
        self._require_running("complete")
        if terminal_reason != "finished":
            raise ValueError("match completion requires terminal_reason finished")
        if self._manifest["room_id"] is None:
            raise RuntimeError("cannot complete an unmatched session")
        if any(game["status"] == "running" for game in self._manifest["games"]):
            raise RuntimeError("cannot complete match run with a running game")
        self._manifest.update(status=self._completion_status(),
                              finished_at=self._now().isoformat(), summary=self._summary(terminal_reason, participant_ranking))
        self._changed()

    def fail(self, error: dict[str, Any], terminal_reason: str = "runtime_error",
             participant_ranking: Mapping[str, object] | None = None) -> None:
        self._require_running("fail")
        reason = self._reason(terminal_reason, "terminal reason")
        self._manifest.update(status="failed", finished_at=self._now().isoformat(),
                              summary=self._summary(reason, participant_ranking), error=error)
        self._changed()

    def interrupt(self, terminal_reason: str = "user_interrupt") -> None:
        self._require_running("interrupt")
        reason = self._reason(terminal_reason, "terminal reason")
        finished_at = self._now().isoformat()
        for game in self._manifest["games"]:
            if game["status"] == "running":
                game.update(status="interrupted", interrupted_reason=reason)
        self._manifest.update(status="interrupted", finished_at=finished_at,
                              summary=self._summary(reason, None))
        self._changed()


def projection_type(mode: str) -> type[Projection]:
    try:
        return {"train": TrainProjection, "race": RaceProjection, "match": MatchProjection}[mode]
    except KeyError:
        raise ValueError("invalid artifact projection mode") from None
