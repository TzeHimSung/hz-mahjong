"""Small, durable stage ledgers; no full state, private hands or replay downloads."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable
import uuid

from .match_statistics import RoundEvidence


class RaceStageStatistics:
    """One authorized worker writes; a checkpoint is replaced only after fsync.

    The per-game adapters capture the stage attempt. Late callbacks cannot add
    results to a new stage or resurrect a stage invalidated by the server.
    """

    def __init__(self, rounds: int, *, path: Path | None = None,
                 warning: Callable[[str], None] = lambda _reason: None) -> None:
        if type(rounds) is not int or rounds < 1:
            raise ValueError("race statistics require positive rounds")
        self.rounds, self.path, self.warning = rounds, path, warning
        self._lock = threading.RLock()
        self._data: dict[str, Any] | None = None
        self._loaded = False
        self._storage_failed = False
        self._corrupt = False
        self._warned: set[str] = set()
        self._evidence: dict[tuple[int, str], RoundEvidence] = {}
        if path is not None and path.exists():
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                data = envelope["data"]
                if envelope["sha256"] != hashlib.sha256(self._encode(data)).hexdigest():
                    raise ValueError("checkpoint checksum mismatch")
                self._validate(data)
                self._data, self._loaded = data, True
            except (OSError, ValueError, KeyError, TypeError):
                self._corrupt = True
                self._warn("checkpoint_invalid")

    @staticmethod
    def _encode(data: Any) -> bytes:
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    def _validate(self, data: Any) -> None:
        if (not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1
                or data.get("rounds") != self.rounds
                or data.get("identity") != (self.path.parent.name if self.path is not None else "memory")):
            raise ValueError("invalid race statistics checkpoint")
        if data.get("stage_no") is not None and (type(data["stage_no"]) is not int or data["stage_no"] < 0):
            raise ValueError("invalid statistics stage")
        if type(data.get("attempt")) is not int or data["attempt"] < 1:
            raise ValueError("invalid statistics attempt")
        for key in ("coverage", "crashed", "started"):
            if type(data.get(key)) is not bool:
                raise ValueError("invalid statistics state")
        if not isinstance(data.get("inventory"), list) or any(type(k) is not str or not k for k in data["inventory"]):
            raise ValueError("invalid statistics inventory")
        if not isinstance(data.get("games"), dict):
            raise ValueError("invalid statistics games")
        for game_id, game in data["games"].items():
            if not isinstance(game_id, str) or not game_id or not isinstance(game, dict):
                raise ValueError("invalid statistics game")
            if type(game.get("completed")) is not int or not 0 <= game["completed"] <= self.rounds:
                raise ValueError("invalid completed round count")
            if type(game.get("observed")) is not bool or not isinstance(game.get("outcomes"), dict):
                raise ValueError("invalid round evidence")
            for number, row in game["outcomes"].items():
                if str(int(number)) != number or not 1 <= int(number) <= game["completed"]:
                    raise ValueError("invalid round number")
                if not isinstance(row, dict) or row.get("outcome") not in {"win", "loss", "draw", "unknown"}:
                    raise ValueError("invalid round outcome")
                if row.get("source") not in {"state_event", "score_delta"} or type(row.get("conflicted", False)) is not bool:
                    raise ValueError("invalid round evidence source")

    def _warn(self, reason: str) -> None:
        if reason not in self._warned:
            self._warned.add(reason)
            self.warning(reason)

    def _save(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            raw = self._encode(self._data)
            envelope = self._encode({"data": self._data, "sha256": hashlib.sha256(raw).hexdigest()})
            with temporary.open("xb") as stream:
                stream.write(envelope)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self._storage_failed = False
            self._warned.discard("checkpoint_write_failed")
        except OSError:
            self._storage_failed = True
            self._warn("checkpoint_write_failed")
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def observe_tournament(self, *, stage_no: int | None, status: str,
                           crashed: bool = False, game_ids: tuple[str, ...] | None = None) -> None:
        with self._lock:
            previous = copy.deepcopy(self._data)
            fresh = self._data is None or self._data["stage_no"] != stage_no
            invalidated = not fresh and crashed and not self._data["crashed"]
            # A reopened attempt is also conclusive if this stage had started.
            reopened = (not fresh and self._data["started"] and status in {"registering", "stage_open"}
                        and self._data.get("status") not in {"registering", "stage_open"})
            if fresh or invalidated or reopened:
                attempt = 1 if fresh else self._data["attempt"] + 1
                self._data = {"version": 1, "identity": self.path.parent.name if self.path is not None else "memory",
                              "rounds": self.rounds, "stage_no": stage_no,
                              "attempt": attempt, "coverage": status in {"registering", "stage_open"},
                              "started": False, "crashed": crashed, "status": status,
                              "inventory": [], "games": {}}
                self._evidence.clear()
                # A pre-start observation is a new, provably empty baseline.
                if self._data["coverage"]:
                    self._corrupt = False
            assert self._data is not None
            data = self._data
            if status in {"registering", "stage_open"} and not data["started"] and not data["games"]:
                data["coverage"] = True
                self._corrupt = False
            if self._loaded:
                # Without an inventory, downtime may have hidden entire tables.
                if status not in {"registering", "stage_open"} and game_ids is None:
                    data["coverage"] = False
                self._loaded = False
            data.update(status=status, crashed=crashed)
            if status == "running" and not crashed:
                data["started"] = True
            if game_ids is not None and status not in {"registering", "stage_open"} and not crashed:
                data["inventory"] = sorted(set(data["inventory"]) | set(game_ids))
            if status in {"stage_done", "finished"} and not crashed:
                # Only real table identifiers contribute rounds; byes have none.
                for game in data["games"].values():
                    game["completed"] = self.rounds
            if data != previous or self._storage_failed:
                self._save()

    @staticmethod
    def _empty_game() -> dict[str, Any]:
        return {"completed": 0, "outcomes": {}, "observed": False}

    def evidence(self, game_id: str) -> RoundEvidence:
        with self._lock:
            if self._data is None:
                raise RuntimeError("statistics stage has not been observed")
            scope = (self._data["stage_no"], self._data["attempt"])
            key = (scope[1], game_id)
            if key not in self._evidence:
                self._data["games"].setdefault(game_id, self._empty_game())
                self._save()
                self._evidence[key] = RoundEvidence(game_id, self.rounds, _StageRecorder(self, scope))
            return self._evidence[key]

    def _game(self, scope: tuple[int | None, int], game_id: str) -> dict[str, Any] | None:
        if self._data is None or (self._data["stage_no"], self._data["attempt"]) != scope or self._data["crashed"]:
            return None
        return self._data["games"].get(game_id)

    def observe(self, game_id: str) -> Callable[..., None]:
        evidence = self.evidence(game_id)
        scope = (self._data["stage_no"], self._data["attempt"])

        def callback(response: Any, *, request_seq: int | None = None) -> None:
            evidence.observe(response, request_seq=request_seq)
            if evidence.own_seat is not None or evidence.completed:
                with self._lock:
                    game = self._game(scope, game_id)
                    if game is not None and not game["observed"]:
                        game["observed"] = True
                        self._save()
        return callback

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._data is None:
                return {"total_rounds": 0, "won_rounds": 0, "unknown_rounds": 0,
                        "statistics_complete": False, "win_rate": None, "attempt": None,
                        "incomplete_reason": "history_missing"}
            data = self._data
            games = data["games"]
            total = sum(g["completed"] for g in games.values())
            known = sum(row["outcome"] != "unknown" for g in games.values() for row in g["outcomes"].values())
            won = sum(row["outcome"] == "win" for g in games.values() for row in g["outcomes"].values())
            reason = None
            if self._storage_failed or self._corrupt:
                reason = "checkpoint_unavailable"
            elif not data["coverage"]:
                reason = "history_missing"
            elif not all(k in games and games[k]["observed"] for k in data["inventory"]):
                reason = "tables_unconfirmed"
            elif total != known:
                reason = "round_results_missing"
            complete = reason is None
            return {"total_rounds": total, "won_rounds": won, "unknown_rounds": total - known,
                    "statistics_complete": complete, "win_rate": won / total if complete and total else None,
                    "attempt": data["attempt"], "incomplete_reason": reason}


class _StageRecorder:
    def __init__(self, owner: RaceStageStatistics, scope: tuple[int | None, int]) -> None:
        self.owner, self.scope = owner, scope

    def record_round_progress(self, game_id: str, completed_rounds: int) -> None:
        with self.owner._lock:
            game = self.owner._game(self.scope, game_id)
            if game is not None and completed_rounds > game["completed"]:
                game["completed"] = completed_rounds
                self.owner._save()

    def record_round_outcome(self, game_id: str, round_no: int, *, outcome: str,
                             source: str, conflicted: bool = False) -> None:
        with self.owner._lock:
            game = self.owner._game(self.scope, game_id)
            if game is None:
                return
            key = str(round_no)
            previous = game["outcomes"].get(key)
            incoming = {"outcome": outcome, "source": source}
            if previous is not None:
                if previous.get("conflicted"):
                    return
                if outcome == "unknown" and not conflicted:
                    return
                if previous["outcome"] == outcome and not conflicted:
                    return
                if previous["outcome"] not in {"unknown", outcome}:
                    conflicted = True
            if conflicted:
                incoming.update(outcome="unknown", conflicted=True)
            game["outcomes"][key] = incoming
            game["completed"] = max(game["completed"], round_no)
            self.owner._save()
