"""Conservative participant round evidence and bounded background reconciliation."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from heuristic_ai.artifacts.replay import extract_match_round_outcomes
from heuristic_ai.protocol.compatibility import round_ended_metadata


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _scores(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(type(v) is int for v in value):
        return tuple(value)
    return None


class RoundEvidence:
    """One game's small amount of state survives worker generation restarts.

    Only an explicit round number, or one ending following a known current-round
    snapshot with no gap, can assign a live event to a round. Sequence tails from
    older snapshots never supply a new-round baseline.
    """

    def __init__(self, game_id: str, rounds: int, recorder: Any) -> None:
        self.game_id, self.rounds, self.recorder = game_id, rounds, recorder
        self.own_seat: int | None = None
        self.round_no: int | None = None
        self.sequence: int | None = None
        self.baseline: tuple[int, ...] | None = None
        self.baseline_completed: int | None = None
        self.completed = 0
        self.ending_sequence = 0
        self.ending_rounds: dict[int, int] = {}
        self.contradictory_rounds: set[int] = set()
        self._lock = threading.RLock()

    def _progress(self, count: int) -> None:
        count = min(self.rounds, max(0, count))
        if count > self.completed:
            self.recorder.record_round_progress(self.game_id, count)
            self.completed = count

    def _outcome(self, round_no: int, outcome: str, source: str) -> None:
        self._progress(round_no)
        self.recorder.record_round_outcome(self.game_id, round_no, outcome=outcome, source=source)

    def _conflict(self, round_no: int) -> None:
        self._progress(round_no)
        self.contradictory_rounds.add(round_no)
        self.recorder.record_round_outcome(self.game_id, round_no, outcome="unknown",
                                           source="state_event", conflicted=True)

    def _event_outcome(self, event: dict[str, Any]) -> str | None:
        data = event.get("data")
        if not isinstance(data, dict):
            return None
        draw, winner = data.get("draw"), event.get("seat")
        if type(draw) not in (bool, int) or draw not in (False, True):
            return None
        if draw:
            return "draw" if type(winner) is int and winner == -1 else None
        if not _integer(winner, 0, 3) or self.own_seat is None:
            return None
        return "win" if winner == self.own_seat else "loss"

    def observe(self, response: Any, *, request_seq: int | None = None) -> None:
        # Validation deliberately precedes recorder calls: bad protocol evidence
        # is ignored, while storage exceptions cross the guarded callback intact.
        if not isinstance(response, dict) or response.get("pending"):
            return
        with self._lock:
            snapshot = response.get("snapshot")
            snapshot = snapshot if isinstance(snapshot, dict) else {}
            phase = snapshot.get("phase")
            valid_snapshot = (
                _integer(snapshot.get("round_no"), 1, self.rounds)
                and _integer(snapshot.get("seat"), 0, 3)
                and _scores(snapshot.get("scores")) is not None
                and isinstance(phase, str)
                and phase in {"deal", "draw", "response_peng", "response_chi", "settled", "finished"}
            )
            legacy_terminal = (response.get("finished") is True and set(snapshot) == {"scores"}
                               and _scores(snapshot.get("scores")) is not None)
            if not valid_snapshot and not legacy_terminal:
                snapshot = {}
            supplied_round = snapshot.get("round_no")
            round_no = supplied_round if _integer(supplied_round, 1, self.rounds) else None
            seq = response.get("seq")
            seq = seq if type(seq) is int and seq >= 0 else None
            if seq is None and not (response.get("finished") is True and _scores(snapshot.get("scores")) is not None):
                return
            if self.sequence is not None and seq is not None and seq < self.sequence:
                return
            if snapshot and seq is not None and seq < self.ending_sequence:
                return
            if self.round_no is not None and round_no is not None and round_no < self.round_no:
                return
            seat = snapshot.get("seat")
            if _integer(seat, 0, 3):
                if self.own_seat is not None and seat != self.own_seat:
                    # Seats may differ across games, never within a game.
                    return
                self.own_seat = seat
            terminal = response.get("finished") is True and _scores(snapshot.get("scores")) is not None
            phase = snapshot.get("phase")
            settled = isinstance(phase, str) and phase in {"settled", "finished"}
            completed = self.rounds if terminal else round_no - (not settled) if round_no else None
            if completed is not None:
                self._progress(completed)

            events = response.get("events")
            events = events if isinstance(events, list) and seq is not None else []
            endings = [event for event in events if isinstance(event, dict)
                       and event.get("type") == "round_ended"
                       and type(event.get("seq")) is int and event["seq"] > 0
                       and (seq is None or event["seq"] <= seq)]
            fresh = [event for event in endings
                     if self.sequence is not None and event["seq"] > self.sequence]
            endings.sort(key=lambda event: event["seq"])
            for event in endings:
                data = event.get("data")
                event_round = self.ending_rounds.get(event["seq"])
                try:
                    explicit_round, _ = round_ended_metadata(event, expected_round_no=event_round)
                    if explicit_round is not None and explicit_round > self.rounds:
                        raise ValueError("round_ended round_no exceeds configured rounds")
                except ValueError:
                    candidates = {value for value in (event.get("round_no"),
                        data.get("round_no") if isinstance(data, dict) else None)
                        if _integer(value, 1, self.rounds)}
                    if event_round is not None:
                        self._conflict(event_round)
                    elif len(candidates) == 1:
                        self._conflict(candidates.pop())
                    continue
                if explicit_round is not None:
                    event_round = explicit_round
                elif (event_round is None and len(fresh) == 1 and event is fresh[0] and not response.get("gap")
                      and self.round_no is not None and self.baseline_completed == self.round_no - 1
                      and self.ending_sequence <= (self.sequence or 0)
                      and (completed == self.round_no or completed is None and type(request_seq) is int
                           and request_seq > 0 and request_seq >= (self.sequence or 0))
                      and (round_no is None or round_no in (self.round_no, self.round_no + 1))):
                    event_round = self.round_no
                if event_round is None:
                    if (event["seq"] > self.ending_sequence > (self.sequence or 0)
                            and round_no is None and not response.get("gap")
                            and type(request_seq) is int and request_seq > 0
                            and self._event_outcome(event) is not None):
                        # A distinct second ending proves another completed hand,
                        # but cannot reuse the old snapshot's round assignment.
                        self._progress(self.completed + 1)
                    self.ending_sequence = max(self.ending_sequence, event["seq"])
                    continue
                # A snapshot from an earlier round cannot confirm a future end.
                if round_no is not None and event_round > round_no:
                    continue
                self._progress(event_round)
                if len(self.ending_rounds) < self.rounds * 2 or event["seq"] in self.ending_rounds:
                    self.ending_rounds[event["seq"]] = event_round
                self.ending_sequence = max(self.ending_sequence, event["seq"])
                outcome = self._event_outcome(event)
                previous_ending = max((item["seq"] for item in endings if item["seq"] < event["seq"]), default=0)
                hu_events = [item for item in events if isinstance(item, dict) and item.get("type") == "hu"
                             and type(item.get("seq")) is int and previous_ending < item["seq"] < event["seq"]
                             and (item.get("round_no") in (None, event_round))]
                if hu_events and (len(hu_events) != 1 or outcome == "draw"
                                  or hu_events[0].get("seat") != event.get("seat")):
                    self._conflict(event_round)
                    outcome = None
                if outcome is not None:
                    self._outcome(event_round, outcome, "state_event")

            scores = _scores(snapshot.get("scores"))
            if (completed is not None and scores is not None and self.baseline is not None
                    and self.baseline_completed is not None and completed == self.baseline_completed + 1
                    and completed not in self.contradictory_rounds
                    and self.own_seat is not None):
                delta = tuple(after - before for before, after in zip(self.baseline, scores))
                winners = [seat for seat, value in enumerate(delta) if value > 0]
                if not any(delta):
                    self._outcome(completed, "draw", "score_delta")
                elif (sum(delta) == 0 and len(winners) == 1
                      and all(value < 0 for seat, value in enumerate(delta) if seat != winners[0])):
                    self._outcome(completed, "win" if winners[0] == self.own_seat else "loss", "score_delta")
            if round_no is not None and seq is not None and scores is not None and completed is not None:
                # Changing scores without a completed-round boundary invalidates
                # the baseline; it must never turn an aggregate delta into a win.
                if (self.baseline_completed == completed and self.baseline is not None
                        and scores != self.baseline):
                    self.baseline = None
                else:
                    self.baseline = scores
                self.baseline_completed = completed
                self.round_no, self.sequence = round_no, seq


class StatisticsCancelled(RuntimeError):
    pass


class StatisticsRequestGate:
    """A per-room bucket shared by listing, replay reads, and every HTTP retry."""

    def __init__(self, stop: Any, *, interval: float = 0.25) -> None:
        self.stop, self.interval = stop, interval
        self.next_allowed = 0.0
        self._lock = threading.Lock()

    def pause(self, seconds: float) -> None:
        if self.stop.is_set() or self.stop.wait(max(0.0, seconds)):
            raise StatisticsCancelled()

    def wait(self, method: str, path: str) -> None:
        with self._lock:
            self.pause(self.next_allowed - time.monotonic())
            self.next_allowed = time.monotonic() + self.interval

    def wait_until(self, method: str, path: str, deadline_monotonic: float) -> None:
        self.wait(method, path)


class MatchStatisticsReader:
    """Read transient public replay data; retain only validated round outcomes."""

    def __init__(self, *, room_id: str, rounds: int, user_id: str,
                 client_factory: Callable[[], Any], stop: Any, recorder: Any) -> None:
        self.room_id, self.rounds, self.user_id = room_id, rounds, user_id
        self.client_factory, self.recorder = client_factory, recorder
        self.gate = StatisticsRequestGate(stop)
        self.client: Any = None
        self.bound_gate = False

    def _request(self, path: str) -> Any:
        if self.client is None:
            self.client = self.client_factory()
            bind_gate = getattr(self.client, "bind_request_gate", None)
            if callable(bind_gate):
                bind_gate(self.gate)
                self.bound_gate = True
            bind_wait = getattr(self.client, "bind_retry_wait", None)
            if callable(bind_wait):
                bind_wait(self.gate.pause)
        if not self.bound_gate:
            self.gate.wait("GET", path)
        return self.client.request("GET", path)

    def read(self, game_id: str, own_seat: int | None) -> None:
        listing = self._request(f"/api/test-rooms/{self.room_id}/games")
        if isinstance(listing, dict):
            if "room_id" in listing and listing["room_id"] != self.room_id:
                raise ValueError("statistics_room_mismatch")
            listing = listing.get("games")
        if not isinstance(listing, list):
            raise ValueError("statistics_invalid_game_list")
        entries = [entry for entry in listing if isinstance(entry, dict) and entry.get("game_id") == game_id]
        if len(entries) != 1 or not _integer(entries[0].get("batch"), 0, 2**63 - 1):
            raise ValueError("statistics_missing_game_batch")
        batch = entries[0]["batch"]
        payload = self._request(f"/api/test-rooms/{self.room_id}/games/{batch}/events")
        outcomes = extract_match_round_outcomes(payload, room_id=self.room_id, game_id=game_id,
                     batch=batch, rounds=self.rounds, user_id=self.user_id, own_seat=own_seat)
        # Wrap only persistence calls, keeping ordinary network/schema failures
        # nonfatal while preserving audit-failure provenance through the future.
        from .recovery import guard_callback
        record = guard_callback(self.recorder.record_round_outcome,
                                getattr(self.gate.stop, "parent", self.gate.stop))
        for outcome in outcomes:
            record(game_id, outcome["round_no"], outcome=outcome["outcome"], source="replay")
