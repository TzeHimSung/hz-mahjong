from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from heuristic_ai.protocol.client import RequestDeadlineExceeded, redact
from heuristic_ai.protocol.compatibility import round_ended_metadata
from heuristic_ai.artifacts.files import OwnedArtifactFile, checked_path

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")
_SENTINEL = object()
_BEIJING = timezone(timedelta(hours=8))
_DRAW_ACTION_TYPES = {
    "tile_discarded": "discard",
    "gang": "gang",
    "hu": "hu",
}
_POST_ROUND_END_GAMEPLAY_TYPES = frozenset(
    {"tile_drawn", "pass", "chi", "peng", *_DRAW_ACTION_TYPES}
)
_REPLAY_REQUEST_INTERVAL = 0.21  # Both public replay endpoints share a 5/s room bucket.


class _ArchiveCancelled(RuntimeError):
    pass


class _ReplayNotReady(ValueError):
    """A finished listing can precede visibility of the final replay event."""


class _ReplayRequestGate:
    """Apply the same room budget to logical requests and transport retries."""

    def __init__(self, archiver: ReplayArchiver) -> None:
        self.archiver = archiver

    def wait(self, method: str, path: str) -> None:
        self.archiver._wait_for_request()

    def wait_until(self, method: str, path: str, deadline_monotonic: float) -> None:
        self.archiver._wait_for_request(deadline_monotonic=deadline_monotonic)


def _validate_identifier(value: str) -> str:
    if not value or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("identifier must contain only letters, digits, '_' or '-'")
    return value


@dataclass(frozen=True, slots=True)
class RoundBundle:
    directory_name: str
    record: dict[str, Any]
    events: list[dict[str, Any]]


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with OwnedArtifactFile(path) as handle:
        handle.append(data)
        handle.seal()
        if handle.digest() != hashlib.sha256(data).hexdigest():
            raise ValueError("round replay verification failed")


def _write_new_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with OwnedArtifactFile(path) as handle:
        for value in values:
            data = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            digest.update(data)
            handle.append(data, sync=False)
        handle.seal()
        if handle.digest() != digest.hexdigest():
            raise ValueError("round replay verification failed")


def _event_integer(event: dict[str, Any], key: str, context: str) -> int:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} event has an invalid {key}")
    return value


def _result_boolean(value: object, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{context} must be a boolean or 0/1")


def _player_identity(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": player["name"],
        "server_name": player["server_name"],
        "user_id": player["user_id"],
        "seat": player["seat"],
    }


def _winner_seat(
    value: object,
    source: str,
    *,
    allow_absent_sentinel: bool = False,
    allow_none: bool = False,
) -> int | None:
    if allow_none and value is None:
        return None
    if (
        allow_absent_sentinel
        and not isinstance(value, bool)
        and isinstance(value, int)
        and value == -1
    ):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(4):
        raise ValueError(f"round outcome inconsistency: {source}={value!r} is invalid")
    return value


def _rounds_result_values(result: dict[str, Any]) -> tuple[bool, int | None]:
    result_draw = _result_boolean(
        result.get("is_draw"),
        "round outcome is_draw",
    )
    rounds_winner = _winner_seat(
        result.get("winner"),
        "rounds.winner",
        allow_none=result_draw,
        allow_absent_sentinel=result_draw,
    )
    if result_draw and rounds_winner is not None:
        raise ValueError(
            "round outcome inconsistency: "
            f"draw has winner source rounds.winner={rounds_winner}"
        )
    return result_draw, rounds_winner


def _terminal_draw_before(
    events: list[dict[str, Any]], ending_index: int, winner_seat: int,
) -> dict[str, Any]:
    index = ending_index - 1
    if index < 0:
        return {}
    marker = events[index]
    if (marker.get("type") == "timeout" and isinstance(marker.get("data"), dict)
            and marker["data"].get("kind") == "discard"):
        if marker.get("seat") == winner_seat:
            index -= 1
        elif index >= 2:
            draw, discard = events[index - 1], events[index - 2]
            seat = marker.get("seat")
            # The server can advance to the next draw before appending the
            # preceding discard's timeout. Match that annotation to its action;
            # keep it in the replay and retain the winner's actual draw.
            if (type(seat) is int and seat in range(4)
                    and discard.get("type") == "tile_discarded" and discard.get("seat") == seat
                    and draw.get("type") == "tile_drawn" and draw.get("seat") == winner_seat
                    and winner_seat == (seat + 1) % 4
                    and _event_integer(discard, "seq", "terminal discard") + 1
                        == _event_integer(draw, "seq", "terminal draw")
                    and _event_integer(draw, "seq", "terminal draw") + 1
                        == _event_integer(marker, "seq", "terminal discard timeout")
                    and _event_integer(discard, "ts", "terminal discard")
                        <= _event_integer(draw, "ts", "terminal draw")
                        <= _event_integer(marker, "ts", "terminal discard timeout")):
                index -= 1
    return events[index] if index >= 0 else {}


def _draw_to_action(
    events: list[dict[str, Any]],
    players: list[dict[str, Any]],
    *,
    terminal_winner_seat: int | None,
    terminal_hu_absent: bool,
) -> list[dict[str, Any]]:
    by_seat = {player["seat"]: player for player in players}
    pending: dict[int, dict[str, Any]] = {}
    timings: list[dict[str, Any]] = []
    round_ended = False
    for index, event in enumerate(events):
        event_type = event.get("type")
        if round_ended and event_type in _POST_ROUND_END_GAMEPLAY_TYPES:
            raise ValueError("draw-to-action gameplay event follows round_ended")
        if event_type == "round_ended":
            if (
                pending
                and terminal_hu_absent
                and terminal_winner_seat is not None
                and len(pending) == 1
            ):
                terminal_draw = pending.get(terminal_winner_seat)
                if terminal_draw is not None and index > 0:
                    preceding_event = _terminal_draw_before(events, index, terminal_winner_seat)
                    if (
                        preceding_event.get("type") == "tile_drawn"
                        and _event_integer(
                            preceding_event, "seat", "draw-to-action"
                        ) == terminal_winner_seat
                        and _event_integer(
                            preceding_event, "seq", "draw-to-action"
                        ) == terminal_draw["seq"]
                        and terminal_draw["ts"]
                        <= _event_integer(event, "ts", "draw-to-action")
                    ):
                        pending.clear()
            if pending:
                raise ValueError("draw-to-action contains an unclosed draw at round_ended")
            round_ended = True
            continue
        if event_type != "tile_drawn" and event_type not in _DRAW_ACTION_TYPES:
            continue
        seat = _event_integer(event, "seat", "draw-to-action")
        if seat not in by_seat:
            raise ValueError("draw-to-action event has an invalid seat")
        timestamp = _event_integer(event, "ts", "draw-to-action")
        sequence = _event_integer(event, "seq", "draw-to-action")
        if event_type == "tile_drawn":
            if seat in pending:
                raise ValueError("draw-to-action contains an unclosed draw")
            pending[seat] = {
                "seq": sequence,
                "tile": event.get("tile"),
                "ts": timestamp,
            }
            continue

        draw = pending.pop(seat, None)
        if draw is None:
            continue
        if timestamp < draw["ts"]:
            raise ValueError("draw-to-action timestamp moves backwards")
        timings.append(
            {
                **_player_identity(by_seat[seat]),
                "draw": draw,
                "action": {
                    "seq": sequence,
                    "kind": _DRAW_ACTION_TYPES[event_type],
                    "tile": event.get("tile"),
                    "ts": timestamp,
                },
                "elapsed_seconds": timestamp - draw["ts"],
            }
        )

    if pending:
        raise ValueError("draw-to-action contains an unclosed draw")
    return timings


def _round_outcome(
    events: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    result_source: str,
    players: list[dict[str, Any]],
) -> dict[str, Any]:
    ending = [event for event in events if event.get("type") == "round_ended"]
    if len(ending) != 1 or not isinstance(ending[0].get("data"), dict):
        raise ValueError("round outcome requires one usable round_ended event")
    _, dealer = round_ended_metadata(ending[0])
    event_draw = _result_boolean(
        ending[0]["data"].get("draw"),
        "round outcome draw",
    )
    rounds_winner = None
    if result_source == "rounds":
        result_draw, rounds_winner = _rounds_result_values(result)
        if result_draw != event_draw:
            raise ValueError("round outcome inconsistency: draw status conflicts with round_ended")

    hu_events = [event for event in events if event.get("type") == "hu"]
    if len(hu_events) > 1:
        raise ValueError("round outcome inconsistency: multiple hu events")

    winner_sources: list[tuple[str, int]] = []
    if hu_events:
        hu_seat = _winner_seat(hu_events[0].get("seat"), "hu.seat")
        if hu_seat is not None:
            winner_sources.append(("hu.seat", hu_seat))

    ending_seat = _winner_seat(
        ending[0].get("seat"),
        "round_ended.seat",
        allow_absent_sentinel=True,
    )
    if ending_seat is not None:
        winner_sources.append(("round_ended.seat", ending_seat))

    if rounds_winner is not None:
        winner_sources.append(("rounds.winner", rounds_winner))

    if event_draw:
        if winner_sources:
            sources = ", ".join(
                f"{source}={seat}" for source, seat in winner_sources
            )
            raise ValueError(
                f"round outcome inconsistency: draw has winner source {sources}"
            )
        return {"kind": "draw", "winner": None}

    if not winner_sources:
        raise ValueError("round outcome inconsistency: win has no winner source")
    if len({seat for _, seat in winner_sources}) != 1:
        sources = ", ".join(
            f"{source}={seat}" for source, seat in winner_sources
        )
        raise ValueError(
            f"round outcome inconsistency: winner sources disagree ({sources})"
        )

    winner_seat = winner_sources[0][1]
    # Initial dealing emits no draw event, including the dealer's 14th tile.
    # An immediate dealer win can therefore contain only settlement markers.
    opening_dealer_win = (
        ending_seat is not None and ending_seat == dealer
        and events[0] is ending[0]
        and all(event.get("type") == "game_ended" for event in events[1:])
    )
    if not hu_events and not opening_dealer_win:
        ending_index = next(
            index for index, event in enumerate(events) if event is ending[0]
        )
        if ending_seat is None or ending_index == 0:
            raise ValueError(
                "round outcome inconsistency: no-hu win lacks terminal winning draw"
            )
        terminal_draw = _terminal_draw_before(events, ending_index, winner_seat)
        if (
            terminal_draw.get("type") != "tile_drawn"
            or _event_integer(terminal_draw, "seat", "round outcome") != winner_seat
        ):
            raise ValueError(
                "round outcome inconsistency: no-hu win lacks terminal winning draw"
            )

    by_seat = {player["seat"]: player for player in players}
    winner_player = by_seat.get(winner_seat)
    if winner_player is None:
        raise ValueError("round outcome winner cannot be mapped to a player")
    return {"kind": "win", "winner": _player_identity(winner_player)}


def _validated_replay_rounds(
    payload: dict[str, Any], *, room_id: str, game_id: str, batch: int,
    rounds_per_game: int,
) -> list[dict[str, Any]]:
    """Validate the shared replay envelope and select one settlement source."""
    if not isinstance(payload, dict):
        raise ValueError("replay payload must be an object")
    if type(batch) is not int or batch < 0 or type(payload.get("batch")) is not int:
        raise ValueError("replay batch must be a non-negative integer")
    if payload.get("room_id") != room_id:
        raise ValueError("replay payload belongs to a different room")
    if payload.get("game_id") != game_id:
        raise ValueError("replay payload belongs to a different game")
    if payload.get("batch") != batch:
        raise ValueError("replay payload belongs to a different batch")
    if payload.get("status") != "finished":
        raise ValueError("replay payload is not marked finished")
    blocks = payload.get("blocks")
    rounds = payload.get("rounds")
    if not isinstance(blocks, list) or not isinstance(rounds, list):
        raise ValueError("replay response must contain blocks and rounds arrays")

    expected_rounds = rounds_per_game
    if (
        isinstance(expected_rounds, bool)
        or not isinstance(expected_rounds, int)
        or expected_rounds < 1
    ):
        raise ValueError("replay config must contain a positive integer Rounds")
    if not all(isinstance(block, dict) for block in blocks):
        raise ValueError("every replay block must be an object")
    ordered_blocks = sorted(blocks, key=lambda item: item.get("seq_start", -1))
    by_round: dict[int, list[dict[str, Any]]] = {}
    last_seq_end = 0
    all_events: list[tuple[int, dict[str, Any]]] = []
    for block in ordered_blocks:
        if block.get("truncated"):
            raise ValueError("replay contains a truncated block")
        seq_start = block.get("seq_start")
        seq_end = block.get("seq_end")
        if (
            isinstance(seq_start, bool)
            or not isinstance(seq_start, int)
            or isinstance(seq_end, bool)
            or not isinstance(seq_end, int)
            or seq_start != last_seq_end + 1
            or seq_end < seq_start
        ):
            raise ValueError("replay block sequence is not continuous")
        last_seq_end = seq_end
        round_no = block.get("round_no")
        if (
            isinstance(round_no, bool)
            or not isinstance(round_no, int)
            or round_no < 1
        ):
            raise ValueError("replay block is missing a valid round_no")
        events = block.get("events")
        if not isinstance(events, list) or not all(
            isinstance(event, dict) for event in events
        ):
            raise ValueError("replay block events must be objects")
        event_sequences = [event.get("seq") for event in events]
        if (
            any(
                isinstance(sequence, bool) or not isinstance(sequence, int)
                for sequence in event_sequences
            )
            or event_sequences != list(range(seq_start, seq_end + 1))
        ):
            raise ValueError("replay event sequence disagrees with block bounds")
        all_events.extend((round_no, event) for event in events)
        by_round.setdefault(round_no, []).append(block)

    results_by_round: dict[int, dict[str, Any]] = {}
    for result in rounds:
        if not isinstance(result, dict):
            raise ValueError("every round result must be an object")
        round_no = result.get("round_no")
        if (
            isinstance(round_no, bool)
            or not isinstance(round_no, int)
            or round_no < 1
            or round_no in results_by_round
        ):
            raise ValueError("round results must have unique positive round_no values")
        results_by_round[round_no] = result
    expected_round_numbers = set(range(1, expected_rounds + 1))
    if not set(results_by_round).issubset(expected_round_numbers):
        raise ValueError("round results fall outside the contiguous configured range")
    if set(by_round) != expected_round_numbers:
        raise ValueError("replay blocks must cover every contiguous configured round")
    summary_is_complete = set(results_by_round) == expected_round_numbers
    if not summary_is_complete:
        for result in results_by_round.values():
            _rounds_result_values(result)

    game_ended_events = [
        (round_no, event)
        for round_no, event in all_events
        if event.get("type") == "game_ended"
    ]
    if not game_ended_events:
        raise _ReplayNotReady("replay game_ended event is not yet available")
    if len(game_ended_events) != 1:
        raise ValueError("replay must contain exactly one game_ended event")
    game_ended_round, game_ended_event = game_ended_events[0]
    if game_ended_event is not all_events[-1][1]:
        # An automatic discard can exhaust the wall and emit both endings
        # before the server appends that discard's timeout annotation.
        # Keep the annotation in the evidence; it is still a server fallback.
        terminal = [event for _, event in all_events[-4:]]
        trailing_discard_timeout = (
            len(terminal) == 4
            and [event.get("type") for event in terminal]
            == ["tile_discarded", "round_ended", "game_ended", "timeout"]
            and all(round_no == expected_rounds for round_no, _ in all_events[-4:])
            and isinstance(terminal[-1].get("data"), dict)
            and terminal[-1]["data"].get("kind") == "discard"
            and type(terminal[-1].get("seat")) is int
            and terminal[-1]["seat"] in range(4)
            and terminal[-1]["seat"] == terminal[0].get("seat")
            and _event_integer(terminal[0], "ts", "terminal discard")
            <= _event_integer(terminal[1], "ts", "terminal round_ended")
            <= _event_integer(terminal[2], "ts", "game_ended")
            <= _event_integer(terminal[3], "ts", "terminal discard timeout")
        )
        if not trailing_discard_timeout:
            raise ValueError("game_ended event must be the final replay event")
    if game_ended_round != expected_rounds:
        raise ValueError("game_ended event must belong to the final configured round")
    final_round_ended = [
        event
        for round_no, event in all_events
        if round_no == expected_rounds and event.get("type") == "round_ended"
    ]
    if (
        len(final_round_ended) != 1
        or game_ended_event["seq"] <= final_round_ended[0]["seq"]
    ):
        raise ValueError(
            "game_ended event must follow the final round_ended event"
        )
    if _event_integer(game_ended_event, "seat", "game_ended") != -1:
        raise ValueError("game_ended event must have seat -1")
    if not isinstance(game_ended_event.get("data"), dict):
        raise ValueError("game_ended event data must be an object")

    validated: list[dict[str, Any]] = []
    for round_no in sorted(expected_round_numbers):
        round_blocks = by_round[round_no]
        events = [
            event
            for block in round_blocks
            for event in block["events"]
        ]
        events.sort(key=lambda event: event.get("seq", -1))
        ending_events = [
            event for event in events if event.get("type") == "round_ended"
        ]
        if len(ending_events) != 1:
            raise ValueError(
                f"round {round_no} must contain exactly one round_ended event"
            )
        round_ended_metadata(ending_events[0], expected_round_no=round_no)
        ended_at_unix = ending_events[0].get("ts")
        if isinstance(ended_at_unix, bool) or not isinstance(ended_at_unix, int):
            raise ValueError(f"round {round_no} has an invalid end timestamp")
        if summary_is_complete:
            result = results_by_round[round_no]
            result_source = "rounds"
        else:
            result = ending_events[0].get("data")
            if not isinstance(result, dict):
                raise ValueError(
                    f"round {round_no} has no usable settlement result"
                )
            result_source = "round_ended.data"
        validated.append({"round_no": round_no, "blocks": round_blocks,
                          "events": events, "result_source": result_source,
                          "result": result, "round_ended": ending_events[0]})
    return validated


def extract_match_round_outcomes(
    payload: dict[str, Any], *, room_id: str, game_id: str, batch: int,
    rounds: int, user_id: str, own_seat: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Return a fully validated replay's minimal outcomes for one participant.

    Match AI nicknames need not be present or unique. Seat order and unique
    user IDs are the identity source; no full states or replay events are kept.
    """
    settlements = _validated_replay_rounds(
        payload, room_id=room_id, game_id=game_id, batch=batch,
        rounds_per_game=rounds,
    )
    seats = payload.get("seats")
    if (not isinstance(seats, list) or len(seats) != 4
            or not all(isinstance(seat, dict)
                       and isinstance(seat.get("user_id"), str)
                       and seat["user_id"] for seat in seats)
            or len({seat["user_id"] for seat in seats}) != 4):
        raise ValueError("match replay must contain four unique user_id seats")
    for index, seat in enumerate(seats):
        if "seat" in seat and (type(seat["seat"]) is not int or seat["seat"] != index):
            raise ValueError("match replay seat identity disagrees with seat order")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("match participant user_id must be a non-empty string")
    matching = [index for index, seat in enumerate(seats) if seat["user_id"] == user_id]
    if len(matching) != 1:
        raise ValueError("match participant cannot be mapped to a replay seat")
    participant_seat = matching[0]
    if own_seat is not None and (type(own_seat) is not int or own_seat != participant_seat):
        raise ValueError("match participant seat conflicts with replay identity")
    players = [{"seat": index, "user_id": seat["user_id"],
                "name": seat["user_id"], "server_name": seat.get("name")}
               for index, seat in enumerate(seats)]
    rows: list[dict[str, object]] = []
    for settlement in settlements:
        result = _round_outcome(settlement["events"], settlement["result"],
                                result_source=settlement["result_source"], players=players)
        outcome = ("draw" if result["kind"] == "draw" else
                   "win" if result["winner"]["seat"] == participant_seat else "loss")
        rows.append({"round_no": settlement["round_no"], "outcome": outcome, "source": "replay"})
    return tuple(rows)


class ReplayArchiver:
    """Download completed game data on a background thread and store it atomically."""

    def __init__(
        self,
        client: Any,
        *,
        room_id: str,
        output_root: Path,
        players: list[dict[str, Any]],
        config: dict[str, Any],
        seed: int,
        run_id: str | None = None,
        poll_interval: float = 1.0,
        max_polls: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        on_round_published: Callable[[Path], None] | None = None,
        on_round_bundle: Callable[[RoundBundle], None] | None = None,
        on_archive_failure: Callable[[str, dict[str, Any], Any], None] | None = None,
    ) -> None:
        self.client = client
        self.room_id = _validate_identifier(room_id)
        self.output_root = Path(output_root)
        self.players = redact(players)
        self.config = redact(config)
        self.seed = seed
        self.run_id = _validate_identifier(run_id) if run_id is not None else None
        self.poll_interval = poll_interval
        self.max_polls = max_polls
        self.sleep = sleep
        self.clock = clock
        self.on_round_published = on_round_published
        self.on_round_bundle = on_round_bundle
        self.on_archive_failure = on_archive_failure
        if on_round_published is not None and on_round_bundle is not None:
            raise ValueError("choose a bundle sink or a published-path callback")
        if poll_interval < 0:
            raise ValueError("poll_interval cannot be negative")
        if max_polls is not None and max_polls < 1:
            raise ValueError("max_polls must be at least one")

        self._queue: queue.Queue[object] = queue.Queue()
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._errors: list[tuple[str, BaseException]] = []
        self._thread: threading.Thread | None = None
        self._closed = False
        self._cancelled = threading.Event()
        self._statuses: dict[str, str] = {}
        self._request_lock = threading.Lock()
        self._next_request_at: float | None = None
        bind_request_gate = getattr(self.client, "bind_request_gate", None)
        self._gates_attempts = callable(bind_request_gate)
        if self._gates_attempts:
            bind_request_gate(_ReplayRequestGate(self))

    def status_for(self, game_id: str) -> str | None:
        """Return the latest authoritative status without making a request."""
        with self._lock:
            return self._statuses.get(game_id)

    def _pause(self, seconds: float) -> bool:
        if self.sleep is time.sleep:
            return self._cancelled.wait(seconds)
        if not self._cancelled.is_set():
            self.sleep(seconds)
        return self._cancelled.is_set()

    def _wait_for_request(self, *, deadline_monotonic: float | None = None) -> None:
        with self._request_lock:
            while True:
                if self._cancelled.is_set():
                    raise _ArchiveCancelled("replay archival cancelled")
                now = self.clock()
                remaining = None if deadline_monotonic is None else deadline_monotonic - now
                if remaining is not None and remaining <= 0:
                    raise RequestDeadlineExceeded("replay request admission deadline exceeded")
                delay = 0.0 if self._next_request_at is None else self._next_request_at - now
                if delay <= 0:
                    self._next_request_at = now + _REPLAY_REQUEST_INTERVAL
                    return
                self._pause(delay if remaining is None else min(delay, remaining))

    def _request(self, path: str) -> Any:
        if not self._gates_attempts:
            self._wait_for_request()
        return self.client.request("GET", path)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("replay archiver already started")
            self._thread = threading.Thread(
                target=self._run,
                name="mahjong-replay-archiver",
                daemon=False,
            )
            self._thread.start()

    def submit(self, game_id: str) -> None:
        safe_game_id = _validate_identifier(game_id)
        with self._lock:
            if self._thread is None:
                raise RuntimeError("replay archiver has not started")
            if self._closed:
                raise RuntimeError("replay archiver is closed")
            if safe_game_id in self._seen:
                return
            self._seen.add(safe_game_id)
            self._queue.put(safe_game_id)

    def close_and_wait(self) -> None:
        with self._lock:
            if self._thread is None:
                raise RuntimeError("replay archiver has not started")
            if not self._closed:
                self._closed = True
                self._queue.put(_SENTINEL)
            thread = self._thread

        self._queue.join()
        thread.join()
        if self._errors:
            game_id, error = self._errors[0]
            raise RuntimeError(
                f"archive failed for {game_id}: {type(error).__name__}: {error}"
            ) from error

    def cancel_and_wait(self) -> None:
        self._cancelled.set()
        with self._lock:
            if self._thread is None:
                raise RuntimeError("replay archiver has not started")
            if not self._closed:
                self._closed = True
                self._queue.put(_SENTINEL)
            thread = self._thread

        self._queue.join()
        thread.join()

    def _run(self) -> None:
        pending: dict[str, int] = {}
        try:
            self._run_pending(pending)
        except BaseException as error:
            # Even an injected clock/sleep failure must not leave queue.join()
            # blocked forever or allow submissions into a dead worker.
            with self._lock:
                self._closed = True
            self._errors.append((self.room_id, error))
            for _ in pending:
                self._queue.task_done()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()

    def _run_pending(self, pending: dict[str, int]) -> None:
        closing = False

        def accept(item: object) -> None:
            nonlocal closing
            if item is _SENTINEL:
                closing = True
                self._queue.task_done()
            else:
                pending[str(item)] = 0

        def complete(game_id: str, error: BaseException | None = None) -> None:
            if error is not None and not isinstance(error, _ArchiveCancelled):
                self._errors.append((game_id, error))
            pending.pop(game_id)
            self._queue.task_done()

        while True:
            if not pending and not closing:
                accept(self._queue.get())
            while True:
                try:
                    accept(self._queue.get_nowait())
                except queue.Empty:
                    break
            if self._cancelled.is_set():
                for game_id in tuple(pending):
                    complete(game_id)
            if not pending:
                if closing:
                    return
                continue

            for game_id in pending:
                pending[game_id] += 1
            try:
                room_status, entries = self._room_listing()
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                # A shared read can recover on the next poll while workers are
                # still playing. Closing must report an exhausted HTTP failure
                # rather than waiting forever for already completed workers.
                for game_id, polls in tuple(pending.items()):
                    if self._cancelled.is_set() or closing or (
                        self.max_polls is not None and polls >= self.max_polls
                    ):
                        complete(game_id, error)
                if pending:
                    self._pause(self.poll_interval)
                continue

            for game_id, polls in tuple(pending.items()):
                if self._cancelled.is_set():
                    complete(game_id)
                    continue
                entry = entries.get(game_id)
                try:
                    if self._check_archiveable(game_id, entry, room_status):
                        assert entry is not None
                        self._archive_finished(game_id, entry)
                        complete(game_id)
                    elif self.max_polls is not None and polls >= self.max_polls:
                        raise TimeoutError(f"game {game_id} did not become archiveable")
                except BaseException as error:
                    complete(game_id, error)
            if pending:
                self._pause(self.poll_interval)

    def _room_listing(self) -> tuple[str | None, dict[str, dict[str, Any]]]:
        listing = self._request(f"/api/test-rooms/{self.room_id}/games")
        if self._cancelled.is_set():
            raise _ArchiveCancelled("replay archival cancelled")
        room_status = listing.get("status") if isinstance(listing, dict) else None
        room_status = room_status if isinstance(room_status, str) else None
        games = listing.get("games") if isinstance(listing, dict) else listing
        if not isinstance(games, list):
            raise ValueError("room games response must contain a games array")
        entries = {item["game_id"]: item for item in games
                   if isinstance(item, dict) and isinstance(item.get("game_id"), str)}
        with self._lock:
            for game_id in self._seen:
                entry_status = entries.get(game_id, {}).get("status")
                entry_status = entry_status if isinstance(entry_status, str) else None
                if entry_status in {"finished", "abandoned", "void", "closed"}:
                    self._statuses[game_id] = entry_status
                elif room_status in {"finished", "void", "closed"}:
                    # A closed room is not evidence that a missing individual
                    # game finished successfully.
                    self._statuses[game_id] = "closed" if room_status == "finished" else room_status
                elif entry_status == "running":
                    self._statuses[game_id] = "running"
        return room_status, entries

    def _check_archiveable(
        self, game_id: str, entry: dict[str, Any] | None, room_status: str | None,
    ) -> bool:
        if entry is not None and entry.get("status") == "finished":
            return True
        entry_status = entry.get("status") if entry is not None else None
        entry_status = entry_status if isinstance(entry_status, str) else None
        terminal_status = (entry_status if entry_status in {"abandoned", "void", "closed"}
                           else room_status if room_status in {"finished", "void", "closed"} else None)
        if terminal_status is not None:
            batch = entry.get("batch") if entry is not None else None
            self._quarantine_failure(
                game_id, batch=batch if type(batch) is int else None, status=terminal_status,
            )
            raise RuntimeError(f"game {game_id} reached {terminal_status} before archival")
        return False

    def _archive_finished(self, game_id: str, entry: dict[str, Any]) -> None:
        batch = entry.get("batch")
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise ValueError("finished game batch must be an integer")

        for attempt in range(3):
            payload = self._request(f"/api/test-rooms/{self.room_id}/games/{batch}/events")
            if self._cancelled.is_set():
                return
            if not isinstance(payload, dict):
                raise ValueError("replay response must be an object")
            try:
                bundles = self._build_round_bundles(payload, game_id=game_id, batch=batch)
            except _ReplayNotReady:
                if attempt == 2:
                    raise
                logging.getLogger(__name__).info(
                    "game=%s replay terminal event not yet visible; reread=%s", game_id, attempt + 1)
                if self._pause(max(.25, self.poll_interval)):
                    return
                continue
            self._publish_round_bundles(bundles)
            return

    def _quarantine_failure(
        self,
        game_id: str,
        *,
        batch: int | None,
        status: str,
    ) -> None:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "room_id": self.room_id,
            "game_id": game_id,
            "batch": batch,
            "status": status,
        }
        raw = None
        if batch is not None:
            try:
                payload = self._request(f"/api/test-rooms/{self.room_id}/games/{batch}/events")
            except BaseException as error:
                failure["replay_error"] = type(error).__name__
            else:
                raw = redact(payload)
        if self.on_archive_failure is not None:
            self.on_archive_failure(game_id, failure, raw)
            return
        target = self.output_root / "failed" / game_id
        while target.exists():
            target = self.output_root / "failed" / f"{game_id}_{uuid.uuid4().hex[:12]}"
        checked_path(target, self.output_root)
        target.mkdir(parents=True, exist_ok=False)
        if raw is not None:
            _write_new_json(target / "raw.json", raw)
        _write_new_json(target / "failure.json", failure)

    def _build_round_bundles(
        self,
        payload: dict[str, Any],
        *,
        game_id: str,
        batch: int,
    ) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
        seats = payload.get("seats")
        if (
            not isinstance(seats, list)
            or len(seats) != 4
            or not all(
                isinstance(seat, dict)
                and isinstance(seat.get("name"), str)
                and bool(seat["name"])
                and isinstance(seat.get("user_id"), str)
                and bool(seat["user_id"])
                for seat in seats
            )
            or len({seat["name"] for seat in seats}) != 4
            or len({seat["user_id"] for seat in seats}) != 4
        ):
            raise ValueError("replay payload must contain four unique seats")
        replay_players = self._players_by_seat(seats)
        settlements = _validated_replay_rounds(
            payload, room_id=self.room_id, game_id=game_id, batch=batch,
            rounds_per_game=self.config.get("Rounds"),
        )
        bundles: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
        for settlement in settlements:
            round_no = settlement["round_no"]
            round_blocks = settlement["blocks"]
            events = settlement["events"]
            result_source = settlement["result_source"]
            result = settlement["result"]
            ending_events = [settlement["round_ended"]]
            ended_at_unix = ending_events[0]["ts"]
            outcome = _round_outcome(
                events,
                result,
                result_source=result_source,
                players=replay_players,
            )
            terminal_winner_seat = (
                outcome["winner"]["seat"] if outcome["kind"] == "win" else None
            )
            draw_to_action = _draw_to_action(
                events,
                replay_players,
                terminal_winner_seat=terminal_winner_seat,
                terminal_hu_absent=not any(
                    event.get("type") == "hu" for event in events
                ),
            )
            ended_at = datetime.fromtimestamp(ended_at_unix, _BEIJING)
            directory_name = ended_at.strftime("%Y_%m_%d_%H_%M_%S")
            record_data = {
                "schema_version": 3,
                "room_id": self.room_id,
                "game_id": game_id,
                "batch": batch,
                "round_no": round_no,
                "ended_at_unix": ended_at_unix,
                "ended_at": ended_at.isoformat(timespec="seconds"),
                "seed": self.seed,
                "players": replay_players,
                "config": self.config,
                "seats": payload.get("seats"),
                "result_source": result_source,
                "result": result,
                "outcome": outcome,
                "draw_to_action": draw_to_action,
                "round_ended": ending_events[0],
                "blocks": round_blocks,
            }
            if self.run_id is not None:
                record_data["run_id"] = self.run_id
            record = redact(record_data)
            bundles.append((directory_name, record, redact(events)))
        return bundles

    def _players_by_seat(
        self,
        seats: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        configured_by_user: dict[str, dict[str, Any]] = {}
        for player in self.players:
            if not isinstance(player, dict):
                raise ValueError("configured replay players must be objects")
            user_id = player.get("user_id")
            if not isinstance(user_id, str) or not user_id:
                if self.run_id is not None:
                    raise ValueError("configured replay player is missing user_id")
                continue
            if user_id in configured_by_user:
                raise ValueError("configured replay players must have unique user_id values")
            configured_by_user[user_id] = player

        seat_user_ids = {seat["user_id"] for seat in seats}
        if self.run_id is not None and set(configured_by_user) != seat_user_ids:
            raise ValueError("configured replay players do not match replay seats")

        normalized: list[dict[str, Any]] = []
        for seat_index, seat in enumerate(seats):
            user_id = seat["user_id"]
            configured = configured_by_user.get(user_id)
            if configured is None:
                normalized.append(
                    {
                        "name": seat["name"],
                        "server_name": seat["name"],
                        "user_id": user_id,
                        "seat": seat_index,
                        "strategy": None,
                    }
                )
                continue
            name = configured.get("name")
            strategy = configured.get("strategy")
            if not isinstance(name, str) or not name:
                raise ValueError("configured replay player is missing a name")
            if not isinstance(strategy, dict):
                raise ValueError("configured replay player is missing strategy metadata")
            mode = strategy.get("mode")
            strategy_name = strategy.get("name")
            version = strategy.get("version")
            if (
                mode not in {None, "heuristic", "model"}
                or not isinstance(strategy_name, str)
                or not strategy_name
                or isinstance(version, bool)
                or not isinstance(version, int)
                or version < 1
            ):
                raise ValueError("configured replay player has invalid strategy metadata")
            normalized.append(
                {
                    "name": name,
                    "server_name": seat["name"],
                    "user_id": user_id,
                    "seat": seat_index,
                    "strategy": {
                        **({"mode": mode} if mode is not None else {}),
                        "name": strategy_name,
                        "version": version,
                    },
                }
            )
        return normalized

    def _publish_round_bundles(
        self,
        bundles: list[tuple[str, dict[str, Any], list[dict[str, Any]]]],
    ) -> None:
        # Validate the entire response's encoding before publishing any round.
        for _, record, events in bundles:
            json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
            for event in events:
                json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if self.on_round_bundle is not None:
            for directory_name, record, events in bundles:
                self.on_round_bundle(RoundBundle(directory_name, record, events))
            return
        self.output_root.mkdir(parents=True, exist_ok=True)
        reserved_names: set[str] = set()
        for directory_name, record, events in bundles:
            target = self._resolve_target(directory_name, record, expected_events=events,
                                          reserved_names=reserved_names)
            reserved_names.add(target.name)
            with OwnedArtifactFile.create_directory(target, ownership_root=self.output_root, temporary=False):
                _write_new_json(target / "record.json", record)
                _write_new_jsonl(target / "events.jsonl", events)
            if self.on_round_published is not None:
                self.on_round_published(target)

    def _resolve_target(
        self,
        directory_name: str,
        expected_record: dict[str, Any],
        *,
        expected_events: list[dict[str, Any]],
        reserved_names: set[str],
    ) -> Path | None:
        round_suffix = f"{directory_name}_round_{expected_record['round_no']}"
        candidate_names = [directory_name, round_suffix]
        collision_index = 2
        while True:
            for candidate_name in candidate_names:
                if candidate_name in reserved_names:
                    continue
                target = self.output_root / candidate_name
                if not target.exists():
                    return target
            candidate_names = [f"{round_suffix}_{collision_index}"]
            collision_index += 1

    @staticmethod
    def _verify(
        target: Path,
        expected_record: dict[str, Any],
        expected_events: list[dict[str, Any]],
    ) -> None:
        record = json.loads((target / "record.json").read_text(encoding="utf-8"))
        event_lines = (target / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        events = [json.loads(line) for line in event_lines]
        if record != expected_record or events != expected_events:
            raise ValueError("round replay verification failed")
