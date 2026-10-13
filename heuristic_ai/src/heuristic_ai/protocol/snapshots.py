from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..game.errors import SnapshotError
from ..game.models import GodState, Meld, Observation, Phase, normalize_melds
from ..game.public_inventory import inventory_from_hand_counts
from ..game.tiles import is_tile_code


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SnapshotError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise SnapshotError(f"{field} must be an array")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotError(f"{field} must be an integer")
    return value


def _bounded_integer(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    result = _integer(value, field)
    if result < minimum or (maximum is not None and result > maximum):
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise SnapshotError(f"{field} must be at least {minimum}{upper}")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise SnapshotError(f"{field} must be a boolean")
    return value


def _optional_tile(value: Any, field: str) -> str | None:
    if value is None or value == "":
        return None
    if not is_tile_code(value):
        raise SnapshotError(f"{field} must be a tile string or null")
    return value


def _tiles(value: Any, field: str) -> tuple[str, ...]:
    values = _sequence(value, field)
    if not all(is_tile_code(item) for item in values):
        raise SnapshotError(f"{field} must contain tile strings")
    return tuple(values)


def normalize_snapshot(snapshot: Mapping[str, Any]) -> Observation:
    source = _mapping(snapshot, "snapshot")
    try:
        phase = Phase(source.get("phase"))
    except ValueError as error:
        raise SnapshotError("phase is not supported") from error

    scores_raw = _sequence(source.get("scores"), "scores")
    if len(scores_raw) != 4:
        raise SnapshotError("scores must contain four values")
    scores = tuple(_integer(item, f"scores[{index}]") for index, item in enumerate(scores_raw))

    discards_raw = _sequence(source.get("discards"), "discards")
    if len(discards_raw) != 4:
        raise SnapshotError("discards must contain four seats")
    discards = tuple(
        _tiles(items, f"discards[{index}]")
        for index, items in enumerate(discards_raw)
    )

    melds_raw = _sequence(source.get("melds"), "melds")
    if len(melds_raw) != 4:
        raise SnapshotError("melds must contain four seats")
    meld_seats: list[tuple[Meld, ...]] = []
    for seat_index, seat_melds in enumerate(melds_raw):
        parsed_melds: list[Meld] = []
        for meld_index, raw_meld in enumerate(_sequence(seat_melds, f"melds[{seat_index}]")):
            item = _mapping(raw_meld, f"melds[{seat_index}][{meld_index}]")
            kind = item.get("kind")
            if not isinstance(kind, str) or not kind:
                raise SnapshotError(f"melds[{seat_index}][{meld_index}].kind must be a string")
            parsed_melds.append(
                Meld(kind, _tiles(item.get("tiles"), f"melds[{seat_index}][{meld_index}].tiles"))
            )
        try:
            normalized_melds = normalize_melds(parsed_melds)
        except ValueError as error:
            raise SnapshotError(f"melds[{seat_index}] is invalid: {error}") from error
        meld_counts = Counter(
            tile
            for meld in normalized_melds
            for tile in meld.tiles
        )
        if any(count > 4 for count in meld_counts.values()):
            raise SnapshotError(
                f"melds[{seat_index}] cannot contain more than four copies "
                "of a tile"
            )
        meld_seats.append(normalized_melds)

    god_raw = _mapping(source.get("god"), "god")
    seat = _bounded_integer(source.get("seat"), "seat", minimum=0, maximum=3)
    # The server field means "a catch-play circle is active". GodState.catch_play
    # means "this seat is restricted by it", which is what the rules need, so the
    # exemption is applied here at the only place the server value enters the
    # model (v26 exempts the god discarder). An explicit null is treated like the
    # documented -1 (no circle) and like a pre-v26 snapshot that omits the key,
    # so neither can turn a live snapshot into a parse error.
    discarder_raw = god_raw.get("god_discarder_seat")
    discarder_seat = (
        -1
        if discarder_raw is None
        else _bounded_integer(
            discarder_raw,
            "god.god_discarder_seat",
            minimum=-1,
            maximum=3,
        )
    )
    god = GodState(
        baotou=_boolean(god_raw.get("baotou"), "god.baotou"),
        chain_count=_bounded_integer(
            god_raw.get("chain_count"),
            "god.chain_count",
            minimum=0,
            maximum=6,
        ),
        catch_play=(
            _boolean(god_raw.get("catch_play"), "god.catch_play")
            and discarder_seat != seat
        ),
    )
    drawn_tile = _optional_tile(source.get("drawn_tile"), "drawn_tile")
    turn = _bounded_integer(source.get("turn"), "turn", minimum=-1, maximum=3)
    if phase in {Phase.DRAW, Phase.RESPONSE_PENG, Phase.RESPONSE_CHI} and turn < 0:
        raise SnapshotError("turn must identify a seat during an action phase")
    if phase is Phase.DRAW and turn == seat and god.catch_play and drawn_tile is None:
        raise SnapshotError("drawn_tile is required during catch play")

    responding_value = source.get("responding_seats")
    if responding_value is None and phase not in {
        Phase.RESPONSE_PENG,
        Phase.RESPONSE_CHI,
    }:
        responding_raw: Sequence[Any] = ()
    else:
        responding_raw = _sequence(responding_value, "responding_seats")
    responding = tuple(
        _bounded_integer(
            item,
            f"responding_seats[{index}]",
            minimum=0,
            maximum=3,
        )
        for index, item in enumerate(responding_raw)
    )
    if len(set(responding)) != len(responding):
        raise SnapshotError("responding_seats must be unique")

    try:
        last_discard = _optional_tile(source.get("last_discard"), "last_discard")
    except SnapshotError:
        if phase not in {Phase.DEAL, Phase.DRAW} or any(discards):
            raise
        last_discard = None

    if (
        phase in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}
        and seat in responding
        and last_discard is None
    ):
        raise SnapshotError("last_discard is required during an eligible response")

    hand = _tiles(source.get("my_hand"), "my_hand")
    if any(count > 4 for count in Counter(hand).values()):
        raise SnapshotError("my_hand cannot contain more than four copies of a tile")
    owned_counts = Counter(hand)
    for meld in meld_seats[seat]:
        owned_counts.update(meld.tiles)
    if any(count > 4 for count in owned_counts.values()):
        raise SnapshotError(
            f"my_hand and melds[{seat}] cannot contain more than four copies "
            "of a tile"
        )
    if drawn_tile is not None and drawn_tile not in hand:
        raise SnapshotError("drawn_tile must be present in my_hand")

    observation = Observation(
        round_no=_bounded_integer(source.get("round_no"), "round_no", minimum=1),
        seat=seat,
        phase=phase,
        dealer=_bounded_integer(
            source.get("dealer"),
            "dealer",
            minimum=0,
            maximum=3,
        ),
        turn=turn,
        scores=scores,  # type: ignore[arg-type]
        wall_remaining=_bounded_integer(
            source.get("wall_remaining"),
            "wall_remaining",
            minimum=0,
        ),
        hand=hand,
        drawn_tile=drawn_tile,
        discards=discards,
        melds=tuple(meld_seats),
        last_discard=last_discard,
        responding_seats=responding,
        god=god,
    )
    if 'hand_counts' in source:
        observation = replace(observation, public_inventory=inventory_from_hand_counts(
            observation, source['hand_counts']))
    return observation
