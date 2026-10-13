from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .errors import SnapshotError
from .hand_solver import is_winning_hand
from .models import (
    Action,
    ActionKind,
    CandidateAction,
    Meld,
    Observation,
    Phase,
    RuleConfig,
    normalize_melds,
)
from .tiles import is_tile_code

_PASS_CANDIDATE = CandidateAction(Action(ActionKind.PASS))
_PASS_CANDIDATES = (_PASS_CANDIDATE,)


class GangKind(StrEnum):
    MING = "gang_ming"
    AN = "gang_an"
    BU = "gang_bu"


class IllegalActionError(ValueError):
    """Raised when an action is not an exact member of the legal action set."""


class ChainStateError(ValueError):
    """Raised when authoritative chain history cannot explain an observation."""


class DiscardHistoryUnavailable(ChainStateError):
    """The current river does not preserve the previously observed history."""


def effective_baotou(
    authoritative_baotou: bool,
    hand: Sequence[str],
) -> bool:
    """Validate and preserve authoritative baotou, including four whites (v21)."""

    if type(authoritative_baotou) is not bool:
        raise ValueError("authoritative baotou must be a boolean")
    try:
        normalized_hand = tuple(hand)
    except TypeError:
        raise ValueError("baotou hand must be a tile sequence") from None
    if any(not is_tile_code(tile) for tile in normalized_hand):
        raise ValueError("baotou hand must contain valid tile codes")
    if any(count > 4 for count in Counter(normalized_hand).values()):
        raise ValueError("baotou hand cannot contain more than four copies")
    return authoritative_baotou


@dataclass(frozen=True, slots=True)
class SuccessorState:
    hand: tuple[str, ...]
    melds: tuple[Meld, ...]
    wall_remaining: int
    chain_count: int
    piao_count: int
    baotou: bool
    continuation_discard: Action | None = None

    def __post_init__(self) -> None:
        try:
            hand = tuple(self.hand)
        except TypeError:
            raise ValueError("successor hand must be a sequence") from None
        if any(not is_tile_code(tile) for tile in hand):
            raise ValueError("successor hand must contain real tile codes")
        melds = normalize_melds(self.melds)
        physical_counts = Counter(hand)
        for meld in melds:
            physical_counts.update(meld.tiles)
        if any(count > 4 for count in physical_counts.values()):
            raise ValueError(
                "successor tile inventory cannot contain more than four copies of a face"
            )
        object.__setattr__(self, "hand", hand)
        object.__setattr__(self, "melds", melds)
        if type(self.wall_remaining) is not int or self.wall_remaining < 0:
            raise ValueError("successor wall_remaining must be nonnegative")
        if type(self.chain_count) is not int or not 0 <= self.chain_count <= 6:
            raise ValueError("successor chain_count must be between zero and six")
        if (
            type(self.piao_count) is not int
            or not 0 <= self.piao_count <= self.chain_count
        ):
            raise ValueError("successor piao_count must be between zero and chain_count")
        if hand.count("白") + self.piao_count > 4:
            raise ValueError("successor retained whites plus piao cannot exceed four")
        if type(self.baotou) is not bool:
            raise ValueError("successor baotou must be a boolean")
        if (
            self.continuation_discard is not None
            and (
                not isinstance(self.continuation_discard, Action)
                or self.continuation_discard.kind is not ActionKind.DISCARD
            )
        ):
            raise ValueError("successor continuation must be a discard action")


@dataclass(frozen=True, slots=True)
class ActionTransition:
    action: Action
    successors: tuple[SuccessorState, ...]
    gang_kind: GangKind | None = None
    supplement_draw: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise ValueError("transition action must be an Action")
        try:
            successors = tuple(self.successors)
        except TypeError:
            raise ValueError("transition successors must be a sequence") from None
        if not successors or any(
            not isinstance(successor, SuccessorState)
            for successor in successors
        ):
            raise ValueError("transition successors must be nonempty successor states")
        object.__setattr__(self, "successors", successors)
        if self.gang_kind is not None and not isinstance(self.gang_kind, GangKind):
            raise ValueError("transition gang_kind must be a GangKind")
        if type(self.supplement_draw) is not bool:
            raise ValueError("transition supplement_draw must be a boolean")
        if self.action.kind is ActionKind.GANG:
            if self.gang_kind is None or not self.supplement_draw:
                raise ValueError("gang transition requires its kind and supplement draw")
        elif self.gang_kind is not None or self.supplement_draw:
            raise ValueError("non-gang transition cannot have gang metadata")


def can_hu(observation: Observation, config: RuleConfig) -> bool:
    if (
        observation.phase is not Phase.DRAW
        or observation.turn != observation.seat
        or observation.drawn_tile is None
    ):
        return False

    meld_count = len(observation.melds[observation.seat])
    expected_tiles = 3 * (4 - meld_count) + 2
    if meld_count > 4 or len(observation.hand) != expected_tiles:
        return False

    baotou = effective_baotou(observation.god.baotou, observation.hand)
    shape_is_complete = baotou or is_winning_hand(
        observation.hand,
        meld_count,
    )
    if not shape_is_complete:
        return False

    if (
        config.you_cai_bi_kao
        and "白" in observation.hand
        and not baotou
        and observation.god.chain_count == 0
    ):
        return False
    return True


def supported_candidates(
    observation: Observation,
    config: RuleConfig,
) -> tuple[CandidateAction, ...]:
    return legal_actions(observation, config)


def validate_actionable_observation(observation: Observation) -> None:
    """Validate the concealed-hand size at the production action boundary."""

    if observation.phase is Phase.DRAW and observation.turn == observation.seat:
        action_window = "draw"
        shape_remainder = 2
    elif (
        observation.phase in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}
        and observation.seat in observation.responding_seats
    ):
        action_window = "response"
        shape_remainder = 1
    else:
        return

    try:
        own_melds = normalize_melds(observation.melds[observation.seat])
    except (IndexError, ValueError) as error:
        raise SnapshotError(
            f"melds[{observation.seat}] is invalid for an actionable observation: "
            f"{error}"
        ) from error
    expected_tiles = 3 * (4 - len(own_melds)) + shape_remainder
    if len(observation.hand) != expected_tiles:
        raise SnapshotError(
            f"my_hand must contain {expected_tiles} tiles during an actionable "
            f"{action_window} window"
        )


def legal_actions(
    observation: Observation,
    config: RuleConfig,
) -> tuple[CandidateAction, ...]:
    return _legal_actions(observation, config)


def _legal_actions(observation: Observation, config: RuleConfig, *, winning=None) -> tuple[CandidateAction, ...]:
    """Same enumeration with an optional independently validated HU predicate."""
    if observation.phase is Phase.DRAW and observation.turn == observation.seat:
        if not observation.hand:
            raise SnapshotError("my_hand cannot be empty when the player must discard")

        candidates: list[CandidateAction] = []
        baotou = effective_baotou(observation.god.baotou, observation.hand)
        own_melds = observation.melds[observation.seat]
        if can_hu(observation, config) if winning is None else winning:
            candidates.append(CandidateAction(Action(ActionKind.HU)))

        gang_allowed = (
            observation.drawn_tile is not None
            and observation.wall_remaining > 20
            and observation.god.chain_count < 6
        )
        counts = Counter(observation.hand)
        if gang_allowed:
            if len(own_melds) < 4:
                candidates.extend(
                    CandidateAction(Action(ActionKind.GANG, tile=tile))
                    for tile, copies in counts.items()
                    if tile != "白" and copies == 4
                )
            if not observation.god.catch_play:
                concealed_gang_tiles = {
                    candidate.action.tile
                    for candidate in candidates
                    if candidate.action.kind is ActionKind.GANG
                }
                candidates.extend(
                    CandidateAction(Action(ActionKind.GANG, tile=meld.tiles[0]))
                    for meld in own_melds
                    if meld.kind == "peng"
                    and len(meld.tiles) == 3
                    and len(set(meld.tiles)) == 1
                    and meld.tiles[0] != "白"
                    and counts[meld.tiles[0]] >= 1
                    and meld.tiles[0] not in concealed_gang_tiles
                )

        if observation.god.catch_play:
            if observation.drawn_tile is None:
                raise SnapshotError("drawn_tile is required during catch play")
            if not (
                baotou
                and observation.god.chain_count == 6
                and observation.drawn_tile == "白"
            ):
                candidates.append(
                    CandidateAction(
                        Action(ActionKind.DISCARD, tile=observation.drawn_tile),
                        copies=1,
                    )
                )
            return tuple(candidates)

        candidates.extend(
            CandidateAction(Action(ActionKind.DISCARD, tile=tile), copies=copies)
            for tile, copies in counts.items()
            if not (
                baotou
                and observation.god.chain_count == 6
                and tile == "白"
            )
        )
        return tuple(candidates)

    if (
        observation.phase in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}
        and observation.seat in observation.responding_seats
    ):
        if observation.last_discard is None:
            raise SnapshotError("last_discard is required during an eligible response")

        candidates = []
        claimed = observation.last_discard
        if observation.phase is Phase.RESPONSE_PENG and observation.hand.count(claimed) < 2:
            return _PASS_CANDIDATES
        counts = Counter(observation.hand)
        own_melds = observation.melds[observation.seat]
        owned_counts = counts.copy()
        for meld in own_melds:
            owned_counts.update(meld.tiles)
        if observation.phase is Phase.RESPONSE_PENG:
            if (
                claimed != "白"
                and not observation.god.catch_play
                and len(own_melds) < 4
                and owned_counts[claimed] <= 3
            ):
                if counts[claimed] >= 2 and len(observation.hand) > 2:
                    candidates.append(
                        CandidateAction(Action(ActionKind.PENG, tile=claimed))
                    )
                if (
                    counts[claimed] >= 3
                    and observation.wall_remaining > 20
                    and observation.god.chain_count < 6
                    and not observation.god.catch_play
                ):
                    candidates.append(
                        CandidateAction(Action(ActionKind.GANG, tile=claimed))
                    )
        elif (
            claimed != "白"
            and len(claimed) == 2
            and claimed[0] in "123456789"
            and claimed[1] in "wbt"
            and not observation.god.catch_play
            and len(observation.hand) > 2
            and len(own_melds) < 4
            and owned_counts[claimed] <= 3
            and sum(
                meld.kind == "chi"
                for meld in own_melds
            ) < 2
        ):
            rank = int(claimed[0])
            suit = claimed[1]
            for start in range(rank - 2, rank + 1):
                if start < 1 or start + 2 > 9:
                    continue
                consumed = tuple(
                    f"{value}{suit}"
                    for value in range(start, start + 3)
                    if value != rank
                )
                if all(counts[tile] >= consumed.count(tile) for tile in consumed):
                    candidates.append(
                        CandidateAction(
                            Action(ActionKind.CHI, tile=claimed, tiles=consumed)
                        )
                    )
        candidates.append(CandidateAction(Action(ActionKind.PASS)))
        return tuple(candidates)

    return ()


def _remove_tiles(hand: tuple[str, ...], tiles: tuple[str, ...]) -> tuple[str, ...]:
    remaining = list(hand)
    for tile in tiles:
        try:
            remaining.remove(tile)
        except ValueError as error:
            raise IllegalActionError(f"hand does not contain required tile {tile}") from error
    return tuple(remaining)


def _state(
    observation: Observation,
    *,
    hand: tuple[str, ...] | None = None,
    melds: tuple[Meld, ...] | None = None,
    wall_remaining: int | None = None,
    chain_count: int | None = None,
    piao_count: int,
    baotou: bool,
    continuation_discard: Action | None = None,
) -> SuccessorState:
    return SuccessorState(
        hand=observation.hand if hand is None else hand,
        melds=(
            observation.melds[observation.seat]
            if melds is None
            else melds
        ),
        wall_remaining=(
            observation.wall_remaining
            if wall_remaining is None
            else wall_remaining
        ),
        chain_count=(
            observation.god.chain_count
            if chain_count is None
            else chain_count
        ),
        piao_count=piao_count,
        baotou=baotou,
        continuation_discard=continuation_discard,
    )


def _mandatory_discard_successors(
    observation: Observation,
    hand: tuple[str, ...],
    melds: tuple[Meld, ...],
    *,
    piao_count: int,
) -> tuple[SuccessorState, ...]:
    # Follow v26's claim + white continuation using the prior baotou cache.
    # v32 describes refresh at dealt/discarded waits and gang, not chi/peng.
    baotou = effective_baotou(observation.god.baotou, hand)
    return tuple(
        _state(
            observation,
            hand=_remove_tiles(hand, (tile,)),
            melds=melds,
            chain_count=observation.god.chain_count + 1 if baotou and tile == "白" else 0,
            piao_count=piao_count + 1 if baotou and tile == "白" else 0,
            baotou=False,
            continuation_discard=Action(ActionKind.DISCARD, tile=tile),
        )
        for tile in dict.fromkeys(hand)
        if not (baotou and observation.god.chain_count == 6 and tile == "白")
    )


def apply_action(
    observation: Observation,
    action: Action,
    config: RuleConfig,
    *,
    piao_count: int,
) -> ActionTransition:
    if not isinstance(action, Action) or not any(
        candidate.action == action for candidate in legal_actions(observation, config)
    ):
        raise IllegalActionError("action is not an exact member of legal_actions")
    return _apply_legal_action(observation, action, config, piao_count=piao_count)


def _apply_legal_action(
    observation: Observation,
    action: Action,
    config: RuleConfig,
    *,
    piao_count: int,
) -> ActionTransition:
    """Shared transition for an action already generated for this exact state.

    The public entrypoint always validates. Search callers may reuse their
    complete legal-action enumeration instead of generating it a second time.
    """

    own_melds = observation.melds[observation.seat]
    if action.kind in {ActionKind.HU, ActionKind.PASS}:
        return ActionTransition(
            action,
            (
                _state(
                    observation,
                    piao_count=piao_count,
                    baotou=observation.god.baotou,
                ),
            ),
        )

    if action.kind is ActionKind.DISCARD:
        hand = _remove_tiles(observation.hand, (action.tile,))  # type: ignore[arg-type]
        is_piao = (
            effective_baotou(observation.god.baotou, observation.hand)
            and action.tile == "白"
        )
        return ActionTransition(
            action,
            (
                _state(
                    observation,
                    hand=hand,
                    chain_count=(
                        observation.god.chain_count + 1 if is_piao else 0
                    ),
                    piao_count=piao_count + 1 if is_piao else 0,
                    baotou=False,
                ),
            ),
        )

    if action.kind is ActionKind.CHI:
        hand = _remove_tiles(observation.hand, action.tiles)
        meld_tiles = tuple(
            sorted(
                (*action.tiles, action.tile),  # type: ignore[arg-type]
                key=lambda tile: int(tile[0]),
            )
        )
        melds = own_melds + (Meld("chi", meld_tiles),)
        return ActionTransition(
            action,
            _mandatory_discard_successors(observation, hand, melds, piao_count=piao_count),
        )

    if action.kind is ActionKind.PENG:
        claimed = action.tile
        hand = _remove_tiles(observation.hand, (claimed, claimed))  # type: ignore[arg-type]
        melds = own_melds + (Meld("peng", (claimed,) * 3),)  # type: ignore[arg-type]
        return ActionTransition(
            action,
            _mandatory_discard_successors(observation, hand, melds, piao_count=piao_count),
        )

    claimed = action.tile
    if observation.phase is Phase.RESPONSE_PENG:
        gang_kind = GangKind.MING
        consumed = (claimed,) * 3
        hand = _remove_tiles(observation.hand, consumed)  # type: ignore[arg-type]
        melds = own_melds + (Meld(gang_kind.value, (claimed,) * 4),)  # type: ignore[arg-type]
    else:
        peng_index = next(
            (
                index
                for index, meld in enumerate(own_melds)
                if meld.kind == "peng" and meld.tiles == (claimed,) * 3
            ),
            None,
        )
        if peng_index is not None:
            gang_kind = GangKind.BU
            hand = _remove_tiles(observation.hand, (claimed,))  # type: ignore[arg-type]
            mutable_melds = list(own_melds)
            mutable_melds[peng_index] = Meld(gang_kind.value, (claimed,) * 4)  # type: ignore[arg-type]
            melds = tuple(mutable_melds)
        else:
            gang_kind = GangKind.AN
            hand = _remove_tiles(observation.hand, (claimed,) * 4)  # type: ignore[arg-type]
            melds = own_melds + (Meld(gang_kind.value, (claimed,) * 4),)  # type: ignore[arg-type]

    successor = _state(
        observation,
        hand=hand,
        melds=melds,
        wall_remaining=observation.wall_remaining - 1,
        chain_count=observation.god.chain_count + 1,
        piao_count=piao_count,
        baotou=False,
    )
    return ActionTransition(
        action,
        (successor,),
        gang_kind=gang_kind,
        supplement_draw=True,
    )


def _is_peng_to_bu(previous: Meld, current: Meld) -> bool:
    try:
        previous = normalize_melds((previous,))[0]
        current = normalize_melds((current,))[0]
    except ValueError:
        return False
    return (
        previous.kind == "peng"
        and current.kind == GangKind.BU.value
        and len(previous.tiles) == 3
        and len(current.tiles) == 4
        and len(set(current.tiles)) == 1
        and previous.tiles == current.tiles[:3]
    )


def _meld_chain_actions(
    previous: tuple[Meld, ...],
    current: tuple[Meld, ...],
) -> int:
    if len(current) < len(previous):
        raise ChainStateError("own meld history cannot shrink within a round")
    upgrades = 0
    for old, new in zip(previous, current):
        if old == new:
            continue
        if _is_peng_to_bu(old, new):
            upgrades += 1
            continue
        raise ChainStateError("own meld history was reordered or replaced")
    appended = current[len(previous):]
    gang_appends = sum(meld.kind in {kind.value for kind in GangKind} for meld in appended)
    return upgrades + gang_appends


def _canonical_history(observation: Observation) -> tuple[Meld, ...]:
    try:
        melds = normalize_melds(observation.melds[observation.seat])
    except ValueError as error:
        raise ChainStateError(f"invalid own meld history: {error}") from error
    physical_counts = Counter(observation.hand)
    for meld in melds:
        physical_counts.update(meld.tiles)
    if any(count > 4 for count in physical_counts.values()):
        raise ChainStateError(
            "invalid own physical inventory: a tile face exceeds four copies"
        )
    return melds


def _new_own_discards(
    previous: Observation,
    current: Observation,
) -> tuple[str, ...]:
    """Return the new suffix of the authoritative append-only discard history.

    A called tile remains in this history. Physical inventory tracks its move
    separately, so an opponent's new meld cannot justify deleting this record.
    """

    old_river = (previous.public_discard_history or previous.discards)[current.seat]
    new_river = (current.public_discard_history or current.discards)[current.seat]
    if new_river[:len(old_river)] == old_river:
        return new_river[len(old_river):]
    raise DiscardHistoryUnavailable("own discard history must retain its complete prefix")


class ChainStateTracker:
    def __init__(self) -> None:
        self._observation: Observation | None = None
        self._piao_count: int | None = None

    def checkpoint(self):
        return self._observation, self._piao_count

    def restore(self, checkpoint):
        observation, piao = checkpoint
        if observation is None or type(piao) is not int or not 0 <= piao <= observation.god.chain_count:
            raise ChainStateError('invalid chain recovery checkpoint')
        _canonical_history(observation)
        if observation.hand.count('白') + piao > 4:
            raise ChainStateError('invalid chain recovery white count')
        self._observation, self._piao_count = observation, piao

    def observe(self, observation: Observation) -> None:
        previous = self._observation
        if previous is None:
            current_melds = _canonical_history(observation)
            if observation.god.chain_count != 0:
                actions = observation.public_actions
                if (actions is None or actions.reason is not None
                        or actions.chain_counts[observation.seat] != observation.god.chain_count
                        or actions.piao_counts[observation.seat] is None):
                    raise ChainStateError("first observation must begin with chain_count zero")
                self.restore((observation, actions.piao_counts[observation.seat]))
                return
            self._observation = observation
            self._piao_count = 0
            return

        if observation.seat != previous.seat:
            raise ChainStateError("player seat cannot change within one game")
        if observation.round_no < previous.round_no:
            raise ChainStateError("round number cannot move backwards")

        current_melds = _canonical_history(observation)
        previous_melds = _canonical_history(previous)
        if observation.round_no > previous.round_no:
            if observation.god.chain_count != 0:
                raise ChainStateError("a new round must begin with chain_count zero")
            self._observation = observation
            self._piao_count = 0
            return

        meld_actions = _meld_chain_actions(previous_melds, current_melds)
        appended_river = _new_own_discards(previous, observation)
        if observation.god.chain_count == 0:
            self._observation = observation
            self._piao_count = 0
            return

        old_chain = previous.god.chain_count
        new_chain = observation.god.chain_count
        if old_chain > new_chain:
            raise ChainStateError("a positive chain cannot decrease")

        if any(tile != "白" for tile in appended_river):
            raise ChainStateError("a retained positive chain cannot append a non-white discard")
        new_piao = appended_river.count("白")

        if new_chain - old_chain != new_piao + meld_actions:
            raise ChainStateError("chain delta is not explained by piao and gang actions")

        previous_piao = self._piao_count if old_chain > 0 else 0
        assert previous_piao is not None
        piao_count = previous_piao + new_piao
        if piao_count > new_chain:
            raise ChainStateError("piao_count cannot exceed chain_count")
        retained_whites = observation.hand.count("白")
        if retained_whites + piao_count > 4:
            raise ChainStateError("retained whites and current-chain piao exceed four")

        self._observation = observation
        self._piao_count = piao_count

    def piao_count_for(self, observation: Observation) -> int:
        if self._observation is None or self._piao_count is None:
            raise ChainStateError("chain tracker has not been initialized")
        if observation != self._observation:
            raise ChainStateError("observation is not the most recently confirmed state")
        return self._piao_count
