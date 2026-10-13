from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Sequence

from .models import Meld, normalize_melds
from .tiles import is_tile_code


_SUITS = ("w", "b", "t")
_HONORS = ("东", "南", "西", "北", "中", "发")
_WILDCARD = "白"
_NON_WILDCARD_TILES = tuple(
    f"{rank}{suit}" for suit in _SUITS for rank in range(1, 10)
) + _HONORS
_ALL_TILES = frozenset((*_NON_WILDCARD_TILES, _WILDCARD))
_TILE_ORDER = {
    tile: index for index, tile in enumerate((*_NON_WILDCARD_TILES, _WILDCARD))
}
_ALL_NON_WILDCARD_MASK = (1 << len(_NON_WILDCARD_TILES)) - 1


class RouteKind(StrEnum):
    REGULAR = "regular"
    SEVEN_PAIRS = "seven_pairs"


@dataclass(frozen=True, slots=True)
class RouteComponent:
    """Logical target slots and their physical, wildcard, or missing fills.

    Targets describe a structural witness rather than additional physical tiles.
    A wildcard substitution can therefore make the logical target multiplicity
    exceed four even though physical entities are capped at four at analysis input.
    """

    kind: str
    target_tiles: tuple[str, ...]
    natural_tiles: tuple[str, ...]
    wildcard_tiles: tuple[str, ...]
    missing_tiles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in {
            "pair",
            "triplet",
            "sequence",
        }:
            raise ValueError(
                "route component kind must be pair, triplet, or sequence"
            )
        try:
            target_tiles = tuple(self.target_tiles)
            natural_tiles = tuple(self.natural_tiles)
            wildcard_tiles = tuple(self.wildcard_tiles)
            missing_tiles = tuple(self.missing_tiles)
        except TypeError:
            raise ValueError(
                "route component tiles must use non-wildcard tile codes"
            ) from None
        all_tiles = (*target_tiles, *natural_tiles, *wildcard_tiles, *missing_tiles)
        if any(tile == _WILDCARD or not is_tile_code(tile) for tile in all_tiles):
            raise ValueError(
                "route component tiles must use non-wildcard tile codes"
            )

        is_pair = (
            len(target_tiles) == 2 and target_tiles[0] == target_tiles[1]
        )
        is_triplet = (
            len(target_tiles) == 3 and len(set(target_tiles)) == 1
        )
        is_sequence = (
            len(target_tiles) == 3
            and all(len(tile) == 2 and tile[1] in _SUITS for tile in target_tiles)
            and len({tile[1] for tile in target_tiles}) == 1
            and tuple(int(tile[0]) for tile in target_tiles)
            == tuple(range(int(target_tiles[0][0]), int(target_tiles[0][0]) + 3))
        )
        shape_is_valid = {
            "pair": is_pair,
            "triplet": is_triplet,
            "sequence": is_sequence,
        }[self.kind]
        if not shape_is_valid:
            raise ValueError("route component target shape is invalid")
        if Counter((*natural_tiles, *wildcard_tiles, *missing_tiles)) != Counter(
            target_tiles
        ):
            raise ValueError("route component slots must conserve target tiles")

        object.__setattr__(self, "target_tiles", target_tiles)
        object.__setattr__(self, "natural_tiles", natural_tiles)
        object.__setattr__(self, "wildcard_tiles", wildcard_tiles)
        object.__setattr__(self, "missing_tiles", missing_tiles)


@dataclass(frozen=True, slots=True)
class WildcardAssignment:
    wildcard_index: int
    target_tile: str
    component_index: int

    def __post_init__(self) -> None:
        if type(self.wildcard_index) is not int or self.wildcard_index < 0:
            raise ValueError("wildcard index must be a nonnegative integer")
        if self.target_tile == _WILDCARD or not is_tile_code(self.target_tile):
            raise ValueError(
                "wildcard target tile must be a non-wildcard tile code"
            )
        if type(self.component_index) is not int or self.component_index < 0:
            raise ValueError(
                "wildcard component index must be a nonnegative integer"
            )


@dataclass(frozen=True, slots=True)
class RouteState:
    """An exact route witness and its structural improving tile faces.

    ``improving_tiles`` unites faces from every tied optimal decomposition.
    Generated incomplete routes include white, without filtering for physical or
    publicly visible availability.
    """

    kind: RouteKind
    distance: int
    components: tuple[RouteComponent, ...]
    wildcard_assignments: tuple[WildcardAssignment, ...]
    open_meld_compatibility: bool
    config_compatibility: bool
    improving_tiles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RouteKind):
            raise ValueError("route kind must be a RouteKind")
        if type(self.distance) is not int or self.distance < 0:
            raise ValueError("route distance must be a nonnegative integer")
        try:
            components = tuple(self.components)
        except TypeError:
            raise ValueError(
                "route components must contain RouteComponent values"
            ) from None
        if any(not isinstance(component, RouteComponent) for component in components):
            raise ValueError("route components must contain RouteComponent values")
        try:
            assignments = tuple(self.wildcard_assignments)
        except TypeError:
            raise ValueError(
                "wildcard assignments must contain WildcardAssignment values"
            ) from None
        if any(
            not isinstance(assignment, WildcardAssignment)
            for assignment in assignments
        ):
            raise ValueError(
                "wildcard assignments must contain WildcardAssignment values"
            )
        wildcard_slot_count = sum(
            len(component.wildcard_tiles) for component in components
        )
        if len(assignments) > 4 or wildcard_slot_count > 4:
            raise ValueError(
                "route state cannot contain more than four wildcard slots"
            )

        pair_count = sum(component.kind == "pair" for component in components)
        regular_shape = (
            1 <= len(components) <= 5
            and pair_count == 1
            and all(
                component.kind in {"pair", "triplet", "sequence"}
                for component in components
            )
        )
        seven_pairs_shape = len(components) == 7 and pair_count == 7
        if (
            self.kind is RouteKind.REGULAR
            and not regular_shape
            or self.kind is RouteKind.SEVEN_PAIRS
            and not seven_pairs_shape
        ):
            raise ValueError("route components do not match route kind")
        if self.distance != sum(
            len(component.missing_tiles) for component in components
        ):
            raise ValueError("route distance must equal total missing slots")

        wildcard_indexes = {
            assignment.wildcard_index for assignment in assignments
        }
        if wildcard_indexes != set(range(len(assignments))):
            raise ValueError(
                "wildcard indexes must be unique and contiguous from zero"
            )
        if any(
            assignment.component_index >= len(components)
            for assignment in assignments
        ):
            raise ValueError("wildcard assignment component index is out of range")
        assigned_slots = Counter(
            (assignment.component_index, assignment.target_tile)
            for assignment in assignments
        )
        component_slots = Counter(
            (component_index, tile)
            for component_index, component in enumerate(components)
            for tile in component.wildcard_tiles
        )
        if assigned_slots != component_slots:
            raise ValueError(
                "wildcard assignments must match component wildcard slots"
            )
        if (
            type(self.open_meld_compatibility) is not bool
            or type(self.config_compatibility) is not bool
        ):
            raise ValueError("route compatibility fields must be booleans")
        if self.open_meld_compatibility is not (
            self.kind is RouteKind.REGULAR
        ):
            raise ValueError("open meld compatibility must match route kind")
        try:
            improving_tiles = tuple(self.improving_tiles)
        except TypeError:
            raise ValueError(
                "route improving tiles must use valid tile codes"
            ) from None
        if any(not is_tile_code(tile) for tile in improving_tiles):
            raise ValueError("route improving tiles must use valid tile codes")
        if len(set(improving_tiles)) != len(improving_tiles):
            raise ValueError(
                "route improving tiles cannot contain duplicate faces"
            )
        if improving_tiles and self.distance == 0:
            raise ValueError("complete route cannot have improving tiles")
        if improving_tiles and _WILDCARD not in improving_tiles:
            raise ValueError(
                "incomplete route improving tiles must include white"
            )

        object.__setattr__(self, "components", components)
        object.__setattr__(self, "wildcard_assignments", assignments)
        object.__setattr__(
            self,
            "improving_tiles",
            tuple(sorted(improving_tiles, key=_TILE_ORDER.__getitem__)),
        )


_InternalComponent = tuple[str, tuple[int, ...]]
_InternalPlanResult = tuple[int, tuple[_InternalComponent, ...], int]
_SEQUENCE_TARGETS: tuple[tuple[int, int, int], ...] = tuple(
    (suit_start + rank, suit_start + rank + 1, suit_start + rank + 2)
    for suit_start in (0, 9, 18)
    for rank in range(7)
)

# The structural solver only constructs these 87 components. Keep their
# original lexical tile ordering, including its distinction from TILE_ORDER,
# without rebuilding string tuples and scanning all sequences at every node.
_GROUP_TARGETS = tuple(
    (("triplet", (index, index, index)),)
    + tuple(("sequence", target) for target in _SEQUENCE_TARGETS if index in target)
    for index in range(len(_NON_WILDCARD_TILES))
)
_COMPONENT_KEYS = {
    component: (component[0], tuple(_NON_WILDCARD_TILES[index] for index in component[1]))
    for component in (
        *(("pair", (index, index)) for index in range(len(_NON_WILDCARD_TILES))),
        *(component for targets in _GROUP_TARGETS for component in targets),
    )
}
_FIRST_LEXICAL_TILE = min(range(len(_NON_WILDCARD_TILES)), key=_NON_WILDCARD_TILES.__getitem__)
_EMPTY_PAIR = ("pair", (_FIRST_LEXICAL_TILE, _FIRST_LEXICAL_TILE))
_EMPTY_GROUP = min(
    (component for targets in _GROUP_TARGETS for component in targets),
    key=_COMPONENT_KEYS.__getitem__,
)


def _component_key(component: _InternalComponent) -> tuple[str, tuple[str, ...]]:
    return _COMPONENT_KEYS[component]


def _normalize_plan(
    components: tuple[_InternalComponent, ...],
) -> tuple[_InternalComponent, ...]:
    return tuple(sorted(components, key=_component_key))


def _plan_key(
    plan: tuple[_InternalComponent, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(_component_key(component) for component in plan)


def _prefer_plan(
    current: _InternalPlanResult | None,
    candidate: _InternalPlanResult,
) -> _InternalPlanResult:
    if current is None or candidate[0] > current[0]:
        return candidate
    if candidate[0] < current[0]:
        return current
    canonical_plan = min((current[1], candidate[1]), key=_plan_key)
    return current[0], canonical_plan, current[2] | candidate[2]


def _consume_target(
    counts: tuple[int, ...], target: tuple[int, ...]
) -> tuple[tuple[int, ...], int, int]:
    remainder = list(counts)
    consumed = 0
    missing_mask = 0
    for index in target:
        if remainder[index]:
            remainder[index] -= 1
            consumed += 1
        else:
            missing_mask |= 1 << index
    return tuple(remainder), consumed, missing_mask


@lru_cache(maxsize=65_536)
def _best_component_plan(
    counts: tuple[int, ...], groups_left: int, pairs_left: int
) -> _InternalPlanResult:
    if groups_left == 0 and pairs_left == 0:
        return 0, (), 0

    first = next((index for index, count in enumerate(counts) if count), None)
    if first is None:
        components = (_EMPTY_PAIR,) * pairs_left + (_EMPTY_GROUP,) * groups_left
        return (
            0,
            _normalize_plan(components),
            _ALL_NON_WILDCARD_MASK,
        )

    discarded = list(counts)
    discarded[first] -= 1
    best: _InternalPlanResult | None = (
        _best_component_plan(tuple(discarded), groups_left, pairs_left)
    )

    if pairs_left:
        target = (first, first)
        remainder, consumed, missing_mask = _consume_target(counts, target)
        reused, plan, child_mask = _best_component_plan(
            remainder, groups_left, pairs_left - 1
        )
        candidate = (
            consumed + reused,
            _normalize_plan((("pair", target), *plan)),
            missing_mask | child_mask,
        )
        best = _prefer_plan(best, candidate)

    if groups_left:
        for component in _GROUP_TARGETS[first]:
            remainder, consumed, missing_mask = _consume_target(
                counts, component[1]
            )
            reused, plan, child_mask = _best_component_plan(
                remainder, groups_left - 1, pairs_left
            )
            candidate = (
                consumed + reused,
                _normalize_plan((component, *plan)),
                missing_mask | child_mask,
            )
            best = _prefer_plan(best, candidate)

    assert best is not None
    return best


def _build_route(
    kind: RouteKind,
    counts: tuple[int, ...],
    wildcard_count: int,
    plan: tuple[_InternalComponent, ...],
    improving_mask: int,
    *,
    config_compatibility: bool,
) -> RouteState:
    available = list(counts)
    natural_slots: list[list[str]] = [[] for _ in plan]
    unfilled: list[tuple[int, str]] = []
    targets_by_component: list[tuple[str, ...]] = []

    for component_index, (_, targets) in enumerate(plan):
        target_tiles = tuple(_NON_WILDCARD_TILES[index] for index in targets)
        targets_by_component.append(target_tiles)
        for target_index, target_tile in zip(targets, target_tiles, strict=True):
            if available[target_index]:
                available[target_index] -= 1
                natural_slots[component_index].append(target_tile)
            else:
                unfilled.append((component_index, target_tile))

    wildcard_slots: list[list[str]] = [[] for _ in plan]
    assignments: list[WildcardAssignment] = []
    for wildcard_index, (component_index, target_tile) in enumerate(
        unfilled[:wildcard_count]
    ):
        wildcard_slots[component_index].append(target_tile)
        assignments.append(
            WildcardAssignment(wildcard_index, target_tile, component_index)
        )

    missing_by_component: list[list[str]] = [[] for _ in plan]
    for component_index, target_tile in unfilled[wildcard_count:]:
        missing_by_component[component_index].append(target_tile)

    components = tuple(
        RouteComponent(
            component_kind,
            targets_by_component[index],
            tuple(natural_slots[index]),
            tuple(wildcard_slots[index]),
            tuple(missing_by_component[index]),
        )
        for index, (component_kind, _) in enumerate(plan)
    )
    distance = len(unfilled) - wildcard_count
    improving_tiles = ()
    if distance:
        improving_tiles = tuple(
            tile
            for index, tile in enumerate(_NON_WILDCARD_TILES)
            if improving_mask & (1 << index)
        ) + (_WILDCARD,)
    return RouteState(
        kind=kind,
        distance=distance,
        components=components,
        wildcard_assignments=tuple(assignments),
        open_meld_compatibility=kind is RouteKind.REGULAR,
        config_compatibility=config_compatibility,
        improving_tiles=improving_tiles,
    )


@lru_cache(maxsize=2_048)
def _analyze_hand_cached(
    counts: tuple[int, ...],
    wildcard_count: int,
    meld_signature: tuple[tuple[str, tuple[str, ...]], ...],
    you_cai_bi_kao: bool,
    baotou: bool,
    chain_count: int,
) -> tuple[RouteState, ...]:
    config_compatibility = (
        not you_cai_bi_kao or wildcard_count == 0 or baotou or chain_count > 0
    )
    meld_count = len(meld_signature)
    _, regular_plan, regular_improving_mask = _best_component_plan(
        counts, 4 - meld_count, 1
    )
    routes = [
        _build_route(
            RouteKind.REGULAR,
            counts,
            wildcard_count,
            regular_plan,
            regular_improving_mask,
            config_compatibility=config_compatibility,
        )
    ]
    if meld_count == 0:
        _, seven_pairs_plan, seven_pairs_improving_mask = _best_component_plan(
            counts, 0, 7
        )
        routes.append(
            _build_route(
                RouteKind.SEVEN_PAIRS,
                counts,
                wildcard_count,
                seven_pairs_plan,
                seven_pairs_improving_mask,
                config_compatibility=config_compatibility,
            )
        )
    return tuple(routes)


def analyze_hand(
    hand: Sequence[str],
    melds: Sequence[Meld],
    *,
    you_cai_bi_kao: bool,
    baotou: bool,
    chain_count: int,
) -> tuple[RouteState, ...]:
    """Return exact regular and eligible seven-pairs distances and witnesses.

    Component targets are logical structural slots. Wildcard substitutions can
    create target multiplicity beyond four; only physical hand and meld entities
    are subject to the four-copy validation limit.
    """

    try:
        normalized_hand = tuple(hand)
        normalized_melds = tuple(melds)
    except TypeError:
        raise ValueError("hand and melds must be sequences") from None

    normalized_melds = normalize_melds(normalized_melds)
    meld_count = len(normalized_melds)
    if type(you_cai_bi_kao) is not bool:
        raise ValueError("you_cai_bi_kao must be a boolean")
    if type(baotou) is not bool:
        raise ValueError("baotou must be a boolean")
    if type(chain_count) is not int or chain_count < 0:
        raise ValueError("chain_count must be a nonnegative integer")
    target_slots = 3 * (4 - meld_count) + 2
    if len(normalized_hand) not in {target_slots, target_slots - 1}:
        raise ValueError(
            f"hand length must be {target_slots} or {target_slots - 1} "
            f"for {meld_count} melds"
        )

    if any(not is_tile_code(tile) for tile in normalized_hand):
        raise ValueError("hand and meld tiles must use valid tile codes")
    meld_signature = tuple(
        sorted((meld.kind, meld.tiles) for meld in normalized_melds)
    )
    meld_tiles = tuple(tile for _, tiles in meld_signature for tile in tiles)
    physical_counts = Counter((*normalized_hand, *meld_tiles))
    if any(count > 4 for count in physical_counts.values()):
        raise ValueError("physical tile count cannot exceed four copies")

    hand_counts = Counter(normalized_hand)
    wildcard_count = hand_counts[_WILDCARD]
    counts = tuple(hand_counts[tile] for tile in _NON_WILDCARD_TILES)
    return _analyze_hand_cached(
        counts,
        wildcard_count,
        meld_signature,
        you_cai_bi_kao,
        baotou,
        chain_count,
    )


@lru_cache(maxsize=65_536)
def _can_form_groups(
    counts: tuple[int, ...],
    wildcards: int,
    groups_needed: int,
) -> bool:
    if sum(counts) + wildcards != groups_needed * 3:
        return False
    if groups_needed == 0:
        return wildcards == 0

    try:
        first = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return wildcards == groups_needed * 3

    triplet_counts = list(counts)
    triplet_size = min(3, triplet_counts[first])
    triplet_missing = 3 - triplet_size
    if triplet_missing <= wildcards:
        triplet_counts[first] -= triplet_size
        if _can_form_groups(
            tuple(triplet_counts),
            wildcards - triplet_missing,
            groups_needed - 1,
        ):
            return True

    if first >= 27:
        return False

    suit_start = first - (first % 9)
    rank = first % 9
    for sequence_start in range(max(0, rank - 2), min(rank, 6) + 1):
        sequence_counts = list(counts)
        sequence_missing = 0
        for offset in range(3):
            index = suit_start + sequence_start + offset
            if sequence_counts[index]:
                sequence_counts[index] -= 1
            else:
                sequence_missing += 1
        if sequence_missing > wildcards:
            continue
        if _can_form_groups(
            tuple(sequence_counts),
            wildcards - sequence_missing,
            groups_needed - 1,
        ):
            return True

    return False


def _is_regular_win(
    counts: tuple[int, ...],
    wildcards: int,
    groups_needed: int,
) -> bool:
    for index, count in enumerate(counts):
        if count >= 2:
            remainder = list(counts)
            remainder[index] -= 2
            if _can_form_groups(tuple(remainder), wildcards, groups_needed):
                return True
        if count >= 1 and wildcards >= 1:
            remainder = list(counts)
            remainder[index] -= 1
            if _can_form_groups(tuple(remainder), wildcards - 1, groups_needed):
                return True

    return wildcards >= 2 and _can_form_groups(
        counts,
        wildcards - 2,
        groups_needed,
    )


def _is_seven_pairs(counts: tuple[int, ...], wildcards: int) -> bool:
    singles = sum(count % 2 for count in counts)
    if singles > wildcards:
        return False
    remaining_wildcards = wildcards - singles
    if remaining_wildcards % 2:
        return False
    pairs = sum(count // 2 for count in counts)
    return pairs + singles + remaining_wildcards // 2 == 7


def is_winning_hand(hand: Sequence[str], meld_count: int) -> bool:
    """Return whether a concealed hand completes a supported winning shape."""

    if meld_count < 0 or meld_count > 4:
        return False
    expected_tiles = 3 * (4 - meld_count) + 2
    if len(hand) != expected_tiles:
        return False

    physical_counts = Counter(hand)
    if any(tile not in _ALL_TILES for tile in physical_counts):
        return False
    if any(count > 4 for count in physical_counts.values()):
        return False

    wildcards = physical_counts.pop(_WILDCARD, 0)
    counts = tuple(physical_counts.get(tile, 0) for tile in _NON_WILDCARD_TILES)
    if meld_count == 0 and _is_seven_pairs(counts, wildcards):
        return True
    return _is_regular_win(counts, wildcards, 4 - meld_count)
