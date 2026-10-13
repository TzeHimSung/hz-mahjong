"""Witness-free structure and option evaluation for the One-Ply Expectation policy.

The semantics mirror ``hand_solver`` and the internal static action evaluation,
but this module owns its optimization and caches.  The shared hand solver and static evaluator
remain the independent reference and fallback.  The regular-hand dynamic
program maximizes naturally reused tiles and unions missing faces across *all*
tied optima.  Suits are independent, so bounded local profiles can be reused
between the thousands of adjacent hands in a one-ply decision.

``summarize_hand`` / ``evaluate_state`` are validated public adapters.  The
counts variants are trusted internal hot paths: the caller has already checked
the 34 physical counts, meld inventory, hand size, rules, and chain history.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

from heuristic_ai.game.hand_solver import RouteKind
from heuristic_ai.game.models import (
    Action,
    ActionKind,
    GodState,
    Meld,
    Observation,
    RuleConfig,
    is_tile_code,
    normalize_melds,
)
from heuristic_ai.game.rules import SuccessorState, can_hu, effective_baotou
from heuristic_ai.game.scoring import (
    ScoreBreakdown,
    ScoreValueKind,
    _build_breakdown,
    _validate_chain,
    _validate_config,
    _validate_seats,
    luxury_quad_count_counts,
)
from heuristic_ai.game.tiles import TILE_INDEX, TILE_ORDER


ALL_TILE_MASK = (1 << len(TILE_ORDER)) - 1
_NATURAL_MASK = (1 << (len(TILE_ORDER) - 1)) - 1
_WHITE_BIT = 1 << (len(TILE_ORDER) - 1)
_LOCAL_MASK = (1 << 9) - 1
_PAIR_MASK = (1 << 18) - 1
_ACTION_PRIORITY = {
    ActionKind.HU: 6,
    ActionKind.PASS: 5,
    ActionKind.DISCARD: 4,
    ActionKind.PENG: 3,
    ActionKind.CHI: 2,
    ActionKind.GANG: 1,
}
ActionStableKey = tuple[int, int, tuple[int, ...]]
DecisionKey = tuple[int, int, int, int, int, int, int, ActionStableKey]


@dataclass(frozen=True, slots=True)
class RouteSummary:
    kind: RouteKind
    distance: int
    improving_mask: int
    config_compatibility: bool

    @property
    def improving_tiles(self) -> tuple[str, ...]:
        return tuple(
            tile for index, tile in enumerate(TILE_ORDER)
            if self.improving_mask & (1 << index)
        )

    def to_details(self) -> dict[str, object]:
        return {
            "route": self.kind.value,
            "distance": self.distance,
            "improving_tiles": self.improving_tiles,
            "open_meld_compatibility": self.kind is RouteKind.REGULAR,
            "config_compatibility": self.config_compatibility,
        }


def stable_action_key(action: Action) -> ActionStableKey:
    """The fixed stable action ordering shared by One-Ply Expectation and its static evaluation."""

    return (
        _ACTION_PRIORITY[action.kind],
        -TILE_INDEX[action.tile] if action.tile is not None else 0,
        tuple(-TILE_INDEX[tile] for tile in action.tiles),
    )


def _prefer(left: int, right: int) -> int:
    left_reused = left >> 9
    right_reused = right >> 9
    if right_reused > left_reused:
        return right
    if right_reused == left_reused:
        return left | right
    return left


def _component_compute(
    code: int, groups: int, pairs: int, suited: bool, recurse
) -> int:
    """Pack maximum reuse above nine bits and optimal missing faces below."""

    if groups == 0 and pairs == 0:
        return 0
    if code == 0:
        return _LOCAL_MASK if suited else (1 << 6) - 1
    first = ((code & -code).bit_length() - 1) // 3
    shift = 3 * first
    unit = 1 << shift
    count = (code >> shift) & 7
    best = recurse(code - unit, groups, pairs, suited)
    if pairs:
        used = min(count, 2)
        child = recurse(code - used * unit, groups, pairs - 1, suited)
        candidate = child + (used << 9)
        if used < 2:
            candidate |= 1 << first
        best = _prefer(best, candidate)
    if groups:
        used = min(count, 3)
        child = recurse(code - used * unit, groups - 1, pairs, suited)
        candidate = child + (used << 9)
        if used < 3:
            candidate |= 1 << first
        best = _prefer(best, candidate)
        if suited:
            for start in range(max(0, first - 2), min(first, 6) + 1):
                remainder = code
                used = 0
                missing = 0
                for index in (start, start + 1, start + 2):
                    if (code >> (3 * index)) & 7:
                        remainder -= 1 << (3 * index)
                        used += 1
                    else:
                        missing |= 1 << index
                child = recurse(remainder, groups - 1, pairs, suited)
                candidate = (child + (used << 9)) | missing
                best = _prefer(best, candidate)
    return best


@lru_cache(maxsize=131_072)
def _component_cached(code: int, groups: int, pairs: int, suited: bool) -> int:
    return _component_compute(code, groups, pairs, suited, _component_cached)


def _profile_compute(code: int, suited: bool, component) -> tuple[int, ...]:
    return tuple(
        component(code, groups, pairs, suited)
        for groups in range(5) for pairs in range(2)
    )


@lru_cache(maxsize=16_384)
def _profile_cached(code: int, suited: bool) -> tuple[int, ...]:
    return _profile_compute(code, suited, _component_cached)


def _paired_profile_compute(
    left_code: int, right_code: int, right_suited: bool, profile,
) -> tuple[int, ...]:
    """Allocate up to four groups and one pair between two tile categories.

    Adjacent search hands usually leave one half unchanged. Caching these
    paired profiles avoids redoing a complete four-category convolution for
    every two-discard / one-draw combination. Local masks occupy eighteen bits.
    """

    left = profile(left_code, True)
    right = profile(right_code, right_suited)
    result: list[int] = []
    for groups in range(5):
        for pairs in range(2):
            best_reuse = -1
            best_mask = 0
            for left_groups in range(groups + 1):
                for left_pairs in range(pairs + 1):
                    first = left[2 * left_groups + left_pairs]
                    second = right[2 * (groups - left_groups) + pairs - left_pairs]
                    reuse = (first >> 9) + (second >> 9)
                    mask = (first & _LOCAL_MASK) | ((second & _LOCAL_MASK) << 9)
                    if reuse > best_reuse:
                        best_reuse, best_mask = reuse, mask
                    elif reuse == best_reuse:
                        best_mask |= mask
            result.append((best_reuse << 18) | best_mask)
    return tuple(result)


@lru_cache(maxsize=32_768)
def _paired_profile_cached(
    left_code: int, right_code: int, right_suited: bool,
) -> tuple[int, ...]:
    return _paired_profile_compute(left_code, right_code, right_suited, _profile_cached)


def _regular_compute(counts: tuple[int, ...], groups: int, paired_profile) -> tuple[int, int]:
    codes = tuple(
        sum(count << (3 * index) for index, count in enumerate(counts[start:stop]))
        for start, stop in ((0, 9), (9, 18), (18, 27), (27, 33))
    )
    left = paired_profile(codes[0], codes[1], True)
    right = paired_profile(codes[2], codes[3], False)
    best_reuse = -1
    best_mask = 0
    # Only the final target (groups, one pair) is needed, so this merge has at
    # most ten alternatives instead of computing every intermediate budget.
    for left_groups in range(groups + 1):
        for left_pairs in range(2):
            first = left[2 * left_groups + left_pairs]
            second = right[2 * (groups - left_groups) + 1 - left_pairs]
            reuse = (first >> 18) + (second >> 18)
            mask = (first & _PAIR_MASK) | ((second & _PAIR_MASK) << 18)
            if reuse > best_reuse:
                best_reuse, best_mask = reuse, mask
            elif reuse == best_reuse:
                best_mask |= mask
    return best_reuse, best_mask


@lru_cache(maxsize=32_768)
def _regular_cached(counts: tuple[int, ...], groups: int) -> tuple[int, int]:
    return _regular_compute(counts, groups, _paired_profile_cached)


def _seven_pairs(counts: tuple[int, ...]) -> tuple[int, int]:
    pairs = sum(count // 2 for count in counts)
    odd_mask = 0
    odd_count = 0
    for index, count in enumerate(counts):
        if count & 1:
            odd_mask |= 1 << index
            odd_count += 1
    remaining = 7 - pairs
    reused = 2 * pairs + min(remaining, odd_count)
    missing_mask = _NATURAL_MASK if remaining > odd_count else odd_mask
    return reused, missing_mask


def _known_planes(known: tuple[int, ...]) -> tuple[int, int, int]:
    """Three bit planes represent the visible count capped at four per face."""

    ones = twos = fours = 0
    for index, count in enumerate(known):
        bit = 1 << index
        if count >= 4:
            fours |= bit
        else:
            if count & 1:
                ones |= bit
            if count & 2:
                twos |= bit
    return ones, twos, fours


@lru_cache(maxsize=4_096)
def _known_planes_cached(known: tuple[int, ...]) -> tuple[int, int, int]:
    return _known_planes(known)


def _summaries_compute(
    counts: tuple[int, ...], meld_count: int, compatible: bool, regular_solver,
) -> tuple[RouteSummary, ...]:
    natural = counts[:33]
    whites = counts[33]
    groups = 4 - meld_count
    regular = regular_solver(natural, groups)
    distance = 3 * groups + 2 - regular[0] - whites
    routes = [RouteSummary(
        RouteKind.REGULAR,
        distance,
        (regular[1] | _WHITE_BIT) if distance else 0,
        compatible,
    )]
    if meld_count == 0:
        reused, mask = _seven_pairs(natural)
        distance = 14 - reused - whites
        routes.append(RouteSummary(
            RouteKind.SEVEN_PAIRS,
            distance,
            (mask | _WHITE_BIT) if distance else 0,
            compatible,
        ))
    return tuple(routes)


@lru_cache(maxsize=65_536)
def _summaries_cached(
    counts: tuple[int, ...], meld_count: int, compatible: bool,
) -> tuple[RouteSummary, ...]:
    return _summaries_compute(counts, meld_count, compatible, _regular_cached)


def summarize_counts(
    counts: tuple[int, ...],
    meld_count: int,
    *,
    you_cai_bi_kao: bool,
    baotou: bool,
    chain_count: int,
    cache_enabled: bool = True,
) -> tuple[RouteSummary, ...]:
    """Analyze trusted physical counts without constructing decomposition witnesses.

    Like ``analyze_hand``, this function takes baotou as supplied.  Guide v21
    permits authoritative baotou with four retained whites as well.
    The compatibility boolean is the complete effect of rules/baotou/chains
    on structural summaries, so no other game state belongs in these caches.
    With shared caching disabled, bounded per-call memoization is still part
    of the dynamic program and is released when the call returns.
    """

    compatible = not you_cai_bi_kao or counts[33] == 0 or baotou or chain_count > 0
    if cache_enabled:
        return _summaries_cached(counts, meld_count, compatible)

    @lru_cache(maxsize=16_384)
    def component(code: int, groups: int, pairs: int, suited: bool) -> int:
        return _component_compute(code, groups, pairs, suited, component)

    def profile(code: int, suited: bool) -> tuple[int, ...]:
        return _profile_compute(code, suited, component)

    def regular(natural: tuple[int, ...], groups: int) -> tuple[int, int]:
        def paired(left_code: int, right_code: int, right_suited: bool) -> tuple[int, ...]:
            return _paired_profile_compute(left_code, right_code, right_suited, profile)

        return _regular_compute(natural, groups, paired)

    result = _summaries_compute(counts, meld_count, compatible, regular)
    component.cache_clear()
    return result


def _validated_counts(hand: Sequence[str], melds: Sequence[Meld]) -> tuple[tuple[int, ...], int]:
    try:
        hand = tuple(hand)
        melds = normalize_melds(tuple(melds))
    except TypeError:
        raise ValueError("hand and melds must be sequences") from None
    target = 3 * (4 - len(melds)) + 2
    if len(hand) not in {target, target - 1}:
        raise ValueError(f"hand length must be {target} or {target - 1} for {len(melds)} melds")
    if any(not is_tile_code(tile) for tile in hand):
        raise ValueError("hand and meld tiles must use valid tile codes")
    hand_counts = Counter(hand)
    physical = hand_counts.copy()
    physical.update(tile for meld in melds for tile in meld.tiles)
    if any(count > 4 for count in physical.values()):
        raise ValueError("physical tile count cannot exceed four copies")
    return tuple(hand_counts[tile] for tile in TILE_ORDER), len(melds)


def summarize_hand(
    hand: Sequence[str],
    melds: Sequence[Meld],
    *,
    you_cai_bi_kao: bool,
    baotou: bool,
    chain_count: int,
    cache_enabled: bool = True,
) -> tuple[RouteSummary, ...]:
    if type(you_cai_bi_kao) is not bool:
        raise ValueError("you_cai_bi_kao must be a boolean")
    if type(baotou) is not bool:
        raise ValueError("baotou must be a boolean")
    if type(chain_count) is not int or chain_count < 0:
        raise ValueError("chain_count must be a nonnegative integer")
    if type(cache_enabled) is not bool:
        raise ValueError("cache_enabled must be a boolean")
    counts, meld_count = _validated_counts(hand, melds)
    return summarize_counts(
        counts, meld_count, you_cai_bi_kao=you_cai_bi_kao,
        baotou=baotou, chain_count=chain_count, cache_enabled=cache_enabled,
    )


def score_drawn_terminal(
    observation: Observation,
    rules: RuleConfig,
    *,
    piao_count: int,
    cache_enabled: bool = True,
) -> ScoreBreakdown:
    """Exactly score a hypothetical drawn win without building route witnesses.

    This preserves the input checks, ``can_hu`` gate, compatible complete-route
    selection, and factor builder of ``scoring.score_terminal``. Only route
    enumeration uses One-Ply Expectation's independently differential-tested summaries. The
    existing factor builder reads route.kind and physical hand multiplicities;
    component witnesses never contribute to terminal factors or payments.
    """

    if not isinstance(observation, Observation):
        raise ValueError("terminal score observation must be an Observation")
    _validate_config(rules)
    _validate_seats(observation.seat, observation.dealer)
    if not isinstance(observation.god, GodState):
        raise ValueError("terminal score god state must be a GodState")
    if type(observation.god.baotou) is not bool:
        raise ValueError("terminal score baotou must be a boolean")
    if type(cache_enabled) is not bool:
        raise ValueError("cache_enabled must be a boolean")
    try:
        white_count = tuple(observation.hand).count("白")
        own_melds = observation.melds[observation.seat]
    except (IndexError, TypeError):
        raise ValueError("terminal score observation is malformed") from None
    _validate_chain(observation.god.chain_count, piao_count, white_count)
    baotou = effective_baotou(observation.god.baotou, observation.hand)
    try:
        legal_hu = can_hu(observation, rules)
    except (AttributeError, IndexError, TypeError):
        raise ValueError("terminal score observation is malformed") from None
    if not legal_hu:
        raise ValueError("terminal score requires a legal hu observation")
    try:
        routes = summarize_hand(
            observation.hand, own_melds,
            you_cai_bi_kao=rules.you_cai_bi_kao, baotou=baotou,
            chain_count=observation.god.chain_count, cache_enabled=cache_enabled,
        )
    except (AttributeError, IndexError, TypeError):
        raise ValueError("terminal score observation is malformed") from None
    candidates = tuple(
        _build_breakdown(
            value_kind=ScoreValueKind.TERMINAL,
            route=route,  # The shared builder requires only route.kind.
            hand=observation.hand,
            chain_count=observation.god.chain_count,
            piao_count=piao_count,
            baotou=baotou,
            seat=observation.seat,
            dealer=observation.dealer,
            base_score=rules.base_score,
        )
        for route in routes
        if route.distance == 0 and route.config_compatibility
        and (route.kind is RouteKind.REGULAR or observation.god.chain_count == piao_count)
    )
    if not candidates:
        raise ValueError("terminal score requires a compliant complete route")
    return max(candidates, key=lambda score: (
        score.normalized_winner_value, int(score.route is RouteKind.SEVEN_PAIRS),
    ))


class CompactEvaluation(NamedTuple):
    """Immutable cheap-to-construct leaf value with lazily built audit detail."""

    minimum_distance: int
    route_count: int
    live_tile_weight: int
    fan_option_value: int
    chain_option_value: int
    decision_key: DecisionKey
    routes: tuple[RouteSummary, ...]
    known: tuple[int, ...]
    draw_available: bool
    option_route: RouteKind | None
    derived_baotou: bool
    white_count: int
    quad_count: int
    chain_count: int
    piao_count: int
    seat: int
    dealer: int
    base_score: int

    @property
    def score_breakdown(self) -> ScoreBreakdown | None:
        """Build the validated rich value only for selected audited leaves."""

        if self.option_route is None:
            return None
        seven_pairs = self.option_route is RouteKind.SEVEN_PAIRS
        luxury = self.quad_count if seven_pairs else 0
        branch = 2 * 2**luxury if seven_pairs else 1
        chain = 2**self.chain_count
        four_white = 2 if self.white_count + self.piao_count == 4 else 1
        baotou = 2 if self.derived_baotou else 1
        fan = branch * chain * four_white * baotou
        deltas = [0] * 4
        for payer in range(4):
            if payer != self.seat:
                multiplier = 8 if self.seat == self.dealer or payer == self.dealer else 1
                deltas[payer] = -self.base_score * fan * multiplier
        deltas[self.seat] = -sum(deltas)
        return ScoreBreakdown(
            value_kind=ScoreValueKind.OPTION,
            route=self.option_route,
            branch_factor=branch,
            luxury_quad_count=luxury,
            chain_count=self.chain_count,
            piao_count=self.piao_count,
            chain_factor=chain,
            hand_white_count=self.white_count,
            four_white_factor=four_white,
            baotou_factor=baotou,
            dealer=self.seat == self.dealer,
            base_score=self.base_score,
            fan=fan,
            score_deltas=tuple(deltas),
            normalized_winner_value=(24 if self.seat == self.dealer else 10) * fan,
        )

    def to_details(self) -> dict[str, object]:
        frontier = tuple(route for route in self.routes if route.distance == self.minimum_distance)
        mask = 0
        for route in frontier:
            mask |= route.improving_mask
        live = tuple({
            "tile": tile,
            "visible_remaining": 4 - self.known[index],
            "weight": 4 - self.known[index] if self.draw_available else 0,
            "improvements": tuple({
                "route": route.kind.value,
                "distance_before": route.distance,
                "distance_after": route.distance - 1,
                "distance_improvement": 1,
                "fan_option_value": None,
            } for route in frontier if route.improving_mask & (1 << index)),
        } for index, tile in enumerate(TILE_ORDER)
            if mask & (1 << index) and self.known[index] < 4)
        breakdown = self.score_breakdown
        return {
            "after_routes": tuple(route.to_details() for route in self.routes),
            "minimum_route_distance": self.minimum_distance,
            "viable_route_count": self.route_count,
            "live_tiles": live,
            "total_live_tile_weight": self.live_tile_weight,
            "score_breakdown": breakdown.to_details() if breakdown else None,
            "fan_option_value": self.fan_option_value,
            "chain_option_value": self.chain_option_value,
            "stable_action_key": self.decision_key[-1],
            "decision_key": self.decision_key,
        }


def evaluate_counts(
    counts: tuple[int, ...],
    meld_count: int,
    rules: RuleConfig,
    known: tuple[int, ...],
    *,
    chain_count: int,
    piao_count: int,
    baotou: bool,
    seat: int,
    dealer: int,
    draw_available: bool,
    stable_action: Action,
    cache_enabled: bool = True,
) -> CompactEvaluation:
    """Evaluate a trusted nonterminal state with precisely the internal static decision key."""

    whites = counts[33]
    authoritative_baotou = baotou
    routes = summarize_counts(
        counts, meld_count, you_cai_bi_kao=rules.you_cai_bi_kao,
        baotou=authoritative_baotou, chain_count=chain_count,
        cache_enabled=cache_enabled,
    )
    minimum = min(route.distance for route in routes)
    frontier = tuple(route for route in routes if route.distance == minimum)
    mask = 0
    for route in frontier:
        mask |= route.improving_mask
    live_weight = 0
    if draw_available:
        ones, twos, fours = (
            _known_planes_cached(known) if cache_enabled else _known_planes(known)
        )
        live_weight = (
            4 * mask.bit_count() - (mask & ones).bit_count()
            - 2 * (mask & twos).bit_count() - 4 * (mask & fours).bit_count()
        )
    derived_baotou = authoritative_baotou or (
        minimum == 1 and mask == ALL_TILE_MASK
    )
    option_route: RouteKind | None = None
    best_branch = 0
    quad_count = luxury_quad_count_counts(counts)
    for route in frontier:
        if route.config_compatibility or derived_baotou:
            branch = 1 if route.kind is RouteKind.REGULAR else 2 * 2**quad_count
            if branch >= best_branch:
                best_branch = branch
                option_route = route.kind
    fan = best_branch * (24 if seat == dealer else 10)
    if whites + piao_count == 4:
        fan *= 2
    if derived_baotou:
        fan *= 2
    chain = 2**chain_count
    key: DecisionKey = (
        0, 0, -minimum, live_weight, fan, chain, len(frontier),
        stable_action_key(stable_action),
    )
    return CompactEvaluation(
        minimum, len(frontier), live_weight, fan, chain, key, routes, known,
        draw_available, option_route, derived_baotou, whites, quad_count,
        chain_count, piao_count, seat, dealer, rules.base_score,
    )


def evaluate_state(
    state: SuccessorState,
    rules: RuleConfig,
    known: tuple[int, ...],
    *,
    seat: int,
    dealer: int,
    draw_available: bool,
    stable_action: Action,
    cache_enabled: bool = True,
) -> CompactEvaluation:
    if not isinstance(state, SuccessorState):
        raise ValueError("option score state must be a SuccessorState")
    if not isinstance(rules, RuleConfig) or type(rules.base_score) is not int or rules.base_score <= 0:
        raise ValueError("option rules require a positive base score")
    if type(rules.you_cai_bi_kao) is not bool:
        raise ValueError("you_cai_bi_kao must be a boolean")
    if any(type(value) is not int or not 0 <= value <= 3 for value in (seat, dealer)):
        raise ValueError("option seat and dealer must be between zero and three")
    if len(known) != len(TILE_ORDER) or any(type(value) is not int or value < 0 for value in known):
        raise ValueError("known counts must contain 34 nonnegative integers")
    if type(draw_available) is not bool or type(cache_enabled) is not bool:
        raise ValueError("draw_available and cache_enabled must be booleans")
    counts, meld_count = _validated_counts(state.hand, state.melds)
    return evaluate_counts(
        counts, meld_count, rules, tuple(known), chain_count=state.chain_count,
        piao_count=state.piao_count, baotou=state.baotou, seat=seat,
        dealer=dealer, draw_available=draw_available, stable_action=stable_action,
        cache_enabled=cache_enabled,
    )


def clear_caches() -> None:
    _known_planes_cached.cache_clear()
    _summaries_cached.cache_clear()
    _regular_cached.cache_clear()
    _paired_profile_cached.cache_clear()
    _profile_cached.cache_clear()
    _component_cached.cache_clear()


def cache_info() -> dict[str, dict[str, int | None]]:
    return {
        name: function.cache_info()._asdict()
        for name, function in (
            ("known_planes", _known_planes_cached),
            ("summaries", _summaries_cached),
            ("regular", _regular_cached),
            ("paired_profiles", _paired_profile_cached),
            ("suit_profiles", _profile_cached),
            ("components", _component_cached),
        )
    }
