"""Evaluate successor-route scores and opportunities to reach baotou."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from heuristic_ai.game.hand_solver import RouteState, analyze_hand
from heuristic_ai.game.models import RuleConfig
from heuristic_ai.game.rules import SuccessorState, effective_baotou
from heuristic_ai.game.scoring import (
    ScoreBreakdown, ScoreValueKind, _build_breakdown,
    _validate_chain, _validate_config, _validate_seats,
)
from heuristic_ai.game.tiles import TILE_ORDER


def _route_witness(route: RouteState) -> tuple[object, ...]:
    """Identify structure while intentionally omitting config compatibility.

    ``route_can_baotou`` has no rule-config input, so its provenance check can
    compare only structural witnesses. Config-aware callers compare complete
    ``RouteState`` values before using this identity after deriving new baotou.
    """

    return (
        route.kind,
        route.distance,
        route.components,
        route.wildcard_assignments,
        route.open_meld_compatibility,
        route.improving_tiles,
    )


def _same_route_witness(left: RouteState, right: RouteState) -> bool:
    return _route_witness(left) == _route_witness(right)


def _canonical_routes(state: SuccessorState) -> tuple[RouteState, ...]:
    try:
        return analyze_hand(
            state.hand,
            state.melds,
            you_cai_bi_kao=False,
            baotou=effective_baotou(state.baotou, state.hand),
            chain_count=state.chain_count,
        )
    except (AttributeError, IndexError, TypeError):
        raise ValueError("score successor state is malformed") from None


def _frontier_can_baotou(
    state: SuccessorState,
    routes: Sequence[RouteState],
) -> bool:
    minimum_distance = min(route.distance for route in routes)
    if minimum_distance != 1:
        return False
    improving_faces = {
        tile
        for route in routes
        if route.distance == minimum_distance
        for tile in route.improving_tiles
    }
    return improving_faces == set(TILE_ORDER)


def score_option(
    state: SuccessorState,
    route: RouteState,
    config: RuleConfig,
    *,
    seat: int,
    dealer: int,
    baotou: bool,
) -> ScoreBreakdown:
    """Project a route using only factors already present in its successor state."""

    if not isinstance(state, SuccessorState):
        raise ValueError("option score state must be a SuccessorState")
    if not isinstance(route, RouteState):
        raise ValueError("option score route must be a RouteState")
    _validate_config(config)
    _validate_seats(seat, dealer)
    if type(baotou) is not bool:
        raise ValueError("option score baotou must be a boolean")
    _validate_chain(state.chain_count, state.piao_count, state.hand.count("白"))
    authoritative_baotou = effective_baotou(state.baotou, state.hand)
    try:
        initial_routes = analyze_hand(
            state.hand,
            state.melds,
            you_cai_bi_kao=config.you_cai_bi_kao,
            baotou=authoritative_baotou,
            chain_count=state.chain_count,
        )
    except (AttributeError, IndexError, TypeError):
        raise ValueError("option score state is malformed") from None
    if route not in initial_routes:
        raise ValueError("option score route does not belong to successor state")
    canonical_routes = _canonical_routes(state)
    expected_baotou = authoritative_baotou or _frontier_can_baotou(
        state, canonical_routes
    )
    if baotou is not expected_baotou:
        raise ValueError("option score baotou does not match successor state")
    try:
        compatible_routes = analyze_hand(
            state.hand,
            state.melds,
            you_cai_bi_kao=config.you_cai_bi_kao,
            baotou=expected_baotou,
            chain_count=state.chain_count,
        )
    except (AttributeError, IndexError, TypeError):
        raise ValueError("option score state is malformed") from None
    if not any(
        candidate.config_compatibility
        and _same_route_witness(route, candidate)
        for candidate in compatible_routes
    ):
        raise ValueError("option score route is not compatible with state and rules")
    return _build_breakdown(
        value_kind=ScoreValueKind.OPTION,
        route=route,
        hand=state.hand,
        chain_count=state.chain_count,
        piao_count=state.piao_count,
        baotou=baotou,
        seat=seat,
        dealer=dealer,
        base_score=config.base_score,
    )


def route_can_baotou(
    state: SuccessorState,
    routes: Sequence[RouteState],
) -> bool:
    """Return whether the minimum one-short frontier accepts every tile face."""

    if not isinstance(state, SuccessorState):
        raise ValueError("baotou state must be a SuccessorState")
    try:
        normalized_routes = tuple(routes)
    except TypeError:
        raise ValueError("baotou routes must be a nonempty route sequence") from None
    if not normalized_routes or any(
        not isinstance(route, RouteState) for route in normalized_routes
    ):
        raise ValueError("baotou routes must be a nonempty route sequence")
    canonical_routes = _canonical_routes(state)
    if Counter(map(_route_witness, normalized_routes)) != Counter(
        map(_route_witness, canonical_routes)
    ):
        raise ValueError("baotou routes must be the complete canonical route set")
    return _frontier_can_baotou(state, canonical_routes)
