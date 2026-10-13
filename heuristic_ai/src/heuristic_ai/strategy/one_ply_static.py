"""One-Ply Expectation internal static action values and legal emergency selection."""
from __future__ import annotations

from dataclasses import dataclass

from heuristic_ai.game.serialization import action_to_json
from heuristic_ai.game.hand_solver import RouteKind, RouteState, analyze_hand
from heuristic_ai.game.models import Action, ActionKind
from heuristic_ai.strategy.models import (
    Decision,
    DecisionContext,
)
from heuristic_ai.game.public_inventory import get_public_inventory, inventory_details
from heuristic_ai.game.rules import (
    ActionTransition,
    SuccessorState,
    apply_action,
    effective_baotou,
)
from heuristic_ai.game.scoring import ScoreBreakdown, score_terminal
from heuristic_ai.strategy.evaluation.option_scoring import route_can_baotou, score_option
from heuristic_ai.game.tiles import TILE_INDEX, TILE_ORDER
from heuristic_ai.strategy.evaluation.features import RouteFrontier, route_details, route_frontier


_ACTION_PRIORITY = {
    ActionKind.HU: 6,
    ActionKind.PASS: 5,
    ActionKind.DISCARD: 4,
    ActionKind.PENG: 3,
    ActionKind.CHI: 2,
    ActionKind.GANG: 1,
}
_ROUTE_PRIORITY = {
    RouteKind.REGULAR: 0,
    RouteKind.SEVEN_PAIRS: 1,
}

ActionStableKey = tuple[int, int, tuple[int, ...]]
DecisionKey = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    ActionStableKey,
]


def _stable_action_key(action: Action) -> ActionStableKey:
    tile_order = -TILE_INDEX[action.tile] if action.tile is not None else 0
    consumed_order = tuple(-TILE_INDEX[tile] for tile in action.tiles)
    return _ACTION_PRIORITY[action.kind], tile_order, consumed_order


def _analyze_state(
    state: SuccessorState,
    context: DecisionContext,
) -> tuple[RouteState, ...]:
    return analyze_hand(
        state.hand,
        state.melds,
        you_cai_bi_kao=context.game.rules.you_cai_bi_kao,
        baotou=effective_baotou(state.baotou, state.hand),
        chain_count=state.chain_count,
    )


def _best_option_score(
    state: SuccessorState,
    routes: tuple[RouteState, ...],
    frontier: RouteFrontier,
    context: DecisionContext,
) -> ScoreBreakdown | None:
    derived_baotou = effective_baotou(state.baotou, state.hand) or route_can_baotou(
        state, routes
    )
    scores = tuple(
        score_option(
            state,
            route,
            context.game.rules,
            seat=context.observation.seat,
            dealer=context.observation.dealer,
            baotou=derived_baotou,
        )
        for route in frontier.routes
        if route.config_compatibility or derived_baotou
    )
    if not scores:
        return None
    return max(
        scores,
        key=lambda breakdown: (
            breakdown.normalized_winner_value,
            _ROUTE_PRIORITY[breakdown.route],
        ),
    )


@dataclass(frozen=True, slots=True)
class _StateEvaluation:
    state: SuccessorState
    routes: tuple[RouteState, ...]
    frontier: RouteFrontier
    score_breakdown: ScoreBreakdown | None
    fan_option_value: int
    chain_option_value: int
    stable_action_key: ActionStableKey
    decision_key: DecisionKey


def _evaluate_state(
    state: SuccessorState,
    context: DecisionContext,
    *,
    draw_available: bool,
    stable_action: Action,
    terminal_score: ScoreBreakdown | None = None,
) -> _StateEvaluation:
    routes = _analyze_state(state, context)
    inventory = get_public_inventory(context.observation)
    if inventory.status == "exact":
        frontier = route_frontier(
            routes, dict(zip(TILE_ORDER, inventory.counts)),
            draw_available=draw_available,
        )
    else:
        minimum = min(route.distance for route in routes)
        frontier = RouteFrontier(
            minimum, tuple(route for route in routes if route.distance == minimum), (), 0,
        )
    breakdown = terminal_score or _best_option_score(
        state,
        routes,
        frontier,
        context,
    )
    fan_option_value = (
        breakdown.normalized_winner_value // breakdown.chain_factor
        if breakdown is not None
        else 0
    )
    chain_option_value = 2**state.chain_count
    stable_key = _stable_action_key(stable_action)
    terminal = terminal_score is not None
    decision_key: DecisionKey = (
        int(terminal),
        terminal_score.normalized_winner_value if terminal_score is not None else 0,
        -frontier.minimum_distance,
        frontier.live_tile_weight,
        fan_option_value,
        chain_option_value,
        frontier.route_count,
        stable_key,
    )
    return _StateEvaluation(
        state=state,
        routes=routes,
        frontier=frontier,
        score_breakdown=breakdown,
        fan_option_value=fan_option_value,
        chain_option_value=chain_option_value,
        stable_action_key=stable_key,
        decision_key=decision_key,
    )


def _continuation_details(evaluation: _StateEvaluation) -> dict[str, object]:
    continuation = evaluation.state.continuation_discard
    if continuation is None:
        raise ValueError("claim successor requires a continuation discard")
    return {
        "action": action_to_json(continuation),
        "after_routes": tuple(route_details(route) for route in evaluation.routes),
        "minimum_route_distance": evaluation.frontier.minimum_distance,
        "viable_route_count": evaluation.frontier.route_count,
        "live_tiles": evaluation.frontier.live_tiles,
        "total_live_tile_weight": evaluation.frontier.live_tile_weight,
        "score_breakdown": (
            evaluation.score_breakdown.to_details()
            if evaluation.score_breakdown is not None
            else None
        ),
        "fan_option_value": evaluation.fan_option_value,
        "chain_option_value": evaluation.chain_option_value,
        "stable_action_key": evaluation.stable_action_key,
        "decision_key": evaluation.decision_key,
    }


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    action: Action
    stable_action_key: ActionStableKey
    decision_key: DecisionKey
    details: dict[str, object]


def _candidate_details(
    action: Action,
    transition: ActionTransition,
    before_routes: tuple[RouteState, ...],
    selected: _StateEvaluation,
    continuations: tuple[_StateEvaluation, ...],
    decision_key: DecisionKey,
) -> dict[str, object]:
    before_kinds = tuple(dict.fromkeys(route.kind for route in before_routes))
    after_kinds = {route.kind for route in selected.routes}
    invalidated = tuple(
        route_kind.value
        for route_kind in before_kinds
        if route_kind not in after_kinds
    )
    selected_continuation = selected.state.continuation_discard
    return {
        "action": action_to_json(action),
        "gang_kind": (
            transition.gang_kind.value if transition.gang_kind is not None else None
        ),
        "before_routes": tuple(route_details(route) for route in before_routes),
        "after_routes": tuple(route_details(route) for route in selected.routes),
        "invalidated_routes": invalidated,
        "minimum_route_distance": selected.frontier.minimum_distance,
        "viable_route_count": selected.frontier.route_count,
        "live_tiles": selected.frontier.live_tiles,
        "total_live_tile_weight": selected.frontier.live_tile_weight,
        "score_breakdown": (
            selected.score_breakdown.to_details()
            if selected.score_breakdown is not None
            else None
        ),
        "fan_option_value": selected.fan_option_value,
        "chain_option_value": selected.chain_option_value,
        "continuation_evaluations": tuple(
            _continuation_details(evaluation)
            for evaluation in sorted(
                continuations,
                key=lambda item: item.stable_action_key,
                reverse=True,
            )
        ),
        "selected_continuation": (
            action_to_json(selected_continuation)
            if selected_continuation is not None
            else None
        ),
        "stable_action_key": _stable_action_key(action),
        "decision_key": decision_key,
    }


def _evaluate_candidate(
    action: Action,
    context: DecisionContext,
    *,
    piao_count: int,
    before_routes: tuple[RouteState, ...],
) -> _CandidateEvaluation:
    transition = apply_action(
        context.observation,
        action,
        context.game.rules,
        piao_count=piao_count,
    )
    terminal_score = (
        score_terminal(
            context.observation,
            context.game.rules,
            piao_count=piao_count,
        )
        if action.kind is ActionKind.HU
        else None
    )
    is_claim = action.kind in {ActionKind.CHI, ActionKind.PENG}
    if is_claim:
        continuations = tuple(
            _evaluate_state(
                successor,
                context,
                draw_available=successor.wall_remaining > 20,
                stable_action=successor.continuation_discard,
            )
            for successor in transition.successors
            if successor.continuation_discard is not None
        )
        if len(continuations) != len(transition.successors):
            raise ValueError("claim successor requires a continuation discard")
        selected = max(continuations, key=lambda item: item.decision_key)
    else:
        if len(transition.successors) != 1:
            raise ValueError("non-claim action requires exactly one successor")
        continuations = ()
        successor = transition.successors[0]
        selected = _evaluate_state(
            successor,
            context,
            draw_available=(
                transition.supplement_draw or successor.wall_remaining > 20
            ),
            stable_action=action,
            terminal_score=terminal_score,
        )

    stable_key = _stable_action_key(action)
    decision_key: DecisionKey = (*selected.decision_key[:7], stable_key)
    return _CandidateEvaluation(
        action=action,
        stable_action_key=stable_key,
        decision_key=decision_key,
        details=_candidate_details(
            action,
            transition,
            before_routes,
            selected,
            continuations,
            decision_key,
        ),
    )



def evaluate_actions(context: DecisionContext, *, piao_count: int) -> Decision:
    """Pure evaluation for an isolated projection with a known chain count."""
    if not context.candidates:
        raise ValueError("One-Ply Expectation static evaluation found no legal candidate")

    observation = context.observation
    before_routes = analyze_hand(
        observation.hand,
        observation.melds[observation.seat],
        you_cai_bi_kao=context.game.rules.you_cai_bi_kao,
        baotou=effective_baotou(observation.god.baotou, observation.hand),
        chain_count=observation.god.chain_count,
    )
    evaluated = tuple(
        _evaluate_candidate(
            candidate.action,
            context,
            piao_count=piao_count,
            before_routes=before_routes,
        )
        for candidate in context.candidates
    )
    selected = max(evaluated, key=lambda item: item.decision_key)
    ordered = tuple(
        item.details
        for item in sorted(
            evaluated,
            key=lambda item: item.stable_action_key,
            reverse=True,
        )
    )
    return Decision(
        selected.action,
        "legal_hu"
        if selected.action.kind is ActionKind.HU
        else "static_action_score",
        {
            "drawable_wall_remaining": max(
                0, observation.wall_remaining - 20
            ),
            "piao_count": piao_count,
            "public_inventory": inventory_details(get_public_inventory(observation)),
            "effective_tile_weight_used": get_public_inventory(observation).status == "exact",
            "candidate_evaluations": ordered,
            "selected_decision_key": selected.decision_key,
        },
    )

def legal_fallback(context, reason, details=None):
    order = {ActionKind.HU: 0, ActionKind.PASS: 1, ActionKind.DISCARD: 2}
    action = min((c.action for c in context.candidates),
                 key=lambda a: (order.get(a.kind, 3), a.tile or "", a.tiles))
    return Decision(action, reason, details or {})
