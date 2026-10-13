"""One chance layer, with decisions on both sides of the draw.

Opponent hands/actions are deliberately not sampled. Ordinary turn progression
only consumes wall slots; the draw distribution is a public-information proxy.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import threading
import time
from typing import Callable, Mapping

from heuristic_ai.game.serialization import action_to_json
from heuristic_ai.game.models import Action, ActionKind, Observation, Phase
from heuristic_ai.strategy.models import Decision, DecisionContext
from heuristic_ai.strategy.search.one_ply_model import (
    DrawPlan,
    SearchModelError,
    apply_draw,
    plan_draw,
)
from heuristic_ai.strategy.search.one_ply_solver import (
    evaluate_counts,
    evaluate_state,
    score_drawn_terminal,
    stable_action_key,
)
from heuristic_ai.game.public_inventory import get_public_inventory, inventory_details
from heuristic_ai.game.rules import apply_action, can_hu, legal_actions
from heuristic_ai.game.scoring import score_terminal, luxury_quad_count_counts
from heuristic_ai.game.tiles import TILE_INDEX, TILE_ORDER

try:
    from heuristic_ai.strategy.native import _one_ply_native
except ImportError:
    _one_ply_native = None


Value = tuple[int, int, int, int, int, int]
ZERO: Value = (0, 0, 0, 0, 0, 0)
_DISCARDS = tuple(Action(ActionKind.DISCARD, tile) for tile in TILE_ORDER)
_DISCARD_KEYS = tuple(stable_action_key(action) for action in _DISCARDS)


def leaf_discard_actions(drawn: Observation, counts: tuple[int, ...]) -> tuple[Action, ...]:
    """The discard subset of rules.legal_actions for a validated fresh draw.

    A leaf never expands another gang. Reusing immutable actions avoids making
    thousands of identical Action objects and evaluating unused gang choices.
    Differential tests lock this subset to the shared rule implementation.
    """
    capped_piao = drawn.god.baotou and drawn.god.chain_count == 6
    if drawn.god.catch_play:
        if drawn.drawn_tile is None:
            raise SearchModelError("catch play requires a fresh drawn tile")
        if capped_piao and drawn.drawn_tile == "白":
            return ()
        return (_DISCARDS[TILE_INDEX[drawn.drawn_tile]],)
    return tuple(
        _DISCARDS[index] for index, count in enumerate(counts)
        if count and not (capped_piao and index == 33)
    )


class SearchBudgetExhausted(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class SearchBudget:
    deadline_monotonic: float
    max_nodes: int
    clock: Callable[[], float] = time.monotonic
    nodes: int = 0
    cancelled: Callable[[], bool] | None = None

    def check(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise SearchBudgetExhausted("search_cancelled")
        if self.clock() >= self.deadline_monotonic:
            raise SearchBudgetExhausted("search_deadline")

    def consume(self, count: int = 1) -> None:
        self.check()
        if self.nodes + count > self.max_nodes:
            raise SearchBudgetExhausted("search_node_budget")
        self.nodes += count


def decision_time_budget(context: DecisionContext, started: float) -> tuple[float, float, float]:
    """Return effective milliseconds, hard deadline and reserved search deadline."""
    rules = context.game.rules
    phase_seconds = {
        Phase.DRAW: rules.discard_timeout_seconds,
        Phase.RESPONSE_PENG: rules.peng_timeout_seconds,
        Phase.RESPONSE_CHI: rules.chi_timeout_seconds,
    }.get(context.observation.phase, rules.discard_timeout_seconds)
    effective_ms = min(context.budget_ms, context.game.decision_budget_ms, phase_seconds * 1000)
    hard = started + effective_ms / 1000
    if context.deadline_monotonic is not None:
        hard = min(hard, context.deadline_monotonic)
        effective_ms = min(effective_ms, max(0.0, (hard - started) * 1000))
    reserve_ms = max(50.0, effective_ms * 0.1)
    return effective_ms, hard, hard - reserve_ms / 1000


class SearchScheduler:
    """FIFO admission for CPU-bound Python searches, charged to their deadlines.

    The GIL already serializes this work. Admission prevents sixteen long
    searches from each waiting through fifteen others at every instruction
    slice. Networking, observations and the One-Ply Expectation internal static fallback remain free
    to run concurrently. No caller receives a fresh budget after waiting.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._waiting: deque[threading.Event] = deque()
        self._running = False

    @property
    def waiting(self) -> int:
        with self._condition:
            return len(self._waiting)

    @contextmanager
    def slot(self, budget: SearchBudget):
        token = threading.Event()
        with self._condition:
            self._waiting.append(token)
            if not self._running and self._waiting[0] is token:
                token.set()
        try:
            while True:
                budget.check()
                remaining = max(0.0, budget.deadline_monotonic - budget.clock())
                token.wait(min(0.02, remaining) if budget.cancelled is not None else remaining)
                with self._condition:
                    budget.check()
                    if not self._running and self._waiting[0] is token:
                        self._waiting.popleft()
                        self._running = True
                        break
        except BaseException:
            with self._condition:
                self._waiting.remove(token)
                if not self._running and self._waiting:
                    self._waiting[0].set()
            raise
        try:
            yield
        finally:
            with self._condition:
                self._running = False
                # A search may yield frequently. Wake only the FIFO successor instead
                # of making all sixteen game workers contend at each quantum.
                if self._waiting:
                    self._waiting[0].set()


_SEARCH_SCHEDULER = SearchScheduler()


@dataclass(frozen=True, slots=True)
class ExpectedValue:
    numerator: Value
    denominator: int = 1

    @property
    def key(self) -> tuple[Fraction, ...]:
        return tuple(Fraction(item, self.denominator) for item in self.numerator)

    def to_details(self) -> dict[str, object]:
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True, slots=True)
class LeafEvaluation:
    value: Value
    action: Action
    details: Mapping[str, object]
    node_cost: int


@dataclass(frozen=True, slots=True)
class RootEvaluation:
    action: Action
    expected: ExpectedValue
    baseline_key: tuple
    details: dict[str, object]

    @property
    def key(self) -> tuple:
        return (*self.expected.key, self.baseline_key)


@dataclass(frozen=True, slots=True)
class OnePlyEvaluation:
    """A complete One-Ply Expectation decision and reusable roots, never a partial ranking.

    Root details are taken from the Decision's recursively frozen JSON tree,
    so callers cannot change either the baseline audit or its continuations.
    A forced decision has no roots and needs no search.
    """

    decision: Decision
    roots: tuple[RootEvaluation, ...]
    level_four_decision: Decision
    search_deadline_monotonic: float


@lru_cache(maxsize=8192)
def _counts(hand: tuple[str, ...]) -> tuple[int, ...]:
    counts = [0] * len(TILE_ORDER)
    for tile in hand:
        counts[TILE_INDEX[tile]] += 1
    return tuple(counts)


def _action_from_details(details) -> Action:
    return Action(
        ActionKind(details["action"]),
        tile=details.get("tile") or None,
        tiles=tuple(details.get("tiles", ())),
    )


class OnePlySearch:
    """All root values must finish before a search decision can be returned."""

    def __init__(
        self, context: DecisionContext, baseline: Decision, budget: SearchBudget,
        *, cache_enabled: bool, native_enabled: bool = False,
    ) -> None:
        self.context = context
        self.baseline = baseline
        self.budget = budget
        self.cache_enabled = cache_enabled
        self.native_enabled = native_enabled and _one_ply_native is not None
        self.piao_count = int(baseline.details["piao_count"])
        self._baseline_keys = {
            _action_from_details(item["action"]): item["decision_key"]
            for item in baseline.details["candidate_evaluations"]
        }
        self._leaves: OrderedDict[tuple, LeafEvaluation] = OrderedDict()
        # Rules, seat, dealer and the post-discard zero chain are constant in
        # this decision. Stable action order is applied outside this cache.
        self._states: OrderedDict[tuple, object] = OrderedDict()
        self._terminal_details: dict[tuple, dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.completed: list[RootEvaluation] = []

    def statistics(self, *, complete: bool, exit_reason: str | None) -> dict[str, object]:
        return {
            "complete": complete,
            "exit_reason": exit_reason,
            "nodes": self.budget.nodes,
            "max_nodes": self.budget.max_nodes,
            "cache_enabled": self.cache_enabled,
            "leaf_backend": "native" if self.native_enabled else "python",
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "completed_candidates": len(self.completed),
            "total_candidates": len(self.context.candidates),
            "distribution": "visible_remaining_proxy",
            "opponent_model": "no_interruption_no_new_catch_play",
            "chance_layers": 1,
            "leaf_policy": "legal_hu_else_best_discard",
            "scheduler": "process_fifo",
            "value_fields": (
                "normalized_win_income", "nonterminal_progress", "live_tile_weight",
                "fan_option_value", "chain_option_value", "route_count",
            ),
            "route_fields": ("route", "distance", "improving_mask", "config_compatibility"),
            "tile_order": TILE_ORDER,
            "public_inventory": inventory_details(get_public_inventory(self.context.observation)),
        }

    def run(self) -> Decision:
        queued = time.perf_counter()
        with _SEARCH_SCHEDULER.slot(self.budget):
            self.queue_wait_ms = (time.perf_counter() - queued) * 1000
            return self._run()

    def _run(self) -> Decision:
        for candidate in sorted(
            self.context.candidates,
            key=lambda candidate: stable_action_key(candidate.action), reverse=True,
        ):
            self.budget.consume()
            action = candidate.action
            if action.kind is ActionKind.HU:
                score = score_terminal(
                    self.context.observation, self.context.game.rules,
                    piao_count=self.piao_count,
                )
                expected = ExpectedValue((score.normalized_winner_value, 0, 0, 0, 0, 0))
                details = {"score_breakdown": score.to_details(), "continuations": ()}
            elif action.kind is ActionKind.PASS:
                expected, details = self._pass()
            else:
                expected, details = self._action(self.context.observation, action)
            self.completed.append(RootEvaluation(
                action, expected, tuple(self._baseline_keys[action]),
                {"action": action_to_json(action), **expected.to_details(),
                 "static_key": self._baseline_keys[action], **details},
            ))
        return self._finish_result()

    def _finish_result(self) -> Decision:
        """Publish only after every root has a value (also used by exact One-Ply Expectation precomputation)."""
        self.budget.check()
        selected = max(self.completed, key=lambda item: item.key)
        observation = self.context.observation
        visible = get_public_inventory(observation).counts
        details = {
                "piao_count": self.piao_count,
                "drawable_wall_remaining": max(0, self.context.observation.wall_remaining - 20),
                "baseline": {
                    "action": action_to_json(self.baseline.action),
                    "reason": self.baseline.reason,
                    "selected_decision_key": self.baseline.details["selected_decision_key"],
                },
                # These inputs plus root/continuation/draw/leaf actions reproduce
                # every audited successor without copying thousands of witnesses.
                "position": {
                    "hand": tuple(sorted(observation.hand, key=TILE_INDEX.__getitem__)),
                    "melds": tuple({"kind": meld.kind, "tiles": meld.tiles}
                                   for meld in observation.melds[observation.seat]),
                    "known_counts": visible,
                    "seat": observation.seat, "dealer": observation.dealer,
                    "phase": observation.phase.value, "turn": observation.turn,
                    "last_discard": observation.last_discard,
                    "wall_remaining": observation.wall_remaining,
                    "drawn_tile": observation.drawn_tile,
                    "baotou": observation.god.baotou,
                    "chain_count": observation.god.chain_count,
                    "catch_play": observation.god.catch_play,
                    "you_cai_bi_kao": self.context.game.rules.you_cai_bi_kao,
                    "base_score": self.context.game.rules.base_score,
                },
                "candidate_evaluations": tuple(item.details for item in self.completed),
                "selected_expected_value": selected.expected.to_details(),
                "selected_static_key": selected.baseline_key,
                "search": self.statistics(complete=True, exit_reason=None),
            }
        details["search"]["queue_wait_ms"] = getattr(self, "queue_wait_ms", 0.0)
        build = lambda: Decision(selected.action, "one_ply_expectation", details)
        if self.context.publish_candidate is not None:
            from heuristic_ai.strategy.models import CandidatePublication
            self.context.publish_candidate(CandidatePublication(selected.action, build))
        return build()

    def _action(self, observation: Observation, action: Action) -> tuple[ExpectedValue, dict]:
        transition = apply_action(
            observation, action, self.context.game.rules, piao_count=self.piao_count,
        )
        evaluations = []
        for state in sorted(
            transition.successors,
            key=lambda state: stable_action_key(state.continuation_discard or action),
            reverse=True,
        ):
            self.budget.consume()
            plan = plan_draw(
                observation, action, state,
                supplement_draw=transition.supplement_draw,
                cache_enabled=self.cache_enabled,
            )
            expected, branches = self._plan(plan)
            continuation = state.continuation_discard
            local = evaluate_state(
                state, self.context.game.rules, plan.known,
                seat=observation.seat, dealer=observation.dealer,
                draw_available=transition.supplement_draw or state.wall_remaining > 20,
                stable_action=continuation or action,
                cache_enabled=self.cache_enabled,
            )
            evaluations.append((expected, local.decision_key, {
                "action": action_to_json(continuation) if continuation is not None else None,
                **expected.to_details(),
                "draws_until_self": plan.draws_until_self,
                "supplement_draw": plan.supplement,
                "draw_available": plan.available,
                "static_key": local.decision_key,
                "branches": branches,
            }))
        selected = max(evaluations, key=lambda item: (*item[0].key, item[1]))
        return selected[0], {
            "gang_kind": transition.gang_kind.value if transition.gang_kind is not None else None,
            "continuations": tuple(item[2] for item in evaluations),
            "selected_continuation": selected[2]["action"],
        }

    def _pass(self) -> tuple[ExpectedValue, dict]:
        observation = self.context.observation
        if (
            observation.phase is Phase.RESPONSE_PENG
            and observation.seat == (observation.turn + 1) % 4
            and not observation.god.catch_play
        ):
            chi_observation = replace(
                observation, phase=Phase.RESPONSE_CHI,
                responding_seats=(observation.seat,),
            )
            choices = legal_actions(chi_observation, self.context.game.rules)
            if any(item.action.kind is ActionKind.CHI for item in choices):
                evaluated = []
                for choice in sorted(choices, key=lambda item: stable_action_key(item.action), reverse=True):
                    self.budget.consume()
                    expected, details = self._action(chi_observation, choice.action)
                    # No future information is available at the deferred chi window.
                    # Its local tie key is the selected mandatory-discard key.
                    continuations = details["continuations"]
                    selected = next(
                        item for item in continuations
                        if item["action"] == details["selected_continuation"]
                    )
                    local_key = (*selected["static_key"][:7], stable_action_key(choice.action))
                    evaluated.append((expected, local_key, {
                        "action": action_to_json(choice.action),
                        **expected.to_details(), **details,
                    }))
                selected = max(evaluated, key=lambda item: (*item[0].key, item[1]))
                return selected[0], {
                    "response_continuations": tuple(item[2] for item in evaluated),
                    "selected_response": selected[2]["action"],
                }
        return self._action(observation, Action(ActionKind.PASS))

    def _plan(self, plan: DrawPlan) -> tuple[ExpectedValue, tuple[dict, ...]]:
        if not plan.available:
            return ExpectedValue(ZERO), ()
        weights = tuple(max(0, 4 - count) for count in plan.known)
        denominator = sum(weights)
        if denominator == 0:
            raise SearchModelError("no visible draw remains for an available wall")
        total = [0] * 6
        branches = []
        native_counts = bytes(_counts(plan.state.hand)) if self.native_enabled else None
        native_known = bytes(plan.known) if self.native_enabled else None
        for index, weight in enumerate(weights):
            if weight == 0:
                continue
            self.budget.consume()
            leaf = (self._native_leaf(plan, index, native_counts, native_known)
                    if self.native_enabled else self._leaf(plan, TILE_ORDER[index]))
            contribution = tuple(weight * value for value in leaf.value)
            total = [left + right for left, right in zip(total, contribution)]
            branches.append({
                "tile": TILE_ORDER[index], "visible_remaining": weight,
                "weight_numerator": weight, "weight_denominator": denominator,
                "value": leaf.value, "contribution_numerator": contribution,
                "selected_action": action_to_json(leaf.action),
                "leaf": leaf.details,
            })
        self.budget.check()
        return ExpectedValue(tuple(total), denominator), tuple(branches)

    def _native_leaf(self, plan, index, counts, known) -> LeafEvaluation:
        """The complete One-Ply Expectation leaf policy with its original audit/node semantics.

        Non-winning branches need only packed physical counts. Build a rich
        hypothetical Observation only when validating a new terminal score.
        One-Ply Expectation uses this exact kernel when the native extension is available.
        """
        self.cache_misses += 1
        wall = plan.state.wall_remaining - (0 if plan.supplement else plan.draws_until_self)
        seat, dealer = plan.observation.seat, plan.observation.dealer
        value, selected, cost, terminal, legal_count, summary = _one_ply_native.leaf(
            counts, known, index, len(plan.state.melds), int(self.context.game.rules.you_cai_bi_kao),
            int(plan.pre_draw_baotou), plan.state.chain_count, plan.state.piao_count,
            int(plan.catch_play), wall, 24 if seat == dealer else 10, int(self.cache_enabled),
        )
        node_cost = cost if terminal == 1 else cost - 1
        self.budget.consume(node_cost)
        if terminal == 1:
            seven_pairs = summary[2] == 0 and plan.state.chain_count == plan.state.piao_count
            whites = counts[33] + int(index == 33)
            quads = luxury_quad_count_counts(counts, index)
            score_key = (value, seven_pairs, quads if seven_pairs else 0, whites,
                         plan.pre_draw_baotou, plan.state.chain_count,
                         plan.state.piao_count, seat, dealer, self.context.game.rules)
            details = self._terminal_details.get(score_key) if self.cache_enabled else None
            if details is None:
                drawn = apply_draw(plan, TILE_ORDER[index], self.context.game.rules)
                score = score_drawn_terminal(drawn, self.context.game.rules, piao_count=plan.state.piao_count,
                                             cache_enabled=self.cache_enabled)
                if score.normalized_winner_value != value[0]:
                    raise RuntimeError("native One-Ply Expectation leaf disagrees with shared scoring")
                details = {"terminal": "hu", "score_breakdown": score.to_details()}
                if self.cache_enabled:
                    self._terminal_details[score_key] = details
            return LeafEvaluation(value, Action(ActionKind.HU), details, node_cost)
        regular_distance, regular_mask, pairs_distance, pairs_mask, compatible = summary
        routes = [("regular", regular_distance, regular_mask, bool(compatible))]
        if pairs_distance >= 0:
            routes.append(("seven_pairs", pairs_distance, pairs_mask, bool(compatible)))
        return LeafEvaluation(value, _DISCARDS[selected], {
            "terminal": "projected_draw" if terminal == 2 else None,
            "routes": tuple(routes), "legal_discard_count": legal_count,
        }, node_cost)

    def _leaf(self, plan: DrawPlan, tile: str) -> LeafEvaluation:
        if self.native_enabled:
            return self._native_leaf(plan, TILE_INDEX[tile], bytes(_counts(plan.state.hand)), bytes(plan.known))
        drawn = apply_draw(plan, tile, self.context.game.rules)
        counts = _counts(drawn.hand)
        known = list(plan.known)
        known[TILE_INDEX[tile]] += 1
        known = tuple(known)
        own_melds = drawn.melds[drawn.seat]
        key = (
            counts, own_melds, known, drawn.wall_remaining,
            drawn.god, plan.state.piao_count, tile if drawn.god.catch_play else None,
        )
        cached = self._leaves.get(key) if self.cache_enabled else None
        if cached is not None:
            self.cache_hits += 1
            self._leaves.move_to_end(key)
            self.budget.consume(cached.node_cost)
            return cached
        self.cache_misses += 1
        if can_hu(drawn, self.context.game.rules):
            self.budget.consume()
            score = score_drawn_terminal(
                drawn, self.context.game.rules, piao_count=plan.state.piao_count,
                cache_enabled=self.cache_enabled,
            )
            result = LeafEvaluation(
                (score.normalized_winner_value, 0, 0, 0, 0, 0),
                Action(ActionKind.HU),
                {"terminal": "hu", "score_breakdown": score.to_details()}, 1,
            )
        else:
            discards = leaf_discard_actions(drawn, counts)
            if not discards:
                raise SearchModelError("non-winning draw has no legal discard")
            best = None
            for action in discards:
                self.budget.consume()
                after = list(counts)
                after[TILE_INDEX[action.tile]] -= 1
                # A non-winning draw cannot be an authoritative baotou. Its
                # discard therefore breaks both gang and piao chains.
                state_key = (tuple(after), len(own_melds), known, drawn.wall_remaining > 20)
                evaluation = self._states.get(state_key) if self.cache_enabled else None
                if evaluation is None:
                    self.cache_misses += 1
                    evaluation = evaluate_counts(
                        state_key[0], len(own_melds), self.context.game.rules, known,
                        chain_count=0, piao_count=0, baotou=False,
                        seat=drawn.seat, dealer=drawn.dealer,
                        draw_available=drawn.wall_remaining > 20,
                        stable_action=action, cache_enabled=self.cache_enabled,
                    )
                    if self.cache_enabled:
                        self._states[state_key] = evaluation
                        if len(self._states) > 16384:
                            self._states.popitem(last=False)
                else:
                    self.cache_hits += 1
                    self._states.move_to_end(state_key)
                evaluation_key = (*evaluation.decision_key[:7], _DISCARD_KEYS[TILE_INDEX[action.tile]])
                if best is None or evaluation_key > best[2]:
                    best = evaluation, action, evaluation_key
            assert best is not None
            selected, action, selected_key = best
            exhausted = drawn.wall_remaining < 24
            value = ZERO if exhausted else (
                0, 14 - selected.minimum_distance, selected.live_tile_weight,
                selected.fan_option_value, selected.chain_option_value, selected.route_count,
            )
            result = LeafEvaluation(value, action, {
                "terminal": "projected_draw" if exhausted else None,
                "routes": tuple(
                    (route.kind.value, route.distance, route.improving_mask, route.config_compatibility)
                    for route in selected.routes
                ),
                "legal_discard_count": len(discards),
            }, len(discards))
        self.budget.check()
        if self.cache_enabled:
            self._leaves[key] = result
            if len(self._leaves) > 16384:
                self._leaves.popitem(last=False)
        return result
