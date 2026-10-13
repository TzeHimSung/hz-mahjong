"""The fixed One-Ply Expectation search and its internal execution and recovery paths."""
from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Callable

from heuristic_ai.game.models import ActionKind, GameOutcome, Observation, Phase
from heuristic_ai.game.rules import ChainStateTracker
from heuristic_ai.game.public_inventory import get_public_inventory
from heuristic_ai.strategy.models import (Decision, DecisionContext, DecisionResult, GameContext,
    MAX_SEARCH_NODES, CACHE_ENABLED)
from heuristic_ai.strategy.errors import AuditIntegrityError, DecisionExecutionError, OnePlyFallback
from heuristic_ai.strategy.one_ply_static import evaluate_actions, legal_fallback
from heuristic_ai.strategy.search.one_ply_model import PublicInventoryUnknown, SearchModelError
from heuristic_ai.strategy.search.one_ply_search import (OnePlyEvaluation, OnePlySearch, RootEvaluation,
    SearchBudget, SearchBudgetExhausted, decision_time_budget)


class OnePlySession:
    def __init__(
        self, *,
        clock: Callable[[], float] = time.monotonic,
        native_enabled: bool = False,
    ) -> None:
        self.max_nodes = MAX_SEARCH_NODES
        self.cache_enabled = CACHE_ENABLED
        self.clock = clock
        self.native_enabled = native_enabled
        self._chain_tracker = ChainStateTracker()

    def observe(self, observation: Observation) -> None:
        self._chain_tracker.observe(observation)

    def decide(self, context: DecisionContext) -> Decision:
        return self.evaluate(context).decision

    def evaluate(
        self, context: DecisionContext, *, search_deadline_monotonic: float | None = None,
    ) -> OnePlyEvaluation:
        started = self.clock()
        effective_ms, _hard, deadline = decision_time_budget(context, started)
        if search_deadline_monotonic is not None:
            deadline = min(deadline, search_deadline_monotonic)
        baseline = self.static_decision(context)
        budget = SearchBudget(
            deadline, self.max_nodes, self.clock, cancelled=context.cancelled,
        )
        search = OnePlySearch(context, baseline, budget, cache_enabled=self.cache_enabled,
                             native_enabled=self.native_enabled)
        try:
            budget.check()
            if len(context.candidates) == 1:
                result = baseline.with_details({
                    "search": {
                        **search.statistics(complete=True, exit_reason=None),
                        "completed_candidates": 1, "effective_budget_ms": effective_ms,
                    },
                }, reason="forced_action")
            else:
                inventory = get_public_inventory(context.observation)
                if inventory.status != "exact":
                    raise PublicInventoryUnknown(inventory.reason)
                result = search.run()
            budget.check()
            frozen_roots = tuple(
                RootEvaluation(root.action, root.expected, root.baseline_key, details)
                for root, details in zip(search.completed, result.details.get("candidate_evaluations", ()))
            )
            return OnePlyEvaluation(result, frozen_roots, baseline, deadline)
        except (SearchBudgetExhausted, SearchModelError) as error:
            reason = getattr(error, "reason", "search_model_unavailable")
            decision = baseline.with_details({
                "search": {
                    **search.statistics(complete=False, exit_reason=reason),
                    "effective_budget_ms": effective_ms,
                },
            })
            raise OnePlyFallback(decision, reason) from None

    def finish(self, outcome: GameOutcome) -> None:
        return None

    def static_decision(self, context):
        return evaluate_actions(context, piao_count=self._chain_tracker.piao_count_for(context.observation))


class OnePlyController:
    """Own one One-Ply Expectation session, its exact prefetch and explicitly recorded fallbacks."""

    def __init__(self, context: GameContext, *, clock=time.perf_counter):
        if context.decision_budget_ms < 1:
            raise ValueError("decision budget must be positive")
        self.game = context
        self._clock = clock
        self._budget_ms = context.decision_budget_ms
        self.session = OnePlySession(native_enabled=True)
        self._degradation_reason = None
        self._compute_error = None
        self._chain_pending = None
        self.background = self.handle = self.publish_prefetch = None
        self._finished = False

    @property
    def degraded(self):
        return self._degradation_reason is not None

    @property
    def degradation_reason(self):
        return self._degradation_reason

    @property
    def background_enabled(self):
        return not self._finished and not self.degraded

    def _disable_compute(self, reason, error=None):
        if self._degradation_reason is None:
            self._degradation_reason = reason
            self._chain_pending = None
            self.cancel_predictions()
            if error is not None:
                from heuristic_ai.runtime.recovery_context import safe_exception
                self._compute_error = {"error_id": uuid.uuid4().hex, "stage": reason,
                                       **safe_exception(error)}

    def restore_state(self, checkpoint):
        if checkpoint is not None:
            self.session._chain_tracker.restore(checkpoint)

    def state_checkpoint(self):
        checkpoint = self.session._chain_tracker.checkpoint()
        return checkpoint if checkpoint[0] is not None else None

    def observe(self, observation):
        try:
            self.session.observe(observation)
        except BaseException as error:
            if not self._await_chain_proof(observation, error):
                raise DecisionExecutionError("One-Ply Expectation failed while observing") from error
            if self._chain_pending is not None:
                return
        self._chain_pending = None
        if self.background is not None and self.background_enabled:
            try:
                self.background.update(self.handle, self.game, observation)
            except BaseException as error:
                self._disable_compute("observe_error", error)

    def _await_chain_proof(self, observation, error):
        """Only independently verified board/history evidence repairs a chain."""
        from heuristic_ai.game.rules import ChainStateError, DiscardHistoryUnavailable
        if not isinstance(error, ChainStateError):
            return False
        from heuristic_ai.runtime.recovery_context import safe_exception
        if self._chain_pending is None:
            self._chain_pending = {"error_id": uuid.uuid4().hex, "stage": "chain_state_pending",
                                   **safe_exception(error)}
            self.cancel_predictions()
        actions = observation.public_actions
        from heuristic_ai.game.public_inventory import current_visible_counts
        inventory = observation.public_inventory
        if (isinstance(error, DiscardHistoryUnavailable)
                and inventory is not None and inventory.status == "exact"
                and inventory.counts == current_visible_counts(observation)):
            chain, seat = observation.god.chain_count, observation.seat
            if chain == 0 or not any(m.kind.startswith("gang_") for m in observation.melds[seat]):
                if chain > 0 and (chain > observation.discards[seat].count("白")
                        or chain + observation.hand.count("白") > 4
                        or actions is not None and (
                            actions.reason not in {None, "public_action_history_unknown", "public_chain_unknown"}
                            or actions.chain_counts[seat] not in {None, chain}
                            or actions.piao_counts[seat] not in {None, chain})):
                    return True
                self.restore_state((observation, chain))
                self._chain_pending = None
                return True
        if (actions is not None and actions.reason is None
                and actions.chain_counts[observation.seat] == observation.god.chain_count
                and actions.piao_counts[observation.seat] is not None
                and (observation.public_discard_history is not None
                     or not any(observation.discards) and not any(observation.melds))):
            self.restore_state((observation, actions.piao_counts[observation.seat]))
            self._chain_pending = None
        return True

    @staticmethod
    def _is_valid(decision, context):
        return isinstance(decision, Decision) and any(
            candidate.action == decision.action for candidate in context.candidates)

    def _pending_decision(self, context):
        observation = context.observation
        hu_offered = any(c.action.kind is ActionKind.HU for c in context.candidates)
        actions = observation.public_actions
        piao_count = 0
        if actions is not None and actions.reason is None:
            value = actions.piao_counts[observation.seat]
            if type(value) is int and value >= 0:
                piao_count = value
        try:
            evaluated = evaluate_actions(context, piao_count=piao_count)
        except Exception:
            evaluated = None
        if (evaluated is not None and (not hu_offered or evaluated.action.kind is ActionKind.HU)
                and self._is_valid(evaluated, context)):
            return evaluated, "static_fallback"
        return legal_fallback(context, "chain_state_pending"), "legal_fallback"

    def decide(self, context):
        if self._chain_pending is not None:
            decision, source = self._pending_decision(context)
            return DecisionResult(decision.with_details({"strategy_error": self._chain_pending}),
                                  source, 0., True, "chain_state_pending")
        started = self._clock()
        if not self.degraded:
            try:
                decision = self.session.decide(context)
            except OnePlyFallback as fallback:
                elapsed = (self._clock() - started) * 1000
                if self._is_valid(fallback.decision, context):
                    reason = fallback.reason
                    if elapsed > self._budget_ms:
                        reason = "budget_exceeded"
                        self._disable_compute(reason)
                    return DecisionResult(fallback.decision, "static_fallback", elapsed, True, reason)
                self._disable_compute("invalid_fallback_action")
            except AuditIntegrityError:
                raise
            except BaseException as error:
                self._disable_compute("decide_error", error)
            else:
                elapsed = (self._clock() - started) * 1000
                if not self._is_valid(decision, context):
                    self._disable_compute("invalid_action")
                elif elapsed > self._budget_ms:
                    self._disable_compute("budget_exceeded")
                else:
                    source = "forced_action" if len(context.candidates) == 1 else "completed_search"
                    return DecisionResult(decision, source, elapsed, False)
        try:
            decision = self.session.static_decision(context)
        except BaseException as error:
            raise DecisionExecutionError("One-Ply Expectation static evaluation failed") from error
        if not self._is_valid(decision, context):
            raise DecisionExecutionError("One-Ply Expectation static evaluation returned an invalid action")
        if self._compute_error is not None:
            decision = decision.with_details({"strategy_error": self._compute_error})
        return DecisionResult(decision, "static_fallback", (self._clock() - started) * 1000,
                              True, self._degradation_reason)

    def bind_background(self, manager, handle, publish):
        if self.background_enabled:
            self.background, self.handle, self.publish_prefetch = manager, handle, publish
            manager.attach(handle, self)

    def accepted_action(self, context, action):
        if not (context.observation.phase is Phase.RESPONSE_PENG and action.kind is ActionKind.PASS):
            self.cancel_predictions()

    def cancel_predictions(self):
        if self.background is not None:
            self.background.cancel(self.handle)

    def receive(self, kind, payload):
        if self._finished:
            return
        if kind == "control":
            control = getattr(self.publish_prefetch, "control", None)
            if control is not None:
                control(payload)
        elif kind == "result":
            from .one_ply_prefetch import signature_digest
            signature, ready = payload
            version = self.background.targets.get(self.handle, {}).get(signature)
            if version is None or ready.task_version != version:
                return
            if ready.input_digest != signature_digest(signature):
                raise AuditIntegrityError("One-Ply Expectation prefetch target identity mismatch")
            self.publish_prefetch(signature, ready)
        elif kind == "error":
            self.cancel_predictions()

    def finish(self, outcome):
        if self._finished:
            return
        if self.background is not None:
            self.background.detach(self.handle)
        self._finished = True
        self.session.finish(outcome)
