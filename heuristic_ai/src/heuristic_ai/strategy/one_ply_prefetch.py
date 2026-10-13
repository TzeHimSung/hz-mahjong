"""Exact One-Ply Expectation response precomputation, input binding and complete receipts."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import pickle
import threading
import time

from heuristic_ai.game.models import Action, ActionKind, Phase
from heuristic_ai.game.public_inventory import get_public_inventory
from heuristic_ai.game.rules import apply_action, legal_actions
from heuristic_ai.game.serialization import action_to_json, decision_key
from heuristic_ai.game.tiles import TILE_ORDER
from .models import (Decision, DecisionContext, POLICY_NAME, POLICY_VERSION, MAX_SEARCH_NODES,
                     CACHE_ENABLED, session_backend)
from .errors import AuditIntegrityError
from .prefetch_audit import plain, BlobRef, decision_value_digest
from .search.one_ply_model import plan_draw
from .search.one_ply_search import (OnePlySearch, ExpectedValue, RootEvaluation, LeafEvaluation,
                                  ZERO, _counts)
from .search.one_ply_solver import evaluate_state, stable_action_key


def decision_wire(decision):
    return decision.action, decision.reason, plain(decision.details), decision.native_enabled

def signature_digest(signature):
    # Unlike pickle memoization, this does not depend on incidental object aliases.
    return hashlib.sha256(repr(signature).encode('utf-8')).hexdigest()

@dataclass(frozen=True, slots=True)
class PrefetchedResult:
    """Only a complete One-Ply Expectation result can produce this versioned receipt."""
    action: Action
    audit_path: str
    audit_sha256: str
    task_version: int
    input_digest: str
    audit_size: int
    raw_size: int
    value_digest: str
    native_enabled: bool = True

    @session_backend
    def build_decision(self, *, check=lambda: None):
        ref = BlobRef("decision", (), self.audit_path, self.audit_sha256,
                      self.audit_size, self.raw_size, self.value_digest)
        try:
            payload = ref.read(check=check)
            decision = Decision(*payload)
        except (ValueError, TypeError, IndexError) as error:
            raise AuditIntegrityError("malformed One-Ply Expectation prefetch decision") from error
        search = decision.details.get("search", {})
        if (decision.action != self.action or decision.native_enabled != self.native_enabled
                or not search.get("complete") or search.get("exit_reason") is not None
                or search.get("completed_depth", 1) != 1
                or decision.details.get("piao_count") != 0
                or not self.input_digest
                or decision.details.get("prefetch_input_digest") != self.input_digest
                or decision_value_digest(decision) != self.value_digest):
            raise AuditIntegrityError("One-Ply Expectation prefetch receipt disagrees with audit")
        check()
        return decision


def input_key(context):
    return ((POLICY_NAME, POLICY_VERSION, MAX_SEARCH_NODES, CACHE_ENABLED, True),
            context.game.rules,
            decision_key(context.game.game_id, context.observation, context.candidates))


def chi_target(game, observation):
    o = observation
    if (o.phase not in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}
            or o.seat not in o.responding_seats or o.seat != (o.turn + 1) % 4
            or o.god.chain_count or o.god.catch_play or get_public_inventory(o).status != "exact"):
        return None
    view = replace(o, phase=Phase.RESPONSE_CHI, responding_seats=(o.seat,))
    candidates = legal_actions(view, game.rules)
    if len(candidates) <= 1:
        return None
    return DecisionContext(game, view, candidates, decision_key(game.game_id, view, candidates),
                           game.decision_budget_ms)


class ResponseCache:
    """Bounded immutable payloads; callers never share mutable audit dictionaries."""
    def __init__(self, max_bytes=128 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.size = 0
        self.hits = 0
        self.lock = threading.RLock()
        self.leaves = {}
        self.costs = {}

    def get(self, key):
        with self.lock:
            item = self.entries.get(key)
            if item is None:
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            if key in self.leaves:
                return self.leaves[key]
        return pickle.loads(item)

    def put(self, key, value):
        blob = pickle.dumps(value, protocol=5)
        self.import_entry(key, blob)
        return blob

    def import_entry(self, key, blob):
        with self.lock:
            self._import_entry(key, blob)

    def _import_entry(self, key, blob):
        cost = len(blob) + len(pickle.dumps(key, protocol=5))
        old = self.entries.pop(key, None)
        if old is not None:
            self.size -= self.costs.pop(key)
            self.leaves.pop(key, None)
        if cost > self.max_bytes:
            return
        self.entries[key] = blob
        self.costs[key] = cost
        value = pickle.loads(blob)
        if isinstance(value, LeafEvaluation):
            self.leaves[key] = replace(value, details=Decision(value.action, 'cached_leaf', value.details).details)
        self.size += cost
        while self.size > self.max_bytes:
            k, v = self.entries.popitem(last=False)
            self.size -= self.costs.pop(k)
            self.leaves.pop(k, None)

class ResumableOnePlySearch(OnePlySearch):
    def __init__(self, *args, shared_cache=None, elapsed_clock=time.perf_counter_ns, **kwargs):
        super().__init__(*args, **kwargs)
        self.elapsed_clock = elapsed_clock
        self.native_requested = kwargs.get("native_enabled", False)
        self.shared_cache = shared_cache if shared_cache is not None else ResponseCache()
        self.resume_hits = 0
        self._iterator = self._walk_roots()
        self.result = None

    def close(self):
        self._iterator.close()
        self.completed.clear()
        self._leaves.clear()
        self._states.clear()
        self._terminal_details.clear()

    def statistics(self, *, complete, exit_reason):
        value = super().statistics(complete=complete, exit_reason=exit_reason)
        value.update(resume_hits=self.resume_hits, scheduler="background_round_robin")
        return value

    def advance(self, milliseconds=25, *, should_yield=None):
        """Yield only at complete leaves/continuations. Node budget survives slices."""
        started = self.elapsed_clock()
        while self.result is None:
            self.budget.check()
            if should_yield is not None and should_yield():
                return None
            try:
                next(self._iterator)
            except StopIteration as completed:
                self.result = completed.value
                break
            if self.elapsed_clock() - started >= milliseconds * 1_000_000:
                break
        return self.result

    def _walk_roots(self):
        for candidate in sorted(self.context.candidates,
                                key=lambda c: stable_action_key(c.action), reverse=True):
            self.budget.consume()
            expected, details = yield from self._walk_action(self.context.observation, candidate.action)
            self.completed.append(RootEvaluation(candidate.action, expected,
                tuple(self._baseline_keys[candidate.action]), {
                    "action": action_to_json(candidate.action), **expected.to_details(),
                    "static_key": self._baseline_keys[candidate.action], **details}))
            yield
        return self._finish_result()

    def _walk_action(self, observation, action, deferred=True):
        if (deferred and action.kind is ActionKind.PASS
                and observation.phase is Phase.RESPONSE_PENG
                and observation.seat == (observation.turn + 1) % 4
                and not observation.god.catch_play):
            chi = replace(observation, phase=Phase.RESPONSE_CHI,
                          responding_seats=(observation.seat,))
            choices = legal_actions(chi, self.context.game.rules)
            if any(c.action.kind is ActionKind.CHI for c in choices):
                evaluated = []
                for choice in sorted(choices, key=lambda c: stable_action_key(c.action), reverse=True):
                    self.budget.consume()
                    expected, details = yield from self._walk_action(chi, choice.action, False)
                    selected = next(c for c in details['continuations']
                                    if c['action'] == details['selected_continuation'])
                    key = (*selected['static_key'][:7], stable_action_key(choice.action))
                    evaluated.append((expected, key, {'action': action_to_json(choice.action),
                                                      **expected.to_details(), **details}))
                best = max(evaluated, key=lambda e: (*e[0].key, e[1]))
                return best[0], {'response_continuations': tuple(e[2] for e in evaluated),
                                 'selected_response': best[2]['action']}
        transition = apply_action(observation, action, self.context.game.rules, piao_count=self.piao_count)
        prepared = []
        for state in transition.successors:
            self.budget.check()
            plan = plan_draw(observation, action, state, supplement_draw=transition.supplement_draw,
                             cache_enabled=self.cache_enabled)
            local = evaluate_state(state, self.context.game.rules, plan.known,
                seat=observation.seat, dealer=observation.dealer,
                draw_available=transition.supplement_draw or state.wall_remaining > 20,
                stable_action=state.continuation_discard or action, cache_enabled=self.cache_enabled)
            prepared.append((state, plan, local))
            yield
        prepared.sort(key=lambda e: stable_action_key(e[0].continuation_discard or action), reverse=True)
        evaluated = []
        for state, plan, local in prepared:
            self.budget.consume()
            expected, branches = yield from self._walk_plan(plan)
            continuation = state.continuation_discard
            evaluated.append((expected, local.decision_key, {
                'action': action_to_json(continuation) if continuation else None,
                **expected.to_details(), 'draws_until_self': plan.draws_until_self,
                'supplement_draw': plan.supplement, 'draw_available': plan.available,
                'static_key': local.decision_key, 'branches': branches}))
            yield
        best = max(evaluated, key=lambda e: (*e[0].key, e[1]))
        return best[0], {'gang_kind': transition.gang_kind.value if transition.gang_kind else None,
                         'continuations': tuple(e[2] for e in evaluated),
                         'selected_continuation': best[2]['action']}

    def _walk_plan(self, plan):
        if not plan.available:
            return ExpectedValue(ZERO), ()
        weights = tuple(4 - n for n in plan.known)
        denominator = sum(weights)
        if not denominator:
            from .search.one_ply_model import SearchModelError
            raise SearchModelError('no visible draw remains')
        total, branches = [0] * 6, []
        counts = _counts(plan.state.hand)
        native_counts, native_known = bytes(counts), bytes(plan.known)
        background = not hasattr(self, 'queue_wait_ms')
        for index, weight in enumerate(weights):
            if not weight:
                continue
            self.budget.consume()
            # Includes every input consumed by apply_draw / leaf evaluation.
            key = ('leaf', self.context.game.rules, counts, plan.state.melds,
                   plan.known, plan.state.wall_remaining, plan.state.chain_count,
                   plan.state.piao_count, plan.draws_until_self, plan.supplement,
                   plan.catch_play, plan.pre_draw_baotou, plan.observation.seat,
                   plan.observation.dealer, index)
            key += (POLICY_NAME, POLICY_VERSION, self.native_requested, self.native_enabled)
            leaf = self.shared_cache.get(key) if self.cache_enabled and self.shared_cache.leaves else None
            if leaf is None:
                leaf = (self._native_leaf(plan, index, native_counts, native_known)
                        if self.native_enabled else self._leaf(plan, TILE_ORDER[index]))
                # The front end consumes transferred checkpoints. Serializing all
                # its cold leaves would cost more than the native solve it saves.
                if self.cache_enabled and background:
                    self.shared_cache.put(key, leaf)
            else:
                self.resume_hits += 1
                self.budget.consume(leaf.node_cost)
            contribution = tuple(weight * v for v in leaf.value)
            total = [a + b for a, b in zip(total, contribution)]
            branches.append({'tile': TILE_ORDER[index], 'visible_remaining': weight,
                'weight_numerator': weight, 'weight_denominator': denominator,
                'value': leaf.value, 'contribution_numerator': contribution,
                'selected_action': action_to_json(leaf.action), 'leaf': leaf.details})
            yield
        return ExpectedValue(tuple(total), denominator), tuple(branches)
