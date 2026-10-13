"""Pure public-information projections for the One-Ply Expectation single-draw search.

The model skips opponents' hidden draws and actions. It charges their draws to
the wall but never invents hidden hands or discards. Observations produced here
are hypothetical and must not be passed to a live ``ChainStateTracker``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from heuristic_ai.game.models import (
    Action,
    ActionKind,
    GodState,
    Observation,
    Phase,
    PublicInventory,
    RuleConfig,
    is_tile_code,
    normalize_melds,
)
from heuristic_ai.strategy.search.one_ply_solver import summarize_hand
from heuristic_ai.game.public_inventory import get_public_inventory
from heuristic_ai.game.rules import SuccessorState
from heuristic_ai.game.tiles import TILE_INDEX, TILE_ORDER

try:
    from heuristic_ai.strategy.native import _one_ply_native
except ImportError:
    _one_ply_native = None
_native_evaluate_counts = getattr(_one_ply_native, "evaluate_counts", None)

class SearchModelError(ValueError):
    """An authoritative observation cannot support this local draw model."""


class PublicInventoryUnknown(SearchModelError):
    reason = "public_inventory_unknown"


@dataclass(frozen=True, slots=True)
class DrawPlan:
    state: SuccessorState
    observation: Observation
    known: tuple[int, ...]
    draws_until_self: int
    supplement: bool
    catch_play: bool
    available: bool
    pre_draw_baotou: bool


def _inventory(observation: Observation) -> tuple[int, ...]:
    if len(observation.discards) != 4 or len(observation.melds) != 4:
        raise SearchModelError("draw model requires four complete public seats")
    try:
        for melds in observation.melds:
            normalize_melds(melds)
        inventory = get_public_inventory(observation)
    except (ValueError, TypeError, AttributeError) as error:
        raise SearchModelError(f"invalid public inventory: {error}") from error
    if inventory.status != "exact":
        raise PublicInventoryUnknown(inventory.reason)
    return inventory.counts


def _response_source(observation: Observation) -> int:
    source = observation.turn
    if (
        type(source) is not int
        or not 0 <= source < 4
        or source == observation.seat
        or observation.seat not in observation.responding_seats
        or observation.last_discard is None
        or not observation.discards[source]
        or observation.discards[source][-1] != observation.last_discard
    ):
        raise SearchModelError("response source does not match its last river tile")
    if (
        observation.phase is Phase.RESPONSE_CHI
        and (source + 1) % 4 != observation.seat
    ):
        raise SearchModelError("chi response must belong to the next seat")
    return source


def _baotou_before_draw(state: SuccessorState, *, cache_enabled: bool,
                        known=None, native_enabled: bool = False) -> bool:
    if native_enabled and _native_evaluate_counts is not None:
        counts = [0] * len(TILE_ORDER)
        for tile in state.hand:
            counts[TILE_INDEX[tile]] += 1
        prefix, routes = _native_evaluate_counts(bytes(counts), bytes(known), len(state.melds),
                                                 0, 0, 0, 0, 0, 10, int(cache_enabled))
        minimum = -prefix[2]
        mask = (routes[1] if routes[0] == minimum else 0) | (routes[3] if routes[2] == minimum else 0)
        return minimum == 1 and mask == (1 << len(TILE_ORDER)) - 1
    routes = summarize_hand(
        state.hand,
        state.melds,
        you_cai_bi_kao=False,
        baotou=False,
        chain_count=state.chain_count,
        cache_enabled=cache_enabled,
    )
    minimum = min(route.distance for route in routes)
    mask = 0
    for route in routes:
        if route.distance == minimum:
            mask |= route.improving_mask
    # This is a structural test over all 34 faces, never a visible-tile filter.
    return minimum == 1 and mask == (1 << len(TILE_ORDER)) - 1


def plan_draw(
    observation: Observation,
    action: Action,
    successor: SuccessorState,
    *,
    supplement_draw: bool = False,
    cache_enabled: bool = True,
    native_enabled: bool = False,
) -> DrawPlan:
    """Project one ``apply_action`` successor to the player's next draw.

    ``apply_action`` has already charged one wall tile for a gang. A normal
    discard (including the mandatory discard after chi/peng) waits through three
    opposing draws and the player's draw. A response pass waits by seat order.
    A claim preserves the source's historical discard and transfers its entity
    into the new meld. Only a fresh draw increases known physical copies.
    """

    if not isinstance(action, Action) or not isinstance(successor, SuccessorState):
        raise SearchModelError("draw plan requires an action and its successor")
    if type(supplement_draw) is not bool or type(cache_enabled) is not bool or type(native_enabled) is not bool:
        raise SearchModelError("draw plan flags must be booleans")
    if (
        type(observation.seat) is not int
        or not 0 <= observation.seat < 4
        or type(observation.wall_remaining) is not int
        or observation.wall_remaining < 0
    ):
        raise SearchModelError("draw model has invalid seat or wall inventory")
    initial_known = _inventory(observation)
    response = observation.phase in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}
    source = _response_source(observation) if response else None
    if not response and not (
        observation.phase is Phase.DRAW and observation.turn == observation.seat
    ):
        raise SearchModelError("draw model requires the player's action window")
    if action.kind is ActionKind.HU:
        raise SearchModelError("a terminal hu has no next draw")
    if supplement_draw is not (action.kind is ActionKind.GANG):
        raise SearchModelError("only gang transitions use a supplement draw")
    if action.kind in {ActionKind.CHI, ActionKind.PENG, ActionKind.PASS} and not response:
        raise SearchModelError("claim or pass requires a response window")
    if action.kind is ActionKind.DISCARD and response:
        raise SearchModelError("a response cannot directly discard")
    if action.kind is ActionKind.CHI and observation.phase is not Phase.RESPONSE_CHI:
        raise SearchModelError("chi requires a chi response window")
    if action.kind is ActionKind.PENG and observation.phase is not Phase.RESPONSE_PENG:
        raise SearchModelError("peng requires a peng response window")
    if action.kind is ActionKind.GANG and response and observation.phase is not Phase.RESPONSE_PENG:
        raise SearchModelError("a claimed gang requires a peng response window")

    expected_wall = observation.wall_remaining - int(supplement_draw)
    if successor.wall_remaining != expected_wall:
        raise SearchModelError("successor has an unexplained wall charge")
    if len(successor.hand) != 3 * (4 - len(successor.melds)) + 1:
        raise SearchModelError("next draw requires a one-short concealed hand")

    rivers = list(observation.discards)
    claims = response and action.kind in {ActionKind.CHI, ActionKind.PENG, ActionKind.GANG}
    if claims:
        if action.tile != observation.last_discard or action.tile == "白":
            raise SearchModelError("claimed tile does not match a non-white discard")
        assert source is not None

    if action.kind in {ActionKind.CHI, ActionKind.PENG}:
        continuation = successor.continuation_discard
        if continuation is None:
            raise SearchModelError("chi and peng require their mandatory discard")
        rivers[observation.seat] += (continuation.tile,)
    elif action.kind is ActionKind.DISCARD:
        if successor.continuation_discard is not None:
            raise SearchModelError("ordinary discard cannot contain a continuation")
        rivers[observation.seat] += (action.tile,)
    elif successor.continuation_discard is not None:
        raise SearchModelError("this action cannot contain a mandatory discard")

    all_melds = list(observation.melds)
    all_melds[observation.seat] = successor.melds
    conserved_discard = False
    if action.kind is ActionKind.DISCARD and successor.melds == observation.melds[observation.seat]:
        try:
            removed = observation.hand.index(action.tile)
        except ValueError:
            pass
        else:
            # The shared transition preserves hand order when removing a tile.
            # This proves conservation directly; reordered inputs still use
            # the general multiset validation below.
            conserved_discard = successor.hand == observation.hand[:removed] + observation.hand[removed + 1:]
    if not conserved_discard:
        delta = Counter(successor.hand)
        delta.subtract(observation.hand)
        delta.update(tile for meld in successor.melds for tile in meld.tiles)
        delta.subtract(tile for meld in observation.melds[observation.seat] for tile in meld.tiles)
        if action.kind in {ActionKind.CHI, ActionKind.PENG}:
            delta.update((successor.continuation_discard.tile,))
        elif action.kind is ActionKind.DISCARD:
            delta.update((action.tile,))
        if claims:
            # The input river already accounted for the incoming entity.
            delta.subtract((action.tile,))
        if any(delta.values()):
            raise SearchModelError("action projection does not conserve known physical tiles")
    projected = replace(
        observation,
        phase=Phase.DRAW,
        turn=observation.seat,
        wall_remaining=successor.wall_remaining,
        hand=successor.hand,
        drawn_tile=None,
        discards=tuple(rivers),
        melds=tuple(all_melds),
        last_discard=None,
        responding_seats=(),
        god=GodState(False, successor.chain_count, observation.god.catch_play if supplement_draw else False),
        public_inventory=PublicInventory(initial_known, "exact"),
    )

    if supplement_draw:
        draws_until_self = 1
    elif action.kind is ActionKind.PASS:
        assert source is not None
        draws_until_self = (observation.seat - source) % 4
    else:
        draws_until_self = 4
    available = observation.wall_remaining - 20 >= draws_until_self
    return DrawPlan(
        state=successor,
        observation=projected,
        known=initial_known,
        draws_until_self=draws_until_self,
        supplement=supplement_draw,
        catch_play=projected.god.catch_play,
        available=available,
        pre_draw_baotou=(
            _baotou_before_draw(successor, cache_enabled=cache_enabled,
                                known=initial_known, native_enabled=native_enabled)
            if available else False
        ),
    )


def apply_draw(plan: DrawPlan, tile: str, config: RuleConfig) -> Observation:
    """Add exactly one known tile and charge all previously uncharged draws.

    ``config`` is required at this rule boundary so callers retain the active
    runtime rules. Shape completion is not itself a hu decision: the returned
    observation must still pass ``can_hu`` with these rules, including YouCaiBiKao.
    """

    if not isinstance(plan, DrawPlan) or not isinstance(config, RuleConfig):
        raise SearchModelError("apply_draw requires a draw plan and runtime rules")
    if not plan.available:
        raise SearchModelError("the reserved last twenty tiles prevent this draw")
    if type(plan.pre_draw_baotou) is not bool:
        raise SearchModelError("pre-draw baotou must be a boolean")
    if not is_tile_code(tile) or plan.known[TILE_INDEX[tile]] >= 4:
        raise SearchModelError("draw face has no unseen physical copies")
    hand = (*plan.state.hand, tile)
    known = list(plan.known)
    known[TILE_INDEX[tile]] += 1
    uncharged_draws = 0 if plan.supplement else plan.draws_until_self
    return replace(
        plan.observation,
        hand=hand,
        drawn_tile=tile,
        wall_remaining=plan.state.wall_remaining - uncharged_draws,
        god=GodState(
            # SuccessorState and plan_draw already validated every retained
            # entity; the unseen-face check above validates the new entity.
            plan.pre_draw_baotou,
            plan.state.chain_count,
            plan.catch_play,
        ),
        public_inventory=PublicInventory(tuple(known), "exact"),
    )
