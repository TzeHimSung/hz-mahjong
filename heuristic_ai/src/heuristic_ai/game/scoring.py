from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .hand_solver import RouteKind, RouteState, analyze_hand
from .models import GodState, Observation, RuleConfig
from .rules import can_hu, effective_baotou


class ScoreValueKind(StrEnum):
    TERMINAL = "terminal"
    OPTION = "option"


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Immutable, self-validating factors and four-seat payment projection."""

    value_kind: ScoreValueKind
    route: RouteKind
    branch_factor: int
    luxury_quad_count: int
    chain_count: int
    piao_count: int
    chain_factor: int
    hand_white_count: int
    four_white_factor: int
    baotou_factor: int
    dealer: bool
    base_score: int
    fan: int
    score_deltas: tuple[int, int, int, int]
    normalized_winner_value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value_kind, ScoreValueKind):
            raise ValueError("score value_kind must be a ScoreValueKind")
        if not isinstance(self.route, RouteKind):
            raise ValueError("score route must be a RouteKind")

        integer_fields = {
            "branch_factor": self.branch_factor,
            "luxury_quad_count": self.luxury_quad_count,
            "chain_count": self.chain_count,
            "piao_count": self.piao_count,
            "chain_factor": self.chain_factor,
            "hand_white_count": self.hand_white_count,
            "four_white_factor": self.four_white_factor,
            "baotou_factor": self.baotou_factor,
            "base_score": self.base_score,
            "fan": self.fan,
            "normalized_winner_value": self.normalized_winner_value,
        }
        if any(type(value) is not int for value in integer_fields.values()):
            raise ValueError("score numeric fields must be integers")
        if type(self.dealer) is not bool:
            raise ValueError("score dealer must be a boolean")
        if not 0 <= self.luxury_quad_count <= 3:
            raise ValueError("score luxury_quad_count must be between zero and three")
        if self.route is RouteKind.REGULAR:
            if self.luxury_quad_count != 0 or self.branch_factor != 1:
                raise ValueError("regular score branch factors are inconsistent")
        else:
            expected_branch = 2 * 2**self.luxury_quad_count
            if self.branch_factor != expected_branch:
                raise ValueError("seven-pairs score branch factors are inconsistent")
            if self.chain_count != self.piao_count:
                raise ValueError(
                    "seven-pairs score chain may contain only piao actions"
                )

        if not 0 <= self.chain_count <= 6:
            raise ValueError("score chain_count must be between zero and six")
        if not 0 <= self.piao_count <= self.chain_count:
            raise ValueError("score piao_count must be between zero and chain_count")
        if self.chain_factor != 2**self.chain_count:
            raise ValueError("score chain_factor is inconsistent")
        if not 0 <= self.hand_white_count <= 4:
            raise ValueError("score hand_white_count must be between zero and four")
        total_whites = self.hand_white_count + self.piao_count
        if total_whites > 4:
            raise ValueError("score retained whites plus piao cannot exceed four")
        expected_four_white = 2 if total_whites == 4 else 1
        if self.four_white_factor != expected_four_white:
            raise ValueError("score four_white_factor is inconsistent")
        if self.baotou_factor not in {1, 2}:
            raise ValueError("score baotou_factor must be one or two")
        if self.base_score <= 0:
            raise ValueError("score base_score must be positive")
        if self.fan > 512:
            raise ValueError("score fan cannot exceed the global maximum of 512")

        expected_fan = (
            self.branch_factor
            * self.chain_factor
            * self.four_white_factor
            * self.baotou_factor
        )
        if self.fan != expected_fan:
            raise ValueError("score fan is inconsistent with its factors")

        try:
            deltas = tuple(self.score_deltas)
        except TypeError:
            raise ValueError("score_deltas must contain four integers") from None
        if len(deltas) != 4 or any(type(delta) is not int for delta in deltas):
            raise ValueError("score_deltas must contain four integers")
        object.__setattr__(self, "score_deltas", deltas)
        if sum(deltas) != 0:
            raise ValueError("score_deltas must sum to zero")

        expected_normalized = (24 if self.dealer else 10) * self.fan
        if self.normalized_winner_value != expected_normalized:
            raise ValueError("normalized winner value is inconsistent")
        winner_income = self.base_score * expected_normalized
        positive = tuple(delta for delta in deltas if delta > 0)
        negative = tuple(-delta for delta in deltas if delta < 0)
        if positive != (winner_income,):
            raise ValueError("score_deltas winner income is inconsistent")
        unit = self.base_score * self.fan
        expected_payments = (unit * 8,) * 3 if self.dealer else (unit, unit, unit * 8)
        if tuple(sorted(negative)) != tuple(sorted(expected_payments)):
            raise ValueError("score_deltas payments are inconsistent")

    def to_details(self) -> dict[str, object]:
        return {
            "value_kind": self.value_kind.value,
            "route": self.route.value,
            "branch_factor": self.branch_factor,
            "luxury_quad_count": self.luxury_quad_count,
            "chain_count": self.chain_count,
            "piao_count": self.piao_count,
            "chain_factor": self.chain_factor,
            "hand_white_count": self.hand_white_count,
            "four_white_factor": self.four_white_factor,
            "baotou_factor": self.baotou_factor,
            "dealer": self.dealer,
            "base_score": self.base_score,
            "fan": self.fan,
            "score_deltas": self.score_deltas,
            "normalized_winner_value": self.normalized_winner_value,
        }


def _validate_config(config: RuleConfig) -> None:
    if not isinstance(config, RuleConfig):
        raise ValueError("score config must be a RuleConfig")
    if type(config.base_score) is not int or config.base_score <= 0:
        raise ValueError("score base_score must be positive")
    if type(config.you_cai_bi_kao) is not bool:
        raise ValueError("score you_cai_bi_kao must be a boolean")


def _validate_seats(seat: int, dealer: int) -> None:
    if type(seat) is not int or not 0 <= seat <= 3:
        raise ValueError("score seat must be between zero and three")
    if type(dealer) is not int or not 0 <= dealer <= 3:
        raise ValueError("score dealer seat must be between zero and three")


def _validate_chain(chain_count: int, piao_count: int, hand_white_count: int) -> None:
    if type(chain_count) is not int or not 0 <= chain_count <= 6:
        raise ValueError("score chain_count must be between zero and six")
    if (
        type(piao_count) is not int
        or not 0 <= piao_count <= chain_count
    ):
        raise ValueError("score piao_count must be between zero and chain_count")
    if hand_white_count + piao_count > 4:
        raise ValueError("score retained whites plus piao cannot exceed four")


def _luxury_quad_count(route: RouteState, hand: Sequence[str]) -> int:
    if route.kind is RouteKind.REGULAR:
        return 0
    return luxury_quad_count(hand)


def luxury_quad_count(hand: Sequence[str]) -> int:
    """Count natural quads without spending the same whites twice (guide v21)."""
    counts = Counter(hand)
    natural = tuple(count for tile, count in counts.items() if tile != "白")
    return sum(count == 4 for count in natural) + int(
        counts["白"] == 4 and all(count % 2 == 0 for count in natural)
    )


def luxury_quad_count_counts(counts: Sequence[int], draw_index: int | None = None) -> int:
    """The packed-count equivalent, optionally including one hypothetical draw."""
    natural = counts[:33]
    quads = natural.count(4)
    if draw_index is not None and draw_index < 33:
        quads += int(counts[draw_index] == 3)
    whites = counts[33] + int(draw_index == 33)
    if whites == 4:
        odds = sum(count & 1 for count in natural)
        if draw_index is not None and draw_index < 33:
            odds += -1 if counts[draw_index] & 1 else 1
        quads += int(odds == 0)
    return quads


def _score_deltas(
    *,
    seat: int,
    dealer: int,
    base_score: int,
    fan: int,
) -> tuple[int, int, int, int]:
    deltas = [0, 0, 0, 0]
    if seat == dealer:
        for payer in range(4):
            if payer != seat:
                deltas[payer] = -(base_score * fan * 8)
    else:
        for payer in range(4):
            if payer == seat:
                continue
            multiplier = 8 if payer == dealer else 1
            deltas[payer] = -(base_score * fan * multiplier)
    deltas[seat] = -sum(deltas)
    return deltas[0], deltas[1], deltas[2], deltas[3]


def _build_breakdown(
    *,
    value_kind: ScoreValueKind,
    route: RouteState,
    hand: Sequence[str],
    chain_count: int,
    piao_count: int,
    baotou: bool,
    seat: int,
    dealer: int,
    base_score: int,
) -> ScoreBreakdown:
    hand_white_count = tuple(hand).count("白")
    luxury_quad_count = _luxury_quad_count(route, hand)
    branch_factor = (
        1
        if route.kind is RouteKind.REGULAR
        else 2 * 2**luxury_quad_count
    )
    chain_factor = 2**chain_count
    four_white_factor = 2 if hand_white_count + piao_count == 4 else 1
    baotou_factor = 2 if baotou else 1
    fan = branch_factor * chain_factor * four_white_factor * baotou_factor
    is_dealer = seat == dealer
    normalized_winner_value = (24 if is_dealer else 10) * fan
    return ScoreBreakdown(
        value_kind=value_kind,
        route=route.kind,
        branch_factor=branch_factor,
        luxury_quad_count=luxury_quad_count,
        chain_count=chain_count,
        piao_count=piao_count,
        chain_factor=chain_factor,
        hand_white_count=hand_white_count,
        four_white_factor=four_white_factor,
        baotou_factor=baotou_factor,
        dealer=is_dealer,
        base_score=base_score,
        fan=fan,
        score_deltas=_score_deltas(
            seat=seat,
            dealer=dealer,
            base_score=base_score,
            fan=fan,
        ),
        normalized_winner_value=normalized_winner_value,
    )


def score_terminal(
    observation: Observation,
    config: RuleConfig,
    *,
    piao_count: int,
) -> ScoreBreakdown:
    """Score the best compliant complete route in an authoritative observation."""

    if not isinstance(observation, Observation):
        raise ValueError("terminal score observation must be an Observation")
    _validate_config(config)
    _validate_seats(observation.seat, observation.dealer)
    if not isinstance(observation.god, GodState):
        raise ValueError("terminal score god state must be a GodState")
    if type(observation.god.baotou) is not bool:
        raise ValueError("terminal score baotou must be a boolean")
    try:
        hand_white_count = tuple(observation.hand).count("白")
        own_melds = observation.melds[observation.seat]
    except (IndexError, TypeError):
        raise ValueError("terminal score observation is malformed") from None
    _validate_chain(observation.god.chain_count, piao_count, hand_white_count)
    baotou = effective_baotou(observation.god.baotou, observation.hand)
    try:
        legal_hu = can_hu(observation, config)
    except (AttributeError, IndexError, TypeError):
        raise ValueError("terminal score observation is malformed") from None
    if not legal_hu:
        raise ValueError("terminal score requires a legal hu observation")
    try:
        routes = analyze_hand(
            observation.hand,
            own_melds,
            you_cai_bi_kao=config.you_cai_bi_kao,
            baotou=baotou,
            chain_count=observation.god.chain_count,
        )
    except (AttributeError, IndexError, TypeError):
        raise ValueError("terminal score observation is malformed") from None

    candidates = tuple(
        _build_breakdown(
            value_kind=ScoreValueKind.TERMINAL,
            route=route,
            hand=observation.hand,
            chain_count=observation.god.chain_count,
            piao_count=piao_count,
            baotou=baotou,
            seat=observation.seat,
            dealer=observation.dealer,
            base_score=config.base_score,
        )
        for route in routes
        if route.distance == 0
        and route.config_compatibility
        and (
            route.kind is RouteKind.REGULAR
            or observation.god.chain_count == piao_count
        )
    )
    if not candidates:
        raise ValueError("terminal score requires a compliant complete route")
    route_priority = {RouteKind.REGULAR: 0, RouteKind.SEVEN_PAIRS: 1}
    return max(
        candidates,
        key=lambda score: (
            score.normalized_winner_value,
            route_priority[score.route],
        ),
    )
