from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from .tiles import is_tile_code


_REASON = re.compile(r"^[a-z][a-z0-9_]*$")


class ActionKind(StrEnum):
    DISCARD = "discard"
    CHI = "chi"
    PENG = "peng"
    GANG = "gang"
    HU = "hu"
    PASS = "pass"


class Phase(StrEnum):
    DEAL = "deal"
    DRAW = "draw"
    RESPONSE_PENG = "response_peng"
    RESPONSE_CHI = "response_chi"
    SETTLED = "settled"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    tile: str | None = None
    tiles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ActionKind):
            raise ValueError("action kind must be an ActionKind")
        try:
            tiles = tuple(self.tiles)
        except TypeError:
            raise ValueError(f"{self.kind.value} action tiles are invalid") from None
        object.__setattr__(self, "tiles", tiles)
        if self.tile is not None and not is_tile_code(self.tile):
            raise ValueError(f"{self.kind.value} action tile is invalid")
        if any(not is_tile_code(tile) for tile in tiles):
            raise ValueError(f"{self.kind.value} action tiles are invalid")
        if self.kind in {ActionKind.PASS, ActionKind.HU}:
            if self.tile is not None or self.tiles:
                raise ValueError(f"{self.kind.value} action cannot contain tile values")
            return
        if self.tile is None:
            raise ValueError(f"{self.kind.value} action requires a tile")
        if self.kind is ActionKind.CHI:
            if len(self.tiles) != 2:
                raise ValueError("chi action requires exactly two consumed tiles")
        elif self.tiles:
            raise ValueError(f"{self.kind.value} action cannot contain consumed tiles")


@dataclass(frozen=True, slots=True)
class CandidateAction:
    action: Action
    copies: int = 1

    def __post_init__(self) -> None:
        if self.copies < 1:
            raise ValueError("candidate copies must be positive")


@dataclass(frozen=True, slots=True)
class Meld:
    kind: str
    tiles: tuple[str, ...]


_MELD_KINDS = frozenset({"chi", "peng", "gang_ming", "gang_an", "gang_bu"})


def _normalize_meld(meld: Meld) -> Meld:
    if not isinstance(meld.kind, str) or meld.kind not in _MELD_KINDS:
        raise ValueError(
            "meld kind must be one of chi, peng, gang_ming, gang_an, gang_bu"
        )
    try:
        tiles = tuple(meld.tiles)
    except TypeError:
        raise ValueError("meld tiles must be a sequence") from None
    if any(not is_tile_code(tile) for tile in tiles):
        raise ValueError("hand and meld tiles must use valid tile codes")
    if "白" in tiles:
        raise ValueError("meld tiles cannot contain white wildcard")

    if meld.kind == "chi":
        if len(tiles) != 3:
            raise ValueError("chi meld must contain exactly three tiles")
        if not (
            all(len(tile) == 2 and tile[1] in "wbt" for tile in tiles)
            and len({tile[1] for tile in tiles}) == 1
        ):
            raise ValueError("chi meld must be one suited consecutive sequence")
        ordered = tuple(sorted(tiles, key=lambda tile: int(tile[0])))
        if tuple(int(tile[0]) for tile in ordered) != tuple(
            range(int(ordered[0][0]), int(ordered[0][0]) + 3)
        ):
            raise ValueError("chi meld must be one suited consecutive sequence")
        return Meld("chi", ordered)

    if meld.kind == "peng":
        if len(tiles) != 3:
            raise ValueError("peng meld must contain exactly three tiles")
        if len(set(tiles)) != 1:
            raise ValueError("peng meld tiles must be identical")
        return Meld("peng", tiles)

    if len(tiles) != 4:
        raise ValueError("gang meld must contain exactly four tiles")
    if len(set(tiles)) != 1:
        raise ValueError("gang meld tiles must be identical")
    return Meld(meld.kind, tiles)


def normalize_melds(melds: object) -> tuple[Meld, ...]:
    """Copy, canonicalize, and validate one player's complete meld collection."""

    try:
        supplied = tuple(melds)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError("melds must be a sequence") from None
    try:
        return _normalized_melds(supplied)
    except TypeError:
        # External callers may supply list-valued Meld.tiles; normalize those
        # before they become immutable cache keys. Validation remains identical.
        return _normalized_melds.__wrapped__(supplied)


@lru_cache(maxsize=1024)
def _normalized_melds(supplied):
    if len(supplied) > 4:
        raise ValueError("meld count must be between 0 and 4")
    if any(not isinstance(meld, Meld) for meld in supplied):
        raise ValueError("melds must contain Meld values")
    if sum(meld.kind == "chi" for meld in supplied) > 2:
        raise ValueError("chi meld count cannot exceed two")
    return tuple(_normalize_meld(meld) for meld in supplied)


@dataclass(frozen=True, slots=True)
class GodState:
    baotou: bool
    chain_count: int
    catch_play: bool


@dataclass(frozen=True, slots=True)
class PublicInventory:
    """Known physical copies, separate from the complete discard history.

    Exact counts use the shared 34-face tile order. Unknown inventories carry
    no numeric estimate; strategies must explicitly omit the affected feature.
    """

    counts: tuple[int, ...] | None
    status: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in {"exact", "unknown"}:
            raise ValueError("public inventory status must be exact or unknown")
        if self.status == "unknown":
            if self.counts is not None:
                raise ValueError("unknown public inventory cannot contain counts")
            if not isinstance(self.reason, str) or _REASON.fullmatch(self.reason) is None:
                raise ValueError("unknown public inventory requires a stable reason")
            return
        if self.reason is not None:
            raise ValueError("exact public inventory cannot contain a failure reason")
        try:
            counts = tuple(self.counts)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError("exact public inventory requires 34 physical counts") from None
        if len(counts) != 34 or any(type(item) is not int or not 0 <= item <= 4 for item in counts):
            raise ValueError("exact public inventory requires 34 integers between zero and four")
        object.__setattr__(self, "counts", counts)


@dataclass(frozen=True, slots=True)
class PublicActionState:
    """Publicly certified action chains; None means unknown, never zero."""
    chain_counts: tuple[int | None, ...]
    piao_counts: tuple[int | None, ...]
    catch_owner: int | None
    reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'chain_counts', tuple(self.chain_counts))
        object.__setattr__(self, 'piao_counts', tuple(self.piao_counts))
        if len(self.chain_counts) != 4 or len(self.piao_counts) != 4:
            raise ValueError('public action state requires four seats')
        for chain, piao in zip(self.chain_counts, self.piao_counts):
            if chain is not None and (type(chain) is not int or not 0 <= chain <= 6):
                raise ValueError('invalid public chain')
            if piao is not None and (type(piao) is not int or not 0 <= piao <= 4 or chain is not None and piao > chain):
                raise ValueError('invalid public piao count')
        if self.catch_owner is not None and (type(self.catch_owner) is not int or self.catch_owner not in range(4)):
            raise ValueError('invalid catch owner')


@dataclass(frozen=True, slots=True)
class Observation:
    round_no: int
    seat: int
    phase: Phase
    dealer: int
    turn: int
    scores: tuple[int, int, int, int]
    wall_remaining: int
    hand: tuple[str, ...]
    drawn_tile: str | None
    discards: tuple[tuple[str, ...], ...]
    melds: tuple[tuple[Meld, ...], ...]
    last_discard: str | None
    responding_seats: tuple[int, ...]
    god: GodState
    public_inventory: PublicInventory | None = None
    public_actions: PublicActionState | None = None
    public_discard_history: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True, slots=True)
class RuleConfig:
    rounds: int
    base_score: int
    you_cai_bi_kao: bool
    discard_timeout_seconds: int
    peng_timeout_seconds: int
    chi_timeout_seconds: int
    source_hash: str


class GameOutcomeStatus(StrEnum):
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class GameOutcome:
    game_id: str
    status: GameOutcomeStatus
    scores: tuple[int, ...] | None = None
    error: str | None = None
