"""Pure scheduling and cursor bookkeeping for training response windows."""

from __future__ import annotations

from dataclasses import dataclass

from heuristic_ai.game.models import Observation, Phase


RESPONSE_PHASES = frozenset({Phase.RESPONSE_PENG, Phase.RESPONSE_CHI})


@dataclass(slots=True)
class EventProgress:
    """Snapshot watermarks never acknowledge receipt of unseen event payloads.

    Zero is a snapshot request in this API. A positive bootstrap anchor is a
    future fetch origin with explicitly unknown earlier history, not consumption.
    Only a covered positive-cursor delta advances ``event_consumed_seq``; an
    attached snapshot tail can supply facts without advancing that cursor.
    """

    snapshot_seq: int | None = None
    event_consumed_seq: int | None = None
    bootstrap_anchor: int | None = None
    round_bootstrap_floor: int | None = None

    def begin_round(self, seq: int, *, has_public_action: bool) -> None:
        """Start an unconsumed fetch origin from the new authoritative epoch.

        Dealing emits no event. Its watermark can still be the old round's
        ending, so an empty new board must wait for a strictly newer watermark.
        A public discard/meld proves that the snapshot is past that boundary.
        """
        self.event_consumed_seq = None
        self.bootstrap_anchor = seq if has_public_action and seq > 0 else None
        self.round_bootstrap_floor = None if has_public_action else seq

    def snapshot(self, seq: int) -> None:
        self.snapshot_seq = seq
        if (self.event_consumed_seq in (None, 0) and self.bootstrap_anchor is None and seq > 0
                and (self.round_bootstrap_floor is None or seq > self.round_bootstrap_floor)):
            self.bootstrap_anchor = seq

    def gap(self) -> None:
        self.event_consumed_seq = None
        self.bootstrap_anchor = None
        self.round_bootstrap_floor = None

    def consumed(self, watermark: int) -> None:
        if self.event_consumed_seq is None or watermark > self.event_consumed_seq:
            self.event_consumed_seq = watermark

    @property
    def fetch_cursor(self) -> int | None:
        if self.event_consumed_seq is not None and self.event_consumed_seq > 0:
            return self.event_consumed_seq
        return self.bootstrap_anchor

    @property
    def needs_backfill(self) -> bool:
        cursor = self.fetch_cursor
        return cursor is not None and self.snapshot_seq is not None and self.snapshot_seq > cursor

    def details(self) -> dict[str, int | None]:
        return {
            "snapshot_seq": self.snapshot_seq,
            "event_consumed_seq": self.event_consumed_seq,
            "bootstrap_anchor": self.bootstrap_anchor,
            "round_bootstrap_floor": self.round_bootstrap_floor,
        }


def response_window_key(observation: Observation) -> tuple | None:
    """Identify one response opportunity independently of inventory enrichment."""

    if observation.phase not in RESPONSE_PHASES:
        return None
    source = observation.turn
    if not 0 <= source < len(observation.discards) or source == observation.seat:
        return None
    history = observation.discards[source]
    tile = observation.last_discard
    if not history or tile is None or history[-1] != tile:
        return None
    key = observation.round_no, observation.phase.value, source, len(history) - 1, tile
    # Live rivers can reuse an ordinal after a claim. The changed public meld
    # board distinguishes that later discard without depending on enrichment.
    return (*key, tuple(tuple((m.kind, m.tiles) for m in seat) for seat in observation.melds)) if any(observation.melds) else key


def action_window_key(observation: Observation) -> tuple:
    """Stable action opportunity identity, excluding inventory enrichment.

    A draw or supplement consumes a wall tile; a claim/mandatory-discard window
    changes the player's hand/melds. Together with river lengths these separate
    consecutive opportunities even when the same tile face is drawn twice.
    """
    response = response_window_key(observation)
    if response is not None:
        return (observation.seat, *response)
    return (
        observation.round_no, observation.seat, observation.phase.value,
        observation.turn, observation.wall_remaining, observation.drawn_tile,
        tuple(sorted(observation.hand)), observation.melds[observation.seat],
        tuple(len(river) for river in observation.discards),
    )
