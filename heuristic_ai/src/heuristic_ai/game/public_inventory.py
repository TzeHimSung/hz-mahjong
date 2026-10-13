"""Physical public inventory for the One-Ply Expectation policy.

``Observation.discards`` is the append-only history of discards. A claimed
entity appears there and in its meld, so open melds contribute one duplicate.
The face is implicit for peng/ming/bu, but a chi requires public provenance.
This module never uses opponents' concealed hands or the post-game replay API.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
import json
import copy
import pickle
from typing import Any, Iterable, Mapping

from .models import Action, ActionKind, Meld, Observation, Phase, PublicInventory, PublicActionState, normalize_melds
from .tiles import TILE_ORDER, is_tile_code


def _unknown(reason: str) -> PublicInventory:
    return PublicInventory(None, "unknown", reason)


def inventory_details(inventory: PublicInventory) -> dict[str, object]:
    return {"status": inventory.status, "reason": inventory.reason, "counts": inventory.counts}


def current_visible_counts(observation: Observation) -> tuple[int, ...]:
    """Count physical live rivers, without inferring a discard history."""
    known = Counter(observation.hand)
    for river in observation.discards:
        known.update(river)
    for group in observation.melds:
        for meld in group:
            known.update(meld.tiles)
    return tuple(known[tile] for tile in TILE_ORDER)


def inventory_from_hand_counts(observation: Observation, hand_counts: Any) -> PublicInventory | None:
    """Certify live rivers using the snapshot's public concealed-hand sizes.

    A complete historical river contains one extra entity per open claim. In
    that case retain the existing provenance path; totals cannot identify chi
    duplicates. Missing fields are handled by the parser, never guessed here.
    """
    if (not isinstance(hand_counts, (list, tuple)) or len(hand_counts) != 4
            or any(type(n) is not int or not 0 <= n <= 14 for n in hand_counts)
            or hand_counts[observation.seat] != len(observation.hand)):
        return _unknown('invalid_public_hand_counts')
    if observation.phase in {Phase.DRAW, Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}:
        if any(n + 3 * len(observation.melds[seat]) !=
               (14 if observation.phase is Phase.DRAW and seat == observation.turn else 13)
               for seat, n in enumerate(hand_counts)):
            return _unknown('invalid_public_hand_counts')
    counts = current_visible_counts(observation)
    expected = 136 - observation.wall_remaining - sum(
        n for seat, n in enumerate(hand_counts) if seat != observation.seat)
    duplicates = sum(meld.kind in {'chi', 'peng', 'gang_ming', 'gang_bu'}
                     for group in observation.melds for meld in group)
    if sum(counts) != expected:
        if duplicates and sum(counts) - expected == duplicates:
            return None
        return _unknown('public_inventory_total_mismatch')
    if any(count > 4 for count in counts):
        return _unknown('invalid_public_inventory')
    return PublicInventory(counts, 'exact')


def _resolve_counts(observation: Observation, chi_claims: tuple[tuple[int, str], ...]) -> PublicInventory:
    try:
        if len(observation.discards) != 4 or len(observation.melds) != 4:
            return _unknown("invalid_public_inventory")
        melds = tuple(normalize_melds(items) for items in observation.melds)
        history = Counter(tile for river in observation.discards for tile in river)
        known = Counter(observation.hand) + history
        duplicates: Counter[str] = Counter()
        chi_count = 0
        for seat_melds in melds:
            for meld in seat_melds:
                known.update(meld.tiles)
                if meld.kind == "chi":
                    chi_count += 1
                elif meld.kind in {"peng", "gang_ming", "gang_bu"}:
                    duplicates[meld.tiles[0]] += 1
        if len(chi_claims) != chi_count:
            return _unknown("chi_claim_source_unknown")
        chi_by_source: Counter[tuple[int, str]] = Counter()
        for claimant, tile in chi_claims:
            duplicates[tile] += 1
            chi_by_source[((claimant - 1) % 4, tile)] += 1
        if any(observation.discards[source].count(tile) < count
               for (source, tile), count in chi_by_source.items()):
            return _unknown("claim_source_mismatch")
        if any(history[tile] < count for tile, count in duplicates.items()):
            return _unknown("claim_discard_missing")
        known.subtract(duplicates)
        if any(not is_tile_code(tile) or not 0 <= count <= 4 for tile, count in known.items()):
            return _unknown("invalid_public_inventory")
        return PublicInventory(tuple(known[tile] for tile in TILE_ORDER), "exact")
    except (AttributeError, IndexError, TypeError, ValueError):
        return _unknown("invalid_public_inventory")


def get_public_inventory(observation: Observation) -> PublicInventory:
    """Read attached provenance, or resolve the snapshot's unambiguous melds.

    Legacy ``strategy.evaluation.features.known_counts`` intentionally remains unchanged
    for historical strategies. Missing chi provenance never yields an estimate.
    """
    if observation.public_inventory is not None:
        return observation.public_inventory
    return _resolve_counts(observation, ())


def get_public_actions(observation: Observation) -> PublicActionState:
    if observation.public_actions is not None:
        return observation.public_actions
    # With no white discards and no gangs, a nonzero chain is impossible.
    if not any('白' in river for river in observation.discards) and not any(
            meld.kind.startswith('gang') for melds in observation.melds for meld in melds):
        return PublicActionState((0,) * 4, (0,) * 4, None)
    return PublicActionState((None,) * 4, (None,) * 4, None, 'public_action_history_unknown')


def project_public_discard(observation, seat, tile, *, chain_count=0, piao_count=0):
    """Advance an attached certificate for a known own or forecast nonwhite discard."""
    summary = observation.public_actions
    if summary is None:
        return None
    chains, piaos = list(summary.chain_counts), list(summary.piao_counts)
    chains[seat], piaos[seat] = chain_count, piao_count
    owner = seat if tile == '白' else None if summary.catch_owner == seat else summary.catch_owner
    reason = summary.reason
    if reason == 'public_chain_unknown' and None not in chains and None not in piaos:
        reason = None
    return PublicActionState(tuple(chains), tuple(piaos), owner, reason)


@dataclass(frozen=True, slots=True)
class _OwnClaim:
    round_no: int
    seat: int
    snapshot_seq: int
    action: Action
    meld_index: int
    expected_meld: Meld
    previous_melds: tuple[Meld, ...]
    hand_after_consumption: tuple[str, ...]
    previous_discards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ConfirmedClaim:
    tile: str
    meld: Meld
    confirmed_at: int


@dataclass(frozen=True, slots=True)
class _ActiveAnchor:
    round_no: int
    seq: int


class StaleInventorySnapshot(ValueError):
    """The caller must discard this snapshot rather than ask a strategy to act."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PublicInventoryTracker:
    """A per-game public claim ledger, published at a snapshot's own watermark.

    The transport owns event coverage / bootstrap cursors. ``ingest`` receives
    facts and explicit coverage separately. A snapshot's round number starts a
    new epoch immediately; cached endings never select or move its anchor.
    Future events remain buffered, but older published snapshots are rejected.
    An unresolved baseline chi needs a certified prefix; later chi facts can
    only explain melds appended after the baseline.
    """

    def __init__(self, game_id: str, *, certify_history: bool = False) -> None:
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("inventory tracker requires a game id")
        self.game_id = game_id
        if type(certify_history) is not bool:
            raise ValueError('history certification must be a boolean')
        self.certify_history = certify_history
        self._events: dict[int, dict[str, Any]] = {}
        self._fingerprints: dict[int, str] = {}
        self._conflicts: set[int] = set()
        self._invalid: set[int] = set()
        self._boundary_conflicts: set[int] = set()
        self._boundary_invalid: set[int] = set()
        self._event_watermark = 0
        self._current_round: int | None = None
        self._round_anchor = 0
        self._round_anchor_phase = Phase.DEAL
        self._verified_round_start: int | None = None
        self._prefix_anchor: _ActiveAnchor | None = None
        self._first_active: _ActiveAnchor | None = None
        # Keep the latest authoritative active snapshot across terminal-only
        # epochs; it can certify several skipped rounds once coverage arrives.
        self._latest_active: _ActiveAnchor | None = None
        self._baseline_melds: tuple[tuple[Meld, ...], ...] = ((), (), (), ())
        self._coverage: list[tuple[int, int]] = []
        self._retention_floor = 0
        self._reassigned: set[int] = set()
        self._last_snapshot_seq = 0
        self._pending: dict[tuple, _OwnClaim] = {}
        self._confirmed: dict[tuple[int, int, int], _ConfirmedClaim] = {}
        self._discard_anchors: dict[int, tuple[int, str]] = {}
        self._duplicates = 0
        self._confirmed_actions = 0
        self._gap_count = 0
        self._last_inventory = _unknown("inventory_not_observed")
        self._history_anchor = None
        self._river_mode = None

    def checkpoint(self):
        # One local, trusted recovery receipt; never a strategy/cache result.
        value = copy.deepcopy(self.__dict__)
        return value if len(pickle.dumps(value, protocol=5)) <= 256 * 1024 else None

    def restore(self, value):
        if (not isinstance(value, dict) or value.get('game_id') != self.game_id
                or set(value) != set(self.__dict__) or len(pickle.dumps(value, protocol=5)) > 256 * 1024):
            raise ValueError('invalid public inventory recovery checkpoint')
        self.__dict__.update(copy.deepcopy(value))

    def _discard_history(self, observation, snapshot_seq):
        """Reconstruct history and live rivers only over certified public deltas."""
        if self._history_anchor is not None and self._history_anchor[0] != observation.round_no:
            self._history_anchor = None
            self._river_mode = None
        if not any(observation.melds):
            if (self._history_anchor is not None and any(new[:len(old)] != old for old, new in
                    zip(self._history_anchor[2], observation.discards))):
                return None, 'public_discard_history_mismatch', False
            self._history_anchor = (observation.round_no, snapshot_seq, observation.discards,
                                    (observation.turn, observation.last_discard)
                                    if observation.phase in {Phase.RESPONSE_CHI, Phase.RESPONSE_PENG} else None)
        if self._history_anchor is not None:
            _, start, rivers, last = self._history_anchor
        else:
            return None, None, False
        history, live = [list(r) for r in rivers], [list(r) for r in rivers]
        source_seq = start
        proved_melds = [[] for _ in range(4)]
        for seq in sorted(self._events):
            if not start < seq <= snapshot_seq:
                continue
            event = self._events[seq]
            kind, seat, tile = event['type'], event['seat'], event['tile']
            if kind == 'tile_discarded':
                history[seat].append(tile)
                live[seat].append(tile)
                last = seat, tile
                source_seq = seq
            elif kind in {'chi', 'peng', 'gang_ming'}:
                if not self._covered(source_seq, seq):
                    return None, 'chi_claim_source_unknown' if kind == 'chi' else 'claim_discard_missing', False
                if (last is None or last[0] == seat or tile not in (None, '', last[1])
                        or kind == 'chi' and last[0] != (seat - 1) % 4):
                    return None, 'claim_source_mismatch', False
                source, claimed = last
                if not live[source] or live[source][-1] != claimed:
                    return None, 'claim_discard_missing', False
                live[source].pop()
                if kind == 'chi':
                    index = len(proved_melds[seat])
                    if index >= len(observation.melds[seat]):
                        return None, 'claim_meld_mismatch', False
                    meld = observation.melds[seat][index]
                    data = event.get('data') or {}
                    tiles = data.get('tiles') if isinstance(data, Mapping) else None
                    if tiles is not None and (not isinstance(tiles, (list, tuple)) or len(tiles) not in {2, 3}
                                              or any(not is_tile_code(t) for t in tiles)):
                        return None, 'invalid_public_event', False
                    if (meld.kind != 'chi' or claimed not in meld.tiles
                            or tiles is not None and Counter(tiles) not in
                                (Counter(meld.tiles), Counter(meld.tiles) - Counter((claimed,)))):
                        return None, 'claim_meld_mismatch', False
                    proved_melds[seat].append(meld)
                else:
                    proved_melds[seat].append(Meld(kind, (claimed,) * (3 if kind == 'peng' else 4)))
                last = None
            elif kind == 'gang_an':
                proved_melds[seat].append(Meld(kind, (tile,) * 4))
            elif kind == 'gang_bu':
                previous = Meld('peng', (tile,) * 3)
                if previous not in proved_melds[seat]:
                    return None, 'claim_discard_missing', False
                proved_melds[seat][proved_melds[seat].index(previous)] = Meld(kind, (tile,) * 4)
            elif kind == 'tile_drawn':
                last = None
        if tuple(map(tuple, proved_melds)) != observation.melds:
            # Unobserved appended melds are different from contradicting a
            # known meld. Counts may repair only the former, without inventing
            # source events. Complete coverage with an extra meld is invalid.
            for proved, current in zip(proved_melds, observation.melds):
                if len(proved) > len(current) or any(old != new and not (
                        old.kind == 'peng' and new.kind == 'gang_bu'
                        and old.tiles[0] == new.tiles[0]) for old, new in zip(proved, current)):
                    return None, 'claim_meld_mismatch', False
            reason = ('chi_claim_source_unknown' if any(m.kind == 'chi'
                      for group in observation.melds for m in group) else 'claim_discard_missing')
            known_seq = max((seq for seq in self._events if start < seq <= snapshot_seq), default=start)
            if self._covered(start, snapshot_seq) or not self._covered(start, known_seq):
                return None, reason, False
            # With a continuous known prefix, only its last pending discard
            # can disappear into a later claim. All other known river tiles
            # must remain a prefix, even if the total still balances.
            candidates = [live]
            if last is not None and any(
                    seat != last[0] and last[1] in meld.tiles
                    and (meld.kind in {'peng', 'gang_ming'}
                         or meld.kind == 'chi' and seat == (last[0] + 1) % 4)
                    for seat, group in enumerate(observation.melds)
                    for meld in group[len(proved_melds[seat]):]):
                removed = [list(river) for river in live]
                if removed[last[0]] and removed[last[0]][-1] == last[1]:
                    removed[last[0]].pop()
                    candidates.append(removed)
            if not any(all(tuple(raw[:len(old)]) == tuple(old)
                           for old, raw in zip(prefix, observation.discards)) for prefix in candidates):
                return None, 'public_discard_history_mismatch', False
            return None, reason, True
        # Public snapshots can be newer than a successful asynchronous delta.
        # Once every current meld has a source proof, an unclaimed river suffix
        # is visible evidence; it does not require inventing an event or cursor.
        complete = self._covered(start, snapshot_seq)
        candidates = {}
        for mode, prefix in (('history', history), ('live', live)):
            if self._river_mode is not None and mode != self._river_mode:
                continue
            if all(tuple(raw[:len(old)]) == tuple(old) and (not complete or len(raw) == len(old))
                   for old, raw in zip(prefix, observation.discards)):
                candidates[mode] = tuple(tuple(old_history) + tuple(raw[len(old_live):])
                    for old_history, old_live, raw in zip(history, prefix, observation.discards))
        values = set(candidates.values())
        if not values:
            return None, 'public_discard_history_mismatch', False
        if len(values) != 1:
            return None, 'ambiguous_discard_history', False
        if len(candidates) == 1:
            self._river_mode = next(iter(candidates))
        return next(iter(values)), None, False

    def ingest(
        self, events: Iterable[Mapping[str, Any]], *, watermark: int,
        coverage_start_seq: int | None = None, covers_watermark: bool = False,
    ) -> None:
        """Store facts; only an explicit complete delta certifies its interval.

        A seq=0 snapshot tail must use ``covers_watermark=False``. Explicit
        coverage starting at zero is reserved for a separately complete replay,
        and is never inferred from the first/last event in a batch.
        """
        if type(watermark) is not int or watermark < 0:
            raise ValueError("event watermark must be a nonnegative integer")
        if type(covers_watermark) is not bool:
            raise ValueError("event coverage flag must be a boolean")
        if coverage_start_seq is not None and (
            type(coverage_start_seq) is not int or not 0 <= coverage_start_seq <= watermark
        ):
            raise ValueError("event coverage start is invalid")
        if covers_watermark and coverage_start_seq is None:
            raise ValueError("complete event coverage requires its start sequence")
        try:
            batch = tuple(events)
        except TypeError:
            self._invalid.add(watermark)
            self._boundary_invalid.add(watermark)
            return
        for supplied in batch:
            if not isinstance(supplied, Mapping):
                self._invalid.add(watermark)
                self._boundary_invalid.add(watermark)
                continue
            seq = supplied.get("seq")
            kind, seat, tile = supplied.get("type"), supplied.get("seat"), supplied.get("tile", "")
            if kind == 'gang' and isinstance(supplied.get('data'), Mapping):
                kind = {'an': 'gang_an', 'ming': 'gang_ming', 'bu': 'gang_bu'}.get(supplied['data'].get('kind'), kind)
            if type(seq) is not int or not 0 < seq <= watermark:
                self._invalid.add(watermark)
                self._boundary_invalid.add(watermark)
                continue
            if seq <= self._retention_floor:
                continue
            if not isinstance(kind, str) or type(seat) is not int or not -1 <= seat <= 3:
                self._invalid.add(seq)
                if kind == "round_ended" or not isinstance(kind, str):
                    self._boundary_invalid.add(seq)
                continue
            if kind in {"tile_discarded", "chi", "peng", "gang_ming", "gang_an", "gang_bu"}:
                if seat < 0 or (kind != "chi" and not is_tile_code(tile)) or (kind == "chi" and tile not in (None, "") and not is_tile_code(tile)):
                    self._invalid.add(seq)
                    continue
            try:
                # Timestamps do not change the identity of an inventory fact.
                record = {"seq": seq, "type": kind, "seat": seat, "tile": tile, "data": supplied.get("data")}
                fingerprint = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError):
                self._invalid.add(seq)
                if kind == "round_ended":
                    self._boundary_invalid.add(seq)
                continue
            if seq in self._fingerprints:
                if self._fingerprints[seq] != fingerprint:
                    self._conflicts.add(seq)
                    if (self._events[seq]["type"] == "round_ended") != (kind == "round_ended"):
                        self._boundary_conflicts.add(seq)
                else:
                    self._duplicates += 1
                continue
            self._fingerprints[seq] = fingerprint
            self._events[seq] = record
        self._event_watermark = max(self._event_watermark, watermark)
        if covers_watermark and watermark > self._retention_floor:
            intervals = sorted((*self._coverage, (max(coverage_start_seq, self._retention_floor), watermark)))
            merged: list[tuple[int, int]] = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            self._coverage = merged

    def _covered(self, start: int, end: int) -> bool:
        return start >= end or any(left <= start and end <= right for left, right in self._coverage)

    def begin_snapshot(self, observation: Observation, *, snapshot_seq: int) -> None:
        """Register the authoritative epoch before ingesting a response's facts."""
        if type(snapshot_seq) is not int or snapshot_seq < 0:
            raise ValueError("snapshot sequence must be a nonnegative integer")
        if type(observation.round_no) is not int or observation.round_no < 1:
            raise ValueError("snapshot round must be a positive integer")
        if self._current_round is not None and observation.round_no < self._current_round:
            raise StaleInventorySnapshot("stale_round_snapshot")
        if self._current_round is not None and snapshot_seq < self._last_snapshot_seq:
            raise StaleInventorySnapshot("stale_snapshot_sequence")
        active = observation.phase not in {Phase.SETTLED, Phase.FINISHED}
        if self._current_round != observation.round_no:
            self._prefix_anchor = self._latest_active
            self._current_round = observation.round_no
            self._round_anchor = snapshot_seq
            self._round_anchor_phase = observation.phase
            self._verified_round_start = None
            self._first_active = None
            self._baseline_melds = observation.melds
            self._pending.clear()
            self._confirmed.clear()
            self._discard_anchors.clear()
            self._reassigned.clear()
            # Keep only the previous active anchor's possible proof interval and
            # all future facts. A late join without an anchor cannot certify an
            # older prefix, except a separately complete first-round replay.
            self._retention_floor = (
                self._prefix_anchor.seq if self._prefix_anchor is not None
                else 0 if observation.round_no == 1 else snapshot_seq
            )
            self._events = {seq: value for seq, value in self._events.items() if seq > self._retention_floor}
            self._fingerprints = {seq: value for seq, value in self._fingerprints.items() if seq > self._retention_floor}
            self._conflicts = {seq for seq in self._conflicts if seq > self._retention_floor}
            self._invalid = {seq for seq in self._invalid if seq > self._retention_floor}
            self._boundary_conflicts = {seq for seq in self._boundary_conflicts if seq > self._retention_floor}
            self._boundary_invalid = {seq for seq in self._boundary_invalid if seq > self._retention_floor}
            self._coverage = [(max(start, self._retention_floor), end) for start, end in self._coverage
                              if end > self._retention_floor]
        if active:
            anchor = _ActiveAnchor(observation.round_no, snapshot_seq)
            if self._first_active is None:
                self._first_active = anchor
            self._latest_active = anchor
        self._last_snapshot_seq = snapshot_seq
        if observation.phase in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI} and observation.last_discard is not None and 0 <= observation.turn <= 3:
            self._discard_anchors[snapshot_seq] = (observation.turn, observation.last_discard)

    def _verify_prefix(self) -> None:
        start = None
        if self._current_round == 1 and self._covered(0, self._round_anchor):
            # Round one has a protocol-defined origin. This still requires
            # actual coverage, never an unverified bootstrap snapshot tail.
            endings = [seq for seq, item in self._events.items()
                       if seq <= self._round_anchor and item["type"] == "round_ended"]
            allowed = 1 if self._round_anchor_phase in {Phase.SETTLED, Phase.FINISHED} else 0
            if len(endings) == allowed:
                start = 0
        elif self._prefix_anchor is not None and self._first_active is not None:
            before, after = self._prefix_anchor, self._first_active
            interval = lambda seq: before.seq < seq <= after.seq
            if (before.round_no < after.round_no and self._covered(before.seq, after.seq)
                    and not any(interval(seq) for seq in self._boundary_conflicts | self._boundary_invalid)):
                endings = sorted(seq for seq, item in self._events.items()
                                 if interval(seq) and item["type"] == "round_ended")
                if len(endings) == after.round_no - before.round_no and endings[-1] <= self._round_anchor:
                    start = endings[-1]
        self._verified_round_start = start
        if start is not None:
            self._reassigned.update(seq for seq in self._events if start < seq <= self._round_anchor)

    def _confirm_actions(self, observation: Observation, snapshot_seq: int) -> None:
        for key, pending in tuple(self._pending.items()):
            if pending.round_no != observation.round_no or pending.seat != observation.seat or snapshot_seq <= pending.snapshot_seq:
                continue
            current = observation.melds[pending.seat]
            if len(current) <= pending.meld_index or current[pending.meld_index] != pending.expected_meld:
                continue
            if current[:pending.meld_index] != pending.previous_melds[:pending.meld_index]:
                continue
            river = observation.discards[pending.seat]
            if river[:len(pending.previous_discards)] != pending.previous_discards:
                continue
            expected = Counter(pending.hand_after_consumption)
            expected.subtract(river[len(pending.previous_discards):])
            draws = [event["tile"] for seq, event in self._events.items()
                     if pending.snapshot_seq < seq <= snapshot_seq and event["type"] == "tile_drawn"
                     and event["seat"] == pending.seat and is_tile_code(event["tile"])]
            expected.update(draws)
            actual = Counter(observation.hand)
            if expected != actual:
                # A fresh owned draw is also authoritative in the snapshot; it
                # can confirm the immediately following turn before its delta.
                if observation.phase is not Phase.DRAW or observation.turn != pending.seat or observation.drawn_tile is None:
                    continue
                expected.update((observation.drawn_tile,))
                if expected != actual:
                    continue
            if pending.expected_meld.kind == "chi":
                self._confirmed[(pending.round_no, pending.seat, pending.meld_index)] = _ConfirmedClaim(
                    pending.action.tile, pending.expected_meld, snapshot_seq,
                )
            self._confirmed_actions += 1
            del self._pending[key]

    def _chi_proofs(self, start_seq: int, snapshot_seq: int) -> tuple[dict[int, list[tuple[str, tuple[str, ...] | None]]], str | None]:
        result: dict[int, list[tuple[str, tuple[str, ...] | None]]] = {seat: [] for seat in range(4)}
        last_discard: tuple[int, int, str] | None = None
        records = [(seq, "event", event) for seq, event in self._events.items() if start_seq < seq <= snapshot_seq]
        records.extend((seq, "anchor", value) for seq, value in self._discard_anchors.items()
                       if start_seq <= seq <= snapshot_seq)
        for seq, record_kind, event in sorted(records, key=lambda item: (item[0], item[1])):
            if record_kind == "anchor":
                last_discard = (seq, *event)
                continue
            kind, seat, tile = event["type"], event["seat"], event["tile"]
            if kind == "round_ended":
                last_discard = None
            elif kind == "tile_discarded":
                last_discard = (seq, seat, tile)
            elif kind == "chi":
                # An incomplete event tail may omit a later discard. An
                # explicitly reported claimed tile is authoritative itself;
                # reconstructing a missing tile needs uninterrupted evidence.
                prior_is_current = last_discard is not None and (
                    seq == last_discard[0] + 1 or self._covered(last_discard[0], seq)
                )
                if not is_tile_code(tile):
                    if not prior_is_current:
                        return result, "chi_claim_source_unknown"
                    tile = last_discard[2]
                if tile == "白" or (prior_is_current and last_discard[1:] != ((seat - 1) % 4, tile)):
                    return result, "claim_source_mismatch"
                target = None
                data = event.get("data")
                if isinstance(data, dict) and "tiles" in data:
                    consumed = data["tiles"]
                    if not isinstance(consumed, (list, tuple)) or len(consumed) not in {2, 3}:
                        return result, "invalid_public_event"
                    try:
                        target = normalize_melds((Meld("chi", tuple(consumed) + ((tile,) if len(consumed) == 2 else ())),))[0].tiles
                    except ValueError:
                        return result, "invalid_public_event"
                result[seat].append((tile, target))
                last_discard = None
            elif kind in {"peng", "gang_ming"}:
                last_discard = None
        return result, None

    @staticmethod
    def _match_chi_slots(
        slots: list[tuple[int, Meld]], events: list[tuple[str, tuple[str, ...] | None]], fixed: dict[int, str],
    ) -> tuple[tuple[str, ...] | None, str | None]:
        if len(events) > len(slots):
            return None, "claim_meld_mismatch"
        possibilities: set[tuple[str, ...]] = set()
        for destinations in combinations(range(len(slots)), len(events)):
            assigned = dict(fixed)
            valid = True
            for (tile, target), destination in zip(events, destinations):
                index, meld = slots[destination]
                if tile not in meld.tiles or target is not None and target != meld.tiles or index in assigned and assigned[index] != tile:
                    valid = False
                    break
                assigned[index] = tile
            if valid and len(assigned) == len(slots):
                possibilities.add(tuple(sorted(assigned.values())))
        if not possibilities:
            return None, "chi_claim_source_unknown"
        if len(possibilities) != 1:
            return None, "ambiguous_chi_claim_source"
        return next(iter(possibilities)), None

    def _chi_claims(self, observation: Observation, snapshot_seq: int) -> tuple[tuple[tuple[int, str], ...] | None, str | None]:
        baseline_proofs: dict[int, list[tuple[str, tuple[str, ...] | None]]] = {seat: [] for seat in range(4)}
        if self._verified_round_start is not None:
            baseline_proofs, error = self._chi_proofs(self._verified_round_start, self._round_anchor)
            if error is not None:
                return None, error
        proofs, error = self._chi_proofs(self._round_anchor, snapshot_seq)
        if error is not None:
            return None, error
        all_claims: list[tuple[int, str]] = []
        for seat, melds in enumerate(observation.melds):
            baseline = self._baseline_melds[seat]
            if len(melds) < len(baseline):
                return None, "claim_meld_mismatch"
            for index, old in enumerate(baseline):
                current = melds[index]
                if old != current and not (old.kind == "peng" and current.kind == "gang_bu"
                                           and old.tiles[0] == current.tiles[0]):
                    return None, "claim_meld_mismatch"
            for initial, events in ((True, baseline_proofs[seat]), (False, proofs[seat])):
                slots = [(index, meld) for index, meld in enumerate(melds)
                         if meld.kind == "chi" and (index < len(baseline)) == initial]
                fixed = {index: proof.tile for index, meld in slots
                         if (proof := self._confirmed.get((observation.round_no, seat, index))) is not None
                         and proof.confirmed_at <= snapshot_seq and proof.meld == meld}
                claims, error = self._match_chi_slots(slots, events, fixed)
                if error is not None:
                    return None, error
                all_claims.extend((seat, tile) for tile in claims)
        return tuple(all_claims), None

    def observe(self, observation: Observation, *, snapshot_seq: int, gap: bool = False) -> Observation:
        if type(gap) is not bool:
            raise ValueError("snapshot gap must be a boolean")
        self.begin_snapshot(observation, snapshot_seq=snapshot_seq)
        self._verify_prefix()
        self._gap_count += int(gap)
        history, history_error, ahead_of_history = (self._discard_history(observation, snapshot_seq)
                                  if self.certify_history else (None, None, False))
        resolved = replace(observation, discards=history) if history is not None else observation
        self._confirm_actions(resolved, snapshot_seq)
        start = self._round_anchor if self._verified_round_start is None else self._verified_round_start
        relevant = lambda seq: start < seq <= snapshot_seq
        if any(relevant(seq) for seq in self._conflicts):
            inventory = _unknown("conflicting_public_event")
        elif any(relevant(seq) for seq in self._invalid):
            inventory = _unknown("invalid_public_event")
        elif observation.public_inventory is not None and observation.public_inventory.status == 'unknown':
            inventory = observation.public_inventory
        elif history_error is not None and not ahead_of_history:
            inventory = _unknown(history_error)
        else:
            chi_claims, error = self._chi_claims(observation, snapshot_seq)
            inventory = (_unknown(error or history_error) if error or history_error
                         else _resolve_counts(resolved, chi_claims))
            physical = observation.public_inventory
            if (error is None and physical is not None and physical.status == 'exact'
                    and physical.counts == current_visible_counts(observation)):
                if ahead_of_history:
                    inventory = physical
                elif inventory.status == 'exact' and inventory != physical:
                    inventory = _unknown('public_inventory_mismatch')
        self._last_inventory = inventory
        return replace(observation, public_inventory=inventory,
                       public_actions=self._action_state(resolved, snapshot_seq), public_discard_history=history)

    def _action_state(self, observation, snapshot_seq):
        start = self._verified_round_start
        if (start is None or not self._covered(start, snapshot_seq) or any(
                start < seq <= snapshot_seq for seq in self._conflicts | self._invalid)):
            return get_public_actions(replace(observation, public_actions=None))
        chains, piaos, owner, supplement = [0] * 4, [0] * 4, None, None
        gangs = [0] * 4
        for seq in sorted(self._events):
            if not start < seq <= snapshot_seq:
                continue
            event = self._events[seq]
            kind, seat, tile = event['type'], event['seat'], event['tile']
            if seat not in range(4):
                continue
            if kind == 'tile_drawn':
                if seat == owner and supplement != seat:
                    owner = None
                supplement = None
            elif kind.startswith('gang_'):
                gangs[seat] += 1
                chains[seat] = None if chains[seat] is None else chains[seat] + 1
                supplement = seat
            elif kind == 'tile_discarded':
                # A chi/peng declaration preserves the chain (including unknown
                # history). Only its actual discard can break or extend it, v26.
                data = event.get('data') or {}
                if tile == '白':
                    owner = seat
                    # A white discard alone does not reveal historical baotou.
                    piao = data.get('piao', data.get('is_piao')) if isinstance(data, Mapping) else None
                    if type(piao) is bool:
                        chains[seat] = (None if chains[seat] is None else chains[seat] + 1) if piao else 0
                        piaos[seat] = (None if piaos[seat] is None else piaos[seat] + 1) if piao else 0
                    else:
                        chains[seat] = piaos[seat] = None
                else:
                    chains[seat] = piaos[seat] = 0
                    if seat == owner or isinstance(data, Mapping) and data.get('catch_play') is False:
                        owner = None
                supplement = None
        reason = 'public_chain_unknown' if None in chains or None in piaos else None
        if gangs != [sum(meld.kind.startswith('gang_') for meld in melds) for melds in observation.melds]:
            reason = 'public_action_state_mismatch'
        if (owner is not None and observation.seat != owner) != observation.god.catch_play:
            reason = 'public_action_state_mismatch'
        try:
            return PublicActionState(tuple(chains), tuple(piaos), owner, reason)
        except ValueError:
            return PublicActionState((None,) * 4, (None,) * 4, None, 'invalid_public_action_state')

    def record_action(self, observation: Observation, action: Action, *, snapshot_seq: int) -> None:
        """Register an accepted own declaration, pending snapshot confirmation."""
        if action.kind not in {ActionKind.CHI, ActionKind.PENG, ActionKind.GANG}:
            return
        if type(snapshot_seq) is not int or snapshot_seq < 0:
            raise ValueError("action snapshot sequence must be nonnegative")
        if self._current_round is None:
            self.begin_snapshot(observation, snapshot_seq=snapshot_seq)
        if observation.round_no != self._current_round or snapshot_seq < self._round_anchor:
            # A late HTTP acknowledgement belongs to its request's epoch. It
            # cannot become evidence for an identical meld in a later round.
            return
        if action.kind in {ActionKind.CHI, ActionKind.PENG}:
            phase = Phase.RESPONSE_CHI if action.kind is ActionKind.CHI else Phase.RESPONSE_PENG
            if (observation.phase is not phase or action.tile != observation.last_discard
                    or observation.seat not in observation.responding_seats
                    or action.kind is ActionKind.CHI and observation.seat != (observation.turn + 1) % 4):
                self._invalid.add(snapshot_seq + 1)
                return
        own = observation.melds[observation.seat]
        index = len(own)
        if action.kind is ActionKind.CHI:
            consumed, kind = action.tiles, "chi"
            targets = (*action.tiles, action.tile)
        elif action.kind is ActionKind.PENG:
            consumed, kind, targets = (action.tile,) * 2, "peng", (action.tile,) * 3
        elif observation.phase is Phase.RESPONSE_PENG:
            consumed, kind, targets = (action.tile,) * 3, "gang_ming", (action.tile,) * 4
        else:
            peng = next((i for i, meld in enumerate(own) if meld.kind == "peng" and meld.tiles[0] == action.tile), None)
            index = len(own) if peng is None else peng
            consumed = (action.tile,) * (4 if peng is None else 1)
            kind, targets = ("gang_an" if peng is None else "gang_bu"), (action.tile,) * 4
        try:
            expected_meld = normalize_melds((Meld(kind, targets),))[0]
            hand = list(observation.hand)
            for tile in consumed:
                hand.remove(tile)
        except ValueError:
            self._invalid.add(snapshot_seq + 1)
            return
        key = (observation.round_no, snapshot_seq, action, index)
        self._pending.setdefault(key, _OwnClaim(
            observation.round_no, observation.seat, snapshot_seq, action, index,
            expected_meld, own, tuple(hand), observation.discards[observation.seat],
        ))

    def diagnostics(self) -> dict[str, object]:
        start = self._verified_round_start
        return {
            "event_watermark": self._event_watermark,
            "snapshot_seq": self._last_snapshot_seq,
            "round_no": self._current_round,
            "inventory_round_no": self._current_round,
            "round_anchor_seq": self._round_anchor,
            "verified_round_start_seq": start,
            "round_boundary_status": "anchored" if start is None else "verified",
            "round_start_seq": self._round_anchor if start is None else start,
            "quarantined_event_count": sum(seq <= self._round_anchor and (start is None or seq <= start)
                                           for seq in self._events),
            "reassigned_event_count": len(self._reassigned),
            "event_count": len(self._events),
            "future_event_count": sum(seq > self._last_snapshot_seq for seq in self._events),
            "duplicate_events": self._duplicates,
            "conflicting_sequences": tuple(sorted(self._conflicts)),
            "invalid_sequences": tuple(sorted(self._invalid)),
            "pending_own_claims": len(self._pending),
            "confirmed_own_actions": self._confirmed_actions,
            "gap_count": self._gap_count,
            "inventory_status": self._last_inventory.status,
            "inventory_reason": self._last_inventory.reason,
        }
