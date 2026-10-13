"""I/O-owned admission, public-action wakeups and conservative window clocks."""
from __future__ import annotations

from heuristic_ai.config.models import DEFAULT_ACTION_RESERVES

from dataclasses import dataclass
from collections import deque, Counter
import sys
import threading
import time

from heuristic_ai.game.models import ActionKind, Phase
from heuristic_ai.protocol.client import RequestDeadlineExceeded
from .response_polling import action_window_key, RESPONSE_PHASES

INVENTORY_REPAIR_REASONS = frozenset({'chi_claim_source_unknown', 'ambiguous_chi_claim_source',
    'claim_source_mismatch', 'claim_discard_missing', 'public_discard_history_mismatch', 'ambiguous_discard_history'})


class PriorityStateGate:
    """One no-burst 15.5/s budget per identity, with EDF and bounded aging."""
    def __init__(self, stop, *, clock=time.monotonic, pacing_clock=time.perf_counter,
                 pacing_sleep=time.sleep, rate=15.5, priority_provider=None):
        self.stop, self.clock = stop, clock
        self.priority_provider = priority_provider
        # Only pacing uses QPC. Request/window deadlines keep their existing
        # monotonic epoch; mixing the two absolute clock domains is unsafe on
        # Python 3.12/Windows. GetTickCount64 would round each 64.5 ms interval
        # up to roughly 78 ms and reduce the configured 15.5/s to 12.8/s.
        self.pacing_clock = pacing_clock
        self.pacing_sleep = pacing_sleep
        self.interval = 1.0 / rate
        self.condition = threading.Condition()
        self.waiting = []
        self.serial = 0
        self.next_at = 0.0
        self.first_wait = {}
        self.last_rate_limit_at = None
        self.rate_limit_streak = 0

    def state_retry_delay(self, attempt_delay, retry_after_seconds):
        """Back off one identity without discarding a whole short window.

        A missing Retry-After does not ask every table to stop for one second.
        Start with two paced slots, escalating closely spaced rejections up to
        one second. Admission still enforces the same no-burst rate. Explicit
        server delays always win, including values longer than this cap.
        """
        with self.condition:
            now = self.pacing_clock()
            if self.last_rate_limit_at is None or now-self.last_rate_limit_at > 1:
                self.rate_limit_streak = 0
            else:
                self.rate_limit_streak = min(3,self.rate_limit_streak+1)
            self.last_rate_limit_at = now
            delay = min(1.0, 2*self.interval*max(attempt_delay,2**self.rate_limit_streak))
            return max(delay,retry_after_seconds or 0)

    def defer_state(self, seconds):
        with self.condition:
            self.next_at = max(self.next_at, self.pacing_clock()+max(0, seconds))
            self.condition.notify_all()

    def wait_for(self, intent):
        if intent.method != 'GET' or '/state' not in intent.path:
            if self.stop.is_set():
                from .bot import GameStopped
                raise GameStopped('I/O is stopping')
            return
        with self.condition:
            now = self.clock()
            table = (intent.path.split('?')[0], intent.purpose == 'inventory_backfill')
            first = self.first_wait.setdefault(table, now)
            self.serial += 1
            # Transport timeouts are not gameplay deadlines. In particular a
            # long poll discovering a discard must not yield to later snapshots.
            ticket = (self.serial, first, intent.priority_deadline_monotonic,
                      intent.purpose == 'inventory_backfill', intent)
            self.waiting.append(ticket)
            def priority(item):
                serial, entered, deadline, background, request = item
                if self.priority_provider is not None and not background:
                    deadline = self.priority_provider(request)
                if self.clock()-entered >= 1:
                    # Aging must compete with urgent reads, not merely join
                    # the ordinary tier that they can indefinitely displace.
                    aged = entered + 1
                    return (0, min(deadline, aged) if deadline is not None and not background else aged, serial)
                if deadline is not None and not background:
                    return (0, deadline, serial)
                return (2 if background else 1, entered, serial)
            try:
                while True:
                    now = self.clock()
                    if self.stop.is_set():
                        from .bot import GameStopped
                        raise GameStopped('I/O is stopping')
                    remaining = None if intent.deadline_monotonic is None else intent.deadline_monotonic - now
                    if remaining is not None and remaining <= 0:
                        raise RequestDeadlineExceeded('state admission expired')
                    pace_now = self.pacing_clock()
                    delay = self.next_at - pace_now
                    selected = min(self.waiting, key=priority) == ticket
                    if selected and delay <= 0:
                        self.waiting.remove(ticket)
                        self.first_wait.pop(table, None)
                        self.next_at = pace_now + self.interval
                        self.condition.notify_all()
                        return
                    if sys.platform == 'win32' and selected and delay > 0:
                        # Condition.wait's lock timeout has a coarse Windows
                        # wakeup even when the pacing clock is QPC. Python 3.11+
                        # sleep uses a high-resolution waitable timer instead.
                        # Release the queue lock and sleep in <=10 ms slices so
                        # stop/priority changes are promptly rechecked. No spin
                        # loop or system-wide timer-resolution change is used.
                        pause = min(.01, delay, remaining if remaining is not None else .01)
                        self.condition.release()
                        try:
                            self.pacing_sleep(pause)
                        finally:
                            self.condition.acquire()
                    else:
                        self.condition.wait(min(0.05, remaining if remaining is not None else 0.05,
                                                max(0.001, delay) if selected else 0.05))
            finally:
                if ticket in self.waiting:
                    self.waiting.remove(ticket)
                    self.condition.notify_all()


@dataclass(frozen=True, slots=True)
class WindowToken:
    user_id: str
    game_id: str
    generation: int
    opportunity: tuple
    fingerprint: str


class ConfirmedDiscardOrigins:
    """Public timing facts from successful controlled-seat discard requests.

    A request start precedes its server execution. Game, round, discarding
    seat, discard ordinal, tile and public meld counts bind it to the later
    snapshot. A claim removes its source tile, so river ordinals can be reused.
    No private hand is stored or shared, and only the latest round is kept.
    """
    def __init__(self):
        self._games = {}
        self._lock = threading.Lock()

    def record(self, game_id, observation, tile, requested_at):
        if (observation is None or observation.phase is not Phase.DRAW
                or observation.turn != observation.seat):
            return
        key = (observation.seat, len(observation.discards[observation.seat]) + 1, tile,
               tuple(len(melds) for melds in observation.melds))
        with self._lock:
            round_no, origins = self._games.get(game_id, (observation.round_no, {}))
            if observation.round_no < round_no:
                return
            if observation.round_no > round_no:
                round_no, origins = observation.round_no, {}
            origins.setdefault(key, requested_at)
            self._games[game_id] = round_no, origins

    def lookup(self, game_id, observation):
        if observation.phase not in RESPONSE_PHASES or observation.last_discard is None:
            return None
        with self._lock:
            round_no, origins = self._games.get(game_id, (None, {}))
            if round_no != observation.round_no:
                return None
            return origins.get((observation.turn, len(observation.discards[observation.turn]),
                                observation.last_discard, tuple(len(melds) for melds in observation.melds)))


@dataclass(frozen=True, slots=True)
class PublicActionHint:
    version: int
    round_no: int
    user_id: str
    seat: int
    kind: str
    tile: str | None
    ordinal: int | None
    requested_at: float
    meld_counts: tuple[int, ...]

    def covered_by(self, observation):
        if observation is None or observation.round_no < self.round_no:
            return False
        if observation.round_no > self.round_no:
            return True
        if self.kind == 'discard':
            if any(len(melds) < count for melds, count in zip(observation.melds, self.meld_counts)):
                return False
            pile = observation.discards[self.seat]
            return len(pile) > self.ordinal and pile[self.ordinal] == self.tile
        if self.kind == 'hu':
            return observation.phase in {Phase.SETTLED, Phase.FINISHED}
        melds = observation.melds[self.seat]
        if len(melds) <= self.ordinal:
            return False
        meld = melds[self.ordinal]
        kind_matches = meld.kind.startswith('gang') if self.kind == 'gang' else meld.kind == self.kind
        return kind_matches and (self.tile is None or self.tile in meld.tiles)


class _ActionHintGroup:
    def __init__(self):
        self.condition = threading.Condition()
        self.version = 0
        self.hints = deque(maxlen=128)


class PublicActionSignals:
    """Bounded, game-local wakeups from acknowledged public mutations.

    Versions are local receipt order, never server event cursors. Keep every
    recent hint so out-of-order ACK arrival cannot hide a newer public action.
    """
    def __init__(self):
        self._groups = {}
        self._lock = threading.Lock()

    def _group(self, game_id):
        with self._lock:
            group = self._groups.get(game_id)
            if group is None:
                group = self._groups[game_id] = _ActionHintGroup()
            return group

    def since(self, game_id, version):
        group = self._group(game_id)
        with group.condition:
            gap = bool(group.hints and version < group.hints[0].version-1)
            return group.version, tuple(h for h in group.hints if h.version > version), gap

    def version(self, game_id):
        group = self._group(game_id)
        with group.condition:
            return group.version

    def publish(self, game, observation, body, requested_at):
        kind = body.get('action') if isinstance(body, dict) else None
        if observation is None or kind not in {'discard','chi','peng','gang','hu'}:
            return
        tile, ordinal = None, None
        if kind == 'discard':
            tile, ordinal = body.get('tile'), len(observation.discards[observation.seat])
        elif kind in {'chi','peng'}:
            tile, ordinal = observation.last_discard, len(observation.melds[observation.seat])
        elif kind == 'gang':
            previous = observation.melds[observation.seat]
            upgraded = next((i for i,m in enumerate(previous) if m.kind=='peng' and body.get('tile') in m.tiles), None)
            ordinal = len(previous) if upgraded is None else upgraded
            # A concealed kong's tile need not be disclosed to other actors.
            if upgraded is not None or observation.phase is Phase.RESPONSE_PENG:
                tile = body.get('tile')
        group = self._group(game.game_id)
        with group.condition:
            group.version += 1
            group.hints.append(PublicActionHint(group.version, observation.round_no, game.user_id,
                                               observation.seat, kind, tile, ordinal, requested_at,
                                               tuple(len(melds) for melds in observation.melds)))
            group.condition.notify_all()

    def wait(self, game_id, version, stop, seconds):
        group = self._group(game_id)
        deadline = time.monotonic()+seconds
        with group.condition:
            while not stop.is_set():
                if group.version != version:
                    return False
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    return False
                group.condition.wait(min(.05, remaining))
        return True


class ManagedState:
    """Never derive private hands from notifications or acknowledge unseen events.

    Snapshot/event inventory validation stays in the shared game loop. This
    policy owns the phase wakeup clock and confirmed public-action signals.
    """
    def __init__(self, game, memory, *, fence=False, discard_origin=None, coordinated_reads=None,
                 public_signals=None, notifications=None, action_reserves=DEFAULT_ACTION_RESERVES):
        self.game, self.memory = game, memory
        self.action_reserves = action_reserves
        self.action_reserve_ms = action_reserves.discard_ms
        self.fence = fence
        self.discard_origin = discard_origin
        self.coordinated_reads = coordinated_reads
        self.public_signals = public_signals
        self.notifications = notifications
        self.next_phase_probe = None
        self.last_state_transport_started_at = None
        self.snapshot_transport_started_at = None
        self.public_version = 0
        self.read_public_version = 0
        self.wait_public_version = 0
        self.snapshot_seq = 0
        self.observation = None
        self.active = False
        self.boundary = None
        self.search_deadline = None
        self.action_deadline = None
        self.basis = 'unknown_initial_window'
        self.confidence = 'unknown'
        self.request_started = None
        self.last_state_request_started_at = None
        self.received_at = None
        self.snapshot_received_at = None
        self.snapshot_request_started_at = None
        self.estimated_window_start = None
        self.causal = None
        self.causal_basis = None
        self.after_submit = False
        self.submitted_pass_seq = None
        self.submitted_discard_seq = None
        self.generation = 0
        self.last_inventory_catchup_signature = None
        self.inventory_retry_at = 0.
        self.inventory_retry_count = 0
        if memory.window_evidence is not None:
            for name, value in memory.window_evidence.items():
                setattr(self, name, value)

    def _snapshot_predecessor_start(self):
        # A successful predecessor snapshot was produced after its transport
        # began, not while it was still waiting for identity admission. This
        # bound is usable only where the caller already proves a new event.
        requested = self.snapshot_request_started_at
        sent, received = self.snapshot_transport_started_at, self.snapshot_received_at
        if (requested is not None and sent is not None and received is not None
                and requested <= sent <= received):
            return sent
        return requested

    def events(self, response, request_seq, started, received):
        self.request_started, self.received_at = started, received
        if response.get('gap'):
            self.causal = None
            self.causal_basis = None
            return
        events = response.get('events', ())
        if not isinstance(events, list):
            events = ()
        relevant = [e for e in events if isinstance(e, dict) and type(e.get('seq')) is int and e['seq'] > self.snapshot_seq and
                    e.get('type') in {'tile_discarded', 'tile_drawn', 'chi', 'peng', 'gang',
                                     'gang_ming', 'gang_an', 'gang_bu', 'round_started'}]
        # A delta identifies the event kind. Its request start is a heuristic
        # origin, not an authoritative server timestamp. Snapshot tails may
        # contain old events and cannot establish a fresh origin.
        if request_seq > 0 and relevant and not response.get('gap'):
            # Only the last observed state is known to precede a new event;
            # a delta request itself may start after that event occurred.
            origin = self._snapshot_predecessor_start()
            self.causal = (origin if origin is not None else started, relevant[-1])
            self.causal_basis = 'previous_snapshot_event_bound' if origin is not None else None
        elif self.notifications is not None and relevant:
            signal = self.notifications.snapshot()
            event = relevant[-1]
            if (signal['notification_connected'] and not signal['notification_pending_initial']
                    and event.get('type') == 'tile_discarded'
                    and event['seq'] == signal['notification_pending_seq']):
                # A continuous notification associated with this exact discard
                # supplies a local estimate, not a server timestamp. Keep 100ms
                # headroom; first/reconnect/coalesced-unknown frames cannot do so.
                self.causal = (signal['notification_pending_received_monotonic']-.1, event)
                self.causal_basis = 'notification_discard_estimate'

    def _matching_causal(self, observation):
        if self.causal is None:
            return None
        if self.observation is not None and self.observation.round_no != observation.round_no:
            return None
        started, event = self.causal
        kind = event.get('type')
        if observation.phase in RESPONSE_PHASES:
            if (kind != 'tile_discarded' or event.get('seat') != observation.turn
                    or event.get('tile') != observation.last_discard):
                return None
        elif (kind not in {'tile_drawn', 'chi', 'peng', 'gang', 'gang_ming', 'gang_an', 'gang_bu'}
                or event.get('seat') != observation.turn):
            return None
        return started, kind

    def observe(self, observation, server_seq, received):
        previous = self.observation
        previous_snapshot_seq = self.snapshot_seq
        previous_transport_start = self.snapshot_transport_started_at
        notification_origin = None
        notification_basis = 'notification_discard_estimate'
        if (self.notifications is not None and previous is not None
                and previous.phase is Phase.DRAW and observation.phase in RESPONSE_PHASES
                and previous.round_no == observation.round_no and previous.turn == observation.turn
                and Counter(previous.hand) == Counter(observation.hand) and previous.melds == observation.melds):
            signal = self.notifications.snapshot()
            source = observation.turn
            expected = list(previous.discards)
            expected[source] = (*expected[source], observation.last_discard)
            if (tuple(expected) == observation.discards and signal['notification_connected']
                    and not signal['notification_pending_initial']
                    and signal['notification_pending_seq'] == self.snapshot_seq+1
                    and server_seq >= signal['notification_pending_seq']):
                # seq=0 has no event tail on the current server. Consecutive
                # authoritative boards bind the first new notification to this
                # source's discard (or its immediately preceding timeout).
                notification_origin = signal['notification_pending_received_monotonic']-.1
        previous_snapshot_request = self._snapshot_predecessor_start()
        board_origin = None
        if (previous is not None and previous.round_no == observation.round_no
                and previous.seat == observation.seat and previous.dealer == observation.dealer
                and server_seq > previous_snapshot_seq
                and previous.phase in {Phase.DRAW, *RESPONSE_PHASES}
                and observation.phase in RESPONSE_PHASES and observation.last_discard is not None
                and previous.melds == observation.melds):
            expected = list(previous.discards)
            expected[observation.turn] = (*expected[observation.turn], observation.last_discard)
            if tuple(expected) == observation.discards:
                board_origin = previous_snapshot_request
                if (previous.phase is Phase.RESPONSE_CHI
                        and previous.wall_remaining == observation.wall_remaining + 1
                        and previous_snapshot_seq + 2 <= server_seq):
                    receipt = getattr(self.notifications, 'event_received_at', None)
                    noticed = receipt(previous_snapshot_seq + 2) if callable(receipt) else None
                    if noticed is not None:
                        # A new draw and discard follow Chi. The second new
                        # event is at or before that discard, even when an old
                        # Chi pass/timeout precedes the draw. Keep the existing
                        # receipt-latency headroom and an estimated confidence.
                        notification_origin = noticed - .1
                        notification_basis = 'notification_transition_estimate'
        round_start_origin = None
        if (previous is not None and previous.phase is Phase.SETTLED
                and observation.round_no == previous.round_no + 1
                and server_seq == self.snapshot_seq and observation.phase is Phase.DRAW
                and observation.turn == observation.seat == observation.dealer
                and len(observation.hand) == 14 and observation.drawn_tile is not None
                and observation.wall_remaining == 83
                and not any(observation.discards) and not any(observation.melds)
                and observation.god.chain_count == 0 and not observation.god.catch_play
                and previous_snapshot_request is not None
                and self.snapshot_transport_started_at == previous_snapshot_request
                and self.snapshot_received_at is not None and self.snapshot_received_at <= received
                and self.request_started is not None and previous_snapshot_request <= self.request_started <= received
                and previous_snapshot_request + self.game.rules.discard_timeout_seconds
                    > received + self.action_reserves.discard_ms / 1000):
            # The settled predecessor was served before the next initial draw.
            # Its transport start is a conservative lower bound, never the new
            # snapshot's arrival time. Keep fallback if this bound has no room
            # for computation plus the existing send reserve.
            round_start_origin = previous_snapshot_request
        if self.uses_public_wakeups:
            # Every hint already acknowledged when this GET started predates
            # the authoritative snapshot. Later ACKs need public-effect checks.
            self.public_version = max(self.public_version, self.read_public_version)
        self.snapshot_received_at = received
        self.snapshot_request_started_at = self.request_started
        self.snapshot_transport_started_at = self.last_state_transport_started_at
        self.snapshot_seq = max(self.snapshot_seq, server_seq)
        if self.notifications is not None:
            self.notifications.covered(self.snapshot_seq)
        opportunity = action_window_key(observation)
        changed = previous is None or opportunity != action_window_key(previous)
        seconds = {Phase.RESPONSE_PENG: self.game.rules.peng_timeout_seconds,
                   Phase.RESPONSE_CHI: self.game.rules.chi_timeout_seconds}.get(
                       observation.phase, self.game.rules.discard_timeout_seconds)
        if changed:
            start = self.request_started if self.request_started is not None else received
            basis, confidence = 'unknown_initial_window', 'unknown'
            origin = self.discard_origin(observation) if self.discard_origin is not None else None
            causal = self._matching_causal(observation)
            same_discard_transition = (previous is not None and previous.round_no == observation.round_no
                and previous.phase is Phase.RESPONSE_PENG and observation.phase is Phase.RESPONSE_CHI
                and previous.last_discard == observation.last_discard and previous.discards == observation.discards
                and self.boundary is not None
                and (self.last_state_transport_started_at is None or self.confidence != 'unknown')
                and (not self.uses_public_wakeups or self.confidence != 'unknown'))
            if origin is not None:
                start = origin + (self.game.rules.peng_timeout_seconds
                                  if observation.phase is Phase.RESPONSE_CHI else 0)
                basis, confidence = 'confirmed_discard_request', 'medium'
            elif round_start_origin is not None:
                start = round_start_origin
                basis, confidence = 'previous_round_settled_bound', 'medium'
            elif notification_origin is not None:
                start = notification_origin + (self.game.rules.peng_timeout_seconds
                    if observation.phase is Phase.RESPONSE_CHI else 0)
                basis, confidence = notification_basis, 'medium'
            elif board_origin is not None:
                start = board_origin + (self.game.rules.peng_timeout_seconds
                    if observation.phase is Phase.RESPONSE_CHI else 0)
                basis = ('previous_snapshot_board_bound_then_peng' if observation.phase is Phase.RESPONSE_CHI
                         else 'previous_snapshot_board_bound')
                confidence = 'medium'
            elif same_discard_transition:
                start = self.boundary
                basis, confidence = 'observed_peng_boundary', 'medium'
            elif causal is not None and not (self.uses_public_wakeups and observation.phase in RESPONSE_PHASES):
                start, event = causal
                basis, confidence = self.causal_basis or 'covered_' + event, 'medium'
                if event == 'tile_discarded' and observation.phase is Phase.RESPONSE_CHI:
                    # Peng runs for its full configured duration, including
                    # pass-only windows. A client can miss that snapshot while
                    # its delta/full read waits for admission. Starting Chi at
                    # the discard would expire it one whole phase too early.
                    start += self.game.rules.peng_timeout_seconds
                    basis = (self.causal_basis + '_then_peng' if self.causal_basis else 'covered_tile_discarded_then_peng')
            elif self.uses_public_wakeups and observation.phase in RESPONSE_PHASES:
                if (previous is not None and previous.round_no == observation.round_no
                        and previous_snapshot_request is not None):
                    # The previous snapshot predates this transition; the
                    # current request may start after an automatic discard.
                    # Its start cannot grant a fresh full response window.
                    start = previous_snapshot_request
                    previous_was_this_peng = (previous.phase is Phase.RESPONSE_PENG
                        and previous.turn == observation.turn
                        and previous.last_discard == observation.last_discard
                        and previous.discards == observation.discards)
                    if observation.phase is Phase.RESPONSE_CHI and not previous_was_this_peng:
                        start += self.game.rules.peng_timeout_seconds
                    basis, confidence = 'previous_snapshot_request', 'medium'
                else:
                    # No bound before this response window is known. Leave
                    # it unsubmitted and seek a new authoritative snapshot.
                    start = received-seconds
            elif previous is not None and previous.round_no == observation.round_no:
                if self.after_submit or previous.phase != observation.phase:
                    basis, confidence = 'observed_transition_request', 'medium'
            self.basis, self.confidence = basis, confidence
            old = self.memory.window_deadlines.get(opportunity)
            estimate = start + seconds
            self.boundary = estimate
            if (observation.phase in RESPONSE_PHASES and origin is None
                    and self.snapshot_transport_started_at is not None
                    and not basis.startswith(('notification_discard_estimate', 'notification_transition_estimate'))
                    and basis not in {'previous_snapshot_event_bound', 'previous_snapshot_event_bound_then_peng',
                                      'previous_snapshot_request', 'previous_snapshot_board_bound',
                                      'previous_snapshot_board_bound_then_peng'}
                    and not (same_discard_transition and self.basis == 'observed_peng_boundary'
                             and self.confidence != 'unknown')):
                # Keep a short budget tied to this actual snapshot transport.
                # The phase origin remains unknown, so search falls back.
                estimate = self.snapshot_transport_started_at + .2
                self.boundary = self.snapshot_transport_started_at + seconds
                self.basis, self.confidence = 'snapshot_send_freshness', 'unknown'
            self.action_deadline = min(old, estimate) if old is not None else estimate
            self.memory.window_deadlines[opportunity] = self.action_deadline
            self.after_submit = False
            self.submitted_pass_seq = None
            self.submitted_discard_seq = None
        else:
            self.action_deadline = self.memory.window_deadlines[opportunity]
        sent = self.snapshot_transport_started_at
        if (observation.phase in RESPONSE_PHASES and sent is not None
                and self.snapshot_request_started_at is not None
                and self.snapshot_request_started_at <= sent <= received < sent + .2
                and (previous_transport_start is None or sent > previous_transport_start)
                and self.action_deadline <= sent
                and self.basis != 'confirmed_discard_request'):
            # The newer authoritative snapshot was requested after this local
            # bound expired and still reports the response phase. Revalidate
            # only the existing 200ms freshness budget. Accepted or uncertain
            # submissions stay fenced; an explicitly unsent window can retry.
            self.action_deadline = sent + .2
            self.basis, self.confidence = 'snapshot_send_freshness_revalidated', 'unknown'
            self.memory.window_deadlines[opportunity] = self.action_deadline
            self.memory.unsubmitted_action_windows.discard(opportunity)
        self.estimated_window_start = (None if self.basis.startswith('snapshot_send_freshness')
                                      else self.action_deadline - seconds)
        self.action_reserve_ms = (self.action_reserves.response_ms if observation.phase in RESPONSE_PHASES
                                  else self.action_reserves.discard_ms)
        self.search_deadline = min(received + self.game.decision_budget_ms / 1000,
                                   self.action_deadline - self.action_reserve_ms / 1000)
        if self.confidence == 'unknown':
            self.search_deadline = min(self.search_deadline, received)
        self.active = observation.phase in RESPONSE_PHASES
        self.observation = observation
        self.next_phase_probe = received + .2
        self._advance_covered_public_hints()
        self.causal = None
        self.causal_basis = None
        self.memory.window_evidence = {name: getattr(self, name) for name in
            ('observation', 'snapshot_seq', 'snapshot_request_started_at', 'snapshot_transport_started_at',
             'snapshot_received_at', 'boundary', 'basis', 'confidence',
              'action_deadline', 'search_deadline', 'after_submit', 'submitted_pass_seq',
              'submitted_discard_seq')}
        if self.fence:
            self.memory.uncertain_action_windows.add(opportunity)
            self.fence = False

    def _needs_chi_boundary_snapshot(self):
        observation = self.observation
        return (observation is not None and observation.phase is Phase.RESPONSE_PENG
                and observation.seat == (observation.turn + 1) % 4
                and observation.seat in observation.responding_seats)

    @property
    def action_pending(self):
        observation = self.observation
        if observation is None or not (
                observation.phase is Phase.DRAW and observation.turn == observation.seat
                or observation.phase in RESPONSE_PHASES and observation.seat in observation.responding_seats):
            return False
        opportunity = action_window_key(observation)
        return not any(opportunity in windows for windows in (
            self.memory.accepted_action_windows, self.memory.uncertain_action_windows,
            self.memory.unsubmitted_action_windows))

    @property
    def response_action_pending(self):
        return self.observation is not None and self.observation.phase in RESPONSE_PHASES and self.action_pending



    @property
    def uses_public_wakeups(self):
        return bool(self.public_signals is not None and self.coordinated_reads is not None
                    and self.coordinated_reads())

    def begin_read(self):
        self.last_state_request_started_at = None
        self.last_state_transport_started_at = None
        if self.uses_public_wakeups:
            self.read_public_version = self.public_signals.version(self.game.game_id)

    def _advance_covered_public_hints(self):
        if not self.uses_public_wakeups:
            return
        _, hints, gap = self.public_signals.since(self.game.game_id, self.public_version)
        if not gap:
            for hint in hints:
                if not hint.covered_by(self.observation):
                    break
                self.public_version = hint.version

    def has_unseen_public_change(self):
        if not self.uses_public_wakeups:
            return False
        self._advance_covered_public_hints()
        _, hints, gap = self.public_signals.since(self.game.game_id, self.public_version)
        return gap or bool(hints)

    def _notification_read_cursor(self):
        observation = self.observation
        # An acknowledged pass produces an event before a new action can open.
        # Another seat's Chi needs a claim/draw and then a discard. This cursor
        # schedules reads only; snapshot/event watermarks stay intact.
        defer_first = self.submitted_pass_seq == self.snapshot_seq
        if (not self.uses_public_wakeups and observation is not None and observation.phase is Phase.DRAW
                and observation.turn == observation.seat):
            # Our acknowledged normal discard itself gives us no new action.
            # Defer only that event; a following claim or round end still wakes.
            defer_first |= self.submitted_discard_seq == self.snapshot_seq
        if observation is not None and observation.seat != (observation.turn + 1) % 4:
            defer_first |= (observation.phase is Phase.RESPONSE_CHI
                and observation.seat not in observation.responding_seats)
            # Once our Peng pass is acknowledged, the next event cannot be our
            # claim. Other seats need a claim/draw and then a discard to open a
            # new action for us. The next seat keeps its timed Chi discovery.
            defer_first |= (observation.phase is Phase.RESPONSE_PENG
                and (observation.seat not in observation.responding_seats
                     or self.submitted_pass_seq is not None))
        return self.snapshot_seq + int(defer_first)

    def _read_notification_signal(self):
        signal = self.notifications.snapshot()
        cursor = self._notification_read_cursor()
        received = (signal.get('notification_pending_received_monotonic')
                    or signal.get('notification_received_monotonic'))
        if signal.get('notification_pending_seq') is not None and signal['notification_pending_seq'] <= cursor:
            receipt = getattr(self.notifications, 'event_received_at', None)
            received = receipt(cursor + 1) if callable(receipt) else None
        return signal, cursor, received

    def wait_for_read(self, stop, seconds):
        if (not self.uses_public_wakeups and self.notifications is not None
                and self.notifications.snapshot()['notification_connected']):
            signal, cursor, _ = self._read_notification_signal()
            if (seconds <= .05 and self.observation is not None
                    and self.observation.phase in RESPONSE_PHASES and not self.response_action_pending
                    and signal['notification_seq'] > cursor):
                # A bounded coalescing delay must not immediately wake on the
                # same already-pending frame that scheduled it. New frames are
                # included by the subsequent snapshot; stop remains prompt.
                return stop.wait(seconds)
            return self.notifications.wait(cursor, stop, seconds)
        if not self.uses_public_wakeups:
            return stop.wait(seconds)
        return self.public_signals.wait(self.game.game_id, self.wait_public_version, stop, seconds)

    def request(self, cursor, now, *, immediate_full=False):
        if immediate_full or self.observation is None:
            return 0, 0.0
        if self._settled_event_read(cursor):
            # Dealing the next round has no SSE event. A positive cursor can
            # wait for that authoritative snapshot, using only the existing
            # consumed/bootstrap cursor supplied by the game loop.
            return cursor, 0.0
        if (self.observation.phase is Phase.DRAW and self.action_pending
                and self.action_deadline is not None and now < self.action_deadline):
            # A pending/429 read did not finish our turn. The next notification
            # might be the server timeout; retain normal admission, but read now.
            return 0, 0.0
        # With all four identities present, acknowledged public mutations wake
        # the existing bounded phase schedule. SSE also announces every PASS;
        # treating those as new action opportunities exhausts shared state quota.
        # Keep SSE for single-identity discovery and window-origin evidence.
        if self.notifications is not None and not self.uses_public_wakeups:
            signal, cursor, received = self._read_notification_signal()
            if signal['notification_connected']:
                if signal['notification_seq'] > cursor:
                    if self.observation.phase in RESPONSE_PHASES and not self.response_action_pending:
                        if received is not None:
                            return 0, min(.05, max(0, received + .05 - now))
                    return 0, 0.0
                # A Chi boundary may have no new event sequence. Only the
                # next seat needs this probe, including after its Peng pass.
                if self._needs_chi_boundary_snapshot():
                    if self.boundary is not None and self.confidence != 'unknown' and now < self.boundary:
                        return 0, self.boundary + .015 - now
                    return 0, max(0, (self.next_phase_probe or now)-now)
                return 0, 2.0
            # No notification connection: bounded full-snapshot discovery;
            # never reintroduce delta+snapshot as the mandatory critical path.
            return 0, .1
        if self.uses_public_wakeups:
            self.wait_public_version = self.public_signals.version(self.game.game_id)
            if self.has_unseen_public_change():
                return 0, 0.0
            if self._needs_chi_boundary_snapshot() or (self.observation.phase is Phase.RESPONSE_CHI
                    and self.observation.seat in self.observation.responding_seats):
                boundary = self.boundary or now
                return 0, max(0, boundary+.015-now) if now < boundary else .08
            if self.observation.phase is Phase.RESPONSE_PENG and self.boundary is not None:
                next_check = self.boundary+self.game.rules.chi_timeout_seconds+.5
                return 0, max(.05, next_check-now) if now < next_check else .5
            return 0, .5
        coordinated = self.coordinated_reads is not None and self.coordinated_reads()
        if self.active and (not coordinated or self._needs_chi_boundary_snapshot()):
            # Keep the prior schedule without four controlled seats. In a
            # coordinated game, only the eligible next seat needs a timed
            # snapshot for Chi's transition without a new event sequence.
            boundary = self.boundary or now
            return 0, max(0, boundary + 0.015 - now) if now < boundary else 0.08
        return cursor, 0.0

    def note_read(self, received):
        self.received_at = received

    def inventory_catchup_deadline(self, now, *, background=False):
        """Keep background admission separate from the foreground repair wait."""
        # Background reads must survive the gate's one-second aging threshold
        # plus a paced slot and transport. The action loop still waits at most
        # 150 ms for optional repair; extending this request does not renew it.
        deadline = now + (1.2 if background else .15 if self.notifications is not None else .5)
        if (self._needs_chi_boundary_snapshot() or (self.observation is not None
                and self.observation.phase is Phase.RESPONSE_CHI
                and self.observation.seat in self.observation.responding_seats)):
            if self.boundary is not None:
                deadline = min(deadline, self.boundary - .05)
        return deadline if deadline > now + .05 else None

    def _inventory_signature(self):
        if self.observation is None:
            return None
        return self.observation.round_no, self.observation.melds

    def needs_inventory_catchup(self, progress, reason):
        if not (self.uses_public_wakeups or self.notifications is not None) or not progress.needs_backfill:
            return False
        if time.monotonic() < self.inventory_retry_at:
            return False
        # The guide bounds retained deltas at 256 events. Read with headroom,
        # even when no chi currently requires provenance; never advance the
        # consumed cursor from a snapshot to make that lag disappear.
        if progress.snapshot_seq - progress.fetch_cursor >= 128:
            return True
        if self.observation is not None and self.observation.phase in {Phase.SETTLED, Phase.FINISHED}:
            return False
        return (reason in INVENTORY_REPAIR_REASONS
                and self._inventory_signature() != self.last_inventory_catchup_signature)

    def inventory_catchup_failed(self):
        self.inventory_retry_count += 1
        self.inventory_retry_at = time.monotonic() + min(1., .1 * 2 ** min(self.inventory_retry_count - 1, 4))

    def inventory_catchup_completed(self):
        # Re-reading later events cannot recover an already missing old claim.
        # Every new/changed meld can need history proof, not only a chi.
        # Try again for that board/round, or for retention maintenance above.
        self.last_inventory_catchup_signature = self._inventory_signature()
        self.inventory_retry_count = 0
        self.inventory_retry_at = 0.

    def _settled_event_read(self, cursor):
        return (cursor > 0 and self.notifications is not None and not self.uses_public_wakeups
                and self.observation is not None and self.observation.phase is Phase.SETTLED)

    def read_deadline(self, cursor, now):
        deadline = now + (1.2 if cursor == 0 else 2.0)
        if cursor == 0 and self.uses_public_wakeups:
            _, hints, _ = self.public_signals.since(self.game.game_id, self.public_version)
            bounds = [hint.requested_at+self.game.rules.peng_timeout_seconds-.1 for hint in hints
                      if hint.kind == 'discard' and hint.tile != '白'
                      and hint.user_id != self.game.user_id
                      and (self.observation is None or hint.round_no >= self.observation.round_no)]
            bounds = [bound for bound in bounds if now < bound]
            if bounds:
                return min(deadline, max(bounds)), True
        boundary = self.boundary
        if self.confidence == 'unknown' and self.snapshot_transport_started_at is not None:
            # A fresh Peng snapshot proves its phase had not ended when this
            # GET began transport. Its following Chi cannot expire before that
            # time plus the Chi duration. An obsolete local estimate must not
            # remove this read's priority after freshness revalidation.
            boundary = self.snapshot_transport_started_at
        own_chi_boundary = (cursor == 0 and self._needs_chi_boundary_snapshot()
            and boundary is not None
            and boundary <= now < boundary + self.game.rules.chi_timeout_seconds - 0.05)
        if own_chi_boundary:
            deadline = min(deadline, boundary + self.game.rules.chi_timeout_seconds - 0.05)
        return deadline, own_chi_boundary

    def state_read_priority(self, cursor, now):
        """Re-evaluate urgency while a state read waits for shared admission."""
        deadline, own_boundary = self.read_deadline(cursor, now)
        # Admission must leave the existing response reserve before this
        # read's unchanged hard cutoff, including timed Chi discovery.
        priorities = [deadline - self.action_reserves.response_ms / 1000] if own_boundary else []
        if self._settled_event_read(cursor) and self.snapshot_received_at is not None:
            # Do not starve the read that must discover an eventless deal.
            # Keep the existing two-second positive-cursor request limit, so a
            # held response cannot make a new draw's request-start bound stale.
            priorities.append(self.snapshot_received_at + 1.2)
        if self.action_pending and self.action_deadline is not None and now < self.action_deadline:
            priorities.append(self.search_deadline if self.search_deadline is not None else self.action_deadline)
        discovery = self.discovery_priority(now)
        if discovery is not None:
            priorities.append(discovery)
        return min(priorities) if priorities else None

    def discovery_priority(self, now=None):
        now = time.monotonic() if now is None else now
        if self.uses_public_wakeups:
            _, hints, _ = self.public_signals.since(self.game.game_id, self.public_version)
            # A claim and its mandatory discard can open a new Peng window
            # while our last snapshot still shows an already handled response.
            # This changes read urgency only; the hint never authorizes a play.
            deadlines = [hint.requested_at + self.game.rules.peng_timeout_seconds
                         - self.action_reserves.response_ms / 1000 for hint in hints
                         if hint.kind == 'discard' and hint.tile != '白'
                         and hint.user_id != self.game.user_id
                         and (self.observation is None or hint.round_no >= self.observation.round_no)]
            future = [deadline for deadline in deadlines if now < deadline]
            if future:
                return min(future)
        # A handled response or our own submitted draw can advance into a new
        # window while this read waits for admission. Keep discovery urgent
        # until an authoritative snapshot replaces that old phase.
        if (self.notifications is None or self.observation is None
                or not (self.observation.phase in RESPONSE_PHASES
                        or self.observation.phase is Phase.DRAW)):
            return None
        signal, cursor, received = self._read_notification_signal()
        if (not signal['notification_connected'] or signal['notification_pending_initial']
                or signal['notification_pending_seq'] is None
                or signal['notification_seq'] <= cursor or received is None):
            return None
        deadline = received + self.game.rules.peng_timeout_seconds - .1
        # Discovery must leave the existing response reserve for the action.
        # An overdue read target stays urgent until the window bound expires.
        if now >= deadline:
            return None
        return deadline - self.action_reserves.response_ms / 1000

    def skip_delta_snapshot(self, response):
        events = response.get('events')
        watermark = response.get('seq')
        if not response.get('gap') and type(watermark) is int and watermark <= self.snapshot_seq:
            return True
        if response.get('gap') or not isinstance(events, list) or not events:
            return False
        coordinated = self.coordinated_reads is not None and self.coordinated_reads()
        if self.observation is None or not coordinated:
            return (all(isinstance(e, dict) and e.get('type') == 'pass' for e in events)
                    and (self.boundary is None or time.monotonic() < self.boundary))
        own_seat = self.observation.seat
        def irrelevant_to_next_private_action(event):
            if not isinstance(event, dict):
                return False
            kind, seat = event.get('type'), event.get('seat')
            if kind == 'pass':
                return True
            if kind == 'timeout':
                data = event.get('data')
                return (isinstance(data, dict) and data.get('kind') == 'response'
                        and data.get('window') in {'peng', 'chi'})
            if type(seat) is not int or not 0 <= seat < 4:
                return False
            return ((kind == 'tile_drawn' and seat != own_seat)
                    or (kind == 'tile_discarded' and seat == own_seat))
        # The caller must ingest this entire delta first. This does not move
        # snapshot_seq, synthesize a private hand, or replace a gap recovery.
        return all(irrelevant_to_next_private_action(event) for event in events)

    def submitted(self, action):
        self.after_submit = True
        self.submitted_pass_seq = self.snapshot_seq if action.kind is ActionKind.PASS else None
        self.submitted_discard_seq = (self.snapshot_seq
            if action.kind is ActionKind.DISCARD and action.tile != '白'
            and self.observation is not None and self.observation.phase is Phase.DRAW
            and self.observation.turn == self.observation.seat else None)
        if self.memory.window_evidence is not None:
            self.memory.window_evidence.update(after_submit=True, submitted_pass_seq=self.submitted_pass_seq,
                                               submitted_discard_seq=self.submitted_discard_seq)

    def details(self):
        return {**(self.notifications.snapshot() if self.notifications is not None else {'notification_seq': 0}),
                'window_deadline_monotonic': self.action_deadline,
                'search_deadline_monotonic': self.search_deadline, 'deadline_basis': self.basis,
                'deadline_confidence': self.confidence, 'action_reserve_ms': self.action_reserve_ms,
                'snapshot_received_monotonic': self.snapshot_received_at,
                'snapshot_request_started_monotonic': self.snapshot_request_started_at,
                'snapshot_transport_started_monotonic': self.snapshot_transport_started_at,
                'estimated_window_start_monotonic': self.estimated_window_start,
                'coordinated_state_reads': bool(self.coordinated_reads is not None and self.coordinated_reads()),
                'local_public_action_version': self.public_version,
                'local_public_wakeups': self.uses_public_wakeups}
