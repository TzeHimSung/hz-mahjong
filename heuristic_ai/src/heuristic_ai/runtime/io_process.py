"""One run-owned I/O process; strategy sessions and audit graphs stay in parent.

Only bounded messages enter either pipe. An overflowing queue is a run failure,
never a dropped decision/audit. The reader can cancel a search even while its
game thread is inside a strategy primitive.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import fields, replace
import multiprocessing
import hashlib
import logging
import pickle
import queue
import threading
import traceback
import time
import uuid
from pathlib import Path

from heuristic_ai.artifacts.models import DecisionTrace
from heuristic_ai.config.models import DEFAULT_ACTION_RESERVES
from heuristic_ai.game.models import ActionKind, Phase
from heuristic_ai.game.serialization import action_to_json
from heuristic_ai.protocol.client import ApiClient, ApiError, RequestDeadlineExceeded, StateChangedBeforeSend
from heuristic_ai.protocol.httpx_transport import HttpxRuntime, HttpxTransport
from heuristic_ai.strategy.models import Decision, DecisionResult
from .managed_state import ConfirmedDiscardOrigins, ManagedState, PriorityStateGate, PublicActionSignals, WindowToken
from .recovery_context import ActionMemory
from .io_channel import IoFailure, _PipeWriter, _read_messages
from .response_polling import action_window_key

def _encode_decision(result, ticket=None, source=None, depth=None, prefetch=None):
    # Audit graphs stay in the parent; IPC carries only the selected action.
    if depth is None:
        depth = 1 if result.execution_source in {"completed_search", "prefetch_hit"} else 0
    metadata = {"native_enabled": result.decision.native_enabled, **(prefetch or {})}
    return (result.decision.action, result.decision.reason, result.execution_source,
            result.elapsed_ms, result.fallback_used, result.fallback_reason,
            ticket, source or result.execution_source, depth, metadata)


def _encode_publication(publication, ticket):
    result = DecisionResult(Decision(publication.action, "published_checkpoint",
                                     native_enabled=publication.native_enabled),
                            "completed_search", 0., False)
    return _encode_decision(result, ticket, "completed_checkpoint", 1)


def _decode_decision(value):
    action, reason, execution_source, elapsed, fallback, why, ticket, source, depth, metadata = value
    return DecisionResult(Decision(action, reason, {"communication": {
        "ticket": ticket, "source": source, "completed_depth": depth, **metadata}},
        metadata.get("native_enabled", True)), execution_source, elapsed, fallback, why)


def _encode_prefetch(signature, ready, ticket, generation, game_id):
    from heuristic_ai.strategy.one_ply_prefetch import signature_digest
    evidence = {"signature_sha256": signature_digest(signature), "generation": generation, "game_id": game_id}
    if ready.input_digest != evidence["signature_sha256"]:
        raise IoFailure("One-Ply Expectation prefetch input identity mismatch")
    metadata = {"task_version": ready.task_version, "checkpoint_revision": 1,
                "complete": True, "value_digest": ready.value_digest, "native_enabled": ready.native_enabled}
    evidence.update(metadata, completed_depth=1)
    result = DecisionResult(Decision(ready.action, "one_ply_expectation", native_enabled=ready.native_enabled),
                            "prefetch_hit", 0., False)
    return _encode_decision(result, ticket, "prefetch_hit", 1, metadata), evidence


def _encode_trace(trace):
    from .supervision_contracts import plain_value
    data = {f.name: getattr(trace, f.name) for f in fields(trace) if f.name != 'decision'}
    data['decision'] = (trace.decision.action, trace.decision.reason, plain_value(trace.decision.details['communication']), trace.decision.native_enabled)
    if data.get('submission_error') is not None:
        data['submission_error'] = plain_value(data['submission_error'])
    return data


def _decode_trace(data):
    value = dict(data)
    action, reason, communication = value['decision'][:3]
    value['decision'] = Decision(action, reason, {'communication': communication},
        value['decision'][3] if len(value['decision']) == 4 else True)
    return DecisionTrace(**value)


class _StatePoolUnresponsive(IoFailure):
    """Repeated state failures after an unresolved connection-pool timeout."""


class _StateSlot:
    """One in-flight read; identical waiters join, different cursors coalesce next."""
    def __init__(self):
        self.condition = threading.Condition()
        self.active = None

    def request(self, call, path, deadline, stop):
        with self.condition:
            while self.active is not None and self.active[0] != path:
                if stop.is_set():
                    raise IoFailure('state request stopped')
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise RequestDeadlineExceeded('coalesced state request expired')
                self.condition.wait(min(0.05, remaining) if remaining is not None else 0.05)
            owner = self.active is None
            if owner:
                self.active = (path, Future())
            future = self.active[1]
        if not owner:
            while True:
                if stop.is_set():
                    raise IoFailure('state request stopped')
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise RequestDeadlineExceeded('joined state request expired')
                try:
                    return future.result(timeout=min(0.05, remaining) if remaining is not None else 0.05)
                except TimeoutError:
                    if future.done():
                        raise
        try:
            result = call()
            future.set_result(result)
            return result
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self.condition:
                self.active = None
                self.condition.notify_all()


class _ChildNetwork:
    def __init__(self, server, send, stop, action_reserves=DEFAULT_ACTION_RESERVES):
        self.server, self.send, self.stop = server, send, stop
        self.action_reserves = action_reserves
        self.runtime = HttpxRuntime()
        self.clients, self.slots, self.labels, self.gates = {}, {}, {}, {}
        self.discard_origins = ConfirmedDiscardOrigins()
        self.public_actions = PublicActionSignals()
        self.lock = threading.RLock()
        self.game_members = {}
        self.notifications = {}
        self.state_policies = {}
        self.state_locks = {}
        self.backfills = ThreadPoolExecutor(max_workers=4, thread_name_prefix='inventory-backfill')

    def subscribe(self, token, game):
        from .notifications import GameNotifications
        key = token, game.game_id
        with self.lock:
            previous = self.notifications.pop(key, None)
            if previous is not None:
                previous.close()
            stream = GameNotifications(self.client(token), game.game_id,
                lambda row: self.send(('metric', None, {'user_id': game.user_id, **row})))
            self.notifications[key] = stream
            return stream

    def register_game_member(self, game_id, user_id):
        with self.lock:
            self.game_members.setdefault(game_id, set()).add(user_id)

    def coordinated_game(self, game_id):
        with self.lock:
            return len(self.game_members.get(game_id, ())) == 4

    def client(self, token, budget_identity=None):
        budget_identity = token if budget_identity is None else budget_identity
        key = (token, budget_identity)
        with self.lock:
            if key not in self.clients:
                transport = HttpxTransport(runtime=self.runtime, insecure=self.server.insecure,
                    action_response_timeout_seconds=min(2.0, self.server.timeout_seconds),
                    state_locks=self.state_locks.setdefault(budget_identity, {}),
                    observer=lambda row: self.send(('metric', None, {
                        **self.labels.get(budget_identity, {}), **row, 'kind': 'http_completion'})))
                client = ApiClient(self.server.url, token=token, insecure=self.server.insecure,
                    timeout=self.server.timeout_seconds, transport=transport, ssl_context=transport.context,
                    observer=lambda row: self.send(('metric', None, {**self.labels.get(budget_identity, {}), **row})))
                client.bind_request_gate(self.gates.setdefault(budget_identity, PriorityStateGate(
                    self.stop, priority_provider=lambda intent: self.state_priority(budget_identity, intent))))
                client.bind_retry_wait(lambda seconds: self._wait_retry(seconds))
                self.clients[key] = client
            return self.clients[key]

    def state_pool_timeout(self, token):
        with self.lock:
            client = self.clients.get((token, token))
            return client._transport.state_pool_timeout_at if client is not None else None

    def _wait_retry(self, seconds):
        if self.stop.wait(seconds):
            raise IoFailure('I/O retry stopped')

    def state_priority(self, budget_identity, intent):
        if not intent.path.startswith('/api/games/'):
            return intent.priority_deadline_monotonic
        with self.lock:
            policy = self.state_policies.get((budget_identity, intent.path.split('/')[3]))
        if policy is None:
            return intent.priority_deadline_monotonic
        cursor = int(intent.path.rsplit('seq=', 1)[1]) if 'seq=' in intent.path else 0
        return policy.state_read_priority(cursor, time.monotonic())

    def request(self, token, method, path, body=None, **options):
        budget_identity = options.pop('budget_identity', token)
        read_context = options.pop('_state_context', None)
        if read_context is not None:
            options['request_timing'] = read_context
        client = self.client(token, budget_identity)
        call = lambda: client.request(method, path, body, **options)
        if method != 'GET' or not path.startswith('/api/games/') or '/state' not in path:
            return call()
        game = path.split('/')[3]
        with self.lock:
            # A low-priority backfill waiting for quota must not own the
            # foreground's coalescing slot. Shared transport locks serialize
            # actual state I/O after admission, including tokenless probes.
            slot = self.slots.setdefault((budget_identity, game,
                options.get('purpose') == 'inventory_backfill'), _StateSlot())
        def read():
            started = time.monotonic()
            version = self.public_actions.version(game)
            result = call()
            return result, version, started, dict(read_context or {})
        response, version, started, timing = slot.request(read, path, options.get('deadline_monotonic'), self.stop)
        if read_context is not None:
            # Joined readers inherit the owner request's start boundary.
            read_context['public_version'] = version
            read_context['request_started_monotonic'] = started
            read_context.update(timing)
        return response

    def prepare(self, token):
        client = self.client(token)
        # Warm notifications as well: a cold SSE handshake can miss the first
        # discard and leave every table with an unanchored response window.
        request = client._make_request('GET', '/api/me', None)
        def warm(channel):
            async def run():
                pool = client._transport._client(channel)
                response = await pool.get(request.full_url, headers=dict(request.header_items()), timeout=10)
                if response.status_code != 200:
                    raise ApiError(response.status_code, response.json())
            client._transport.runtime.submit(run()).result(timeout=12)
        for channel in ('state', 'action', 'notify'):
            warm(channel)


class _GameClient:
    def __init__(self, network, token, game, memory, stop, send, *, fence):
        self.network, self.token, self.stop, self.send = network, token, stop, send
        self.game, self.memory = game, memory
        self.consecutive_state_deadlines = 0
        self.state_deadline_first_at = None
        self.backfill_future = None
        network.labels[token] = {'user_id': game.user_id, 'player_name': game.player_name}
        network.register_game_member(game.game_id, game.user_id)
        self.managed_state = ManagedState(game, memory, fence=fence,
            action_reserves=getattr(network, "action_reserves", DEFAULT_ACTION_RESERVES),
            discard_origin=lambda observation: network.discard_origins.lookup(game.game_id, observation),
            coordinated_reads=lambda: network.coordinated_game(game.game_id), public_signals=network.public_actions,
            notifications=network.subscribe(token, game) if hasattr(network, 'subscribe') else None)
        if hasattr(network, 'state_policies'):
            with network.lock:
                network.state_policies[token, game.game_id] = self.managed_state

    def submit_backfill(self, cursor, deadline):
        def read():
            timing = {}
            response = self.network.request(self.token, 'GET', f'/api/games/{self.game.game_id}/state?seq={cursor}',
                deadline_monotonic=deadline, purpose='inventory_backfill', retry_policy='never', _state_context=timing)
            return response, timing, time.monotonic()
        self.backfill_future = self.network.backfills.submit(read)
        return self.backfill_future
    def request(self, method, path, body=None, **options):
        policy = self.managed_state
        read_context = None
        if method == 'GET' and '/state' in path:
            cursor = int(path.rsplit('seq=', 1)[1]) if 'seq=' in path else 0
            options.setdefault('purpose', 'event_poll' if cursor > 0 else
                               'response_snapshot' if policy.response_action_pending else 'state_snapshot')
            now = time.monotonic()
            deadline, own_boundary = policy.read_deadline(cursor, now)
            if own_boundary:
                options['purpose'] = 'own_action_snapshot'
            options['priority_deadline_monotonic'] = policy.state_read_priority(cursor, now)
            options.setdefault('deadline_monotonic', deadline)
            policy.begin_read()
            read_context = {}
            options['_state_context'] = read_context
        if method == 'POST':
            options['before_send'] = self._check_public_action_context
            if (policy.observation is not None and (policy.uses_public_wakeups or policy.confidence != 'unknown')
                    and not policy.basis.startswith('snapshot_send_freshness')
                    and policy.observation.phase in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}):
                deadline = options.get('deadline_monotonic')
                if deadline is not None:
                    # This seat can legally pass again in the following Chi
                    # window, and the API has no expected-phase field. Leave
                    # more send headroom for that ambiguous Peng pass. A
                    # skipped late action remains an unconfirmed opportunity.
                    ambiguous_pass = (policy.observation.phase is Phase.RESPONSE_PENG
                        and policy.observation.seat == (policy.observation.turn+1)%4
                        and isinstance(body, dict) and body.get('action') == 'pass')
                    options['deadline_monotonic'] = deadline-(.3 if ambiguous_pass else .1)
            self.send('memory', self.memory)
        try:
            observation = policy.observation
            requested_at = time.monotonic()
            if method == 'POST' and path.endswith('/action') and hasattr(self.network, 'send'):
                self.network.send(('metric', None, {'kind': 'action_handoff', 'game_id': self.game.game_id,
                    'user_id': self.game.user_id, 'observation_phase': observation.phase.value if observation is not None else None,
                    'requested_monotonic': requested_at, 'decision_ready_monotonic': getattr(policy, 'decision_ready_at', None)}))
            response = self.network.request(self.token, method, path, body, **options)
            if read_context is not None:
                policy.last_state_request_started_at = read_context.get('request_started_monotonic')
                policy.last_state_transport_started_at = read_context.get('transport_started_monotonic')
            if read_context is not None and options.get('purpose') != 'inventory_backfill':
                self.consecutive_state_deadlines = 0
                self.state_deadline_first_at = None
            if read_context is not None and 'public_version' in read_context:
                policy.read_public_version = read_context['public_version']
            if (method == 'POST' and path == f'/api/games/{self.game.game_id}/action'
                    and isinstance(body, dict) and body.get('action') == 'discard'):
                self.network.discard_origins.record(self.game.game_id, observation, body.get('tile'), requested_at)
            if method == 'POST' and path == f'/api/games/{self.game.game_id}/action':
                self.network.public_actions.publish(self.game, observation, body, requested_at)
            return response
        except ApiError as error:
            if (read_context is not None and options.get('purpose') != 'inventory_backfill'
                    and error.status == 429):
                # The shared state gate already owns the server backoff. A
                # bounded read that used all retries is still a live server
                # response, not evidence that its connection needs restarting.
                self.consecutive_state_deadlines = 0
                self.state_deadline_first_at = None
                return {'pending': True}
            raise
        except RequestDeadlineExceeded as error:
            if method == 'GET':
                if read_context is not None and options.get('purpose') != 'inventory_backfill':
                    if error.response_status is not None:
                        self.consecutive_state_deadlines = 0
                        self.state_deadline_first_at = None
                        return {'pending': True}
                    now = time.monotonic()
                    if self.state_deadline_first_at is None:
                        self.state_deadline_first_at = now
                    self.consecutive_state_deadlines += 1
                    if self.consecutive_state_deadlines >= 8 and now-self.state_deadline_first_at >= 2:
                        pool_timeout = getattr(self.network, 'state_pool_timeout', None)
                        failure = (_StatePoolUnresponsive if callable(pool_timeout)
                                   and pool_timeout(self.token) is not None else IoFailure)
                        raise failure(f'{self.consecutive_state_deadlines} consecutive state request deadlines across '
                                      f'{now-self.state_deadline_first_at:.3f}s') from None
                return {'pending': True}
            raise

    def _check_public_action_context(self):
        policy = self.managed_state
        if policy.action_deadline is not None and time.monotonic() >= policy.action_deadline:
            raise StateChangedBeforeSend('action window expired before send')
        observation = policy.observation
        newer_state = (policy.notifications is not None and observation is not None
            and (observation.phase in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}
                 or observation.phase is Phase.DRAW and observation.turn == observation.seat)
            and policy.notifications.snapshot()['notification_seq'] > policy.snapshot_seq)
        if self.stop.is_set() or newer_state or policy.has_unseen_public_change():
            raise StateChangedBeforeSend('new public state or stop invalidated the action before send')

    def request_once(self, method, path, body=None, **options):
        return self.request(method, path, body, retry_policy='never', **options)

    def close(self):
        if hasattr(self.network, 'state_policies'):
            with self.network.lock:
                key = self.token, self.game.game_id
                if self.network.state_policies.get(key) is self.managed_state:
                    self.network.state_policies.pop(key)
        if self.managed_state.notifications is not None:
            self.managed_state.notifications.close()
            with self.network.lock:
                key = self.token, self.game.game_id
                if self.network.notifications.get(key) is self.managed_state.notifications:
                    self.network.notifications.pop(key)
        if self.backfill_future is not None:
            self.backfill_future.cancel()
        self.stop.set()


class _RemoteController:
    def __init__(self, send, inbox, stop, policy, generation):
        self.send, self.inbox, self.stop, self.policy = send, inbox, stop, policy
        self.generation = generation
        self.degraded, self.degradation_reason = False, None

    def observe(self, observation):
        self.send("observe", (self.policy.snapshot_seq, observation))

    def accepted_action(self, context, action):
        self.send("accepted_action", (context, action))

    def validate_decision(self, context, result):
        communication = result.decision.details.get("communication", {})
        version = communication.get("task_version")
        if version is None:
            return communication.get("source") != "prefetch_hit"
        from heuristic_ai.strategy.one_ply_prefetch import input_key
        versions = getattr(self.inbox, "prefetch_versions", None)
        return (not self.stop.is_set() and versions is not None
                and versions.accepts(input_key(context), version))

    def decide(self, context):
        from heuristic_ai.strategy.one_ply_prefetch import input_key, signature_digest
        from heuristic_ai.strategy.one_ply_static import legal_fallback
        started, stopwatch = time.monotonic(), time.perf_counter()
        timing = {"io_decision_started_monotonic": started,
                  "io_effective_budget_ms": max(0., ((context.deadline_monotonic or started)-started)*1000)}
        job = uuid.uuid4().hex
        token = WindowToken(context.game.user_id, context.game.game_id, self.generation,
                            action_window_key(context.observation), context.decision_key)
        binding = {"game_id": token.game_id, "user_id": token.user_id, "generation": token.generation,
                   "fingerprint": token.fingerprint,
                   "opportunity_hash": hashlib.sha256(repr(token.opportunity).encode("utf8")).hexdigest()}
        forced, chosen = len(context.candidates) == 1, None
        if not forced and context.observation.phase in {Phase.RESPONSE_PENG, Phase.RESPONSE_CHI}:
            supplied = getattr(self.inbox, "prefetch", {}).get(input_key(context))
            if supplied is not None and supplied[0] == self.generation:
                decoded = _decode_decision(supplied[1])
                if (any(c.action == decoded.decision.action for c in context.candidates)
                        and self.validate_decision(context, decoded)):
                    chosen = decoded
                    timing["prefetch_signature_sha256"] = signature_digest(input_key(context))
        if chosen is None and not forced and context.deadline_monotonic is not None and started < context.deadline_monotonic:
            self.send("decide", (job, token, context))
            while not self.stop.is_set():
                remaining = context.deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    supplied_job, supplied_token, result, complete = self.inbox.get(timeout=min(.02, remaining))
                except queue.Empty:
                    continue
                if supplied_job != job or supplied_token != token:
                    continue
                decoded = _decode_decision(result)
                if not any(c.action == decoded.decision.action for c in context.candidates):
                    raise IoFailure("compute returned an illegal action")
                if not self.validate_decision(context, decoded):
                    continue
                chosen = decoded
                if complete:
                    break
            self.send("cancel", job)
        if chosen is not None and not self.validate_decision(context, chosen):
            chosen = None
        if chosen is None:
            source = "forced_action" if forced else "legal_fallback"
            reason = "forced_action" if forced else "deadline_fallback"
            decision = legal_fallback(context, reason, {"communication": {"ticket": None,
                "source": source, "job": job, "completed_depth": 0, "window_binding": binding,
                **self.policy.details(), **timing}})
            chosen = DecisionResult(decision, source, (time.perf_counter()-stopwatch)*1000,
                                    not forced, None if forced else "communication_deadline")
        else:
            communication = dict(chosen.decision.details["communication"])
            communication.update({"job": job, "window_binding": binding, **self.policy.details(), **timing})
            chosen = replace(chosen, elapsed_ms=(time.perf_counter()-stopwatch)*1000,
                decision=Decision(chosen.decision.action, chosen.decision.reason, {"communication": communication},
                                  chosen.decision.native_enabled))
        self.policy.decision_ready_at = time.monotonic()
        if chosen.fallback_used:
            self.degraded, self.degradation_reason = True, chosen.fallback_reason
        return chosen

    def finish(self, outcome):
        self.send("finish", outcome)


def _child_main(connection, server, action_reserves=DEFAULT_ACTION_RESERVES):
    from .bot import play_game
    stop = threading.Event()
    writer = _PipeWriter(connection, lambda error: stop.set())
    send = writer.send
    network = _ChildNetwork(server, send, stop, action_reserves)
    games = {}
    def reply(kind, key, payload):
        send((kind, key, payload))
    def run_rpc(key, args):
        try:
            token, method, path, body, options = args
            result = network.request(token, method, path, body, **options)
            reply('rpc_result', key, result)
        except BaseException as error:
            reply('rpc_error', key, (type(error).__name__, getattr(error, 'status', None)))
    def prepare(key, tokens):
        try:
            for token in tokens:
                network.prepare(token)
            reply('rpc_result', key, True)
        except BaseException as error:
            reply('rpc_error', key, (type(error).__name__, getattr(error, 'status', None)))
    def run_game(key, args, game_stop, inbox):
        token, game, memory, generation, fence, observe_states, health_checkpoints = args
        local_send = lambda kind, payload: reply(kind, key, payload)
        client = _GameClient(network, token, game, memory, game_stop, local_send, fence=fence)
        controller = _RemoteController(local_send, inbox, game_stop, client.managed_state, generation)
        try:
            def on_trace(trace):
                local_send('memory', memory)
                local_send('trace', _encode_trace(trace))
            result = play_game(client, game, controller, game_stop, on_decision=on_trace,
                on_seat=lambda *a: local_send('seat', a),
                on_state_poll=lambda p: local_send('poll', p),
                on_state_response=(lambda r, **kw: local_send('state_response', (r, kw))) if observe_states else None,
                recovery_memory=memory, generation=generation,
                on_ready=lambda: local_send('ready', None),
                on_failure=lambda e, c: local_send('failure', (type(e).__name__, c)),
                on_health_checkpoint=(lambda s: local_send('checkpoint', s)) if health_checkpoints else None)
            local_send('memory', memory)
            local_send('done', result)
        except BaseException as error:
            local_send('memory', memory)
            if isinstance(error, _StatePoolUnresponsive):
                pool_timeout = network.state_pool_timeout(token)
                if pool_timeout is not None:
                    local_send('state_pool_unresponsive', {'game_id': game.game_id,
                        'user_id': game.user_id, 'generation': generation,
                        'pool_timeout_monotonic': pool_timeout, 'reason': str(error)})
            import traceback
            frames = tuple((Path(f.filename).name, f.lineno, f.name) for f in traceback.extract_tb(error.__traceback__))
            local_send('game_error', (type(error).__name__, getattr(error, 'status', None), frames))
        finally:
            client.close()
            games.pop(key, None)
    try:
        reply('started', None, None)
        for kind, key, payload in _read_messages(connection, stop):
            active_game = games.get(key)
            if kind == 'close':
                break
            if kind == 'start_game':
                game_stop, inbox = threading.Event(), queue.Queue(256)
                inbox.prefetch = OrderedDict()
                from heuristic_ai.strategy.prefetch_audit import PrefetchVersions
                inbox.prefetch_versions = PrefetchVersions()
                inbox.generation = payload[3]
                games[key] = (game_stop, inbox)
                threading.Thread(target=run_game, args=(key, payload, game_stop, inbox), daemon=True).start()
            elif kind == 'result' and active_game is not None:
                active_game[1].put_nowait(payload)
            elif kind == 'prefetch' and active_game is not None:
                generation, signature, encoded = payload
                inbox = active_game[1]
                if generation == inbox.generation:
                    metadata = encoded[9]
                    if not inbox.prefetch_versions.accepts(signature, metadata.get('task_version')):
                        continue
                    previous = inbox.prefetch.get(signature)
                    if (previous and previous[1][9].get('task_version') == metadata.get('task_version')
                            and (previous[1][0] != encoded[0]
                                 or previous[1][9].get('value_digest') != metadata.get('value_digest'))):
                        reply('candidate_conflict', key, {'old': previous[1][9], 'new': metadata,
                            'old_ticket': previous[1][6], 'new_ticket': encoded[6]})
                        reply('game_error', key, ('CandidateConflict', None, ()))
                        active_game[0].set()
                        continue
                    inbox.prefetch[signature] = (generation, encoded)
                    while len(inbox.prefetch) > 128:
                        inbox.prefetch.popitem(last=False)
            elif kind == 'prefetch_control' and active_game is not None:
                generation, control = payload
                inbox = active_game[1]
                if generation == inbox.generation:
                    for signature in inbox.prefetch_versions.apply(control):
                        inbox.prefetch.pop(signature, None)
            elif kind == 'stop_game' and active_game is not None:
                active_game[1].prefetch_versions.active.clear()
                active_game[1].prefetch.clear()
                active_game[0].set()
            elif kind == 'rpc':
                threading.Thread(target=run_rpc, args=(key, payload), daemon=True).start()
            elif kind == 'prepare':
                threading.Thread(target=prepare, args=(key, payload), daemon=True).start()
    except (EOFError, OSError):
        pass
    finally:
        stop.set()
        for game_stop, inbox in tuple(games.values()):
            game_stop.set()
        for stream in tuple(network.notifications.values()):
            stream.close()
        network.backfills.shutdown(wait=True, cancel_futures=True)
        try:
            network.runtime.close()
        finally:
            writer.close()
            connection.close()


class _IoGeneration:
    """State retained by its callers until this process generation drains."""
    def __init__(self, epoch, process, connection):
        self.epoch, self.process, self.connection = epoch, process, connection
        self.writer = None
        self.error = None
        self.closing = False
        self.ready = threading.Event()
        self.pending = {}
        self.cancellations = {}


class IoService:
    def __init__(self, server, record, *, action_reserves=DEFAULT_ACTION_RESERVES):
        self.server, self.record = server, record
        self.action_reserves = action_reserves
        self.lock = threading.RLock()
        self._channel = None
        self.closed = False
        self.prefetch = None
        self._ensure()

    @property
    def process(self):
        return self._channel.process if self._channel is not None else None

    @property
    def epoch(self):
        return self._channel.epoch if self._channel is not None else 0

    def _available(self, channel):
        return (channel is not None and channel is self._channel and not self.closed
                and not channel.closing and channel.error is None)

    def _ensure(self):
        with self.lock:
            if self.closed:
                raise IoFailure('I/O service closed')
            channel = self._channel
            if channel is None or not channel.process.is_alive() or channel.error is not None:
                if channel is not None:
                    if channel.error is None:
                        self._fail(channel, IoFailure('I/O process exited before its handoff'), source='process_exit')
                    self._dispose(channel)
                parent, child = multiprocessing.get_context('spawn').Pipe()
                process = multiprocessing.get_context('spawn').Process(target=_child_main,
                    args=(child, self.server, self.action_reserves), name='mahjong-io', daemon=True)
                channel = _IoGeneration(self.epoch + 1, process, parent)
                self._channel = channel
                try:
                    process.start()
                    channel.writer = _PipeWriter(parent, lambda error: self._fail(channel, error, source='writer'))
                    threading.Thread(target=self._read, args=(channel,), name='mahjong-ipc-reader', daemon=True).start()
                except BaseException as error:
                    self._fail(channel, error, source='startup')
                    self._dispose(channel)
                    raise
                finally:
                    child.close()
        if not channel.ready.wait(15):
            self._fail(channel, IoFailure('I/O startup timed out'), source='startup')
        if not self._available(channel):
            self._raise_failure(channel)
        return channel

    def _fail(self, channel, error, *, source='service', message_kind=None, message_key=None):
        with self.lock:
            if channel is None or channel.error is not None or channel.closing:
                return
            channel.error = error
            for event in channel.cancellations.values():
                event.set()
            for pending in channel.pending.values():
                try:
                    pending.put_nowait(('channel_error', type(error).__name__))
                except queue.Full:
                    pass  # The same generation's failure remains visible after its FIFO drains.
            channel.ready.set()
            if not self.closed and channel.process.is_alive():
                channel.process.terminate()
            try:
                pending = channel.pending.get(message_key)
                frames = tuple((Path(frame.filename).name, frame.lineno, frame.name)
                               for frame in traceback.extract_tb(error.__traceback__))
                logging.getLogger(__name__).error(
                    'ipc_generation_failure unix=%.6f monotonic=%.6f epoch=%s pid=%s '
                    'source=%s error=%s failed_kind=%s pending=%s inbox_queued=%s frames=%s',
                    time.time(), time.monotonic(), channel.epoch, channel.process.pid, source,
                    type(error).__name__, message_kind, len(channel.pending),
                    pending.qsize() if pending is not None else None, frames)
            except Exception:
                pass  # Failure propagation must not depend on diagnostic output.

    def _raise_failure(self, channel):
        from .recovery import RecorderCallbackFailure
        error = channel.error
        if isinstance(error, RecorderCallbackFailure):
            raise error
        if error is not None:
            raise IoFailure('I/O channel failed: ' + type(error).__name__) from error
        raise IoFailure('I/O generation is no longer available')

    def _register(self, channel, key, inbox):
        with self.lock:
            if not self._available(channel):
                self._raise_failure(channel)
            channel.pending[key] = inbox

    def _send(self, channel, message):
        with self.lock:
            if not self._available(channel):
                self._raise_failure(channel)
            writer = channel.writer
        # The bound writer cannot become a newer generation's writer.
        writer.send(message)

    def _read(self, channel):
        source, kind, key = 'reader_receive', None, None
        try:
            for kind, key, payload in _read_messages(channel.connection):
                terminal_at = time.monotonic() if kind in ('done', 'game_error') else None
                source = 'reader_dispatch'
                with self.lock:
                    if channel is not self._channel:
                        return
                    pending = channel.pending.get(key)
                if kind == 'started':
                    channel.ready.set()
                elif kind == 'metric':
                    self.record(payload)
                elif kind == 'state_pool_unresponsive':
                    self.record({'kind': kind, 'io_epoch': channel.epoch, **payload})
                    self._fail(channel, _StatePoolUnresponsive(payload['reason']),
                               source='state_pool_unresponsive', message_kind=kind, message_key=key)
                    return
                elif kind == 'cancel':
                    with self.lock:
                        event = channel.cancellations.get(payload)
                    if event is not None:
                        event.set()
                elif pending is not None:
                    if terminal_at is not None and hasattr(pending, 'game_identity'):
                        if pending.terminal_received_monotonic is None:
                            pending.terminal_received_monotonic = terminal_at
                            game_id, user_id, generation = pending.game_identity
                            self.record({'kind': 'io_game_terminal_received', 'io_epoch': channel.epoch,
                                'game_id': game_id, 'user_id': user_id, 'generation': generation,
                                'message_kind': kind, 'received_monotonic': terminal_at,
                                'inbox_queued': pending.qsize()})
                    if kind == 'poll':
                        self.record({'kind': 'state_poll', **payload})
                    if kind == 'decide':
                        with self.lock:
                            channel.cancellations.setdefault(payload[0], threading.Event())
                    pending.put_nowait((kind, payload))
                source = 'reader_receive'
        except BaseException as error:
            self._fail(channel, error, source=source,
                       message_kind=kind if source == 'reader_dispatch' else None,
                       message_key=key if source == 'reader_dispatch' else None)

    def _rpc(self, kind, payload, deadline=None):
        channel = self._ensure()
        key, inbox = uuid.uuid4().hex, queue.Queue(4)
        self._register(channel, key, inbox)
        try:
            self._send(channel, (kind, key, payload))
            end = deadline if deadline is not None else time.monotonic() + max(60, self.server.timeout_seconds + 5)
            while time.monotonic() < end:
                try:
                    event, value = inbox.get(timeout=min(0.1, max(0.001, end - time.monotonic())))
                except queue.Empty:
                    if not self._available(channel):
                        self._raise_failure(channel)
                    continue
                if event == 'rpc_result':
                    return value
                if event == 'rpc_error' and value[1] is not None:
                    raise ApiError(value[1], {'error': value[0]})
                if event == 'channel_error':
                    self._raise_failure(channel)
                raise IoFailure('I/O request failed')
            raise RequestDeadlineExceeded('I/O RPC expired')
        finally:
            with self.lock:
                channel.pending.pop(key, None)

    def request(self, token, method, path, body=None, **options):
        return self._rpc('rpc', (token, method, path, body, options), options.get('deadline_monotonic'))

    def prepare(self, tokens):
        return self._rpc('prepare', tuple(tokens))

    def play_game(self, token, game, controller, stop, **callbacks):
        from .bot import GameResult, GameStopped
        channel = self._ensure()
        key, inbox = uuid.uuid4().hex, queue.Queue(1024)
        memory = callbacks.get('recovery_memory')
        if memory is None:
            memory = ActionMemory()
        generation = callbacks.get('generation') or 1
        inbox.game_identity = (game.game_id, game.user_id, generation)
        inbox.terminal_received_monotonic = None
        epoch = channel.epoch
        cached, jobs = {}, {}
        reported_strategy_errors = set()
        prefetch_closed = threading.Event()
        finished = False
        stopping_at = None
        def callback(name, *a, **kw):
            hook = callbacks.get(name)
            if hook is not None:
                return hook(*a, **kw)
        def operation(label, seconds=None):
            hook = callbacks.get('health_operation')
            return hook(label, seconds) if hook is not None else nullcontext()
        try:
            self._register(channel, key, inbox)
            restore = getattr(controller, 'restore_state', None)
            if restore is not None:
                restore(memory.chain_checkpoint)
            if controller.background_enabled:
                from .response_prefetch import ResponsePrefetch
                from heuristic_ai.strategy.one_ply_prefetch import signature_digest
                with self.lock:
                    if self.prefetch is None:
                        self.prefetch = ResponsePrefetch(lambda reason: self._fail(self._channel, IoFailure(reason), source='prefetch'))
                def send_prefetch(kind, payload):
                    with self.lock:
                        if prefetch_closed.is_set() or stop.is_set() or not self._available(channel):
                            return
                        writer = channel.writer
                        try:
                            writer.send((kind, key, payload))
                        except Exception as error:
                            # The I/O generation owns recorded channel failures;
                            # a late callback must not disable the shared reader.
                            unavailable = (isinstance(error, IoFailure)
                                and (writer.error is not None or writer.closed))
                            if error is not writer.error and not unavailable:
                                raise
                def publish_prefetch(signature, ready):
                    if (prefetch_closed.is_set() or stop.is_set() or not self._available(channel)
                            or controller.degraded):
                        return
                    ticket = uuid.uuid4().hex
                    encoded, evidence = _encode_prefetch(signature, ready,
                                                         ticket, generation, game.game_id)
                    cached[ticket] = lambda ready=ready, evidence=evidence: ready.build_decision().with_details({'prefetch': evidence})
                    send_prefetch('prefetch', (generation, signature, encoded))
                def control_prefetch(payload):
                    send_prefetch('prefetch_control', (generation, payload))
                publish_prefetch.control = control_prefetch
                controller.bind_background(self.prefetch, key, publish_prefetch)
            self._send(channel, ('start_game', key, (token, game, memory,
                generation, epoch > 1, callbacks.get('on_state_response') is not None,
                callbacks.get('on_health_checkpoint') is not None)))
            while True:
                if stop.is_set() and stopping_at is None:
                    command_enqueued = self._available(channel)
                    if command_enqueued:
                        self._send(channel, ('stop_game', key, None))
                    stopping_at = time.monotonic()
                    self.record({'kind': 'io_game_stop_wait', 'io_epoch': channel.epoch,
                        'game_id': game.game_id, 'user_id': game.user_id, 'generation': generation,
                        'stop_wait_started_monotonic': stopping_at, 'command_enqueued': command_enqueued})
                if stopping_at is not None:
                    grace = max(5, min(self.server.timeout_seconds + 2,
                        max(game.rules.discard_timeout_seconds, game.rules.peng_timeout_seconds, game.rules.chi_timeout_seconds) + 2))
                    terminal_at = inbox.terminal_received_monotonic
                    if (time.monotonic() - stopping_at > grace
                            and (terminal_at is None or terminal_at - stopping_at > grace)):
                        self._fail(channel, IoFailure('I/O game did not acknowledge stop before its request deadline'),
                                   source='stop_handoff_timeout', message_kind='stop_game', message_key=key)
                        raise GameStopped('I/O stopped without a safe game handoff')
                try:
                    kind, payload = inbox.get(timeout=0.1)
                except queue.Empty:
                    if not self._available(channel):
                        self._raise_failure(channel)
                    continue
                if kind == 'observe':
                    if stopping_at is None:
                        observed_seq, payload = payload if isinstance(payload, tuple) else (None, payload)
                        with operation('strategy_observe'):
                            controller.observe(payload)
                        checkpoint = getattr(controller, 'state_checkpoint', None)
                        if checkpoint is not None:
                            memory.chain_checkpoint = checkpoint()
                        self.record({'kind': 'strategy_observation', 'game_id': game.game_id,
                            'user_id': game.user_id, 'generation': generation, 'round_no': payload.round_no,
                            'server_seq': observed_seq,
                            'phase': payload.phase.value, 'chain_pending': getattr(controller, '_chain_pending', None),
                            'discards': payload.discards, 'public_discard_history': payload.public_discard_history,
                            'melds': tuple(tuple((m.kind, m.tiles) for m in seat) for seat in payload.melds),
                            'chain_count': payload.god.chain_count})
                elif kind == 'accepted_action':
                    if stopping_at is None:
                        controller.accepted_action(*payload)
                elif kind == 'decide':
                    job, window, context = payload
                    with self.lock:
                        cancelled = channel.cancellations.setdefault(job, threading.Event())
                    jobs[job] = set()
                    def publish(publication, job=job, window=window, cancelled=cancelled):
                        if cancelled.is_set() or time.monotonic() >= context.deadline_monotonic:
                            return
                        ticket = uuid.uuid4().hex
                        cached[ticket] = publication.build_decision
                        jobs[job].add(ticket)
                        self._send(channel, ('result', key, (job, window,
                            _encode_publication(publication, ticket), False)))
                    compute_started, compute_started_ns = time.monotonic(), time.perf_counter_ns()
                    result, failure, terminal = None, None, 'invalidated'
                    try:
                        if stopping_at is None and not cancelled.is_set() and time.monotonic() < context.deadline_monotonic:
                            context = replace(context, publish_candidate=publish, cancelled=cancelled.is_set)
                            if self.prefetch is not None:
                                self.prefetch.foreground(key, True)
                            with operation('strategy_decide', max(0, context.deadline_monotonic - time.monotonic())):
                                result = controller.decide(context)
                            terminal = 'fallback' if result.fallback_used else 'completed'
                            error = result.decision.details.get('strategy_error')
                            if error is not None and error['error_id'] not in reported_strategy_errors:
                                from .supervision_contracts import plain_value
                                reported_strategy_errors.add(error['error_id'])
                                self.record({'kind': 'strategy_error', 'job': job, 'game_id': game.game_id,
                                    'user_id': game.user_id, 'generation': generation, **plain_value(error)})
                            if not cancelled.is_set() and time.monotonic() < context.deadline_monotonic:
                                ticket = uuid.uuid4().hex
                                cached[ticket] = lambda result=result: result.decision
                                jobs[job].add(ticket)
                                depth = result.decision.details.get('search', {}).get('completed_depth')
                                self._send(channel, ('result', key, (job, window, _encode_decision(result, ticket, depth=depth), True)))
                            else:
                                terminal = 'cancelled' if cancelled.is_set() else 'expired'
                    except BaseException as error:
                        from .recovery_context import safe_exception
                        failure, terminal = safe_exception(error), 'error'
                        raise
                    finally:
                        if self.prefetch is not None:
                            self.prefetch.foreground(key, False)
                        from .supervision_contracts import plain_value
                        detail = result.decision.details if result is not None else {}
                        self.record({'kind': 'compute', 'job': job, 'game_id': game.game_id, 'user_id': game.user_id,
                            'generation': generation, 'decision_key': context.decision_key,
                            'observation_phase': context.observation.phase.value,
                            'effective_budget_ms': max(0, (context.deadline_monotonic-compute_started)*1000),
                            'started_monotonic': compute_started, 'finished_monotonic': time.monotonic(),
                            'elapsed_ms': (time.perf_counter_ns() - compute_started_ns) / 1_000_000,
                            'queue_wait_ms': detail.get('search', {}).get('queue_wait_ms'),
                            'cancelled': cancelled.is_set(),
                            'completed_depth': detail.get('search', {}).get('completed_depth'),
                            'fallback_used': result.fallback_used if result is not None else None,
                            'terminal': terminal, 'error': failure,
                            'execution_source': result.execution_source if result is not None else None,
                            'strategy_error_id': detail.get('strategy_error', {}).get('error_id')})
                elif kind == 'trace':
                    trace = _decode_trace(payload)
                    communication = dict(trace.decision.details['communication'])
                    ticket = communication.get('ticket')
                    if ticket is not None:
                        build = cached.get(ticket)
                        if build is None:
                            raise IoFailure('chosen audit checkpoint missing')
                        try:
                            trace = replace(trace, decision=build().with_details({'communication': communication}))
                            evidence = trace.decision.details.get('prefetch', {})
                            if 'task_version' in evidence and any(communication.get(field) != evidence.get(field)
                                    for field in ('task_version', 'checkpoint_revision', 'completed_depth',
                                                  'complete', 'value_digest')):
                                raise IoFailure('prefetch transport metadata disagrees with audit')
                        except (OSError, pickle.UnpicklingError, EOFError, ValueError) as error:
                            self._fail(channel, error)
                            raise
                    callback('on_decision', trace)
                    job = communication.get('job')
                    for ticket in jobs.pop(job, ()):
                        cached.pop(ticket, None)
                    with self.lock:
                        channel.cancellations.pop(job, None)
                elif kind == 'memory':
                    for field in fields(memory):
                        if field.name != 'chain_checkpoint':
                            setattr(memory, field.name, getattr(payload, field.name))
                elif kind == 'candidate_conflict':
                    self.record({'kind': kind, 'game_id': game.game_id, 'user_id': game.user_id,
                                 'generation': generation, **payload})
                elif kind == 'seat':
                    callback('on_seat', *payload)
                elif kind == 'poll':
                    callback('on_state_poll', payload)
                elif kind == 'state_response':
                    callback('on_state_response', payload[0], **payload[1])
                elif kind == 'checkpoint':
                    callback('on_health_checkpoint', payload)
                elif kind == 'ready':
                    callback('on_ready')
                elif kind == 'failure':
                    error = GameStopped(payload[0]) if payload[0] == 'GameStopped' else IoFailure(payload[0])
                    callback('on_failure', error, payload[1])
                elif kind == 'finish':
                    with operation('strategy_finish'):
                        controller.finish(payload)
                    finished = True
                elif kind == 'done':
                    if stopping_at is not None:
                        raise GameStopped('game coordinator stopped after I/O handoff')
                    return GameResult(game.game_id, payload.scores, payload.degraded or controller.degraded,
                                      payload.degradation_reason or controller.degradation_reason)
                elif kind == 'game_error':
                    if payload[0] == 'GameStopped' or stopping_at is not None:
                        raise GameStopped('game coordinator stopped after I/O handoff')
                    if payload[1] is not None:
                        raise ApiError(payload[1], {'error': payload[0]})
                    location = repr(payload[2]) if len(payload) > 2 else ''
                    raise IoFailure('I/O game failed: ' + payload[0] + ' ' + location)
                elif kind == 'channel_error':
                    self._raise_failure(channel)
        except BaseException as error:
            from .recovery import recording_failure
            fatal = recording_failure(error)
            if fatal is not None:
                self._fail(channel, fatal)
            raise
        finally:
            prefetch_closed.set()
            try:
                if self._available(channel):
                    self._send(channel, ('stop_game', key, None))
            finally:
                with self.lock:
                    channel.pending.pop(key, None)
                    for job in jobs:
                        event = channel.cancellations.pop(job, None)
                        if event is not None:
                            event.set()
                if not finished:
                    from heuristic_ai.game.models import GameOutcome, GameOutcomeStatus
                    controller.finish(GameOutcome(game.game_id, GameOutcomeStatus.ABORTED, error='IoFailure'))

    def _dispose(self, channel):
        channel.closing = True
        channel.ready.set()
        try:
            channel.connection.close()
            if channel.process.pid is not None:
                channel.process.join(1)
                if channel.process.is_alive():
                    channel.process.terminate()
                    channel.process.join(3)
                if channel.process.is_alive():
                    raise IoFailure('owned I/O process failed to stop')
        finally:
            if channel.writer is not None:
                channel.writer.close()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            channel = self._channel
            if channel is not None:
                channel.closing = True
                channel.ready.set()
        try:
            if channel is not None:
                if channel.writer is not None and channel.error is None:
                    channel.writer.send(('close', None, None))
                if channel.process.pid is not None:
                    channel.process.join(3)
        finally:
            try:
                if channel is not None:
                    self._dispose(channel)
            finally:
                if self.prefetch is not None:
                    self.prefetch.close()
