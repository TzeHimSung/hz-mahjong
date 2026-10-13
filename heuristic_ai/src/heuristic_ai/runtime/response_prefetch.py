"""One run-owned low-priority process for exact One-Ply Expectation Chi precomputation."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, replace
import hashlib
import logging
import multiprocessing
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace

from heuristic_ai.game.models import Phase
from heuristic_ai.strategy.models import MAX_SEARCH_NODES, CACHE_ENABLED
from heuristic_ai.strategy.one_ply_static import evaluate_actions
from heuristic_ai.strategy.one_ply_prefetch import (ResponseCache, ResumableOnePlySearch,
    PrefetchedResult, decision_wire, input_key, chi_target, signature_digest)
from heuristic_ai.strategy.search.one_ply_search import SearchBudget, SearchBudgetExhausted
from heuristic_ai.strategy.search.one_ply_model import SearchModelError
from heuristic_ai.strategy.prefetch_audit import AuditStore, decision_value_digest


def _worker(commands, results, directory, paused=None, control_revision=None):
    paused = paused if paused is not None else SimpleNamespace(value=0)
    is_paused = lambda: bool(paused.value)
    tracked_control = control_revision is not None
    control_revision = control_revision if tracked_control else SimpleNamespace(value=0)
    processed_revision = 0
    if threading.current_thread() is threading.main_thread() and os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    elif threading.current_thread() is threading.main_thread():
        os.nice(5)
    cache, store = ResponseCache(), AuditStore(directory)
    tasks, busy = OrderedDict(), set()
    stats = {"slices": 0, "completed_targets": 0, "superseded_tasks": 0,
             "cancelled_tasks": 0, "unfinished_target_nodes": 0, "max_slice_ms": 0., "search_exits": 0}
    last_stats = time.monotonic()

    def close_search(task):
        if task is not None and task["search"] is not None:
            stats["unfinished_target_nodes"] += task["search"].budget.nodes
            task["search"].close()

    def publish_stats(revision=0):
        try:
            results.put_nowait(("stats", None, revision, {**stats, "cpu_ms": time.process_time()*1000,
                "cache_payload_bytes": cache.size, "audit_bytes": store.size, "audit_refusals": store.refused,
                "audit_files": sum(map(len, store.files.values())), "audit_peak_bytes": store.peak_bytes,
                "audit_peak_files": store.peak_files}))
        except queue.Full:
            stats["dropped_notifications"] = stats.get("dropped_notifications", 0) + 1

    try:
        while True:
            waiting = time.perf_counter_ns()
            try:
                command = commands.get(timeout=.025 if not tasks or busy or is_paused()
                                       or control_revision.value != processed_revision else 0)
            except queue.Empty:
                command = None
            if busy or is_paused():
                stats["foreground_busy_ms"] = stats.get("foreground_busy_ms", 0) + (time.perf_counter_ns()-waiting)/1_000_000
            while command is not None:
                kind, handle, version, data = command
                if tracked_control:
                    processed_revision += 1
                if kind == "close":
                    publish_stats()
                    return
                if kind == "stats":
                    if data:
                        stats["max_slice_ms"] = 0.
                        store.peak_bytes = store.size
                        store.peak_files = sum(map(len, store.files.values()))
                    publish_stats(version)
                elif kind == "update":
                    context, versions = data
                    previous = tasks.pop(handle, None)
                    if previous is not None:
                        stats["superseded_tasks"] += 1
                    retained = (previous is not None and context is not None
                                and input_key(previous["context"]) == input_key(context))
                    if not retained:
                        close_search(previous)
                    if context is not None:
                        tasks[handle] = {"version": versions[input_key(context)], "context": context,
                                         "search": previous["search"] if retained else None}
                elif kind in {"cancel", "detach"}:
                    previous = tasks.pop(handle, None)
                    if previous is not None:
                        stats["cancelled_tasks"] += 1
                        close_search(previous)
                    busy.discard(handle)
                    if kind == "detach":
                        store.detach(handle)
                elif kind == "foreground":
                    if data:
                        busy.add(handle)
                    else:
                        busy.discard(handle)
                else:
                    raise ValueError("unknown One-Ply Expectation prefetch control")
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    command = None
            if not tasks or busy or is_paused() or control_revision.value != processed_revision:
                if time.monotonic() - last_stats >= .25:
                    publish_stats()
                    last_stats = time.monotonic()
                continue
            handle, task = tasks.popitem(last=False)
            try:
                search, context = task["search"], task["context"]
                if search is None:
                    baseline = evaluate_actions(context, piao_count=0)
                    search = ResumableOnePlySearch(context, baseline,
                        SearchBudget(float("inf"), MAX_SEARCH_NODES), shared_cache=cache,
                        cache_enabled=CACHE_ENABLED, native_enabled=True)
                    task["search"] = search
                started = time.perf_counter()
                result = search.advance(25, should_yield=is_paused)
                stats["slices"] += 1
                stats["max_slice_ms"] = max(stats["max_slice_ms"], (time.perf_counter()-started)*1000)
                if result is None:
                    tasks[handle] = task
                else:
                    stats["completed_targets"] += 1
                    signature = input_key(context)
                    input_digest = signature_digest(signature)
                    result = result.with_details({"async": {"source": "prefetch",
                        "last_slice_ms": (time.perf_counter()-started)*1000,
                        "cache_payload_bytes": cache.size}, "prefetch_input_digest": input_digest,
                        "prefetch_rules": asdict(context.game.rules)})
                    value_digest = decision_value_digest(result)
                    ref = store.write(handle, "decision", signature, decision_wire(result), digest=value_digest)
                    if ref is not None:
                        ready = PrefetchedResult(result.action, ref.path, ref.sha256, task["version"],
                            input_digest, ref.size, ref.raw_size, value_digest, result.native_enabled)
                        results.put(("result", handle, task["version"], (signature, ready)))
                    search.close()
            except (SearchBudgetExhausted, SearchModelError) as error:
                stats["search_exits"] += 1
                results.put(("diagnostic", handle, task["version"], getattr(error, "reason", "model_unavailable")))
                close_search(task)
            if time.monotonic() - last_stats >= .25:
                publish_stats()
                last_stats = time.monotonic()
    except BaseException as error:
        results.put(("fatal", None, 0, type(error).__name__))
        raise
    finally:
        for task in tasks.values():
            close_search(task)


class ResponsePrefetch:
    def __init__(self, audit_failure=None):
        ctx = multiprocessing.get_context('spawn')
        self.directory = tempfile.TemporaryDirectory(prefix='mahjong-one-ply-audits-')
        self.audit_failure = audit_failure
        self.commands = ctx.Queue(256)
        self.results = ctx.Queue(256)
        # ponytail: advisory byte avoids kernel-lock priority inversion; command
        # messages and deadline checks remain authoritative for correctness.
        self.paused = ctx.RawValue('b', 0)
        self.control_revision = ctx.RawValue('Q', 0)
        self.active_foregrounds = set()
        self.process = ctx.Process(target=_worker, args=(self.commands, self.results, self.directory.name, self.paused, self.control_revision),
                                   name='mahjong-response-prefetch', daemon=True)
        self.lock = threading.RLock()
        self.stats_ready = threading.Condition(self.lock)
        self.stats_request = self.stats_received = 0
        self.sessions = {}
        self.versions = {}
        self.targets = {}
        self.control_seq = {}
        self.closed = False
        self.failed = None
        self.metrics = {'stale_messages': 0}
        self.process.start()
        self.reader = threading.Thread(target=self._read, name='mahjong-prefetch-reader', daemon=True)
        self.reader.start()

    def attach(self, handle, session):
        with self.lock:
            self.sessions[handle] = session
            self.versions.setdefault(handle, 0)

    def _send(self, command):
        if self.closed or self.failed:
            return
        try:
            with self.lock:
                self.commands.put_nowait(command)
                # An event for a newly busy table must interrupt a slice even if
                # another table already held the shared pause byte.
                self.control_revision.value += 1
        except queue.Full:
            self._fail('command_queue_full')

    def snapshot_metrics(self, *, reset_peaks=False, timeout=2.):
        """Acknowledge all prior commands before reading phase resource counters."""
        with self.stats_ready:
            self.stats_request += 1
            revision = self.stats_request
            self._send(('stats', None, revision, reset_peaks))
            received = self.stats_ready.wait_for(
                lambda: self.stats_received >= revision or self.failed or self.closed, timeout)
            if not received or self.failed or self.closed:
                raise RuntimeError('prefetch metrics snapshot unavailable')
            return dict(self.metrics)

    def update(self, handle, game, observation):
        with self.lock:
            previous = self.targets.get(handle, {})
            can_start = (observation.phase is Phase.RESPONSE_PENG
                         and observation.seat in observation.responding_seats)
            # Our PASS removes us from the PENG responders while others still
            # decide. Retain the identical CHI target, without starting another.
            retained_view = (replace(observation, responding_seats=(observation.seat,))
                             if previous and observation.phase is Phase.RESPONSE_PENG and not can_start
                             else observation)
            context = chi_target(game, retained_view)
            signature = input_key(context) if context is not None else None
            # Only an actionable PENG can launch work; later views only retain it.
            if not can_start and signature not in previous:
                context = signature = None
            versions = {}
            if signature is not None:
                version = previous.get(signature)
                if version is None:
                    version = self.versions[handle] = self.versions.get(handle, 0) + 1
                versions[signature] = version
            if versions == previous:
                return
            self._control(handle, 'invalidate', tuple((key, value) for key, value in previous.items() if key not in versions))
            self.targets[handle] = versions
            self._control(handle, 'activate', tuple((key, value) for key, value in versions.items() if key not in previous))
            self._send(('update', handle, 0, (context, versions)))

    def cancel(self, handle):
        with self.lock:
            if handle in self.targets:
                self._control(handle, 'invalidate', tuple(self.targets.pop(handle).items()))
            self.versions[handle] = self.versions.get(handle, 0) + 1
            self._send(('cancel', handle, self.versions[handle], None))

    def _control(self, handle, kind, entries):
        if not entries:
            return
        seq = self.control_seq[handle] = self.control_seq.get(handle, 0) + 1
        session = self.sessions.get(handle)
        if session is not None:
            session.receive('control', (seq, kind, entries))

    def foreground(self, handle, active):
        with self.lock:
            if active:
                self.active_foregrounds.add(handle)
            else:
                self.active_foregrounds.discard(handle)
            if self.active_foregrounds:
                self.paused.value = 1
            else:
                self.paused.value = 0
        self._send(('foreground', handle, 0, active))

    def detach(self, handle):
        self.foreground(handle, False)
        self.cancel(handle)
        with self.lock:
            self.sessions.pop(handle, None)
            self._send(('detach', handle, self.versions[handle], None))

    def _fail(self, reason):
        with self.lock:
            if self.failed:
                return
            self.failed = reason
            logging.getLogger(__name__).warning('response prefetch disabled: %s', reason)
            for session in self.sessions.values():
                session.receive('error', reason)
            if reason in {'AuditWriteError', 'AuditIntegrityError'} and self.audit_failure is not None:
                self.audit_failure(reason)

    def _read(self):
        try:
            while not self.closed:
                try:
                    kind, handle, version, payload = self.results.get(timeout=.1)
                except queue.Empty:
                    if not self.closed and not self.process.is_alive():
                        self._fail('worker_exit')
                        return
                    continue
                if kind == 'fatal':
                    self._fail(payload)
                    return
                if kind == 'stats':
                    with self.stats_ready:
                        self.metrics.update(payload)
                        self.stats_received = max(self.stats_received, version)
                        self.stats_ready.notify_all()
                    continue
                with self.lock:
                    session = self.sessions.get(handle)
                    if session is not None and version in self.targets.get(handle, {}).values():
                        session.receive(kind, payload)
                    else:
                        self.metrics['stale_messages'] += 1
        except BaseException as error:
            self._fail(type(error).__name__)

    def close(self):
        if self.closed:
            return
        self._send(('close', None, 0, None))
        self.closed = True
        self.reader.join(1)
        self.process.join(2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)
        for channel in (self.commands, self.results):
            channel.cancel_join_thread()
            channel.close()
        self.directory.cleanup()
