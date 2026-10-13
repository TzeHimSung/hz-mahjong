"""Participant state-request budgets shared by train and race orchestration."""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any, Callable

from heuristic_ai.protocol.client import RequestDeadlineExceeded
from .bot import GameStopped, JsonClient


_GAME_STATE_GET = re.compile(r"^/api/games/[^/?]+/state(?:\?.*)?$")


_STATE_REQUEST_RATE = 15.5


class ParticipantStateRequestGate:
    """No-burst state-poll budget shared by one race participant.

    Admission is earliest-deadline-first rather than plain FIFO. Every table of
    one identity draws on the same 15.5/s budget, and a live match keeps ten
    tables polling, so the budget is saturated: measured admission for a
    response-window snapshot was 312 ms at p50 (531 ms at p90) against 31 ms of
    actual transport, out of a one-second window that also owes 300 ms to the
    submission reserve. Callers already pass their own deadline, so the urgent
    window read is admitted ahead of background discovery without new plumbing.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._interval = 1.0 / _STATE_REQUEST_RATE
        self._condition = threading.Condition()
        self._waiters: deque[tuple[float | None, int, object]] = deque()
        self._serial = 0
        self._next_allowed_at: float | None = None

    def _head(self) -> tuple[float | None, int, object]:
        """The waiter that may take the next slot: earliest deadline, then arrival."""
        return min(self._waiters, key=lambda item: (
            (0, item[0], item[1]) if item[0] is not None else (1, 0.0, item[1])))

    def wait(
        self,
        stop: threading.Event,
        *,
        deadline_monotonic: float | None = None,
    ) -> None:
        with self._condition:
            self._serial += 1
            ticket = (deadline_monotonic, self._serial, object())
            self._waiters.append(ticket)
            try:
                while True:
                    if stop.is_set():
                        raise GameStopped("game stopped before state request")
                    now = self._clock()
                    remaining = (
                        None if deadline_monotonic is None
                        else deadline_monotonic - now
                    )
                    if remaining is not None and remaining <= 0:
                        raise RequestDeadlineExceeded("state request admission deadline exceeded")
                    wait_slice = 0.05 if remaining is None else min(0.05, remaining)
                    if self._head() is not ticket:
                        self._condition.wait(wait_slice)
                        continue
                    delay = (
                        0.0
                        if self._next_allowed_at is None
                        else self._next_allowed_at - now
                    )
                    if delay > 0:
                        self._condition.wait(min(delay, wait_slice))
                        continue
                    self._waiters.remove(ticket)
                    self._next_allowed_at = now + self._interval
                    self._condition.notify_all()
                    return
            finally:
                if ticket in self._waiters:
                    self._waiters.remove(ticket)
                    self._condition.notify_all()


class _StateRequestAttemptGate:
    """Adapt a participant gate to ApiClient's per-attempt request hook."""

    def __init__(
        self,
        request_gate: ParticipantStateRequestGate,
        stop: threading.Event,
    ) -> None:
        self._request_gate = request_gate
        self._stop = stop

    def wait(self, method: str, path: str) -> None:
        if method == "POST" and path.startswith("/api/games/") and path.endswith("/action") and self._stop.is_set():
            raise GameStopped("game stopped before action request attempt")
        if method == "GET" and _GAME_STATE_GET.fullmatch(path) is not None:
            self._request_gate.wait(self._stop)

    def wait_until(self, method: str, path: str, deadline_monotonic: float) -> None:
        if method == "POST" and path.startswith("/api/games/") and path.endswith("/action") and self._stop.is_set():
            raise GameStopped("game stopped before action request attempt")
        if method == "GET" and _GAME_STATE_GET.fullmatch(path) is not None:
            self._request_gate.wait(self._stop, deadline_monotonic=deadline_monotonic)


class StateLimitedClient:
    """Gate only game-state GETs while delegating every other API request."""

    def __init__(
        self,
        delegate: JsonClient,
        request_gate: ParticipantStateRequestGate,
        stop: threading.Event,
    ) -> None:
        self.delegate = delegate
        self.request_gate = request_gate
        self._stop = stop
        bind_request_gate = getattr(delegate, "bind_request_gate", None)
        self._delegate_gates_attempts = callable(bind_request_gate)
        if self._delegate_gates_attempts:
            bind_request_gate(_StateRequestAttemptGate(request_gate, stop))
        bind_retry_wait = getattr(delegate, "bind_retry_wait", None)
        if callable(bind_retry_wait):
            bind_retry_wait(self._retry_wait)

    def _retry_wait(self, seconds: float) -> None:
        if self._stop.wait(seconds):
            raise GameStopped("game stopped while waiting for HTTP retry")

    def request(self, method: str, path: str, body: Any = None, **options: Any) -> Any:
        if method == "POST" and path.startswith("/api/games/") and path.endswith("/action") and self._stop.is_set():
            raise GameStopped("game stopped before action request")
        if (
            not self._delegate_gates_attempts
            and method == "GET"
            and _GAME_STATE_GET.fullmatch(path) is not None
        ):
            self.request_gate.wait(self._stop)
        return self.delegate.request(method, path, body, **options)

    def request_once(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        deadline_monotonic: float,
        **options: Any,
    ) -> Any:
        bounded_request = getattr(self.delegate, "request_once", None)
        if not callable(bounded_request):
            raise RuntimeError("client does not support bounded state reads")
        if (
            not self._delegate_gates_attempts
            and method == "GET"
            and _GAME_STATE_GET.fullmatch(path) is not None
        ):
            self.request_gate.wait(self._stop, deadline_monotonic=deadline_monotonic)
        return bounded_request(
            method, path, body, deadline_monotonic=deadline_monotonic, **options,
        )
