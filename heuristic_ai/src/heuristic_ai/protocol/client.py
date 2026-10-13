from __future__ import annotations

import json
import math
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
import uuid
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping, Protocol

SENSITIVE_KEYS = {"access_token", "authorization", "secret", "token", "global_token"}


def redact(value: Any) -> Any:
    """Recursively replace credential-shaped fields before persistence or logging."""

    if isinstance(value, dict):
        return {
            key: "<redacted>" if key.lower() in SENSITIVE_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    data: bytes
    timings: Mapping[str, Any] | None = None


class ApiError(RuntimeError):
    """An HTTP failure whose response body has already been redacted."""

    def __init__(self, status: int, body: Any, *, retry_after: str | None = None) -> None:
        safe = redact(body)
        self.status = status
        self.body = safe
        self.code = safe.get("code") if isinstance(safe, dict) else None
        # Expose only the scheduling header, never arbitrary response headers.
        self.retry_after = retry_after
        super().__init__(f"HTTP {status}: {safe}")


class RequestDeadlineExceeded(TimeoutError):
    """A request reached its send/response deadline; sent preserves uncertainty."""

    def __init__(self, message: str, *, sent: bool = False, response_status: int | None = None) -> None:
        self.sent = sent
        self.response_status = response_status
        super().__init__(message)


class StateChangedBeforeSend(RuntimeError):
    """A newer public action invalidated a request before any mutation was sent."""
    sent = False


class RetryPolicy(StrEnum):
    SAFE_READ = "safe_read"
    NEVER = "never"


@dataclass(frozen=True, slots=True)
class RequestIntent:
    method: str
    path: str
    purpose: str = "control"
    deadline_monotonic: float | None = None
    priority_deadline_monotonic: float | None = None


Transport = Callable[[urllib.request.Request, float, ssl.SSLContext], HttpResponse]


class RequestAttemptGate(Protocol):
    """Optional hook invoked immediately before every transport attempt."""

    def wait(self, method: str, path: str) -> None:
        raise NotImplementedError

    def wait_until(self, method: str, path: str, deadline_monotonic: float) -> None:
        raise NotImplementedError


class ApiClient:
    """Small JSON client with bounded retry and optional Bearer authentication."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        insecure: bool = False,
        timeout: float = 35.0,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        transport: Transport,
        request_gate: RequestAttemptGate | None = None,
        clock: Callable[[], float] = time.monotonic,
        health_operation: Callable[[str, float | None], AbstractContextManager[Any]] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._transport = transport
        self._request_gate = request_gate
        self._clock = clock
        self._health_operation = health_operation
        self._observer = observer
        self._context = ssl_context or (
            ssl._create_unverified_context()
            if insecure
            else ssl.create_default_context()
        )

    def bind_request_gate(self, request_gate: RequestAttemptGate) -> None:
        """Attach a per-attempt gate before this client is shared or used."""

        if self._request_gate is not None and self._request_gate is not request_gate:
            raise RuntimeError("request gate is already configured")
        self._request_gate = request_gate

    def bind_retry_wait(self, wait: Callable[[float], None]) -> None:
        """Use the owner's cancellation-aware wait before this worker starts."""
        self._sleep = wait

    def _operation(self, label: str, seconds: float | None = None) -> AbstractContextManager[Any]:
        # A plain optional callback keeps the protocol layer independent of runtime.
        return self._health_operation(label, seconds) if self._health_operation else nullcontext()

    def _retry_wait(self, seconds: float) -> None:
        with self._operation("http.retry_wait", seconds):
            self._sleep(seconds)

    def _make_request(self, method: str, path: str, body: Any) -> urllib.request.Request:
        data = (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        )
        request = urllib.request.Request(
            self._base_url + path,
            data=data,
            method=method,
        )
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if self._token:
            request.add_header("Authorization", "Bearer " + self._token)
        return request

    def request_once(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        deadline_monotonic: float,
        purpose: str = "inventory_backfill",
    ) -> Any:
        """Perform one bounded GET; admission and transport share its budget.

        This is used for optional public-event backfill. It never retries or
        changes the normal request policy, especially action POST semantics.
        """
        if method != "GET" or body is not None:
            raise ValueError("bounded single requests support bodyless GET only")
        if (
            type(deadline_monotonic) not in {int, float}
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("request deadline must be a finite monotonic timestamp")
        result = self.request(method, path, deadline_monotonic=deadline_monotonic,
                              purpose=purpose, retry_policy=RetryPolicy.NEVER)
        if self._clock() >= deadline_monotonic:
            raise RequestDeadlineExceeded("bounded state read deadline exceeded")
        return result

    @staticmethod
    def _retry_after(response: HttpResponse) -> str | None:
        return next((value for key, value in response.headers.items()
                     if key.lower() == "retry-after"), None)

    def request(
        self, method: str, path: str, body: Any = None, *,
        deadline_monotonic: float | None = None, purpose: str = "control",
        retry_policy: RetryPolicy | str = RetryPolicy.SAFE_READ,
        before_send: Callable[[], None] | None = None,
        priority_deadline_monotonic: float | None = None,
        request_timing: dict[str, Any] | None = None,
    ) -> Any:
        retry_policy = RetryPolicy(retry_policy)
        router = getattr(self, "request_router", None)
        if callable(router) and method == "GET" and path.startswith("/api/games/") and "/state" in path:
            return router(method, path, body, deadline_monotonic=deadline_monotonic,
                          purpose=purpose, retry_policy=retry_policy,
                          priority_deadline_monotonic=priority_deadline_monotonic)
        if priority_deadline_monotonic is not None and (
            type(priority_deadline_monotonic) not in {int, float} or not math.isfinite(priority_deadline_monotonic)
        ):
            raise ValueError("priority deadline must be a finite monotonic timestamp")
        if deadline_monotonic is not None:
            if type(deadline_monotonic) not in {int, float} or not math.isfinite(deadline_monotonic):
                raise ValueError("request deadline must be a finite monotonic timestamp")
            if deadline_monotonic <= self._clock():
                raise RequestDeadlineExceeded("request expired before admission")
        request = self._make_request(method, path, body)
        request.deadline_monotonic = deadline_monotonic
        request.purpose = purpose
        request.before_send = before_send
        request.request_timing = request_timing
        intent = RequestIntent(method, path, purpose, deadline_monotonic, priority_deadline_monotonic)
        safe_read = method in {"GET", "HEAD"}
        tournament_post = method == "POST" and re.fullmatch(
            r"/api/tournaments/[^/?]+/(?:register|ready)", path
        ) is not None
        # v35 permits retrying an explicit TOURNAMENT_GONE for these idempotent
        # writes. A lost ACK, action POST or matching request is never replayed.
        attempts = (self._max_attempts if retry_policy is RetryPolicy.SAFE_READ
                    and (safe_read or tournament_post) else 1)

        last_response_status = None
        for attempt in range(attempts):
            delay = float(2**attempt)
            try:
                response, parsed = self._attempt(request, intent, attempt + 1)
            except RequestDeadlineExceeded as error:
                if error.response_status is None:
                    error.response_status = last_response_status
                raise
            except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
                if not safe_read or attempt + 1 == attempts:
                    raise RuntimeError(f"transport failed after {attempt + 1} attempts: {type(error).__name__}") from error
            else:
                if 200 <= response.status < 300:
                    return parsed
                last_response_status = response.status
                value = self._retry_after(response)
                retry_after_seconds = None
                if value is not None:
                    try:
                        seconds = float(value)
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        try:
                            seconds = parsedate_to_datetime(value).timestamp() - time.time()
                        except (ValueError, TypeError, OverflowError):
                            seconds = 0
                    if math.isfinite(seconds):
                        retry_after_seconds = max(0,seconds)
                        delay = max(delay, seconds)
                if response.status == 429 and method == 'GET' and '/state' in path:
                    state_delay = getattr(self._request_gate, 'state_retry_delay', None)
                    if callable(state_delay):
                        delay = state_delay(delay,retry_after_seconds)
                    defer_state = getattr(self._request_gate, 'defer_state', None)
                    if callable(defer_state):
                        # Share the backoff with the next bounded call and
                        # other tables using this identity. Otherwise a retry
                        # that cannot fit its deadline restarts immediately.
                        defer_state(delay)
                        if self._observer:
                            self._observer({'kind':'state_backoff','request_id':request.request_id,
                                'method':method,'path':path,'purpose':purpose,
                                'started_monotonic':self._clock(),'delay_seconds':delay,
                                'server_retry_after_seconds':retry_after_seconds})
                tournament_gone = (path.startswith("/api/tournaments/")
                                   and response.status == 404 and isinstance(parsed, dict)
                                   and parsed.get("code") == "TOURNAMENT_GONE")
                retryable = (tournament_gone and (safe_read or tournament_post)
                             or safe_read and (response.status == 429 or response.status >= 500))
                if not retryable or attempt + 1 == attempts:
                    raise ApiError(response.status, parsed, retry_after=self._retry_after(response))
            if deadline_monotonic is not None and self._clock() + delay >= deadline_monotonic:
                raise RequestDeadlineExceeded("read retry would exceed request deadline",
                                              response_status=last_response_status)
            self._retry_wait(delay)
        raise RuntimeError("unreachable retry loop")

    def _attempt(self, request: urllib.request.Request, intent: RequestIntent, number: int):
        started = self._clock()
        request.request_id = uuid.uuid4().hex
        request.timings = {"request_id": request.request_id}
        admitted = None
        response = None
        failure = None
        failure_sent = None
        decoded_ms = 0.0
        def remaining():
            if intent.deadline_monotonic is None:
                return self._timeout
            value = intent.deadline_monotonic - self._clock()
            if value <= 0:
                raise RequestDeadlineExceeded("request expired before transport")
            return min(self._timeout, value)
        try:
            remaining()
            if self._request_gate is not None:
                with self._operation("http.admission"):
                    contextual_wait = getattr(self._request_gate, "wait_for", None)
                    if callable(contextual_wait):
                        contextual_wait(intent)
                    elif intent.deadline_monotonic is not None:
                        self._request_gate.wait_until(intent.method, intent.path, intent.deadline_monotonic)
                    else:
                        self._request_gate.wait(intent.method, intent.path)
            admitted = self._clock()
            if request.request_timing is not None:
                request.request_timing.update(request_started_monotonic=started,
                    transport_started_monotonic=admitted)
            timeout = remaining()
            guard = getattr(request, 'before_send', None)
            if guard is not None and not getattr(self._transport, 'supports_pre_send_guard', False):
                guard()
            response_seconds = (getattr(self._transport, 'action_response_timeout_seconds', 0)
                if intent.method == 'POST' and intent.path.endswith('/action') else 0)
            with self._operation("http.transport", timeout + response_seconds):
                response = self._transport(request, timeout, self._context)
            decode_started = self._clock()
            parsed = json.loads(response.data.decode("utf-8")) if response.data else None
            decoded_ms = (self._clock() - decode_started) * 1000
            return response, parsed
        except BaseException as error:
            failure = type(error).__name__
            if isinstance(error, (RequestDeadlineExceeded, StateChangedBeforeSend)):
                failure_sent = error.sent
            raise
        finally:
            if self._observer:
                ended = self._clock()
                row = dict(response.timings or {}) if response is not None else dict(getattr(request, "timings", {}))
                row.update(kind="http", method=intent.method, path=intent.path, purpose=intent.purpose,
                           attempt=number, admission_ms=((admitted if admitted is not None else ended) - started) * 1000,
                           total_ms=(ended - started) * 1000, json_decode_ms=decoded_ms,
                           requested_monotonic=started, deadline_monotonic=intent.deadline_monotonic,
                           priority_deadline_monotonic=intent.priority_deadline_monotonic,
                           http_status=response.status if response is not None else None, error_type=failure)
                row.setdefault("elapsed_ms", (ended - admitted) * 1000 if admitted is not None else 0.0)
                if row.get('sent') is None and failure_sent is not None:
                    row['sent'] = failure_sent
                self._observer(row)
