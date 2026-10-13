"""Owned persistent pools with cancellable absolute deadlines and safe telemetry.

One asyncio loop serves the synchronous game workers in the I/O process,
isolated from strategy computation.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
import ssl
import logging
import os
import socket
import sys
import math
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit
import urllib.error
import urllib.request
import uuid
import types

import anyio
import httpx
import httpcore
import h2.exceptions

from .client import HttpResponse, RequestDeadlineExceeded


if sys.platform == "win32":
    class _OwnedProactorLoop(asyncio.ProactorEventLoop):
        """Recover the wakeup socket of this background HTTP loop only."""

        def _loop_self_reading(self, future=None):
            if (future is not None and future is self._self_reading_future
                    and not future.cancelled()):
                reason = None
                try:
                    if future.result() == b"":
                        reason = "EOF"
                except OSError as error:
                    reason = type(error).__name__
                if reason is not None:
                    # CPython gh-156333: re-arming a dead self-pipe spins forever.
                    # Allocate first, retaining the old handles if setup fails.
                    reader, writer = socket.socketpair()
                    try:
                        reader.setblocking(False)
                        writer.setblocking(False)
                    except BaseException:
                        reader.close()
                        writer.close()
                        raise
                    previous_reader, previous_writer = self._ssock, self._csock
                    self._ssock, self._csock = reader, writer
                    self._self_reading_future = None
                    previous_reader.close()
                    previous_writer.close()
                    logging.getLogger(__name__).warning(
                        "http_wakeup_rebuilt pid=%s thread=%s monotonic=%.6f reason=%s",
                        os.getpid(), threading.get_ident(), time.monotonic(), reason)
                    future = None
            super()._loop_self_reading(future)


class HttpxRuntime:
    """A run-owned event loop; no global clients, threads or credentials."""

    def __init__(self) -> None:
        self._ready = Future()
        self._closed = False
        self._owners: list[HttpxTransport] = []
        self._thread = threading.Thread(target=self._serve, name="mahjong-http", daemon=True)
        self._thread.start()
        self._ready.result()

    def _serve(self) -> None:
        # Create on the owning thread; an HTTP loop must not install a process
        # signal wakeup fd from the application's main thread.
        try:
            self.loop = _OwnedProactorLoop() if sys.platform == "win32" else asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        except BaseException as error:
            self._ready.set_exception(error)
            return
        self._ready.set_result(None)
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def submit(self, coroutine):
        if self._closed:
            coroutine.close()
            raise RuntimeError("HTTP runtime is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def close(self) -> None:
        if self._closed:
            return
        async def close_owners():
            for owner in self._owners:
                await owner._close_clients()
        future = self.submit(close_owners())
        try:
            future.result(timeout=5)
        finally:
            self._closed = True
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(5)
        if self._thread.is_alive():
            raise RuntimeError("HTTP runtime failed to stop")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _CancelledWriteStream:
    """Do not reuse HTTP/2 framing/TLS state after an interrupted write.

    httpcore 1.0.9 removes frames from h2's outgoing buffer before awaiting
    the stream write, and catches Exception there. CancelledError bypasses
    that handler, even though bytes may already have left the TLS BIO.
    Only cancellation inside write closes the connection. Cancellation while
    waiting for the HTTP/2 write lock or awaiting a response is unaffected.
    """
    def __init__(self, connection, stream, on_abort):
        self.connection, self.stream, self.on_abort = connection, stream, on_abort

    async def read(self, max_bytes, timeout=None):
        return await self.stream.read(max_bytes, timeout)

    async def write(self, buffer, timeout=None):
        try:
            await self.stream.write(buffer, timeout)
        except asyncio.CancelledError:
            self.connection._connection_error = True
            self.connection._write_exception = httpcore.WriteError('HTTP/2 connection closed after cancelled write')
            self.on_abort()
            try:
                await self.connection.aclose()
            except Exception:
                # Availability was revoked before cleanup; preserve cancellation.
                pass
            raise

    async def aclose(self):
        await self.stream.aclose()

    async def start_tls(self, *args, **kwargs):
        return await self.stream.start_tls(*args, **kwargs)

    def get_extra_info(self, name):
        return self.stream.get_extra_info(name)


class HttpxTransport:
    """Separate state/action/control pools for one participant identity."""

    supports_pre_send_guard = True

    def __init__(
        self, *, runtime: HttpxRuntime | None = None, insecure: bool = False,
        trust_env: bool = True, max_connections: int = 16,
        observer: Callable[[dict[str, Any]], None] | None = None,
        action_response_timeout_seconds: float = 0,
        state_locks: dict[str, asyncio.Lock] | None = None,
    ) -> None:
        if max_connections < 1 or max_connections > 32:
            raise ValueError("connection count must be between one and 32")
        if (not math.isfinite(action_response_timeout_seconds)
                or not 0 <= action_response_timeout_seconds <= 5):
            raise ValueError('action response timeout must be between zero and five seconds')
        self.runtime = runtime or HttpxRuntime()
        self._own_runtime = runtime is None
        self.context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        self.trust_env = trust_env
        self.max_connections, self.observer = max_connections, observer
        self.action_response_timeout_seconds = action_response_timeout_seconds
        self._clients: dict[tuple[str, ssl.SSLContext], httpx.AsyncClient] = {}
        self._slots: dict[tuple[str, ssl.SSLContext], asyncio.Semaphore] = {}
        self._state_locks = state_locks if state_locks is not None else {}
        self._orphan_reaped: dict[str, int] = {}
        self.state_pool_timeout_at: float | None = None
        self._http2_write_aborts: dict[str, int] = {}
        self._http2_stream_resets: dict[str, int] = {}
        self._closed = False
        self.runtime._owners.append(self)

    @staticmethod
    def channel(request: urllib.request.Request) -> str:
        path = urlsplit(request.full_url).path
        if path.startswith("/api/games/"):
            for channel in ("state", "action"):
                if path.endswith("/" + channel):
                    return channel
        return "control"

    def _client(self, channel: str, context: ssl.SSLContext | None = None) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("HTTP transport is closed")
        context = context or self.context
        key = (channel, context)
        if key not in self._clients:
            count = self.max_connections if channel != "control" else 4
            # Short calls are cancelled at gameplay deadlines. Give them
            # HTTP/1.1 connections so cancellation cannot corrupt shared H2
            # stream permits; only long-lived notification streams use H2.
            self._clients[key] = httpx.AsyncClient(
                verify=context, http2=channel == "notify", trust_env=self.trust_env,
                limits=httpx.Limits(max_connections=count, max_keepalive_connections=count,
                                   keepalive_expiry=60),
            )
            self._slots[key] = asyncio.Semaphore(count)
            self._guard_pools(self._clients[key], channel)
        return self._clients[key]

    def _guard_pools(self, client, channel):
        """Narrow guard for httpcore 1.0.9's cancelled-assignment orphan race.

        Only this owner's instances are modified, including proxy transports.
        Pinning the dependency makes this internal integration explicit
        and testable on upgrades.
        """
        if httpcore.__version__ != '1.0.9':
            raise RuntimeError('HTTP pool cancellation guard requires httpcore 1.0.9')
        transports = {id(t): t for t in (client._transport, *client._mounts.values()) if t is not None}
        for transport in transports.values():
            pool = transport._pool
            original = pool._assign_requests_to_connections
            def assign(instance, original=original):
                owners = {id(request.connection) for request in instance._requests if request.connection is not None}
                orphans = [connection for connection in instance._connections
                           if not connection.is_idle() and not connection.is_closed() and id(connection) not in owners]
                for connection in orphans:
                    instance._connections.remove(connection)
                self._orphan_reaped[channel] = self._orphan_reaped.get(channel, 0) + len(orphans)
                return orphans + original()
            pool._assign_requests_to_connections = types.MethodType(assign, pool)

    def _protect_http2_connections(self, client, channel):
        from httpcore._async.http2 import AsyncHTTP2Connection
        transports = {id(t): t for t in (client._transport, *client._mounts.values()) if t is not None}
        for transport in transports.values():
            for connection in transport._pool._connections:
                seen = set()
                while connection is not None and id(connection) not in seen:
                    seen.add(id(connection))
                    if isinstance(connection, AsyncHTTP2Connection):
                        if not isinstance(connection._network_stream, _CancelledWriteStream):
                            def aborted():
                                self._http2_write_aborts[channel] = self._http2_write_aborts.get(channel, 0) + 1
                            connection._network_stream = _CancelledWriteStream(
                                connection, connection._network_stream, aborted)
                        if not getattr(connection, '_mahjong_response_reset_guard', False):
                            original_close = connection._response_closed
                            async def response_closed(instance, stream_id, original_close=original_close):
                                # httpcore 1.0.9 releases only its local stream
                                # slot on early close. Without RST_STREAM the
                                # server can keep a cancelled long poll open.
                                # Reset this stream without closing its peers.
                                import h2.errors
                                import h2.events
                                try:
                                    stream = instance._h2_state.streams.get(stream_id)
                                    if instance.is_available():
                                        flush = False
                                        for event in instance._events.get(stream_id, ()):
                                            if isinstance(event, h2.events.DataReceived):
                                                instance._h2_state.acknowledge_received_data(
                                                    event.flow_controlled_length, stream_id)
                                                flush = True
                                        reset = stream is not None and not stream.closed
                                        if reset:
                                            instance._h2_state.reset_stream(stream_id, h2.errors.ErrorCodes.CANCEL)
                                            flush = True
                                        # Cleanup has its own small bound. The
                                        # original request has already expired;
                                        # no new application request is sent.
                                        if flush:
                                            reset_request = httpcore.Request('GET', httpcore.URL(),
                                                extensions={'timeout': {'write': .2}})
                                            async with asyncio.timeout(.2):
                                                await instance._write_outgoing_data(reset_request)
                                        if reset:
                                            self._http2_stream_resets[channel] = self._http2_stream_resets.get(channel, 0)+1
                                except Exception:
                                    # If the reset cannot be written, revoke
                                    # reuse so abandoned streams cannot pile up.
                                    instance._connection_error = True
                                    try:
                                        await instance.aclose()
                                    except Exception:
                                        pass
                                finally:
                                    await original_close(stream_id=stream_id)
                            connection._response_closed = types.MethodType(response_closed, connection)
                            connection._mahjong_response_reset_guard = True
                        break
                    connection = getattr(connection, '_connection', None)

    async def _close_clients(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(*(client.aclose() for client in self._clients.values()))
        self._clients.clear()
        self._state_locks.clear()

    def close(self) -> None:
        if self._own_runtime:
            self.runtime.close()
        elif not self._closed:
            self.runtime.submit(self._close_clients()).result(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __call__(self, request: urllib.request.Request, timeout: float, context: ssl.SSLContext) -> HttpResponse:
        deadline = getattr(request, "deadline_monotonic", None)
        intent = {"sent": False}
        future = self.runtime.submit(self._request(request, timeout, deadline, intent, context))
        try:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            try:
                return future.result(timeout=remaining)
            except FutureTimeout:
                # Once the body was sent before the deadline, cancelling its
                # reply cannot undo the mutation. Preserve its ACK within a
                # separate bounded response interval. Never resend the POST.
                reply_deadline = intent.get('response_deadline_monotonic')
                if reply_deadline is None or future.done():
                    raise
                return future.result(timeout=max(0, reply_deadline-time.monotonic()))
        except FutureTimeout:
            # The coroutine owns the sole cancellation timer. A second
            # Task.cancel() from this waiting thread can interrupt httpcore's
            # cancellation cleanup and strand a connection in the pool.
            # Admission/trace checks forbid a queued request from sending late.
            raise RequestDeadlineExceeded("HTTP request deadline exceeded", sent=intent["sent"]) from None
        except (httpx.HTTPError, h2.exceptions.ProtocolError, ssl.SSLError) as error:
            # Preserve the legacy transport-error contract without exposing
            # response bodies, headers, credentials or proxy URLs in exceptions.
            # The backend can also expose raw HTTP/2 and TLS exceptions.
            timings = getattr(request, "timings", None)
            if isinstance(timings, dict):
                timings["transport_error_type"] = type(error).__name__
            raise urllib.error.URLError(type(error).__name__) from None

    async def _request(self, request, timeout, deadline, intent, context):
        started = time.monotonic()
        if deadline is not None and started >= deadline:
            raise RequestDeadlineExceeded("HTTP request expired in loop queue")
        row: dict[str, Any] = {
            "kind": "http", "request_id": getattr(request, "request_id", uuid.uuid4().hex),
            "method": request.get_method(), "path": urlsplit(request.full_url).path,
            "purpose": getattr(request, "purpose", "unspecified"),
            "pool": self.channel(request), "http_status": None, "http_version": None,
            "started_monotonic": started, "request_bytes": len(request.data or b""),
            "response_bytes": 0, "connect_ms": 0.0, "tls_ms": 0.0,
            "write_ms": 0.0, "response_headers_ms": 0.0, "response_body_ms": 0.0,
            "pool_wait_ms": 0.0, "pool_wait_scope": "slot_admission_and_pre_io",
        }
        traces: dict[str, float] = {}
        request.timings = row
        first_io: float | None = None
        response_seconds = (self.action_response_timeout_seconds
            if request.get_method() == 'POST' and row['pool'] == 'action' and deadline is not None else 0)
        timeout_scope = None
        async def trace(name: str, info: dict[str, Any]):
            nonlocal first_io
            now = time.monotonic()
            operation, _, stage = name.rpartition(".")
            # httpcore traces cleanup before releasing the HTTP/2 stream
            # semaphore. A deadline exception here skips that release and can
            # permanently exhaust an otherwise reusable connection's streams.
            if operation.endswith(('response_closed', 'close')):
                return
            if stage == "started":
                sending = operation.endswith(('send_request_headers', 'send_request_body'))
                effective_deadline = deadline if sending else intent.get('response_deadline_monotonic', deadline)
                if effective_deadline is not None and now >= effective_deadline:
                    raise RequestDeadlineExceeded("HTTP phase deadline exceeded", sent=intent["sent"])
                if operation.endswith(('send_connection_init', 'send_request_headers')):
                    # This trace runs after protocol negotiation and before its
                    # first write, including HTTP/2 inside CONNECT proxies.
                    self._protect_http2_connections(client, row['pool'])
                first_io = now if first_io is None else first_io
                traces[operation] = now
                traced_request = info.get("request")
                is_connect = getattr(traced_request, "method", None) == b"CONNECT"
                if operation.endswith("send_request_headers") and not is_connect:
                    guard = getattr(request, 'before_send', None)
                    if guard is not None:
                        guard()
                    intent["sent"] = True
            elif stage in {"complete", "failed"} and operation in traces:
                elapsed = (now - traces.pop(operation)) * 1000
                field = (
                    "connect_ms" if operation.endswith("connect_tcp") else
                    "tls_ms" if operation.endswith("start_tls") else
                    "write_ms" if operation.endswith(("send_request_headers", "send_request_body")) else
                    "response_headers_ms" if operation.endswith("receive_response_headers") else
                    "response_body_ms" if operation.endswith("receive_response_body") else None
                )
                if field:
                    row[field] += elapsed
                if (stage == 'complete' and operation.endswith('send_request_body')
                        and response_seconds and now < deadline and timeout_scope is not None
                        and not timeout_scope.cancel_called):
                    reply_deadline = now + response_seconds
                    row['request_body_finished_monotonic'] = now
                    row['response_deadline_monotonic'] = reply_deadline
                    intent['response_deadline_monotonic'] = reply_deadline
                    timeout_scope.deadline = anyio.current_time() + response_seconds
        try:
            client = self._client(row["pool"], context)
            seconds = None if deadline is None else max(0, deadline - time.monotonic())
            # httpcore shields response cleanup with AnyIO cancel scopes.
            # Use the same cancellation mechanism so the request deadline
            # cannot interrupt cleanup after a stream marked itself closed.
            with anyio.fail_after(seconds) as timeout_scope:
                # Cancellation is asynchronous. Keep the per-table lock inside
                # the coroutine so a replacement cannot overlap its cleanup.
                @asynccontextmanager
                async def state_slot():
                    if row['pool'] == 'state':
                        lock = self._state_locks.setdefault(row['path'], asyncio.Lock())
                        async with lock:
                            yield
                    else:
                        yield
                async with state_slot(), self._slots[(row["pool"], context)]:
                    admitted = time.monotonic()
                    row["pool_wait_ms"] = (admitted - started) * 1000
                    # httpcore 1.0.9 broadcasts a socket ReadTimeout to all H2
                    # streams. With an absolute deadline, let this coroutine's
                    # timer cancel/reset only its stream; it also bounds a sent
                    # action's reply grace. Unbounded calls keep the read limit.
                    response = await client.request(
                        request.get_method(), request.full_url,
                        headers=dict(request.header_items()), content=request.data,
                        timeout=httpx.Timeout(timeout, read=None if deadline is not None else timeout),
                        extensions={"trace": trace},
                    )
                    row.update(http_status=response.status_code, http_version=response.http_version,
                               response_bytes=len(response.content))
                    if row['pool'] == 'state':
                        # Any actual HTTP response proves this pool is delivering again.
                        self.state_pool_timeout_at = None
                    if first_io is not None:
                        row["pool_wait_ms"] += max(0, first_io - admitted) * 1000
                    return HttpResponse(response.status_code, dict(response.headers), response.content, row)
        except BaseException as error:
            row["error_type"] = type(error).__name__
            if row['pool'] == 'state' and isinstance(error, httpx.PoolTimeout):
                self.state_pool_timeout_at = time.monotonic()
                try:
                    transports = {id(t): t for t in (client._transport, *client._mounts.values())
                                  if t is not None}
                    connections = [c for t in transports.values() for c in t._pool._connections]
                    requests = [r for t in transports.values() for r in t._pool._requests]
                    owners = {id(r.connection) for r in requests if r.connection is not None}
                    row['state_pool'] = {'connections': len(connections), 'requests': len(requests),
                        'queued_requests': sum(r.is_queued() for r in requests),
                        'idle_connections': sum(c.is_idle() for c in connections),
                        'closed_connections': sum(c.is_closed() for c in connections),
                        'available_connections': sum(c.is_available() for c in connections),
                        'unowned_active_connections': sum(not c.is_idle() and not c.is_closed()
                            and id(c) not in owners for c in connections),
                        'locked_state_paths': sum(lock.locked() for lock in self._state_locks.values())}
                except Exception as diagnostic_error:
                    row['state_pool_diagnostic_error'] = type(diagnostic_error).__name__
            raise
        finally:
            row.update(finished_monotonic=time.monotonic(), sent=intent["sent"])
            if row['pool'] == 'state':
                row['state_pool_timeout_at'] = self.state_pool_timeout_at
            row['pool_orphans_reclaimed_total'] = self._orphan_reaped.get(row['pool'], 0)
            row['http2_cancelled_writes_closed_total'] = self._http2_write_aborts.get(row['pool'], 0)
            row['http2_incomplete_streams_reset_total'] = self._http2_stream_resets.get(row['pool'], 0)
            row["elapsed_ms"] = (row["finished_monotonic"] - started) * 1000
            if self.observer:
                self.observer(dict(row))
