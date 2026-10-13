"""Per-run transport ownership shared by train, race and continuous match."""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading
from typing import Any

from heuristic_ai.config.models import DEFAULT_ACTION_RESERVES, ActionReserves, ServerConfig
from heuristic_ai.protocol.client import ApiClient, redact
from heuristic_ai.protocol.httpx_transport import HttpxRuntime, HttpxTransport
from .communication_metrics import CommunicationJournal


class RunNetwork:
    def __init__(self, server: ServerConfig, output_root: Path | None = None, *,
                 action_reserves: ActionReserves = DEFAULT_ACTION_RESERVES) -> None:
        self.server = server
        self.action_reserves = action_reserves
        self.output_root = output_root
        self._runtime: HttpxRuntime | None = None
        self._transports: dict[str, HttpxTransport] = {}
        self._labels: dict[str, dict[str, str]] = {}
        self._tokens: set[str] = set()
        self._lock = threading.RLock()
        self._journal: CommunicationJournal | None = None
        self._journals: list[CommunicationJournal] = []
        self._closed = False
        self._service = None
        self._uses_managed_io = False

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise BaseExceptionGroup("execution and network cleanup failed", [error, cleanup])
            raise

    @staticmethod
    def _identity(token: str | None) -> str:
        return hashlib.sha256(token.encode()).hexdigest()[:16] if token else "public"

    def _sanitize(self, value):
        with self._lock:
            secrets = tuple(self._tokens)
        def clean(item):
            if isinstance(item, str):
                for secret in secrets:
                    item = item.replace(secret, "<redacted>")
                return item
            if isinstance(item, dict):
                return {key: clean(value) for key, value in redact(item).items()}
            if isinstance(item, (tuple, list)):
                return [clean(value) for value in item]
            return item
        return clean(value)

    def bind_recorder(self, recorder) -> None:
        if self._journal is not None:
            self._journal.close()
            self._journal = None
        run_id = getattr(recorder, "run_id", None)
        bind = getattr(recorder, "bind_before_publish", None)
        if self.output_root is not None and isinstance(run_id, str) and callable(bind):
            journal = CommunicationJournal(self.output_root / "communication_records" / (run_id + ".jsonl"),
                                           sanitize=self._sanitize)
            self._journal = journal
            self._journals.append(journal)
            journal.record({"kind": "configuration", "run_id": run_id,
                            "mode": "managed", "transport": "httpx",
                            "notifications": "sse", "http2": False,
                            "notifications_http2": True,
                            "action_reserves": {"response_ms": self.action_reserves.response_ms,
                                                "discard_ms": self.action_reserves.discard_ms}})
        if callable(bind):
            bind(self.flush)

    def bind_identity(self, token: str, user_id: str, player_name: str | None = None) -> None:
        with self._lock:
            self._labels[self._identity(token)] = {"user_id": user_id,
                                                  **({"player_name": player_name} if player_name else {})}

    def record(self, row: dict[str, Any], *, identity: str | None = None) -> None:
        if self._journal is None:
            return
        with self._lock:
            labels = dict(self._labels.get(identity, {}))
        try:
            self._journal.record({**labels, **row})
        except Exception as error:
            from .recovery import RecorderCallbackFailure
            raise RecorderCallbackFailure(error) from None

    def client(self, factory=ApiClient, *, token: str | None = None, **options):
        if self._closed:
            raise RuntimeError("run network is closed")
        identity = self._identity(token)
        with self._lock:
            if token:
                self._tokens.add(token)
            real_client = isinstance(factory, type) and issubclass(factory, ApiClient)
            if real_client:
                if self._runtime is None:
                    self._runtime = HttpxRuntime()
                if identity not in self._transports:
                    self._transports[identity] = HttpxTransport(runtime=self._runtime,
                        insecure=self.server.insecure)
                transport = self._transports[identity]
                options.update(transport=transport, ssl_context=transport.context)
                options.setdefault("observer", lambda row: self.record(row, identity=identity))
            options.setdefault("insecure", self.server.insecure)
            options.setdefault("timeout", self.server.timeout_seconds)
            client = factory(self.server.url, token=token, **options)
            if real_client:
                self._uses_managed_io = True
                client.managed_network = self
                client.managed_token = token
                client.request_router = lambda method, path, body, **kw: self.service.request(
                    token, method, path, body, budget_identity=self._state_identity(token), **kw)
            return client

    @property
    def uses_managed_io(self) -> bool:
        """Whether created clients actually route through the I/O process.

        Injected JSON clients keep their supplied request implementation and
        do not need a real I/O process for fixture requests.
        """
        return self._uses_managed_io

    def _state_identity(self, token):
        if token is not None:
            return token
        # Race/match terminal probes may use a public view, but still consume
        # their participant's single state budget.
        with self._lock:
            identified = [candidate for candidate in self._tokens if self._identity(candidate) in self._labels]
            return identified[0] if len(identified) == 1 else None

    @property
    def service(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("run network is closed")
            if self._service is None:
                from .io_process import IoService
                self._service = IoService(self.server, self.record, action_reserves=self.action_reserves)
            return self._service

    def flush(self) -> None:
        if self._journal is not None:
            self._journal.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures = []
        for resource in (self._service, self._runtime, *self._journals):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    failures.append(error)
        if failures:
            raise BaseExceptionGroup("network cleanup failed", failures)
