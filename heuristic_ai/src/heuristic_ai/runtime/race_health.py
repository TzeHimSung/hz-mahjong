"""Business-thread progress, independent of the process heartbeat and log sinks."""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


OPERATION_GRACE_SECONDS = 10.0


def evaluate_health(snapshot: dict[str, Any], now: float,
                    stall_timeout_seconds: float) -> dict[str, Any] | None:
    """Evaluate the original business timestamps; receiving a heartbeat changes none.

    Explicit operation deadlines already include their grace. All enclosing
    deadlines remain effective while a nested operation runs. An unbudgeted
    innermost operation uses actual checkpoints and the ordinary stall limit.
    """
    for component, state in snapshot.get("components", {}).items():
        operations = state.get("operations", ())
        expired = [operation for operation in operations
                   if operation["deadline_at"] is not None and now > operation["deadline_at"]]
        top = operations[-1] if operations else None
        if expired:
            operation = min(expired, key=lambda item: item["deadline_at"])
            reason, label, deadline = "operation_timeout", operation["label"], operation["deadline_at"]
        elif (top is None or top["deadline_at"] is None) and now - state["last_progress_at"] > stall_timeout_seconds:
            reason, label, deadline = "stall", state["label"], state["last_progress_at"] + stall_timeout_seconds
        else:
            continue
        return {"reason": reason, "component": component, "label": label,
                "generation": state["generation"], "thread_id": state["thread_id"],
                "last_progress_at": state["last_progress_at"], "deadline_at": deadline}
    return None


@dataclass(eq=False)
class HealthComponent:
    tracker: RaceHealth
    name: str
    generation: int
    thread_id: int | None
    last_progress_at: float
    label: str
    operations: list[dict[str, Any]] = field(default_factory=list)

    def checkpoint(self, label: str) -> None:
        self.tracker._checkpoint(self, label)

    def close(self) -> None:
        self.tracker._close(self)


class RaceHealth:
    """Thread-safe registry updated only by the code actually doing the work.

    The sink must only replace an in-memory IPC snapshot, never log or perform
    I/O. Components are registered before executor submission so a queued task
    cannot disappear from monitoring. Instance identity rejects late updates
    from retired worker generations.
    """

    def __init__(self, sink: Callable[[dict[str, Any]], None], *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._sink = sink
        self.clock = clock
        self._lock = threading.RLock()
        self._local = threading.local()
        self._components: dict[str, HealthComponent] = {}
        self._revision = 0
        self._operation_id = 0

    def _publish(self) -> None:
        # Caller owns the registry lock; the nonblocking sink preserves order.
        self._revision += 1
        self._sink({"revision": self._revision, "captured_at": self.clock(),
                    "components": {name: {
                        "generation": item.generation, "thread_id": item.thread_id,
                        "last_progress_at": item.last_progress_at, "label": item.label,
                        "operations": [dict(operation) for operation in item.operations],
                    } for name, item in self._components.items()}})

    def register(self, name: str, generation: int = 1) -> HealthComponent:
        with self._lock:
            existing = self._components.get(name)
            if existing is not None:
                raise RuntimeError("health component is still active")
            component = HealthComponent(self, name, generation, None, self.clock(), "queued")
            self._components[name] = component
            self._publish()
            return component

    @contextmanager
    def bind(self, component: HealthComponent) -> Iterator[HealthComponent]:
        previous = getattr(self._local, "component", None)
        self._local.component = component
        with self._lock:
            component.thread_id = threading.get_ident()
        component.checkpoint("started")
        try:
            yield component
        finally:
            component.close()
            self._local.component = previous

    @contextmanager
    def component(self, name: str, generation: int = 1) -> Iterator[HealthComponent]:
        with self.bind(self.register(name, generation)) as component:
            yield component

    def _checkpoint(self, component: HealthComponent, label: str) -> None:
        with self._lock:
            if self._components.get(component.name) is not component:
                return
            component.last_progress_at = self.clock()
            component.label = label
            self._publish()

    def checkpoint(self, label: str) -> None:
        component = getattr(self._local, "component", None)
        if component is not None:
            component.checkpoint(label)

    def _close(self, component: HealthComponent) -> None:
        with self._lock:
            if self._components.get(component.name) is component:
                del self._components[component.name]
                self._publish()

    def is_active(self, component: HealthComponent | None) -> bool:
        with self._lock:
            return component is not None and self._components.get(component.name) is component

    @contextmanager
    def operation(self, label: str, budget_seconds: float | None = None) -> Iterator[None]:
        component = getattr(self._local, "component", None)
        if component is None:
            yield
            return
        if budget_seconds is not None and (not math.isfinite(budget_seconds) or budget_seconds < 0):
            raise ValueError("health operation budget must be finite and nonnegative")
        with self._lock:
            self._operation_id += 1
            operation_id = self._operation_id
            now = self.clock()
            operation = {"id": operation_id, "label": label, "started_at": now,
                         "deadline_at": None if budget_seconds is None else now + budget_seconds + OPERATION_GRACE_SECONDS}
            active = self._components.get(component.name) is component
            if active:
                component.operations.append(operation)
                component.last_progress_at = now
                component.label = label
                self._publish()
        try:
            yield
        finally:
            with self._lock:
                if self._components.get(component.name) is component and active:
                    component.operations = [item for item in component.operations if item["id"] != operation_id]
                    component.last_progress_at = self.clock()
                    component.label = component.operations[-1]["label"] if component.operations else label + ":complete"
                    self._publish()
