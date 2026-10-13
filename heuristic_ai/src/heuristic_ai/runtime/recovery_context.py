"""Bounded failure evidence and action safety state shared by player generations."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from collections.abc import Callable
import math
import re
from typing import Any

from heuristic_ai.game.models import Observation, Phase
from heuristic_ai.game.public_inventory import StaleInventorySnapshot
from heuristic_ai.protocol.client import ApiError


@dataclass(slots=True)
class ActionMemory:
    """Keep only the validated current round; never retain strategy decisions.

    One player owns this object at a time. Coordinators must join the old
    generation before handing it to its replacement.
    """

    round_no: int | None = None
    snapshot_seq: int | None = None
    accepted_keys: set[str] = field(default_factory=set)
    rejected_keys: set[str] = field(default_factory=set)
    accepted_response_windows: set[tuple] = field(default_factory=set)
    accepted_action_windows: set[tuple] = field(default_factory=set)
    uncertain_action_windows: set[tuple] = field(default_factory=set)
    unsubmitted_action_windows: set[tuple] = field(default_factory=set)
    first_observed_at: dict[str, float] = field(default_factory=dict)
    window_first_observed_at: dict[tuple, float] = field(default_factory=dict)
    window_deadlines: dict[tuple, float] = field(default_factory=dict)
    inventory_checkpoint: dict | None = None
    event_progress: dict | None = None
    chain_checkpoint: tuple | None = None
    window_evidence: dict | None = None

    def check_snapshot(self, round_no: int, seq: int) -> None:
        if self.round_no is not None and round_no < self.round_no:
            raise StaleInventorySnapshot("stale_round_snapshot")
        if round_no == self.round_no and self.snapshot_seq is not None and seq < self.snapshot_seq:
            raise StaleInventorySnapshot("stale_snapshot_sequence")

    def accept_snapshot(self, round_no: int, seq: int) -> None:
        self.check_snapshot(round_no, seq)
        if self.round_no != round_no:
            self.accepted_keys.clear()
            self.rejected_keys.clear()
            self.accepted_response_windows.clear()
            self.accepted_action_windows.clear()
            self.uncertain_action_windows.clear()
            self.unsubmitted_action_windows.clear()
            self.first_observed_at.clear()
            self.window_first_observed_at.clear()
            self.window_deadlines.clear()
            self.inventory_checkpoint = None
            self.event_progress = None
            self.chain_checkpoint = None
            self.window_evidence = None
        self.round_no = round_no
        self.snapshot_seq = seq


_MAX_EXCEPTION_NODES = 32
_MAX_EXCEPTION_EDGES = 64
_MAX_EXCEPTION_DEPTH = 8
_MAX_GROUP_CHILDREN = 8
_MAX_EXCEPTION_TEXT = 32768
_TYPE_NAME = type.__dict__["__name__"]
_TYPE_MRO = type.__dict__["__mro__"]


Sanitizer = Callable[[str], str]


def _clean_text(value: str, sanitize: Sanitizer | None) -> str:
    """Redact complete strings before any bounded diagnostic projection."""
    if sanitize is None:
        return value
    result = sanitize(value)
    if type(result) is not str:
        raise TypeError("failure sanitizer must return a plain string")
    return result


def _clip_text(value: str, limit: int, sanitize: Sanitizer | None, *, tail: bool = False) -> str:
    value = _clean_text(value, sanitize)
    return value[-limit:] if tail and limit else value[:limit]


def _safe_type_name(value: Any, sanitize: Sanitizer | None) -> str:
    # Calling type(value).__name__ can dispatch to a custom metaclass.
    try:
        name = _TYPE_NAME.__get__(type(value), type)
    except BaseException:
        name = "unknown"
    return _clip_text(name, 128, sanitize)


def _actual_instance(value: Any, base: type) -> bool:
    # isinstance can consult an overridden __class__; use the actual MRO and
    # identity comparisons so neither object nor metaclass hooks are invoked.
    try:
        return any(cls is base for cls in _TYPE_MRO.__get__(type(value), type))
    except BaseException:
        return False


def _exception_attribute(error: BaseException, name: str, default: Any = None) -> Any:
    """Read only native BaseException slots, bypassing subclass properties."""
    try:
        return BaseException.__dict__[name].__get__(error, BaseException)
    except BaseException:
        return default


def _argument_text(value: Any, sanitize: Sanitizer | None) -> str:
    if type(value) is str:
        value = re.sub(r"HTTP ([1-5][0-9]{2}):.*", r"HTTP \1", value, flags=re.DOTALL)
        return _clip_text(value, 2048, sanitize)
    if value is None or type(value) is bool or type(value) is float:
        return str(value)
    if type(value) is int:
        return str(value) if value.bit_length() <= 256 else "<large integer>"
    # Exception args often contain arbitrary objects. Their str/repr methods
    # can throw process-control exceptions or reveal whole private payloads.
    return f"<{_safe_type_name(value, sanitize)} argument>"


def _base_message(error: BaseException, sanitize: Sanitizer | None) -> str:
    args = _exception_attribute(error, "args", ())
    if type(args) is not tuple or not args:
        return ""
    if _actual_instance(error, BaseExceptionGroup):
        message = _argument_text(args[0], sanitize)
    elif len(args) == 1:
        message = _argument_text(args[0], sanitize)
    else:
        message = ", ".join(_argument_text(arg, sanitize) for arg in args[:4])
        if len(args) > 4:
            message += ", <additional arguments omitted>"
    # A wrapper may already have interpolated an ApiError's body into a plain
    # message before reaching us. Suppress that HTTP suffix as well.
    return _clip_text(re.sub(r"HTTP ([1-5][0-9]{2}):.*", r"HTTP \1", message, flags=re.DOTALL), 2048, sanitize)


def _frame_locations(error: BaseException, sanitize: Sanitizer | None) -> str:
    frames: deque[tuple[str, int, str] | str] = deque(maxlen=32)
    current = _exception_attribute(error, "__traceback__")
    # Read code locations directly. traceback extraction can load source text;
    # diagnostic recording needs neither source loaders nor frame locals.
    try:
        for _ in range(128):
            if current is None:
                break
            code = current.tb_frame.f_code
            filename, name = code.co_filename, code.co_name
            filename = filename if type(filename) is str else "<unknown>"
            name = name if type(name) is str else "<unknown>"
            frames.append((filename, current.tb_lineno, name))
            current = current.tb_next
        if current is not None:
            frames.append("  <additional traceback frames omitted>")
    except BaseException:
        frames.append("  <traceback unavailable>")
    lines = [frame if type(frame) is str else
             f'  File "{_clip_text(frame[0], 512, sanitize)}", line {frame[1]}, in {_clip_text(frame[2], 128, sanitize)}'
             for frame in frames]
    return _clip_text("\n".join(lines), 8192, sanitize, tail=True)


def _exception_item(error: BaseException, text_budget: list[int], sanitize: Sanitizer | None) -> dict[str, Any]:
    item: dict[str, Any] = {"exception_type": _safe_type_name(error, sanitize)}
    if _actual_instance(error, ApiError):
        raw = _exception_attribute(error, "__dict__", {})
        try:
            status, code = dict.get(raw, "status"), dict.get(raw, "code")
        except BaseException:
            status, code = None, None
        status = status if type(status) is int and 100 <= status <= 599 else None
        code = code if type(code) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) else None
        code = _clean_text(code, sanitize) if code is not None else None
        message = f"HTTP {status}" if status is not None else "HTTP error"
        if code is not None:
            message += f": {code}"
        item.update(http_status=status, code=code)
    else:
        message = _base_message(error, sanitize)
    for key, value in (("message", message), ("traceback", _frame_locations(error, sanitize))):
        value = _clean_text(value, sanitize)
        item[key] = value[:text_budget[0]]
        text_budget[0] -= len(item[key])
        if len(item[key]) < len(value):
            item["text_truncated"] = True
    return item


def safe_exception(error: BaseException, *, sanitize: Sanitizer | None = None) -> dict[str, Any]:
    """Project a bounded exception graph without calling exception renderers.

    The existing root fields and flat exception_chain list remain compatible.
    Entries additionally name their parent/node IDs, so group children, causes,
    suppressed contexts and repeated references retain their relationships.
    """
    budget = [_MAX_EXCEPTION_TEXT]
    result = {"node_id": 0, **_exception_item(error, budget, sanitize)}
    chain: list[dict[str, Any]] = []
    seen = {id(error): 0}
    pending = deque([(error, result, 0)])
    while pending:
        current, parent, depth = pending.popleft()
        edges: list[tuple[BaseException, dict[str, Any]]] = []
        cause = _exception_attribute(current, "__cause__")
        context = _exception_attribute(current, "__context__")
        if cause is not None:
            edges.append((cause, {"relationship": "cause"}))
        if context is not None:
            edges.append((context, {"relationship": "context", "suppressed":
                                    _exception_attribute(current, "__suppress_context__", False) is True}))
        if _actual_instance(current, BaseExceptionGroup):
            try:
                children = BaseExceptionGroup.__dict__["exceptions"].__get__(current, BaseExceptionGroup)
            except BaseException:
                children = ()
                parent["group_children_unavailable"] = True
            for index, child in enumerate(children[:_MAX_GROUP_CHILDREN]):
                edges.append((child, {"relationship": "group_child", "child_index": index}))
            if len(children) > _MAX_GROUP_CHILDREN:
                parent["omitted_group_children"] = len(children) - _MAX_GROUP_CHILDREN
                result["exception_chain_truncated"] = True
        for child, metadata in edges:
            if len(chain) >= _MAX_EXCEPTION_EDGES:
                result["exception_chain_truncated"] = True
                pending.clear()
                break
            entry = {"parent_id": parent["node_id"], **metadata}
            identity = id(child)
            if identity in seen:
                entry["reference_id"] = seen[identity]
            elif depth >= _MAX_EXCEPTION_DEPTH or len(seen) >= _MAX_EXCEPTION_NODES:
                entry.update(exception_type=_safe_type_name(child, sanitize),
                             truncated="depth_limit" if depth >= _MAX_EXCEPTION_DEPTH else "node_limit")
                result["exception_chain_truncated"] = True
            else:
                entry.update(node_id=len(seen), **_exception_item(child, budget, sanitize))
                seen[identity] = entry["node_id"]
                pending.append((child, entry, depth + 1))
            chain.append(entry)
    result["exception_chain"] = chain
    return result


def _scalar(value: Any, sanitize: Sanitizer | None) -> Any:
    if type(value) is float and not math.isfinite(value):
        return "<non_finite>"
    if value is None or type(value) is bool or type(value) is int or type(value) is float:
        return value
    if _actual_instance(value, str):
        return _clip_text(str.__str__(value), 128, sanitize)
    return "<invalid_type>"


def _sequence(value: Any, limit: int, sanitize: Sanitizer | None) -> Any:
    if not isinstance(value, (list, tuple)):
        return _scalar(value, sanitize)
    return [_scalar(item, sanitize) for item in value[:limit]]


def snapshot_evidence(snapshot: Any, *, sanitize: Sanitizer | None = None) -> dict[str, Any]:
    """Whitelist a player's own hand and the protocol's public board only."""
    if not isinstance(snapshot, dict):
        return {"invalid_snapshot_type": _safe_type_name(snapshot, sanitize)}
    result = {key: _scalar(snapshot[key], sanitize) for key in (
        "round_no", "seat", "phase", "dealer", "turn", "wall_remaining",
        "drawn_tile", "last_discard", "waited_seat",
    ) if key in snapshot}
    for key, limit in (("my_hand", 32), ("scores", 4), ("responding_seats", 4)):
        if key in snapshot:
            result[key] = _sequence(snapshot[key], limit, sanitize)
    if "discards" in snapshot:
        discards = snapshot["discards"]
        result["discards"] = ([_sequence(river, 144, sanitize) for river in discards[:4]]
                              if isinstance(discards, (list, tuple)) else _scalar(discards, sanitize))
    if "melds" in snapshot:
        melds = snapshot["melds"]
        result["melds"] = ([
            [{"kind": _scalar(meld.get("kind"), sanitize), "tiles": _sequence(meld.get("tiles"), 4, sanitize)}
             if isinstance(meld, dict) else _scalar(meld, sanitize) for meld in seat_melds[:4]]
            if isinstance(seat_melds, (list, tuple)) else _scalar(seat_melds, sanitize)
            for seat_melds in melds[:4]
        ] if isinstance(melds, (list, tuple)) else _scalar(melds, sanitize))
    god = snapshot.get("god")
    if isinstance(god, dict):
        result["god"] = {key: _scalar(god[key], sanitize) for key in ("baotou", "chain_count", "catch_play") if key in god}
    return result


def failure_context(
    error: BaseException,
    *,
    game_id: str | None = None,
    generation: int | None = None,
    request_seq: int | None = None,
    stage: str | None = None,
    response: Any = None,
    observation: Observation | None = None,
    sanitize: Sanitizer | None = None,
) -> dict[str, Any]:
    """Build a bounded, token-redactable journal entry for a player failure."""
    result: dict[str, Any] = {
        "game_id": game_id, "request_seq": request_seq, "stage": stage,
        **safe_exception(error, sanitize=sanitize),
    }
    if generation is not None:
        result["generation"] = generation
    snapshot = response.get("snapshot") if isinstance(response, dict) else None
    if isinstance(response, dict):
        result["response"] = {key: _scalar(response[key], sanitize) for key in ("seq", "pending", "gap", "finished") if key in response}
        if "snapshot" in response:
            result["snapshot"] = snapshot_evidence(snapshot, sanitize=sanitize)
    if observation is not None:
        result.update(round_no=observation.round_no, seat=observation.seat, phase=observation.phase.value)
    if isinstance(snapshot, dict):
        hand, melds, seat = snapshot.get("my_hand"), snapshot.get("melds"), snapshot.get("seat")
        if isinstance(hand, (list, tuple)):
            result["hand_actual_count"] = len(hand)
        if type(seat) is int and 0 <= seat < 4 and isinstance(melds, (list, tuple)) and len(melds) > seat and isinstance(melds[seat], (list, tuple)):
            phase, responding = snapshot.get("phase"), snapshot.get("responding_seats")
            if phase == Phase.DRAW.value and snapshot.get("turn") == seat:
                result["hand_expected_count"] = 3 * (4 - len(melds[seat])) + 2
            elif phase in (Phase.RESPONSE_PENG.value, Phase.RESPONSE_CHI.value) and isinstance(responding, (list, tuple)) and seat in responding:
                result["hand_expected_count"] = 3 * (4 - len(melds[seat])) + 1
    return result
