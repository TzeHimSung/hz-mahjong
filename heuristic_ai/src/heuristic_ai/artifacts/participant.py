"""Shared participant-only decision recording and safe error descriptors."""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from heuristic_ai.artifacts.models import DecisionTrace
from heuristic_ai.artifacts.recorder import RecorderBase
from heuristic_ai.protocol.client import ApiError

_TRUSTED_EXCEPTION_TYPES = frozenset({OSError, RuntimeError})
_TRUSTED_GROUP_TYPES = frozenset({BaseExceptionGroup, ExceptionGroup})
_SAFE_API_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class ParticipantRunRecorder(RecorderBase):
    def _safe_exception(
        self,
        error: BaseException,
        seen: set[int] | None = None,
    ) -> BaseException:
        try:
            return self._safe_exception_node(error, seen)
        except BaseException:
            return RuntimeError("exception graph inspection failed")

    def _safe_exception_node(
        self,
        error: BaseException,
        seen: set[int] | None,
    ) -> BaseException:
        if seen is None:
            seen = set()
        if id(error) in seen:
            return RuntimeError("cyclic exception reference redacted")
        seen.add(id(error))

        error_type = type(error)
        if error_type not in _TRUSTED_EXCEPTION_TYPES | _TRUSTED_GROUP_TYPES:
            try:
                type_name = type.__getattribute__(error_type, "__name__")
                safe_type_name = (
                    self._serializer.sanitize(type_name)
                    if type(type_name) is str
                    else "untrusted_exception"
                )
            except BaseException:
                safe_type_name = "untrusted_exception"
            return RuntimeError(
                f"{safe_type_name}: untrusted exception details redacted"
            )

        if error_type in _TRUSTED_GROUP_TYPES:
            message = BaseExceptionGroup.message.__get__(error, error_type)
            safe_message = (
                self._serializer.sanitize(message)
                if type(message) is str
                else "exception group message redacted"
            )
            exceptions = BaseExceptionGroup.exceptions.__get__(error, error_type)
            safe_nested = tuple(
                self._safe_exception(nested, seen) for nested in exceptions
            )
            detached: BaseException = BaseExceptionGroup(
                safe_message,
                safe_nested,
            )
        else:
            arguments = BaseException.args.__get__(error, error_type)
            if type(arguments) is not tuple or not arguments:
                safe_message = ""
            elif len(arguments) == 1 and type(arguments[0]) is str:
                safe_message = self._serializer.sanitize(arguments[0])
            else:
                safe_message = "exception arguments redacted"
            detached = error_type(safe_message)

        cause = BaseException.__cause__.__get__(error, error_type)
        context = BaseException.__context__.__get__(error, error_type)
        suppress_context = BaseException.__suppress_context__.__get__(
            error,
            error_type,
        )
        if cause is not None:
            BaseException.__cause__.__set__(
                detached,
                self._safe_exception(cause, seen),
            )
        if context is not None:
            BaseException.__context__.__set__(
                detached,
                self._safe_exception(context, seen),
            )
        BaseException.__suppress_context__.__set__(
            detached,
            suppress_context if type(suppress_context) is bool else True,
        )
        return detached

    def _error_descriptor(self, error: BaseException) -> dict[str, Any]:
        if type(error) is not ApiError:
            label = {
                RuntimeError: "RuntimeError",
                OSError: "OSError",
                ValueError: "ValueError",
            }.get(type(error), "RuntimeError")
            return {"type": label}

        descriptor: dict[str, Any] = {"type": "ApiError"}
        try:
            status = object.__getattribute__(error, "status")
        except BaseException:
            status = None
        if type(status) is int and status in range(100, 600):
            descriptor["http_status"] = status
        try:
            code = object.__getattribute__(error, "code")
        except BaseException:
            code = None
        if type(code) is str and _SAFE_API_CODE.fullmatch(code) is not None:
            safe_code = self._serializer.sanitize(code)
            if safe_code == code:
                descriptor["code"] = code
        return descriptor

    def _trace_json(self, trace: DecisionTrace) -> dict[str, Any]:
        return self._serializer.trace_json(trace)

    def _safe_storage_error(self, error: BaseException) -> BaseException:
        return self._safe_exception(error)

    def record_guide_version(self, version: int | None) -> None:
        self._control("record_guide_version", version=version)

    def identify_participant(self, user_id: str) -> None:
        self._control("identify_participant", user_id=user_id)

    def record_lifecycle(self, tournament_status: str, *, stage_no: int | None = None,
                         stage_name: str | None = None, stage_status: str | None = None,
                         stage_crashed: bool | None = None, qualified: bool | None = None,
                         qualify_role: str | None = None) -> None:
        self._control("record_lifecycle", tournament_status=tournament_status, stage_no=stage_no,
                      stage_name=stage_name, stage_status=stage_status, stage_crashed=stage_crashed,
                      qualified=qualified, qualify_role=qualify_role)

    def register_game(self, game_id: str, *, stage_no: int | None = None, stage_name: str | None = None) -> None:
        self._control("register_game", game_id=game_id, stage_no=stage_no, stage_name=stage_name)

    def identify_player(self, game_id: str, player_name: str, user_id: str, seat: int) -> None:
        self._control("identify_player", game_id=game_id, player_name=player_name, user_id=user_id, seat=seat)

    def mark_game_completed(self, game_id: str, final_scores: Sequence[int]) -> None:
        self._control("mark_game_completed", game_id=game_id, final_scores=final_scores)

    def mark_game_aborted(self, game_id: str, reason: str) -> None:
        self._control("mark_game_aborted", game_id=game_id, reason=reason)

    def mark_degraded(self, game_id: str, player_name: str, reason: str) -> None:
        self._control("mark_degraded", game_id=game_id, player_name=player_name, reason=reason)
