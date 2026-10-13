from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .models import Action, CandidateAction, Observation


def decision_key(
    game_id: str,
    observation: Observation,
    candidates: tuple[CandidateAction, ...],
) -> str:
    observation_fields = asdict(observation)
    if observation.public_inventory is None:
        # Unannotated views preserve the canonical identity used by 000-003.
        observation_fields.pop("public_inventory")
    if observation.public_actions is None:
        observation_fields.pop('public_actions')
    if observation.public_discard_history is None:
        observation_fields.pop('public_discard_history')
    canonical = json.dumps(
        {
            "game_id": game_id,
            "observation": observation_fields,
            "candidates": [asdict(candidate) for candidate in candidates],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def action_to_json(action: Action) -> dict[str, Any]:
    body: dict[str, Any] = {
        "action": action.kind.value,
        "tile": action.tile or "",
    }
    if action.tiles:
        body["tiles"] = list(action.tiles)
    return body
