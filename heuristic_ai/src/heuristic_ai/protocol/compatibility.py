from __future__ import annotations

import hashlib
import json
from typing import Any

from ..game.models import RuleConfig


def guide_compatibility_profile(mode: str) -> dict[str, Any]:
    """Describe room scope; API guide versions never restrict admission."""
    if mode not in {"train", "race", "match"}:
        raise ValueError("guide compatibility mode must be train, race or match")
    return {
        "mode": mode,
        "required_room_kind": {"train": "test", "race": "", "match": "auto"}[mode],
    }


def required_int(
    config: dict[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"tournament config {key} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(
            f"tournament config {key} must be an integer between {minimum} and {maximum}"
        )
    return value


def round_ended_metadata(
    event: dict[str, Any], *, expected_round_no: int | None = None,
) -> tuple[int | None, int | None]:
    """Read the additive v19 settlement fields, retaining legacy replay support."""
    data = event.get("data")
    if not isinstance(data, dict):
        raise ValueError("round_ended data must be an object")
    numbers = []
    for source in (event, data):
        if "round_no" in source:
            number = source["round_no"]
            if type(number) is not int or number < 1:
                raise ValueError("round_ended round_no must be a positive integer")
            numbers.append(number)
    if len(set(numbers)) > 1:
        raise ValueError("round_ended round_no sources disagree")
    number = numbers[0] if numbers else None
    if number is not None and expected_round_no is not None and number != expected_round_no:
        raise ValueError("round_ended round_no disagrees with its round")
    dealer = data.get("dealer")
    if "dealer" in data and (type(dealer) is not int or dealer not in range(4)):
        raise ValueError("round_ended dealer must be a seat between zero and three")
    return number, dealer


def normalize_rules(config: dict[str, Any]) -> RuleConfig:
    """Validate and normalize one complete tournament rules configuration."""

    you_cai_bi_kao = config.get("YouCaiBiKao")
    if not isinstance(you_cai_bi_kao, bool):
        raise RuntimeError("tournament config YouCaiBiKao must be a boolean")
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return RuleConfig(
        rounds=required_int(config, "Rounds", minimum=1),
        base_score=required_int(config, "BaseScore", minimum=1),
        you_cai_bi_kao=you_cai_bi_kao,
        discard_timeout_seconds=required_int(
            config,
            "DiscardTimeoutSec",
            minimum=1,
        ),
        peng_timeout_seconds=required_int(config, "PengTimeoutSec", minimum=1),
        chi_timeout_seconds=required_int(config, "ChiTimeoutSec", minimum=1),
        source_hash=hashlib.sha256(canonical).hexdigest(),
    )
