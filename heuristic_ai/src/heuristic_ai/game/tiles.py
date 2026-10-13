"""Canonical tile codes and public 34-face ordering."""
from __future__ import annotations

import re


_SUITED_TILE = re.compile(r"^[1-9][wbt]$")


_HONOR_TILES = frozenset("东南西北中发白")


_TILE_CODES = _HONOR_TILES | frozenset(f"{rank}{suit}" for rank in range(1, 10) for suit in "wbt")


def is_tile_code(value: object) -> bool:
    if type(value) is str:
        return value in _TILE_CODES
    return isinstance(value, str) and (
        _SUITED_TILE.fullmatch(value) is not None or value in _HONOR_TILES
    )


TILE_ORDER = (
    *(f"{rank}w" for rank in range(1, 10)),
    *(f"{rank}b" for rank in range(1, 10)),
    *(f"{rank}t" for rank in range(1, 10)),
    "东",
    "南",
    "西",
    "北",
    "中",
    "发",
    "白",
)


TILE_INDEX = {tile: index for index, tile in enumerate(TILE_ORDER)}
