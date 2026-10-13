from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml

from .models import (
    AppConfig,
    CompetitionConfig,
    MatchCompetitionConfig,
    MatchConfig,
    MatchmakingConfig,
    PlayerConfig,
    RaceConfig,
    RaceParticipantConfig,
    RaceSupervisionConfig,
    ServerConfig,
    TrainConfig,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")
_PLACEHOLDERS = ("REPLACE_ME", "YOUR_ROOM", "PARTICIPANT_TOKEN")


class ConfigError(ValueError):
    """Raised when a YAML launch configuration is unsafe or incomplete."""


class _StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _StrictSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError:
            raise ConfigError("YAML mapping keys must be hashable") from None
        if duplicate:
            raise ConfigError("duplicate YAML key is not allowed")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _object(
    value: Any,
    path: str,
    *,
    keys: set[str],
    optional_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} must be an object")
    unknown = sorted(set(value) - keys - (optional_keys or set()))
    if unknown:
        raise ConfigError(f"{path} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(keys - set(value))
    if missing:
        raise ConfigError(f"{path} is missing fields: {', '.join(missing)}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value.strip()


def _mode(value: Any) -> Literal["train", "race", "match"]:
    if not isinstance(value, str) or value not in {"train", "race", "match"}:
        raise ConfigError("mode must be 'train', 'race', or 'match'")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be finite")
    if result < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _reject_placeholder(value: str, path: str) -> None:
    upper = value.upper()
    if any(marker in upper for marker in _PLACEHOLDERS):
        raise ConfigError(f"{path} still contains a placeholder")


def _parse_server(value: Any, path: str) -> ServerConfig:
    if isinstance(value, dict) and "communication" in value:
        raise ConfigError(f"{path}.communication has been removed; delete this block. "
                          "I/O is fixed to HTTPX with HTTP/2 negotiation, SSE and a 300 ms action reserve.")
    source = _object(
        value,
        path,
        keys={"url", "insecure", "timeout_seconds"},
    )
    url = _string(source["url"], f"{path}.url")
    _reject_placeholder(url, f"{path}.url")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"{path}.url must be an absolute HTTP(S) URL")
    return ServerConfig(
        url=url.rstrip("/"),
        insecure=_boolean(source["insecure"], f"{path}.insecure"),
        timeout_seconds=_number(
            source["timeout_seconds"],
            f"{path}.timeout_seconds",
            minimum=0.001,
        ),
    )


def _parse_competition(
    value: Any,
    config_dir: Path,
    path: str,
) -> CompetitionConfig:
    source = _object(
        value,
        path,
        keys={
            "room_id",
            "seed",
            "poll_interval_seconds",
            "decision_budget_ms",
            "output_dir",
        },
    )
    room_id = _string(source["room_id"], f"{path}.room_id")
    _reject_placeholder(room_id, f"{path}.room_id")
    if _IDENTIFIER.fullmatch(room_id) is None:
        raise ConfigError(f"{path}.room_id contains unsafe characters")
    output_value = _string(source["output_dir"], f"{path}.output_dir")
    output_dir = Path(output_value)
    if not output_dir.is_absolute():
        output_dir = config_dir / output_dir
    return CompetitionConfig(
        room_id=room_id,
        seed=_integer(source["seed"], f"{path}.seed"),
        poll_interval_seconds=_number(
            source["poll_interval_seconds"],
            f"{path}.poll_interval_seconds",
            minimum=0.0,
        ),
        decision_budget_ms=_integer(
            source["decision_budget_ms"],
            f"{path}.decision_budget_ms",
            minimum=1,
        ),
        output_dir=output_dir.resolve(),
    )




def _parse_players(value: Any, path: str) -> tuple[PlayerConfig, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ConfigError(f"{path} must contain exactly four players")
    players: list[PlayerConfig] = []
    for index, raw_player in enumerate(value):
        source = _object(
            raw_player,
            f"{path}[{index}]",
            keys={"name", "token"},
        )
        name = _string(source["name"], f"{path}[{index}].name")
        token = _string(source["token"], f"{path}[{index}].token")
        _reject_placeholder(token, f"{path}[{index}].token")
        players.append(PlayerConfig(name, token))

    if len({player.name for player in players}) != 4:
        raise ConfigError("player names must be unique")
    if len({player.token for player in players}) != 4:
        raise ConfigError("player tokens must be unique")
    return tuple(players)


def _parse_train(value: Any, config_dir: Path) -> TrainConfig:
    keys = {"server", "competition", "players"}
    if isinstance(value, dict) and "save_to_official" in value:
        keys.add("save_to_official")
    source = _object(value, "train", keys=keys)
    competition = _parse_competition(
        source["competition"], config_dir, "train.competition"
    )
    return TrainConfig(
        server=_parse_server(source["server"], "train.server"),
        competition=competition,
        players=_parse_players(source["players"], "train.players"),
        save_to_official=_boolean(source.get("save_to_official", False), "train.save_to_official"),
    )


def _parse_race_supervision(value: Any) -> RaceSupervisionConfig:
    path = "race.supervision"
    source = _object(value, path, keys=set(), optional_keys={
        "enabled", "heartbeat_timeout_seconds", "stall_timeout_seconds",
    })
    timeouts = {}
    for key, default in (("heartbeat_timeout_seconds", 30.0), ("stall_timeout_seconds", 180.0)):
        timeout = _number(source.get(key, default), f"{path}.{key}", minimum=0.0)
        if timeout <= 0:
            raise ConfigError(f"{path}.{key} must be greater than zero")
        timeouts[key] = timeout
    return RaceSupervisionConfig(
        enabled=_boolean(source.get("enabled", True), f"{path}.enabled"), **timeouts,
    )


def _parse_race(value: Any, config_dir: Path) -> RaceConfig:
    source = _object(value, "race", keys={"server", "competition", "participant"},
                     optional_keys={"supervision"})
    competition = _parse_competition(
        source["competition"], config_dir, "race.competition"
    )
    participant_source = _object(
        source["participant"],
        "race.participant",
        keys={"token"},
    )
    token = _string(participant_source["token"], "race.participant.token")
    _reject_placeholder(token, "race.participant.token")
    return RaceConfig(
        server=_parse_server(source["server"], "race.server"),
        competition=competition,
        supervision=_parse_race_supervision(source.get("supervision", {})),
        participant=RaceParticipantConfig(
            token=token,
        ),
    )


def _parse_match_competition(value: Any, config_dir: Path) -> MatchCompetitionConfig:
    path = "match.competition"
    source = _object(
        value,
        path,
        keys={"seed", "poll_interval_seconds", "output_dir"},
        optional_keys={"decision_budget_ms"},
    )
    poll_interval = _number(
        source["poll_interval_seconds"], f"{path}.poll_interval_seconds", minimum=0.0
    )
    if poll_interval <= 0:
        raise ConfigError(f"{path}.poll_interval_seconds must be greater than zero")
    output_dir = Path(_string(source["output_dir"], f"{path}.output_dir"))
    if not output_dir.is_absolute():
        output_dir = config_dir / output_dir
    return MatchCompetitionConfig(
        seed=_integer(source["seed"], f"{path}.seed"),
        poll_interval_seconds=poll_interval,
        decision_budget_ms=_integer(
            source.get("decision_budget_ms", 3000), f"{path}.decision_budget_ms", minimum=1
        ),
        output_dir=output_dir.resolve(),
    )


def _parse_matchmaking(value: Any) -> MatchmakingConfig:
    path = "match.matchmaking"
    source = _object(
        value,
        path,
        keys=set(),
        optional_keys={"max_parallel_games", "max_rounds_per_game"},
    )
    max_parallel_games = _integer(
        source.get("max_parallel_games", 10), f"{path}.max_parallel_games", minimum=1
    )
    if max_parallel_games > 16:
        raise ConfigError(f"{path}.max_parallel_games must be at most 16")
    return MatchmakingConfig(
        max_parallel_games=max_parallel_games,
        max_rounds_per_game=_integer(
            source.get("max_rounds_per_game", 8), f"{path}.max_rounds_per_game", minimum=1
        ),
    )


def _parse_match(value: Any, config_dir: Path) -> MatchConfig:
    source = _object(
        value,
        "match",
        keys={"server", "competition"},
        optional_keys={"matchmaking", "continuous"},
    )
    competition = _parse_match_competition(source["competition"], config_dir)
    return MatchConfig(
        server=_parse_server(source["server"], "match.server"),
        competition=competition,
        matchmaking=_parse_matchmaking(source.get("matchmaking", {})),
        continuous=_boolean(source.get("continuous", False), "match.continuous"),
    )


def load_config(path: Path) -> AppConfig:
    """Load one strict, secret-bearing YAML launch configuration."""

    try:
        source = yaml.load(
            path.read_text(encoding="utf-8-sig"),
            Loader=_StrictSafeLoader,
        )
    except yaml.YAMLError:
        raise ConfigError("invalid YAML configuration") from None
    if isinstance(source, dict) and source.get("schema_version") in (1, 2):
        raise ConfigError(
            "config schema_version 1/2 is obsolete; migrate to schema_version 3 and remove strategy blocks"
        )
    root_keys = {"schema_version", "mode", "train", "race"}
    if isinstance(source, dict) and "global_token" in source:
        root_keys.add("global_token")
    if isinstance(source, dict) and "match" in source:
        root_keys.add("match")
    root = _object(
        source,
        "config",
        keys=root_keys,
    )
    schema_version = _integer(root["schema_version"], "schema_version")
    if schema_version != 3:
        raise ConfigError("unsupported config schema_version; expected schema_version 3")
    global_token = None
    if "global_token" in root:
        global_token = _string(root["global_token"], "global_token")
        _reject_placeholder(global_token, "global_token")
    mode = _mode(root["mode"])
    train = _parse_train(root["train"], path.parent)
    race = _parse_race(root["race"], path.parent)
    match = _parse_match(root["match"], path.parent) if "match" in root else None
    if mode == "match":
        if match is None:
            raise ConfigError("mode 'match' requires a match configuration")
        if global_token is None:
            raise ConfigError("mode 'match' requires global_token")
    return AppConfig(
        schema_version=schema_version,
        mode=mode,
        train=train,
        race=race,
        global_token=global_token,
        match=match,
    )
