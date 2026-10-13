from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, TypedDict

from heuristic_ai.game.hand_solver import RouteState
from heuristic_ai.game.models import Observation
from heuristic_ai.game.tiles import TILE_INDEX


class RouteImprovementDetail(TypedDict):
    route: str
    distance_before: int
    distance_after: int
    distance_improvement: int
    fan_option_value: None


class LiveTileDetail(TypedDict):
    tile: str
    visible_remaining: int
    weight: int
    improvements: tuple[RouteImprovementDetail, ...]


@dataclass(frozen=True, slots=True)
class RouteFrontier:
    minimum_distance: int
    routes: tuple[RouteState, ...]
    live_tiles: tuple[LiveTileDetail, ...]
    live_tile_weight: int

    @property
    def route_count(self) -> int:
        return len(self.routes)


def route_details(route: RouteState) -> dict[str, object]:
    return {
        "route": route.kind.value,
        "distance": route.distance,
        "components": tuple(
            {
                "kind": component.kind,
                "target_tiles": component.target_tiles,
                "natural_tiles": component.natural_tiles,
                "wildcard_tiles": component.wildcard_tiles,
                "missing_tiles": component.missing_tiles,
            }
            for component in route.components
        ),
        "wildcard_assignments": tuple(
            {
                "wildcard_index": assignment.wildcard_index,
                "target_tile": assignment.target_tile,
                "component_index": assignment.component_index,
            }
            for assignment in route.wildcard_assignments
        ),
        "open_meld_compatibility": route.open_meld_compatibility,
        "config_compatibility": route.config_compatibility,
        "improving_tiles": route.improving_tiles,
    }


def known_counts(observation: Observation) -> Counter[str]:
    known = Counter(observation.hand)
    for seat_discards in observation.discards:
        known.update(seat_discards)
    for seat_melds in observation.melds:
        for meld in seat_melds:
            known.update(meld.tiles)
    return known


def route_frontier(
    routes: Iterable[RouteState],
    counts: Mapping[str, int],
    *,
    draw_available: bool,
) -> RouteFrontier:
    routes = tuple(routes)
    if not routes:
        raise ValueError("routes must not be empty")
    minimum_distance = min(route.distance for route in routes)
    minimum_routes = tuple(
        route for route in routes if route.distance == minimum_distance
    )
    live_faces = {
        tile
        for route in minimum_routes
        for tile in route.improving_tiles
        if max(0, 4 - counts.get(tile, 0)) > 0
    }
    live_tiles = tuple(
        LiveTileDetail(
            tile=tile,
            visible_remaining=visible_remaining,
            weight=visible_remaining if draw_available else 0,
            improvements=tuple(
                RouteImprovementDetail(
                    route=route.kind.value,
                    distance_before=route.distance,
                    distance_after=route.distance - 1,
                    distance_improvement=1,
                    fan_option_value=None,
                )
                for route in minimum_routes
                if tile in route.improving_tiles
            ),
        )
        for tile in sorted(live_faces, key=TILE_INDEX.__getitem__)
        for visible_remaining in (max(0, 4 - counts.get(tile, 0)),)
    )
    return RouteFrontier(
        minimum_distance,
        minimum_routes,
        live_tiles,
        sum(item["weight"] for item in live_tiles),
    )
