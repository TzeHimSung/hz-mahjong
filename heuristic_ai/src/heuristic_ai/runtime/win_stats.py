"""One console representation for cumulative match small-round statistics."""
from __future__ import annotations

from collections.abc import Mapping

from heuristic_ai.artifacts.models import MatchWinStats


def format_match_win_stats(stats: MatchWinStats | Mapping[str, object]) -> str:
    if not isinstance(stats, MatchWinStats):
        stats = MatchWinStats.from_dict(stats)
    if not stats.statistics_complete:
        return (
            f"总胜率=数据不完整（已确认胜利局数={stats.won_rounds}，"
            f"总局数={stats.total_rounds}，赛果缺失局数={stats.unknown_rounds}）"
        )
    percentage = "--" if stats.win_rate is None else f"{stats.win_rate:.2%}"
    return f"总胜率={percentage}（胜利局数={stats.won_rounds}，总局数={stats.total_rounds}）"
