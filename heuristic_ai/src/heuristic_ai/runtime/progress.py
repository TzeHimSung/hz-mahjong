"""Console-only progress sourced from recorder memory, never active files."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .win_stats import format_match_win_stats


class RunProgress:
    def __init__(self, recorder: Any, logger: logging.Logger, *, interval: float = 5.0) -> None:
        self.recorder = recorder
        self.logger = logger
        self.interval = interval
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def emit(self, phase: str) -> None:
        progress = self.recorder.progress()
        counts = progress["decisions"]
        win_stats = progress.get("win_stats")
        win_summary = "" if win_stats is None else f" {format_match_win_stats(win_stats)}"
        rounds_label = "archived_rounds" if getattr(self.recorder, "mode", None) == "race" else "rounds"
        self.logger.info(
            "run=%s phase=%s status=%s games=%s " + rounds_label + "=%s decisions=%s "
            "accepted=%s conflict=%s fallback=%s audit_bytes=%s record_p99_ms=%.3f%s",
            progress["run_id"], phase, progress["status"], progress["games"],
            progress["archived_rounds"], counts["total"], counts["accepted"],
            counts["conflict"], counts["fallback"], progress["synced_bytes"],
            progress["record_p99_ms"], win_summary,
        )
        recovering = [item for item in progress.get("recoveries", ())
                      if item.get("state") in {"failure", "restart_scheduled", "restarted"}]
        if recovering:
            self.logger.info("run=%s recovering=%s details=%s", progress["run_id"], len(recovering),
                             [{key: item.get(key) for key in ("game_id", "state", "restart_count", "next_retry_at")}
                              for item in recovering])

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            self.emit("running")

    def __enter__(self) -> RunProgress:
        self.emit("start")
        self.thread = threading.Thread(target=self._run, name="artifact-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.emit("end")
        result = self.recorder._final_result
        if result is not None:
            self.logger.info("run=%s artifacts=%s cleanup=%s output=%s work=%s",
                             result.run_id, result.artifact_status.value,
                             result.cleanup_status.value, result.artifact_dir, result.work_dir)
            if result.artifact_error:
                self.logger.error("artifact export failed: %s", result.artifact_error)
            if result.cleanup_error or result.remaining_temp_paths:
                self.logger.warning("temporary artifacts=%s cleanup_detail=%s",
                                    result.remaining_temp_paths, result.cleanup_error)


def format_race_status(status: Any, *, now: float | None = None) -> str:
    """Present the coordinator's values; never infer settlement from artifacts."""
    def known(value: Any) -> str:
        return "未知" if value is None else str(value)

    counts = status.decisions
    phase = {"starting": "连接中", "playing": "比赛中", "waiting": "等待开赛",
             "reconnecting": "重连等待", "stopping": "正在停止", "stopped": "已停止",
             "finalizing": "正在收尾", "exporting": "正在导出"}.get(status.phase, status.phase)
    reason = {"not_started": "尚未开赛", "ready": "已确认到位，等待开赛",
              "assignment": "等待分桌", "between_stages": "本阶段结束，等待后续安排",
              "stage_replay": "本阶段作废，等待重赛"}.get(status.waiting_reason)
    if reason and status.phase == "waiting":
        phase = reason
    now = time.monotonic() if now is None else now
    age = None if status.last_server_poll_at is None else max(0.0, now - status.last_server_poll_at)
    if status.server_poll_error:
        connection = "网络重试中，赛事状态已过期"
    elif age is not None and age > 10.0:
        connection = "赛事状态待刷新"
    else:
        connection = "查询中" if status.server_poll_in_flight else "已同步" if age is not None else "尚未同步"
    # A cached waiting fact must never look like a fresh server confirmation.
    if status.phase in {"starting", "waiting", "playing"} and (status.server_poll_error or age is not None and age > 10.0):
        phase = f"{connection}（上次状态：{phase}）"
    age_label = "未知" if age is None else f"{age:.1f}秒前"
    stats = status.stage_stats
    if stats is None:
        statistics = "已打=未知 胡牌=未知 阶段胜率=--"
    else:
        complete = stats.get("statistics_complete") is True
        rate = stats.get("win_rate")
        percentage = "--" if rate is None else f"{rate:.2%}"
        if not complete:
            reason = {"history_missing": "历史覆盖未知", "tables_unconfirmed": "牌局数据待确认",
                      "checkpoint_unavailable": "检查点不可用"}.get(stats.get("incomplete_reason"))
            missing = f"已确认小局中赛果缺失={stats.get('unknown_rounds', 0)}"
            percentage = f"统计不完整（{reason + '；' if reason else ''}{missing}）"
        prefix = "" if complete else "已确认"
        statistics = (f"{prefix}已打={known(stats.get('total_rounds'))} "
                      f"胡牌={known(stats.get('won_rounds'))} 阶段胜率={percentage}")
    ranking = {"previous": "上一阶段", "void": "作废阶段"}.get(status.ranking_scope, "")
    return (
        f"{phase} | {statistics} | 最近同步={age_label} 连接={connection} "
        f"run={known(status.run_id)} phase={status.phase} "
        f"status={known(status.tournament_status)} stage={known(status.stage_no)} "
        f"stage_name={known(status.stage_name)} stage_status={known(status.stage_status)} "
        f"qualified={known(status.qualified)} qualify_role={known(status.qualify_role)} "
        f"{ranking}排名={known(status.rank)} {ranking}得分={known(status.total_score)} "
        f"服务端{ranking or '本阶段'}结算数={known(status.stage_settled_rounds)}（含轮空） "
        f"active_games={known(status.active_games)} decisions={known(counts.get('total'))} "
        f"accepted={known(counts.get('accepted'))} conflict={known(counts.get('conflict'))} "
        f"error={known(counts.get('error'))} fallback={known(counts.get('fallback'))} "
        f"audit_bytes={known(status.audit_synced_bytes)} record_p99_ms={known(status.record_p99_ms)}"
    )


def format_race_health(health: dict[str, Any] | None, now: float) -> str:
    """Re-evaluate original timestamps, never the time this log was printed."""
    from .race_health import evaluate_health
    if health is None:
        return "心跳=未知 业务=未知"
    state = health.get("state", "starting")
    if state in {"reconnecting", "stopping", "stopped"}:
        label = {"reconnecting": "等待重启", "stopping": "停止中", "stopped": "已停止"}[state]
        return f"心跳={label} 业务={label}"
    heartbeat = health.get("heartbeat_at")
    heartbeat_age = None if heartbeat is None else max(0.0, now - heartbeat)
    heartbeat_label = "等待首个心跳" if heartbeat_age is None else "正常"
    if now - (heartbeat if heartbeat is not None else health["started_at"]) > health["heartbeat_timeout_seconds"]:
        heartbeat_label = "超时"
    if health.get("telemetry_error"):
        heartbeat_label = "异常"
    snapshot = health.get("snapshot")
    business = "待确认"
    if snapshot is not None:
        try:
            fault = evaluate_health(snapshot, now, health["stall_timeout_seconds"])
            business = "异常：" + fault["reason"] if fault else "正常"
        except (KeyError, TypeError, ValueError):
            business = "健康数据异常"
    if now - health.get("observed_at", now) > 5.0:
        business = "监控状态待刷新"
    age = "未知" if heartbeat_age is None else f"{heartbeat_age:.1f}秒前"
    return f"心跳={heartbeat_label}（{age}） 业务={business}"


@dataclass(frozen=True, slots=True)
class RaceProgressMessage:
    message: str
    generation: int
    pid: int | None
    restart_count: int
    important: bool = False


class RaceProgress:
    """A caller-serialized, in-memory race presentation clock.

    Activity comes from real coordinator checkpoints. This clock has no health
    authority and never touches recorder locks, files or network endpoints.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 active_interval: float = 5.0, idle_interval: float = 5.0) -> None:
        self.clock = clock
        self.active_interval = active_interval
        self.idle_interval = idle_interval
        self.generation = 0
        self.status: Any = None
        self.pid: int | None = None
        self.restart_count = 0
        self.last_emitted: float | None = None
        self.next_summary_at: float | None = None
        self.health: dict[str, Any] | None = None

    def advance_generation(self, generation: int) -> None:
        if generation > self.generation:
            self.generation = generation
            self.status = None
            self.health = None

    @staticmethod
    def _identity(status: Any) -> tuple[Any, ...]:
        return (status.run_id, status.phase, status.tournament_status, status.stage_no,
                status.stage_name, status.stage_status, status.qualified, status.qualify_role,
                status.waiting_reason, status.ready, status.stage_stats, status.server_poll_error)

    def update(self, status: Any, *, generation: int = 0, pid: int | None = None,
               restart_count: int = 0) -> RaceProgressMessage | None:
        if generation < self.generation:
            return None
        self.advance_generation(generation)
        previous = self.status
        if previous is not None and status.revision < previous.revision:
            return None
        if previous is not None and previous.run_id is not None and status.run_id != previous.run_id:
            return None
        self.status, self.pid, self.restart_count = status, pid, restart_count
        if previous is None or self._identity(previous) != self._identity(status):
            return self._message(important=True)
        return self.tick()

    def _message(self, *, important: bool = False) -> RaceProgressMessage:
        self.last_emitted = self.clock()
        if self.next_summary_at is None:
            self.next_summary_at = self.last_emitted + self.active_interval
        message = format_race_status(self.status, now=self.last_emitted)
        message += " | " + format_race_health(self.health, self.last_emitted)
        return RaceProgressMessage(message, self.generation,
                                   self.pid, self.restart_count, important)

    def tick(self) -> RaceProgressMessage | None:
        if self.status is None or self.status.phase == "stopped":
            return None
        interval = self.idle_interval if self.status.phase == "waiting" else self.active_interval
        if self.next_summary_at is None or self.clock() >= self.next_summary_at:
            self.next_summary_at = self.clock() + interval
            return self._message()
        return None
