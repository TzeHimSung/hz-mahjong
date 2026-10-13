"""Competition mode dispatch and the command-line entry point."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from heuristic_ai.artifacts.models import ArtifactStatus, CleanupStatus, RunResult, RunStatus
from heuristic_ai.config.loader import load_config
from heuristic_ai.config.models import AppConfig
from .race import run_race
from .train import run_train
from .win_stats import format_match_win_stats
from .supervision_contracts import RaceSupervisionResult

if TYPE_CHECKING:
    from .match import MatchLoopResult


LOGGER = logging.getLogger("heuristic_ai.runner")


def run_competition(config: AppConfig) -> RunResult | MatchLoopResult:
    """Dispatch the selected mode from one fully validated root configuration."""

    if config.mode == "train":
        return run_train(config.train)
    if config.mode == "race":
        return run_race(config.race)
    if config.mode == "match":
        if config.match is None or not config.global_token:
            raise ValueError("match mode requires match configuration and global_token")
        from .match import run_match

        return run_match(config.match, global_token=config.global_token)
    raise ValueError("unsupported competition mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed One-Ply Expectation Mahjong client.")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def run_configured_competition(config: AppConfig) -> RunResult | MatchLoopResult | RaceSupervisionResult:
    """CLI dispatch; embedded run_competition callers keep their direct behavior."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if config.mode != "race":
        return run_competition(config)
    from .process_guard import MachineRaceLock
    with MachineRaceLock():
        if config.race.supervision.enabled:
            from .race_supervision import run_supervised_race
            return run_supervised_race(config.race)
        return run_competition(config)


def _report_session(result: RunResult, logger: logging.Logger) -> None:
    if result.artifact_status is ArtifactStatus.INCOMPLETE:
        logger.error("run=%s status=%s artifact export incomplete work=%s error=%s",
                     result.run_id, result.status, result.work_dir, result.artifact_error)
    elif result.status is RunStatus.COMPLETED:
        logger.info("run=%s completed artifacts=%s", result.run_id, result.artifact_dir)
    elif result.status is RunStatus.DEGRADED:
        logger.warning("run=%s degraded artifacts=%s", result.run_id, result.artifact_dir)
    elif result.status.value == "interrupted":
        logger.info("run=%s interrupted artifacts=%s", result.run_id, result.artifact_dir)
    else:
        logger.error("run=%s failed artifacts=%s", result.run_id, result.artifact_dir)
    if result.cleanup_status is CleanupStatus.PENDING:
        logger.warning("run=%s cleanup pending paths=%s error=%s", result.run_id,
                       result.remaining_temp_paths, result.cleanup_error)


def report_competition_result(
    result: RunResult | MatchLoopResult | RaceSupervisionResult,
    logger: logging.Logger = LOGGER,
) -> None:
    """Report a single run or a bounded-memory online matching summary."""

    if isinstance(result, RunResult):
        _report_session(result, logger)
        return
    if result.last_result is not None:
        _report_session(result.last_result, logger)
    level = (
        logging.INFO if result.exit_code in {0, 130}
        else logging.WARNING if result.exit_code == 2
        else logging.ERROR
    )
    if isinstance(result, RaceSupervisionResult):
        logger.log(level, "race_supervision=%s restarts=%s stop=%s exit_code=%s logs=%s",
                   result.supervision_id, result.restart_count, result.stop_reason,
                   result.exit_code, result.log_dir)
        return
    logger.log(
        level,
        "match_loop=%s sessions=%s completed=%s degraded=%s stop=%s exit_code=%s %s",
        result.session_group_id, result.session_count, result.completed_sessions,
        result.degraded_sessions, result.stop_reason, result.exit_code,
        format_match_win_stats(result.win_stats),
    )


def startup_error(error: Exception, config: AppConfig | None) -> str:
    """Mask all configured credentials if execution fails before a recorder exists."""

    if config is None:
        return type(error).__name__
    message = str(error)
    tokens = [config.global_token, config.race.participant.token]
    tokens.extend(player.token for player in config.train.players)
    for token in sorted((token for token in tokens if token), key=len, reverse=True):
        message = message.replace(token, "<redacted>")
    return message


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config: AppConfig | None = None
    try:
        config = load_config(args.config)
        result = run_configured_competition(config)
    except Exception as error:
        LOGGER.error("run failed before artifact creation: %s", startup_error(error, config))
        return 1

    report_competition_result(result)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
