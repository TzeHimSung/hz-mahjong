from __future__ import annotations

import logging
from pathlib import Path

from heuristic_ai.config.loader import load_config
from heuristic_ai.runtime.runner import report_competition_result, run_configured_competition, startup_error

LOGGER = logging.getLogger("heuristic_ai.launch")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    heuristic_root = Path(__file__).resolve().parent
    config = None
    try:
        config = load_config(heuristic_root / "config.yaml")
        result = run_configured_competition(config)
    except Exception as error:
        LOGGER.error("launch failed before artifact creation: %s", startup_error(error, config))
        return 1

    report_competition_result(result, LOGGER)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
