#!/usr/bin/env python3
"""
Main entry point. Run locally with `python -m scripts.run_pipeline`
or let GitHub Actions call it on a schedule.

Steps:
  1. for each configured league: fit model, predict upcoming fixtures,
     render per-match PNGs
  2. append all predictions to predictions_log.csv (the calibration ledger)
  3. redraw the calibration curve from whatever results are settled so far
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, logger, viz          # noqa: E402
from src.predict import run_league           # noqa: E402


def main() -> None:
    cfg = config.load()
    all_rows: list[dict] = []

    for lg in cfg["leagues"]:
        try:
            all_rows += run_league(lg["code"], lg["name"], cfg)
        except Exception as e:
            print(f"[error] {lg['name']}: {e}")

    if all_rows:
        logger.append(all_rows, cfg["log_path"])

    viz.calibration_curve(
        cfg["log_path"],
        str(Path(cfg["output_dir"]) / "calibration.png"))

    print("done.")


if __name__ == "__main__":
    main()
