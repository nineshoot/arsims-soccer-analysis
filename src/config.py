"""Tiny config loader so every module reads the same settings."""
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def dated_path(cfg: dict, folder: str, stem: str) -> str:
    """output/<folder>/<stem>_<utc-date>.png

    The aggregate sheets are a running record, not a current-state file:
    each run keeps its own dated PNG instead of overwriting yesterday's, so
    the calibration curve and the hit rate can be read as a series. UTC to
    match the cron and the logged_at stamps in predictions_log.csv. Two runs
    on the same day land on the same name - the later one is the day's word.
    """
    out = Path(cfg["output_dir"]) / folder
    out.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    return str(out / f"{stem}_{today}.png")


def load() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve output paths relative to repo root.
    cfg["output_dir"] = str(ROOT / cfg["output_dir"])
    cfg["log_path"] = str(ROOT / cfg["log_path"])
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
