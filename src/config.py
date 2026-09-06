"""Tiny config loader so every module reads the same settings."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve output paths relative to repo root.
    cfg["output_dir"] = str(ROOT / cfg["output_dir"])
    cfg["log_path"] = str(ROOT / cfg["log_path"])
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
