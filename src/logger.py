"""
The calibration ledger — the single most important file in the system.

Every pre-match prediction is appended here BEFORE the match is played.
Later, when results are known, you can score the model honestly:
'do the games I called 70% actually win ~70% of the time?'

Rows are keyed on (date, home, away) so re-running the pipeline the same
day updates in place instead of duplicating.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS = [
    "date", "league", "team_home", "team_away",
    "p_home", "p_draw", "p_away", "over_2_5", "btts",
    "mkt_home", "mkt_draw", "mkt_away", "odds_source",
    "logged_at",
    # filled in later, after the match:
    "goals_home", "goals_away", "result",
]


def _read_existing(path: Path) -> pd.DataFrame | None:
    """Return the existing log, or None if there's nothing usable yet.

    A previous run can leave a 0-byte or header-only file behind (e.g. an
    interrupted write, or a stray `touch`). pandas raises EmptyDataError
    on a genuinely empty file, so we check size first rather than let
    that propagate — a missing/empty log is just "start fresh", not a bug.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def append(rows: list[dict], log_path: str) -> None:
    new = pd.DataFrame(rows)
    new["logged_at"] = pd.Timestamp.utcnow().isoformat()

    path = Path(log_path)
    old = _read_existing(path)
    combined = pd.concat([old, new], ignore_index=True) if old is not None else new
    if old is not None:
        # keep the latest prediction per fixture
        combined = combined.drop_duplicates(
            subset=["date", "team_home", "team_away"], keep="last")

    for col in COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined[COLUMNS].to_csv(path, index=False)
    print(f"  logged {len(new)} predictions -> {path}")
