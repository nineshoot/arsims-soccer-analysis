"""
The calibration ledger — the single most important file in the system.

Every pre-match prediction is appended here BEFORE the match is played.
Later, when results are known, you can score the model honestly:
'do the games I called 70% actually win ~70% of the time?'

Rows are keyed on (date, home, away) so re-running the pipeline before
kickoff updates in place instead of duplicating. A settled row is final:
no later prediction for the same fixture may replace it.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

KEY = ["date", "team_home", "team_away"]

COLUMNS = [
    "date", "league", "team_home", "team_away",
    "p_home", "p_draw", "p_away", "over_2_5", "btts",
    "mkt_home", "mkt_draw", "mkt_away", "odds_source",
    "logged_at", "post_match",
    # filled in later, after the match:
    "goals_home", "goals_away", "result",
]


def post_match(df: pd.DataFrame) -> pd.Series:
    """True where a prediction was logged after its match day.

    Such a row is not a forecast, and scoring it would flatter the model.
    The pipeline no longer writes them, but early runs predicted matches
    that fixtures.csv still listed after they were played; those rows stay
    in the ledger, flagged rather than deleted, and are left out of scoring.
    """
    match_day = pd.to_datetime(df["date"]).dt.date
    logged_day = pd.to_datetime(df["logged_at"], utc=True, format="ISO8601").dt.date
    return logged_day > match_day


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
    if old is not None:
        # A settled row is final. Re-predicting its fixture used to replace
        # it, which threw away the result and swapped a pre-match forecast
        # for one made after the match.
        settled = old.loc[old["result"].notna()].set_index(KEY).index
        new = new[~new.set_index(KEY).index.isin(settled)]
        combined = pd.concat([old, new], ignore_index=True)
        # an unsettled fixture re-predicted before kickoff takes the fresher row
        combined = combined.drop_duplicates(subset=KEY, keep="last")
    else:
        combined = new
    combined["post_match"] = post_match(combined)

    for col in COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined[COLUMNS].to_csv(path, index=False)
    print(f"  logged {len(new)} predictions -> {path}")
