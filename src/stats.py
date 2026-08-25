"""
Accuracy scoreboard — distinct from calibration_curve in viz.py.

Calibration asks: "when the model says 70%, does it actually win ~70%
of the time?" (probability quality).

This asks the simpler, more video-friendly question: "how many times
did the model's top pick actually happen, and what's the hit rate?"
(win count / win %). Both matter, but this one is the number people
actually want to see on screen.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

_OUTCOME_MAP = {"p_home": "H", "p_draw": "D", "p_away": "A"}


def compute_accuracy(log_path: str) -> dict | None:
    """Returns None if there's nothing settled yet to score."""
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["result"])
    if df.empty:
        return None

    probs = df[["p_home", "p_draw", "p_away"]]
    df = df.copy()
    df["predicted"] = probs.idxmax(axis=1).map(_OUTCOME_MAP)
    df["correct"] = df["predicted"] == df["result"]

    total = len(df)
    wins = int(df["correct"].sum())

    # breakdown by predicted outcome type — are home favourites more
    # reliable than away/draw picks? useful for a follow-up video.
    by_outcome = (
        df.groupby("predicted")["correct"]
        .agg(n="size", hits="sum")
        .to_dict(orient="index")
    )

    return {
        "total": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": wins / total,
        "by_outcome": by_outcome,   # {'H': {'n':.., 'hits':..}, 'D': {...}, 'A': {...}}
    }
