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

from . import logger


def best(pred) -> int:
    """Index of the most likely outcome: 0 home, 1 draw, 2 away.

    The single definition of "the model's pick" - the scoreboard scores it
    and the sheet headlines it, so both must agree. A tie reads as a draw,
    which is what "too close to call" means.
    """
    ps = [pred["p_home"], pred["p_draw"], pred["p_away"]]
    return 1 if ps[1] == max(ps) else max(range(3), key=ps.__getitem__)


def compute_accuracy(log_path: str) -> dict | None:
    """Returns None if there's nothing settled yet to score."""
    got = logger.scorable(log_path)
    if got is None:
        return None
    df, excluded = got

    df = df.copy()
    df["predicted"] = df.apply(best, axis=1).map({0: "H", 1: "D", 2: "A"})
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
    # first-appearance order is config order, since each run logs its
    # leagues in the order config.toml lists them
    by_league = (
        df.groupby("league", sort=False)["correct"]
        .agg(n="size", hits="sum")
        .to_dict(orient="index")
    )

    return {
        "total": total,
        "wins": wins,
        "win_rate": wins / total,
        "by_outcome": by_outcome,   # {'H': {'n':.., 'hits':..}, 'D': {...}, 'A': {...}}
        "by_league": by_league,     # {'Premier League': {'n':.., 'hits':..}, ...}
        "post_match_excluded": excluded,
    }
