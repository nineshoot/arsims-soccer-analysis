"""
Turn bookmaker decimal odds into a fair probability the model can be
compared against. The raw 1/odds sums to >1 (the bookmaker's margin /
overround); penaltyblog strips it so home+draw+away = 1.
"""
from __future__ import annotations
import penaltyblog as pb


def implied_from_row(row, priority: list[str]) -> dict | None:
    """
    row: a fixtures/results row (pandas Series) carrying odds columns like
         AvgCH/AvgCD/AvgCA, PSCH/..., B365H/...
    priority: prefixes to try in order, e.g. ["AvgC", "PSC", "B365C", ...]
    Returns {'p_home','p_draw','p_away','odds_source'} or None.
    """
    for pref in priority:
        h, d, a = f"{pref}H", f"{pref}D", f"{pref}A"
        # all three present and not NaN (NaN != NaN)
        if all(c in row and row[c] == row[c] for c in (h, d, a)):
            fair = pb.implied.calculate_implied(
                [float(row[h]), float(row[d]), float(row[a])],
                method="multiplicative")
            p = fair.probabilities
            return {"p_home": round(float(p[0]), 4),
                    "p_draw": round(float(p[1]), 4),
                    "p_away": round(float(p[2]), 4),
                    "odds_source": pref}
    return None
