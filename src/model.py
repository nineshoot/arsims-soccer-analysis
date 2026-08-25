"""
Model layer — thin wrapper around penaltyblog's Dixon-Coles goal model
with exponential time decay (recent games weighted more heavily).

penaltyblog does the heavy lifting (Cython Poisson + rho correction).
We just fit and expose a clean predict() that returns everything a
prediction row needs.

NOTE: penaltyblog's Cython loss function rejects read-only buffers, and
pandas-backed arrays often come through read-only. So we hand it FRESH,
writable numpy arrays built with _wr(). Passing df["col"] directly will
raise "buffer source array is read-only".
"""
from __future__ import annotations
import numpy as np
import penaltyblog as pb


def _wr(seq, dtype):
    """Fresh, writable, contiguous array — the shape penaltyblog needs."""
    a = np.array(list(seq), dtype=dtype)
    a.setflags(write=True)
    return a


class DixonColes:
    def __init__(self, xi: float = 0.0018, max_goals: int = 15):
        self.xi = xi
        self.max_goals = max_goals
        self.model = None
        self.teams: set[str] = set()

    def fit(self, df) -> "DixonColes":
        gh = _wr(df["goals_home"], np.int64)
        ga = _wr(df["goals_away"], np.int64)
        th = _wr(df["team_home"], object)
        ta = _wr(df["team_away"], object)
        weights = _wr(pb.models.dixon_coles_weights(df["date"], xi=self.xi),
                      np.float64)
        self.model = pb.models.DixonColesGoalModel(gh, ga, th, ta, weights)
        self.model.fit()
        self.teams = set(df["team_home"]).union(df["team_away"])
        return self

    def can_predict(self, home: str, away: str) -> bool:
        # A promoted team with no history can't be rated yet.
        return home in self.teams and away in self.teams

    def predict(self, home: str, away: str) -> dict:
        g = self.model.predict(home, away, max_goals=self.max_goals)
        p_home, p_draw, p_away = g.home_draw_away
        return {
            "p_home": round(float(p_home), 4),
            "p_draw": round(float(p_draw), 4),
            "p_away": round(float(p_away), 4),
            "over_2_5": round(float(g.total_goals("over", 2.5)), 4),
            "btts": round(float(g.btts_yes), 4),
            "_grid": g.grid,   # 2D scoreline matrix for the heatmap
        }
