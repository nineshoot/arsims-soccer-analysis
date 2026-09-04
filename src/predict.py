"""
Per-league orchestration: fit the model, predict the upcoming fixtures,
build chart rows, and hand back prediction dicts for logging.

Charts are grouped into a per-matchweek subfolder under output/, named
by the Monday of the week the fixtures fall in (e.g. output/2026-08-24/).
That keeps months of accumulated PNGs from turning output/ into one flat
pile — each week's video assets live in their own folder.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from . import data, viz
from .model import DixonColes
from .odds import implied_from_row


def _week_folder(base: Path, dates) -> Path:
    """Monday of the ISO week the earliest fixture date falls in."""
    monday = (dates.min() - pd.Timedelta(days=int(dates.min().weekday()))).date()
    folder = base / monday.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _h2h_rows(hist: pd.DataFrame, home: str, away: str, seasons: list[str],
             n: int = 4) -> list[dict]:
    """Last `n` meetings between these two teams in the most recent 2 seasons."""
    recent = set(seasons[-2:])
    mask = (hist["season"].isin(recent) &
           (((hist["team_home"] == home) & (hist["team_away"] == away)) |
            ((hist["team_home"] == away) & (hist["team_away"] == home))))
    games = hist.loc[mask].sort_values("date", ascending=False).head(n)
    return [
        {"date": r["date"].strftime("%Y-%m-%d"),
         "line": f"{r['team_home']} {r['goals_home']}-{r['goals_away']} {r['team_away']}"}
        for _, r in games.iterrows()
    ]


def run_league(code: str, name: str, cfg: dict) -> list[dict]:
    print(f"[{name}] downloading results…")
    hist = data.results(code, cfg["train_seasons"])

    print(f"[{name}] fitting Dixon-Coles (xi={cfg['model']['xi']})…")
    model = DixonColes(xi=cfg["model"]["xi"],
                       max_goals=cfg["model"]["max_goals"]).fit(hist)

    print(f"[{name}] fetching upcoming fixtures…")
    try:
        fix = data.fixtures(code)
    except Exception as e:
        print(f"  [warn] no fixtures for {code}: {e}")
        return []

    if fix.empty:
        print(f"  [warn] no fixtures for {code}")
        return []

    out_dir = _week_folder(Path(cfg["output_dir"]), fix["date"])
    rows: list[dict] = []

    for _, f in fix.iterrows():
        home, away = f["team_home"], f["team_away"]
        if not model.can_predict(home, away):
            print(f"  [skip] unrated team: {home} / {away}")
            continue

        pred = model.predict(home, away)
        market = implied_from_row(f, cfg["odds_priority"])
        h2h = _h2h_rows(hist, home, away, cfg["train_seasons"])

        rows.append({
            "date": f["date"].date().isoformat(),
            "league": name,
            "team_home": home, "team_away": away,
            "p_home": pred["p_home"], "p_draw": pred["p_draw"],
            "p_away": pred["p_away"],
            "over_2_5": pred["over_2_5"], "btts": pred["btts"],
            "mkt_home": market["p_home"] if market else None,
            "mkt_draw": market["p_draw"] if market else None,
            "mkt_away": market["p_away"] if market else None,
            "odds_source": market["odds_source"] if market else None,
        })

        slug = f"{code}_{home}_{away}".replace(" ", "-")
        try:
            viz.match_dashboard(pred, market, home, away, h2h,
                                str(out_dir / f"{slug}_dashboard.png"),
                                league=name, date=f["date"].date().isoformat())
        except Exception as e:
            # Chart rendering is a nice-to-have, not the source of truth —
            # a failure here (e.g. a broken kaleido/playwright install on
            # some CI runner) must never cost us the numeric prediction,
            # which is already in `rows` and is what predictions_log.csv
            # depends on.
            print(f"  [warn] dashboard render failed for {home} vs {away}: {e}")

    print(f"[{name}] {len(rows)} fixtures predicted -> {out_dir}")
    return rows
