#!/usr/bin/env python3
"""
Back-fills actual results into predictions_log.csv so the calibration
curve has something to score against.

Matches logged predictions to played games by (date, home, away) and
writes goals_home / goals_away / result. Run it a few days after each
match round (or let CI run it every time — settled rows just stay set).
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, data, stats, viz  # noqa: E402


def main() -> None:
    cfg = config.load()
    log_path = Path(cfg["log_path"])
    if not log_path.exists():
        print("no log yet, nothing to settle.")
        return

    log = pd.read_csv(log_path)
    unsettled = log[log["result"].isna()]
    if unsettled.empty:
        print("all predictions already settled.")
    else:
        # pull recent results for every league we track
        played = []
        for lg in cfg["leagues"]:
            try:
                r = data.results(lg["code"], cfg["train_seasons"][-1:])
                played.append(r[["date", "team_home", "team_away",
                                 "goals_home", "goals_away", "result"]])
            except Exception as e:
                print(f"  [warn] {lg['name']}: {e}")
        if played:
            played = pd.concat(played, ignore_index=True)
            played["date"] = pd.to_datetime(played["date"]).dt.date.astype(str)

            key = ["date", "team_home", "team_away"]
            log = log.merge(played, on=key, how="left", suffixes=("", "_actual"))
            for col in ["goals_home", "goals_away", "result"]:
                log[col] = log[col].fillna(log[f"{col}_actual"])
                log = log.drop(columns=[f"{col}_actual"])

            log.to_csv(log_path, index=False)
            now_settled = log["result"].notna().sum()
            print(f"settled. {now_settled} rows now have results.")

    # win count / win rate — the on-screen number, separate from calibration
    acc = stats.compute_accuracy(str(log_path))
    if acc:
        print(f"  scoreboard: {acc['wins']}/{acc['total']} correct "
              f"({acc['win_rate']*100:.1f}%)")
        out = Path(cfg["output_dir"]) / "accuracy_scoreboard.png"
        viz.accuracy_scoreboard(acc, str(out))
    else:
        print("  [skip] scoreboard: no settled results yet")


if __name__ == "__main__":
    main()
