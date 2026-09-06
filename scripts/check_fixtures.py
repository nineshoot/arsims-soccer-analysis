#!/usr/bin/env python3
"""
Quick fixtures.csv freshness check — no model fitting, no chart
rendering, just: "has football-data.co.uk published the games I want
yet?" Useful when you're impatient waiting for a specific matchday
(e.g. Saturday's Premier League fixtures) to show up.

Usage:
    python -m scripts.check_fixtures          # shows every league in the file
    python -m scripts.check_fixtures E0        # filter to one league code
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import _get_csv, FIXTURES_URL  # noqa: E402


def main() -> None:
    df = _get_csv(FIXTURES_URL)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    if len(sys.argv) > 1:
        code = sys.argv[1].upper()
        df = df[df["Div"] == code]
        if df.empty:
            print(f"No fixtures for {code} yet. Try again later.")
            return

    print(f"Leagues currently listed: {sorted(df['Div'].dropna().unique())}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")
    cols = [c for c in ["Div", "Date", "Time", "HomeTeam", "AwayTeam"]
            if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
