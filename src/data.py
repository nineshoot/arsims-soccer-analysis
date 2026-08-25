"""
Data layer.

Two jobs:
  1. results()  -> historical match results + bookmaker odds for TRAINING
                   and for calibration benchmarking.
  2. fixtures() -> the upcoming (not-yet-played) fixtures we predict.

Everything comes from football-data.co.uk, which is free, needs no
API key, ships closing odds alongside results, and is stable enough to
run unattended in CI. soccerdata is left as an optional richer source
(FBref xG etc.) — wire it in later if you want more features.

NOTE ON ENCODING: the per-season result files (mmz4281/.../E0.csv) are
plain latin-1. The all-leagues fixtures.csv, however, ships with a
UTF-8 byte-order-mark (BOM) at the very start of the file. Decoding
that file as latin-1 turns "Div" into "\ufeffDiv" (or worse, mojibake),
so every column lookup on 'Div' silently fails. _get_csv() sniffs for
the BOM and picks the right decoding per-file.
"""
from __future__ import annotations
import io
import requests
import pandas as pd

BASE = "https://www.football-data.co.uk"
FIXTURES_URL = f"{BASE}/fixtures.csv"          # all leagues, next ~1 week
SEASON_URL = f"{BASE}/mmz4281/{{season}}/{{code}}.csv"

# football-data.co.uk uses these column names; we normalise to ours.
_RENAME = {
    "Date": "date",
    "HomeTeam": "team_home",
    "AwayTeam": "team_away",
    "FTHG": "goals_home",
    "FTAG": "goals_away",
    "FTR": "result",  # H / D / A
}


def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    raw = r.content
    # BOM-aware decode: fixtures.csv ships UTF-8-BOM, season files are latin-1.
    if raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        text = raw.decode("latin-1")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


def results(code: str, seasons: list[str]) -> pd.DataFrame:
    """Historical results + odds for one league across several seasons."""
    frames = []
    for season in seasons:
        try:
            df = _get_csv(SEASON_URL.format(season=season, code=code))
            df["season"] = season
            frames.append(df)
        except Exception as e:  # a season file may not exist yet
            print(f"  [warn] {code} {season}: {e}")
    if not frames:
        raise RuntimeError(f"No result data downloaded for {code}")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "goals_home", "goals_away",
                           "team_home", "team_away"])
    df["goals_home"] = df["goals_home"].astype(int)
    df["goals_away"] = df["goals_away"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def fixtures(code: str) -> pd.DataFrame:
    """Upcoming fixtures for one league (with pre-match odds if present)."""
    df = _get_csv(FIXTURES_URL)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]  # belt & braces
    df = df[df["Div"] == code].copy()
    df = df.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    keep = ["date", "team_home", "team_away"]
    # carry through whatever odds columns exist for the benchmark overlay
    odds_cols = [c for c in df.columns
                 if c[:-1] in ("B365", "PS", "Avg") and c[-1] in "HDA"]
    return df[keep + odds_cols].dropna(subset=["team_home", "team_away"])
