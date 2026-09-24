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
that file as latin-1 turns "Div" into "﻿Div" (or worse, mojibake),
so every column lookup on 'Div' silently fails. _get_csv() sniffs for
the BOM and picks the right decoding per-file.

FALLBACK: if football-data.co.uk itself is unreachable (not just one
missing file — the retry-with-backoff below already absorbs those), both
functions fall back to openfootball/football.json on GitHub. It has no
odds, which the rest of the pipeline already treats as optional. The one
rule that matters: football-data.co.uk spells teams "Liverpool",
openfootball spells them "Liverpool FC" — mixing the two sources within
one league's fit() would silently split a team's history into two
"different" teams. So the fallback is all-or-nothing per league per run:
_used_fallback remembers that results() had to switch, and fixtures()
checks it before even trying football-data.co.uk, so both always agree.
"""
from __future__ import annotations
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
}

# The site occasionally 503s under load.  Do not honour its Retry-After
# header: it has returned hour-long values, which makes an unattended CI run
# look hung inside urllib3.  Two bounded retries are enough to absorb a blip;
# after that results() switches to the fallback source.
_session = requests.Session()
_retry_adapter = HTTPAdapter(max_retries=Retry(
    total=2,
    backoff_factor=0.5,
    backoff_max=2,
    status_forcelist=[502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=False,
))
_session.mount("https://", _retry_adapter)
_session.mount("http://", _retry_adapter)


def _get_csv(url: str) -> pd.DataFrame:
    r = _session.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    raw = r.content
    # BOM-aware decode: fixtures.csv ships UTF-8-BOM, season files are latin-1.
    if raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        text = raw.decode("latin-1")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


# ---- openfootball fallback (github.com/openfootball/football.json) -------
_OF_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
_OF_CODE = {"E0": "en.1", "SP1": "es.1", "I1": "it.1", "D1": "de.1", "F1": "fr.1"}
# openfootball times are league-local. Checked for en.1 and es.1 against
# football-data.co.uk kickoffs; the other three follow the same convention.
_OF_TZ = {"E0": "Europe/London", "SP1": "Europe/Madrid", "I1": "Europe/Rome",
          "D1": "Europe/Berlin", "F1": "Europe/Paris"}
_used_fallback: dict[str, bool] = {}  # code -> did results() have to switch?
FIXTURE_WINDOW_DAYS = 8  # roughly what football-data.co.uk's fixtures.csv holds


def _of_season(season: str) -> str:
    """'2526' -> '2025-26' (football-data.co.uk season code -> openfootball folder)."""
    return f"20{season[:2]}-{season[2:]}"


def _of_matches(code: str, season: str) -> pd.DataFrame:
    of_code = _OF_CODE.get(code)
    if not of_code:
        raise RuntimeError(f"no openfootball mapping for league code {code}")
    url = f"{_OF_BASE}/{_of_season(season)}/{of_code}.json"
    r = _session.get(url, timeout=30)
    r.raise_for_status()
    rows = []
    for m in r.json()["matches"]:
        score = m.get("score")
        ft = score.get("ft") if isinstance(score, dict) else score  # schema varies
        rows.append({
            "date": m["date"], "time": m.get("time"),
            "team_home": m["team1"], "team_away": m["team2"],
            "goals_home": ft[0] if ft else None, "goals_away": ft[1] if ft else None,
        })
    return pd.DataFrame(rows)


def _kickoff(date: pd.Series, time: pd.Series, tz: str) -> pd.Series:
    """Local date + local HH:MM -> UTC kickoff. NaT wherever the time is unknown."""
    local = pd.to_datetime(date.dt.strftime("%Y-%m-%d") + " " + time.astype("string"),
                           errors="coerce")
    return local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def _upcoming(df: pd.DataFrame) -> pd.Series:
    """Not yet kicked off. A fixture with no known time falls back to its date."""
    now = pd.Timestamp.now("UTC")
    known = df["kickoff_utc"].notna()
    return ((known & (df["kickoff_utc"] > now))
            | (~known & (df["date"] >= now.tz_localize(None).normalize())))


def results(code: str, seasons: list[str]) -> pd.DataFrame:
    """Historical results + odds for one league across several seasons."""
    frames = []
    for season in seasons:
        try:
            df = _get_csv(SEASON_URL.format(season=season, code=code))
            df = df.rename(columns=_RENAME)
            df["season"] = season
            frames.append(df)
        except Exception as e:  # a season file may not exist yet
            print(f"  [warn] {code} {season}: {e}")
            # A missing season (404) is local to that URL, so try the next
            # one.  A connection error or 5xx after the bounded retries means
            # the primary service is unavailable; hammering every remaining
            # season only repeats the same failure and delays the fallback.
            response = getattr(e, "response", None)
            status = getattr(response, "status_code", None)
            if not frames and isinstance(e, requests.RequestException) \
                    and (status is None or status >= 500):
                break

    if not frames:
        print(f"  [warn] {code}: football-data.co.uk unreachable, "
              f"falling back to openfootball (no odds)")
        _used_fallback[code] = True
        for season in seasons:
            try:
                df = _of_matches(code, season)
                df["season"] = season
                frames.append(df)
            except Exception as e:
                print(f"  [warn] openfootball {code} {season}: {e}")
        if not frames:
            raise RuntimeError(f"No result data downloaded for {code}")

    df = pd.concat(frames, ignore_index=True)
    # football-data.co.uk ships dd/mm/yyyy, openfootball ISO. The two are
    # never mixed in one call, so the flag alone picks the right reading.
    df["date"] = pd.to_datetime(df["date"], dayfirst=not _used_fallback.get(code),
                                errors="coerce")
    df = df.dropna(subset=["date", "goals_home", "goals_away",
                           "team_home", "team_away"])
    df["goals_home"] = df["goals_home"].astype(int)
    df["goals_away"] = df["goals_away"].astype(int)
    # derived fresh (rather than trusting football-data.co.uk's own FTR
    # column) so it works the same regardless of which source supplied the row
    df["result"] = "D"
    df.loc[df["goals_home"] > df["goals_away"], "result"] = "H"
    df.loc[df["goals_home"] < df["goals_away"], "result"] = "A"
    return df.sort_values("date").reset_index(drop=True)


def fixtures(code: str, season: str | None = None) -> pd.DataFrame:
    """Upcoming fixtures for one league (with pre-match odds if present).

    `season` (football-data.co.uk code, e.g. "2526") is only used if the
    openfootball fallback kicks in - pass the latest of your train_seasons.
    """
    if not _used_fallback.get(code):
        try:
            df = _get_csv(FIXTURES_URL)
            df.columns = [c.strip().lstrip("﻿") for c in df.columns]  # belt & braces
            df = df[df["Div"] == code].copy()
            df = df.rename(columns=_RENAME)
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            # fixtures.csv lists kickoff in UK time, whatever the league
            time = df["Time"] if "Time" in df else pd.Series(pd.NA, index=df.index)
            df["kickoff_utc"] = _kickoff(df["date"], time, "Europe/London")
            # fixtures.csv keeps listing matches for a while after they are
            # played. Predicting one then is not a forecast - and the model
            # may already have trained on its result.
            df = df[_upcoming(df)]
            keep = ["date", "kickoff_utc", "team_home", "team_away"]
            # carry through whatever odds columns exist for the benchmark overlay
            odds_cols = [c for c in df.columns
                         if c[:-1] in ("B365", "PS", "Avg") and c[-1] in "HDA"]
            return df[keep + odds_cols].dropna(subset=["team_home", "team_away"])
        except Exception as e:
            print(f"  [warn] {code} fixtures: {e}")

    if not season:
        raise RuntimeError(f"No fixtures for {code}: football-data.co.uk "
                           f"unreachable and no season given for the fallback")
    print(f"  [warn] {code}: falling back to openfootball for fixtures (no odds)")
    df = _of_matches(code, season)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["kickoff_utc"] = _kickoff(df["date"], df["time"], _OF_TZ[code])
    df = df[df["goals_home"].isna() & _upcoming(df)]  # unplayed and not yet started
    # openfootball carries the WHOLE remaining season; fixtures.csv only ever
    # holds the next ~week. Match that window, or one fallback run would
    # render hundreds of sheets per league.
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    df = df[df["date"] <= today + pd.Timedelta(days=FIXTURE_WINDOW_DAYS)]
    return (df[["date", "kickoff_utc", "team_home", "team_away"]]
            .dropna(subset=["team_home", "team_away"]))


def _demo() -> None:
    """`python -m src.data` - checks both sources and the switch between them."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    assert _of_season("2526") == "2025-26"

    # --- primary path: a stand-in football-data.co.uk, odds and all
    season_csv = (b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\r\n"
                  b"E0,16/08/2024,Man United,Fulham,1,0,H,2.10,3.40,3.60\r\n"
                  b"E0,17/08/2024,Arsenal,Wolves,2,2,D,1.30,5.80,9.00\r\n")
    # relative dates: a fixed one would drift into the past and get filtered
    day = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    past, soon = day - pd.Timedelta(days=1), day + pd.Timedelta(days=3)
    fix_csv = ("﻿Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\r\n"
               f"E0,{past:%d/%m/%Y},15:00,Played,Already,2.50,3.30,2.80\r\n"
               f"E0,{soon:%d/%m/%Y},15:00,Everton,Man United,2.50,3.30,2.80\r\n").encode()

    class _Stub(BaseHTTPRequestHandler):
        def do_GET(self):
            body = fix_csv if "fixtures" in self.path else season_csv
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    global SEASON_URL, FIXTURES_URL
    keep = SEASON_URL, FIXTURES_URL
    SEASON_URL, FIXTURES_URL = f"{base}/{{season}}/{{code}}.csv", f"{base}/fixtures.csv"
    _used_fallback.clear()
    hist = results("E0", ["2526"])
    assert not _used_fallback.get("E0"), "must not switch while the site answers"
    assert list(hist["result"]) == ["H", "D"], hist["result"].tolist()
    assert str(hist["date"].iloc[0].date()) == "2024-08-16", "dd/mm/yyyy is dayfirst"
    assert "B365H" in hist.columns, "odds must survive for the market benchmark"
    fx = fixtures("E0")
    assert "B365H" in fx.columns
    assert list(fx["team_home"]) == ["Everton"], "a match already played must be dropped"
    ko = fx["kickoff_utc"].iloc[0]
    want = (soon + pd.Timedelta(hours=15)).tz_localize("Europe/London").tz_convert("UTC")
    assert ko == want, f"15:00 UK must convert to UTC: {ko} != {want}"
    srv.shutdown()

    # A hostile Retry-After must never park CI for the requested hour.  The
    # adapter should use only our short bounded backoff, then raise.
    class _UnavailableStub(BaseHTTPRequestHandler):
        calls = 0

        def do_GET(self):
            type(self).calls += 1
            self.send_response(503)
            self.send_header("Retry-After", "3600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    unavailable = HTTPServer(("127.0.0.1", 0), _UnavailableStub)
    threading.Thread(target=unavailable.serve_forever, daemon=True).start()
    started = time.monotonic()
    try:
        _get_csv(f"http://127.0.0.1:{unavailable.server_port}/season.csv")
        raise AssertionError("503 response should fail after bounded retries")
    except requests.RequestException:
        pass
    finally:
        unavailable.shutdown()
    elapsed = time.monotonic() - started
    assert _UnavailableStub.calls == 3, _UnavailableStub.calls
    assert elapsed < 5, f"Retry-After was not bounded: {elapsed:.1f}s"

    # --- fallback: a URL requests refuses outright, so no retry wait
    SEASON_URL, FIXTURES_URL = "dead://{season}/{code}", "dead://fixtures"
    _used_fallback.clear()
    hist = results("E0", ["2425"])
    assert _used_fallback.get("E0") is True, "should have switched to openfootball"
    assert len(hist) == 380, f"a full EPL season is 380 matches, got {len(hist)}"
    assert hist["date"].notna().all(), "ISO dates must parse"
    wrong = hist[hist["result"] != hist.apply(
        lambda r: "H" if r.goals_home > r.goals_away
        else "A" if r.goals_home < r.goals_away else "D", axis=1)]
    assert wrong.empty, f"{len(wrong)} rows have the wrong H/D/A"
    # the window matters: openfootball holds the whole season, not one week
    assert len(fixtures("E0", "2627")) < 25, "fixtures must stay near one matchweek"
    SEASON_URL, FIXTURES_URL = keep
    print("ok - primary path, openfootball fallback, and the switch between them")


if __name__ == "__main__":
    _demo()
