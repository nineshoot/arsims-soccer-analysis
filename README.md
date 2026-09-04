# ⚽ ArSim's Soccer Analysis

A **zero-server, zero-cost** system that predicts upcoming football matches
from **pre-match data only** (no live/in-play), logs every prediction before
kickoff, and grades itself against the market. Runs unattended on GitHub
Actions and commits fresh charts back to the repo.

Built on [`penaltyblog`](https://github.com/martineastwood/penaltyblog)
(Dixon-Coles goal model with time decay) + free data from
[football-data.co.uk](https://www.football-data.co.uk/).

\---

## Why it's built this way

|Decision|Reason|
|-|-|
|**Dixon-Coles + time decay**, not just Poisson|Corrects Poisson's known under-count of low-scoring draws; recent form weighted heavier.|
|**Benchmark vs closing odds**|The honest test isn't "% correct" — it's whether the model tracks the market. Closing odds are the sharpest public probability.|
|**`predictions\_log.csv` written *before* kickoff**|The one file that makes the whole thing trustworthy: no hindsight, real calibration.|
|**GitHub Actions cron = the "automation"**|No Dataiku, no server, no bill. The green run-history *is* the automation.|
|**Static PNGs (Plotly + Playwright)**|Drop straight into a video-editing pipeline — no dashboard to host, no server to run.|

## How it works

```
                 football-data.co.uk (results + odds, free)
                              │
        ┌─────────────────────┴─────────────────────┐
        │  scrape historical results  │  scrape upcoming fixtures │
        └─────────────────────┬─────────────────────┘
                              │
     Dixon-Coles fit (time-decay)     strip bookmaker overround
                              │                    │
                       predict 1X2 / O-U / BTTS    │
                              │                    │
                     ┌────────┴─────────┐          │
              predictions\_log.csv    Plotly PNGs ◄─┘  (model vs market)
                     │
             (later) settle results ──► calibration curve
```

## Layout

```
football-predictor/
├── config.yaml                # leagues, seasons, xi, odds priority
├── requirements.txt
├── src/
│   ├── data.py                # download results + upcoming fixtures
│   ├── model.py               # Dixon-Coles wrapper (penaltyblog)
│   ├── odds.py                # bookmaker odds → fair probabilities
│   ├── logger.py              # the calibration ledger
│   ├── viz.py                 # mono-color print sheets: match, calibration, scoreboard
│   └── predict.py             # per-league orchestration
├── scripts/
│   ├── run\_pipeline.py        # main entry: predict + log + charts
│   └── settle\_results.py      # back-fill actual results
├── output/                    # generated PNGs (auto-committed by CI)
├── predictions\_log.csv        # created on first run
└── .github/workflows/predict.yml   # Thu predict · Mon settle
```

## Run it locally

```bash
pip install -r requirements.txt
playwright install chromium        # one-time: headless browser for chart rendering
python -m scripts.run\_pipeline     # predict this week's fixtures
# ...after the matches are played:
python -m scripts.settle\_results   # fill in results
python -m scripts.run\_pipeline     # redraw calibration curve
```

Charts land in `output/`. Predictions accumulate in `predictions\_log.csv`.

## Automate it (GitHub Actions)

Push to GitHub — `.github/workflows/predict.yml` already schedules:

* **Thursday 07:00 UTC** — predict the weekend's fixtures + render charts
* **Monday 09:00 UTC** — settle results + redraw calibration

It commits `predictions\_log.csv` and `output/\*.png` back automatically.
Trigger a manual run anytime from the **Actions** tab (`workflow\_dispatch`).

## Tuning

* **`config.yaml → model.xi`** — decay speed. Higher = forget old games
faster. Backtest to find the sweet spot for each league.
* **`odds\_priority`** — which bookmaker columns to benchmark against.
* **`leagues`** — add `I1` (Serie A), `D1` (Bundesliga), `F1` (Ligue 1).

## Where to take it next

* Swap in **xG-based** attack/defence strengths via
[`soccerdata`](https://github.com/probberechts/soccerdata) (FBref/Understat).
* Add an **ensemble** second model (XGBoost on Elo gap + market value),
weighted like the reference design in `caroescm/football-odds`.
* Track **Ranked Probability Score / log-loss vs closing odds** in the log —
the real scoreboard.

## Data credit

Results \& odds: football-data.co.uk. If you later use StatsBomb Open Data
(`hudl/open-data`) for xG features, credit StatsBomb per their terms.

