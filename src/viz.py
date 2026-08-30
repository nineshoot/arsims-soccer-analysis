"""
Static, high-res PNGs for the video pipeline. Three outputs:

  1. match_dashboard    - one 2x2 glass-card image per fixture:
                           model-vs-market bars, scoreline heatmap,
                           head-to-head history, and the verdict.
  2. calibration_curve  - reliability plot built from predictions_log.csv
  3. accuracy_scoreboard - win-count KPI card

Rendering pipeline: Plotly still draws every chart (bars, heatmap, curve,
KPI number) exactly as before - kaleido rasterizes each to a transparent
PNG, which gets embedded into an HTML "glass card" template, and
Playwright's headless Chromium screenshots the assembled page. Plotly
alone can't do frosted-glass/backdrop-blur; CSS can, so the two are
combined rather than reimplementing the charts in a new stack.

Design language: Linear/Notion dark aesthetic + data-ink discipline
(principles borrowed from indi256s/dataviz-skill and Anthropic's
data-visualization skill):
  - Data-ink ratio is sacred - every gridline/border earns its place.
  - Titles are conclusions ("City strongly favoured"), not axis labels.
  - Gray is the default; ONE accent colour highlights the actual story
    (the favoured outcome), everything else recedes to grayscale - this
    also means colour is never the only signal (labels always present),
    satisfying colourblind-safety by construction.
  - Sequential, single-hue colour scales for the heatmap (never a
    red/green diverging scale) - also colourblind-safe by construction.
"""
from __future__ import annotations
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from playwright.sync_api import sync_playwright

CANVAS_W, CANVAS_H = 1920, 1080
SUB_SCALE = 2  # oversample embedded Plotly PNGs so they stay crisp at CSS display size

# ---- Linear-style dark palette ---------------------------------------
BG = "#08090a"            # Linear's near-black canvas
PANEL = "#0f1115"
BORDER = "#1c1f26"
GRID = "#181b21"
FG = "#f7f8f8"
MUTED = "#8a8f98"         # Linear's muted label gray
GRAY_FILL = "#2a2e37"     # de-emphasised series - "gray is your best friend"
GRAY_FILL_2 = "#3d434f"   # second de-emphasised series, distinguishable in greyscale
ACCENT = "#5e6ad2"        # Linear purple - the ONE highlight colour
ACCENT_SOFT = "#8f97e8"
MARKET = "#c7ccd4"

WATERMARK = "@nineshoot"   # set to "" to disable

# single-hue sequential scale (dark -> Linear purple), colourblind-safe
HEAT_SCALE = [
    [0.0, "#111318"], [0.15, "#1c1f3d"], [0.35, "#2f3480"],
    [0.6, "#4750c4"], [0.8, "#5e6ad2"], [1.0, "#b8bdf5"],
]

_CARD_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: __W__px; height: __H__px; position: relative;
    background: radial-gradient(circle at 12% 8%, #14162a 0%, __BG__ 45%, #050506 100%);
    font-family: Inter, -apple-system, Arial, sans-serif;
    color: __FG__;
    padding: 40px;
    display: flex; flex-direction: column; gap: 22px;
  }
  .header h1 { font-size: 32px; font-weight: 700; margin-bottom: 4px; }
  .header p { font-size: 16px; color: __MUTED__; }
  .grid {
    flex: 1; display: grid;
    grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
    gap: 20px;
  }
  .grid.solo { grid-template-columns: 1fr; grid-template-rows: 1fr; }
  .card {
    position: relative; border-radius: 20px;
    background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.09);
    box-shadow: 0 24px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    padding: 20px 26px; display: flex; flex-direction: column; overflow: hidden;
  }
  .card .eyebrow {
    font-size: 12px; letter-spacing: 0.08em; color: __MUTED__;
    text-transform: uppercase; font-weight: 600; margin-bottom: 6px;
  }
  .card img { flex: 1; width: 100%; height: 100%; object-fit: contain; }
  .h2h-list { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 12px; }
  .h2h-row { display: flex; justify-content: space-between; font-size: 18px; }
  .h2h-row .date { color: __MUTED__; }
  .h2h-row .score { font-weight: 600; }
  .h2h-empty { margin: auto; color: __MUTED__; font-size: 16px; }
  .verdict { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; }
  .verdict .big {
    font-size: 78px; font-weight: 800; color: __ACCENT__;
    text-shadow: 0 0 44px rgba(94,106,210,0.55); text-align: center;
  }
  .verdict .sub { font-size: 20px; color: __MUTED__; }
  .watermark { position: absolute; bottom: 18px; right: 26px; font-size: 13px; color: __MUTED__; }
</style>
"""


def _card_css(width: int, height: int) -> str:
    return (_CARD_CSS
            .replace("__W__", str(width)).replace("__H__", str(height))
            .replace("__BG__", BG).replace("__FG__", FG)
            .replace("__MUTED__", MUTED).replace("__ACCENT__", ACCENT))


def _fig_png_datauri(fig: go.Figure, width: int, height: int) -> str:
    """Render a transparent-background Plotly figure to a data URI for HTML embedding."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    png = fig.to_image(format="png", width=width, height=height, scale=SUB_SCALE)
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _render_html(html: str, out: str, width: int = CANVAS_W, height: int = CANVAS_H) -> None:
    """Screenshot an HTML string to `out` with headless Chromium.

    ponytail: launches a fresh browser per call - simplest correct thing for
    a weekly batch job. If per-fixture loops become a bottleneck, share one
    browser/context across a run_league() call instead.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html, wait_until="networkidle")
        page.screenshot(path=out)
        browser.close()


def _page(body: str, width: int = CANVAS_W, height: int = CANVAS_H) -> str:
    return f"<html><head>{_card_css(width, height)}</head><body>{body}</body></html>"


def _insight_title(p_home: float, p_draw: float, p_away: float,
                   home: str, away: str) -> str:
    """Lead with the story, not the axis labels."""
    best = max(p_home, p_draw, p_away)
    if best == p_draw:
        return f"{home} vs {away} — too close to call"
    fav, dog, p = (home, away, p_home) if best == p_home else (away, home, p_away)
    if p >= 0.60:
        return f"{fav} strongly favoured over {dog}"
    if p >= 0.45:
        return f"{fav} favoured, but {dog} has a real shot"
    return f"{home} vs {away} — tight matchup"


def _heatmap_fig(grid, home: str, away: str) -> go.Figure:
    g = np.asarray(grid)[:4, :4]  # 0-3 goals is the interesting corner at this scale
    ai, aj = np.unravel_index(np.argmax(g), g.shape)  # most likely scoreline

    fig = go.Figure(go.Heatmap(
        z=g * 100, colorscale=HEAT_SCALE, showscale=False,
        text=np.round(g * 100, 1), texttemplate="%{text}",
        textfont=dict(size=15, color=FG),
        xgap=4, ygap=4))
    fig.update_xaxes(title=f"{away} goals", dtick=1, gridcolor=GRID, zeroline=False, color=FG)
    fig.update_yaxes(title=f"{home} goals", dtick=1, gridcolor=GRID, zeroline=False, color=FG)
    fig.add_annotation(x=aj, y=ai, text="most likely", showarrow=True,
                       arrowhead=2, arrowcolor=ACCENT_SOFT,
                       font=dict(color=FG, size=12), ax=30, ay=-30)
    fig.update_layout(
        font=dict(color=FG, family="Inter, -apple-system, Arial", size=13),
        margin=dict(l=55, r=10, t=10, b=45),
    )
    return fig


def _market_fig(pred: dict, market: dict | None, home: str, away: str) -> go.Figure:
    labels = [home, "Draw", away]
    model = [pred["p_home"] * 100, pred["p_draw"] * 100, pred["p_away"] * 100]

    best_idx = int(np.argmax(model))
    grays = [GRAY_FILL, GRAY_FILL_2]
    colors = [ACCENT if i == best_idx else grays[i % 2] for i in range(3)]

    fig = go.Figure()
    fig.add_bar(x=labels, y=model, name="Model", marker_color=colors,
               text=[f"{v:.0f}%" for v in model],
               textposition="inside", insidetextanchor="middle",
               textfont=dict(size=16, color=FG), width=0.55,
               cliponaxis=False, marker_line_width=0, showlegend=False)
    legend = None
    if market:
        mkt = [market["p_home"] * 100, market["p_draw"] * 100, market["p_away"] * 100]
        fig.add_scatter(x=labels, y=mkt, mode="markers+text", name="Market",
                        marker=dict(color=MARKET, size=14, symbol="diamond",
                                   line=dict(color=BG, width=2)),
                        text=[f"{v:.0f}%" for v in mkt],
                        textposition="top center", textfont=dict(size=12, color=FG))
        legend = dict(orientation="h", y=1.15, x=1, xanchor="right",
                      bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11))

    fig.update_yaxes(title="Probability %", range=[0, 108], gridcolor=GRID,
                     zeroline=False, color=FG)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, color=FG)
    fig.update_layout(
        font=dict(color=FG, family="Inter, -apple-system, Arial", size=13),
        margin=dict(l=55, r=10, t=10, b=35),
        legend=legend, bargap=0.4,
    )
    return fig


def _calibration_fig(log_path: str, bins: int) -> tuple[go.Figure | None, dict | None]:
    """Returns (fig, {'title','subtitle'}), or (None, None) if nothing to plot yet."""
    if not Path(log_path).exists() or Path(log_path).stat().st_size == 0:
        print("  [skip] calibration: predictions_log.csv doesn't exist yet")
        return None, None
    try:
        df = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        print("  [skip] calibration: predictions_log.csv is empty")
        return None, None
    df = df.dropna(subset=["result"])
    if df.empty:
        print("  [skip] calibration: no settled results yet")
        return None, None

    rows = []
    for _, r in df.iterrows():
        rows += [(r["p_home"], r["result"] == "H"),
                 (r["p_draw"], r["result"] == "D"),
                 (r["p_away"], r["result"] == "A")]
    c = pd.DataFrame(rows, columns=["p", "hit"])
    c["bucket"] = pd.cut(c["p"], np.linspace(0, 1, bins + 1))
    agg = c.groupby("bucket", observed=True).agg(
        pred=("p", "mean"), actual=("hit", "mean"), n=("hit", "size")).dropna()
    mae = float(np.average(np.abs(agg["pred"] - agg["actual"]), weights=agg["n"]))

    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                    line=dict(color=MUTED, dash="dash", width=1.5))
    fig.add_scatter(x=agg["pred"], y=agg["actual"], mode="markers+lines",
                    name="Model", line=dict(color=ACCENT, width=3),
                    marker=dict(color=ACCENT, size=13,
                               line=dict(color=BG, width=1.5)))
    fig.update_xaxes(title="Predicted probability", range=[0, 1],
                     gridcolor=GRID, zeroline=False, tickformat=".0%", color=FG)
    fig.update_yaxes(title="Actual hit rate", range=[0, 1],
                     gridcolor=GRID, zeroline=False, tickformat=".0%", color=FG)
    fig.update_layout(
        font=dict(color=FG, family="Inter, -apple-system, Arial", size=14),
        margin=dict(l=60, r=30, t=20, b=55),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
    )
    title = f"Model is off by {mae*100:.1f} points on average" if mae >= 0.02 \
        else "Model tracks its own confidence closely"
    subtitle = f"{int(agg['n'].sum())} settled outcomes, {bins} buckets"
    return fig, {"title": title, "subtitle": subtitle}


def _scoreboard_fig(stats: dict) -> go.Figure:
    total, wins, rate = stats["total"], stats["wins"], stats["win_rate"]

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number", value=rate * 100,
        number=dict(suffix="%", font=dict(size=130, color=ACCENT)),
        domain=dict(x=[0, 1], y=[0.4, 1]),
    ))
    fig.add_annotation(
        text=f"{wins} / {total} correct picks", xref="paper", yref="paper",
        x=0.5, y=0.3, xanchor="center", showarrow=False,
        font=dict(size=24, color=FG))

    order = ["H", "D", "A"]
    display_names = {"H": "Home picks", "D": "Draw picks", "A": "Away picks"}
    by = stats["by_outcome"]
    parts = []
    for k in order:
        if k in by and by[k]["n"] > 0:
            n, hits = int(by[k]["n"]), int(by[k]["hits"])
            parts.append(f"{display_names[k]}: {hits}/{n}")
    if parts:
        fig.add_annotation(
            text="   ·   ".join(parts), xref="paper", yref="paper",
            x=0.5, y=0.15, xanchor="center", showarrow=False,
            font=dict(size=17, color=MUTED))

    fig.update_layout(
        font=dict(color=FG, family="Inter, -apple-system, Arial"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _single_card_body(eyebrow: str, img_uri: str, title: str, subtitle: str) -> str:
    return f"""
    <div class="header"><h1>{title}</h1><p>{subtitle}</p></div>
    <div class="grid solo">
      <div class="card">
        <div class="eyebrow">{eyebrow}</div>
        <img src="{img_uri}">
      </div>
    </div>
    <div class="watermark">{WATERMARK}</div>
    """


def match_dashboard(pred: dict, market: dict | None, home: str, away: str,
                    h2h: list[dict], out: str) -> None:
    """One glass-card 2x2 image per fixture: model vs market, scoreline
    heatmap, head-to-head history, and the verdict."""
    market_uri = _fig_png_datauri(_market_fig(pred, market, home, away), 860, 460)
    heat_uri = _fig_png_datauri(_heatmap_fig(pred["_grid"], home, away), 860, 460)

    if h2h:
        h2h_html = "".join(
            f'<div class="h2h-row"><span class="date">{r["date"]}</span>'
            f'<span class="score">{r["line"]}</span></div>' for r in h2h)
    else:
        h2h_html = '<div class="h2h-empty">No meetings in the last 2 seasons</div>'

    best = max(pred["p_home"], pred["p_draw"], pred["p_away"])
    if best == pred["p_draw"]:
        verdict, verdict_p = "Draw", pred["p_draw"]
    elif best == pred["p_home"]:
        verdict, verdict_p = home, pred["p_home"]
    else:
        verdict, verdict_p = away, pred["p_away"]

    title = _insight_title(pred["p_home"], pred["p_draw"], pred["p_away"], home, away)

    body = f"""
    <div class="header"><h1>{title}</h1><p>{home} vs {away}</p></div>
    <div class="grid">
      <div class="card">
        <div class="eyebrow">Model vs Market</div>
        <img src="{market_uri}">
      </div>
      <div class="card">
        <div class="eyebrow">Scoreline Probability (0-3 goals)</div>
        <img src="{heat_uri}">
      </div>
      <div class="card">
        <div class="eyebrow">Head-to-Head — Last 2 Seasons</div>
        <div class="h2h-list">{h2h_html}</div>
      </div>
      <div class="card">
        <div class="eyebrow">Verdict</div>
        <div class="verdict">
          <div class="big">{verdict}</div>
          <div class="sub">{verdict_p*100:.0f}% probability</div>
        </div>
      </div>
    </div>
    <div class="watermark">{WATERMARK}</div>
    """
    _render_html(_page(body), out)


def calibration_curve(log_path: str, out: str, bins: int = 10) -> None:
    """Reliability plot from settled predictions (needs results filled in)."""
    fig, meta = _calibration_fig(log_path, bins)
    if fig is None:
        return
    img_uri = _fig_png_datauri(fig, 1600, 760)
    body = _single_card_body("Calibration", img_uri, meta["title"], meta["subtitle"])
    _render_html(_page(body), out)


def accuracy_scoreboard(stats: dict, out: str) -> None:
    """Linear-style KPI card: win count + win rate, big and readable."""
    img_uri = _fig_png_datauri(_scoreboard_fig(stats), 1500, 700)
    body = _single_card_body("Scoreboard", img_uri, "Model Hit Rate to Date",
                             f"{stats['total']} settled predictions")
    _render_html(_page(body), out)
