"""
Static, high-res PNGs for the video pipeline. Three outputs:

  1. match_dashboard    - one printed sheet per fixture: model-vs-market
                          bars, scoreline heatmap, head-to-head, verdict.
  2. calibration_curve  - reliability plot built from predictions_log.csv
  3. accuracy_scoreboard - win-count KPI sheet (pure HTML, no chart)

Rendering pipeline: Plotly draws the charts, kaleido rasterizes each to a
transparent PNG, the PNGs get embedded into an HTML sheet, and Playwright's
headless Chromium screenshots the assembled page.

Design language: the `mono-color` editorial print system
(github.com/yanliudesign/mono-color-skill) applied to a data sheet:

  - Substrate: pale beige paper, never a screen-dark canvas.
  - Exactly TWO inks: charcoal carries 70-85% of the marks, signal red is
    the single accent reserved for the story (the favoured outcome).
    Everything else is a *screen* (a tint) of charcoal, not a new colour.
  - Composition `ruled_information`: headline and facts share one rule,
    blocks are separated by hairlines, not boxes. No glass, no shadows,
    no gradients, no rounded corners - a print sheet has none of those.
  - Typography `programmatic`: grotesk display, mono support type,
    numerals as anchors. Titles are conclusions, not axis labels.
  - One focal event per sheet (the verdict), one release zone (open paper).
  - Because colour is a tint ladder plus one accent, every chart still
    reads in greyscale, and labels are always present - colourblind-safe
    by construction.
"""
from __future__ import annotations
import base64
from html import escape as e
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from playwright.sync_api import sync_playwright

CANVAS_W, CANVAS_H = 1920, 1080
SUB_SCALE = 2  # oversample embedded Plotly PNGs so they stay crisp at CSS display size

# ---- mono-color: substrate_pale_beige + palette_charcoal_signal_red ----
PAPER = "#F5F1E8"      # substrate - not an ink
INK = "#30343A"        # ink_charcoal   - dominant ink, 70-85% of the marks
ACCENT = "#C83232"     # ink_signal_red - the ONE accent, reserved for the story
MUTED = "#6E6960"      # 60% charcoal screen, for mono microcopy
HAIR = "#C8C1B5"       # hairline rule, paper-toned
SCREEN_1 = "#B4B1AB"   # ~32% charcoal screen - de-emphasised series
SCREEN_2 = "#8E8B85"   # ~55% charcoal screen - second de-emphasised series

WATERMARK = "@nineshoot"   # set to "" to disable

# Charcoal screen ladder for the scoreline plate. Capped at a 45% screen so
# solid-charcoal numerals stay >=4.5:1 on the darkest cell - in print terms,
# image plates are screened and solid ink is reserved for type.
HEAT_SCALE = [
    [0.0, PAPER], [0.2, "#E5E2DA"], [0.4, "#D2CFC9"],
    [0.6, "#C0BEB9"], [0.8, "#AEADA9"], [1.0, "#9C9B9A"],
]

DISPLAY = "Archivo, 'Ubuntu Sans', 'DejaVu Sans', Helvetica, sans-serif"
MONO = "'Space Mono', 'DejaVu Sans Mono', 'Ubuntu Mono', monospace"
PLOT_FONT = "DejaVu Sans Mono, monospace"  # kaleido sees system fonts only

_CSS = """
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: __W__px; height: __H__px; position: relative; overflow: hidden;
    background: __PAPER__; color: __INK__;
    font-family: __DISPLAY__;
    padding: 52px 64px 40px;
    display: flex; flex-direction: column;
    -webkit-font-smoothing: antialiased;
  }
  /* paper grain - the substrate is printed on, not emitted */
  body::after {
    content: ""; position: absolute; inset: 0; pointer-events: none; opacity: 0.055;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }
  .micro {
    font-family: __MONO__; font-size: 12px; letter-spacing: 0.14em;
    text-transform: uppercase; color: __MUTED__;
    display: flex; justify-content: space-between; align-items: baseline;
  }
  .micro .ix { color: __ACCENT__; font-weight: 700; }
  .rule { height: 3px; background: __INK__; margin: 10px 0 22px; flex-shrink: 0; }
  h1 {
    font-size: 52px; font-weight: 700; line-height: 1.04;
    letter-spacing: -0.018em; max-width: 1240px;
  }
  .sub {
    font-family: __MONO__; font-size: 14px; letter-spacing: 0.2em;
    text-transform: uppercase; color: __INK__; margin-top: 12px;
  }
  .blocks {
    flex: 1; min-height: 0; display: grid; margin-top: 26px;
    grid-template-columns: 1fr 1fr; grid-template-rows: 1.1fr 0.9fr;
    column-gap: 56px; row-gap: 22px;
  }
  .blocks.solo { grid-template-columns: 1fr; grid-template-rows: 1fr; }
  .block {
    min-height: 0; display: flex; flex-direction: column;
    border-top: 1px solid __HAIR__; padding-top: 12px;
  }
  .block > .micro { flex-shrink: 0; margin-bottom: 8px; }
  .block img { flex: 1; min-height: 0; width: 100%; object-fit: contain; }

  /* head-to-head: a ruled table, the way a results page is actually set */
  .h2h { flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: center; }
  .h2h-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 11px 2px; border-bottom: 1px dotted __HAIR__;
    font-family: __MONO__; font-size: 17px;
  }
  .h2h-row .date { color: __MUTED__; font-size: 13px; letter-spacing: 0.08em; }
  .h2h-empty { margin: auto; font-family: __MONO__; font-size: 15px; color: __MUTED__; }

  /* the focal event */
  .verdict { flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: center; }
  .verdict .name {
    font-weight: 800; line-height: 0.92; letter-spacing: -0.035em;
    text-transform: uppercase; color: __ACCENT__;
  }
  .verdict .plate { height: 12px; background: __ACCENT__; margin: 16px 0 14px; width: 62%; }
  .verdict .pct {
    font-size: 92px; font-weight: 700; line-height: 1;
    letter-spacing: -0.04em; font-variant-numeric: tabular-nums;
    display: flex; align-items: baseline; gap: 4px;
  }
  .verdict .pct em { font-style: normal; font-size: 40px; font-weight: 500; color: __MUTED__; }
  .verdict .note {
    font-family: __MONO__; font-size: 12px; letter-spacing: 0.16em;
    text-transform: uppercase; color: __MUTED__; margin-top: 12px;
  }

  /* KPI sheet */
  .kpi {
    flex: 1; min-height: 0; display: grid; align-items: center;
    grid-template-columns: auto 1fr; gap: 96px;
  }
  .kpi .big {
    font-size: 300px; font-weight: 800; line-height: 0.82;
    letter-spacing: -0.055em; color: __ACCENT__;
    font-variant-numeric: tabular-nums; padding: 34px 52px 52px;
    /* the numeral prints over a screened halftone plate */
    background-image: radial-gradient(__HAIR__ 1.2px, transparent 1.3px);
    background-size: 8px 8px;
  }
  .kpi .big em { font-style: normal; font-size: 124px; }
  .kpi table { width: 100%; border-collapse: collapse; font-family: __MONO__; font-size: 26px; }
  .kpi td { padding: 22px 0; border-bottom: 1px solid __HAIR__; }
  .kpi td.n { text-align: right; font-weight: 700; }
  .kpi caption {
    text-align: left; font-family: __MONO__; font-size: 12px; letter-spacing: 0.16em;
    text-transform: uppercase; color: __MUTED__; padding-bottom: 14px;
  }

  footer { flex-shrink: 0; margin-top: 20px; border-top: 1px solid __HAIR__; padding-top: 10px; }
</style>
"""


def _css(width: int, height: int) -> str:
    subs = {"__W__": str(width), "__H__": str(height), "__PAPER__": PAPER,
            "__INK__": INK, "__ACCENT__": ACCENT, "__MUTED__": MUTED,
            "__HAIR__": HAIR, "__DISPLAY__": DISPLAY, "__MONO__": MONO}
    css = _CSS
    for k, v in subs.items():
        css = css.replace(k, v)
    return css


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
        page.evaluate("document.fonts.ready")  # webfonts land before the shutter
        page.screenshot(path=out)
        browser.close()


def _sheet(kicker: str, meta: str, title: str, sub: str, blocks: str,
           solo: bool = False, width: int = CANVAS_W, height: int = CANVAS_H) -> str:
    """One printed sheet: mono masthead, heavy rule, headline, ruled blocks, colophon."""
    return f"""<html><head>{_css(width, height)}</head><body>
      <div class="micro"><span><span class="ix">■</span>&nbsp;&nbsp;{kicker}</span><span>{meta}</span></div>
      <div class="rule"></div>
      <h1>{title}</h1>
      <div class="sub">{sub}</div>
      <div class="blocks{' solo' if solo else ''}">{blocks}</div>
      <footer><div class="micro"><span>{WATERMARK}</span>
        <span>Dixon–Coles · two-ink print</span></div></footer>
    </body></html>"""


def _block(n: str, label: str, meta: str, inner: str) -> str:
    return (f'<div class="block"><div class="micro"><span><span class="ix">{n}</span>'
            f'&nbsp;&nbsp;{label}</span><span>{meta}</span></div>{inner}</div>')


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


def _axes(fig: go.Figure) -> None:
    """One ink, hairline rules, mono numerals - no chartjunk on paper."""
    fig.update_xaxes(showgrid=False, zeroline=False, color=INK,
                     linecolor=INK, linewidth=1, ticks="outside",
                     tickcolor=HAIR, ticklen=5)
    fig.update_yaxes(gridcolor=HAIR, gridwidth=1, zeroline=False, color=INK,
                     showline=False, ticks="")
    fig.update_layout(font=dict(color=INK, family=PLOT_FONT, size=13))


def _heatmap_fig(grid, home: str, away: str) -> go.Figure:
    g = np.asarray(grid)[:4, :4]  # 0-3 goals is the interesting corner at this scale
    ai, aj = np.unravel_index(np.argmax(g), g.shape)  # most likely scoreline

    fig = go.Figure(go.Heatmap(
        z=g * 100, colorscale=HEAT_SCALE, showscale=False,
        text=g * 100, texttemplate="%{text:.1f}",
        textfont=dict(size=15, color=INK, family=PLOT_FONT),
        xgap=5, ygap=5))
    # the peak cell is registered in the accent ink, not by being darker
    fig.add_shape(type="rect", x0=aj - 0.5, x1=aj + 0.5, y0=ai - 0.5, y1=ai + 0.5,
                  line=dict(color=ACCENT, width=3))
    fig.add_annotation(x=aj, y=ai + 0.5, text="MOST LIKELY", showarrow=False,
                       bgcolor=PAPER, borderpad=3,
                       font=dict(color=ACCENT, size=11, family=PLOT_FONT))
    _axes(fig)
    fig.update_xaxes(title=f"{away} goals".upper(), dtick=1, showline=False, ticks="")
    fig.update_yaxes(title=f"{home} goals".upper(), dtick=1, gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=62, r=10, t=8, b=46))
    return fig


def _market_fig(pred: dict, market: dict | None, home: str, away: str) -> go.Figure:
    labels = [home.upper(), "DRAW", away.upper()]
    model = [pred["p_home"] * 100, pred["p_draw"] * 100, pred["p_away"] * 100]

    best_idx = int(np.argmax(model))
    screens = [SCREEN_1, SCREEN_2]
    # one accent for the story; everything else is a screen of the same ink
    colors = [ACCENT if i == best_idx else screens[i % 2] for i in range(3)]

    fig = go.Figure()
    fig.add_bar(x=labels, y=model, name="Model", marker_color=colors,
                text=[f"{v:.0f}%" for v in model],
                textposition="outside",  # numerals sit on the paper, always legible
                textfont=dict(size=17, color=INK, family=PLOT_FONT),
                width=0.5, cliponaxis=False, marker_line_width=0, showlegend=False)
    if market:
        mkt = [market["p_home"] * 100, market["p_draw"] * 100, market["p_away"] * 100]
        # marker only: a second number this close to the bar's own label just
        # collides with it, and the gap between diamond and bar is the point
        fig.add_scatter(x=labels, y=mkt, mode="markers", name="MARKET",
                        marker=dict(color=PAPER, size=16, symbol="diamond",
                                    line=dict(color=INK, width=2)))

    _axes(fig)
    fig.update_yaxes(title="PROBABILITY %", range=[0, 112])
    fig.update_layout(margin=dict(l=64, r=10, t=16, b=38),
                      showlegend=False, bargap=0.45)
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
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="PERFECT CALIBRATION",
                    line=dict(color=SCREEN_2, dash="dash", width=1.5))
    fig.add_scatter(x=agg["pred"], y=agg["actual"], mode="markers+lines",
                    name="MODEL", line=dict(color=ACCENT, width=3),
                    marker=dict(color=ACCENT, size=12,
                                line=dict(color=PAPER, width=2)))
    _axes(fig)
    fig.update_xaxes(title="PREDICTED PROBABILITY", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="ACTUAL HIT RATE", range=[0, 1], tickformat=".0%")
    fig.update_layout(
        margin=dict(l=72, r=30, t=30, b=58),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=MUTED, size=11, family=PLOT_FONT)))
    title = f"Model is off by {mae*100:.1f} points on average" if mae >= 0.02 \
        else "Model tracks its own confidence closely"
    subtitle = f"{int(agg['n'].sum())} settled outcomes · {bins} buckets"
    return fig, {"title": title, "subtitle": subtitle}


def match_dashboard(pred: dict, market: dict | None, home: str, away: str,
                    h2h: list[dict], out: str,
                    league: str = "", date: str = "") -> None:
    """One printed sheet per fixture: model vs market, scoreline plate,
    head-to-head, and the verdict as the focal event."""
    # matches the block aspect so `object-fit: contain` has nothing to letterbox
    market_uri = _fig_png_datauri(_market_fig(pred, market, home, away), 868, 380)
    heat_uri = _fig_png_datauri(_heatmap_fig(pred["_grid"], home, away), 868, 380)

    if h2h:
        h2h_html = "".join(
            f'<div class="h2h-row"><span class="date">{e(r["date"])}</span>'
            f'<span class="score">{e(r["line"])}</span></div>' for r in h2h)
    else:
        h2h_html = '<div class="h2h-empty">No meetings in the last 2 seasons</div>'

    best = max(pred["p_home"], pred["p_draw"], pred["p_away"])
    if best == pred["p_draw"]:
        verdict, verdict_p = "Draw", pred["p_draw"]
    elif best == pred["p_home"]:
        verdict, verdict_p = home, pred["p_home"]
    else:
        verdict, verdict_p = away, pred["p_away"]
    # display type is set to the word, not the other way round
    name_px = 66 if len(verdict) <= 9 else 52 if len(verdict) <= 14 else 40

    blocks = (
        _block("01", "Model vs Market", "bars = model · ◇ = market", f'<img src="{market_uri}">')
        + _block("02", "Scoreline Plate", "0–3 goals · %", f'<img src="{heat_uri}">')
        + _block("03", "Head to Head", "last 2 seasons", f'<div class="h2h">{h2h_html}</div>')
        + _block("04", "Verdict", "model pick", f"""
            <div class="verdict">
              <div class="name" style="font-size:{name_px}px">{e(verdict)}</div>
              <div class="plate"></div>
              <div class="pct">{verdict_p*100:.0f}<em>%</em></div>
              <div class="note">probability · most likely outcome</div>
            </div>""")
    )
    meta = f"{date} · match forecast" if date else "match forecast"
    _render_html(_sheet(
        kicker=e(league) or "Match forecast",
        meta=e(meta),
        title=e(_insight_title(pred["p_home"], pred["p_draw"], pred["p_away"], home, away)),
        sub=f"{e(home)} &nbsp;vs&nbsp; {e(away)}",
        blocks=blocks), out)


def calibration_curve(log_path: str, out: str, bins: int = 10) -> None:
    """Reliability plot from settled predictions (needs results filled in)."""
    fig, meta = _calibration_fig(log_path, bins)
    if fig is None:
        return
    img_uri = _fig_png_datauri(fig, 1660, 700)
    _render_html(_sheet(
        kicker="Calibration", meta="reliability plate",
        title=meta["title"], sub=meta["subtitle"],
        blocks=_block("01", "Predicted vs Actual", "all settled outcomes",
                      f'<img src="{img_uri}">'),
        solo=True), out)


def accuracy_scoreboard(stats: dict, out: str) -> None:
    """Hit-rate sheet. Pure HTML - a number and a ruled table need no chart."""
    names = {"H": "Home picks", "D": "Draw picks", "A": "Away picks"}
    by = stats["by_outcome"]
    rows = "".join(
        f'<tr><td>{names[k]}</td><td class="n">{int(by[k]["hits"])} / {int(by[k]["n"])}</td></tr>'
        for k in ("H", "D", "A") if k in by and by[k]["n"] > 0)

    _render_html(_sheet(
        kicker="Scoreboard", meta="hit rate to date",
        title="Model Hit Rate to Date",
        sub=f"{stats['total']} settled predictions · {stats['wins']} correct",
        blocks=_block("01", "Correct Picks", "1X2 · share", f"""
            <div class="kpi">
              <div class="big">{stats['win_rate']*100:.0f}<em>%</em></div>
              <table><caption>By picked outcome</caption>{rows}</table>
            </div>"""),
        solo=True), out)


def _demo() -> None:
    """`python -m src.viz [outdir]` - self-check plus a preview of every sheet."""
    import sys
    import tempfile

    # the four verdict branches, which are the only real logic on this page
    assert "strongly favoured" in _insight_title(.70, .20, .10, "A", "B")
    assert "real shot" in _insight_title(.50, .20, .30, "A", "B")
    assert "too close to call" in _insight_title(.30, .40, .30, "A", "B")
    assert "tight matchup" in _insight_title(.40, .20, .40, "A", "B")

    # exactly one accent ink per chart - the whole point of a two-ink system
    pred = {"p_home": .642, "p_draw": .221, "p_away": .137}
    bars = _market_fig(pred, None, "A", "B").data[0].marker.color
    assert list(bars).count(ACCENT) == 1, bars

    grid = np.outer([.30, .36, .22, .09, .03], [.28, .35, .22, .11, .04])
    pred["_grid"] = grid / grid.sum()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)
    match_dashboard(pred, {"p_home": .60, "p_draw": .24, "p_away": .16}, "Man City",
                    "Bournemouth",
                    [{"date": "2026-02-11", "line": "Man City 2-1 Bournemouth"},
                     {"date": "2025-09-21", "line": "Bournemouth 0-3 Man City"}],
                    str(out / "sheet_match.png"),
                    league="Premier League", date="2026-08-17")
    accuracy_scoreboard(
        {"total": 148, "wins": 79, "win_rate": .534,
         "by_outcome": {"H": {"n": 71, "hits": 44}, "D": {"n": 22, "hits": 5},
                        "A": {"n": 55, "hits": 30}}},
        str(out / "sheet_scoreboard.png"))
    print(f"ok - sheets in {out}")


if __name__ == "__main__":
    _demo()
