"""
Static, high-res PNGs for the video pipeline. Three outputs:

  1. match_dashboard    - one printed sheet per fixture: model-vs-market
                          bars, scoreline heatmap, head-to-head, verdict.
  2. calibration_curve  - reliability plot built from predictions_log.csv
  3. accuracy_scoreboard - win-count KPI sheet (pure HTML, no chart)

Rendering pipeline: the sheet is an HTML page that Playwright's headless
Chromium screenshots. The Plotly charts are drawn by plotly.js inside that
same page - no kaleido, no second browser to install, and the charts land
as vector SVG in the DOM rather than a rasterised PNG in an <img>.

Design language: the `diagram-design` editorial system
(github.com/cathrynlavery/diagram-design), default light skin:

  - Tokens by semantic role, never by hex: paper, ink, muted, soft, rule,
    accent. White-smoke paper, jet-black ink, atomic-tangerine accent.
  - The focal rule: accent marks at most ONE thing per block - the picked
    outcome. Everything else is ink/muted/soft. Accent on the block
    indices too would erase the signal, so those are muted.
  - Borders, never shadows. Radius 6-8px or none. No gradients, no glow.
  - Typography: Instrument Serif for the headline, Geist for names, Geist
    Mono for technical content only (axis values, dates, scores).
  - Clean paper - the dotted-paper variant is opt-in and not used here.
  - 4px grid: every size, coordinate and gap divides by 4.

The sheets are composites, not one of the skill's 39 diagram types - a
scoreline matrix and a KPI have no type there, and it explicitly sends
lists to a table. So this takes the design system, not the type grammar.
"""
from __future__ import annotations
from functools import cache
from html import escape as e
from itertools import count
from pathlib import Path
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from playwright.sync_api import sync_playwright

CANVAS_W, CANVAS_H = 1920, 1080

# ---- diagram-design tokens, default light skin ----
PAPER = "#f5f5f5"      # white-smoke - page background
INK = "#2d3142"        # jet-black   - primary text and stroke
MUTED = "#4f5d75"      # blue-slate  - secondary text, non-focal series
SOFT = "#7a8399"       # sublabels
RULE = "rgba(45,49,66,0.12)"    # hairline borders
RULE_SOLID = "#bfc0c0"          # silver - stronger baselines
ACCENT = "#eb6c36"              # atomic-tangerine - 1 focal element per block
ACCENT_TINT = "rgba(235,108,54,0.12)"
GRID_LINE = "rgba(45,49,66,0.08)"   # y-gridlines, per the bar-chart spec
BASELINE = "rgba(45,49,66,0.25)"    # x-axis baseline
SERIES = "rgba(79,93,117,0.15)"     # non-focal bar fill

WATERMARK = "@nineshoot"   # set to "" to disable

# Ink tints for the scoreline matrix. Capped well short of full ink so the
# ink numerals stay legible on the densest cell; the peak is marked with an
# accent border rather than by being darkest, keeping the one-focal rule.
HEAT_SCALE = [
    [0.0, PAPER], [0.2, "#e6e7ea"], [0.4, "#d3d5db"],
    [0.6, "#c0c3cc"], [0.8, "#adb1bd"], [1.0, "#9aa0ae"],
]

DISPLAY = "Geist, 'Ubuntu Sans', 'DejaVu Sans', Helvetica, sans-serif"
SERIF = "'Instrument Serif', Georgia, serif"
MONO = "'Geist Mono', 'DejaVu Sans Mono', 'Ubuntu Mono', monospace"
PLOT_FONT = "DejaVu Sans Mono, monospace"  # a font the renderer always has

_CSS = """
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
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
  /* clean paper - the dotted variant is opt-in and would fight a video frame */
  .micro {
    font-family: __MONO__; font-size: 12px; font-weight: 500;
    letter-spacing: 0.18em; text-transform: uppercase; color: __MUTED__;
    display: flex; justify-content: space-between; align-items: baseline;
  }
  .micro .ix { color: __SOFT__; font-weight: 600; }
  .rule { height: 1px; background: __RULE_SOLID__; margin: 12px 0 24px; flex-shrink: 0; }
  h1 {
    font-family: __SERIF__; font-size: 56px; font-weight: 400;
    line-height: 1.08; letter-spacing: 0; max-width: 1280px;
  }
  .sub {
    font-family: __MONO__; font-size: 12px; letter-spacing: 0.18em;
    text-transform: uppercase; color: __MUTED__; margin-top: 12px;
  }
  .blocks {
    flex: 1; min-height: 0; display: grid; margin-top: 28px;
    grid-template-columns: 1fr 1fr; grid-template-rows: 1.1fr 0.9fr;
    column-gap: 56px; row-gap: 24px;
  }
  .blocks.solo { grid-template-columns: 1fr; grid-template-rows: 1fr; }
  .block {
    min-height: 0; display: flex; flex-direction: column;
    border-top: 1px solid __RULE__; padding-top: 12px;
  }
  .block > .micro { flex-shrink: 0; margin-bottom: 8px; }
  .block .plot {
    flex: 1; min-height: 0; width: 100%;
    display: flex; align-items: center; justify-content: center;
  }

  /* head-to-head: a table, which is what the skill says a list should be */
  .h2h { flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: center; }
  .h2h-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 12px 4px; border-bottom: 1px solid __RULE__;
  }
  .h2h-row .score { font-size: 16px; font-weight: 600; }
  .h2h-row .date {
    font-family: __MONO__; font-size: 12px; color: __SOFT__; letter-spacing: 0.08em;
  }
  .h2h-empty { margin: auto; font-family: __MONO__; font-size: 12px; color: __SOFT__; }

  /* the one focal element on this sheet */
  .verdict { flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: center; }
  .verdict .name {
    font-weight: 600; line-height: 0.96; letter-spacing: -0.02em; color: __ACCENT__;
  }
  .verdict .plate { height: 4px; background: __ACCENT__; margin: 20px 0 16px; width: 56%; }
  .verdict .pct {
    font-family: __SERIF__; font-size: 96px; font-weight: 400; line-height: 1;
    font-variant-numeric: tabular-nums; color: __INK__;
    display: flex; align-items: baseline; gap: 4px;
  }
  .verdict .pct em { font-family: __DISPLAY__; font-style: normal; font-size: 32px; font-weight: 500; color: __MUTED__; }
  .verdict .note {
    font-family: __MONO__; font-size: 12px; letter-spacing: 0.18em;
    text-transform: uppercase; color: __SOFT__; margin-top: 16px;
  }

  /* KPI sheet */
  .kpi {
    flex: 1; min-height: 0; display: grid; align-items: center;
    grid-template-columns: auto 1fr; gap: 96px;
  }
  .kpi .big {
    font-family: __SERIF__; font-size: 280px; font-weight: 400; line-height: 0.88;
    color: __ACCENT__; font-variant-numeric: tabular-nums; padding: 32px 48px;
    border: 1px solid __RULE__; border-radius: 8px;   /* borders, never shadows */
  }
  .kpi .big em { font-family: __DISPLAY__; font-style: normal; font-size: 96px; font-weight: 500; color: __MUTED__; }
  .kpi table { width: 100%; border-collapse: collapse; font-size: 24px; }
  .kpi td { padding: 24px 0; border-bottom: 1px solid __RULE__; }
  .kpi td.n { text-align: right; font-family: __MONO__; font-weight: 600; }
  .kpi caption {
    text-align: left; font-family: __MONO__; font-size: 12px; letter-spacing: 0.18em;
    text-transform: uppercase; color: __SOFT__; padding-bottom: 16px;
  }

  footer { flex-shrink: 0; margin-top: 20px; border-top: 1px solid __RULE__; padding-top: 12px; }
</style>
"""


def _css() -> str:
    subs = {"__W__": str(CANVAS_W), "__H__": str(CANVAS_H), "__PAPER__": PAPER,
            "__INK__": INK, "__ACCENT__": ACCENT, "__MUTED__": MUTED,
            "__SOFT__": SOFT, "__RULE__": RULE, "__RULE_SOLID__": RULE_SOLID,
            "__DISPLAY__": DISPLAY, "__SERIF__": SERIF, "__MONO__": MONO}
    css = _CSS
    for k, v in subs.items():
        css = css.replace(k, v)
    return css


@cache
def _plotlyjs() -> str:
    """plotly.min.js as it ships inside the installed plotly package.

    Inlined rather than pulled from a CDN so a render never depends on the
    network, and cached so the 4.6MB read happens once per process.
    """
    return (Path(plotly.__file__).parent / "package_data"
            / "plotly.min.js").read_text(encoding="utf-8")


_plot_ids = count(1)


def _fig_html(fig: go.Figure, width: int, height: int) -> str:
    """A Plotly figure as a div the page draws itself.

    The sheet is already going through headless Chromium, so the chart is
    drawn there too — no kaleido, no second browser, and the chart lands as
    vector SVG in the DOM instead of a rasterised PNG pasted into an <img>.
    """
    div = f"plot{next(_plot_ids)}"
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      width=width, height=height, autosize=False)
    return (f'<div class="plot" id="{div}"></div>'
            f'<script>window.__plots.push(Plotly.newPlot({div!r},'
            f'{fig.to_json()},{{}},{{staticPlot:true}}));</script>')


def _render_html(html: str, out: str) -> None:
    """Screenshot an HTML string to `out` with headless Chromium.

    ponytail: launches a fresh browser per call - simplest correct thing for
    a weekly batch job. If per-fixture loops become a bottleneck, share one
    browser/context across a run_league() call instead.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": CANVAS_W, "height": CANVAS_H})
        # "load" means every script has run, so __plots is complete - and it
        # is legitimately empty on the sheets that carry no chart at all
        page.set_content(html, wait_until="load")
        page.evaluate("Promise.all(window.__plots || [])")
        page.evaluate("document.fonts.ready")  # webfonts land before the shutter
        page.screenshot(path=out)
        browser.close()


def _sheet(kicker: str, meta: str, title: str, sub: str, blocks: str,
           solo: bool = False) -> str:
    """One printed sheet: mono masthead, heavy rule, headline, ruled blocks, colophon."""
    return f"""<html><head>{_css()}
      <script>window.__plots=[];</script>
      <script>{_plotlyjs()}</script></head><body>
      <div class="micro"><span><span class="ix">■</span>&nbsp;&nbsp;{kicker}</span><span>{meta}</span></div>
      <div class="rule"></div>
      <h1>{title}</h1>
      <div class="sub">{sub}</div>
      <div class="blocks{' solo' if solo else ''}">{blocks}</div>
      <footer><div class="micro"><span>{WATERMARK}</span>
        <span>Dixon–Coles · diagram-design</span></div></footer>
    </body></html>"""


def _block(n: str, label: str, meta: str, inner: str) -> str:
    return (f'<div class="block"><div class="micro"><span><span class="ix">{n}</span>'
            f'&nbsp;&nbsp;{label}</span><span>{meta}</span></div>{inner}</div>')


def _best(pred: dict) -> int:
    """Index of the most likely outcome: 0 home, 1 draw, 2 away.

    A tie reads as a draw, which is what "too close to call" means - and it
    keeps the headline, the focal bar and the verdict picking the same
    outcome, which three separate argmaxes did not guarantee.
    """
    ps = [pred["p_home"], pred["p_draw"], pred["p_away"]]
    return 1 if ps[1] == max(ps) else max(range(3), key=ps.__getitem__)


def _insight_title(pred: dict, home: str, away: str) -> str:
    """Lead with the story, not the axis labels."""
    i = _best(pred)
    if i == 1:
        return f"{home} vs {away} — too close to call"
    fav, dog, p = ((home, away, pred["p_home"]) if i == 0
                   else (away, home, pred["p_away"]))
    if p >= 0.60:
        return f"{fav} strongly favoured over {dog}"
    if p >= 0.45:
        return f"{fav} favoured, but {dog} has a real shot"
    return f"{home} vs {away} — tight matchup"


def _axes(fig: go.Figure) -> None:
    """Faint gridlines, a solid baseline, mono numerals - the skill's axis spec."""
    fig.update_xaxes(showgrid=False, zeroline=False, color=MUTED,
                     linecolor=BASELINE, linewidth=1, ticks="outside",
                     tickcolor=RULE, ticklen=4)
    fig.update_yaxes(gridcolor=GRID_LINE, gridwidth=1, zeroline=False, color=MUTED,
                     showline=False, ticks="")
    fig.update_layout(font=dict(color=MUTED, family=PLOT_FONT, size=12))


def _heatmap_fig(grid, home: str, away: str) -> go.Figure:
    g = np.asarray(grid)[:4, :4]  # 0-3 goals is the interesting corner at this scale
    ai, aj = np.unravel_index(np.argmax(g), g.shape)  # most likely scoreline

    fig = go.Figure(go.Heatmap(
        z=g * 100, colorscale=HEAT_SCALE, showscale=False,
        text=g * 100, texttemplate="%{text:.1f}",
        textfont=dict(size=16, color=INK, family=PLOT_FONT),
        xgap=4, ygap=4))
    # the peak cell is the one focal element here - marked by an accent
    # border, not by being the darkest tint
    fig.add_shape(type="rect", x0=aj - 0.5, x1=aj + 0.5, y0=ai - 0.5, y1=ai + 0.5,
                  line=dict(color=ACCENT, width=2), fillcolor=ACCENT_TINT)
    fig.add_annotation(x=aj, y=ai + 0.5, text="MOST LIKELY", showarrow=False,
                       bgcolor=PAPER, borderpad=4,
                       font=dict(color=ACCENT, size=12, family=PLOT_FONT))
    _axes(fig)
    fig.update_xaxes(title=f"{away} goals".upper(), dtick=1, showline=False, ticks="")
    fig.update_yaxes(title=f"{home} goals".upper(), dtick=1, gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=62, r=10, t=8, b=46))
    return fig


def _market_fig(pred: dict, market: dict | None, home: str, away: str) -> go.Figure:
    labels = [home.upper(), "DRAW", away.upper()]
    model = [pred["p_home"] * 100, pred["p_draw"] * 100, pred["p_away"] * 100]

    # exactly one focal bar; the rest take the non-focal series treatment
    focal = _best(pred)
    fills, lines, label_ink = zip(*(
        (ACCENT_TINT, ACCENT, ACCENT) if i == focal else (SERIES, MUTED, MUTED)
        for i in range(3)))

    fig = go.Figure()
    fig.add_bar(x=labels, y=model, name="Model", marker_color=fills,
                width=0.56, cliponaxis=False, showlegend=False,
                marker_line=dict(color=lines, width=1))

    mkt = None
    if market:
        mkt = [market["p_home"] * 100, market["p_draw"] * 100, market["p_away"] * 100]
        # marker only: a second number this close to the bar's own label just
        # collides with it, and the gap between diamond and bar is the point
        fig.add_scatter(x=labels, y=mkt, mode="markers", name="MARKET",
                        marker=dict(color=PAPER, size=14, symbol="diamond",
                                    line=dict(color=MUTED, width=1.5)))

    # The value label clears whichever is higher, the bar top or the market
    # diamond - when the model agrees with the market they land on the same
    # spot, and a label hidden under a marker is the same failure as a label
    # sitting on its own connector.
    for i, (lab, v) in enumerate(zip(labels, model)):
        fig.add_annotation(x=lab, y=max(v, mkt[i] if mkt else v), yshift=18,
                           text=f"{v:.0f}%", showarrow=False,
                           font=dict(size=16, color=label_ink[i], family=PLOT_FONT))

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
                    line=dict(color=MUTED, dash="dash", width=1))
    # the model trace is the one focal element on the calibration sheet
    fig.add_scatter(x=agg["pred"], y=agg["actual"], mode="markers+lines",
                    name="MODEL", line=dict(color=ACCENT, width=2),
                    marker=dict(color=ACCENT, size=12,
                                line=dict(color=INK, width=1)))
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
    market_html = _fig_html(_market_fig(pred, market, home, away), 868, 380)
    heat_html = _fig_html(_heatmap_fig(pred["_grid"], home, away), 868, 380)

    if h2h:
        h2h_html = "".join(
            f'<div class="h2h-row"><span class="date">{e(r["date"])}</span>'
            f'<span class="score">{e(r["line"])}</span></div>' for r in h2h)
    else:
        h2h_html = '<div class="h2h-empty">No meetings in the last 2 seasons</div>'

    i = _best(pred)
    verdict = (home, "Draw", away)[i]
    verdict_p = (pred["p_home"], pred["p_draw"], pred["p_away"])[i]
    # display type is set to the word, not the other way round
    name_px = 64 if len(verdict) <= 9 else 48 if len(verdict) <= 14 else 40

    blocks = (
        _block("01", "Model vs Market" if market else "Model Probability",
               "bars = model · ◇ = market" if market else "1X2 · no odds for this source",
               market_html)
        + _block("02", "Scoreline Plate", "0–3 goals · %", heat_html)
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
        title=e(_insight_title(pred, home, away)),
        sub=f"{e(home)} &nbsp;vs&nbsp; {e(away)}",
        blocks=blocks), out)


def calibration_curve(log_path: str, out: str, bins: int = 10) -> None:
    """Reliability plot from settled predictions (needs results filled in)."""
    fig, meta = _calibration_fig(log_path, bins)
    if fig is None:
        return
    plot_html = _fig_html(fig, 1660, 700)
    _render_html(_sheet(
        kicker="Calibration", meta="reliability plate",
        title=meta["title"], sub=meta["subtitle"],
        blocks=_block("01", "Predicted vs Actual", "all settled outcomes",
                      plot_html),
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
    def p3(h, d, a):
        return {"p_home": h, "p_draw": d, "p_away": a}
    assert "strongly favoured" in _insight_title(p3(.70, .20, .10), "A", "B")
    assert "real shot" in _insight_title(p3(.50, .20, .30), "A", "B")
    assert "too close to call" in _insight_title(p3(.30, .40, .30), "A", "B")
    assert "tight matchup" in _insight_title(p3(.40, .20, .40), "A", "B")
    # headline, focal bar and verdict must agree on the pick
    assert _best(p3(.40, .40, .20)) == 1, "a tie reads as a draw everywhere"

    # exactly one focal bar - diagram-design allows 1-2 accents, never 3
    pred = {"p_home": .642, "p_draw": .221, "p_away": .137}
    bar = _market_fig(pred, None, "A", "B").data[0].marker
    assert list(bar.color).count(ACCENT_TINT) == 1, bar.color
    assert list(bar.line.color).count(ACCENT) == 1, bar.line.color

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
