"""
Static, high-res PNGs for the video pipeline. Three charts:

  1. scoreline_heatmap  – the 15x15 joint-probability grid for one match
  2. model_vs_market    – model 1X2 probs overlaid on bookmaker-implied
  3. calibration_curve  – reliability plot built from predictions_log.csv

Design language: Linear/Notion dark aesthetic + data-ink discipline
(principles borrowed from indi256s/dataviz-skill and Anthropic's
data-visualization skill):
  - Data-ink ratio is sacred — every gridline/border earns its place.
  - Titles are conclusions ("City strongly favoured"), not axis labels.
  - Gray is the default; ONE accent colour highlights the actual story
    (the favoured outcome), everything else recedes to grayscale — this
    also means colour is never the only signal (labels always present),
    satisfying colourblind-safety by construction.
  - Sequential, single-hue colour scales for the heatmap (never a
    red/green diverging scale) — also colourblind-safe by construction.

All exported at 2x scale so they stay crisp when scaled in OpenMontage.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

SCALE = 2

# ---- Linear-style dark palette ---------------------------------------
BG = "#08090a"            # Linear's near-black canvas
PANEL = "#0f1115"
BORDER = "#1c1f26"
GRID = "#181b21"
FG = "#f7f8f8"
MUTED = "#8a8f98"         # Linear's muted label gray
GRAY_FILL = "#2a2e37"     # de-emphasised series — "gray is your best friend"
GRAY_FILL_2 = "#3d434f"   # second de-emphasised series, distinguishable in greyscale
ACCENT = "#5e6ad2"        # Linear purple — the ONE highlight colour
ACCENT_SOFT = "#8f97e8"
MARKET = "#c7ccd4"

WATERMARK = "@nineshoot"   # set to "" to disable

# single-hue sequential scale (dark -> Linear purple), colourblind-safe
HEAT_SCALE = [
    [0.0, "#111318"], [0.15, "#1c1f3d"], [0.35, "#2f3480"],
    [0.6, "#4750c4"], [0.8, "#5e6ad2"], [1.0, "#b8bdf5"],
]


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


def _watermark(fig: go.Figure) -> None:
    if not WATERMARK:
        return
    fig.add_annotation(
        text=WATERMARK, xref="paper", yref="paper",
        x=1, y=0, xanchor="right", yanchor="bottom",
        xshift=0, yshift=-46,
        showarrow=False, font=dict(size=13, color=MUTED))


def _base(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    full_title = f"<b>{title}</b>"
    if subtitle:
        full_title += f"<br><span style='font-size:14px;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=full_title, font=dict(color=FG, size=24),
                  x=0.03, xanchor="left"),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=FG, family="Inter, -apple-system, Arial", size=15),
        margin=dict(l=70, r=50, t=95, b=70),
        width=1280, height=720,
        shapes=[dict(  # subtle 1px card border around the plot — Linear signature
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color=BORDER, width=1), fillcolor="rgba(0,0,0,0)",
        )],
    )
    _watermark(fig)
    return fig


def scoreline_heatmap(grid, home: str, away: str, out: str) -> None:
    g = np.asarray(grid)[:6, :6]  # 0-5 goals is the interesting corner
    ai, aj = np.unravel_index(np.argmax(g), g.shape)  # most likely scoreline

    fig = go.Figure(go.Heatmap(
        z=g * 100, colorscale=HEAT_SCALE,
        colorbar=dict(title=dict(text="Prob %", font=dict(color=FG)),
                      tickfont=dict(color=FG), outlinewidth=0),
        text=np.round(g * 100, 1), texttemplate="%{text}",
        textfont=dict(size=13, color=FG),
        xgap=3, ygap=3))
    fig.update_xaxes(title=f"{away} goals", dtick=1, gridcolor=GRID,
                     zeroline=False)
    fig.update_yaxes(title=f"{home} goals", dtick=1, gridcolor=GRID,
                     zeroline=False)
    fig.add_annotation(x=aj, y=ai, text="most likely", showarrow=True,
                       arrowhead=2, arrowcolor=ACCENT_SOFT,
                       font=dict(color=FG, size=13), ax=40, ay=-40)
    _base(fig, f"Most likely scoreline: {home} {ai}–{aj} {away}",
          "Probability of every scoreline 0–5 goals each side")
    fig.write_image(out, scale=SCALE)


def model_vs_market(pred: dict, market: dict | None,
                    home: str, away: str, out: str) -> None:
    labels = [home, "Draw", away]
    model = [pred["p_home"] * 100, pred["p_draw"] * 100, pred["p_away"] * 100]

    # "Gray is your best friend": highlight ONLY the favoured outcome in
    # the one accent colour; the other two recede to two distinguishable
    # grays. This also means colour is never load-bearing on its own —
    # the x-axis labels and in-bar numbers already carry the information.
    best_idx = int(np.argmax(model))
    grays = [GRAY_FILL, GRAY_FILL_2]
    gray_cycle = iter(grays * 2)
    colors = [ACCENT if i == best_idx else next(gray_cycle) for i in range(3)]

    fig = go.Figure()
    fig.add_bar(x=labels, y=model, name="Model", marker_color=colors,
               text=[f"{v:.0f}%" for v in model],
               textposition="inside", insidetextanchor="middle",
               textfont=dict(size=18, color=FG), width=0.55,
               cliponaxis=False,
               marker_line_width=0,
               showlegend=False)
    # dummy trace purely to give "Model" a clean, always-visible legend swatch
    fig.add_bar(x=[None], y=[None], name="Model", marker_color=ACCENT,
               showlegend=True)
    subtitle = "Model probability"
    if market:
        mkt = [market["p_home"] * 100, market["p_draw"] * 100,
               market["p_away"] * 100]
        fig.add_scatter(x=labels, y=mkt, mode="markers+text", name="Market",
                        marker=dict(color=MARKET, size=16, symbol="diamond",
                                   line=dict(color=BG, width=2)),
                        text=[f"{v:.0f}%" for v in mkt],
                        textposition="top center",
                        textfont=dict(size=13, color=FG))
        subtitle = "Model probability vs bookmaker-implied (overround removed)"

    fig.update_yaxes(title="Probability %", range=[0, 108], gridcolor=GRID,
                     zeroline=False)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=1, xanchor="right",
                                  bgcolor="rgba(0,0,0,0)"),
                      bargap=0.4)
    title = _insight_title(pred["p_home"], pred["p_draw"], pred["p_away"],
                           home, away)
    _base(fig, title, subtitle)
    fig.write_image(out, scale=SCALE)


def calibration_curve(log_path: str, out: str, bins: int = 10) -> None:
    """Reliability plot from settled predictions (needs results filled in)."""
    if not Path(log_path).exists() or Path(log_path).stat().st_size == 0:
        print("  [skip] calibration: predictions_log.csv doesn't exist yet")
        return
    try:
        df = pd.read_csv(log_path)
    except pd.errors.EmptyDataError:
        print("  [skip] calibration: predictions_log.csv is empty")
        return
    df = df.dropna(subset=["result"])
    if df.empty:
        print("  [skip] calibration: no settled results yet")
        return
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
                     gridcolor=GRID, zeroline=False, tickformat=".0%")
    fig.update_yaxes(title="Actual hit rate", range=[0, 1],
                     gridcolor=GRID, zeroline=False, tickformat=".0%")
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=1, xanchor="right",
                                  bgcolor="rgba(0,0,0,0)"))
    title = f"Model is off by {mae*100:.1f} points on average" if mae >= 0.02 \
        else "Model tracks its own confidence closely"
    _base(fig, title, f"{int(agg['n'].sum())} settled outcomes, {bins} buckets")
    fig.write_image(out, scale=SCALE)


def accuracy_scoreboard(stats: dict, out: str) -> None:
    """
    Linear-style KPI card: win count + win rate, big and readable —
    the number that actually goes on screen in a video, as opposed to
    the more technical calibration_curve.
    """
    total, wins, rate = stats["total"], stats["wins"], stats["win_rate"]

    fig = go.Figure()
    # Big headline number as an indicator — no axes, pure KPI card.
    fig.add_trace(go.Indicator(
        mode="number",
        value=rate * 100,
        number=dict(suffix="%", font=dict(size=110, color=ACCENT)),
        domain=dict(x=[0, 1], y=[0.35, 1]),
    ))
    fig.add_annotation(
        text=f"{wins} / {total} correct picks", xref="paper", yref="paper",
        x=0.5, y=0.28, xanchor="center", showarrow=False,
        font=dict(size=22, color=FG))

    # small breakdown row: Home / Draw / Away hit rates
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
            x=0.5, y=0.12, xanchor="center", showarrow=False,
            font=dict(size=15, color=MUTED))

    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=FG, family="Inter, -apple-system, Arial"),
        width=1280, height=720,
        margin=dict(l=40, r=40, t=40, b=40),
        shapes=[dict(type="rect", xref="paper", yref="paper",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(color=BORDER, width=1),
                    fillcolor="rgba(0,0,0,0)")],
    )
    fig.add_annotation(
        text="Model hit rate to date", xref="paper", yref="paper",
        x=0.5, y=0.92, xanchor="center", showarrow=False,
        font=dict(size=18, color=MUTED))
    _watermark(fig)
    fig.write_image(out, scale=SCALE)
