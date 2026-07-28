"""Entry zone + target price chart (split out for reliable imports)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def entry_target_chart(
    df: pd.DataFrame,
    entry_lo: float | None,
    entry_hi: float | None,
    stop: float | None,
    short_bull: float | None,
    short_bear: float | None,
    med_bull: float | None,
    med_bear: float | None,
    title: str = "入场区与目标价",
    ultra_bull: float | None = None,
    ultra_bear: float | None = None,
) -> go.Figure:
    """Price chart with entry zone, stop, and ultra/short/medium targets."""
    fig = go.Figure()
    if df is None or df.empty or "Close" not in getattr(df, "columns", []):
        fig.update_layout(title=title or "无数据", height=400, template="plotly_dark")
        return fig

    plot_df = df.tail(80).copy()
    for col in ("Open", "High", "Low", "Close"):
        if col not in plot_df.columns:
            plot_df[col] = plot_df["Close"]
    if "Date" not in plot_df.columns:
        plot_df = plot_df.reset_index()
        if "Date" not in plot_df.columns and "index" in plot_df.columns:
            plot_df = plot_df.rename(columns={"index": "Date"})

    x = plot_df["Date"] if "Date" in plot_df.columns else plot_df.index
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="K线",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        )
    )

    def _band(y0, y1, color, name):
        if y0 is None or y1 is None:
            return
        fig.add_hrect(
            y0=min(y0, y1),
            y1=max(y0, y1),
            fillcolor=color,
            opacity=0.15,
            line_width=0,
            annotation_text=name,
            annotation_position="top left",
        )

    if entry_lo is not None and entry_hi is not None:
        _band(entry_lo, entry_hi, "rgba(30,136,229,0.9)", "入场区")
        fig.add_hline(
            y=entry_lo,
            line_dash="dot",
            line_color="#1e88e5",
            annotation_text=f"买区下 {entry_lo:.2f}",
        )
        fig.add_hline(
            y=entry_hi,
            line_dash="dot",
            line_color="#1e88e5",
            annotation_text=f"买区上 {entry_hi:.2f}",
        )

    if stop is not None:
        fig.add_hline(
            y=stop,
            line_dash="dash",
            line_color="#6d4c41",
            annotation_text=f"止损 {stop:.2f}",
        )

    for price, label, color, width in (
        (ultra_bull, "一周目标", "#ab47bc", 1.6),
        (ultra_bear, "一周下看", "#7e57c2", 1.2),
        (short_bull, "短期目标", "#ef5350", 1.4),
        (short_bear, "短期下看", "#26a69a", 1.1),
        (med_bull, "中期目标", "#c62828", 1.5),
        (med_bear, "中期下看", "#00695c", 1.1),
    ):
        if price is not None:
            fig.add_hline(
                y=price,
                line_dash="solid" if "目标" in label else "dash",
                line_color=color,
                line_width=width,
                annotation_text=f"{label} {price:.2f}",
            )

    fig.update_layout(
        title=title,
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=110, t=50, b=30),
        template="plotly_dark",
        showlegend=False,
    )
    return fig
