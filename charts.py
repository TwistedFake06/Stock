"""Plotly chart builders for the stock app."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def price_volume_chart(
    df: pd.DataFrame,
    title: str = "",
    show_sma: bool = True,
    show_bb: bool = False,
    show_volume: bool = True,
) -> go.Figure:
    rows = 2 if show_volume else 1
    row_heights = [0.72, 0.28] if show_volume else [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K线",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        ),
        row=1,
        col=1,
    )

    if show_sma:
        for col, color in (("SMA5", "#f9a825"), ("SMA20", "#1e88e5"), ("SMA60", "#8e24aa")):
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df[col],
                        mode="lines",
                        name=col,
                        line=dict(width=1.2, color=color),
                    ),
                    row=1,
                    col=1,
                )

    if show_bb and "BB_UPPER" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["BB_UPPER"],
                mode="lines",
                name="布林上轨",
                line=dict(width=1, color="rgba(100,100,100,0.5)", dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["BB_LOWER"],
                mode="lines",
                name="布林下轨",
                line=dict(width=1, color="rgba(100,100,100,0.5)", dash="dot"),
                fill="tonexty",
                fillcolor="rgba(100,100,100,0.08)",
            ),
            row=1,
            col=1,
        )

    if show_volume and "Volume" in df.columns:
        colors = [
            "#ef5350" if c >= o else "#26a69a"
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                name="成交量",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=560 if show_volume else 420,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_dark",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI" not in df.columns:
        return fig
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["RSI"],
            mode="lines",
            name="RSI(14)",
            line=dict(color="#5e35b1", width=1.5),
        )
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="超买 70")
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="超卖 30")
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(94,53,177,0.05)", line_width=0)
    fig.update_layout(
        title="RSI 相对强弱指标",
        height=240,
        margin=dict(l=40, r=20, t=40, b=30),
        template="plotly_dark",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
    )
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    if "MACD" not in df.columns:
        return fig
    colors = ["#ef5350" if v >= 0 else "#26a69a" for v in df["MACD_HIST"]]
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["MACD_HIST"], name="柱状", marker_color=colors, opacity=0.7)
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color="#1e88e5", width=1.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MACD_SIGNAL"],
            mode="lines",
            name="Signal",
            line=dict(color="#fb8c00", width=1.2),
        )
    )
    fig.update_layout(
        title="MACD",
        height=260,
        margin=dict(l=40, r=20, t=40, b=30),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


def compare_chart(df: pd.DataFrame, title: str = "相对走势 (基期=100)") -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(width=1.8))
        )
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=40, r=20, t=50, b=30),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="归一化价格",
    )
    return fig


def drawdown_chart(equity: pd.Series, drawdown: pd.Series, title: str = "净值与回撤") -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=("净值曲线 (起点=100)", "回撤 %"),
    )
    if equity is not None and len(equity):
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name="净值",
                line=dict(color="#1e88e5", width=1.6),
            ),
            row=1,
            col=1,
        )
    if drawdown is not None and len(drawdown):
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name="回撤",
                fill="tozeroy",
                line=dict(color="#26a69a", width=1),
                fillcolor="rgba(38,166,154,0.25)",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=40, r=20, t=60, b=30),
        template="plotly_dark",
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def sr_chart(df: pd.DataFrame, levels: list, title: str = "支撑 / 阻力") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K线",
            increasing_line_color="#ef5350",
            decreasing_line_color="#26a69a",
        )
    )
    color_map = {"支撑": "#26a69a", "阻力": "#ef5350", "枢轴": "#fb8c00"}
    # Only draw nearest few levels to avoid clutter
    last = float(df["Close"].iloc[-1])
    ranked = sorted(levels, key=lambda lv: abs(lv.price - last))[:8]
    for lv in ranked:
        fig.add_hline(
            y=lv.price,
            line_dash="dot" if lv.strength == "弱" else "dash",
            line_color=color_map.get(lv.kind, "#90a4ae"),
            annotation_text=f"{lv.kind} {lv.price:.2f}",
            annotation_position="right",
            opacity=0.85,
        )
    fig.update_layout(
        title=title,
        height=460,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=80, t=50, b=30),
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def relative_strength_chart(rel_df: pd.DataFrame, title: str = "相对强弱") -> go.Figure:
    fig = go.Figure()
    for col, color in (("股票", "#ef5350"), ("基准", "#1e88e5"), ("相对强度", "#8e24aa")):
        if col in rel_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=rel_df.index,
                    y=rel_df[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1.8 if col != "相对强度" else 1.4, color=color),
                )
            )
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=40, r=20, t=50, b=30),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="归一化 (起点=100)",
    )
    return fig


def scorecard_radar(
    tech: float | None,
    funda: float | None,
    risk: float | None,
    rs: float | None,
) -> go.Figure:
    labels = ["技术面", "基本面", "风险调整", "相对强弱"]
    values = [
        tech if tech is not None else 0,
        funda if funda is not None else 0,
        risk if risk is not None else 0,
        rs if rs is not None else 0,
    ]
    # close polygon
    labels = labels + [labels[0]]
    values = values + [values[0]]
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name="得分",
            line=dict(color="#5e35b1"),
            fillcolor="rgba(94,53,177,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=40, b=30),
        title="综合能力雷达",
        template="plotly_dark",
    )
    return fig


# Re-export for callers that still import from charts
try:
    from chart_entry import entry_target_chart  # noqa: F401
except Exception:  # pragma: no cover
    entry_target_chart = None  # type: ignore


def bias_gauge(score: float, bias: str) -> go.Figure:
    """Horizontal-style gauge for multi/空 score (-100 ~ +100)."""
    color = "#ef5350" if score > 0 else "#26a69a" if score < 0 else "#78909c"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={"suffix": " 分", "font": {"size": 36}},
            title={"text": f"多空规则分 · {bias}", "font": {"size": 16}},
            gauge={
                "axis": {"range": [-100, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#eceff1",
                "steps": [
                    {"range": [-100, -45], "color": "#b2dfdb"},
                    {"range": [-45, -18], "color": "#e0f2f1"},
                    {"range": [-18, 18], "color": "#f5f5f5"},
                    {"range": [18, 45], "color": "#ffebee"},
                    {"range": [45, 100], "color": "#ffcdd2"},
                ],
                "threshold": {
                    "line": {"color": "#37474f", "width": 2},
                    "thickness": 0.8,
                    "value": score,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=50, b=10),
        template="plotly_dark",
    )
    return fig
