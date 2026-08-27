"""簡易規則策略回測畫面。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.predict import model_available, score_features
from screens.common import disclaimer, sidebar_controls
from utils.costs import default_cost_rate
from utils.metrics import performance


def _ma_strategy(features: pd.DataFrame, cost: float) -> tuple[pd.Series, pd.DataFrame]:
    frame = features.sort_values(["Ticker", "Date"]).copy()
    frame["ma5"] = frame.groupby("Ticker").Close.transform(lambda data: data.rolling(5).mean())
    frame["ma20"] = frame.groupby("Ticker").Close.transform(lambda data: data.rolling(20).mean())
    frame["position"] = (frame.ma5 > frame.ma20).astype(float)
    frame["turnover"] = frame.groupby("Ticker").position.diff().abs().fillna(0)
    frame["net_return"] = frame.groupby("Ticker").Close.pct_change().fillna(0) * frame.groupby("Ticker").position.shift(1).fillna(0) - frame.turnover * cost
    equity = (1 + frame.groupby("Date").net_return.mean()).cumprod()
    trades = frame[frame.turnover > 0][["Date", "Ticker", "Close", "position"]].rename(columns={"Close": "價格", "position": "持倉狀態"})
    return equity, trades


def render_backtest() -> None:
    st.title("簡易回測")
    disclaimer()
    _, _, features, failed = sidebar_controls()
    if failed:
        st.warning("下載失敗：" + ", ".join(failed))
    strategy = st.radio("策略", ["雙均線", "模型分數"], horizontal=True, key="workbench_strategy")
    initial = st.number_input("起始資金", min_value=1000.0, value=100000.0, step=1000.0)
    enabled = st.checkbox("扣除交易成本", value=True)
    default_cost = default_cost_rate(features.Ticker.dropna().unique().tolist()) * 100
    cost = st.number_input("單邊交易成本 (%)", min_value=0.0, value=default_cost, step=.01) / 100 if enabled else 0.0
    if strategy == "模型分數":
        st.info("模型策略以保存的模型對歷史可用特徵打分，每 5 日等權持有 Top K；請先訓練模型。")
        if not model_available():
            return
        k = st.slider("持有 Top K", 1, min(10, features.Ticker.nunique()), min(3, features.Ticker.nunique()))
        historical = score_features(features, latest_only=False)
        historical["bucket"] = historical.Date.dt.to_period("W").astype(str)
        historical["rank"] = historical.groupby("bucket").score.rank(ascending=False, method="first")
        historical["position"] = (historical["rank"] <= k).astype(float)
        historical["net_return"] = historical.groupby("Ticker").Close.pct_change().fillna(0) * historical.groupby("Ticker").position.shift(1).fillna(0)
        historical["turnover"] = historical.groupby("Ticker").position.diff().abs().fillna(0)
        historical["net_return"] -= historical.turnover * cost
        equity = (1 + historical.groupby("Date").net_return.mean()).cumprod()
        trades = historical[historical.turnover > 0][["Date", "Ticker", "Close", "position"]].rename(columns={"Close": "價格", "position": "持倉狀態"})
    else:
        equity, trades = _ma_strategy(features, cost)
    benchmark = (1 + features.groupby("Date").Close.pct_change().groupby(features.Date).mean().fillna(0)).cumprod()
    stats = performance(equity)
    a, b, c, d = st.columns(4)
    a.metric("總回報", f"{stats['total_return']:.2%}"); b.metric("年化", f"{stats['annual_return']:.2%}"); c.metric("最大回撤", f"{stats['max_drawdown']:.2%}"); d.metric("交易次數", len(trades))
    fig = go.Figure(); fig.add_scatter(x=equity.index, y=equity * initial, name="策略"); fig.add_scatter(x=benchmark.index, y=benchmark * initial, name="等權 Buy & Hold")
    fig.update_layout(title="資金曲線", template="plotly_dark", yaxis_title="資金")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(trades.sort_values("Date", ascending=False), width="stretch", hide_index=True)
    st.warning("回測未計印花稅細節、極端滑價及借貨成本，結果可能偏樂觀。")