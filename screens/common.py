"""短炒工作台畫面共用函式。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import DEFAULT_UNIVERSE
from data.prices import download_prices
from data.universe import selected_universe
from features.engineering import build_features


def disclaimer() -> None:
    st.caption("本工具只供研究學習，不構成投資建議。")


def sidebar_controls() -> tuple[dict[str, str], str, pd.DataFrame, tuple[str, ...]]:
    with st.sidebar:
        st.divider()
        st.markdown("### 短炒工作台")
        market = st.selectbox("市場", ["全部", "只港股", "只美股"], key="workbench_market")
        candidates = list(
            DEFAULT_UNIVERSE
            if market == "全部"
            else {
                ticker: name
                for ticker, name in DEFAULT_UNIVERSE.items()
                if ticker.endswith(".HK") == (market == "只港股")
            }
        )
        chosen = st.multiselect(
            "股票池",
            candidates,
            default=candidates,
            format_func=lambda ticker: f"{ticker} · {DEFAULT_UNIVERSE[ticker]}",
            key=f"workbench_tickers_{market}",
        )
        custom = st.text_input("自訂代碼（逗號分隔）", key="workbench_custom")
        period_label = st.selectbox("數據年期", ["1y", "2y"], index=1, key="workbench_period")
        if st.button("重新整理數據", key="workbench_refresh"):
            download_prices.clear()
        universe = selected_universe(market, chosen, custom)
    prices, failed = download_prices(tuple(universe), period_label)
    return universe, period_label, build_features(prices), failed