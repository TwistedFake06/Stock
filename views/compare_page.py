"""Streamlit page: 多股对比."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import compare_chart
from stock_service import compare_symbols, normalize_symbol
from ui_mobile import plotly_chart as mobile_plotly


def render_compare(period: str, interval: str, period_label: str) -> None:
    st.header("多股对比")
    st.caption("价格归一化到起点 = 100，便于比较相对强弱。")

    default_compare = ",".join(st.session_state.watchlist[:4])
    raw = st.text_input(
        "对比代码（逗号分隔）",
        value=default_compare,
        placeholder="AAPL,MSFT,0700.HK,600519",
    )
    symbols = [normalize_symbol(s) for s in raw.split(",") if s.strip()]

    if not symbols:
        st.info("请输入至少一个股票代码。")
        return

    with st.spinner("拉取对比数据..."):
        cmp_df = compare_symbols(symbols, period=period, interval=interval)

    if cmp_df.empty:
        st.warning("没有可用的对比数据，请检查代码。")
        return

    mobile_plotly(
        compare_chart(cmp_df, title=f"相对走势 · {period_label}"),
        width="stretch",
    )

    rows = []
    for col in cmp_df.columns:
        series = cmp_df[col].dropna()
        if series.empty:
            continue
        ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
        rows.append(
            {
                "代码": col,
                "区间涨跌%": round(ret, 2),
                "最新指数": round(float(series.iloc[-1]), 2),
            }
        )
    if rows:
        table = pd.DataFrame(rows).sort_values("区间涨跌%", ascending=False)
        st.dataframe(table, width="stretch", hide_index=True)
