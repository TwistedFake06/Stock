"""Streamlit page: 行情看板."""
from __future__ import annotations

import streamlit as st

from analysis import analyze_bias
from charts import price_volume_chart
from indicators import enrich
from stock_service import (
    cache_bucket,
    cached_info,
    compute_returns,
    fetch_history,
)
from ui_mobile import plotly_chart as mobile_plotly
from views.common import (
    fmt_number,
    fmt_pct,
    get_price_fields,
    render_bias_banner,
    render_session_quote_card,
    save_watchlist,
)


def render_dashboard(
    symbol: str,
    period: str,
    interval: str,
    period_label: str,
    interval_label: str,
) -> None:
    st.header(f"行情看板 · `{symbol}`")

    with st.spinner("加载行情..."):
        info = cached_info(symbol, cache_bucket(5))
        hist = fetch_history(symbol, period=period, interval=interval)

    fields = get_price_fields(info)
    st.subheader(fields["name"] or symbol)
    meta_bits = [b for b in [fields["exchange"], fields["sector"], fields["industry"]] if b]
    if meta_bits:
        st.caption(" · ".join(meta_bits))

    # 盘前 / 常规 / 盘后 / 夜盘时段
    st.markdown("#### 交易时段 · 盘前盘后")
    render_session_quote_card(symbol, info)
    st.markdown("---")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cur = fields["currency"]
    c1.metric(
        "最新价",
        fmt_number(fields["price"], suffix=f" {cur}" if cur else ""),
        fmt_pct(fields["change_pct"]),
    )
    c2.metric("开盘", fmt_number(fields["open"]))
    c3.metric("最高", fmt_number(fields["high"]))
    c4.metric("最低", fmt_number(fields["low"]))
    c5.metric("52周高", fmt_number(fields["year_high"]))
    c6.metric("市值", fmt_number(fields["mcap"]))

    m1, m2, m3 = st.columns(3)
    m1.metric("前收", fmt_number(fields["prev"]))
    m2.metric("市盈率 PE", fmt_number(fields["pe"]))
    rets = compute_returns(hist) if not hist.empty else {}
    m3.metric(
        f"区间涨跌 ({period_label})",
        fmt_pct(rets.get("total_return_pct")),
        help="所选时间范围内首末收盘价涨跌幅",
    )

    if hist.empty:
        st.warning(
            f"未能获取 `{symbol}` 的历史数据。请检查代码是否正确。"
            "\n\n示例：`AAPL`、`0700.HK`、`600519`（茅台）、`000001`（平安银行）"
        )
    else:
        df = enrich(hist)
        report = analyze_bias(df)
        render_bias_banner(report, compact=True)

        fig = price_volume_chart(df, title=f"{symbol} · {period_label} · {interval_label}")
        mobile_plotly(fig, use_container_width=True)

        with st.expander("原始数据", expanded=False):
            show = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(show.iloc[::-1], use_container_width=True, hide_index=True)

    if st.button("⭐ 加入自选"):
        if symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(symbol)
            save_watchlist(st.session_state.watchlist)
            st.success(f"已添加 {symbol}")
        else:
            st.info(f"{symbol} 已在自选列表中")
