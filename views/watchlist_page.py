"""Streamlit page: 自选股."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import analyze_bias, bias_emoji
from indicators import enrich
import stock_service as _ss

if not hasattr(_ss, "filter_us_only"):
    import importlib

    _ss = importlib.reload(_ss)

DEFAULT_WATCHLIST = _ss.DEFAULT_WATCHLIST
cache_bucket = _ss.cache_bucket
cached_info = _ss.cached_info
fetch_history = _ss.fetch_history
filter_us_only = _ss.filter_us_only
is_us_symbol = _ss.is_us_symbol
normalize_symbol = _ss.normalize_symbol

from views.common import fmt_number, get_price_fields, save_watchlist


def render_watchlist(period: str, interval: str) -> None:
    st.header("自选股管理")

    add_col, _ = st.columns([2, 3])
    with add_col:
        new_sym = st.text_input("添加代码", placeholder="例如 NVDA 或 300750")
        if st.button("添加", type="primary") and new_sym.strip():
            ns = normalize_symbol(new_sym)
            if not is_us_symbol(ns):
                st.error("快速自选仅支持美股代码（已过滤港股/A股）。")
            elif ns not in st.session_state.watchlist:
                st.session_state.watchlist.append(ns)
                st.session_state.watchlist = filter_us_only(st.session_state.watchlist)
                save_watchlist(st.session_state.watchlist)
                st.success(f"已添加 {ns}")
                st.rerun()
            else:
                st.info("已存在")

    st.divider()

    if not st.session_state.watchlist:
        st.info("自选列表为空，请添加股票。")
        return

    rows = []
    progress = st.progress(0, text="刷新自选行情与多空...")
    n = len(st.session_state.watchlist)
    for i, s in enumerate(st.session_state.watchlist):
        info = cached_info(s, cache_bucket(5))
        f = get_price_fields(info)
        hist_s = fetch_history(s, period=period, interval=interval)
        if hist_s.empty:
            bias_label, bias_score = "—", None
        else:
            rep = analyze_bias(enrich(hist_s))
            bias_label = f"{bias_emoji(rep.bias)} {rep.bias}"
            bias_score = rep.score
        rows.append(
            {
                "代码": s,
                "名称": f["name"] or s,
                "最新价": f["price"],
                "涨跌%": f["change_pct"],
                "多空": bias_label,
                "得分": bias_score,
                "市值": f["mcap"],
                "PE": f["pe"],
                "交易所": f["exchange"],
            }
        )
        progress.progress((i + 1) / n, text=f"分析 {s}...")
    progress.empty()

    table = pd.DataFrame(rows)
    display = table.copy()
    if "涨跌%" in display.columns:
        display["涨跌%"] = display["涨跌%"].apply(
            lambda x: f"{float(x):+.2f}%" if x is not None and pd.notna(x) else "—"
        )
    if "得分" in display.columns:
        display["得分"] = display["得分"].apply(
            lambda x: f"{float(x):+.0f}" if x is not None and pd.notna(x) else "—"
        )
    for col in ("最新价", "市值", "PE"):
        if col in display.columns:
            display[col] = display[col].apply(lambda x: fmt_number(x))

    if "得分" in table.columns and table["得分"].notna().any():
        order = table["得分"].fillna(0).sort_values(ascending=False).index
        display = display.loc[order]

    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("自选列表按多空得分从高到低排序（看多在前）。")

    st.subheader("操作")
    for s in list(st.session_state.watchlist):
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.write(f"`{s}`")
        if c2.button("查看", key=f"view_{s}"):
            st.session_state.symbol = s
            st.info(f"已选择 {s}，请到「行情看板」或「技术分析」查看。")
        if c3.button("删除", key=f"del_{s}"):
            st.session_state.watchlist = [x for x in st.session_state.watchlist if x != s]
            save_watchlist(st.session_state.watchlist)
            st.rerun()

    if st.button("恢复默认自选"):
        st.session_state.watchlist = list(DEFAULT_WATCHLIST)
        save_watchlist(st.session_state.watchlist)
        st.rerun()
