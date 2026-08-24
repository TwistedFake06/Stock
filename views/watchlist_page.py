"""Streamlit page: 自选股."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import analyze_bias, bias_emoji
from indicators import enrich
import stock_service as _ss
from trade_sop import build_trade_sop

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
    progress = st.progress(0, text="刷新自选行情与 SOP 风控...")
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
        sop_verdict = "资料不足"
        risk_units = None
        rr_net = None
        earnings_note = "—"
        earnings_soon = False
        sop_brief = ""
        sop_status = "成功"
        try:
            # Watchlist 是预筛选：不取额外 1H 资料，执行前仍应到投资 SOP 确认。
            sop = build_trade_sop(
                s,
                period=period,
                interval=interval,
                include_h1=False,
            )
            sop_verdict = sop.enter_ok
            risk_units = sop.risk_units
            rr_net = getattr(sop.primary_plan, "rr_net", None) if sop.primary_plan else None
            earnings_note = sop.earnings_note or "—"
            earnings_soon = bool(sop.earnings_soon)
            sop_brief = sop.one_liner_reason or sop.decision_brief.split("\n", 1)[0]
        except Exception as exc:
            sop_status = f"失败：{type(exc).__name__}"
        rows.append(
            {
                "代码": s,
                "名称": f["name"] or s,
                "最新价": f["price"],
                "涨跌%": f["change_pct"],
                "多空": bias_label,
                "得分": bias_score,
                "SOP结论": sop_verdict,
                "SOP状态": sop_status,
                "允许风险": risk_units,
                "净R:R": rr_net,
                "事件": earnings_note,
                "财报临近": earnings_soon,
                "执行重点": sop_brief,
                "市值": f["mcap"],
                "PE": f["pe"],
                "交易所": f["exchange"],
            }
        )
        progress.progress((i + 1) / n, text=f"分析 {s} 的 SOP...")
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
    if "允许风险" in display.columns:
        display["允许风险"] = display["允许风险"].apply(
            lambda x: f"{float(x):g}R" if x is not None and pd.notna(x) else "—"
        )
    if "净R:R" in display.columns:
        display["净R:R"] = display["净R:R"].apply(
            lambda x: f"{float(x):.2f}" if x is not None and pd.notna(x) else "—"
        )
    for col in ("最新价", "市值", "PE"):
        if col in display.columns:
            display[col] = display[col].apply(lambda x: fmt_number(x))

    verdict_rank = {"适合入场": 3, "谨慎试仓": 2, "观望": 1, "回避": 0}
    table["_sop_rank"] = table["SOP结论"].map(verdict_rank).fillna(-1)
    table["_risk_rank"] = table["允许风险"].fillna(-1)
    table["_status_rank"] = (table["SOP状态"] == "成功").astype(int)
    order = table.sort_values(
        ["_status_rank", "_sop_rank", "_risk_rank", "得分"], ascending=False
    ).index
    display = display.loc[order]

    action_columns = [
        "代码", "名称", "最新价", "涨跌%", "SOP结论", "允许风险",
        "净R:R", "事件", "执行重点", "SOP状态",
    ]
    action_columns = [column for column in action_columns if column in display.columns]
    ready_mask = (
        (table["SOP状态"] == "成功")
        & table["SOP结论"].isin(["适合入场", "谨慎试仓"])
        & (table["允许风险"].fillna(0) > 0)
        & ~table["财报临近"]
    )
    wait_mask = (
        (table["SOP状态"] == "成功")
        & ~ready_mask
        & ~table["SOP结论"].isin(["回避"])
    )
    pause_mask = ~ready_mask & ~wait_mask

    groups = [
        ("今天可准备", ready_mask, "资料完整、允许新增风险且没有近期财报风险；仍只挂计划 E，不追价。"),
        ("等待条件", wait_mask, "等回到入场区、财报后或结构改善；不要因为排名靠前就提前买。"),
        ("暂缓 / 资料问题", pause_mask, "趋势或赔率不合格、财报风险，或 SOP 资料失败；不建立新仓。"),
    ]
    for title, mask, note in groups:
        group = display.loc[mask.reindex(display.index, fill_value=False), action_columns]
        st.subheader(f"{title} · {len(group)}")
        st.caption(note)
        if group.empty:
            st.caption("目前没有标的。")
        else:
            st.dataframe(group, width="stretch", hide_index=True)

    st.caption("分组只使用当前 SOP 结果，资料失败不会伪装成观望；下单前仍须进入「投资SOP」确认 E/S/T。")

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
        try:
            from stock_service import QUICK_PIN

            pins = list(QUICK_PIN)
        except Exception:
            pins = ["MU", "SNDK"]
        rest = [x for x in DEFAULT_WATCHLIST if x not in pins]
        st.session_state.watchlist = pins + rest
        save_watchlist(st.session_state.watchlist)
        st.rerun()
