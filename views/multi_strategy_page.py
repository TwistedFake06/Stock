"""Streamlit: 多策略掃描 — aggregate bullish + bearish entry lenses on watchlist."""
from __future__ import annotations

from entry_labels import ENTER_MAYBE, ENTER_NO, ENTER_YES, label_multi_tier

import pandas as pd
import streamlit as st

from multi_strategy_scan import (
    BEAR_TIER_NONE,
    BEAR_TIER_STRONG,
    BEAR_TIER_WATCH,
    CORE_WATCHLIST,
    scan_symbols,
)
from stock_service import DEFAULT_WATCHLIST, normalize_symbol


def render_multi_strategy_scan(period: str, interval: str, period_label: str) -> None:
    st.markdown("## 多策略掃描")
    st.caption(
        f"並行 **6 看多**（看多／超賣／入場區／量能／趨勢對齊／MACD轉強）+ "
        f"**6 看空**（看空／超買／壓力／回避區／量能偏空／趨勢逆勢／MACD轉弱）· {period_label} · "
        "看多／看空命中分開計 · 非投資建議 · 實盤仍建議開市頭 2 小時掛單"
    )
    st.info(
        "同 Watchlist 互補：呢頁係 **多鏡頭集合**（多空各 6）；"
        "可入場 係完整 SOP 門檻。兩邊都靚先優先做。"
        "偏空分級只係回避提示，唔等同做空指令。"
    )

    core_only = st.toggle(
        "只掃核心高勝率名單",
        value=bool(st.session_state.get("ms_core_only", True)),
        key="ms_core_only",
    )
    min_hits = st.select_slider(
        "最少看多命中數",
        options=[1, 2, 3, 4],
        value=int(st.session_state.get("ms_min_hits", 2)),
        key="ms_min_hits",
        help="預設 ≥2：減少單一指標噪音（只過濾看多側）",
    )
    min_bear = st.select_slider(
        "最少看空命中數（可選過濾）",
        options=[0, 1, 2, 3, 4],
        value=int(st.session_state.get("ms_min_bear", 0)),
        key="ms_min_bear",
        help="0＝唔過濾看空；≥1 時只顯示達標嘅偏空票（可同看多表並存）",
    )
    sort_by = st.radio(
        "排序",
        ["看多優先", "看空優先"],
        horizontal=True,
        key="ms_sort_by",
    )

    session_wl = list(st.session_state.get("watchlist") or DEFAULT_WATCHLIST)
    src = st.radio(
        "清單",
        ["核心名單", "App 自選股"],
        horizontal=True,
        index=0 if core_only else 1,
        key="ms_list_src",
    )
    if src == "核心名單":
        symbols = list(CORE_WATCHLIST)
        core_only = True
    else:
        symbols = [normalize_symbol(s) for s in session_wl if normalize_symbol(s)]
        core_only = False

    st.caption(
        f"將掃描 **{len(symbols)}** 隻："
        + ", ".join(symbols[:14])
        + ("…" if len(symbols) > 14 else "")
    )

    if not st.button("開始多策略掃描", type="primary", width="stretch", key="ms_run"):
        cached = st.session_state.get("ms_results")
        if cached:
            st.caption("顯示上次掃描快取；要最新請再按開始。")
            _render_results(cached, min_hits, min_bear, sort_by)
        else:
            st.info("按「開始多策略掃描」對清單跑 6 看多 + 6 看空策略。")
        return

    progress = st.progress(0, text="多策略掃描中…")
    results = []
    from multi_strategy_scan import evaluate_strategies

    core_set = {normalize_symbol(s) for s in CORE_WATCHLIST}
    cleaned = []
    seen = set()
    for s in symbols:
        nsym = normalize_symbol(s)
        if not nsym or nsym in seen:
            continue
        if core_only and nsym not in core_set:
            continue
        seen.add(nsym)
        cleaned.append(nsym)
    if core_only and not cleaned:
        cleaned = [normalize_symbol(s) for s in CORE_WATCHLIST]

    for i, sym in enumerate(cleaned):
        results.append(evaluate_strategies(sym, period=period, interval=interval))
        progress.progress((i + 1) / len(cleaned), text=f"掃描 {sym}…")
    progress.empty()

    results = _sort_results(results, sort_by)
    st.session_state["ms_results"] = results
    _render_results(results, min_hits, min_bear, sort_by)


def _sort_results(results, sort_by: str):
    if sort_by == "看空優先":
        return sorted(
            results,
            key=lambda r: (
                {BEAR_TIER_STRONG: 0, BEAR_TIER_WATCH: 1, BEAR_TIER_NONE: 2}.get(
                    getattr(r, "bear_tier", BEAR_TIER_NONE), 9
                ),
                -getattr(r, "bear_hit_count", 0),
                -r.hit_count,
            ),
        )
    return sorted(
        results,
        key=lambda r: (
            {ENTER_YES: 0, ENTER_MAYBE: 1, ENTER_NO: 2}.get(r.suggest_tier, 9),
            -r.hit_count,
            -getattr(r, "bear_hit_count", 0),
        ),
    )


def _render_results(results, min_hits: int, min_bear: int = 0, sort_by: str = "看多優先") -> None:
    if not results:
        st.warning("無結果。")
        return

    results = _sort_results(results, sort_by)

    rows = []
    for r in results:
        if r.error:
            rows.append(
                {
                    "代码": r.symbol,
                    "建議": "—",
                    "看多命中": 0,
                    "看多策略": "—",
                    "偏空分級": "—",
                    "看空命中": 0,
                    "看空策略": "—",
                    "多空": "—",
                    "入場評估": "—",
                    "現價": r.last_price,
                    "主因": r.error,
                }
            )
            continue
        bull_ok = r.hit_count >= min_hits
        bear_ok = getattr(r, "bear_hit_count", 0) >= min_bear if min_bear > 0 else False
        # min_bear==0: legacy bull-only filter; else show if bull OR bear threshold met
        if min_bear == 0:
            if not bull_ok:
                continue
        elif not (bull_ok or bear_ok):
            continue
        rows.append(
            {
                "代码": r.symbol,
                "建議": r.suggest_tier,
                "看多命中": r.hit_count,
                "看多策略": r.hit_labels,
                "偏空分級": getattr(r, "bear_tier", BEAR_TIER_NONE),
                "看空命中": getattr(r, "bear_hit_count", 0),
                "看空策略": getattr(r, "bear_hit_labels", "—"),
                "多空": r.bias,
                "入場評估": r.entry_opportunity,
                "現價": r.last_price,
                "主因": r.primary_reason,
            }
        )

    n_pri = sum(1 for r in results if r.suggest_tier == ENTER_YES)
    n_att = sum(1 for r in results if r.suggest_tier == ENTER_MAYBE)
    n_bear_s = sum(1 for r in results if getattr(r, "bear_tier", "") == BEAR_TIER_STRONG)
    n_bear_w = sum(1 for r in results if getattr(r, "bear_tier", "") == BEAR_TIER_WATCH)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("可入場", n_pri)
    c2.metric("可考慮", n_att)
    c3.metric("偏空強", n_bear_s)
    c4.metric("偏空留意", n_bear_w)
    c5.metric("表上顯示", len(rows))

    if not rows:
        st.warning(
            f"沒有符合過濾嘅票（看多≥{min_hits}"
            + (f" 或 看空≥{min_bear}" if min_bear else "")
            + "）。可調低命中門檻。"
        )
    else:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=420)

    # detail + SOP jump
    interesting = []
    for r in results:
        if r.error:
            continue
        if min_bear == 0:
            if r.hit_count >= min_hits:
                interesting.append(r)
        else:
            if r.hit_count >= min_hits or getattr(r, "bear_hit_count", 0) >= min_bear:
                interesting.append(r)
    if interesting:
        st.markdown("### 明細")
        for r in interesting:
            bt = getattr(r, "bear_tier", BEAR_TIER_NONE)
            title = (
                f"{r.symbol} · {r.suggest_tier} · 看多 {r.hit_count}"
                f" · {bt} · 看空 {getattr(r, 'bear_hit_count', 0)}"
            )
            with st.expander(title, expanded=(r.suggest_tier == ENTER_YES)):
                st.caption(
                    f"多空 **{r.bias}** · 入場評估 **{r.entry_opportunity}** · "
                    f"現價 **{r.last_price}**"
                )
                st.markdown("**看多鏡頭**")
                for h in r.hits:
                    mark = "✅" if h.fired else "·"
                    st.markdown(f"{mark} **{h.label}** — {h.reason}")
                st.markdown("**看空鏡頭**")
                for h in getattr(r, "bear_hits", []) or []:
                    mark = "🔻" if h.fired else "·"
                    st.markdown(f"{mark} **{h.label}** — {h.reason}")
                if getattr(r, "bear_primary_reason", ""):
                    st.caption(f"偏空主因：{r.bear_primary_reason}")
                if st.button(f"開投資SOP · {r.symbol}", key=f"ms_sop_{r.symbol}"):
                    st.session_state.symbol = r.symbol
                    st.session_state._pending_symbol = r.symbol
                    st.session_state._goto_sop = True
                    st.rerun()

    st.caption(
        "策略互不要求全部同意；命中數係「幾種鏡頭同時覺得有機會／風險」。"
        "看多命中同看空命中分開計，唔會混入同一 hit_count。"
        "落單前仍用投資SOP／可入場規則同開市頭 2 小時紀律。"
    )
