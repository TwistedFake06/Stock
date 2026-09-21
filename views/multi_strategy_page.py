"""Streamlit: 多策略掃描 — aggregate entry lenses on watchlist."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from multi_strategy_scan import CORE_WATCHLIST, scan_symbols
from stock_service import DEFAULT_WATCHLIST, normalize_symbol


def render_multi_strategy_scan(period: str, interval: str, period_label: str) -> None:
    st.markdown("## 多策略掃描")
    st.caption(
        f"並行檢查 **看多／超賣／入場區／量能／趨勢對齊／MACD轉強** · {period_label} · "
        "命中越多越值得再入投資SOP · 非投資建議 · 實盤仍建議開市頭 2 小時掛單"
    )
    st.info(
        "同 Watchlist Confirm 互補：呢頁係 **多鏡頭集合**；"
        "Confirm 係完整 SOP 門檻。兩邊都靚先優先做。"
    )

    core_only = st.toggle(
        "只掃核心高勝率名單",
        value=bool(st.session_state.get("ms_core_only", True)),
        key="ms_core_only",
    )
    min_hits = st.select_slider(
        "最少命中策略數",
        options=[1, 2, 3, 4],
        value=int(st.session_state.get("ms_min_hits", 2)),
        key="ms_min_hits",
        help="預設 ≥2：減少單一指標噪音",
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
            _render_results(cached, min_hits)
        else:
            st.info("按「開始多策略掃描」對清單跑六種策略。")
        return

    progress = st.progress(0, text="多策略掃描中…")
    # scan_symbols does all; update progress coarsely
    results = []
    n = max(len(symbols), 1)
    # run one-by-one for progress UX
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

    results.sort(
        key=lambda r: (
            {"優先看": 0, "關注": 1, "觀察": 2}.get(r.suggest_tier, 9),
            -r.hit_count,
        )
    )
    st.session_state["ms_results"] = results
    _render_results(results, min_hits)


def _render_results(results, min_hits: int) -> None:
    if not results:
        st.warning("無結果。")
        return

    rows = []
    for r in results:
        if r.error:
            rows.append(
                {
                    "代码": r.symbol,
                    "建議": "—",
                    "命中數": 0,
                    "命中策略": "—",
                    "多空": "—",
                    "入場評估": "—",
                    "現價": r.last_price,
                    "主因": r.error,
                }
            )
            continue
        if r.hit_count < min_hits:
            continue
        rows.append(
            {
                "代码": r.symbol,
                "建議": r.suggest_tier,
                "命中數": r.hit_count,
                "命中策略": r.hit_labels,
                "多空": r.bias,
                "入場評估": r.entry_opportunity,
                "現價": r.last_price,
                "主因": r.primary_reason,
            }
        )

    n_pri = sum(1 for r in results if r.suggest_tier == "優先看")
    n_att = sum(1 for r in results if r.suggest_tier == "關注")
    c1, c2, c3 = st.columns(3)
    c1.metric("優先看", n_pri)
    c2.metric("關注", n_att)
    c3.metric(f"表上顯示（≥{min_hits} 命中）", len(rows))

    if not rows:
        st.warning(f"沒有 ≥{min_hits} 個策略同時命中嘅票。可調低「最少命中策略數」。")
    else:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=420)

    # detail + SOP jump
    interesting = [r for r in results if r.hit_count >= min_hits and not r.error]
    if interesting:
        st.markdown("### 明細")
        for r in interesting:
            title = f"{r.symbol} · {r.suggest_tier} · 命中 {r.hit_count}"
            with st.expander(title, expanded=(r.suggest_tier == "優先看")):
                st.caption(f"多空 **{r.bias}** · 入場評估 **{r.entry_opportunity}** · 現價 **{r.last_price}**")
                for h in r.hits:
                    mark = "✅" if h.fired else "·"
                    st.markdown(f"{mark} **{h.label}** — {h.reason}")
                if st.button(f"開投資SOP · {r.symbol}", key=f"ms_sop_{r.symbol}"):
                    st.session_state.symbol = r.symbol
                    st.session_state._pending_symbol = r.symbol
                    st.session_state._goto_sop = True
                    st.rerun()

    st.caption(
        "策略互不要求全部同意；命中數係「幾種鏡頭同時覺得有機會」。"
        "落單前仍用投資SOP／Confirm 規則同開市頭 2 小時紀律。"
    )
