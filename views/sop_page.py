"""Investment SOP page — short-term swing (0–2w / 2–4w) decision card."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from stock_service import cache_bucket, cached_info
from trade_sop import build_trade_sop
from views.common import render_session_quote_card


def _verdict_box(verdict: str):
    if verdict == "可以入場":
        return st.success
    if verdict == "可以試倉":
        return st.warning
    if verdict == "不做多":
        return st.error
    return st.info


def _render_horizon_card(h, capital_note: str) -> None:
    """One swing horizon: entry / stop / target / win rate."""
    if h is None:
        st.caption("无该周期数据")
        return
    box = _verdict_box(h.verdict)
    box(f"### {h.label} · {h.verdict}")
    st.caption(h.note or f"路径窗口约 {h.bars} 个交易日")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "入場價（区）",
        f"{h.entry_low:.2f}" if h.entry_low is not None else "—",
        f"至 {h.entry_high:.2f}" if h.entry_high is not None else None,
    )
    k2.metric(
        "建議掛單",
        f"{h.entry_plan:.2f}" if h.entry_plan is not None else "—",
    )
    k3.metric(
        "止蝕價",
        f"{h.stop_loss:.2f}" if h.stop_loss is not None else "—",
    )
    k4.metric(
        "目標價",
        f"{h.target:.2f}" if h.target is not None else "—",
    )
    k5.metric(
        "勝率",
        f"{h.win_rate_pct:.0f}%" if h.win_rate_pct is not None else "—",
        f"R:R {h.rr:.2f}" if h.rr is not None else None,
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("期望 E[R]", f"{h.expectancy_r:+.2f}" if h.expectancy_r is not None else "—")
    r2.metric(
        "每股風險",
        f"{h.risk_per_share:.2f}" if h.risk_per_share is not None else "—",
    )
    r3.metric(
        "每股潛在賺",
        f"{h.reward_per_share:.2f}" if h.reward_per_share is not None else "—",
    )
    st.caption(capital_note)


def render_sop(
    symbol: str,
    period: str,
    interval: str,
    period_label: str,
    interval_label: str,
) -> None:
    st.markdown("## 短线波段计划")
    st.caption(
        f"`{symbol}` · {period_label}/{interval_label} · "
        "**0–2周 / 2–4周**：能否入場 · 勝率 · 入場價 · 止蝕 · 目標價"
    )

    HKD_PER_USD = 7.8
    c1, c2 = st.columns(2)
    with c1:
        capital_hkd = st.number_input(
            "账户本金 (HKD)",
            min_value=1000.0,
            value=float(st.session_state.get("sop_capital_hkd", 50_000.0)),
            step=1000.0,
            key="sop_capital_input",
        )
    with c2:
        risk_pct = st.number_input(
            "单笔风险 1R (%)",
            min_value=0.25,
            max_value=5.0,
            value=float(st.session_state.get("sop_risk_pct", 1.0)),
            step=0.25,
            key="sop_risk_input",
        )
    st.session_state["sop_capital_hkd"] = capital_hkd
    st.session_state["sop_risk_pct"] = risk_pct
    capital_usd = float(capital_hkd) / HKD_PER_USD
    st.caption(
        f"约合 USD {capital_usd:,.0f} · 1R ≈ HKD {capital_hkd * risk_pct / 100:,.0f} · "
        "谨慎试仓自动用 0.5R 算股数"
    )

    with st.spinner("分析走势与短线计划…"):
        sop = build_trade_sop(
            symbol,
            period=period,
            interval=interval,
            capital=float(capital_usd),
            risk_pct=float(risk_pct),
        )

    h1 = getattr(sop, "swing_h1", None)
    h2 = getattr(sop, "swing_h2", None)

    # ---- Primary: dual horizon ----
    st.markdown("---")
    st.markdown(f"### {sop.name} · 现价 **{sop.last_price if sop.last_price is not None else '—'}**")
    if getattr(sop, "trend_note", None):
        st.markdown(sop.trend_note)
    st.markdown(sop.summary)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            _render_horizon_card(
                h1,
                capital_note=f"建议股数（按本计划风险）约 **{sop.position_shares}** · {sop.position_note}",
            )
    with right:
        with st.container(border=True):
            _render_horizon_card(
                h2,
                capital_note="2–4周目标通常更远；若只做 0–2周，以左卡目标先减仓。",
            )

    # ---- What to do now (action language) ----
    st.markdown("### 现在怎么做")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**立刻**")
        if sop.actions_now:
            for i, line in enumerate(sop.actions_now, 1):
                st.markdown(f"{i}. {line}")
        else:
            st.write("—")
    with a2:
        st.markdown("**等待条件**")
        if sop.actions_wait:
            for i, line in enumerate(sop.actions_wait, 1):
                st.markdown(f"{i}. {line}")
        else:
            st.write("—")
    st.error(f"**失效 / 止蚀逻辑：** {sop.invalidation}")

    # ---- Compact price strip ----
    st.markdown("### 价格一览")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("现价", f"{sop.last_price:.2f}" if sop.last_price is not None else "—")
    p2.metric(
        "入場低",
        f"{sop.entry_low:.2f}" if sop.entry_low is not None else "—",
    )
    p3.metric(
        "入場高",
        f"{sop.entry_high:.2f}" if sop.entry_high is not None else "—",
    )
    p4.metric(
        "止蝕",
        f"{sop.stop_loss:.2f}" if sop.stop_loss is not None else "—",
    )
    p5.metric(
        "目标0–2周",
        f"{sop.target_t1:.2f}" if sop.target_t1 is not None else "—",
    )
    p6.metric(
        "目标2–4周",
        f"{sop.target_t2:.2f}" if sop.target_t2 is not None else "—",
    )

    # Session + edge collapsed
    with st.expander("交易时段 · 盘前盘后", expanded=False):
        try:
            info = cached_info(symbol, cache_bucket(2))
        except Exception:
            info = {}
        render_session_quote_card(symbol, info)

    with st.expander("辅助：跟势 / 假突破 / 板块 / 量能 / IV", expanded=True):
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric(
            "跟势",
            getattr(sop, "trend_align_label", "—") or "—",
            f"{sop.trend_align_score:.0f}"
            if getattr(sop, "trend_align_score", None) is not None
            else None,
        )
        e2.metric(
            "假突破",
            getattr(sop, "false_break_label", "—") or "—",
            f"{sop.false_break_score:.0f}"
            if getattr(sop, "false_break_score", None) is not None
            else None,
        )
        e3.metric(
            "板块 RS",
            getattr(sop, "sector_rs_label", "—") or "—",
            f"{sop.sector_rs_score:.0f}"
            if getattr(sop, "sector_rs_score", None) is not None
            else None,
        )
        e4.metric(
            "量能",
            getattr(sop, "volume_confirm_label", "—") or "—",
            f"{sop.volume_confirm_score:.0f}"
            if getattr(sop, "volume_confirm_score", None) is not None
            else None,
        )
        e5.metric(
            "IV",
            getattr(sop, "iv_label", "—") or "—",
            f"{sop.iv_score:.0f}" if getattr(sop, "iv_score", None) is not None else None,
        )
        if getattr(sop, "against_trend", False):
            st.warning("逆势警告：大盘/板块偏空时个股硬多 → 短线优先暂缓")
        if getattr(sop, "false_break_risk", False):
            st.warning("假突破风险：近日刺破前高后回落 → 不追多")
        for txt in (
            getattr(sop, "trend_align_summary", ""),
            getattr(sop, "false_break_summary", ""),
            getattr(sop, "sector_rs_summary", ""),
            getattr(sop, "volume_confirm_summary", ""),
            getattr(sop, "iv_summary", ""),
            getattr(sop, "regime_summary", ""),
        ):
            if txt:
                st.caption(txt)

    st.markdown("### 检查清单")
    if sop.checklist:
        rows = [
            {
                "项": c.get("name", ""),
                "结果": c.get("status", "").upper(),
                "说明": c.get("detail", ""),
            }
            for c in sop.checklist
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("怎么读这张卡（短线）", expanded=False):
        st.markdown(
            """
**你要的输出（0–2周 / 2–4周）：**

| 字段 | 含义 |
|------|------|
| **能否入場** | 可以入場 / 可以試倉 / 暫緩觀望 / 不做多 |
| **入場價** | 结构买区（限价挂这里，不追高） |
| **止蝕價** | 破了计划作废，认亏离场 |
| **目標價** | 该周期看多目标；到了先减仓 |
| **勝率** | 历史上「同样止蚀距离与目标距离」先碰到目标的比例（模型估算） |
| **R:R** | 赚：亏；太低则暂缓 |

**操作习惯建议：**
1. 只在入場区内限价  
2. 先设止蝕再挂单  
3. 0–2周到目标减半仓；剩仓看 2–4周  
4. 胜率是历史统计，不是保证  

非投资建议。
            """
        )
        for n in sop.notes:
            st.caption(f"· {n}")
