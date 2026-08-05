"""Investment SOP page — action card when a symbol is selected."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_sop import build_trade_sop


def render_sop(
    symbol: str,
    period: str,
    interval: str,
    period_label: str,
    interval_label: str,
) -> None:
    st.markdown("## 投资 SOP")
    st.caption(
        f"标的 `{symbol}` · {period_label} / {interval_label} · "
        "选股后直接给出：适不适合入场、价位、胜率、稳定度、该做什么"
    )

    # HKD account → convert to USD for share sizing (US quotes are USD)
    HKD_PER_USD = 7.8
    c1, c2 = st.columns(2)
    with c1:
        capital_hkd = st.number_input(
            "账户本金 (HKD)",
            min_value=1000.0,
            value=float(st.session_state.get("sop_capital_hkd", 50_000.0)),
            step=1000.0,
            key="sop_capital_input",
            help="默认 50,000 HKD；仓位按约 7.8 HKD/USD 换成美元计价",
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
        f"约合 USD {capital_usd:,.0f}（{HKD_PER_USD} HKD/USD）· 1R ≈ HKD {capital_hkd * risk_pct / 100:,.0f}"
    )

    with st.spinner("生成实盘 SOP…"):
        sop = build_trade_sop(
            symbol,
            period=period,
            interval=interval,
            capital=float(capital_usd),
            risk_pct=float(risk_pct),
        )

    # ---- Verdict banner ----
    st.markdown("---")
    head_l, head_r = st.columns([1.4, 2])
    with head_l:
        if sop.enter_ok == "适合入场":
            st.success(f"### {sop.enter_ok}")
        elif sop.enter_ok == "谨慎试仓":
            st.warning(f"### {sop.enter_ok}")
        elif sop.enter_ok == "观望":
            st.info(f"### {sop.enter_ok}")
        else:
            st.error(f"### {sop.enter_ok}")
        st.metric("入场适合度", f"{sop.enter_score:.0f}/100")
        st.caption(f"{sop.name} · 现价 {sop.last_price if sop.last_price is not None else '—'}")
    with head_r:
        st.markdown(sop.summary)
        st.caption(f"方向 {sop.bias}（{sop.bias_score:+.0f}）· 机会 {sop.opportunity} · 侧 {sop.side}")

    # ---- KPI row ----
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "入场区",
        f"{sop.entry_low:.2f}" if sop.entry_low is not None else "—",
        f"至 {sop.entry_high:.2f}" if sop.entry_high is not None else None,
    )
    k2.metric(
        "建议挂单价",
        f"{sop.entry_plan:.2f}" if sop.entry_plan is not None else "—",
    )
    k3.metric(
        "止损",
        f"{sop.stop_loss:.2f}" if sop.stop_loss is not None else "—",
    )
    k4.metric(
        "T1 / T2",
        f"{sop.target_t1:.2f}" if sop.target_t1 is not None else "—",
        f"T2 {sop.target_t2:.2f}" if sop.target_t2 is not None else None,
    )
    k5.metric(
        "胜率",
        f"{sop.win_rate_pct:.0f}%" if sop.win_rate_pct is not None else "—",
        sop.win_rate_label,
    )
    k6.metric(
        "稳定度",
        f"{sop.stability_score:.0f}",
        sop.stability_label,
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("R:R (T1)", f"{sop.rr_t1:.2f}" if sop.rr_t1 is not None else "—")
    m2.metric(
        "期望 E[R]",
        f"{sop.expectancy_r:+.2f}R" if getattr(sop, "expectancy_r", None) is not None else "—",
    )
    m3.metric("建议股数", f"{sop.position_shares}")
    m4.metric("风险等级", sop.risk_level)
    m5.metric(
        "市场环境",
        getattr(sop, "regime_label", "—") or "—",
        f"{sop.regime_score:.0f}" if getattr(sop, "regime_score", None) is not None else None,
    )
    m6.metric(
        "流动性",
        getattr(sop, "liquidity_label", "—") or "—",
        f"{sop.liquidity_score:.0f}" if getattr(sop, "liquidity_score", None) is not None else None,
    )
    st.caption(sop.position_note)

    # ---- Market regime / quality context ----
    if getattr(sop, "regime_summary", None) or getattr(sop, "multi_rs_summary", None):
        with st.expander("市场环境 · 强弱 · 质量（免费数据）", expanded=True):
            if sop.regime_summary:
                st.markdown(sop.regime_summary)
            if sop.regime_bullets:
                for b in sop.regime_bullets[:6]:
                    st.caption(b)
            if sop.multi_rs_summary:
                st.markdown(sop.multi_rs_summary)
            if sop.quality_notes:
                st.markdown("**标的质量提示**")
                for q in sop.quality_notes[:5]:
                    st.caption(f"· {q}")
            if sop.news_summary:
                st.caption(sop.news_summary)
            if sop.scorecard_bullets:
                st.caption("评分分项：" + "；".join(sop.scorecard_bullets))
            if sop.data_sources:
                st.caption("数据源：" + " · ".join(sop.data_sources))
            st.caption(
                "可选免费 API：`FRED_API_KEY`（宏观）· `FINNHUB_API_KEY`（新闻条数）· "
                "`ALPHAVANTAGE_API_KEY`（基本面 OVERVIEW + 新闻情绪）。"
                "不设也能用 yfinance 的 VIX/SPY/10Y。AV 免费额度紧，已做长缓存。"
            )

    # ---- What to do ----
    st.markdown("### 现在该做什么")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**立刻执行**")
        if sop.actions_now:
            for i, line in enumerate(sop.actions_now, 1):
                st.markdown(f"{i}. {line}")
        else:
            st.write("—")
    with a2:
        st.markdown("**等待 / 条件**")
        if sop.actions_wait:
            for i, line in enumerate(sop.actions_wait, 1):
                st.markdown(f"{i}. {line}")
        else:
            st.write("—")
    st.error(f"**失效条件：** {sop.invalidation}")

    # ---- Checklist ----
    st.markdown("### SOP 检查清单")
    if sop.checklist:
        rows = []
        for c in sop.checklist:
            rows.append(
                {
                    "项": c.get("name", ""),
                    "结果": c.get("status", "").upper(),
                    "说明": c.get("detail", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("模型说明（实盘向）", expanded=False):
        st.markdown(
            """
**投资 SOP 如何算（偏真实下单决策）：**

| 字段 | 方法 |
|------|------|
| 适不适合入场 | 入场结构 + 多空 + 稳定度 + 路径胜率 + R:R + **期望E[R]** + **市场环境** + **流动性** |
| 入场价 | 技术结构买卖区；区内用现价，区外用中位限价 |
| 止损 / T1 T2 | ATR/结构止损 + 短中期目标 |
| 胜率 | 历史路径：先到 T1 再触止损（70%）+ 日线上涨日占比（30%） |
| 期望 E[R] | 胜率×盈亏比 − (1−胜率)×1R；负期望默认不应满仓做多 |
| 稳定度 | 波动、最大回撤、夏普、趋势强度 |
| 市场环境 | 免费：VIX + SPY 均线结构 + 10Y + HYG；可选 FRED 利差 |
| 可选 Alpha Vantage | OVERVIEW 补全基本面空字段 + NEWS_SENTIMENT 情绪（需免费 key） |
| 综合分 | 技术22 / 基本面22 / 风险18 / 多周期RS15 / 环境13 / 流动性10 |
| 股数 | 本金 × 1R% ÷ (入场−止损)，按手数取整 |

硬门槛示例：VIX/大盘避险、临近财报、流动性过差 → 不会给「适合入场」。

数据可能延迟。**非投顾、非保证收益。**
            """
        )
        for n in sop.notes:
            st.caption(f"· {n}")

    if sop.scorecard_total is not None:
        st.caption(
            f"综合评分卡：{sop.scorecard_stance} · {sop.scorecard_total:.0f} 分"
        )
