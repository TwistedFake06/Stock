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

    c1, c2 = st.columns(2)
    with c1:
        capital = st.number_input(
            "账户本金 (USD)",
            min_value=1000.0,
            value=float(st.session_state.get("sop_capital", 50_000.0)),
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
    st.session_state["sop_capital"] = capital
    st.session_state["sop_risk_pct"] = risk_pct

    with st.spinner("生成实盘 SOP…"):
        sop = build_trade_sop(
            symbol,
            period=period,
            interval=interval,
            capital=float(capital),
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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R:R (T1)", f"{sop.rr_t1:.2f}" if sop.rr_t1 is not None else "—")
    m2.metric("建议股数", f"{sop.position_shares}")
    m3.metric("风险等级", sop.risk_level)
    m4.metric(
        "最大回撤",
        f"{sop.max_dd_pct:.1f}%" if sop.max_dd_pct is not None else "—",
    )
    st.caption(sop.position_note)

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

    with st.expander("模型说明（简）", expanded=False):
        st.markdown(
            """
**投资 SOP（实盘辅助）如何算出：**

| 字段 | 方法 |
|------|------|
| 适不适合入场 | 入场评级 + 多空分 + 稳定度 + 路径胜率 + R:R |
| 入场价 | 技术结构给出的买卖区；区内用现价，区外用中位挂单价 |
| 止损 / T1 T2 | ATR/结构止损 + 短中期目标 |
| 胜率 | 历史路径：先到 T1 再触止损 的比例，混合日线胜率 |
| 稳定度 | 波动、最大回撤、夏普、趋势强度 |
| 股数 | 本金 × 1R% ÷ (入场−止损)，按手数取整 |

数据源 Yahoo 可能延迟。**非投顾、非保证收益。**
            """
        )
        for n in sop.notes:
            st.caption(f"· {n}")

    if sop.scorecard_total is not None:
        st.caption(
            f"综合评分卡：{sop.scorecard_stance} · {sop.scorecard_total:.0f} 分"
        )
