"""Streamlit page: 综合分析."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import analyze_bias
from charts import (
    drawdown_chart,
    price_volume_chart,
    relative_strength_chart,
    scorecard_radar,
    sr_chart,
)
from extra_analysis import (
    analyze_fundamentals,
    analyze_relative_strength,
    analyze_risk,
    analyze_support_resistance,
    analyze_trend,
    analyze_volume,
    build_scorecard,
    default_benchmark,
)
from indicators import enrich
from stock_service import cache_bucket, cached_info, fetch_history
from ui_mobile import plotly_chart as mobile_plotly
from views.common import fmt_number, fmt_pct, html_plain, render_bias_banner


def render_extra(symbol: str, period: str, interval: str, period_label: str, interval_label: str) -> None:
    st.header(f"综合分析 · `{symbol}`")
    st.caption(
        f"周期 {period_label} · {interval_label} · "
        "风险 / 支撑阻力 / 趋势 / 基本面 / 相对强弱 / 量价 / 评分卡"
    )

    with st.spinner("加载多维分析..."):
        info = cached_info(symbol, cache_bucket(5))
        hist = fetch_history(symbol, period=period, interval=interval)
        bench_sym, bench_label = default_benchmark(symbol)
        bench_hist = fetch_history(bench_sym, period=period, interval=interval)

    if hist.empty:
        st.warning(f"未能获取 `{symbol}` 数据。")
    else:
        df = enrich(hist)
        bias = analyze_bias(df)
        risk = analyze_risk(df)
        sr = analyze_support_resistance(df)
        trend = analyze_trend(df)
        funda = analyze_fundamentals(info)
        vol_rep = analyze_volume(df)
        rs = analyze_relative_strength(
            hist, bench_hist, benchmark=bench_sym, bench_label=bench_label
        )
        card = build_scorecard(bias.score, funda, risk, rs)

        # ---- Scorecard header ----
        st.subheader("📋 综合评分卡")
        sc1, sc2 = st.columns([1.2, 1.3])
        with sc1:
            mobile_plotly(
                scorecard_radar(
                    card.technical_score,
                    card.funda_score,
                    card.risk_score,
                    card.rs_score,
                ),
                use_container_width=True,
            )
        with sc2:
            stance_color = (
                "#ef5350"
                if "偏多" in card.stance
                else "#26a69a"
                if "偏空" in card.stance
                else "#78909c"
            )
            st.markdown(
                f"""
                <div class="bias-card">
                  <p class="bias-title" style="color:{stance_color};">
                    {card.stance}
                    <span style="font-size:1.1rem;color:#90a4ae;">
                      · {card.total_score:.0f} 分 · {card.total_grade}
                    </span>
                  </p>
                  <p class="bias-sub">{html_plain(card.summary)}</p>
                </div>
                """
                if card.total_score is not None
                else "<div class='bias-card'>评分数据不足</div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("技术", f"{card.technical_score:.0f}" if card.technical_score is not None else "—")
            m2.metric("基本面", f"{card.funda_score:.0f}" if card.funda_score is not None else "—")
            m3.metric("风险调整", f"{card.risk_score:.0f}" if card.risk_score is not None else "—")
            m4.metric("相对强弱", f"{card.rs_score:.0f}" if card.rs_score is not None else "—")
            render_bias_banner(bias, compact=True)

        tabs = st.tabs(
            ["⚠️ 风险", "🧱 支撑阻力", "📡 趋势结构", "🏢 基本面", "⚔️ 相对强弱", "📊 量价"]
        )

        # ---- Risk ----
        with tabs[0]:
            st.markdown(risk.summary.replace("**", ""))
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            r1.metric("区间收益", fmt_pct(risk.total_return_pct))
            r2.metric("年化收益(估)", fmt_pct(risk.ann_return_pct))
            r3.metric("年化波动", fmt_pct(risk.ann_vol_pct))
            r4.metric("最大回撤", fmt_pct(risk.max_drawdown_pct))
            r5.metric("夏普(估)", f"{risk.sharpe:.2f}" if risk.sharpe is not None else "—")
            r6.metric("风险等级", risk.risk_level)

            r7, r8, r9, r10 = st.columns(4)
            r7.metric("日VaR 95%", fmt_pct(risk.var_95_pct))
            r8.metric("Calmar", f"{risk.calmar:.2f}" if risk.calmar is not None else "—")
            r9.metric("胜率", fmt_pct(risk.win_rate_pct))
            r10.metric(
                "回撤区间",
                f"{risk.max_dd_start or '—'} → {risk.max_dd_end or '—'}",
            )
            if risk.equity_curve is not None and risk.drawdown_curve is not None:
                mobile_plotly(
                    drawdown_chart(risk.equity_curve, risk.drawdown_curve),
                    use_container_width=True,
                )
            st.caption(
                f"平均阳线 {fmt_pct(risk.avg_up_pct)} · 平均阴线 {fmt_pct(risk.avg_down_pct)}"
            )

        # ---- Support / Resistance ----
        with tabs[1]:
            st.markdown(sr.summary)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("现价", fmt_number(sr.last_price))
            s2.metric("最近支撑", fmt_number(sr.nearest_support), fmt_pct(sr.downside_pct))
            s3.metric("最近阻力", fmt_number(sr.nearest_resistance), fmt_pct(sr.upside_pct))
            s4.metric(
                "区间位置",
                f"{sr.position_in_range:.0%}" if sr.position_in_range is not None else "—",
            )
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Pivot", fmt_number(sr.pivot))
            p2.metric("R1", fmt_number(sr.r1))
            p3.metric("S1", fmt_number(sr.s1))
            p4.metric("R2", fmt_number(sr.r2))
            p5.metric("S2", fmt_number(sr.s2))

            mobile_plotly(
                sr_chart(df.tail(120), sr.levels, title=f"{symbol} 支撑/阻力"),
                use_container_width=True,
            )
            if sr.levels:
                lv_df = pd.DataFrame(
                    [
                        {
                            "价格": round(lv.price, 2),
                            "类型": lv.kind,
                            "强度": lv.strength,
                            "说明": lv.detail,
                        }
                        for lv in sorted(sr.levels, key=lambda x: -x.price)
                    ]
                )
                st.dataframe(lv_df, use_container_width=True, hide_index=True)

        # ---- Trend ----
        with tabs[2]:
            st.markdown(trend.summary.replace("**", ""))
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("短期趋势", trend.short_trend)
            t2.metric("中期趋势", trend.medium_trend)
            t3.metric("趋势强度", trend.strength_label)
            t4.metric(
                "强度指数",
                f"{trend.adx_proxy:.0f}" if trend.adx_proxy is not None else "—",
            )
            st.info(f"价格结构：{trend.structure}")
            hh = "是" if trend.higher_highs else "否" if trend.higher_highs is False else "—"
            hl = "是" if trend.higher_lows else "否" if trend.higher_lows is False else "—"
            st.caption(f"更高高点：{hh} · 更高低点：{hl}")
            mobile_plotly(
                price_volume_chart(
                    df,
                    title=f"{symbol} 趋势参考",
                    show_sma=True,
                    show_bb=False,
                    show_volume=True,
                ),
                use_container_width=True,
            )

        # ---- Fundamentals ----
        with tabs[3]:
            st.markdown(funda.summary.replace("**", ""))
            if not funda.available:
                st.warning("该标的在 Yahoo 上基本面字段较少（部分 A股/港股常见）。")
            else:
                f1, f2 = st.columns(2)
                f1.metric(
                    "基本面得分",
                    f"{funda.score:.0f}" if funda.score is not None else "—",
                )
                f2.metric("评级", funda.grade)
                rows = [
                    {
                        "指标": it.name,
                        "数值": it.display,
                        "倾向": it.verdict,
                        "说明": it.note,
                    }
                    for it in funda.items
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Company snapshot
            with st.expander("公司快照（原始字段）", expanded=False):
                snap_keys = [
                    ("longName", "名称"),
                    ("sector", "板块"),
                    ("industry", "行业"),
                    ("fullTimeEmployees", "员工数"),
                    ("website", "官网"),
                    ("longBusinessSummary", "简介"),
                ]
                for k, label in snap_keys:
                    if info.get(k):
                        st.markdown(f"**{label}**：{info.get(k)}")

        # ---- Relative strength ----
        with tabs[4]:
            st.markdown(rs.summary.replace("**", ""))
            st.caption(f"基准：`{bench_sym}`（{bench_label}）· 可按市场自动选择")
            x1, x2, x3, x4, x5 = st.columns(5)
            x1.metric("股票涨跌", fmt_pct(rs.stock_return_pct))
            x2.metric("基准涨跌", fmt_pct(rs.bench_return_pct))
            x3.metric("超额收益", fmt_pct(rs.alpha_pct))
            x4.metric("Beta", f"{rs.beta:.2f}" if rs.beta is not None else "—")
            x5.metric("相关性", f"{rs.corr:.2f}" if rs.corr is not None else "—")
            if rs.relative_curve is not None and not rs.relative_curve.empty:
                mobile_plotly(
                    relative_strength_chart(
                        rs.relative_curve,
                        title=f"{symbol} vs {bench_label}",
                    ),
                    use_container_width=True,
                )

        # ---- Volume ----
        with tabs[5]:
            st.markdown(vol_rep.summary.replace("**", ""))
            v1, v2, v3 = st.columns(3)
            v1.metric(
                "量比(vs20日均)",
                f"{vol_rep.vol_ratio:.2f}x" if vol_rep.vol_ratio is not None else "—",
            )
            v2.metric("量能状态", vol_rep.trend)
            v3.metric("量价关系", vol_rep.price_volume)
            st.caption(vol_rep.obv_trend)
            mobile_plotly(
                price_volume_chart(df, title=f"{symbol} 量价", show_sma=True, show_volume=True),
                use_container_width=True,
            )

        st.info(
            "综合分析整合技术多空、基本面、风险与相对强弱，仅供学习研究，**不构成投资建议**。"
        )
