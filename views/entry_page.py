"""Streamlit page: 入场与目标价."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import analyze_bias
from chart_entry import entry_target_chart
from entry_targets import analyze_entry, analyze_targets
from helper_analysis import analyze_helpers
from indicators import enrich
from stock_service import cache_bucket, cached_calendar, cached_info, fetch_history
from trade_plan import (
    analyze_events,
    build_trade_plan_card,
    calc_position,
    suggest_lot_size,
)
from ui_mobile import plotly_chart as mobile_plotly
from views.common import fmt_number, fmt_pct, html_plain, render_bias_banner


def render_entry(symbol: str, period: str, interval: str, period_label: str, interval_label: str) -> None:
    st.header(f"入场 · 仓位 · 目标价 · `{symbol}`")
    st.caption(
        f"周期 {period_label} · {interval_label} · "
        "超短≈1周 · 短期2周–1月 · 中期1–2月 · 含辅助分析与交易计划"
    )

    with st.spinner("评估入场 / 目标 / 事件 / 辅助..."):
        info = cached_info(symbol, cache_bucket(5))
        cal = cached_calendar(symbol, cache_bucket(30))
        hist = fetch_history(symbol, period=period, interval=interval)

    if hist.empty:
        st.warning(f"未能获取 `{symbol}` 数据。")
    else:
        df = enrich(hist)
        entry = analyze_entry(df)
        targets = analyze_targets(df, info=info, entry=entry)
        events = analyze_events(info, cal)
        bias = analyze_bias(df)
        helpers = analyze_helpers(df, entry, targets, bias=bias, events=events)
        entry.risk_reward_short = targets.rr_short
        entry.risk_reward_medium = targets.rr_medium
        company_name = (
            info.get("shortName") or info.get("longName") or ""
        )

        opp_color = {
            "较佳入场": "#ef5350",
            "可关注": "#ff8a65",
            "观望": "#78909c",
            "不宜追高": "#ffb74d",
            "偏空回避": "#26a69a",
        }.get(entry.opportunity, "#78909c")

        st.markdown(
            f"""
            <div class="bias-card">
              <p class="bias-title" style="color:{opp_color};">
                🎯 {entry.opportunity}
                <span style="font-size:1.05rem;color:#90a4ae;font-weight:500;">
                  · 机会分 {entry.score:.0f} · {entry.side_bias}
                </span>
              </p>
              <p class="bias-sub">{html_plain(entry.summary)}</p>
              <p class="bias-sub">{html_plain(helpers.one_liner)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if events.caution:
            st.warning(events.caution)
        elif events.summary:
            st.caption(events.summary)
        st.caption(helpers.week_focus)

        # Target ladder metrics
        t1, t2, t3, t4, t5, t6, t7 = st.columns(7)
        t1.metric("现价", fmt_number(entry.current_price))
        t2.metric(
            "一周上看",
            fmt_number(targets.ultra.bull_target if targets.ultra else None),
            fmt_pct(targets.ultra.upside_pct if targets.ultra else None),
        )
        t3.metric(
            "短期上看",
            fmt_number(targets.short.bull_target),
            fmt_pct(targets.short.upside_pct),
        )
        t4.metric(
            "中期上看",
            fmt_number(targets.medium.bull_target),
            fmt_pct(targets.medium.upside_pct),
        )
        t5.metric(
            "止损",
            fmt_number(entry.stop_loss),
            (
                f"{(entry.stop_loss / entry.current_price - 1) * 100:+.1f}%"
                if entry.stop_loss and entry.current_price
                else None
            ),
        )
        t6.metric(
            "超短R:R",
            f"{targets.rr_ultra:.2f}" if targets.rr_ultra is not None else "—",
        )
        t7.metric(
            "中线R:R",
            f"{targets.rr_medium:.2f}" if targets.rr_medium is not None else "—",
        )

        e2, e3, e4 = st.columns(3)
        e2.metric("建议买区下", fmt_number(entry.suggested_entry_low))
        e3.metric("建议买区上", fmt_number(entry.suggested_entry_high))
        e4.metric(
            "短线R:R",
            f"{targets.rr_short:.2f}" if targets.rr_short is not None else "—",
        )

        mobile_plotly(
            entry_target_chart(
                df,
                entry.suggested_entry_low,
                entry.suggested_entry_high,
                entry.stop_loss,
                targets.short.bull_target,
                targets.short.bear_target,
                targets.medium.bull_target,
                targets.medium.bear_target,
                title=f"{symbol} · 入场 / 止损 / 一周·短·中目标",
                ultra_bull=targets.ultra.bull_target if targets.ultra else None,
                ultra_bear=targets.ultra.bear_target if targets.ultra else None,
            ),
            width="stretch",
        )

        # ========== 实用工具 ==========
        tab_help, tab_pos, tab_plan, tab_evt, tab_tgt, tab_sig = st.tabs(
            [
                "🧭 辅助分析",
                "💰 仓位计算",
                "📋 交易计划卡",
                "📅 事件提醒",
                "🎯 目标价明细",
                "📌 信号与清单",
            ]
        )

        with tab_help:
            st.subheader("本周怎么做")
            st.info(helpers.week_focus)
            st.markdown(f"**一句话：** {helpers.one_liner}")
            if helpers.extension_note:
                st.caption(helpers.extension_note)

            st.markdown("**操作剧本**")
            for i, step in enumerate(helpers.playbook, 1):
                st.markdown(f"{i}. {step}")

            st.markdown("**分批止盈阶梯**")
            if helpers.take_profits:
                tp_rows = [
                    {
                        "阶段": s.label,
                        "价格": s.price,
                        "相对入场%": f"{s.pct_from_entry:+.2f}%" if s.pct_from_entry is not None else "—",
                        "建议动作": s.action,
                    }
                    for s in helpers.take_profits
                ]
                st.dataframe(pd.DataFrame(tp_rows), width="stretch", hide_index=True)

            st.markdown("**本周观察清单**")
            if helpers.watchlist:
                w_rows = [
                    {
                        "类型": w.kind,
                        "优先级": w.level,
                        "标题": w.title,
                        "说明": w.detail,
                    }
                    for w in helpers.watchlist
                ]
                st.dataframe(pd.DataFrame(w_rows), width="stretch", hide_index=True)

            st.markdown("**附近关键价位（按距现价排序）**")
            if helpers.key_levels:
                st.dataframe(
                    pd.DataFrame(helpers.key_levels),
                    width="stretch",
                    hide_index=True,
                )
            st.caption("辅助分析帮助把「看多/看空」落成可执行清单，仍非投资建议。")

        default_lot = suggest_lot_size(symbol)
        mid_entry = None
        if entry.suggested_entry_low and entry.suggested_entry_high:
            mid_entry = (entry.suggested_entry_low + entry.suggested_entry_high) / 2
        elif entry.current_price:
            mid_entry = entry.current_price

        # ---- Position ----
        with tab_pos:
            st.markdown("按 **单笔最大亏损占本金比例** 计算建议股数（风险仓位法）。")
            p1, p2, p3, p4 = st.columns(4)
            capital = p1.number_input(
                "交易本金",
                min_value=1000.0,
                value=100_000.0,
                step=10_000.0,
                help="可用于交易的总资金",
            )
            risk_pct = p2.number_input(
                "单笔风险 %",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="止损打到时，最多亏本金的百分之几（常见 0.5%–2%）",
            )
            entry_px = p3.number_input(
                "计划入场价",
                min_value=0.01,
                value=float(mid_entry or entry.current_price or 1.0),
                step=0.01,
                format="%.2f",
            )
            stop_px = p4.number_input(
                "计划止损价",
                min_value=0.01,
                value=float(entry.stop_loss or (entry_px * 0.95)),
                step=0.01,
                format="%.2f",
            )
            lot = st.number_input(
                "每手股数（A股/港股常 100）",
                min_value=1,
                value=int(default_lot),
                step=1,
            )

            effective_risk = risk_pct
            if events.near_earnings:
                half = st.checkbox(
                    "财报临近：风险预算减半（推荐）",
                    value=True,
                    help="14 天内有财报时，默认把单笔风险砍半控制事件波动",
                )
                if half:
                    effective_risk = risk_pct * 0.5
                    st.caption(f"实际用于计算的风险：{effective_risk:.2f}%")

            pos = calc_position(
                capital=capital,
                risk_pct=effective_risk,
                entry_price=entry_px,
                stop_price=stop_px,
                short_target=targets.short.bull_target,
                medium_target=targets.medium.bull_target,
                lot_size=int(lot),
            )
            # show ultra target reward if available
            if (
                pos.valid
                and pos.shares > 0
                and targets.ultra
                and targets.ultra.bull_target
                and targets.ultra.bull_target > entry_px
            ):
                ultra_reward = pos.shares * (targets.ultra.bull_target - entry_px)
                ultra_rr = (targets.ultra.bull_target - entry_px) / pos.risk_per_share if pos.risk_per_share else None
            else:
                ultra_reward = ultra_rr = None

            if not pos.valid:
                st.error(pos.error)
            else:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("建议股数", f"{pos.shares:,}")
                c2.metric("仓位市值", f"{pos.position_value:,.0f}")
                c3.metric("占本金", f"{pos.position_pct_of_capital:.1f}%")
                c4.metric("风险金额", f"{pos.risk_amount:,.0f}")
                c5.metric("每股风险", f"{pos.risk_per_share:.2f}")

                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("止损约亏%", f"{pos.max_loss_pct:.2f}%")
                r2.metric(
                    "一周 R:R",
                    f"{ultra_rr:.2f}" if ultra_rr is not None else "—",
                )
                r3.metric(
                    "到一周目标约盈",
                    f"{ultra_reward:,.0f}" if ultra_reward is not None else "—",
                )
                r4.metric(
                    "到短目标约盈",
                    f"{pos.reward_short:,.0f}" if pos.reward_short is not None else "—",
                )
                r5.metric(
                    "到中目标约盈",
                    f"{pos.reward_medium:,.0f}" if pos.reward_medium is not None else "—",
                )

                # Simple risk bar
                st.progress(
                    min(1.0, pos.position_pct_of_capital / 100.0),
                    text=f"仓位占用本金 {pos.position_pct_of_capital:.1f}%",
                )

                for n in pos.notes:
                    st.warning(n) if ("不足" in n or "偏高" in n or "< 1" in n) else st.info(n)

                st.caption(
                    f"公式：股数 = (本金 × 风险%) ÷ (入场价 − 止损价)，再按 {lot} 股取整。"
                )

            # stash for plan tab via session
            st.session_state["_last_position"] = pos
            st.session_state["_last_entry_px"] = entry_px
            st.session_state["_last_stop_px"] = stop_px

        # ---- Trade plan card ----
        with tab_plan:
            pos_for_plan = st.session_state.get("_last_position")
            # Rebuild position with current targets if missing
            if pos_for_plan is None and mid_entry and entry.stop_loss:
                pos_for_plan = calc_position(
                    100_000.0,
                    1.0,
                    float(mid_entry),
                    float(entry.stop_loss),
                    targets.short.bull_target,
                    targets.medium.bull_target,
                    default_lot,
                )

            # Allow override entry/stop from position tab already in session;
            # optionally recompute plan entry using user prices
            plan_entry = entry
            # If user set custom prices, annotate in card via position fields

            card = build_trade_plan_card(
                symbol=symbol,
                entry=plan_entry,
                targets=targets,
                position=pos_for_plan if pos_for_plan and pos_for_plan.valid else None,
                events=events,
                name=company_name,
            )

            st.markdown("**一键交易计划**（可复制保存）")
            st.code(card.text, language=None)

            # Download as text file
            st.download_button(
                label="⬇️ 下载计划卡 (.txt)",
                data=card.text.encode("utf-8"),
                file_name=f"trade_plan_{symbol.replace('.', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )

            st.markdown("**摘要**")
            st.write(" · ".join(card.bullets))

            # Action recommendation box
            if entry.opportunity in ("较佳入场", "可关注"):
                st.success(
                    f"系统倾向：**可制定做多计划**（{entry.opportunity}）。"
                    "建议只在买区内分批，严格止损。"
                    + (" 财报临近请减仓或观望。" if events.near_earnings else "")
                )
            elif entry.opportunity == "不宜追高":
                st.warning("系统倾向：**先不追**，等回踩买区或指标冷却。")
            elif entry.opportunity == "偏空回避":
                st.error("系统倾向：**回避做多**，优先观望或仅极小仓试错。")
            else:
                st.info("系统倾向：**观望**，等待更清晰的回踩/突破信号。")

        # ---- Events ----
        with tab_evt:
            st.markdown(events.summary)
            if events.items:
                ev_df = pd.DataFrame(
                    [
                        {
                            "事件": it.name,
                            "日期": it.when.isoformat() if it.when else "—",
                            "剩余天数": it.days_left if it.days_left is not None else "—",
                            "关注度": it.level,
                            "说明": it.detail,
                        }
                        for it in events.items
                    ]
                )
                st.dataframe(ev_df, width="stretch", hide_index=True)
            else:
                st.info("暂无财报/除息日期（部分 A股/港股字段可能缺失）。")

            if events.near_earnings:
                st.error(
                    "财报窗口建议：\n"
                    "1. 单笔风险降至平时的 50% 或更低\n"
                    "2. 避免在财报前一夜重仓\n"
                    "3. 短线目标可提前止盈，不必扛过财报\n"
                    "4. 若已有浮盈，可先减仓锁定部分利润"
                )
            st.caption("事件数据来自 Yahoo Finance calendar / info，可能有延迟或估计日期。")

        # ---- Targets detail ----
        with tab_tgt:
            st.markdown(targets.summary)

            def _horizon_block(h, title_prefix: str):
                st.markdown(f"### {title_prefix}（{h.horizon_note}）")
                a, b, c = st.columns(3)
                a.metric("看多目标", fmt_number(h.bull_target), fmt_pct(h.upside_pct))
                b.metric("中性目标", fmt_number(h.base_target))
                c.metric("看空/下看", fmt_number(h.bear_target), fmt_pct(h.downside_pct))
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "方法": m.name,
                                "价格": round(m.price, 2),
                                "方向": m.side,
                                "权重": m.weight,
                                "说明": m.detail,
                            }
                            for m in h.methods
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

            _horizon_block(targets.ultra, "⚡ 超短 · 约1周")
            st.divider()
            c_short, c_med = st.columns(2)
            with c_short:
                _horizon_block(targets.short, "短期")
            with c_med:
                _horizon_block(targets.medium, "中期")
                if targets.analyst_target is not None:
                    st.caption(
                        f"分析师目标均价：{targets.analyst_target:.2f}"
                        f"（{targets.analyst_upside_pct:+.1f}%）"
                    )

        # ---- Signals ----
        with tab_sig:
            st.subheader("入场信号明细")
            if entry.signals:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "信号": s.name,
                                "类型": s.side,
                                "质量": s.score,
                                "紧迫度": s.urgency,
                                "说明": s.detail,
                                "区间下": s.entry_zone_low,
                                "区间上": s.entry_zone_high,
                            }
                            for s in entry.signals
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            st.subheader("操作检查清单")
            for item in entry.checklist:
                st.markdown(f"- {item}")
            st.warning(entry.invalidation)

        st.info(
            "仓位、目标价与计划卡均为技术规则推算，**不是保证收益**。"
            "仅供学习研究，**不构成投资建议**。"
        )
