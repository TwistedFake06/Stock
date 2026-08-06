"""Investment SOP — short-term swing plan + exit rules + journal."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from position_coach import advise_open_position
from stock_service import cache_bucket, cached_info
from trade_journal import add_trade, close_trade, journal_stats, load_trades
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


def _render_horizon_card(h, *, is_primary: bool, capital_note: str) -> None:
    if h is None:
        st.caption("无该周期数据")
        return
    box = _verdict_box(h.verdict)
    title = f"### {'⭐ 主周期 · ' if is_primary else ''}{h.label} · {h.verdict}"
    box(title)
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
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("纸面 E[R]", f"{h.expectancy_r:+.2f}" if h.expectancy_r is not None else "—")
    r2.metric("净 R:R", f"{h.rr_net:.2f}" if getattr(h, "rr_net", None) is not None else "—")
    r3.metric(
        "净 E[R]",
        f"{h.expectancy_net:+.2f}" if getattr(h, "expectancy_net", None) is not None else "—",
    )
    r4.metric(
        "每股风险",
        f"{h.risk_per_share:.2f}" if h.risk_per_share is not None else "—",
    )
    if getattr(h, "slip_note", None):
        st.caption(h.slip_note)
    st.caption(capital_note)


def render_sop(
    symbol: str,
    period: str,
    interval: str,
    period_label: str,
    interval_label: str,
) -> None:
    st.markdown("## 短线波段计划")
    st.caption(f"`{symbol}` · {period_label}/{interval_label}")
    st.warning(
        "**已买入？** 请到左侧菜单选 **「我已买入」**（在「投资SOP」正下方）→ 填你的成交价。"
        " 本页是「还没买」的入场计划。"
    )
    # 页顶直接再放一个迷你入口，避免找不到标签
    with st.expander("⚡ 快捷：我已买入（填买入价）", expanded=True):
        st.caption("填完点按钮，或去侧栏「我已买入」看完整建议。")
        mx1, mx2, mx3 = st.columns([1.2, 1, 1])
        with mx1:
            quick_buy = st.number_input(
                "买入价",
                min_value=0.01,
                value=float(st.session_state.get(f"pos_buy_{symbol}", 100.0)),
                step=0.01,
                format="%.4f",
                key=f"quick_buy_top_{symbol}",
            )
        with mx2:
            st.write("")
            st.write("")
            if st.button("去「我已买入」页", type="primary", key=f"goto_hold_{symbol}"):
                sym_u = str(symbol).strip().upper()
                st.session_state[f"hold_buy_{sym_u}"] = quick_buy
                st.session_state[f"pos_buy_{symbol}"] = quick_buy
                # 下一轮 app 开头切页（勿在此改 nav_page）
                st.session_state._goto_hold = True
                st.rerun()
        with mx3:
            st.write("")
            st.write("")
            st.caption("侧栏也可直接点「我已买入」")

    HKD_PER_USD = 7.8
    c1, c2, c3 = st.columns(3)
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
    with c3:
        horizon_ui = st.selectbox(
            "主周期（决定做不做）",
            ["0–2周", "2–4周"],
            index=0 if st.session_state.get("sop_primary_horizon", "h1") == "h1" else 1,
            key="sop_horizon_select",
            help="只按此周期给出最终结论、出场纪律与净R:R",
        )
    primary_horizon = "h1" if horizon_ui == "0–2周" else "h2"
    st.session_state["sop_capital_hkd"] = capital_hkd
    st.session_state["sop_risk_pct"] = risk_pct
    st.session_state["sop_primary_horizon"] = primary_horizon

    capital_usd = float(capital_hkd) / HKD_PER_USD
    st.caption(
        f"约合 USD {capital_usd:,.0f} · 1R ≈ HKD {capital_hkd * risk_pct / 100:,.0f} · "
        f"主周期 **{horizon_ui}** · 谨慎试仓自动 0.5R"
    )

    with st.spinner("分析走势与短线计划…"):
        sop = build_trade_sop(
            symbol,
            period=period,
            interval=interval,
            capital=float(capital_usd),
            risk_pct=float(risk_pct),
            primary_horizon=primary_horizon,
        )

    h1 = getattr(sop, "swing_h1", None)
    h2 = getattr(sop, "swing_h2", None)
    primary = getattr(sop, "primary_plan", None) or (h1 if primary_horizon == "h1" else h2)
    exit_pl = getattr(sop, "exit_plan", None)
    slip = getattr(sop, "slip_rr", None)

    # ---- Header ----
    st.markdown("---")
    st.markdown(f"### {sop.name} · 现价 **{sop.last_price if sop.last_price is not None else '—'}**")
    if getattr(sop, "trend_note", None):
        st.markdown(sop.trend_note)
    if primary:
        _verdict_box(primary.verdict)(
            f"**主周期 {primary.label} → {primary.verdict}**（映射 {sop.enter_ok}）"
        )

    # 两个大标签：计划 vs 已买入（最显眼）
    tab_plan, tab_hold = st.tabs(["📋 未买入·计划", "💰 我已买入（填买入价）"])

    # ========== TAB: 已买入 ==========
    with tab_hold:
        st.markdown("## 💰 我已买入")
        st.markdown("在这里填 **成交买入价** → 建议 **持有 / 止盈 / 止蚀**")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            default_buy = float(
                st.session_state.get(
                    f"pos_buy_{sop.symbol}",
                    primary.entry_plan
                    if primary and primary.entry_plan
                    else (sop.last_price or 100.0),
                )
            )
            buy_px = st.number_input(
                "你的买入价 *",
                min_value=0.01,
                value=float(default_buy),
                step=0.01,
                format="%.4f",
                key=f"pos_buy_input_{sop.symbol}",
            )
        with b2:
            buy_shares = st.number_input(
                "股数（可选）",
                min_value=0,
                value=int(
                    st.session_state.get(f"pos_sh_{sop.symbol}", sop.position_shares or 0)
                ),
                step=1,
                key=f"pos_shares_input_{sop.symbol}",
            )
        with b3:
            buy_date = st.date_input(
                "买入日期（可选）",
                value=None,
                key=f"pos_date_{sop.symbol}",
                help="用于时间止损判断",
            )
        with b4:
            st.write("")
            st.write("")
            run_coach = st.button(
                "生成持仓建议",
                type="primary",
                use_container_width=True,
                key=f"btn_coach_{sop.symbol}",
            )

        # 默认就显示建议（有买入价 + 现价即算，不必强依赖按钮状态）
        if buy_px and sop.last_price:
            if run_coach:
                st.session_state[f"coach_on_{sop.symbol}"] = True
                st.session_state[f"pos_buy_{sop.symbol}"] = buy_px
                st.session_state[f"pos_sh_{sop.symbol}"] = buy_shares

            max_days = None
            if exit_pl is not None:
                max_days = exit_pl.max_hold_days
            elif primary is not None:
                max_days = primary.bars
            advice = advise_open_position(
                buy_price=float(buy_px),
                last_price=float(sop.last_price),
                plan_stop=primary.stop_loss if primary else sop.stop_loss,
                plan_t1=primary.target if primary else sop.target_t1,
                plan_t2=(
                    h2.target
                    if primary and primary.key == "h1" and h2
                    else sop.target_t2
                ),
                plan_entry=primary.entry_plan if primary else sop.entry_plan,
                max_hold_days=max_days,
                buy_date=buy_date.isoformat() if buy_date else None,
                shares=int(buy_shares) if buy_shares else None,
                bias_label=sop.bias,
                bias_score=float(sop.bias_score or 0),
            )
            if advice.color == "red":
                st.error(f"### {advice.action}")
            elif advice.color == "amber":
                st.warning(f"### {advice.action}")
            elif advice.color == "green":
                st.success(f"### {advice.action}")
            else:
                st.info(f"### {advice.action}")
            st.markdown(f"**{advice.headline}**")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(
                "浮盈%",
                f"{advice.pnl_pct:+.2f}%" if advice.pnl_pct is not None else "—",
            )
            m2.metric(
                "浮盈R",
                f"{advice.pnl_r:+.2f}R" if advice.pnl_r is not None else "—",
            )
            m3.metric(
                "建议止蚀",
                f"{advice.suggested_stop:.2f}"
                if advice.suggested_stop is not None
                else "—",
            )
            m4.metric(
                "距T1",
                f"{advice.dist_to_t1_pct:+.1f}%"
                if advice.dist_to_t1_pct is not None
                else "—",
            )
            m5.metric(
                "距止蚀",
                f"{advice.dist_to_stop_pct:+.1f}%"
                if advice.dist_to_stop_pct is not None
                else "—",
            )
            for b in advice.bullets:
                st.markdown(f"- {b}")

            if st.button("写入交易日志（用此买入价）", key=f"coach_journal_{sop.symbol}"):
                add_trade(
                    symbol=sop.symbol,
                    name=sop.name,
                    horizon=primary.label if primary else "0–2周",
                    entry=float(buy_px),
                    stop=float(
                        advice.suggested_stop
                        or (primary.stop_loss if primary else 0)
                        or buy_px * 0.97
                    ),
                    target=primary.target if primary else sop.target_t1,
                    shares=int(buy_shares) if buy_shares else 0,
                    model_wr=primary.win_rate_pct if primary else None,
                    model_rr=(primary.rr_net or primary.rr) if primary else None,
                    model_verdict=advice.action,
                    notes=f"持仓教练:{advice.action}; 现价{sop.last_price}",
                    opened=buy_date.isoformat() if buy_date else None,
                )
                st.success("已写入交易日志")
                st.rerun()
            st.caption("非投资建议 · 以券商成交与风控为准")
        else:
            st.warning("请填写「你的买入价」（上方数字框）")

        # 主计划对照（方便对照止蚀目标）
        if primary:
            st.markdown("#### 主周期计划对照")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("计划止蚀", f"{primary.stop_loss:.2f}" if primary.stop_loss else "—")
            p2.metric("T1", f"{primary.target:.2f}" if primary.target else "—")
            p3.metric("建议挂单", f"{primary.entry_plan:.2f}" if primary.entry_plan else "—")
            p4.metric("现价", f"{sop.last_price:.2f}" if sop.last_price else "—")

    # ========== TAB: 未买入计划 ==========
    with tab_plan:
        st.markdown(sop.summary)

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                _render_horizon_card(
                    h1,
                    is_primary=(primary_horizon == "h1"),
                    capital_note=f"建议股数约 **{sop.position_shares}** · {sop.position_note}",
                )
        with right:
            with st.container(border=True):
                _render_horizon_card(
                    h2,
                    is_primary=(primary_horizon == "h2"),
                    capital_note="非主周期仅对照；操作以主周期结论为准。",
                )

        st.markdown("### 出场纪律（硬规则）")
        if exit_pl:
            st.info(exit_pl.summary)
            for b in exit_pl.bullets:
                st.markdown(f"- {b}")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric(
                "初始止蚀",
                f"{exit_pl.stop_initial:.2f}" if exit_pl.stop_initial else "—",
            )
            e2.metric("T1 减半", f"{exit_pl.t1_price:.2f}" if exit_pl.t1_price else "—")
            e3.metric(
                "T1后止蚀",
                f"{exit_pl.stop_after_t1:.2f}" if exit_pl.stop_after_t1 else "—",
            )
            e4.metric("最多持有", f"{exit_pl.max_hold_days} 交易日")
        else:
            st.caption("无出场计划")

        if slip is not None:
            st.markdown("### 滑点后可执行赔率")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric(
                "纸面 R:R",
                f"{slip.rr_paper:.2f}" if slip.rr_paper is not None else "—",
            )
            s2.metric("净 R:R", f"{slip.rr_net:.2f}" if slip.rr_net is not None else "—")
            s3.metric(
                "净 E[R]",
                f"{slip.exp_net:+.2f}" if slip.exp_net is not None else "—",
            )
            s4.metric("单边滑点", f"{slip.slip_pct * 100:.2f}%")
            st.caption(slip.note)

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

    # ---- Journal ----
    st.markdown("### 交易日志（对照真胜率）")
    stats = journal_stats()
    j1, j2, j3, j4 = st.columns(4)
    j1.metric("已平仓", stats.get("closed") or 0)
    j2.metric(
        "实盘胜率",
        f"{stats['win_rate']:.0f}%" if stats.get("win_rate") is not None else "—",
    )
    j3.metric(
        "平均 R",
        f"{stats['avg_r']:+.2f}" if stats.get("avg_r") is not None else "—",
    )
    j4.metric("持仓中", stats.get("open") or 0)
    if stats.get("total_pnl") is not None:
        st.caption(f"已平仓合计 PnL ≈ ${stats['total_pnl']:,.2f} USD（按日志股数）")

    with st.expander("记一笔进场（用主周期计划）", expanded=False):
        if primary and primary.entry_plan and primary.stop_loss:
            if st.button("📥 按主计划写入日志", type="primary", key="btn_journal_add"):
                row = add_trade(
                    symbol=sop.symbol,
                    name=sop.name,
                    horizon=primary.label,
                    entry=primary.entry_plan,
                    stop=primary.stop_loss,
                    target=primary.target,
                    shares=int(sop.position_shares or 0),
                    model_wr=primary.win_rate_pct,
                    model_rr=primary.rr_net or primary.rr,
                    model_verdict=primary.verdict,
                    notes=f"模型:{sop.enter_ok}; 净R:R={primary.rr_net}",
                )
                st.success(f"已记录 #{row['id']} {row['symbol']} @ {row['entry']}")
                st.rerun()
        else:
            st.caption("主计划缺入场/止蚀，无法一键记录")

    trades = load_trades()
    open_trades = [t for t in trades if t.get("status") == "open"]
    if open_trades:
        st.markdown("**持仓中**")
        for t in open_trades[:15]:
            with st.container(border=True):
                st.markdown(
                    f"**#{t.get('id')} {t.get('symbol')}** · {t.get('horizon')} · "
                    f"入 {t.get('entry')} 止 {t.get('stop')} 目标 {t.get('target')} · "
                    f"{t.get('shares')} 股 · 模型 {t.get('model_verdict')} "
                    f"WR {t.get('model_wr')}%"
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    xp = st.number_input(
                        "平仓价",
                        min_value=0.01,
                        value=float(t.get("entry") or sop.last_price or 1.0),
                        key=f"exit_px_{t['id']}",
                    )
                with c2:
                    reason = st.selectbox(
                        "原因",
                        ["t1", "t2", "stop", "time", "manual"],
                        key=f"exit_rs_{t['id']}",
                    )
                with c3:
                    st.write("")
                    st.write("")
                    if st.button("平仓", key=f"btn_close_{t['id']}"):
                        close_trade(t["id"], exit_price=float(xp), exit_reason=reason)
                        st.success("已平仓")
                        st.rerun()

    closed = [t for t in trades if t.get("status") == "closed"][:20]
    if closed:
        with st.expander("最近平仓记录", expanded=False):
            st.dataframe(pd.DataFrame(closed), use_container_width=True, hide_index=True)

    # ---- Price strip ----
    st.markdown("### 价格一览")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("现价", f"{sop.last_price:.2f}" if sop.last_price is not None else "—")
    p2.metric("入場低", f"{sop.entry_low:.2f}" if sop.entry_low is not None else "—")
    p3.metric("入場高", f"{sop.entry_high:.2f}" if sop.entry_high is not None else "—")
    p4.metric("止蝕", f"{sop.stop_loss:.2f}" if sop.stop_loss is not None else "—")
    p5.metric("目标0–2周", f"{sop.target_t1:.2f}" if sop.target_t1 is not None else "—")
    p6.metric("目标2–4周", f"{sop.target_t2:.2f}" if sop.target_t2 is not None else "—")

    with st.expander("多周期 + 辅助指标", expanded=False):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("周线", getattr(sop, "weekly_label", "—") or "—")
        r2.metric("1H", getattr(sop, "h1_label", "—") or "—")
        r3.metric(
            "ADX",
            getattr(sop, "adx_label", "—") or "—",
            f"{sop.adx_value:.0f}" if getattr(sop, "adx_value", None) is not None else None,
        )
        r4.metric("跟势", getattr(sop, "trend_align_label", "—") or "—")
        for txt in (
            getattr(sop, "weekly_summary", ""),
            getattr(sop, "h1_summary", ""),
            getattr(sop, "adx_summary", ""),
            getattr(sop, "fib_summary", ""),
            getattr(sop, "trend_align_summary", ""),
            getattr(sop, "false_break_summary", ""),
            getattr(sop, "volume_confirm_summary", ""),
        ):
            if txt:
                st.caption(txt)

    with st.expander("交易时段 · 盘前盘后", expanded=False):
        try:
            info = cached_info(symbol, cache_bucket(2))
        except Exception:
            info = {}
        render_session_quote_card(symbol, info)

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

    with st.expander("怎么读（短线）", expanded=False):
        st.markdown(
            """
1. **选主周期**（0–2周 或 2–4周）→ 只听这个结论  
2. **净 R:R / 净 E[R]**：扣滑点后仍要划算才做  
3. **出场纪律**：T1 减半 → 止蚀保本 → 到期时间止损 → 破止蚀全出  
4. **交易日志**：进场一键记录，平仓填价 → 看实盘胜率 vs 模型胜率  

非投资建议。
            """
        )
        for n in sop.notes:
            st.caption(f"· {n}")
