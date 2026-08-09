"""
投资SOP — 一屏决策卡（精简默认）

默认只显示：结论、入場/止蚀/目标、胜率、净R:R、出场 5 条、去「我已买入」。
其余全部折叠。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from stock_service import cache_bucket, cached_info
from trade_journal import add_trade, close_trade, journal_stats, load_trades
from trade_sop import build_trade_sop, format_win_rate
from views.common import render_session_quote_card


def _verdict_box(verdict: str):
    if verdict in ("可以入場", "适合入场"):
        return st.success
    if verdict in ("可以試倉", "谨慎试仓"):
        return st.warning
    if verdict in ("不做多", "回避"):
        return st.error
    return st.info


def _fmt(x, nd=2):
    if x is None:
        return "—"
    try:
        v = float(x)
        if v != v:  # NaN
            return "—"
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _wr_text(sop_or_plan) -> str:
    """Never show nan% — use preformatted display or format_win_rate."""
    if sop_or_plan is None:
        return "样本不足"
    disp = getattr(sop_or_plan, "win_rate_display", None)
    if disp:
        return str(disp)
    return format_win_rate(
        getattr(sop_or_plan, "win_rate_pct", None),
        getattr(sop_or_plan, "win_rate_samples", None),
    )


def render_sop(
    symbol: str,
    period: str,
    interval: str,
    period_label: str,
    interval_label: str,
) -> None:
    st.markdown("## 短线计划（一屏）")
    st.caption(
        f"`{symbol}` · {period_label}/{interval_label} · "
        "已买入请侧栏点 **我已买入**"
    )

    # ---- compact controls ----
    HKD_PER_USD = 7.8
    c1, c2, c3, c4, c5 = st.columns([1.0, 0.85, 0.95, 1.15, 0.9])
    with c1:
        capital_hkd = st.number_input(
            "本金 HKD",
            min_value=1000.0,
            value=float(st.session_state.get("sop_capital_hkd", 50_000.0)),
            step=1000.0,
            key="sop_capital_input",
        )
    with c2:
        risk_pct = st.number_input(
            "1R %",
            min_value=0.25,
            max_value=5.0,
            value=float(st.session_state.get("sop_risk_pct", 1.0)),
            step=0.25,
            key="sop_risk_input",
        )
    with c3:
        horizon_ui = st.selectbox(
            "主周期",
            ["0–2周", "2–4周"],
            index=0 if st.session_state.get("sop_primary_horizon", "h1") == "h1" else 1,
            key="sop_horizon_select",
        )
    with c4:
        mode_ui = st.selectbox(
            "模式",
            ["A 防守版", "B 进攻版"],
            index=0
            if st.session_state.get("sop_mode", "defensive") != "aggressive"
            else 1,
            key="sop_mode_select",
            help="A 高胜率稳健 · B 有根据的搏（默认 0.5R）· 大盘弱/VIX 高会强制 A",
        )
    with c5:
        st.write("")
        st.write("")
        if st.button("→ 我已买入", use_container_width=True, key="sop_to_hold"):
            st.session_state._goto_hold = True
            st.rerun()

    primary_horizon = "h1" if horizon_ui == "0–2周" else "h2"
    mode_key = "aggressive" if mode_ui.startswith("B") else "defensive"
    st.session_state["sop_capital_hkd"] = capital_hkd
    st.session_state["sop_risk_pct"] = risk_pct
    st.session_state["sop_primary_horizon"] = primary_horizon
    st.session_state["sop_mode"] = mode_key
    capital_usd = float(capital_hkd) / HKD_PER_USD

    with st.spinner("生成计划…"):
        try:
            sop = build_trade_sop(
                symbol,
                period=period,
                interval=interval,
                capital=float(capital_usd),
                risk_pct=float(risk_pct),
                primary_horizon=primary_horizon,
                mode=mode_key,
            )
        except Exception as exc:
            st.error(f"分析失败：{type(exc).__name__}: {exc}")
            st.stop()

    h1 = getattr(sop, "swing_h1", None)
    h2 = getattr(sop, "swing_h2", None)
    primary = getattr(sop, "primary_plan", None) or (h1 if primary_horizon == "h1" else h2)
    exit_pl = getattr(sop, "exit_plan", None)
    slip = getattr(sop, "slip_rr", None)
    other = h2 if primary_horizon == "h1" else h1

    # ========== 一屏决策卡（默认全部可见）==========
    st.markdown("---")
    name = sop.name or symbol
    last = sop.last_price
    st.markdown(f"### {name} · `{sop.symbol}` · 现价 **{_fmt(last)}**")

    mode_lab = getattr(sop, "mode_label", "") or "A 防守版"
    if getattr(sop, "mode_forced", False):
        st.warning(f"**模式已强制为 {mode_lab}**" + (
            f"：{sop.mode_note}" if getattr(sop, "mode_note", "") else ""
        ))
    if primary:
        v = primary.verdict
        _verdict_box(v)(f"## {v} · {primary.label} · {mode_lab}")
    else:
        st.warning(sop.enter_ok)

    # 决策摘要（自动：结论 + 根据 + 主风险）
    brief = getattr(sop, "decision_brief", "") or ""
    if brief:
        st.markdown("#### 决策摘要")
        st.info(brief)

    # 核心 6 格
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "入場区",
        _fmt(primary.entry_low if primary else sop.entry_low),
        f"至 {_fmt(primary.entry_high if primary else sop.entry_high)}",
    )
    k2.metric("掛單", _fmt(primary.entry_plan if primary else sop.entry_plan))
    k3.metric("止蝕", _fmt(primary.stop_loss if primary else sop.stop_loss))
    k4.metric("目標", _fmt(primary.target if primary else sop.target_t1))
    k5.metric("勝率", _wr_text(primary if primary else sop))
    net_rr = None
    if primary and getattr(primary, "rr_net", None) is not None:
        net_rr = primary.rr_net
    elif slip and slip.rr_net is not None:
        net_rr = slip.rr_net
    paper_rr = primary.rr if primary else sop.rr_t1
    k6.metric(
        "净R:R",
        _fmt(net_rr),
        f"纸面 {_fmt(paper_rr)}" if paper_rr is not None else None,
    )

    # 一句话走势
    if getattr(sop, "trend_note", None):
        st.caption(sop.trend_note)

    # 出场 5 条（硬规则，默认展开短）
    st.markdown("#### 出场纪律")
    if exit_pl:
        for b in exit_pl.bullets[:6]:
            st.markdown(f"- {b}")
        e1, e2, e3 = st.columns(3)
        e1.caption(f"T1后止蚀≈ **{_fmt(exit_pl.stop_after_t1)}**（保本）")
        e2.caption(f"最多持有 **{exit_pl.max_hold_days}** 交易日")
        e3.caption(f"初始止蚀 **{_fmt(exit_pl.stop_initial)}**")
    else:
        st.caption("无出场计划")

    # 立刻动作（最多 4 条）
    if sop.actions_now:
        st.markdown("#### 现在做什么")
        for i, line in enumerate(sop.actions_now[:4], 1):
            st.markdown(f"{i}. {line}")
    if sop.invalidation:
        st.error(f"**失效：** {sop.invalidation}")

    st.caption(
        f"**{mode_lab}** · 建议股数 **{sop.position_shares}** · "
        f"允许 **{getattr(sop, 'risk_units', 0):g}R** · {sop.position_note} · "
        f"1R≈HKD {capital_hkd * risk_pct / 100:,.0f}"
    )

    # ========== 以下全部折叠 ==========
    with st.expander("另一周期对照", expanded=False):
        if other:
            wr_s = _wr_text(other)
            st.markdown(
                f"**{other.label}** → {other.verdict} · "
                f"目标 {_fmt(other.target)} · 胜率 {wr_s} · "
                f"R:R {_fmt(other.rr)} · 净R:R {_fmt(getattr(other, 'rr_net', None))}"
            )
        else:
            st.caption("无对照")

    with st.expander("滑点 / 期望详情", expanded=False):
        if slip is not None:
            s1, s2, s3 = st.columns(3)
            s1.metric("纸面 R:R", _fmt(slip.rr_paper))
            s2.metric("净 R:R", _fmt(slip.rr_net))
            s3.metric("净 E[R]", f"{slip.exp_net:+.2f}" if slip.exp_net is not None else "—")
            st.caption(slip.note)
        if primary and primary.expectancy_r is not None:
            st.caption(f"纸面 E[R]={primary.expectancy_r:+.2f}")

    with st.expander("辅助信号（周线/1H/ADX/假突破/跟势/量能）", expanded=False):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("周线", getattr(sop, "weekly_label", "—") or "—")
        r2.metric("1H", getattr(sop, "h1_label", "—") or "—")
        r3.metric(
            "ADX",
            getattr(sop, "adx_label", "—") or "—",
            f"{sop.adx_value:.0f}" if getattr(sop, "adx_value", None) is not None else None,
        )
        r4.metric("跟势", getattr(sop, "trend_align_label", "—") or "—")
        e1, e2, e3 = st.columns(3)
        e1.metric("假突破", getattr(sop, "false_break_label", "—") or "—")
        e2.metric("量能", getattr(sop, "volume_confirm_label", "—") or "—")
        e3.metric("板块RS", getattr(sop, "sector_rs_label", "—") or "—")
        for txt in (
            getattr(sop, "weekly_summary", ""),
            getattr(sop, "h1_summary", ""),
            getattr(sop, "adx_summary", ""),
            getattr(sop, "fib_summary", ""),
            getattr(sop, "false_break_summary", ""),
            getattr(sop, "trend_align_summary", ""),
            getattr(sop, "volume_confirm_summary", ""),
            getattr(sop, "sector_rs_summary", ""),
            getattr(sop, "iv_summary", ""),
        ):
            if txt:
                st.caption(txt)

    with st.expander("检查清单", expanded=False):
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
        else:
            st.caption("无清单")

    with st.expander("交易日志（实盘对照）", expanded=False):
        stats = journal_stats()
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("已平仓", stats.get("closed") or 0)
        j2.metric(
            "实盘胜率",
            f"{stats['win_rate']:.0f}%" if stats.get("win_rate") is not None else "—",
        )
        j3.metric(
            "平均R",
            f"{stats['avg_r']:+.2f}" if stats.get("avg_r") is not None else "—",
        )
        j4.metric("持仓中", stats.get("open") or 0)
        if primary and primary.entry_plan and primary.stop_loss:
            if st.button("按主计划写入日志", key="sop_journal_add"):
                add_trade(
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
                    mode=getattr(sop, "mode", "") or "",
                    mode_label=getattr(sop, "mode_label", "") or "",
                    notes=f"模式={getattr(sop, 'mode_label', '')} · 胜率={_wr_text(primary)}",
                )
                st.success(f"已写入（模式 {getattr(sop, 'mode_label', '')}）")
                st.rerun()
        trades = load_trades()
        open_t = [t for t in trades if t.get("status") == "open"][:8]
        for t in open_t:
            st.caption(
                f"#{t.get('id')} {t.get('symbol')} "
                f"[{t.get('mode_label') or t.get('mode') or '?'}] "
                f"入{t.get('entry')} 止{t.get('stop')} "
                f"目标{t.get('target')}"
            )
            if st.button(f"平仓 #{t.get('id')}", key=f"sop_close_{t.get('id')}"):
                px = float(sop.last_price or t.get("entry") or 0)
                close_trade(t["id"], exit_price=px, exit_reason="manual")
                st.rerun()

    with st.expander("盘前盘后 / 时段", expanded=False):
        try:
            info = cached_info(symbol, cache_bucket(2))
        except Exception:
            info = {}
        render_session_quote_card(symbol, info)

    with st.expander("模型说明", expanded=False):
        st.markdown(
            """
- **模式 A 防守**：胜率≥55%/50%，净R:R≥1.2，E[R]≥0.15，稳定度≥45  
- **模式 B 进攻**：胜率≥52%/48%，净R:R≥1.0，默认 0.5R，≥3 项加分可升 1R  
- **路径胜率**：历史先到目标再触止损；样本&lt;12 显示「样本不足」（禁止当高胜率）  
- **主周期**决定做不做、出场、净R:R  
- **胜率**＝历史路径先到目标再触止蚀（非保证）  
- **净R:R**＝扣约 0.15% 单边滑点  
- **出场**：T1 减半 → 止蚀保本 → 时间止损 → 破止蚀全出  
- 已买入请用侧栏 **我已买入**
            """
        )
        for n in (sop.notes or [])[:8]:
            st.caption(f"· {n}")
