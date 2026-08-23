"""
投资SOP — 一屏决策卡（精简默认）

默认只显示：结论、入場/止蚀/目标、胜率、净R:R、出场 5 条、去「我已买入」。
其余全部折叠。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from stock_service import cache_bucket, cached_info
from trade_sop import build_trade_sop, format_win_rate
from views.common import render_session_quote_card
from views.journal_panel import render_journal_panel


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
    try:
        from trade_sop import SOP_BUILD as _b
    except Exception:
        _b = "—"
    st.caption(
        f"`{symbol}` · {period_label}/{interval_label} · "
        f"build `{_b}` · **v1 极简定版** · 已买入 → 侧栏 **我已买入**"
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
        if st.button("→ 我已买入", width="stretch", key="sop_to_hold"):
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

    # ========== 一屏决策（默认极简，减少噪音）==========
    st.markdown("---")
    simple = st.toggle(
        "极简模式（推荐）",
        value=bool(st.session_state.get("sop_simple_mode", True)),
        key="sop_simple_mode",
        help="开启：只看三灯+主因+挂单/止蚀/目标。关闭：显示完整明细。",
    )

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

    # 三灯（位置 · 胜率 · 划算）— 日常唯一必看
    def _light_emoji(x: str) -> str:
        return {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(x or "", "⚪")

    pl = getattr(sop, "position_light", None)
    wl = getattr(sop, "wr_light", None)
    rl = getattr(sop, "rr_light", None)
    if pl and wl and rl:
        st.markdown("#### 只看这三灯")
        t1, t2, t3 = st.columns(3)
        t1.metric(
            f"{_light_emoji(pl)} 位置",
            {"green": "可挂/区内", "yellow": "略高", "red": "追高/远离"}.get(pl, pl),
        )
        t2.metric(
            f"{_light_emoji(wl)} 胜率",
            _wr_text(primary if primary else sop),
        )
        t3.metric(
            f"{_light_emoji(rl)} 划算",
            _fmt(getattr(primary, "rr_net", None) if primary else None),
        )

    one = getattr(sop, "one_liner_reason", "") or ""
    if one:
        st.info(f"**主因：** {one}")
    if getattr(sop, "earnings_soon", False) or getattr(sop, "earnings_note", ""):
        ed = getattr(sop, "earnings_days_left", None)
        en = getattr(sop, "earnings_note", "") or "临近财报窗口"
        if ed is not None and 0 <= int(ed) <= 3:
            st.error(f"**财报风险：** {en}")
        else:
            st.warning(f"**财报：** {en}")

    # 执行价位 + 阻力/支撑（极简必看）
    st.markdown("#### 执行价位")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("掛單 E", _fmt(primary.entry_plan if primary else sop.entry_plan))
    k2.metric("止蝕 S", _fmt(primary.stop_loss if primary else sop.stop_loss))
    k3.metric("目標 T", _fmt(primary.target if primary else sop.target_t1))
    win_h = getattr(sop, "pnl_if_win_hkd", None)
    loss_h = getattr(sop, "pnl_if_loss_hkd", None)
    notional = getattr(sop, "notional_hkd", 5000) or 5000
    k4.metric(
        f"约盈/亏({notional:.0f}HKD)",
        (
            f"+{win_h:.0f}/−{loss_h:.0f}"
            if win_h is not None and loss_h is not None
            else "—"
        ),
    )
    r1, r2 = st.columns(2)
    nr = getattr(sop, "nearest_resistance", None)
    ns = getattr(sop, "nearest_support", None)
    r1.metric(
        "最近阻力",
        _fmt(nr),
        (
            f"上方 {sop.resistance_pct:+.1f}%"
            if getattr(sop, "resistance_pct", None) is not None
            else None
        ),
    )
    r2.metric(
        "最近支撑",
        _fmt(ns),
        (
            f"下方 {sop.support_pct:+.1f}%"
            if getattr(sop, "support_pct", None) is not None
            else None
        ),
    )
    if getattr(sop, "resistance_levels_txt", ""):
        st.caption(f"上方阻力：{sop.resistance_levels_txt}")
    if getattr(sop, "support_levels_txt", ""):
        st.caption(f"下方支撑：{sop.support_levels_txt}")
    st.caption(
        f"入場区 {_fmt(primary.entry_low if primary else sop.entry_low)}–"
        f"{_fmt(primary.entry_high if primary else sop.entry_high)} · "
        "现价仅判断追不追；按 **掛單 E** 等，勿市价追。"
    )

    # 出场：极简只 3 条
    st.markdown("#### 出场（记住这三条）")
    if exit_pl:
        st.markdown(
            f"1. 到 T1 减半，止蚀改保本  \n"
            f"2. 最多持有约 **{exit_pl.max_hold_days}** 个交易日  \n"
            f"3. 收盘破止蚀 **{_fmt(exit_pl.stop_initial)}** → 全出"
        )
    else:
        st.caption("无出场计划")

    # 交易日志：始终显示（永久样本），极简也要能写
    render_journal_panel(
        key_prefix="sop_jr",
        default_symbol=sop.symbol,
        default_name=sop.name or sop.symbol,
        default_horizon=primary.label if primary else "0–2周",
        default_entry=(
            float(primary.entry_plan)
            if primary and primary.entry_plan
            else float(sop.entry_plan or 0) or None
        ),
        default_stop=(
            float(primary.stop_loss)
            if primary and primary.stop_loss
            else float(sop.stop_loss or 0) or None
        ),
        default_target=(
            float(primary.target)
            if primary and primary.target
            else float(sop.target_t1 or 0) or None
        ),
        default_shares=int(sop.position_shares or 0),
        default_wr=primary.win_rate_pct if primary else sop.win_rate_pct,
        default_rr=(
            (primary.rr_net or primary.rr) if primary else sop.rr_t1
        ),
        default_verdict=(primary.verdict if primary else sop.enter_ok) or "",
        default_mode=getattr(sop, "mode", "") or "",
        default_mode_label=getattr(sop, "mode_label", "") or "",
        default_exit_px=float(sop.last_price) if sop.last_price else None,
        expanded=True,
    )

    if simple:
        st.caption(
            "极简模式：上方日志可写样本；辅助信号/清单在下方折叠。"
        )
    else:
        # 完整明细
        card = getattr(sop, "plain_card", None) or getattr(sop, "decision_brief", "") or ""
        if card:
            with st.expander("白话全文", expanded=True):
                st.markdown(card)
        if pl and wl and rl:
            st.caption(getattr(sop, "position_light_note", "") or "")
            st.caption(getattr(sop, "wr_light_note", "") or "")
            st.caption(getattr(sop, "rr_light_note", "") or "")
        net_rr = None
        if primary and getattr(primary, "rr_net", None) is not None:
            net_rr = primary.rr_net
        elif slip and slip.rr_net is not None:
            net_rr = slip.rr_net
        paper_rr = primary.rr if primary else sop.rr_t1
        x1, x2 = st.columns(2)
        x1.metric("净R:R", _fmt(net_rr), f"纸面 {_fmt(paper_rr)}" if paper_rr else None)
        x2.metric("胜率", _wr_text(primary if primary else sop))
        if getattr(sop, "trend_note", None):
            st.caption(sop.trend_note)
        if exit_pl:
            with st.expander("出场纪律全文", expanded=False):
                for b in exit_pl.bullets[:8]:
                    st.markdown(f"- {b}")
        if sop.actions_now:
            st.markdown("#### 现在做什么")
            for i, line in enumerate(sop.actions_now[:4], 1):
                st.markdown(f"{i}. {line}")
        if sop.invalidation:
            st.error(f"**失效：** {sop.invalidation}")
        st.caption(
            f"**{mode_lab}** · 建议股数 **{sop.position_shares}** · "
            f"{sop.position_note}"
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
            if getattr(primary, "win_rate_pct", None) is None and (
                not getattr(primary, "win_rate_display", "")
                or "样本不足" in str(getattr(primary, "win_rate_display", ""))
            ):
                s3.metric("净 E[R]", "—")
                st.caption("样本不足时 E[R] 不估算（避免假负期望）。")
            else:
                s3.metric(
                    "净 E[R]",
                    f"{slip.exp_net:+.2f}" if slip.exp_net is not None else "—",
                )
            st.caption(slip.note)
        if primary and primary.expectancy_r is not None:
            st.caption(
                f"纸面 E[R]={primary.expectancy_r:+.2f} "
                f"（门檻用纸面；滑点只在净R:R扣一次）"
            )
        st.caption(
            "E[R]≈ 胜率×赔率 − (1−胜率)×1。"
            "胜率低 + 赔率不够高 → E[R] 为负是数学结果，不是显示错误。"
        )

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
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("无清单")

    with st.expander("盘前盘后 / 时段", expanded=False):
        try:
            info = cached_info(symbol, cache_bucket(2))
        except Exception:
            info = {}
        render_session_quote_card(symbol, info)

    with st.expander("模型说明", expanded=False):
        st.markdown(
            """
- **三灯裁决（默认）**：**位置** + **胜率** + **划算(净R:R)**  
  - 三绿 → 可以入場；有黄无红 → 試倉；任一红 → 先别做  
  - 强烈看空 / 放量下跌 → 不做多；假突破/周线空/逆势 → 最多試倉  
- **模式 A**：胜率满仓≥52% / 试仓≥48%；净R:R 满仓≥1.10  
- **模式 B**：胜率满仓≥50% / 试仓≥45%；默认 0.5R  
- **每笔 5000 HKD**：白话卡显示赚到目标/止损大约多少钱  
- **E/S/T**：计划限价=区中下（不追现价算R:R）；目标至少1:1；止损约0.6–1.5×ATR  
- **路径胜率**：历史 K 线先到目标再触止损；样本数只作说明  
- **主周期**决定做不做、出场、净R:R  
- **胜率**＝历史路径先到目标再触止蚀（非保证）  
- **净R:R**＝扣约 0.15% 单边滑点  
- **出场**：T1 减半 → 止蚀保本 → 时间止损 → 破止蚀全出  
- 已买入请用侧栏 **我已买入**
            """
        )
        for n in (sop.notes or [])[:8]:
            st.caption(f"· {n}")
