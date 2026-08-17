"""Dedicated page: I already bought — hold / take-profit / stop advice."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import streamlit as st

from position_coach import advise_dual_hold, build_follow_levels
from stock_service import cache_bucket, cached_info, fetch_history, normalize_symbol
from trade_sop import DEFAULT_NOTIONAL_HKD, build_trade_sop
from views.journal_panel import render_journal_panel


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except (TypeError, ValueError):
        return None


def _plan_quality(plan: Any) -> float:
    """Score a swing plan for hold management (higher = better structure)."""
    if plan is None:
        return -1e9
    score = 0.0
    e = _f(getattr(plan, "entry_plan", None))
    s = _f(getattr(plan, "stop_loss", None))
    t = _f(getattr(plan, "target", None))
    if e and s and t and e > s and t > e:
        score += 20.0
    else:
        return -1e9
    rr = _f(getattr(plan, "rr_net", None))
    if rr is None:
        rr = _f(getattr(plan, "rr", None))
    if rr is not None:
        # Prefer tradable RR; cap contribution
        score += min(12.0, max(0.0, rr) * 6.0)
    wr = _f(getattr(plan, "win_rate_pct", None))
    if wr is not None:
        score += min(8.0, wr / 12.5)
    v = getattr(plan, "verdict", "") or ""
    if v == "可以入場":
        score += 4.0
    elif v == "可以試倉":
        score += 2.0
    return score


def pick_best_hold_plan(
    sop: Any,
    *,
    prefer: str = "auto",
) -> tuple[Any, Any, str, str]:
    """
    Returns (primary_plan, secondary_for_t2, horizon_label, why_pick).
    prefer: auto | h1 | h2
    """
    h1 = getattr(sop, "swing_h1", None)
    h2 = getattr(sop, "swing_h2", None)
    if prefer == "h1":
        return h1, h2, getattr(h1, "label", "0–2周") or "0–2周", "你指定 0–2周"
    if prefer == "h2":
        return h2, h1, getattr(h2, "label", "2–4周") or "2–4周", "你指定 2–4周"

    s1, s2 = _plan_quality(h1), _plan_quality(h2)
    if s2 > s1 + 0.5:
        why = (
            f"自动选 2–4周（结构分 {s2:.1f} > 0–2周 {s1:.1f}："
            f"更完整/赔率或胜率更好）"
        )
        return h2, h1, getattr(h2, "label", "2–4周") or "2–4周", why
    why = (
        f"自动选 0–2周（结构分 {s1:.1f}"
        + (f" ≥ 2–4周 {s2:.1f}" if s2 > -1e8 else "")
        + "）"
    )
    return h1, h2, getattr(h1, "label", "0–2周") or "0–2周", why


@dataclass
class HoldLevels:
    """Extracted E/S/T1/T2/SR for hold dual view (entry-day or live)."""

    tag: str  # entry | live
    as_of: str  # YYYY-MM-DD or "即时"
    close: float | None
    entry_plan: float | None
    zone_lo: float | None
    zone_hi: float | None
    stop: float | None
    t1: float | None
    t2: float | None
    resistance: float | None
    support: float | None
    resistance_txt: str = ""
    support_txt: str = ""
    resistance_pct: float | None = None
    support_pct: float | None = None
    verdict: str = "—"
    wr_display: str | None = None
    rr: float | None = None
    rr_net: float | None = None
    bias: str = "—"
    bias_score: float = 0.0
    horizon_label: str = "0–2周"
    pick_why: str = ""
    max_days: int = 10
    structure_bits: list[str] | None = None


def extract_hold_levels(
    sop: Any,
    *,
    prefer: str = "auto",
    tag: str = "live",
) -> HoldLevels:
    """Pull ordered T1/T2 + SR from a TradeSOP (live or as_of)."""
    primary, secondary, horizon_label, pick_why = pick_best_hold_plan(sop, prefer=prefer)
    plan_stop = plan_t1 = plan_t2 = plan_entry = None
    plan_zone_lo = plan_zone_hi = None
    plan_rr = plan_rr_net = plan_wr = None
    plan_verdict = "—"
    max_days = 10
    if primary:
        plan_stop = _f(primary.stop_loss)
        plan_t1 = _f(primary.target)
        plan_entry = _f(primary.entry_plan)
        plan_zone_lo = _f(primary.entry_low)
        plan_zone_hi = _f(primary.entry_high)
        plan_rr = _f(primary.rr)
        plan_rr_net = _f(primary.rr_net)
        plan_wr = getattr(primary, "win_rate_display", None) or (
            f"{primary.win_rate_pct:.0f}%"
            if getattr(primary, "win_rate_pct", None) is not None
            else None
        )
        plan_verdict = primary.verdict or "—"
        max_days = int(getattr(primary, "bars", None) or 10)
    if secondary and getattr(secondary, "target", None) is not None:
        plan_t2 = _f(secondary.target)
    ta, tb = plan_t1, plan_t2
    if ta is not None and tb is not None:
        plan_t1, plan_t2 = min(ta, tb), max(ta, tb)
    elif ta is not None:
        plan_t1, plan_t2 = ta, None
    elif tb is not None:
        plan_t1, plan_t2 = tb, None
    # Prefer SOP top-level T1/T2 if present (already ordered)
    st1, st2 = _f(getattr(sop, "target_t1", None)), _f(getattr(sop, "target_t2", None))
    if st1 is not None and st2 is not None:
        plan_t1, plan_t2 = min(st1, st2), max(st1, st2)
    elif st1 is not None and plan_t1 is None:
        plan_t1 = st1
    elif st2 is not None and plan_t2 is None:
        plan_t2 = st2
    exit_pl = getattr(sop, "exit_plan", None)
    if exit_pl and getattr(exit_pl, "max_hold_days", None):
        max_days = int(exit_pl.max_hold_days)

    structure_bits: list[str] = []
    if getattr(sop, "notes", None):
        for n in sop.notes:
            if any(
                k in str(n)
                for k in ("计划限价", "止损", "目标", "E=", "ATR", "1:1", "阻力")
            ):
                structure_bits.append(str(n))
                if len(structure_bits) >= 4:
                    break

    as_of_raw = getattr(sop, "as_of", None)
    as_of_label = str(as_of_raw) if as_of_raw else ("即时" if tag == "live" else "—")

    return HoldLevels(
        tag=tag,
        as_of=as_of_label,
        close=_f(getattr(sop, "last_price", None)),
        entry_plan=plan_entry or _f(getattr(sop, "entry_plan", None)),
        zone_lo=plan_zone_lo or _f(getattr(sop, "entry_low", None)),
        zone_hi=plan_zone_hi or _f(getattr(sop, "entry_high", None)),
        stop=plan_stop or _f(getattr(sop, "stop_loss", None)),
        t1=plan_t1,
        t2=plan_t2,
        resistance=_f(getattr(sop, "nearest_resistance", None)),
        support=_f(getattr(sop, "nearest_support", None)),
        resistance_txt=str(getattr(sop, "resistance_levels_txt", "") or ""),
        support_txt=str(getattr(sop, "support_levels_txt", "") or ""),
        resistance_pct=_f(getattr(sop, "resistance_pct", None)),
        support_pct=_f(getattr(sop, "support_pct", None)),
        verdict=plan_verdict,
        wr_display=plan_wr,
        rr=plan_rr,
        rr_net=plan_rr_net,
        bias=str(getattr(sop, "bias", None) or "—"),
        bias_score=float(getattr(sop, "bias_score", None) or 0),
        horizon_label=horizon_label,
        pick_why=pick_why,
        max_days=max_days,
        structure_bits=structure_bits,
    )


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _render_levels_card(lv: HoldLevels, *, title: str, subtitle: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(subtitle)
    st.caption(
        f"周期 **{lv.horizon_label}** · 结论 **{lv.verdict}** · "
        f"多空 **{lv.bias}**（{lv.bias_score:+.0f}）· "
        f"最多约 **{lv.max_days}** 交易日"
        + (f" · 当日收 {lv.close:.2f}" if lv.close is not None else "")
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(
        "计划限价 E",
        _fmt(lv.entry_plan),
        (
            f"区 {_fmt(lv.zone_lo)}–{_fmt(lv.zone_hi)}"
            if lv.zone_lo is not None and lv.zone_hi is not None
            else None
        ),
    )
    c2.metric("止蚀 S", _fmt(lv.stop))
    c3.metric("目标 T1", _fmt(lv.t1))
    c4.metric("目标 T2", _fmt(lv.t2))
    c5.metric(
        "最近阻力",
        _fmt(lv.resistance),
        f"+{lv.resistance_pct:.1f}%" if lv.resistance_pct is not None else None,
    )
    c6.metric(
        "最近支撑",
        _fmt(lv.support),
        f"{lv.support_pct:+.1f}%" if lv.support_pct is not None else None,
    )
    bits: list[str] = []
    rr_show = lv.rr_net if lv.rr_net is not None else lv.rr
    if rr_show is not None:
        bits.append(f"净R:R **{rr_show:.2f}**")
    if lv.wr_display:
        bits.append(f"胜率 {lv.wr_display}")
    if lv.resistance_txt:
        bits.append(f"上方阻力：{lv.resistance_txt}")
    if lv.support_txt:
        bits.append(f"下方支撑：{lv.support_txt}")
    if bits:
        st.caption(" · ".join(bits))
    if lv.resistance is not None and lv.t1 is not None and lv.t1 > lv.resistance:
        st.caption(
            f"提示：T1={lv.t1:.2f} 高于最近阻力 {lv.resistance:.2f}，"
            "短线可先在阻力减仓，站稳再看 T1"
        )
    if lv.pick_why:
        st.caption(lv.pick_why)


def _levels_ruler(lv: HoldLevels, *, buy: float | None = None, last: float | None = None) -> str:
    levels: list[tuple[str, float]] = []
    if lv.stop is not None:
        levels.append(("止蚀S", lv.stop))
    if lv.support is not None:
        levels.append(("支撑", lv.support))
    if buy is not None:
        levels.append(("你的买入", buy))
    if lv.entry_plan is not None:
        levels.append(("计划E", lv.entry_plan))
    if last is not None:
        levels.append(("现价", last))
    if lv.close is not None and (last is None or abs(lv.close - (last or 0)) > 1e-6):
        if lv.tag == "entry":
            levels.append(("入场日收", lv.close))
    if lv.resistance is not None:
        levels.append(("阻力", lv.resistance))
    if lv.t1 is not None:
        levels.append(("T1", lv.t1))
    if lv.t2 is not None:
        levels.append(("T2", lv.t2))
    levels_sorted = sorted(levels, key=lambda x: x[1])
    return "  <  ".join(f"**{n}** {v:.2f}" for n, v in levels_sorted)


def render_hold_page(symbol: str, period: str = "1y", interval: str = "1d") -> None:
    """Always-visible buy-price coach (does not depend on long tabs)."""
    st.markdown("# 💰 我已买入")
    st.markdown(
        f"股票 **`{symbol}`** · 填**买入价 + 买入日期** → "
        "系统给出 **唯一一套「你现在跟这些」**（不用自己选入场日或即时）。"
    )

    sym = normalize_symbol(symbol)
    last = None
    name = sym
    try:
        info = cached_info(sym, cache_bucket(5))
        last = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("last_price")
        )
        name = info.get("shortName") or info.get("longName") or sym
        if last is None:
            hist = fetch_history(sym, period="5d", interval="1d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                last = float(hist["Close"].iloc[-1])
    except Exception:
        pass

    if last is not None:
        st.metric("现价（参考）", f"{float(last):.4f}")
    else:
        st.warning("暂时拉不到现价，仍可填买入价；生成建议时会再试一次。")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        buy_px = st.number_input(
            "你的买入价（必填）",
            min_value=0.01,
            value=float(st.session_state.get(f"hold_buy_{sym}", last or 100.0)),
            step=0.01,
            format="%.4f",
            key="hold_page_buy_px",
        )
    with c2:
        shares = st.number_input(
            "股数（可选，用于港币盈亏）",
            min_value=0,
            value=int(st.session_state.get(f"hold_sh_{sym}", 0)),
            step=1,
            key="hold_page_shares",
        )
    with c3:
        default_buy_date = st.session_state.get(f"hold_dt_{sym}", date.today())
        if not isinstance(default_buy_date, date):
            default_buy_date = date.today()
        buy_date = st.date_input(
            "买入日期（用于入场日计划）",
            value=default_buy_date,
            max_value=date.today(),
            key="hold_page_buy_date",
            help="系统会用该日及以前的日线，重算入场当天的 E/S/T1/T2/阻力/支撑",
        )

    c4, c5 = st.columns(2)
    with c4:
        use_plan = st.checkbox(
            "加载入场日 + 即时计划（止蚀/T1/T2/阻力）",
            value=True,
            key="hold_use_plan",
            help="与投资SOP同源：区中下计划限价、止蚀ATR、目标至少1:1；入场日无未来K线",
        )
    with c5:
        horizon_pref = st.selectbox(
            "计划周期",
            ["自动选最优", "强制 0–2周", "强制 2–4周"],
            index=0,
            key="hold_horizon_pref",
            help="自动：比较两套计划的完整性、净R:R、胜率后选更好的一套做管仓",
        )

    entry_lv: HoldLevels | None = None
    live_lv: HoldLevels | None = None
    structure_bits: list[str] = []
    load_err: str | None = None

    if st.button("生成持仓建议", type="primary", use_container_width=True, key="hold_page_go"):
        st.session_state["hold_page_ran"] = True
        st.session_state[f"hold_buy_{sym}"] = buy_px
        st.session_state[f"hold_sh_{sym}"] = shares
        st.session_state[f"hold_dt_{sym}"] = buy_date

    if not st.session_state.get("hold_page_ran") and not st.session_state.get("hold_auto"):
        st.session_state["hold_auto"] = True

    if buy_px and (last or use_plan):
        if use_plan:
            with st.spinner("加载入场日计划 + 即时计划…"):
                try:
                    prefer_map = {
                        "自动选最优": "auto",
                        "强制 0–2周": "h1",
                        "强制 2–4周": "h2",
                    }
                    prefer = prefer_map.get(horizon_pref, "auto")
                    ph = "h2" if prefer == "h2" else "h1"
                    # —— 即时 ——
                    sop_live = build_trade_sop(
                        sym,
                        period=period,
                        interval=interval,
                        capital=50_000 / 7.8,
                        risk_pct=1.0,
                        primary_horizon=ph,
                    )
                    live_lv = extract_hold_levels(sop_live, prefer=prefer, tag="live")
                    if sop_live.last_price is not None:
                        last = float(sop_live.last_price)
                    name = sop_live.name or name
                    structure_bits = list(live_lv.structure_bits or [])

                    # —— 入场日（买入日期 as_of）——
                    if buy_date:
                        sop_entry = build_trade_sop(
                            sym,
                            period=period,
                            interval=interval,
                            capital=50_000 / 7.8,
                            risk_pct=1.0,
                            primary_horizon=ph,
                            as_of=buy_date,
                        )
                        entry_lv = extract_hold_levels(
                            sop_entry, prefer=prefer, tag="entry"
                        )
                except Exception as exc:
                    load_err = str(exc)
                    st.warning(f"完整计划加载失败，改用买入价 vs 现价：{exc}")

        if last is None:
            st.error("没有现价，无法比较。请检查网络或股票代码。")
            return

        buy_f = float(buy_px)
        last_f = float(last)

        # Fallbacks if plans failed
        if live_lv is None:
            live_lv = HoldLevels(
                tag="live",
                as_of="即时",
                close=last_f,
                entry_plan=buy_f,
                zone_lo=None,
                zone_hi=None,
                stop=buy_f * 0.97,
                t1=buy_f * 1.05,
                t2=buy_f * 1.10,
                resistance=None,
                support=None,
                max_days=10,
            )
        if live_lv.stop is None:
            live_lv.stop = buy_f * 0.97
        if live_lv.t1 is None:
            live_lv.t1 = buy_f * 1.05

        t1_use = (
            entry_lv.t1 if entry_lv and entry_lv.t1 is not None else live_lv.t1
        )
        t2_use = (
            entry_lv.t2 if entry_lv and entry_lv.t2 is not None else live_lv.t2
        )
        stop_hard = (
            entry_lv.stop if entry_lv and entry_lv.stop is not None else live_lv.stop
        )
        max_days = (
            entry_lv.max_days if entry_lv is not None else live_lv.max_days
        )
        entry_as_of = (
            buy_date.isoformat()
            if buy_date
            else (entry_lv.as_of if entry_lv else "")
        )

        # 入场日 + 即时 → 综合 suggestion + 唯一跟单价位
        advice, dual_lines = advise_dual_hold(
            buy_price=buy_f,
            last_price=last_f,
            buy_date=buy_date.isoformat() if buy_date else None,
            shares=int(shares) if shares else None,
            max_hold_days=max_days,
            bias_label=live_lv.bias,
            bias_score=live_lv.bias_score,
            entry_stop=entry_lv.stop if entry_lv else None,
            entry_t1=entry_lv.t1 if entry_lv else None,
            entry_t2=entry_lv.t2 if entry_lv else None,
            entry_e=entry_lv.entry_plan if entry_lv else None,
            entry_res=entry_lv.resistance if entry_lv else None,
            entry_sup=entry_lv.support if entry_lv else None,
            entry_close=entry_lv.close if entry_lv else None,
            entry_as_of=str(entry_as_of or ""),
            live_stop=live_lv.stop,
            live_t1=live_lv.t1,
            live_t2=live_lv.t2,
            live_e=live_lv.entry_plan,
            live_res=live_lv.resistance,
            live_sup=live_lv.support,
        )
        follow = build_follow_levels(
            buy_price=buy_f,
            last_price=last_f,
            advice=advice,
            entry_stop=entry_lv.stop if entry_lv else None,
            entry_t1=entry_lv.t1 if entry_lv else None,
            entry_t2=entry_lv.t2 if entry_lv else None,
            entry_res=entry_lv.resistance if entry_lv else None,
            live_t1=live_lv.t1,
            live_t2=live_lv.t2,
            live_res=live_lv.resistance,
        )

        # ========== 唯一决策区（置顶，不逼你选入场日/即时）==========
        st.markdown("---")
        st.markdown("### ⚡ 你现在怎么做（只跟这一套）")
        st.info(follow.rule_one_liner)

        if advice.color == "red":
            st.error(f"## {advice.action}")
        elif advice.color == "amber":
            st.warning(f"## {advice.action}")
        else:
            st.success(f"## {advice.action}")
        st.markdown(f"**{advice.headline}**")

        # 价位尺：现在立刻做 ≠ 剩仓目标（避免「1026=减半」的混淆）
        st.markdown(f"**现在立刻做：** {follow.now_do}")
        st.caption(follow.now_do_detail)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("现价", f"{last_f:.2f}")
        k2.metric(
            "浮盈%",
            f"{advice.pnl_pct:+.2f}%" if advice.pnl_pct is not None else "—",
        )
        k3.metric(
            "现在止蚀（跟这个）",
            _fmt(follow.now_stop),
            (
                f"硬底线 {_fmt(follow.hard_stop)}"
                if follow.hard_stop is not None
                else None
            ),
        )
        # 未到 T1：显示「等待价」；已到 T1：显示「剩仓下一目标」（不再把减半绑在 T2 价上）
        if follow.stage == "before_t1":
            k4.metric(
                "等待价（到了再减）",
                _fmt(follow.wait_price),
                follow.wait_label or None,
            )
            k5.metric(
                "之后目标(T2)",
                _fmt(follow.remain_target),
                follow.remain_label or None,
            )
        elif follow.stage == "past_t1":
            k4.metric(
                "剩仓下一目标",
                _fmt(follow.remain_target),
                follow.remain_label or "减半之后才看",
            )
            k5.metric(
                "更远目标",
                _fmt(follow.far_target) if follow.far_target is not None else "—",
                (
                    follow.far_label
                    if follow.far_target is not None
                    else (
                        f"阻力 {_fmt(follow.resistance)}"
                        if follow.resistance is not None
                        else "无更远·收紧止蚀"
                    )
                ),
            )
        else:
            k4.metric("等待/目标", "—", follow.wait_label or follow.now_do)
            k5.metric(
                "备注",
                _fmt(follow.far_target) if follow.far_target else "—",
                (
                    f"阻力 {_fmt(follow.resistance)}"
                    if follow.resistance is not None
                    else None
                ),
            )

        if follow.resistance is not None and follow.stage == "past_t1":
            st.caption(
                f"参考阻力 ≈ **{_fmt(follow.resistance)}**（可贴着减/观察，"
                f"但**减半动作不依赖阻力**，T1 已到就该减）"
            )

        st.markdown(f"**为什么：** {follow.next_why}")

        st.markdown("#### 3 步执行（步骤已拆开）")
        if follow.stage == "past_t1":
            st.markdown(
                f"1. **现在**减约一半（入场日 T1 已到，**不必等到** "
                f"{_fmt(follow.remain_target)}）\n"
                f"2. 券商止蚀改到 **{_fmt(follow.now_stop)}**"
                f"（硬底线 {_fmt(follow.hard_stop)}，只上移不下移）\n"
                f"3. **剩仓**拿到 **{_fmt(follow.remain_target)}** 再减/清"
                + (
                    f"；更远可看 {_fmt(follow.far_target)}"
                    if follow.far_target is not None
                    else ""
                )
                + "\n4. 禁止：摊平、止蚀下移、把「减半」拖到 T2 才做"
            )
        elif follow.stage == "before_t1":
            st.markdown(
                f"1. 券商止蚀设在 **{_fmt(follow.now_stop)}**"
                f"（硬底线 {_fmt(follow.hard_stop)}）\n"
                f"2. 等到 **{_fmt(follow.wait_price)}** → {follow.wait_label}\n"
                f"3. 再远目标 **{_fmt(follow.remain_target)}**；禁止摊平/下移止蚀"
            )
        else:
            st.markdown(
                f"1. **{follow.now_do}**\n"
                f"2. 止蚀参考 **{_fmt(follow.now_stop)}**\n"
                f"3. 禁止摊平、下移止蚀"
            )
        st.caption(follow.follow_note)

        # 短依据（最多 4 条，含综合）
        short_bits: list[str] = []
        for line in dual_lines:
            if line.startswith("【综合】"):
                short_bits.append(line.replace("【综合】", "").strip())
        for b in advice.bullets:
            if not isinstance(b, str):
                continue
            if b.startswith("【综合】"):
                continue
            if any(
                k in b
                for k in ("买入 ", "浮盈", "建议", "止蚀", "约 ")
            ):
                short_bits.append(b)
            if len(short_bits) >= 4:
                break
        if short_bits:
            with st.expander("简要依据（可收起）", expanded=False):
                for b in short_bits[:5]:
                    st.markdown(f"- {b}")

        # ========== 进阶：双计划明细（默认折叠，防信息过载）==========
        with st.expander("进阶：入场日 / 即时明细（一般不用看）", expanded=False):
            st.caption(
                "入场日 = 合同（硬止蚀 + 原 T1/T2）；"
                "即时 = 天气预报（抬止蚀 / 延伸目标）。"
                "日常只跟上方「现在止蚀 + 下一动作价」。"
            )
            if entry_lv is None or entry_lv.stop is None:
                if buy_date:
                    st.info(
                        f"未能重建 {buy_date.isoformat()} 的入场日计划"
                        + (f"：{load_err}" if load_err else "")
                    )
            col_a, col_b = st.columns(2)
            with col_a:
                if entry_lv is not None and entry_lv.stop is not None:
                    _render_levels_card(
                        entry_lv,
                        title=f"🔒 入场日（{entry_as_of or entry_lv.as_of}）",
                        subtitle="合同：硬止蚀与原目标",
                    )
            with col_b:
                _render_levels_card(
                    live_lv,
                    title="📡 即时",
                    subtitle="天气：延伸 / 阻力",
                )
            for line in dual_lines:
                st.caption(line)

        # ---- 对照表 ----
        with st.expander("对照表 · 盈亏估算 · 日志前参考", expanded=False):
            st.markdown("#### 入场日 vs 即时（一览）")
            rows = [
                ("计划限价 E", entry_lv.entry_plan if entry_lv else None, live_lv.entry_plan),
                ("止蚀 S", entry_lv.stop if entry_lv else None, live_lv.stop),
                ("目标 T1", entry_lv.t1 if entry_lv else None, live_lv.t1),
                ("目标 T2", entry_lv.t2 if entry_lv else None, live_lv.t2),
                ("最近阻力", entry_lv.resistance if entry_lv else None, live_lv.resistance),
                ("最近支撑", entry_lv.support if entry_lv else None, live_lv.support),
                ("区下沿", entry_lv.zone_lo if entry_lv else None, live_lv.zone_lo),
                ("区上沿", entry_lv.zone_hi if entry_lv else None, live_lv.zone_hi),
            ]
            e_label = entry_as_of or "入场日"
            table_md = (
                f"| 项目 | 入场日（{e_label}） | 即时 | 相对你的买入 |\n"
                "|------|------------------|------|-------------|\n"
            )
            for name_r, ev, lv in rows:
                note = "—"
                ref = ev if ev is not None else lv
                if ref is not None and buy_f > 0:
                    note = f"{(ref / buy_f - 1.0) * 100:+.1f}%"
                table_md += f"| {name_r} | {_fmt(ev)} | {_fmt(lv)} | {note} |\n"
            st.markdown(table_md)

            st.markdown("#### 你的成交 vs 计划")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("你的买入", f"{buy_f:.2f}")
            d2.metric("现价", f"{last_f:.2f}")
            e_ref = (
                entry_lv.entry_plan
                if entry_lv and entry_lv.entry_plan is not None
                else live_lv.entry_plan
            )
            if e_ref is not None:
                vs_e = (buy_f / float(e_ref) - 1.0) * 100.0
                d3.metric(
                    "买入 vs 入场E",
                    f"{vs_e:+.2f}%",
                    "贵了" if vs_e > 0.5 else ("更便宜" if vs_e < -0.5 else "接近"),
                )
            else:
                d3.metric("买入 vs 入场E", "—")
            if stop_hard is not None and buy_f > float(stop_hard):
                risk_pct = (buy_f - float(stop_hard)) / buy_f * 100.0
                d4.metric("买入到入场止蚀", f"{risk_pct:.2f}%")
            else:
                d4.metric("买入到止蚀", "—")

            st.markdown("#### 若按现价平仓 / 打到计划位（约）")
            notional = DEFAULT_NOTIONAL_HKD
            if shares and int(shares) > 0:
                sh = int(shares)
                pnl_now = sh * (last_f - buy_f)
                h1, h2, h3 = st.columns(3)
                h1.metric("现价浮动(股币)", f"{pnl_now:+,.2f}")
                if t1_use:
                    h2.metric(
                        "若到入场T1",
                        f"{sh * (float(t1_use) - buy_f):+,.2f}",
                    )
                if stop_hard:
                    h3.metric(
                        "若打入场止蚀",
                        f"{sh * (float(stop_hard) - buy_f):+,.2f}",
                    )
                st.caption("股数 × 股价币种（美股美元）。")
            else:

                def _hkd_move(px: float) -> float:
                    return notional * (px / buy_f - 1.0)

                h1, h2, h3 = st.columns(3)
                h1.metric("现价浮动(5k名义)", f"{_hkd_move(last_f):+,.0f} HKD")
                if t1_use:
                    h2.metric("到入场T1", f"{_hkd_move(float(t1_use)):+,.0f} HKD")
                if stop_hard:
                    h3.metric("到入场止蚀", f"{_hkd_move(float(stop_hard)):+,.0f} HKD")
                st.caption(f"未填股数时按名义 **{notional:.0f} HKD**。")

        if structure_bits or (entry_lv and entry_lv.structure_bits):
            with st.expander("计划结构说明（E/S/T）", expanded=False):
                if entry_lv and entry_lv.structure_bits:
                    st.markdown("**入场日**")
                    for b in entry_lv.structure_bits:
                        st.caption(f"· {b}")
                if structure_bits:
                    st.markdown("**即时**")
                    for b in structure_bits:
                        st.caption(f"· {b}")

        render_journal_panel(
            key_prefix="hold_jr",
            default_symbol=sym,
            default_name=str(name),
            default_horizon=(
                entry_lv.horizon_label if entry_lv else live_lv.horizon_label
            ),
            default_entry=buy_f,
            default_stop=float(advice.suggested_stop or stop_hard or buy_f * 0.97),
            default_target=float(t1_use) if t1_use else None,
            default_shares=int(shares) if shares else 0,
            default_verdict=advice.action,
            default_exit_px=last_f,
            expanded=False,
        )
    else:
        st.info("👆 填好买入价与买入日期后，会自动加载入场日 + 即时计划与持仓建议。")
        render_journal_panel(key_prefix="hold_jr_empty", expanded=False)
