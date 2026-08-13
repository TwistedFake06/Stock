"""Dedicated page: I already bought — hold / take-profit / stop advice."""
from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from position_coach import advise_open_position
from stock_service import cache_bucket, cached_info, fetch_history, normalize_symbol
from trade_journal import add_trade
from trade_sop import DEFAULT_NOTIONAL_HKD, build_trade_sop


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


def render_hold_page(symbol: str, period: str = "1y", interval: str = "1d") -> None:
    """Always-visible buy-price coach (does not depend on long tabs)."""
    st.markdown("# 💰 我已买入")
    st.markdown(
        f"股票 **`{symbol}`** · 填你的**成交买入价**，系统选出 **最优短线计划** "
        "（止蚀 / T1 / T2），再对照现价建议 **持有 / 止盈 / 止蚀**。"
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
        buy_date = st.date_input(
            "买入日期（可选）",
            value=date.today(),
            key="hold_page_buy_date",
        )

    c4, c5 = st.columns(2)
    with c4:
        use_plan = st.checkbox(
            "加载最优短线计划（止蚀/T1/T2）",
            value=True,
            key="hold_use_plan",
            help="与投资SOP同源：区中下计划限价、止蚀ATR、目标至少1:1",
        )
    with c5:
        horizon_pref = st.selectbox(
            "计划周期",
            ["自动选最优", "强制 0–2周", "强制 2–4周"],
            index=0,
            key="hold_horizon_pref",
            help="自动：比较两套计划的完整性、净R:R、胜率后选更好的一套做管仓",
        )

    plan_stop = plan_t1 = plan_t2 = plan_entry = None
    plan_zone_lo = plan_zone_hi = None
    plan_rr = plan_rr_net = plan_wr = None
    max_days = 10
    bias_label, bias_score = "—", 0.0
    horizon_label = "0–2周"
    pick_why = ""
    plan_verdict = "—"
    sop = None
    structure_bits: list[str] = []

    if st.button("生成持仓建议", type="primary", use_container_width=True, key="hold_page_go"):
        st.session_state["hold_page_ran"] = True
        st.session_state[f"hold_buy_{sym}"] = buy_px
        st.session_state[f"hold_sh_{sym}"] = shares

    if not st.session_state.get("hold_page_ran") and not st.session_state.get("hold_auto"):
        st.session_state["hold_auto"] = True

    if buy_px and (last or use_plan):
        if use_plan:
            with st.spinner("加载最优计划中…"):
                try:
                    prefer_map = {
                        "自动选最优": "auto",
                        "强制 0–2周": "h1",
                        "强制 2–4周": "h2",
                    }
                    prefer = prefer_map.get(horizon_pref, "auto")
                    # primary_horizon only seeds default; we re-pick below
                    ph = "h2" if prefer == "h2" else "h1"
                    sop = build_trade_sop(
                        sym,
                        period=period,
                        interval=interval,
                        capital=50_000 / 7.8,
                        risk_pct=1.0,
                        primary_horizon=ph,
                    )
                    primary, secondary, horizon_label, pick_why = pick_best_hold_plan(
                        sop, prefer=prefer
                    )
                    exit_pl = getattr(sop, "exit_plan", None)
                    if primary:
                        plan_stop = primary.stop_loss
                        plan_t1 = primary.target
                        plan_entry = primary.entry_plan
                        plan_zone_lo = primary.entry_low
                        plan_zone_hi = primary.entry_high
                        plan_rr = primary.rr
                        plan_rr_net = primary.rr_net
                        plan_wr = getattr(primary, "win_rate_display", None) or (
                            f"{primary.win_rate_pct:.0f}%"
                            if primary.win_rate_pct is not None
                            else None
                        )
                        plan_verdict = primary.verdict or "—"
                        max_days = int(getattr(primary, "bars", None) or 10)
                    if secondary and getattr(secondary, "target", None):
                        # 另一周期目标作 T2（若主计划目标更远则互换逻辑）
                        t_sec = _f(secondary.target)
                        t_pri = _f(plan_t1)
                        if t_sec is not None:
                            if t_pri is None or t_sec > t_pri:
                                plan_t2 = t_sec
                            else:
                                # secondary nearer: keep as T1 if missing, else T2 still secondary
                                plan_t2 = t_sec
                    if exit_pl and getattr(exit_pl, "max_hold_days", None):
                        max_days = int(exit_pl.max_hold_days)
                    if sop.last_price is not None:
                        last = float(sop.last_price)
                    bias_label = sop.bias
                    bias_score = float(sop.bias_score or 0)
                    name = sop.name or name
                    # 结构说明
                    if getattr(sop, "notes", None):
                        for n in sop.notes:
                            if any(
                                k in str(n)
                                for k in ("计划限价", "止损", "目标", "E=", "ATR", "1:1")
                            ):
                                structure_bits.append(str(n))
                                if len(structure_bits) >= 4:
                                    break
                except Exception as exc:
                    st.warning(f"完整计划加载失败，改用买入价 vs 现价：{exc}")

        if last is None:
            st.error("没有现价，无法比较。请检查网络或股票代码。")
            return

        # Fallback levels from fill if no plan
        if plan_stop is None:
            plan_stop = float(buy_px) * 0.97
        if plan_t1 is None:
            plan_t1 = float(buy_px) * 1.05

        # ---- 价位总览卡（计划）----
        st.markdown("---")
        st.markdown("### 📋 最优计划价位（管仓用）")
        if pick_why:
            st.caption(pick_why)
        st.caption(
            f"周期 **{horizon_label}** · 计划结论参考 **{plan_verdict}** · "
            f"多空 **{bias_label}**（{bias_score:+.0f}）· 最多约 **{max_days}** 交易日"
        )

        z1, z2, z3, z4, z5 = st.columns(5)
        z1.metric(
            "计划限价 E",
            f"{plan_entry:.2f}" if plan_entry else "—",
            (
                f"区 {plan_zone_lo:.2f}–{plan_zone_hi:.2f}"
                if plan_zone_lo and plan_zone_hi
                else None
            ),
        )
        z2.metric("计划止蚀 S", f"{plan_stop:.2f}" if plan_stop else "—")
        z3.metric("目标 T1", f"{plan_t1:.2f}" if plan_t1 else "—")
        z4.metric("目标 T2", f"{plan_t2:.2f}" if plan_t2 else "—")
        rr_show = plan_rr_net if plan_rr_net is not None else plan_rr
        z5.metric(
            "净R:R / 胜率",
            f"{rr_show:.2f}" if rr_show is not None else "—",
            plan_wr or None,
        )

        # 你的成交 vs 计划
        st.markdown("#### 你的成交 vs 计划")
        buy_f = float(buy_px)
        last_f = float(last)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("你的买入", f"{buy_f:.2f}")
        d2.metric("现价", f"{last_f:.2f}")
        if plan_entry:
            vs_e = (buy_f / float(plan_entry) - 1.0) * 100.0
            d3.metric(
                "买入 vs 计划E",
                f"{vs_e:+.2f}%",
                "贵了（相对计划）" if vs_e > 0.5 else ("更便宜" if vs_e < -0.5 else "接近计划"),
            )
        else:
            d3.metric("买入 vs 计划E", "—")
        if plan_stop and buy_f > float(plan_stop):
            risk_pct = (buy_f - float(plan_stop)) / buy_f * 100.0
            d4.metric("买入到止蚀", f"{risk_pct:.2f}%")
        else:
            d4.metric("买入到止蚀", "—")

        # 价位图示（文字）
        levels = []
        if plan_stop:
            levels.append(("止蚀 S", float(plan_stop)))
        levels.append(("你的买入", buy_f))
        if plan_entry:
            levels.append(("计划E", float(plan_entry)))
        levels.append(("现价", last_f))
        if plan_t1:
            levels.append(("T1", float(plan_t1)))
        if plan_t2:
            levels.append(("T2", float(plan_t2)))
        levels_sorted = sorted(levels, key=lambda x: x[1])
        st.caption(
            "价位由低到高： "
            + "  <  ".join(f"**{n}** {v:.2f}" for n, v in levels_sorted)
        )

        # HKD 空间（按股数或 5000 名义）
        st.markdown("#### 若按现价平仓 / 打到计划位（约）")
        notional = DEFAULT_NOTIONAL_HKD
        if shares and int(shares) > 0:
            sh = int(shares)
            pnl_now = sh * (last_f - buy_f)
            # rough HKD if US stock: user often thinks HKD; show USD and note
            h1, h2, h3 = st.columns(3)
            h1.metric("现价浮动(股币)", f"{pnl_now:+,.2f}")
            if plan_t1:
                h2.metric(
                    "若到 T1(相对买入)",
                    f"{sh * (float(plan_t1) - buy_f):+,.2f}",
                )
            if plan_stop:
                h3.metric(
                    "若打止蚀(相对买入)",
                    f"{sh * (float(plan_stop) - buy_f):+,.2f}",
                )
            st.caption("上表按「股数 × 股价币种」；美股为美元。")
        else:
            # 5000 HKD notional equivalent move from buy
            def _hkd_move(px: float) -> float:
                return notional * (px / buy_f - 1.0)

            h1, h2, h3 = st.columns(3)
            h1.metric("现价浮动(按5k名义)", f"{_hkd_move(last_f):+,.0f} HKD")
            if plan_t1:
                h2.metric("到T1(按5k名义)", f"{_hkd_move(float(plan_t1)):+,.0f} HKD")
            if plan_stop:
                h3.metric("到止蚀(按5k名义)", f"{_hkd_move(float(plan_stop)):+,.0f} HKD")
            st.caption(f"未填股数时，按名义 **{notional:.0f} HKD** 估算比例盈亏。")

        if structure_bits:
            with st.expander("计划结构说明（E/S/T 优化）", expanded=False):
                for b in structure_bits:
                    st.caption(f"· {b}")

        # ---- 持仓动作建议 ----
        advice = advise_open_position(
            buy_price=buy_f,
            last_price=last_f,
            plan_stop=plan_stop,
            plan_t1=plan_t1,
            plan_t2=plan_t2,
            plan_entry=plan_entry or buy_f,
            max_hold_days=max_days,
            buy_date=buy_date.isoformat() if buy_date else None,
            shares=int(shares) if shares else None,
            bias_label=bias_label,
            bias_score=bias_score,
        )

        st.markdown("---")
        st.markdown("### 持仓动作建议")
        if advice.color == "red":
            st.error(f"## {advice.action}")
        elif advice.color == "amber":
            st.warning(f"## {advice.action}")
        else:
            st.success(f"## {advice.action}")
        st.markdown(f"### {advice.headline}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("买入价", f"{buy_f:.2f}")
        m2.metric("现价", f"{last_f:.2f}")
        m3.metric(
            "浮盈%",
            f"{advice.pnl_pct:+.2f}%" if advice.pnl_pct is not None else "—",
        )
        m4.metric(
            "建议止蚀",
            f"{advice.suggested_stop:.2f}" if advice.suggested_stop else "—",
        )
        m5.metric(
            "浮盈R",
            f"{advice.pnl_r:+.2f}" if advice.pnl_r is not None else "—",
        )

        st.markdown("#### 依据")
        for b in advice.bullets:
            st.markdown(f"- {b}")

        st.markdown("#### 执行清单")
        _sug = (
            f"{advice.suggested_stop:.2f}"
            if advice.suggested_stop is not None
            else "—"
        )
        _t1 = f"{plan_t1:.2f}" if plan_t1 is not None else "—"
        _t2 = f"{plan_t2:.2f}" if plan_t2 is not None else "—"
        _ps = f"{plan_stop:.2f}" if plan_stop is not None else "—"
        st.markdown(
            f"1. **硬止蚀**：{_ps}（或建议止蚀 {_sug}）\n"
            f"2. **T1 减仓**：{_t1} 到价减约一半，止蚀抬到保本\n"
            f"3. **T2 / 时间**：{_t2} · 最多约 {max_days} 个交易日\n"
            "4. 禁止：摊平、下移止蚀、无计划死扛"
        )

        if st.button("写入交易日志", key="hold_page_journal"):
            add_trade(
                symbol=sym,
                name=str(name),
                horizon=horizon_label,
                entry=buy_f,
                stop=float(advice.suggested_stop or plan_stop or buy_f * 0.97),
                target=plan_t1,
                shares=int(shares) if shares else 0,
                model_verdict=advice.action,
                notes=(
                    f"持仓页:{advice.action}; 现价{last_f}; "
                    f"计划E={plan_entry}; S={plan_stop}; T1={plan_t1}; {pick_why}"
                ),
                opened=buy_date.isoformat() if buy_date else None,
            )
            st.success("已写入日志")
    else:
        st.info("👆 填好买入价后，会自动加载最优计划价位与持仓建议。")
