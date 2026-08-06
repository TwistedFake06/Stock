"""Dedicated page: I already bought — hold / take-profit / stop advice."""
from __future__ import annotations

from datetime import date

import streamlit as st

from position_coach import advise_open_position
from stock_service import cache_bucket, cached_info, fetch_history, normalize_symbol
from trade_journal import add_trade
from trade_sop import build_trade_sop


def render_hold_page(symbol: str, period: str = "1y", interval: str = "1d") -> None:
    """Always-visible buy-price coach (does not depend on long tabs)."""
    st.markdown("# 💰 我已买入")
    st.markdown(
        f"股票 **`{symbol}`** · 填你的**成交买入价**，根据现价与计划止蚀/目标，"
        "建议 **持有 / 止盈 / 止蚀**。"
    )

    sym = normalize_symbol(symbol)
    # Light quote for last price even if full SOP fails
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
            "股数（可选）",
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

    st.caption("可选：加载完整短线计划（止蚀/T1/T2）。若网络慢可先不勾，仅用买入价 vs 现价。")
    use_plan = st.checkbox("结合主周期计划（止蚀/T1/T2）", value=True, key="hold_use_plan")

    plan_stop = plan_t1 = plan_t2 = plan_entry = None
    max_days = 10
    bias_label, bias_score = "—", 0.0
    horizon_label = "0–2周"

    if st.button("生成持仓建议", type="primary", use_container_width=True, key="hold_page_go"):
        st.session_state["hold_page_ran"] = True
        st.session_state[f"hold_buy_{sym}"] = buy_px
        st.session_state[f"hold_sh_{sym}"] = shares

    if not st.session_state.get("hold_page_ran") and not st.session_state.get("hold_auto"):
        # First visit: still compute once when they have a price
        st.session_state["hold_auto"] = True

    # Auto-run advice whenever we have prices (so user always sees result area)
    if buy_px and (last or use_plan):
        sop = None
        if use_plan:
            with st.spinner("加载计划中…"):
                try:
                    sop = build_trade_sop(
                        sym,
                        period=period,
                        interval=interval,
                        capital=50_000 / 7.8,
                        risk_pct=1.0,
                        primary_horizon="h1",
                    )
                    primary = getattr(sop, "primary_plan", None) or getattr(sop, "swing_h1", None)
                    h2 = getattr(sop, "swing_h2", None)
                    exit_pl = getattr(sop, "exit_plan", None)
                    if primary:
                        plan_stop = primary.stop_loss
                        plan_t1 = primary.target
                        plan_entry = primary.entry_plan
                        max_days = primary.bars
                        horizon_label = primary.label
                    if h2:
                        plan_t2 = h2.target
                    if exit_pl:
                        max_days = exit_pl.max_hold_days
                    if sop.last_price is not None:
                        last = float(sop.last_price)
                    bias_label = sop.bias
                    bias_score = float(sop.bias_score or 0)
                    name = sop.name or name
                except Exception as exc:
                    st.warning(f"完整计划加载失败，改用买入价 vs 现价：{exc}")

        if last is None:
            st.error("没有现价，无法比较。请检查网络或股票代码。")
            return

        # If no plan stop, use 3% default for risk framing only
        if plan_stop is None:
            plan_stop = float(buy_px) * 0.97
        if plan_t1 is None:
            plan_t1 = float(buy_px) * 1.05

        advice = advise_open_position(
            buy_price=float(buy_px),
            last_price=float(last),
            plan_stop=plan_stop,
            plan_t1=plan_t1,
            plan_t2=plan_t2,
            plan_entry=plan_entry or float(buy_px),
            max_hold_days=max_days,
            buy_date=buy_date.isoformat() if buy_date else None,
            shares=int(shares) if shares else None,
            bias_label=bias_label,
            bias_score=bias_score,
        )

        st.markdown("---")
        if advice.color == "red":
            st.error(f"## {advice.action}")
        elif advice.color == "amber":
            st.warning(f"## {advice.action}")
        else:
            st.success(f"## {advice.action}")
        st.markdown(f"### {advice.headline}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("买入价", f"{float(buy_px):.2f}")
        m2.metric("现价", f"{float(last):.2f}")
        m3.metric("浮盈%", f"{advice.pnl_pct:+.2f}%" if advice.pnl_pct is not None else "—")
        m4.metric("建议止蚀", f"{advice.suggested_stop:.2f}" if advice.suggested_stop else "—")
        m5.metric("浮盈R", f"{advice.pnl_r:+.2f}" if advice.pnl_r is not None else "—")

        st.markdown("#### 依据")
        for b in advice.bullets:
            st.markdown(f"- {b}")

        if plan_stop or plan_t1:
            st.markdown("#### 计划对照")
            p1, p2, p3 = st.columns(3)
            p1.metric("计划止蚀", f"{plan_stop:.2f}" if plan_stop else "—")
            p2.metric("T1", f"{plan_t1:.2f}" if plan_t1 else "—")
            p3.metric("T2", f"{plan_t2:.2f}" if plan_t2 else "—")

        if st.button("写入交易日志", key="hold_page_journal"):
            add_trade(
                symbol=sym,
                name=str(name),
                horizon=horizon_label,
                entry=float(buy_px),
                stop=float(advice.suggested_stop or plan_stop or buy_px * 0.97),
                target=plan_t1,
                shares=int(shares) if shares else 0,
                model_verdict=advice.action,
                notes=f"持仓页:{advice.action}; 现价{last}",
                opened=buy_date.isoformat() if buy_date else None,
            )
            st.success("已写入日志")
    else:
        st.info("👆 填好买入价后，建议会显示在下方。")
