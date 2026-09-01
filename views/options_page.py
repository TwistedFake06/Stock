"""Streamlit page: 期权价差."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from indicators import enrich
from options_greeks import calc_spread_greeks
from options_payoff import (
    build_daily_mark_calendar,
    build_payoff_ladder,
    payoff_per_contract,
    payoff_zones_summary,
)
from options_plain import plain_spread_steps
from options_position import calc_options_position
from options_spreads import (
    INDEX_ETF_WHITELIST,
    analyze_options_spreads,
    ideas_to_frame,
    is_options_eligible,
    legs_to_frame,
    options_symbol,
)

try:
    from options_methods import methods_to_rows
except Exception:  # pragma: no cover
    def methods_to_rows(idea):  # type: ignore
        return []

from stock_service import fetch_history
from ui_mobile import plotly_chart as mobile_plotly


def render_options(symbol: str) -> None:
    st.markdown(
        """
        <div class="hero-wrap">
          <p class="hero-kicker">Home · Vertical Spread</p>
          <p class="hero-title">期权价差</p>
          <p class="hero-desc">
            今天<strong style="color:#90caf9">卖出</strong>一组价差先收钱
            （或买进先付钱）→
            <strong style="color:#d2a8ff">几天后买回</strong>平仓。
            只做 QQQ / VOO / SPY 等大盘 ETF。
          </p>
          <div class="pill-row">
            <span class="pill pill-blue">卖出 / 买回</span>
            <span class="pill pill-purple">高胜率规则</span>
            <span class="pill pill-green">流动性</span>
            <span class="pill pill-amber">不用行权</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    opt_default = options_symbol(symbol)
    if not is_options_eligible(opt_default):
        opt_default = "QQQ"

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        etf_list = sorted(INDEX_ETF_WHITELIST.keys())
        for hot in ("QQQ", "SPY", "VOO"):
            if hot in etf_list:
                etf_list.remove(hot)
                etf_list.insert(0, hot)
        idx = etf_list.index(opt_default) if opt_default in etf_list else 0
        opt_sym = st.selectbox(
            "ETF",
            etf_list,
            index=idx,
            format_func=lambda s: f"{s} · {INDEX_ETF_WHITELIST[s]}",
        )
    with c2:
        target_dte = st.select_slider(
            "大约几天到期",
            options=[14, 21, 30, 45, 60],
            value=30,
        )
    with c3:
        width_mode = st.selectbox("宽度", ["自动", "1", "2", "5", "10"], index=0)

    st.markdown("#### 風險參數（1R 風控）")
    HKD_PER_USD = 7.8
    r1, r2 = st.columns(2)
    with r1:
        account_hkd = st.number_input(
            "帳戶資金 (HKD)",
            min_value=1000.0,
            value=50000.0,
            step=1000.0,
            help="默认 50,000 HKD；期權美元標的會按匯率折算",
        )
    with r2:
        risk_pct_opt = st.number_input(
            "單筆風險 1R (%)",
            min_value=0.25,
            max_value=5.0,
            value=1.0,
            step=0.25,
            help="相對帳戶資金的百分比",
        )
    account_size = float(account_hkd) / HKD_PER_USD
    risk_per_trade = account_size * (float(risk_pct_opt) / 100.0)
    st.caption(
        f"約合 USD {account_size:,.0f} · 1R ≈ USD {risk_per_trade:,.0f} "
        f"(HKD {account_hkd * risk_pct_opt / 100:,.0f})"
    )

    with st.spinner("扫描方向与价差…"):
        hist_opt = fetch_history(opt_sym, period="6mo", interval="1d")
        width_val = None if width_mode == "自动" else float(width_mode)
        opt_rep = analyze_options_spreads(
            opt_sym,
            target_dte=int(target_dte),
            width=width_val,
            hist_df=hist_opt if not hist_opt.empty else None,
        )

    # Always resolve timing (fallback if report missing field / old cache)
    timing = getattr(opt_rep, "timing", None)
    if timing is None:
        try:
            from options_timing import assess_spread_timing

            best_for_t = (
                getattr(opt_rep, "best_playbook", None)
                or getattr(opt_rep, "best", None)
            )
            timing = assess_spread_timing(
                direction=getattr(opt_rep, "direction", None),
                best=best_for_t,
                after_hours=bool(getattr(opt_rep, "after_hours", False)),
                dte=getattr(opt_rep, "dte", None),
                iv_atm=getattr(opt_rep, "iv_atm", None),
                ideas_count=len(getattr(opt_rep, "ideas", None) or []),
                quote_warning=getattr(opt_rep, "quote_warning", "") or "",
                pricing_note=getattr(opt_rep, "pricing_note", "") or "",
            )
        except Exception as _exc:
            timing = None
            st.caption(f"(适合度模块暂不可用: {_exc})")

    def _render_timing_banner(t) -> None:
        """Big, hard-to-miss block at top of results."""
        st.markdown("---")
        st.markdown("## 现在适合做 Spread 吗？")
        if t is None:
            st.warning("暂时无法评估适合度（分析未返回 timing）。")
            return
        box = st.container(border=True)
        with box:
            t1, t2, t3 = st.columns([1.2, 1, 2.2])
            with t1:
                if t.color == "green":
                    st.success(f"### {t.verdict}")
                elif t.color == "amber":
                    st.warning(f"### {t.verdict}")
                else:
                    st.error(f"### {t.verdict}")
            with t2:
                st.metric("适合度评分", f"{t.score:.0f} / 100")
            with t3:
                st.markdown(f"**{t.headline}**")
                st.markdown(f"👉 {t.action}")
            if t.preferred_style:
                st.info(f"风格建议：{t.preferred_style}")
            for b in (t.bullets or [])[:8]:
                st.markdown(f"- {b}")
            with st.expander("适合度检查清单（明细）", expanded=True):
                rows = []
                for c in t.checklist or []:
                    mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(
                        c.get("status", ""), "-"
                    )
                    rows.append(
                        {
                            "项": c.get("name", ""),
                            "结果": mark,
                            "说明": c.get("detail", ""),
                        }
                    )
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        width="stretch",
                        hide_index=True,
                    )
                st.caption(
                    "实盘辅助判断，不是下单指令。盘后请只做计划，开盘后再核价。"
                )
        st.markdown("---")

    if not opt_rep.eligible:
        st.error(opt_rep.message)
        st.write(", ".join(f"`{k}`" for k in sorted(INDEX_ETF_WHITELIST)))
        _render_timing_banner(timing)
    else:
        # Show timing FIRST so it is never missed
        _render_timing_banner(timing)

        d = opt_rep.direction
        dir_pill = {
            "看多": "pill-red",
            "看空": "pill-green",
            "中性": "pill",
        }.get(d.direction if d else "中性", "pill")

        # 盘后 / 报价质量提示
        qw = getattr(opt_rep, "quote_warning", "") or ""
        if qw:
            st.warning(qw)
        pn = getattr(opt_rep, "pricing_note", "") or ""
        fo = getattr(opt_rep, "filtered_out", 0) or 0
        if pn or fo:
            cap_bits = []
            if pn:
                cap_bits.append(pn)
            if fo:
                cap_bits.append(f"硬过滤剔除 {fo} 个（流动性 / 权利金宽度不合规）")
            st.caption(" · ".join(cap_bits))

        iv_txt = f"{opt_rep.iv_atm * 100:.0f}%" if opt_rep.iv_atm is not None else "—"

        st.markdown(
            f"""
            <div class="glass-card">
              <p class="section-label">Market snapshot</p>
              <div class="pill-row">
                <span class="pill pill-blue">{opt_sym} · {opt_rep.spot:.2f}</span>
                <span class="pill {dir_pill}">方向 {(d.direction if d else '—')}</span>
                <span class="pill">{(d.strength if d else '—')} · {(f'{d.score:+.0f}' if d else '')}</span>
                <span class="pill">到期 {opt_rep.selected_expiry or '—'}</span>
                <span class="pill">还剩 {opt_rep.dte if opt_rep.dte is not None else '—'} 天</span>
                <span class="pill pill-amber">波动 ~{iv_txt}</span>
              </div>
              <p class="mini-note">{(d.style_hint if d else '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if d and d.reasons:
            with st.expander("方向依据", expanded=False):
                for r in d.reasons:
                    st.markdown(f"- {r}")

        if not opt_rep.ideas or not opt_rep.best:
            st.warning("暂时算不出可用价差（网络或盘后报价问题）。")
            st.info(
                "当前无可用推荐时，不会显示 Greeks 与 1R 仓位建议。"
                "可尝试把到期天数调到 21/30/45，或在美股交易时段再刷新。"
            )
        else:
            hi = getattr(opt_rep, "best_winrate", None)
            hi_al = getattr(opt_rep, "best_winrate_aligned", None)
            pb = getattr(opt_rep, "best_playbook", None) or opt_rep.best
            pb_wr = getattr(opt_rep, "best_playbook_wr", None) or hi_al or hi
            best = pb
            credit_candidates = list(getattr(opt_rep, "credit_candidates", None) or [])
            is_range_market = bool(d and getattr(d, "market_regime", "") == "震荡")

            if is_range_market:
                st.markdown("### 震荡市：先选要卖哪一侧的风险")
                if d.range_low is not None and d.range_high is not None:
                    st.caption(
                        f"近 20 日参考区间：{d.range_low:.2f} - {d.range_high:.2f}。"
                        "靠近下缘且下方有支撑，比较 OTM 卖 Put；靠近上缘且上方有压力，比较 OTM 卖 Call。"
                        "两侧都不清楚时不做，不把双侧风险当成单一仓位。"
                    )

                if credit_candidates:
                    candidate_cols = st.columns(len(credit_candidates))
                    for column, candidate in zip(candidate_cols, credit_candidates):
                        with column:
                            side = "OTM 卖 Put" if candidate.code == "bull_put" else "OTM 卖 Call"
                            pop = getattr(candidate, "win_rate_profit", None)
                            st.markdown(f"**{side}**")
                            st.caption(candidate.name)
                            st.metric("短腿 OTM", candidate.otm_label or "-" )
                            st.metric("价宽 / 最大风险", f"${candidate.width:.0f} / ${candidate.max_loss:.0f}")
                            st.caption(
                                f"净收 ${candidate.net_credit:.2f}/股 · POP "
                                f"{f'{pop:.0f}%' if pop is not None else '-'}"
                            )

                    selected_credit_index = st.radio(
                        "这次要看哪一侧？",
                        options=list(range(len(credit_candidates))),
                        horizontal=True,
                        format_func=lambda index: (
                            "卖 OTM Put（偏多 / 守区间下缘）"
                            if credit_candidates[index].code == "bull_put"
                            else "卖 OTM Call（偏空 / 守区间上缘）"
                        ),
                        key="opt_range_side",
                    )
                    best = credit_candidates[selected_credit_index]
                else:
                    st.warning("震荡市未找到可成交的 OTM 信用价差；等待盘中报价或改选到期日。")

            is_credit = best.net_credit is not None
            do_word = "卖出价差 · 先收钱" if is_credit else "买进价差 · 先付钱"
            close_word = "几天后买回" if is_credit else "几天后卖出"

            st.caption(
                "**实盘辅助**：POP=对数正态到期；**到期EV**=分段盈亏积分；"
                "**管理EV**=50%止盈/2R止损路径；开仓=自然成交(bid/ask)。Yahoo 可能延迟。"
            )
            ev_m = getattr(best, "expected_value_managed", None)
            ev0 = getattr(best, "expected_value", None)
            if ev0 is not None or ev_m is not None:
                e1, e2, e3 = st.columns(3)
                if ev0 is not None:
                    e1.metric("到期 EV $/张", f"{ev0:+.0f}")
                if ev_m is not None:
                    e2.metric("管理 EV $/张", f"{ev_m:+.0f}", help="50%止盈 / 2R止损路径期望")
                wr_show = getattr(best, "win_rate_profit", None)
                if wr_show is not None:
                    e3.metric("POP %", f"{wr_show:.0f}")
            if getattr(best, "win_rate_method", None):
                with st.expander("POP / EV 模型假设（实盘）", expanded=False):
                    st.write(best.win_rate_method)
                    st.markdown(
                        """
- **POP**：到期结算盈亏 > 0 的概率
- **满盈概率**：价格落在最大利润区的概率
- **到期 EV**：完整 payoff 曲线积分（含中间亏损区）+ 折现 + 摩擦
- **管理 EV**：日路径 BS 盯市，**50% 最大利润止盈 / 2R 止损**（更接近实盘管仓）
- **成本**：摩擦约 $0.03/股 + 管理路径含约 $2.6 往返佣金
- **下单**：以券商实时 NBBO 限价为准
                        """
                    )
            wr_p = getattr(best, "win_rate_profit", None) or getattr(best, "pop_est", None)
            liq_lab = getattr(best, "liquidity_label", "") or "—"
            style = getattr(best, "playbook_style", "") or "实战扫描"
            plain = getattr(best, "playbook_plain", "") or ""
            src = getattr(best, "playbook_source", "") or ""
            fit = getattr(best, "playbook_fit", None)
            c_w = getattr(best, "metric_credit_width", None)
            half_p = getattr(best, "metric_half_profit", None)
            half_bb = getattr(best, "metric_half_buyback", None)
            mc = getattr(best, "method_composite", None)
            pricing_mode = getattr(best, "pricing_mode", "") or ""
            prem = f"${best.net_credit:.2f}" if is_credit else f"${best.net_debit:.2f}"
            prem_lbl = "今天大约收" if is_credit else "今天大约付"
            close_lbl = "50%止盈买回价" if is_credit else "50%止盈卖出价"
            be = best.breakevens[0] if best.breakevens else None
            fit_line = (
                f'<p class="mini-note">贴合 {fit:.0f}/100 · {src}</p>'
                if fit is not None
                else ""
            )
            liq_line = (
                f'<p class="mini-note">{getattr(best, "liquidity_detail", "")}</p>'
                if getattr(best, "liquidity_detail", "")
                else ""
            )
            price_note = ""
            if pricing_mode == "natural":
                price_note = (
                    '<p class="mini-note">定价：自然成交（卖腿=bid · 买腿=ask），比中间价更保守</p>'
                )
            elif pricing_mode == "mid_mixed":
                price_note = (
                    '<p class="mini-note">定价：部分腿无买卖价，已混用中间价</p>'
                )
            elif pricing_mode == "mid_only":
                price_note = (
                    '<p class="mini-note">定价：无有效 bid/ask，整组用中间价估算（盘后常见）</p>'
                )
            half_note = ""
            if half_bb is not None and half_p is not None:
                if is_credit:
                    half_note = (
                        f'<p class="mini-note">常见做法：浮盈到约 ${half_p:.0f}/张时，'
                        f'把价差<strong style="color:#d2a8ff">买回约 ${half_bb:.2f}/股</strong>离场</p>'
                    )
                else:
                    half_note = (
                        f'<p class="mini-note">常见做法：浮盈到约 ${half_p:.0f}/张时，'
                        f'把价差<strong style="color:#d2a8ff">卖出约 ${half_bb:.2f}/股</strong>落袋</p>'
                    )
            mc_pill = (
                f'<span class="pill pill-amber">规则 {mc:.0f}</span>'
                if mc is not None
                else ""
            )
            wr_txt = f"{wr_p:.0f}%" if wr_p is not None else "—"
            be_txt = f"{be:.2f}" if be is not None else "—"
            cw_txt = f"{c_w:.0f}%" if c_w is not None else "—"
            half_txt = f"${half_p:.0f}" if half_p is not None else "—"
            half_bb_txt = f"${half_bb:.2f}" if half_bb is not None else "—"

            st.markdown(
                f"""
                <div class="glass-card glass-card-accent">
                  <p class="section-label">Best pick · 推荐</p>
                  <p class="hero-title" style="font-size:1.25rem;">{do_word}</p>
                  <p class="hero-desc" style="margin-top:0.25rem;">
                    然后 <strong style="color:#d2a8ff">{close_word}</strong>
                    · 方向 {(d.direction if d else '—')}
                    · 像「{style}」
                  </p>
                  <div class="pill-row">
                    <span class="pill pill-blue">{best.name}</span>
                    <span class="pill pill-green">赢面 {wr_txt}</span>
                    <span class="pill pill-purple">流动性 {liq_lab}</span>
                    {mc_pill}
                  </div>
                  <div class="kpi-grid">
                    <div class="kpi">
                      <p class="kpi-label">{prem_lbl}</p>
                      <p class="kpi-value accent">{prem}<span style="font-size:0.75rem;font-weight:500"> /股</span></p>
                    </div>
                    <div class="kpi">
                      <p class="kpi-label">{close_lbl}</p>
                      <p class="kpi-value accent">{half_bb_txt}<span style="font-size:0.75rem;font-weight:500"> /股</span></p>
                    </div>
                    <div class="kpi">
                      <p class="kpi-label">一半利润约</p>
                      <p class="kpi-value up">{half_txt}<span style="font-size:0.75rem;font-weight:500"> /张</span></p>
                    </div>
                    <div class="kpi">
                      <p class="kpi-label">最多赚 / 张</p>
                      <p class="kpi-value up">${best.max_profit:.0f}</p>
                    </div>
                    <div class="kpi">
                      <p class="kpi-label">最多亏 / 张</p>
                      <p class="kpi-value down">${best.max_loss:.0f}</p>
                    </div>
                    <div class="kpi">
                      <p class="kpi-label">不赚不亏价</p>
                      <p class="kpi-value">{be_txt}</p>
                    </div>
                    <div class="kpi">
                      <p class="kpi-label">权利金/宽度</p>
                      <p class="kpi-value">{cw_txt}</p>
                    </div>
                  </div>
                  <p class="mini-note">{plain}</p>
                  {half_note}
                  {price_note}
                  {fit_line}
                  {liq_line}
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                greeks = (
                    calc_spread_greeks(best.legs, opt_rep.spot, best.dte)
                    if best.legs and len(best.legs) >= 2
                    else None
                )
                pos_plan = calc_options_position(
                    max_loss_per_contract=best.max_loss,
                    max_profit_per_contract=best.max_profit,
                    account_size=account_size,
                    risk_per_trade=risk_per_trade,
                )
            except Exception:
                greeks = None
                pos_plan = None

            if greeks is not None:
                st.markdown("#### Greeks")
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("短腿 Delta", f"{greeks.short_delta:.2f}")
                g2.metric("淨 Delta", f"{greeks.net_delta:.2f}")
                g3.metric("淨 Theta /日", f"${greeks.net_theta:.1f}")
                g4.metric("淨 Vega", f"{greeks.net_vega:.1f}")

            if pos_plan is not None:
                st.markdown("#### 倉位建議（依你的 1R 風控）")
                if pos_plan.contracts >= 1:
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("建議張數", f"{pos_plan.contracts} 張")
                    p2.metric("總最大虧損", f"${pos_plan.total_max_loss:.0f}")
                    p3.metric("總最大利潤", f"${pos_plan.total_max_profit:.0f}")
                    p4.metric("盈虧比", f"{pos_plan.r_multiple:.2f}")
                    st.caption(f"風險佔帳戶：{pos_plan.risk_pct:.2f}%")
                    for note in pos_plan.notes:
                        st.info(note)
                else:
                    st.error(pos_plan.notes[0] if pos_plan.notes else "風險過大，不建議開倉")

            steps = plain_spread_steps(best)
            steps_html = '<div class="glass-card"><p class="section-label">How to trade · 两步</p>'
            nstep = 0
            for line in steps:
                clean = (
                    line.replace("**", "")
                    .replace("第1步（今天）：", "")
                    .replace("第2步（几天后）：", "")
                )
                safe = html.escape(clean)
                if "分两步" in line or line.startswith("做法"):
                    steps_html += f'<p class="mini-note">{safe}</p>'
                    continue
                nstep += 1
                steps_html += (
                    f'<div class="step-row"><div class="step-num">{nstep}</div>'
                    f'<div class="step-body">{safe}</div></div>'
                )
            steps_html += "</div>"
            st.markdown(steps_html, unsafe_allow_html=True)

            with st.expander("买卖两腿 + 流动性明细", expanded=False):
                st.dataframe(legs_to_frame(best), width="stretch", hide_index=True)

            mrows = methods_to_rows(best)
            if mrows:
                with st.expander("热门规则打分（真实盘面 + 常见做法）", expanded=False):
                    st.caption("Yahoo 期权链 · 实盘规则筛子 · 非保证收益")
                    st.dataframe(pd.DataFrame(mrows), width="stretch", hide_index=True)

            ops = getattr(best, "ops_plan", None)
            if ops is not None:
                with st.expander("怎么进 / 管 / 出", expanded=False):
                    st.markdown(f"**进场** — {ops.entry}")
                    st.markdown(f"**持有** — {ops.manage}")
                    for e in ops.exit_rules:
                        st.markdown(f"- {e}")

            reasons = getattr(best, "playbook_reasons", None) or []
            if reasons:
                with st.expander("为什么像这种做法", expanded=False):
                    for r in reasons:
                        st.markdown(f"- {r}")

            if pb_wr is not None and getattr(pb_wr, "win_rate_profit", None) is not None:
                same = pb_wr is best
                if same:
                    extra = (
                        f" · 大赢约 {pb_wr.win_rate_max:.0f}%"
                        if pb_wr.win_rate_max is not None
                        else ""
                    )
                    st.markdown(
                        f"""
                        <div class="glass-card">
                          <p class="section-label">Highest win rate · 最高赢面</p>
                          <p class="kpi-value accent" style="font-size:1.35rem;">{pb_wr.win_rate_profit:.0f}%</p>
                          <p class="mini-note">与推荐是同一套{extra}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    prem2 = (
                        f"收 ${pb_wr.net_credit:.2f}"
                        if pb_wr.net_credit is not None
                        else f"付 ${pb_wr.net_debit:.2f}"
                    )
                    st.markdown(
                        f"""
                        <div class="glass-card">
                          <p class="section-label">Highest win rate · 最高赢面</p>
                          <p class="hero-title" style="font-size:1.1rem;">{pb_wr.name}</p>
                          <div class="pill-row">
                            <span class="pill pill-green">赢面 {pb_wr.win_rate_profit:.0f}%</span>
                            <span class="pill">{prem2}</span>
                            <span class="pill">赚 ${pb_wr.max_profit:.0f} · 亏 ${pb_wr.max_loss:.0f}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("最高赢面 · 两腿", expanded=False):
                        st.dataframe(
                            legs_to_frame(pb_wr), width="stretch", hide_index=True
                        )

            strategy_comparison = getattr(opt_rep, "strategy_comparison", None) or []
            if strategy_comparison:
                st.markdown("### 跟坊间常见做法比一比")
                st.caption(
                    "公平口径：所有数字都取自同一到期日、当前期权链、IV 与自然成交（卖=bid / 买=ask）。"
                    "POP 与 EV 是本工具模型估算，不是任何教学平台的历史保证胜率。"
                )
                st.dataframe(
                    pd.DataFrame(strategy_comparison),
                    width="stretch",
                    height=330,
                    hide_index=True,
                )
                st.info(
                    "看法：高 POP 往往以较低的最大利润/风险报酬作交换；"
                    "比较时先确保最大亏损符合 1R，再看流动性、OTM 距离、EV 与管理计划。"
                )

            ptable = getattr(opt_rep, "playbook_table", None) or []
            if ptable:
                with st.expander("策略排行榜（可滚）", expanded=False):
                    st.dataframe(
                        pd.DataFrame(ptable),
                        width="stretch",
                        height=300,
                        hide_index=True,
                    )

            if opt_rep.action_plan:
                with st.expander("步骤清单", expanded=False):
                    for line in opt_rep.action_plan:
                        st.markdown(f"- {line}")

            # payoff / daily sections need these
            pay_idea = best
            spot_px = opt_rep.spot

            st.markdown(
                """
                <div class="glass-card" style="margin-bottom:0.4rem;">
                  <p class="section-label">P&amp;L by price · 到什么价赚/蚀</p>
                  <p class="mini-note">一般不用行权。到期或平仓时，看股票价在「不赚不亏价」哪一边。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            payoff_choices = {"目前选择": best}
            for candidate in credit_candidates:
                if candidate is best:
                    continue
                side = "卖 OTM Put" if candidate.code == "bull_put" else "卖 OTM Call"
                payoff_choices[f"比较：{side}"] = candidate
            if hi is not None:
                payoff_choices["赢面最高"] = hi
            if hi_al is not None and hi_al is not hi:
                payoff_choices["跟方向一致·赢面高"] = hi_al
            payoff_pick = st.radio(
                "看哪一套价差",
                list(payoff_choices.keys()),
                horizontal=True,
                key="opt_payoff_pick",
            )
            pay_idea = payoff_choices[payoff_pick]
            spot_px = opt_rep.spot
            pay_credit = pay_idea.net_credit is not None

            zones = payoff_zones_summary(pay_idea, spot_px)
            for line in zones["lines"]:
                st.markdown(f"- {line}")

            z1, z2, z3 = st.columns(3)
            z1.metric("股价不变到期", f"${zones['spot_pnl']:.0f}/张")
            z2.metric("最多赚", f"${zones['max_profit']:.0f}/张")
            z3.metric("最多亏", f"${zones['max_loss']:.0f}/张")

            strike_prices = sorted({float(leg.strike) for leg in pay_idea.legs})
            key_prices = sorted({spot_px, *strike_prices, *(pay_idea.breakevens or [])})
            strike_payoffs = pd.DataFrame(
                [
                    {
                        "到期标的价": round(price, 2),
                        "位置": (
                            "现价"
                            if abs(price - spot_px) < 0.01
                            else "打和点"
                            if any(abs(price - be) < 0.01 for be in (pay_idea.breakevens or []))
                            else "履约价"
                        ),
                        "到期赚亏$/张": round(payoff_per_contract(pay_idea, price), 2),
                    }
                    for price in key_prices
                ]
            )
            st.markdown("**关键履约价：每张到期赚 / 蚀**")
            st.dataframe(strike_payoffs, width="stretch", hide_index=True)

            ladder = build_payoff_ladder(pay_idea, spot_px).rename(
                columns={
                    "标的价": "股票价",
                    "相对现价%": "比今天%",
                    "到期盈亏$/张": "到期赚亏$/张",
                    "结果": "赚或蚀",
                    "区间": "说明",
                    "标记": "备注",
                }
            )
            st.markdown("**价位表（可往下滚）**")
            st.dataframe(ladder, width="stretch", height=320, hide_index=True)

            be0 = zones.get("breakeven")
            if be0 is not None and pay_idea.code in ("bull_put", "bull_call"):
                st.success(f"记一句：股票 **高于 {be0:.2f} 賺**，**低于 {be0:.2f} 蝕**。")
            elif be0 is not None and pay_idea.code in ("bear_call", "bear_put"):
                st.success(f"记一句：股票 **低于 {be0:.2f} 賺**，**高于 {be0:.2f} 蝕**。")

            st.line_chart(
                ladder.rename(columns={"股票价": "股价", "到期赚亏$/张": "赚亏"})[
                    ["股价", "赚亏"]
                ].set_index("股价")
            )

            # ---- 卖出后第几天买回 ----
            st.markdown(
                """
                <div class="glass-card" style="margin-bottom:0.4rem;">
                  <p class="section-label">Day by day · 第几天买回</p>
                  <p class="mini-note">选持有天数（如 14 天），看每天若平仓大约赚还是蚀（估算）。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pay_credit:
                st.caption("今天卖出价差收了钱 → 下面问：第 N 天买回，赚还是蚀？")
            else:
                st.caption("今天买进价差付了钱 → 下面问：第 N 天卖出，赚还是蚀？")

            dte_cap = int(pay_idea.dte or opt_rep.dte or 30)
            dte_cap = max(dte_cap, 1)
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                hold_days = st.slider(
                    "想看到第几天（例如 14 = 两周后）",
                    min_value=1,
                    max_value=dte_cap,
                    value=min(14, dte_cap),
                )
            with dc2:
                path = st.selectbox(
                    "这段时间股票怎么走",
                    ["flat", "+1%", "-1%", "+2%", "-2%", "+3%", "-3%"],
                    format_func=lambda x: {
                        "flat": "股价差不多不变",
                        "+1%": "慢慢涨约 1%",
                        "-1%": "慢慢跌约 1%",
                        "+2%": "慢慢涨约 2%",
                        "-2%": "慢慢跌约 2%",
                        "+3%": "慢慢涨约 3%",
                        "-3%": "慢慢跌约 3%",
                    }.get(x, x),
                )
            with dc3:
                day_focus = st.slider(
                    "点选：看第几天详情",
                    min_value=0,
                    max_value=int(hold_days),
                    value=min(7, int(hold_days)),
                )

            daily = build_daily_mark_calendar(
                pay_idea,
                spot=spot_px,
                sigma=opt_rep.iv_atm,
                dte_total=dte_cap,
                hold_days=int(hold_days),
                spot_path=path,
            )
            st.dataframe(daily, width="stretch", height=380, hide_index=True)

            # 焦点日（兼容新旧列名）
            day_col = "第几天后" if "第几天后" in daily.columns else "第N日"
            focus_row = daily[daily[day_col] == day_focus]
            if not focus_row.empty:
                fr = focus_row.iloc[0]
                st.markdown(f"#### 若在第 **{day_focus}** 天平仓")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("日期", str(fr["日期"]))
                f2.metric(
                    "那天股票大约",
                    f"{fr['股票大约价']:.2f}" if "股票大约价" in fr.index else f"{fr.get('假设标的价', 0):.2f}",
                )
                if pay_credit:
                    buy_col = "这天买回要付$/股"
                    if buy_col in fr.index:
                        f3.metric("买回大约要付", f"${fr[buy_col]:.2f}/股")
                else:
                    sell_col = "这天卖出收回$/股"
                    if sell_col in fr.index:
                        f3.metric("卖出大约收回", f"${fr[sell_col]:.2f}/股")
                pnl_col = "若这天平仓赚亏$/张" if "若这天平仓赚亏$/张" in fr.index else "浮动盈亏$/张"
                res_col = "赚或蚀" if "赚或蚀" in fr.index else "结果"
                f4.metric("这一张大约", f"${fr[pnl_col]:.0f}", str(fr[res_col]))

                if pay_credit and day_focus > 0:
                    st.markdown(
                        f"→ 白话：**今天卖出价差**，**{day_focus} 天后买回**，"
                        f"估算结果：**{fr[res_col]}** 约 **${fr[pnl_col]:.0f}/张**。"
                    )
                elif day_focus > 0:
                    st.markdown(
                        f"→ 白话：**今天买进价差**，**{day_focus} 天后卖出**，"
                        f"估算结果：**{fr[res_col]}** 约 **${fr[pnl_col]:.0f}/张**。"
                    )

            st.caption("下图：过了第几天，平仓大约赚亏多少（$/张）")
            pnl_col = "若这天平仓赚亏$/张" if "若这天平仓赚亏$/张" in daily.columns else "浮动盈亏$/张"
            st.line_chart(daily.set_index(day_col)[[pnl_col]])

            st.info(
                "卖出价差：买回越便宜 → 越容易賺。"
                " 买进价差：卖出越贵 → 越容易賺。"
                " 数字是估算，真钱下单请看券商报价。"
            )

            # 其他候选
            with st.expander("还有哪些候选价差（可比较）", expanded=False):
                sort_mode = st.radio(
                    "怎么排",
                    ["赢面高的在前", "流动性好的在前", "综合分高的在前"],
                    horizontal=True,
                )
                show_n = st.slider("显示几条", 5, max(5, len(opt_rep.ideas)), min(12, len(opt_rep.ideas)))
                if "赢面" in sort_mode:
                    sort_by = "winrate"
                elif "流动" in sort_mode:
                    sort_by = "liquidity"
                else:
                    sort_by = "score"
                table = ideas_to_frame(opt_rep.ideas, sort_by=sort_by).head(show_n)
                st.dataframe(table, width="stretch", hide_index=True)

            with st.expander("这些「常见做法」从哪来（白话）", expanded=False):
                st.markdown(
                    """
            本页**不是抄某一家荐股**，而是把很多人教学里反复出现的规则写成筛子：

            | 常见叫法 | 白话规则 |
            |----------|----------|
            | 高胜率收钱价差 | 卖出价差；短腿离现价远一点；大约 3–6 周到期；赢面偏好高 |
            | 权利金约 1/3 宽 | 收到的钱大约是两腿间距的 1/4～1/3 |
            | 偏多收租 | 看多时卖 Put 价差，先收钱，再买回 |
            | 偏空收租 | 看空时卖 Call 价差，先收钱，再买回 |
            | 顺势买进 | 方向很强才买进价差（先付钱） |

            **最好** = 最像上述规则 + 方向对 + 报价合理。  
            **最高赢面** = 在还能算「像常见做法」的里面，赢面% 最高。

            | 你怎么想 | 今天 | 几天后 |
            |----------|------|--------|
            | 不会大跌 | 卖出看跌价差收钱 | 买回 |
            | 会大涨 | 买进看涨价差付钱 | 卖出 |
            | 不会大涨 | 卖出看涨价差收钱 | 买回 |
            | 会大跌 | 买进看跌价差付钱 | 卖出 |

            赢面是估算，**不是保证**。
            """
                )

            st.warning(
                "**实盘辅助，不是自动下单/投顾。** 下单以券商实时报价与风控为准。"
            )

            if not is_options_eligible(symbol):
                st.info(f"侧边栏是 `{symbol}`，本页在分析 `{opt_sym}`。")
