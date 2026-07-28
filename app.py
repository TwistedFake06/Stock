"""
股票分析助手 — Streamlit 桌面/浏览器应用
支持美股、港股、A股（Yahoo Finance 代码）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path (fixes odd launch cwd / IDE issues)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

try:
    from analysis import BiasReport, analyze_bias, bias_color, bias_emoji
    from chart_entry import entry_target_chart
    from charts import (
        bias_gauge,
        compare_chart,
        drawdown_chart,
        macd_chart,
        price_volume_chart,
        relative_strength_chart,
        rsi_chart,
        scorecard_radar,
        sr_chart,
    )
    from entry_targets import analyze_entry, analyze_targets
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
    from helper_analysis import analyze_helpers
    from indicators import enrich
    from options_payoff import (
        build_daily_mark_calendar,
        build_payoff_ladder,
        payoff_zones_summary,
    )
    from options_plain import plain_spread_steps
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
    except Exception:
        def methods_to_rows(idea):  # type: ignore
            return []

    from stock_service import (
        DEFAULT_WATCHLIST,
        INTERVAL_MAP,
        PERIOD_MAP,
        cache_bucket,
        cached_calendar,
        cached_info,
        compare_symbols,
        compute_returns,
        fetch_history,
        normalize_symbol,
    )
    from trade_plan import (
        analyze_events,
        build_trade_plan_card,
        calc_position,
        suggest_lot_size,
    )
except Exception as exc:
    # Friendly page when deps/modules fail (e.g. ran without venv / missing plotly)
    try:
        st.set_page_config(page_title="启动失败", page_icon="⚠️")
    except Exception:
        pass
    st.error("启动失败：模块导入错误")
    st.code(f"{type(exc).__name__}: {exc}")
    st.markdown(
        """
**推荐启动方式（请用项目虚拟环境）：**

1. 双击 `run.bat`  
或在终端执行：

```bat
cd /d C:\\Users\\Eddie\\Python\\Stock
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m streamlit run app.py
```

不要用未安装依赖的系统 Python 直接 `streamlit run`。
"""
    )
    st.stop()

WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"

st.set_page_config(
    page_title="股票分析助手",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",  # better for iPhone; open menu when needed
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "股票 / 期权价差分析 · 手机浏览器可用 · Streamlit Cloud",
    },
)

from ui_mobile import inject_mobile_css, metric_grid, plotly_chart as mobile_plotly

inject_mobile_css()


def render_bias_banner(report: BiasReport, compact: bool = False) -> None:
    """Render multi/空 conclusion card."""
    color = bias_color(report.bias)
    emoji = bias_emoji(report.bias)
    if compact:
        st.markdown(
            f"""
            <div class="bias-card">
              <p class="bias-title" style="color:{color};">{emoji} {report.bias}
                <span style="font-size:1rem;color:#90a4ae;font-weight:500;">
                  · 得分 {report.score:+.0f} · 置信度 {report.confidence}
                </span>
              </p>
              <p class="bias-sub">{report.summary.replace('**', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Stack vertically for phone; still fine on desktop
    st.markdown(
        f"""
        <div class="bias-card">
          <p class="bias-title" style="color:{color};">{emoji} 综合判断：{report.bias}</p>
          <p class="bias-sub">{report.summary.replace('**', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mobile_plotly(bias_gauge(report.score, report.bias))
    st.caption(
        f"多头 **{report.bull_count}** · 空头 **{report.bear_count}** · "
        f"中性 **{report.neutral_count}** · 置信度 **{report.confidence}**"
    )
    snap = report.snapshot or {}
    metric_grid(
        [
            ("收盘", f"{snap['close']:.2f}" if snap.get("close") is not None else "—"),
            ("RSI", f"{snap['rsi']:.1f}" if snap.get("rsi") is not None else "—"),
            (
                "MACD柱",
                f"{snap['macd_hist']:.4f}" if snap.get("macd_hist") is not None else "—",
            ),
            (
                "布林位置",
                f"{snap.get('bb_position'):.0%}" if snap.get("bb_position") is not None else "—",
            ),
        ],
        per_row=2,
    )
    with st.expander("分项信号", expanded=False):
        for sig in report.signals:
            if sig.bias == "看多":
                cls, mark = "sig-bull", "▲ 看多"
            elif sig.bias == "看空":
                cls, mark = "sig-bear", "▼ 看空"
            else:
                cls, mark = "sig-flat", "● 中性"
            st.markdown(
                f"<span class='{cls}'>{mark}</span> · **{sig.name}** — {sig.detail}",
                unsafe_allow_html=True,
            )

    st.info(
        "仅供学习参考，**不构成投资建议**。"
    )


def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        try:
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [str(x) for x in data]
        except Exception:
            pass
    return list(DEFAULT_WATCHLIST)


def save_watchlist(items: list[str]) -> None:
    WATCHLIST_FILE.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fmt_number(val, prefix: str = "", suffix: str = "", digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        n = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(n) >= 1e12:
        return f"{prefix}{n / 1e12:.{digits}f}T{suffix}"
    if abs(n) >= 1e8:
        return f"{prefix}{n / 1e8:.{digits}f}亿{suffix}"
    if abs(n) >= 1e6:
        return f"{prefix}{n / 1e6:.{digits}f}M{suffix}"
    return f"{prefix}{n:,.{digits}f}{suffix}"


def fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{float(val):+.2f}%"


def get_price_fields(info: dict) -> dict:
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("last_price")
        or info.get("previousClose")
        or info.get("previous_close")
    )
    prev = info.get("previousClose") or info.get("previous_close") or info.get("regularMarketPreviousClose")
    change = None
    change_pct = None
    if price is not None and prev is not None and prev != 0:
        change = float(price) - float(prev)
        change_pct = change / float(prev) * 100
    return {
        "price": price,
        "prev": prev,
        "change": change,
        "change_pct": change_pct,
        "open": info.get("open") or info.get("regularMarketOpen") or info.get("open"),
        "high": info.get("day_high") or info.get("dayHigh") or info.get("regularMarketDayHigh"),
        "low": info.get("day_low") or info.get("dayLow") or info.get("regularMarketDayLow"),
        "year_high": info.get("year_high") or info.get("fiftyTwoWeekHigh"),
        "year_low": info.get("year_low") or info.get("fiftyTwoWeekLow"),
        "mcap": info.get("market_cap") or info.get("marketCap"),
        "pe": info.get("trailingPE") or info.get("forwardPE"),
        "currency": info.get("currency") or "",
        "name": info.get("shortName") or info.get("longName") or info.get("_symbol", ""),
        "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
    }


# ---- session state ----
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if "symbol" not in st.session_state:
    st.session_state.symbol = st.session_state.watchlist[0]


# ---- sidebar ----
with st.sidebar:
    st.markdown("### 📈 股票分析")
    st.caption("手机点左上角 « 打开菜单 · Yahoo 数据")

    # selectbox is easier than long radio on iPhone
    page = st.selectbox(
        "功能页面",
        [
            "行情看板",
            "多空分析",
            "入场与目标价",
            "期权价差",
            "综合分析",
            "技术分析",
            "多股对比",
            "自选股",
        ],
    )

    st.divider()
    symbol_input = st.text_input(
        "股票代码",
        value=st.session_state.symbol,
        placeholder="AAPL / 0700.HK / 600519",
        help="A股 6 位代码；港股 .HK；美股代码",
    )
    if symbol_input:
        st.session_state.symbol = normalize_symbol(symbol_input)

    period_label = st.selectbox("时间范围", list(PERIOD_MAP.keys()), index=3)
    interval_label = st.selectbox("K线周期", list(INTERVAL_MAP.keys()), index=0)
    period = PERIOD_MAP[period_label]
    interval = INTERVAL_MAP[interval_label]

    st.divider()
    st.markdown("**快速选择**")
    cols = st.columns(2)
    for i, s in enumerate(st.session_state.watchlist[:10]):
        if cols[i % 2].button(s, key=f"quick_{s}", use_container_width=True):
            st.session_state.symbol = s
            st.rerun()

    st.divider()
    st.caption("数据来源: Yahoo Finance · 仅供学习参考，不构成投资建议")


symbol = st.session_state.symbol


# ===================== 行情看板 =====================
if page == "行情看板":
    st.header(f"行情看板 · `{symbol}`")

    with st.spinner("加载行情..."):
        info = cached_info(symbol, cache_bucket(5))
        hist = fetch_history(symbol, period=period, interval=interval)

    fields = get_price_fields(info)
    st.subheader(fields["name"] or symbol)
    meta_bits = [b for b in [fields["exchange"], fields["sector"], fields["industry"]] if b]
    if meta_bits:
        st.caption(" · ".join(meta_bits))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cur = fields["currency"]
    c1.metric(
        "最新价",
        fmt_number(fields["price"], suffix=f" {cur}" if cur else ""),
        fmt_pct(fields["change_pct"]),
    )
    c2.metric("开盘", fmt_number(fields["open"]))
    c3.metric("最高", fmt_number(fields["high"]))
    c4.metric("最低", fmt_number(fields["low"]))
    c5.metric("52周高", fmt_number(fields["year_high"]))
    c6.metric("市值", fmt_number(fields["mcap"]))

    m1, m2, m3 = st.columns(3)
    m1.metric("前收", fmt_number(fields["prev"]))
    m2.metric("市盈率 PE", fmt_number(fields["pe"]))
    rets = compute_returns(hist) if not hist.empty else {}
    m3.metric(
        f"区间涨跌 ({period_label})",
        fmt_pct(rets.get("total_return_pct")),
        help="所选时间范围内首末收盘价涨跌幅",
    )

    if hist.empty:
        st.warning(
            f"未能获取 `{symbol}` 的历史数据。请检查代码是否正确。"
            "\n\n示例：`AAPL`、`0700.HK`、`600519`（茅台）、`000001`（平安银行）"
        )
    else:
        df = enrich(hist)
        report = analyze_bias(df)
        render_bias_banner(report, compact=True)

        fig = price_volume_chart(df, title=f"{symbol} · {period_label} · {interval_label}")
        mobile_plotly(fig, use_container_width=True)

        with st.expander("原始数据", expanded=False):
            show = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(show.iloc[::-1], use_container_width=True, hide_index=True)

    # Add to watchlist
    if st.button("⭐ 加入自选"):
        if symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(symbol)
            save_watchlist(st.session_state.watchlist)
            st.success(f"已添加 {symbol}")
        else:
            st.info(f"{symbol} 已在自选列表中")


# ===================== 多空分析 =====================
elif page == "多空分析":
    st.header(f"多空分析 · `{symbol}`")
    st.caption(
        f"基于当前所选周期：{period_label} · {interval_label} · "
        "综合均线 / MACD / RSI / 布林带 / 动量 / 量能"
    )

    with st.spinner("分析多空..."):
        hist = fetch_history(symbol, period=period, interval=interval)

    if hist.empty:
        st.warning(f"未能获取 `{symbol}` 数据，无法分析。")
    else:
        df = enrich(hist)
        report = analyze_bias(df)
        render_bias_banner(report, compact=False)

        st.subheader("价格与关键指标")
        fig = price_volume_chart(
            df,
            title=f"{symbol} · 多空参考图",
            show_sma=True,
            show_bb=True,
            show_volume=True,
        )
        mobile_plotly(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            mobile_plotly(rsi_chart(df), use_container_width=True)
        with col_b:
            mobile_plotly(macd_chart(df), use_container_width=True)

        # Signal table
        st.subheader("信号明细表")
        rows = [
            {
                "指标": s.name,
                "方向": s.bias,
                "权重分": round(s.score, 2),
                "依据": s.detail,
            }
            for s in report.signals
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        rets = compute_returns(df)
        m1, m2, m3 = st.columns(3)
        m1.metric("区间收益率", fmt_pct(rets.get("total_return_pct")))
        m2.metric("年化波动率(估)", fmt_pct(rets.get("volatility_pct")))
        m3.metric("多空得分", f"{report.score:+.1f}", report.bias)


# ===================== 综合分析 =====================
elif page == "综合分析":
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
                  <p class="bias-sub">{card.summary.replace('**', '')}</p>
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


# ===================== 入场与目标价 =====================
elif page == "入场与目标价":
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
              <p class="bias-sub">{entry.summary.replace('**', '')}</p>
              <p class="bias-sub">{helpers.one_liner}</p>
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
            use_container_width=True,
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
                st.dataframe(pd.DataFrame(tp_rows), use_container_width=True, hide_index=True)

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
                st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

            st.markdown("**附近关键价位（按距现价排序）**")
            if helpers.key_levels:
                st.dataframe(
                    pd.DataFrame(helpers.key_levels),
                    use_container_width=True,
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
                st.dataframe(ev_df, use_container_width=True, hide_index=True)
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
                    use_container_width=True,
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
                    use_container_width=True,
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


# ===================== 期权价差（白话版） =====================
elif page == "期权价差":
    st.header("期权价差 · 卖出 / 买回")
    st.caption(
        "用白话看：今天**卖出一组价差先收钱**（或买进先付钱），"
        "**几天后买回（或卖出）平仓**，看赚还是蚀。只做 QQQ / VOO / SPY 等大盘 ETF。"
    )

    opt_default = options_symbol(symbol)
    if not is_options_eligible(opt_default):
        opt_default = "QQQ"

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        etf_list = sorted(INDEX_ETF_WHITELIST.keys())
        idx = etf_list.index(opt_default) if opt_default in etf_list else 0
        opt_sym = st.selectbox(
            "选哪只 ETF",
            etf_list,
            index=idx,
            format_func=lambda s: f"{s} — {INDEX_ETF_WHITELIST[s]}",
        )
    with c2:
        target_dte = st.select_slider(
            "打算拿多久（大约几天到期）",
            options=[14, 21, 30, 45, 60],
            value=30,
        )
    with c3:
        width_mode = st.selectbox("两腿价差宽度", ["自动", "1", "2", "5", "10"], index=0)

    with st.spinner("正在判断方向、找适合的价差..."):
        hist_opt = fetch_history(opt_sym, period="6mo", interval="1d")
        width_val = None if width_mode == "自动" else float(width_mode)
        opt_rep = analyze_options_spreads(
            opt_sym,
            target_dte=int(target_dte),
            width=width_val,
            hist_df=hist_opt if not hist_opt.empty else None,
        )

    if not opt_rep.eligible:
        st.error(opt_rep.message)
        st.write(", ".join(f"`{k}`" for k in sorted(INDEX_ETF_WHITELIST)))
    else:
        d = opt_rep.direction
        dir_color = {
            "看多": "#ef5350",
            "看空": "#26a69a",
            "中性": "#78909c",
        }.get(d.direction if d else "中性", "#78909c")

        st.markdown(
            f"""
            <div class="bias-card">
              <p class="bias-title" style="color:{dir_color};">
                现在更像：{(d.direction if d else '—')}
                <span style="font-size:1.05rem;color:#90a4ae;font-weight:500;">
                  · {(d.strength if d else '—')} · 分数 {(f'{d.score:+.0f}' if d else '—')}
                </span>
              </p>
              <p class="bias-sub">{(d.style_hint if d else '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if d and d.reasons:
            with st.expander("为什么这样判断（可展开）", expanded=False):
                for r in d.reasons:
                    st.markdown(f"- {r}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("现在股价", f"{opt_rep.spot:.2f}")
        m2.metric("期权到期日", opt_rep.selected_expiry or "—")
        m3.metric("还剩几天", f"{opt_rep.dte}" if opt_rep.dte is not None else "—")
        m4.metric(
            "波动大约",
            f"{opt_rep.iv_atm * 100:.0f}%" if opt_rep.iv_atm is not None else "—",
        )

        if not opt_rep.ideas or not opt_rep.best:
            st.warning("暂时算不出可用价差（网络或盘后报价问题）。")
        else:
            hi = getattr(opt_rep, "best_winrate", None)
            hi_al = getattr(opt_rep, "best_winrate_aligned", None)
            pb = getattr(opt_rep, "best_playbook", None) or opt_rep.best
            pb_wr = getattr(opt_rep, "best_playbook_wr", None) or hi_al or hi
            best = pb  # 默认展示实战推荐
            is_credit = best.net_credit is not None
            do_word = "卖出价差（先收钱）" if is_credit else "买进价差（先付钱）"
            close_word = "几天后买回平仓" if is_credit else "几天后卖出平仓"

            st.success(
                f"建议：先 **{do_word}**，再 **{close_word}**。"
                f"方向：{d.direction if d else '—'}。"
            )

            # ---- 参考常见策略 ----
            st.subheader("① 按常见做法推荐")
            style = getattr(best, "playbook_style", "") or "实战扫描"
            plain = getattr(best, "playbook_plain", "") or ""
            src = getattr(best, "playbook_source", "") or ""
            fit = getattr(best, "playbook_fit", None)
            st.markdown(f"**像哪种常见做法：** {style}")
            if plain:
                st.markdown(f"**白话：** {plain}")
            if src:
                st.caption(f"规则参考：{src}（教学归纳，不是荐股）")
            if fit is not None:
                st.caption(f"和该做法的贴合程度大约 {fit:.0f}/100")

            for line in plain_spread_steps(best):
                st.markdown(line)

            wr_p = getattr(best, "win_rate_profit", None) or getattr(best, "pop_est", None)
            wr_m = getattr(best, "win_rate_max", None)

            b1, b2, b3, b4, b5, b6 = st.columns(6)
            if is_credit:
                b1.metric("今天卖出大约收", f"${best.net_credit:.2f}/股")
            else:
                b1.metric("今天买进大约付", f"${best.net_debit:.2f}/股")
            b2.metric("赢面大约", f"{wr_p:.0f}%" if wr_p is not None else "—")
            liq_lab = getattr(best, "liquidity_label", "") or "—"
            liq_sc = getattr(best, "liquidity_score", None)
            b3.metric(
                "流动性",
                liq_lab,
                f"{liq_sc:.0f}分" if liq_sc is not None else None,
            )
            b4.metric("最多能赚", f"${best.max_profit:.0f}/张")
            b5.metric("最多会亏", f"${best.max_loss:.0f}/张")
            b6.metric("不赚不亏价", f"{best.breakevens[0]:.2f}" if best.breakevens else "—")

            if getattr(best, "liquidity_detail", ""):
                st.caption(f"流动性：{best.liquidity_detail}")

            # 热门数字一览
            c_w = getattr(best, "metric_credit_width", None)
            roc = getattr(best, "metric_roc", None)
            half_p = getattr(best, "metric_half_profit", None)
            mc = getattr(best, "method_composite", None)
            x1, x2, x3, x4 = st.columns(4)
            x1.metric(
                "权利金/宽度",
                f"{c_w:.0f}%" if c_w is not None else "—",
                help="很多人用：收到的钱 ÷ 两腿间距 ≈ 25%–33%",
            )
            x2.metric(
                "赚亏比 ROC",
                f"{roc:.2f}" if roc is not None else "—",
                help="最多赚 ÷ 最多亏",
            )
            x3.metric(
                "一半利润可走",
                f"${half_p:.0f}/张" if half_p is not None else "—",
                help="常见操作：赚到最大利润约 50% 就提前买回",
            )
            x4.metric(
                "热门规则总分",
                f"{mc:.0f}" if mc is not None else "—",
                help="多条常见规则加权平均",
            )

            st.caption(f"具体组合：{best.name}")
            with st.expander("买卖明细 + 每腿流动性", expanded=False):
                st.dataframe(legs_to_frame(best), use_container_width=True, hide_index=True)
                st.caption("未平仓/成交量越大、买卖差越小 → 越好成交。")

            mrows = methods_to_rows(best)
            if mrows:
                with st.expander("用了哪些真实/热门规则打分（可看）", expanded=True):
                    st.caption(
                        "报价来自 Yahoo 真实期权链；规则来自常见教学归纳"
                        "（高概率价差、1/3 宽度、30–45 天、50% 止盈等），不是保证赚钱。"
                    )
                    st.dataframe(
                        pd.DataFrame(mrows),
                        use_container_width=True,
                        hide_index=True,
                    )

            ops = getattr(best, "ops_plan", None)
            if ops is not None:
                with st.expander("热门操作方式：怎么进、怎么管、怎么出", expanded=True):
                    st.markdown(f"**进场：** {ops.entry}")
                    st.markdown(f"**持有中：** {ops.manage}")
                    st.markdown("**出场常见做法：**")
                    for e in ops.exit_rules:
                        st.markdown(f"- {e}")

            reasons = getattr(best, "playbook_reasons", None) or []
            if reasons:
                with st.expander("为什么说像这种做法", expanded=False):
                    for r in reasons:
                        st.markdown(f"- {r}")

            # 最高赢面（实战过滤）
            if pb_wr is not None and getattr(pb_wr, "win_rate_profit", None) is not None:
                same = pb_wr is best
                st.subheader("② 最高赢面（在常见规则里挑）")
                if same:
                    st.info(
                        f"刚好就是上面这一套：赢面大约 **{pb_wr.win_rate_profit:.0f}%**"
                        + (
                            f"，大赢约 {pb_wr.win_rate_max:.0f}%"
                            if pb_wr.win_rate_max is not None
                            else ""
                        )
                    )
                else:
                    st.markdown(
                        f"**{pb_wr.name}** · 赢面约 **{pb_wr.win_rate_profit:.0f}%**"
                        + (
                            f" · 像「{getattr(pb_wr, 'playbook_style', '')}」"
                            if getattr(pb_wr, "playbook_style", "")
                            else ""
                        )
                    )
                    if pb_wr.net_credit is not None:
                        st.caption(
                            f"今天卖出约收 ${pb_wr.net_credit:.2f}/股，"
                            f"最多赚 ${pb_wr.max_profit:.0f}/张，最多亏 ${pb_wr.max_loss:.0f}/张"
                        )
                    else:
                        st.caption(
                            f"今天买进约付 ${pb_wr.net_debit:.2f}/股，"
                            f"最多赚 ${pb_wr.max_profit:.0f}/张，最多亏 ${pb_wr.max_loss:.0f}/张"
                        )
                    with st.expander("最高赢面 · 两腿明细", expanded=False):
                        st.dataframe(
                            legs_to_frame(pb_wr), use_container_width=True, hide_index=True
                        )
                    for line in plain_spread_steps(pb_wr):
                        st.markdown(line)

            # 实战对照表
            ptable = getattr(opt_rep, "playbook_table", None) or []
            if ptable:
                with st.expander("对照：常见策略排行（含流动性，可滚）", expanded=True):
                    st.caption(
                        "按「常见做法 + 赢面 + 流动性 + 报价」打分。"
                        "流动性看：未平仓、成交量、买卖价差。"
                    )
                    st.dataframe(
                        pd.DataFrame(ptable),
                        use_container_width=True,
                        height=300,
                        hide_index=True,
                    )

            if opt_rep.action_plan:
                with st.expander("步骤清单", expanded=False):
                    for line in opt_rep.action_plan:
                        st.markdown(f"- {line}")

            # ---- 到期到什么价赚/蚀 ----
            st.subheader("③ 拿到到期：股票到什么价是赚/蚀")
            st.caption("一般**不用行权**。到期或平仓时，看股票价在「不赚不亏价」哪一边。")

            payoff_choices = {"系统建议": best}
            if hi is not None:
                payoff_choices["赢面最高"] = hi
            if hi_al is not None and hi_al is not hi:
                payoff_choices["跟方向一致·赢面高"] = hi_al
            payoff_pick = st.radio(
                "看哪一套价差",
                list(payoff_choices.keys()),
                horizontal=True,
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
            st.dataframe(ladder, use_container_width=True, height=320, hide_index=True)

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
            st.subheader("④ 卖出后，第几天买回？赚亏大概多少")
            if pay_credit:
                st.caption(
                    "假设你**今天卖出价差收了钱**，下面每一天问："
                    "若**那天买回平仓**，大概赚还是蚀？（估算，不是保证）"
                )
            else:
                st.caption(
                    "假设你**今天买进价差付了钱**，下面每一天问："
                    "若**那天卖出平仓**，大概赚还是蚀？（估算，不是保证）"
                )

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
            st.dataframe(daily, use_container_width=True, height=380, hide_index=True)

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
                st.dataframe(table, use_container_width=True, hide_index=True)

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

            st.warning("仅供学习，**不是投资建议**。下单以券商实时报价为准。")

    if not is_options_eligible(symbol):
        st.info(f"侧边栏是 `{symbol}`，本页在分析 `{opt_sym}`。")


# ===================== 技术分析 =====================
elif page == "技术分析":
    st.header(f"技术分析 · `{symbol}`")

    show_sma = st.checkbox("均线 SMA 5/20/60", value=True)
    show_bb = st.checkbox("布林带 Bollinger", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)

    with st.spinner("计算指标..."):
        hist = fetch_history(symbol, period=period, interval=interval)

    if hist.empty:
        st.warning(f"未能获取 `{symbol}` 数据。")
    else:
        df = enrich(hist)
        report = analyze_bias(df)
        render_bias_banner(report, compact=True)

        fig = price_volume_chart(
            df,
            title=f"{symbol} 技术图表",
            show_sma=show_sma,
            show_bb=show_bb,
            show_volume=True,
        )
        mobile_plotly(fig, use_container_width=True)

        if show_rsi:
            mobile_plotly(rsi_chart(df), use_container_width=True)
            last_rsi = df["RSI"].dropna()
            if not last_rsi.empty:
                v = float(last_rsi.iloc[-1])
                if v >= 70:
                    st.warning(f"当前 RSI ≈ **{v:.1f}**，处于超买区域，注意回调风险。")
                elif v <= 30:
                    st.info(f"当前 RSI ≈ **{v:.1f}**，处于超卖区域，可能存在反弹机会。")
                else:
                    st.caption(f"当前 RSI ≈ **{v:.1f}**（中性区间 30–70）")

        if show_macd:
            mobile_plotly(macd_chart(df), use_container_width=True)

        rets = compute_returns(df)
        a, b = st.columns(2)
        a.metric("区间收益率", fmt_pct(rets.get("total_return_pct")))
        b.metric("年化波动率(估)", fmt_pct(rets.get("volatility_pct")))


# ===================== 多股对比 =====================
elif page == "多股对比":
    st.header("多股对比")
    st.caption("价格归一化到起点 = 100，便于比较相对强弱。")

    default_compare = ",".join(st.session_state.watchlist[:4])
    raw = st.text_input(
        "对比代码（逗号分隔）",
        value=default_compare,
        placeholder="AAPL,MSFT,0700.HK,600519",
    )
    symbols = [normalize_symbol(s) for s in raw.split(",") if s.strip()]

    if not symbols:
        st.info("请输入至少一个股票代码。")
    else:
        with st.spinner("拉取对比数据..."):
            cmp_df = compare_symbols(symbols, period=period, interval=interval)

        if cmp_df.empty:
            st.warning("没有可用的对比数据，请检查代码。")
        else:
            mobile_plotly(
                compare_chart(cmp_df, title=f"相对走势 · {period_label}"),
                use_container_width=True,
            )

            # End-point performance table
            rows = []
            for col in cmp_df.columns:
                series = cmp_df[col].dropna()
                if series.empty:
                    continue
                ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
                rows.append({"代码": col, "区间涨跌%": round(ret, 2), "最新指数": round(series.iloc[-1], 2)})
            if rows:
                table = pd.DataFrame(rows).sort_values("区间涨跌%", ascending=False)
                st.dataframe(table, use_container_width=True, hide_index=True)


# ===================== 自选股 =====================
elif page == "自选股":
    st.header("自选股管理")

    add_col, _ = st.columns([2, 3])
    with add_col:
        new_sym = st.text_input("添加代码", placeholder="例如 NVDA 或 300750")
        if st.button("添加", type="primary") and new_sym.strip():
            ns = normalize_symbol(new_sym)
            if ns not in st.session_state.watchlist:
                st.session_state.watchlist.append(ns)
                save_watchlist(st.session_state.watchlist)
                st.success(f"已添加 {ns}")
                st.rerun()
            else:
                st.info("已存在")

    st.divider()

    if not st.session_state.watchlist:
        st.info("自选列表为空，请添加股票。")
    else:
        rows = []
        progress = st.progress(0, text="刷新自选行情与多空...")
        for i, s in enumerate(st.session_state.watchlist):
            info = cached_info(s, cache_bucket(5))
            f = get_price_fields(info)
            hist_s = fetch_history(s, period=period, interval=interval)
            if hist_s.empty:
                bias_label, bias_score = "—", None
            else:
                rep = analyze_bias(enrich(hist_s))
                bias_label = f"{bias_emoji(rep.bias)} {rep.bias}"
                bias_score = rep.score
            rows.append(
                {
                    "代码": s,
                    "名称": f["name"] or s,
                    "最新价": f["price"],
                    "涨跌%": f["change_pct"],
                    "多空": bias_label,
                    "得分": bias_score,
                    "市值": f["mcap"],
                    "PE": f["pe"],
                    "交易所": f["exchange"],
                }
            )
            progress.progress(
                (i + 1) / len(st.session_state.watchlist), text=f"分析 {s}..."
            )
        progress.empty()

        table = pd.DataFrame(rows)
        display = table.copy()
        if "涨跌%" in display.columns:
            display["涨跌%"] = display["涨跌%"].apply(
                lambda x: f"{float(x):+.2f}%" if x is not None and pd.notna(x) else "—"
            )
        if "得分" in display.columns:
            display["得分"] = display["得分"].apply(
                lambda x: f"{float(x):+.0f}" if x is not None and pd.notna(x) else "—"
            )
        for col in ("最新价", "市值", "PE"):
            if col in display.columns:
                display[col] = display[col].apply(lambda x: fmt_number(x))

        # Sort by score descending when available
        if "得分" in table.columns and table["得分"].notna().any():
            order = table["得分"].fillna(0).sort_values(ascending=False).index
            display = display.loc[order]

        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("自选列表按多空得分从高到低排序（看多在前）。")

        st.subheader("操作")
        for s in list(st.session_state.watchlist):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"`{s}`")
            if c2.button("查看", key=f"view_{s}"):
                st.session_state.symbol = s
                st.info(f"已选择 {s}，请到「行情看板」或「技术分析」查看。")
            if c3.button("删除", key=f"del_{s}"):
                st.session_state.watchlist = [x for x in st.session_state.watchlist if x != s]
                save_watchlist(st.session_state.watchlist)
                st.rerun()

        if st.button("恢复默认自选"):
            st.session_state.watchlist = list(DEFAULT_WATCHLIST)
            save_watchlist(st.session_state.watchlist)
            st.rerun()
