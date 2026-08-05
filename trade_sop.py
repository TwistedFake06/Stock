"""
Investment SOP engine — practical trading checklist when a symbol is selected.

Produces: enter or not, entry price zone, stop, targets, win-rate estimate,
stability score, and ordered actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from analysis import analyze_bias
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
from free_data import (
    analyze_liquidity,
    extract_quality_extras,
    free_data_status,
    get_market_regime,
    get_news_pulse,
    merge_av_overview_into_info,
    multi_horizon_rs,
)
from indicators import enrich
from stock_service import (
    cached_calendar,
    cached_info,
    cache_bucket,
    compute_returns,
    fetch_history,
    normalize_symbol,
)
from trade_plan import analyze_events, calc_position, suggest_lot_size


@dataclass
class TradeSOP:
    symbol: str
    name: str
    last_price: float | None

    # Core SOP fields user asked for
    enter_ok: str  # 适合入场 | 谨慎试仓 | 观望 | 回避
    enter_score: float  # 0-100
    entry_low: float | None
    entry_high: float | None
    entry_plan: float | None  # recommended limit / mid of zone
    stop_loss: float | None
    target_t1: float | None
    target_t2: float | None
    win_rate_pct: float | None  # estimated path win rate
    win_rate_label: str  # 高 | 中 | 低
    stability_score: float  # 0-100
    stability_label: str  # 高 | 中 | 低

    side: str  # 做多 | 观望 | 偏空
    risk_per_share: float | None
    rr_t1: float | None
    position_shares: int
    position_note: str

    # What to do now
    actions_now: list[str] = field(default_factory=list)
    actions_wait: list[str] = field(default_factory=list)
    invalidation: str = ""
    checklist: list[dict[str, str]] = field(default_factory=list)

    # Context
    bias: str = "—"
    bias_score: float = 0.0
    opportunity: str = "—"
    risk_level: str = "—"
    max_dd_pct: float | None = None
    ann_vol_pct: float | None = None
    scorecard_stance: str = "—"
    scorecard_total: float | None = None
    summary: str = ""
    period: str = "1y"
    notes: list[str] = field(default_factory=list)

    # Real-invest extras
    expectancy_r: float | None = None  # expected R per trade from path WR + R:R
    regime_label: str = "—"
    regime_score: float | None = None
    regime_summary: str = ""
    regime_bullets: list[str] = field(default_factory=list)
    liquidity_label: str = "—"
    liquidity_score: float | None = None
    multi_rs_summary: str = ""
    quality_notes: list[str] = field(default_factory=list)
    news_summary: str = ""
    data_sources: list[str] = field(default_factory=list)
    scorecard_bullets: list[str] = field(default_factory=list)


def _path_win_rate(
    close: pd.Series,
    risk_per_share: float,
    reward_per_share: float,
    lookback: int = 80,
    horizon: int = 15,
) -> float | None:
    """
    Historical path win rate: from each past bar, long at close;
    win if +reward hit before -risk within horizon bars.
    """
    if close is None or len(close) < lookback + horizon + 5:
        return None
    if risk_per_share <= 0 or reward_per_share <= 0:
        return None

    c = close.astype(float).values
    wins = 0
    total = 0
    start = max(20, len(c) - lookback - horizon)
    end = len(c) - horizon
    for i in range(start, end):
        entry = c[i]
        stop = entry - risk_per_share
        target = entry + reward_per_share
        path = c[i + 1 : i + 1 + horizon]
        hit_t = hit_s = False
        for px in path:
            if px <= stop:
                hit_s = True
                break
            if px >= target:
                hit_t = True
                break
        if not hit_t and not hit_s:
            continue  # no resolution — skip
        total += 1
        if hit_t:
            wins += 1
    if total < 12:
        return None
    return 100.0 * wins / total


def _stability_score(risk, trend, bias_score: float) -> tuple[float, str]:
    """0-100: higher = calmer / more stable trend quality."""
    score = 50.0
    vol = getattr(risk, "ann_vol_pct", None)
    dd = getattr(risk, "max_drawdown_pct", None)
    sharpe = getattr(risk, "sharpe", None)
    wr = getattr(risk, "win_rate_pct", None)

    if vol is not None:
        # 12% vol ~ +20, 40% vol ~ -15
        score += max(-20.0, min(20.0, (22.0 - float(vol)) * 0.9))
    if dd is not None:
        # dd is negative; -10% good, -40% bad
        score += max(-25.0, min(15.0, (abs(float(dd)) * -0.7) + 12))
    if sharpe is not None:
        score += max(-15.0, min(20.0, float(sharpe) * 8.0))
    if wr is not None:
        score += max(-10.0, min(10.0, (float(wr) - 50.0) * 0.4))

    # choppy trend reduces stability of signal
    st = getattr(trend, "strength_label", "") or ""
    if "强" in st:
        score += 6
    elif "弱" in st or "震荡" in st:
        score -= 6

    # extreme bias can mean unstable chase
    if abs(bias_score) >= 55:
        score -= 4

    score = float(max(0.0, min(100.0, score)))
    if score >= 68:
        label = "高"
    elif score >= 45:
        label = "中"
    else:
        label = "低"
    return round(score, 1), label


def _expectancy_r(win_rate_pct: float | None, rr: float | None) -> float | None:
    """E[R] = p*R - (1-p)*1  assuming 1R risk, R:R = reward/risk."""
    if win_rate_pct is None or rr is None or rr <= 0:
        return None
    p = float(win_rate_pct) / 100.0
    return round(p * float(rr) - (1.0 - p) * 1.0, 3)


def _enter_decision(
    entry_opp: str,
    entry_score: float,
    bias_label: str,
    bias_score: float,
    stability: float,
    win_rate: float | None,
    rr: float | None,
    risk_level: str,
    *,
    regime_score: float | None = None,
    liquidity_score: float | None = None,
    expectancy_r: float | None = None,
    earnings_soon: bool = False,
    chase_high: bool = False,
    multi_rs_score: float | None = None,
) -> tuple[str, float, str]:
    """Map models → 适合入场 / 谨慎试仓 / 观望 / 回避 + score."""
    s = 38.0 + entry_score * 0.32
    if entry_opp in ("较佳入场",):
        s += 16
    elif entry_opp in ("可关注",):
        s += 8
    elif entry_opp in ("观望",):
        s -= 5
    elif entry_opp in ("不宜追高",):
        s -= 15
    elif entry_opp in ("偏空回避",):
        s -= 25

    if "强烈看多" in bias_label or bias_label == "看多":
        s += 10 if bias_score > 0 else 0
    if "看空" in bias_label:
        s -= 12
    if "强烈看空" in bias_label:
        s -= 20

    s += (stability - 50) * 0.12
    if win_rate is not None:
        s += (win_rate - 50) * 0.18
    if rr is not None:
        if rr >= 2.0:
            s += 8
        elif rr >= 1.2:
            s += 3
        elif rr < 0.8:
            s -= 10

    if risk_level in ("高", "极高"):
        s -= 8

    # Market regime (VIX / SPY structure) — real gate for new risk
    if regime_score is not None:
        s += (float(regime_score) - 50.0) * 0.22
        if regime_score < 35:
            s -= 8
    if liquidity_score is not None:
        s += (float(liquidity_score) - 50.0) * 0.10
        if liquidity_score < 35:
            s -= 10
    if multi_rs_score is not None:
        s += (float(multi_rs_score) - 50.0) * 0.08
    if expectancy_r is not None:
        if expectancy_r >= 0.35:
            s += 8
        elif expectancy_r >= 0.10:
            s += 3
        elif expectancy_r < 0:
            s -= 12
    if earnings_soon:
        s -= 14
    if chase_high:
        s -= 8

    s = float(max(0.0, min(100.0, s)))

    # Hard gates for long SOP
    if entry_opp in ("偏空回避",) or ("强烈看空" in bias_label and entry_score < 50):
        return "回避", min(s, 35.0), "偏空"
    if liquidity_score is not None and liquidity_score < 28:
        return "观望", min(s, 48.0), "观望"
    if earnings_soon:
        # Within earnings window: never full "适合入场" (gap risk)
        if s >= 55:
            return "谨慎试仓", min(s, 68.0), "做多"
        if s >= 40:
            return "观望", min(s, 52.0), "观望"
        return "回避", min(s, 40.0), "观望"
    if regime_score is not None and regime_score < 32:
        if s >= 70 and entry_opp in ("较佳入场",):
            return "谨慎试仓", min(s, 65.0), "做多"
        if s >= 50:
            return "观望", min(s, 55.0), "观望"
        return "回避", min(s, 40.0), "观望" if bias_score >= -15 else "偏空"

    if (
        s >= 74
        and entry_opp in ("较佳入场", "可关注")
        and bias_score >= -10
        and (expectancy_r is None or expectancy_r >= 0.05)
        and (regime_score is None or regime_score >= 42)
    ):
        return "适合入场", s, "做多"
    if s >= 55 and entry_opp not in ("偏空回避", "不宜追高"):
        return "谨慎试仓", s, "做多"
    if s >= 40:
        return "观望", s, "观望"
    return "回避", s, "观望" if bias_score >= -15 else "偏空"


def build_trade_sop(
    symbol: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    capital: float = 50_000.0,
    risk_pct: float = 1.0,
) -> TradeSOP:
    """Full SOP for one symbol (long-bias equity playbook)."""
    sym = normalize_symbol(symbol)
    info = cached_info(sym, cache_bucket(5))
    hist = fetch_history(sym, period=period, interval=interval)
    name = (
        (info or {}).get("shortName")
        or (info or {}).get("longName")
        or sym
    )

    if hist is None or hist.empty or "Close" not in hist.columns:
        return TradeSOP(
            symbol=sym,
            name=str(name),
            last_price=None,
            enter_ok="观望",
            enter_score=0,
            entry_low=None,
            entry_high=None,
            entry_plan=None,
            stop_loss=None,
            target_t1=None,
            target_t2=None,
            win_rate_pct=None,
            win_rate_label="—",
            stability_score=0,
            stability_label="—",
            side="观望",
            risk_per_share=None,
            rr_t1=None,
            position_shares=0,
            position_note="无行情数据",
            actions_now=["检查代码 / 网络 / 稍后再试"],
            summary="无法拉取历史数据，SOP 不可用。",
            period=period,
        )

    df = enrich(hist)
    last = float(df["Close"].iloc[-1])
    bias = analyze_bias(df)
    entry = analyze_entry(df)
    risk = analyze_risk(df)
    trend = analyze_trend(df)
    vol = analyze_volume(df)
    sr = analyze_support_resistance(df)
    rets = compute_returns(df)

    # Free market regime + optional Alpha Vantage (funda fill + news sentiment)
    try:
        regime = get_market_regime()
    except Exception:
        regime = None

    info_use = dict(info or {})
    av_filled: list[str] = []
    try:
        info_use, av_filled = merge_av_overview_into_info(info_use, sym)
    except Exception:
        av_filled = []

    targets = analyze_targets(df, info=info_use, entry=entry)
    funda = analyze_fundamentals(info_use)
    liq = analyze_liquidity(df, info_use)
    quality = extract_quality_extras(info_use, last_price=last)
    if av_filled:
        quality.notes.append(
            f"Alpha Vantage 补全 {len(av_filled)} 项基本面字段（仅填 yfinance 缺失）"
        )
    try:
        news = get_news_pulse(sym)
    except Exception:
        news = None

    # Soft note from news sentiment (Alpha Vantage)
    if news and getattr(news, "available", False) and news.sentiment_score is not None:
        sc = float(news.sentiment_score)
        if sc <= -0.20:
            quality.notes.append(f"新闻情绪偏空（AV {sc:+.2f}）：消息面逆风，宜减仓/观望")
        elif sc >= 0.25:
            quality.notes.append(f"新闻情绪偏多（AV {sc:+.2f}）：注意是否已定价/追高")

    bench_sym, bench_label = default_benchmark(sym)
    bench = None
    try:
        bench = fetch_history(bench_sym, period=period, interval=interval)
        rs = analyze_relative_strength(
            hist, bench, benchmark=bench_sym, bench_label=bench_label
        )
    except Exception:
        rs = None
    multi_rs = multi_horizon_rs(hist, bench) if bench is not None else {
        "score": None,
        "summary": "",
        "label": "—",
    }
    card = build_scorecard(
        bias.score,
        funda,
        risk,
        rs,
        regime_score=getattr(regime, "score", None) if regime else None,
        liquidity_score=liq.score,
        multi_rs_score=multi_rs.get("score"),
    )

    try:
        cal = cached_calendar(sym, cache_bucket(30))
        events = analyze_events(info_use, cal)
    except Exception:
        events = None

    earnings_soon = bool(getattr(events, "near_earnings", False)) if events else False
    if events is not None and not earnings_soon:
        for it in getattr(events, "items", []) or []:
            if getattr(it, "name", "") == "财报日":
                dleft = getattr(it, "days_left", None)
                if dleft is not None and 0 <= int(dleft) <= 7:
                    earnings_soon = True
                    break

    chase_high = bool(
        quality.fifty_two_week_pct is not None and quality.fifty_two_week_pct >= 92
    )

    e_low = entry.suggested_entry_low
    e_high = entry.suggested_entry_high
    stop = entry.stop_loss
    # Plan entry: mid of zone, or last if inside zone
    if e_low and e_high:
        if e_low <= last <= e_high:
            entry_plan = round(last, 4)
        else:
            entry_plan = round((float(e_low) + float(e_high)) / 2.0, 4)
    else:
        entry_plan = round(last, 4)

    t1 = getattr(getattr(targets, "short", None), "bull_target", None)
    t2 = getattr(getattr(targets, "medium", None), "bull_target", None)
    if t1 is None:
        t1 = getattr(getattr(targets, "ultra", None), "bull_target", None)

    risk_ps = None
    if entry_plan and stop and entry_plan > stop:
        risk_ps = float(entry_plan) - float(stop)
    reward_ps = None
    if entry_plan and t1 and t1 > entry_plan:
        reward_ps = float(t1) - float(entry_plan)

    rr = None
    if risk_ps and reward_ps and risk_ps > 0:
        rr = reward_ps / risk_ps

    path_wr = None
    if risk_ps and reward_ps:
        path_wr = _path_win_rate(df["Close"], risk_ps, reward_ps)

    # Blend path WR with daily up-win rate (path more relevant for stops/targets)
    day_wr = getattr(risk, "win_rate_pct", None)
    if path_wr is not None and day_wr is not None:
        win_rate = 0.70 * path_wr + 0.30 * float(day_wr)
    elif path_wr is not None:
        win_rate = path_wr
    elif day_wr is not None:
        win_rate = float(day_wr)
    else:
        win_rate = None

    if win_rate is not None:
        if win_rate >= 58:
            wr_label = "高"
        elif win_rate >= 48:
            wr_label = "中"
        else:
            wr_label = "低"
    else:
        wr_label = "—"

    exp_r = _expectancy_r(win_rate, rr)

    stab, stab_label = _stability_score(risk, trend, bias.score)
    enter_ok, enter_score, side = _enter_decision(
        entry.opportunity,
        entry.score,
        bias.bias,
        bias.score,
        stab,
        win_rate,
        rr,
        getattr(risk, "risk_level", "—") or "—",
        regime_score=getattr(regime, "score", None) if regime else None,
        liquidity_score=liq.score,
        expectancy_r=exp_r,
        earnings_soon=earnings_soon,
        chase_high=chase_high,
        multi_rs_score=multi_rs.get("score"),
    )

    # Position
    lot = suggest_lot_size(sym)
    plan_entry = float(entry_plan or last)
    plan_stop = float(stop or last * 0.97)
    pos = calc_position(
        capital=capital,
        risk_pct=risk_pct,
        entry_price=plan_entry,
        stop_price=plan_stop,
        short_target=float(t1) if t1 else None,
        medium_target=float(t2) if t2 else None,
        lot_size=lot,
    )
    pos_note = ""
    if not pos.valid:
        pos_note = pos.error or "仓位无效"
    elif pos.shares <= 0:
        pos_note = "；".join(pos.notes) if pos.notes else "风险预算不足 1 股"
    else:
        pos_note = (
            f"约 {pos.shares} 股 · 仓位市值 ${pos.position_value:,.0f} "
            f"· 占本金 {pos.position_pct_of_capital:.1f}%（本金按 HKD→USD 折算）"
        )

    # Checklist
    checklist: list[dict[str, str]] = []
    checklist.append(
        {
            "name": "入场评级",
            "status": "pass"
            if entry.opportunity in ("较佳入场", "可关注")
            else ("warn" if entry.opportunity == "观望" else "fail"),
            "detail": f"{entry.opportunity}（{entry.score:.0f}分）· {entry.side_bias}",
        }
    )
    checklist.append(
        {
            "name": "多空方向",
            "status": "pass"
            if bias.score >= 18
            else ("warn" if abs(bias.score) < 18 else "fail"),
            "detail": f"{bias.bias}（{bias.score:+.0f}）置信度 {bias.confidence}",
        }
    )
    checklist.append(
        {
            "name": "稳定度",
            "status": "pass"
            if stab >= 68
            else ("warn" if stab >= 45 else "fail"),
            "detail": f"{stab_label}（{stab:.0f}/100）· 风险等级 {getattr(risk, 'risk_level', '—')}",
        }
    )
    if win_rate is not None:
        checklist.append(
            {
                "name": "路径胜率",
                "status": "pass"
                if win_rate >= 55
                else ("warn" if win_rate >= 48 else "fail"),
                "detail": f"约 {win_rate:.0f}%（历史：先到 T1 再触止损 的比例，混合日线胜率）",
            }
        )
    if rr is not None:
        checklist.append(
            {
                "name": "盈亏比 R:R",
                "status": "pass" if rr >= 1.5 else ("warn" if rr >= 1.0 else "fail"),
                "detail": f"T1 R:R ≈ {rr:.2f}",
            }
        )
    if events is not None and getattr(events, "caution", None):
        checklist.append(
            {
                "name": "事件风险",
                "status": "fail" if earnings_soon else "warn",
                "detail": str(events.caution),
            }
        )
    if regime is not None:
        rstat = (
            "pass"
            if regime.score >= 60
            else ("warn" if regime.score >= 40 else "fail")
        )
        checklist.append(
            {
                "name": "市场环境",
                "status": rstat,
                "detail": f"{regime.label}（{regime.score:.0f}）· VIX {regime.vix_label}"
                + (f" {regime.vix:.1f}" if regime.vix is not None else "")
                + f" · SPY {regime.spy_trend}",
            }
        )
    checklist.append(
        {
            "name": "流动性",
            "status": "pass"
            if liq.score >= 65
            else ("warn" if liq.score >= 40 else "fail"),
            "detail": liq.summary.replace("**", ""),
        }
    )
    if exp_r is not None:
        checklist.append(
            {
                "name": "期望值 E[R]",
                "status": "pass" if exp_r >= 0.15 else ("warn" if exp_r >= 0 else "fail"),
                "detail": f"约 {exp_r:+.2f}R / 笔（用胜率×盈亏比估算，含1R亏损假设）",
            }
        )
    if multi_rs.get("score") is not None:
        checklist.append(
            {
                "name": "多周期强弱",
                "status": "pass"
                if multi_rs["score"] >= 58
                else ("warn" if multi_rs["score"] >= 42 else "fail"),
                "detail": multi_rs.get("summary", ""),
            }
        )

    # Actions
    actions_now: list[str] = []
    actions_wait: list[str] = []

    if enter_ok == "适合入场":
        actions_now.append(
            f"限价买入区 {e_low:.2f}–{e_high:.2f}"
            if e_low and e_high
            else f"参考价附近挂单 ≈ {entry_plan:.2f}"
        )
        if pos.shares > 0:
            actions_now.append(f"按 1R={risk_pct:.1f}% 下单约 {pos.shares} 股（本金 {capital:,.0f}）")
        else:
            actions_now.append("先调本金/风险% 或收紧止损，使股数 ≥ 1")
        actions_now.append(f"止损设在 {stop:.2f}" if stop else "设定止损")
        if t1:
            actions_now.append(f"T1 目标 {t1:.2f}（到则减仓 50% 或移动止损）")
        if t2:
            actions_now.append(f"T2 目标 {t2:.2f}（趋势持有）")
        actions_now.append("成交后写交易日志：理由 / 止损 / 目标")
    elif enter_ok == "谨慎试仓":
        half = max(1, pos.shares // 2) if pos.shares > 0 else 0
        actions_now.append(
            f"最多 0.5R（约 {half} 股）" if half else "最多半仓或 0.5R（不是满 1R）"
        )
        if e_low and e_high:
            actions_now.append(f"只在 {e_low:.2f}–{e_high:.2f} 回踩挂限价，不追高")
        actions_now.append(f"止损 {stop:.2f} 必须预设" if stop else "先定止损再考虑")
        if earnings_soon:
            actions_now.append("临近财报：减小仓位或等财报后再做")
        if regime is not None and regime.score < 45:
            actions_now.append("大盘环境偏弱：优先高流动性标的 + 更紧风控")
        actions_wait.append("若跌破止损或机会评级变「回避」→ 立刻停手")
        actions_wait.append("等方向更清晰或稳定度回升再加仓")
    elif enter_ok == "观望":
        actions_now.append("今天不开新仓")
        if e_low:
            actions_wait.append(f"价格进入 {e_low:.2f}–{e_high:.2f} 且多空转正再评估")
        if regime is not None and regime.score < 40:
            actions_wait.append("等待 VIX 回落或 SPY 重回均线上方再开多")
        actions_wait.append("继续观察量价 / 均线结构，不预判抄底")
    else:
        actions_now.append("回避做多；不加仓")
        actions_now.append("已有多单：检查是否触及止损 / 考虑减仓")
        actions_wait.append("等待空头结构结束、入场评级改善后再做多")

    if vol and getattr(vol, "trend", "") == "放量" and bias.score < 0:
        actions_wait.append("放量偏空：避免在恐慌杀跌中接刀")
    if chase_high:
        actions_wait.append("靠近 52 周高位：等回踩入场区，避免 FOMO 追高")
    if liq.score < 40:
        actions_now.append("流动性偏弱：减小单笔、用限价、避免市价扫单")

    invalidation = entry.invalidation or (
        f"收盘跌破止损 {stop:.2f} 则本计划作废" if stop else "结构破坏则作废"
    )

    reg_bit = ""
    if regime is not None:
        reg_bit = f"市场 {regime.label}（{regime.score:.0f}）；"
    exp_bit = f"期望 {exp_r:+.2f}R；" if exp_r is not None else ""
    if win_rate is not None:
        summary = (
            f"{sym} 现价 {last:.2f} → **{enter_ok}**（{enter_score:.0f}分）。"
            f"{reg_bit}方向 {bias.bias}；入场 {entry.opportunity}；"
            f"路径胜率约 {win_rate:.0f}%（{wr_label}）；{exp_bit}稳定度 {stab_label}。"
        )
    else:
        summary = (
            f"{sym} 现价 {last:.2f} → **{enter_ok}**（{enter_score:.0f}分）。"
            f"{reg_bit}方向 {bias.bias}；入场 {entry.opportunity}；稳定度 {stab_label}。"
        )

    notes = [
        "胜率=历史路径模拟（先触 T1 再触止损），非未来保证",
        "期望值 E[R]=胜率×盈亏比 − (1−胜率)×1R，用于比较「值不值得做」",
        "稳定度综合波动、回撤、夏普、趋势强度",
        "综合分权重：技术22/基本面22/风险18/多周期RS15/市场环境13/流动性10",
        f"区间报酬 {rets.get('total_return_pct'):.1f}%，年化波动 {rets.get('volatility_pct'):.1f}%"
        if rets.get("total_return_pct") is not None and rets.get("volatility_pct") is not None
        else "数据仅供实盘辅助",
    ]
    if sr.nearest_support:
        notes.append(f"近支撑 ≈ {sr.nearest_support:.2f} · 近阻力 ≈ {sr.nearest_resistance}")
    notes.extend(quality.notes[:4])

    status = free_data_status()
    data_sources = list(getattr(regime, "sources", None) or ["yfinance"])
    if status.get("fred"):
        data_sources.append("FRED")
    if status.get("alphavantage") and (av_filled or (news and getattr(news, "source", "") == "AlphaVantage")):
        data_sources.append("AlphaVantage")
    if status.get("finnhub") and news and getattr(news, "source", "") == "Finnhub":
        data_sources.append("Finnhub")
    elif status.get("finnhub") and news and getattr(news, "available", False) and "AlphaVantage" not in data_sources:
        data_sources.append("Finnhub")

    return TradeSOP(
        symbol=sym,
        name=str(name),
        last_price=round(last, 4),
        enter_ok=enter_ok,
        enter_score=round(enter_score, 1),
        entry_low=round(float(e_low), 4) if e_low else None,
        entry_high=round(float(e_high), 4) if e_high else None,
        entry_plan=entry_plan,
        stop_loss=round(float(stop), 4) if stop else None,
        target_t1=round(float(t1), 4) if t1 else None,
        target_t2=round(float(t2), 4) if t2 else None,
        win_rate_pct=round(win_rate, 1) if win_rate is not None else None,
        win_rate_label=wr_label,
        stability_score=stab,
        stability_label=stab_label,
        side=side,
        risk_per_share=round(risk_ps, 4) if risk_ps else None,
        rr_t1=round(rr, 2) if rr else None,
        position_shares=int(pos.shares),
        position_note=pos_note,
        actions_now=actions_now,
        actions_wait=actions_wait,
        invalidation=invalidation,
        checklist=checklist,
        bias=bias.bias,
        bias_score=bias.score,
        opportunity=entry.opportunity,
        risk_level=getattr(risk, "risk_level", "—") or "—",
        max_dd_pct=getattr(risk, "max_drawdown_pct", None),
        ann_vol_pct=getattr(risk, "ann_vol_pct", None),
        scorecard_stance=card.stance,
        scorecard_total=card.total_score,
        summary=summary,
        period=period,
        notes=notes,
        expectancy_r=exp_r,
        regime_label=getattr(regime, "label", "—") if regime else "—",
        regime_score=getattr(regime, "score", None) if regime else None,
        regime_summary=getattr(regime, "summary", "") if regime else "",
        regime_bullets=list(getattr(regime, "bullets", []) or []) if regime else [],
        liquidity_label=liq.label,
        liquidity_score=liq.score,
        multi_rs_summary=str(multi_rs.get("summary") or ""),
        quality_notes=list(quality.notes or []),
        news_summary=getattr(news, "summary", "") if news else "",
        data_sources=data_sources,
        scorecard_bullets=list(card.bullets or []),
    )
