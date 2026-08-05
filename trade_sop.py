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
from edge_signals import edge_bundle
from exit_plan import (
    DEFAULT_SLIP_PCT,
    ExitPlan,
    SlippageRR,
    apply_long_slippage,
    build_exit_plan,
)
from mtf_signals import mtf_bundle
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
class SwingHorizonPlan:
    """短线波段单周期计划：0–2周 或 2–4周。"""

    key: str  # h1 | h2
    label: str  # 0–2周 | 2–4周
    bars: int  # 交易日路径窗口
    verdict: str  # 可以入場 | 可以試倉 | 暫緩觀望 | 不做多
    win_rate_pct: float | None
    entry_low: float | None
    entry_high: float | None
    entry_plan: float | None
    stop_loss: float | None
    target: float | None
    rr: float | None
    expectancy_r: float | None
    risk_per_share: float | None
    reward_per_share: float | None
    note: str = ""
    # 滑点后可执行赔率
    rr_net: float | None = None
    expectancy_net: float | None = None
    slip_note: str = ""


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

    # 短线波段双周期（主展示）
    swing_h1: SwingHorizonPlan | None = None  # 0–2周
    swing_h2: SwingHorizonPlan | None = None  # 2–4周
    trend_note: str = ""  # 走势一句话
    # 主周期选择：h1 | h2 — 决定 enter_ok / 出场卡
    primary_horizon: str = "h1"
    primary_plan: SwingHorizonPlan | None = None
    exit_plan: ExitPlan | None = None
    slip_rr: SlippageRR | None = None

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
    # Edge signals (sector RS / volume / IV)
    sector_rs_summary: str = ""
    sector_rs_score: float | None = None
    sector_rs_label: str = "—"
    volume_confirm_summary: str = ""
    volume_confirm_score: float | None = None
    volume_confirm_label: str = "—"
    iv_summary: str = ""
    iv_score: float | None = None
    iv_label: str = "—"
    iv_high_event: bool = False
    false_break_summary: str = ""
    false_break_score: float | None = None
    false_break_label: str = "—"
    false_break_risk: bool = False
    trend_align_summary: str = ""
    trend_align_score: float | None = None
    trend_align_label: str = "—"
    against_trend: bool = False
    weekly_label: str = "—"
    weekly_summary: str = ""
    weekly_allow_long: bool = True
    adx_label: str = "—"
    adx_value: float | None = None
    adx_summary: str = ""
    adx_trending: bool = False
    fib_summary: str = ""
    h1_label: str = "—"
    h1_summary: str = ""
    h1_ready: bool = False


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


def _swing_verdict(
    *,
    entry_opp: str,
    bias_label: str,
    bias_score: float,
    wr: float | None,
    rr: float | None,
    exp_r: float | None,
    price_far_chase: bool,
    vol_label: str = "",
    false_break_risk: bool = False,
    block_breakout_chase: bool = False,
    against_trend: bool = False,
    trend_label: str = "",
    trend_score: float | None = None,
    weekly_allow_long: bool = True,
    adx_trending: bool | None = None,
    adx_value: float | None = None,
    h1_ready: bool | None = None,
) -> str:
    """
    短线波段结论（偏操作、仍挡明显亏钱结构）:
      可以入場 | 可以試倉 | 暫緩觀望 | 不做多
    """
    if entry_opp == "偏空回避" or "强烈看空" in bias_label:
        return "不做多"
    if "看空" in bias_label and bias_score <= -25:
        return "不做多"
    if not weekly_allow_long:
        # 周线空头：短线不做「可以入場」；若其他极差直接不做多
        if bias_score < 0 or entry_opp in ("偏空回避", "不宜追高"):
            return "不做多"
    if price_far_chase or entry_opp == "不宜追高":
        return "暫緩觀望"
    if vol_label == "放量下跌":
        return "暫緩觀望"
    # 假突破：禁止追多；明显失败则暂缓
    if false_break_risk:
        return "暫緩觀望"
    if block_breakout_chase and price_far_chase:
        return "暫緩觀望"
    # 逆势（大盘/板块空、个股硬多）→ 不做满仓，最多暂缓
    if against_trend and trend_label == "逆势":
        return "暫緩觀望"
    # 震荡市 + 追突破形态 → 暂缓（ADX 低）
    if adx_trending is False and adx_value is not None and adx_value < 18:
        if entry_opp in ("较佳入场",) and block_breakout_chase:
            return "暫緩觀望"
    # 赔率太差：短线也难赚
    if rr is not None and rr < 0.95:
        return "暫緩觀望"
    if exp_r is not None and exp_r < -0.08:
        return "暫緩觀望"

    good_setup = entry_opp in ("较佳入场", "可关注")
    ok_bias = bias_score >= -8 and "强烈看空" not in bias_label
    wr_ok = wr is None or wr >= 50
    wr_good = wr is None or wr >= 55
    rr_ok = rr is None or rr >= 1.05
    rr_good = rr is not None and rr >= 1.25
    exp_ok = exp_r is None or exp_r >= 0.0
    exp_good = exp_r is not None and exp_r >= 0.10
    trend_ok = trend_score is None or trend_score >= 48
    trend_good = trend_score is None or trend_score >= 60
    no_fbo = not false_break_risk and not (
        block_breakout_chase and entry_opp == "不宜追高"
    )

    if (
        good_setup
        and ok_bias
        and wr_good
        and rr_good
        and exp_good
        and not price_far_chase
        and not against_trend
        and trend_good
        and no_fbo
        and not block_breakout_chase  # 满仓不要在「仅待确认突破」时追
        and weekly_allow_long
        and (adx_trending is not False or (adx_value is not None and adx_value >= 18))
    ):
        # 可以入場：日线OK；1H 未就绪仍可挂限价（ready 仅影响文案）
        return "可以入場"
    if ok_bias and wr_ok and rr_ok and exp_ok and entry_opp not in ("偏空回避",):
        if against_trend:
            return "暫緩觀望"
        if not weekly_allow_long:
            # 周线空头最多试仓，且要求日线结构尚可
            if good_setup and not false_break_risk and wr_ok:
                return "可以試倉"
            return "暫緩觀望"
        if block_breakout_chase and not good_setup:
            pass
        if (good_setup or bias_score >= 10) and trend_ok and not false_break_risk:
            return "可以試倉"
        if entry_opp in ("观望",) and bias_score >= 18 and trend_ok:
            return "可以試倉"
    return "暫緩觀望"


def _build_swing_plan(
    *,
    key: str,
    label: str,
    bars: int,
    close: pd.Series,
    entry_low: float | None,
    entry_high: float | None,
    entry_mid: float | None,
    display_limit: float | None,
    stop: float | None,
    target: float | None,
    entry_opp: str,
    bias_label: str,
    bias_score: float,
    price_far_chase: bool,
    vol_label: str,
    false_break_risk: bool = False,
    block_breakout_chase: bool = False,
    against_trend: bool = False,
    trend_label: str = "",
    trend_score: float | None = None,
    weekly_allow_long: bool = True,
    adx_trending: bool | None = None,
    adx_value: float | None = None,
    h1_ready: bool | None = None,
) -> SwingHorizonPlan:
    risk_ps = None
    reward_ps = None
    mid = entry_mid
    if mid is not None and stop is not None and mid > stop:
        risk_ps = float(mid) - float(stop)
    if mid is not None and target is not None and target > mid:
        reward_ps = float(target) - float(mid)
    rr = None
    if risk_ps and reward_ps and risk_ps > 0:
        rr = reward_ps / risk_ps
    wr = None
    if risk_ps and reward_ps and close is not None and len(close) > bars + 30:
        wr = _path_win_rate(
            close, risk_ps, reward_ps, lookback=100, horizon=bars
        )
    exp_r = _expectancy_r(wr, rr)
    verdict = _swing_verdict(
        entry_opp=entry_opp,
        bias_label=bias_label,
        bias_score=bias_score,
        wr=wr,
        rr=rr,
        exp_r=exp_r,
        price_far_chase=price_far_chase,
        vol_label=vol_label,
        false_break_risk=false_break_risk,
        block_breakout_chase=block_breakout_chase,
        against_trend=against_trend,
        trend_label=trend_label,
        trend_score=trend_score,
        weekly_allow_long=weekly_allow_long,
        adx_trending=adx_trending,
        adx_value=adx_value,
        h1_ready=h1_ready,
    )
    note_bits = []
    if wr is not None:
        note_bits.append(f"历史路径约{bars}个交易日内先到目标再触止蚀的比例")
    if rr is not None:
        note_bits.append(f"R:R≈{rr:.2f}")
    if exp_r is not None:
        note_bits.append(f"E[R]≈{exp_r:+.2f}")
    if h1_ready is True:
        note_bits.append("1H可掛單")
    elif h1_ready is False:
        note_bits.append("等1H/回踩")

    # 滑点后 R:R / E[R]（可执行）
    entry_ref = (
        float(display_limit)
        if display_limit
        else (float(mid) if mid is not None else None)
    )
    slip = apply_long_slippage(
        entry_ref, stop, target, win_rate_pct=wr, slip_pct=DEFAULT_SLIP_PCT
    )
    if slip.rr_net is not None:
        note_bits.append(f"净R:R≈{slip.rr_net:.2f}")

    return SwingHorizonPlan(
        key=key,
        label=label,
        bars=bars,
        verdict=verdict,
        win_rate_pct=round(wr, 1) if wr is not None else None,
        entry_low=round(float(entry_low), 4) if entry_low else None,
        entry_high=round(float(entry_high), 4) if entry_high else None,
        entry_plan=round(float(display_limit), 4)
        if display_limit
        else (round(float(mid), 4) if mid else None),
        stop_loss=round(float(stop), 4) if stop else None,
        target=round(float(target), 4) if target else None,
        rr=round(rr, 2) if rr is not None else None,
        expectancy_r=exp_r,
        risk_per_share=round(risk_ps, 4) if risk_ps else None,
        reward_per_share=round(reward_ps, 4) if reward_ps else None,
        note=" · ".join(note_bits),
        rr_net=slip.rr_net,
        expectancy_net=slip.exp_net,
        slip_note=slip.note,
    )


# ---- Profit-focused hard floors (raise only if you accept worse expectancy) ----
# 适合入场：赔率+期望值都要过线，避免「高胜率低赔率」假好票
MIN_RR_FULL = 1.35
MIN_RR_CAUTIOUS = 1.05
MIN_EXP_FULL = 0.12
MIN_EXP_CAUTIOUS = 0.0
MIN_WR_FULL = 52.0
MIN_STAB_FULL = 48.0
MIN_STAB_CAUTIOUS = 30.0
MIN_REGIME_FULL = 48.0
MIN_LIQ_FULL = 42.0
MIN_RS_FULL = 42.0


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
    price_above_zone: bool = False,
    price_far_chase: bool = False,
    sector_rs_score: float | None = None,
    volume_confirm_score: float | None = None,
    iv_score: float | None = None,
    iv_high_event: bool = False,
) -> tuple[str, float, str]:
    """
    Map models → 适合入场 / 谨慎试仓 / 观望 / 回避.

    Soft score is for ranking; **hard gates** protect expectancy:
    bad R:R / negative E[R] / chase / earnings / thin liquidity cannot full-size.
    Edge: sector RS, volume confirm, IV event risk.
    """
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
            s += 10
        elif rr >= 1.35:
            s += 6
        elif rr >= 1.05:
            s += 2
        elif rr < 1.0:
            s -= 14  # 赔率不足，强烈减分
        elif rr < 0.8:
            s -= 18

    if risk_level in ("高", "极高"):
        s -= 8

    if regime_score is not None:
        s += (float(regime_score) - 50.0) * 0.22
        if regime_score < 35:
            s -= 8
    if liquidity_score is not None:
        s += (float(liquidity_score) - 50.0) * 0.10
        if liquidity_score < 35:
            s -= 10
    if multi_rs_score is not None:
        s += (float(multi_rs_score) - 50.0) * 0.10
    if expectancy_r is not None:
        if expectancy_r >= 0.35:
            s += 10
        elif expectancy_r >= 0.12:
            s += 5
        elif expectancy_r >= 0:
            s += 1
        else:
            s -= 16  # 负期望：长期必亏结构
    if earnings_soon:
        s -= 14
    if chase_high or price_above_zone:
        s -= 10
    if price_far_chase:
        s -= 12

    # Edge signals: sector / volume / IV
    if sector_rs_score is not None:
        s += (float(sector_rs_score) - 50.0) * 0.12
        if sector_rs_score < 38:
            s -= 6
    if volume_confirm_score is not None:
        s += (float(volume_confirm_score) - 50.0) * 0.14
        if volume_confirm_score <= 32:
            s -= 10  # 放量下跌等
        elif volume_confirm_score >= 68:
            s += 4
    if iv_score is not None:
        s += (float(iv_score) - 50.0) * 0.08
    if iv_high_event:
        s -= 8

    s = float(max(0.0, min(100.0, s)))

    # ---- Absolute blocks (protect capital / expectancy) ----
    if entry_opp in ("偏空回避",) or ("强烈看空" in bias_label and entry_score < 50):
        return "回避", min(s, 35.0), "偏空"
    if "看空" in bias_label and bias_score <= -18:
        return "回避", min(s, 38.0), "偏空"
    if liquidity_score is not None and liquidity_score < 28:
        return "观望", min(s, 48.0), "观望"
    # 赔率不足 1：做多数学期望很难正 → 最多观望
    if rr is not None and rr < MIN_RR_CAUTIOUS:
        if s >= 40:
            return "观望", min(s, 52.0), "观望"
        return "回避", min(s, 40.0), "观望" if bias_score >= -15 else "偏空"
    # 负期望：禁止开新多
    if expectancy_r is not None and expectancy_r < MIN_EXP_CAUTIOUS:
        if s >= 40:
            return "观望", min(s, 50.0), "观望"
        return "回避", min(s, 38.0), "观望"
    # 远离入场区追高
    if price_far_chase or entry_opp in ("不宜追高",):
        if s >= 45:
            return "观望", min(s, 50.0), "观望"
        return "回避", min(s, 40.0), "观望"
    if earnings_soon:
        # 财报窗口：永不适合入场；期望仍为正才允许极小试仓
        can_half = (
            s >= 58
            and (rr is None or rr >= MIN_RR_CAUTIOUS)
            and (expectancy_r is None or expectancy_r >= 0.05)
            and not price_far_chase
        )
        if can_half:
            return "谨慎试仓", min(s, 62.0), "做多"
        if s >= 40:
            return "观望", min(s, 52.0), "观望"
        return "回避", min(s, 40.0), "观望"
    if regime_score is not None and regime_score < 32:
        if s >= 50:
            return "观望", min(s, 55.0), "观望"
        return "回避", min(s, 40.0), "观望" if bias_score >= -15 else "偏空"

    # ---- 适合入场：全部硬条件 ----
    full_ok = (
        s >= 74
        and entry_opp in ("较佳入场", "可关注")
        and bias_score >= -5
        and "看空" not in bias_label
        and (rr is not None and rr >= MIN_RR_FULL)
        and (expectancy_r is not None and expectancy_r >= MIN_EXP_FULL)
        and (win_rate is None or win_rate >= MIN_WR_FULL)
        and stability >= MIN_STAB_FULL
        and (regime_score is None or regime_score >= MIN_REGIME_FULL)
        and (liquidity_score is None or liquidity_score >= MIN_LIQ_FULL)
        and (multi_rs_score is None or multi_rs_score >= MIN_RS_FULL)
        and (sector_rs_score is None or sector_rs_score >= 42)
        and (volume_confirm_score is None or volume_confirm_score >= 40)
        and not iv_high_event
        and not earnings_soon
        and not chase_high
        and not price_above_zone
        and not price_far_chase
        and risk_level not in ("极高",)
    )
    if full_ok:
        return "适合入场", s, "做多"

    # ---- 谨慎试仓：允许稍差，但赔率/期望不能穿底 ----
    cautious_ok = (
        s >= 55
        and entry_opp not in ("偏空回避", "不宜追高")
        and bias_score >= -12
        and "强烈看空" not in bias_label
        and (rr is None or rr >= MIN_RR_CAUTIOUS)
        and (expectancy_r is None or expectancy_r >= MIN_EXP_CAUTIOUS)
        and stability >= MIN_STAB_CAUTIOUS
        and not price_far_chase
        and (liquidity_score is None or liquidity_score >= 32)
        and (regime_score is None or regime_score >= 38)
        and (volume_confirm_score is None or volume_confirm_score >= 28)
        # 放量下跌禁止试仓
        and not (volume_confirm_score is not None and volume_confirm_score < 28)
    )
    if cautious_ok:
        return "谨慎试仓", min(s, 72.0) if not full_ok else s, "做多"

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
    primary_horizon: str = "h1",
) -> TradeSOP:
    """
    Full SOP for one symbol (short-term swing playbook).

    primary_horizon: ``h1`` = 0–2周（默认）, ``h2`` = 2–4周
    — 决定 enter_ok / 出场纪律 / 主展示计划。
    """
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

    # 多周期：周线过滤 + ADX + Fib  refinement + 1H 触发
    weekly = adx_r = fib_r = h1_r = None
    fib_note = ""
    try:
        mtf = mtf_bundle(sym, df, e_low, e_high)
        weekly = mtf.get("weekly")
        adx_r = mtf.get("adx")
        fib_r = mtf.get("fib")
        h1_r = mtf.get("h1")
        fib_note = str(mtf.get("fib_note") or "")
        if mtf.get("entry_low") is not None and mtf.get("entry_high") is not None:
            e_low = mtf["entry_low"]
            e_high = mtf["entry_high"]
    except Exception:
        pass

    # 评分用「入场区中位」固定 R:R / 期望 / 路径胜率
    if e_low and e_high:
        zone_mid = round((float(e_low) + float(e_high)) / 2.0, 4)
        entry_plan = zone_mid
        in_zone = float(e_low) <= last <= float(e_high)
        display_limit = round(last, 4) if in_zone else zone_mid
    else:
        zone_mid = round(last, 4)
        entry_plan = zone_mid
        display_limit = zone_mid
        in_zone = True

    # 现价相对入场区
    price_above_zone = False
    price_far_chase = False
    if e_high is not None and last > 0:
        eh = float(e_high)
        if last > eh * 1.02:
            price_above_zone = True
        if last > eh * 1.045:
            price_far_chase = True

    weekly_allow = True if weekly is None else bool(getattr(weekly, "allow_long", True))
    adx_trending = getattr(adx_r, "trending", None) if adx_r else None
    adx_val = getattr(adx_r, "adx", None) if adx_r else None
    h1_ready = getattr(h1_r, "ready", None) if h1_r else None

    # 短线目标：0–2周用超短/短期；2–4周用短期/中期
    t_ultra = getattr(getattr(targets, "ultra", None), "bull_target", None)
    t_short = getattr(getattr(targets, "short", None), "bull_target", None)
    t_med = getattr(getattr(targets, "medium", None), "bull_target", None)
    t1 = t_ultra or t_short  # 0–2周目标
    t2 = t_short or t_med  # 2–4周目标
    if t2 is not None and t1 is not None and float(t2) < float(t1):
        t2 = t_med or t2

    # 结构赔率一律相对 zone mid（稳定）— 默认用 0–2 周目标
    risk_ps = None
    if entry_plan and stop and entry_plan > stop:
        risk_ps = float(entry_plan) - float(stop)
    reward_ps = None
    if entry_plan and t1 and t1 > entry_plan:
        reward_ps = float(t1) - float(entry_plan)

    rr = None
    if risk_ps and reward_ps and risk_ps > 0:
        rr = reward_ps / risk_ps

    # 路径用已收盘日线（去掉未完成当日 bar）
    close_for_path = df["Close"]
    if len(close_for_path) >= 40:
        try:
            from market_session import us_session_clock

            sess = us_session_clock().session
            if sess in ("pre_market", "rth", "after_hours", "overnight"):
                close_for_path = close_for_path.iloc[:-1]
        except Exception:
            pass

    path_wr = None
    if risk_ps and reward_ps:
        path_wr = _path_win_rate(
            close_for_path, risk_ps, reward_ps, lookback=100, horizon=10
        )

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

    # Edge: sector RS + volume confirm + IV regime
    try:
        edges = edge_bundle(sym, df, info_use, period=period)
        sector_rs = edges["sector_rs"]
        vol_cf = edges["volume"]
        iv_reg = edges["iv"]
        fbo = edges.get("false_break")
        trend_al = edges.get("trend_align")
    except Exception:
        sector_rs = None
        vol_cf = None
        iv_reg = None
        fbo = None
        trend_al = None

    vol_label = getattr(vol_cf, "label", "") if vol_cf else ""
    fbo_risk = bool(getattr(fbo, "false_break_risk", False)) if fbo else False
    fbo_block = bool(getattr(fbo, "block_breakout_chase", False)) if fbo else False
    against_tr = bool(getattr(trend_al, "against_trend", False)) if trend_al else False
    trend_lab = getattr(trend_al, "label", "") if trend_al else ""
    trend_sc = getattr(trend_al, "score", None) if trend_al else None

    # 双周期短线计划（主输出）
    _mtf_kw = dict(
        false_break_risk=fbo_risk,
        block_breakout_chase=fbo_block,
        against_trend=against_tr,
        trend_label=trend_lab,
        trend_score=trend_sc,
        weekly_allow_long=weekly_allow,
        adx_trending=adx_trending,
        adx_value=adx_val,
        h1_ready=h1_ready,
    )
    swing_h1 = _build_swing_plan(
        key="h1",
        label="0–2周",
        bars=10,
        close=close_for_path,
        entry_low=e_low,
        entry_high=e_high,
        entry_mid=entry_plan,
        display_limit=display_limit,
        stop=stop,
        target=t1,
        entry_opp=entry.opportunity,
        bias_label=bias.bias,
        bias_score=bias.score,
        price_far_chase=price_far_chase,
        vol_label=vol_label,
        **_mtf_kw,
    )
    swing_h2 = _build_swing_plan(
        key="h2",
        label="2–4周",
        bars=20,
        close=close_for_path,
        entry_low=e_low,
        entry_high=e_high,
        entry_mid=entry_plan,
        display_limit=display_limit,
        stop=stop,
        target=t2,
        entry_opp=entry.opportunity,
        bias_label=bias.bias,
        bias_score=bias.score,
        price_far_chase=price_far_chase,
        vol_label=vol_label,
        **_mtf_kw,
    )
    trend_note = (
        f"走势：{bias.bias}（{bias.score:+.0f}）· 入场 {entry.opportunity}· "
        f"周线 {getattr(weekly, 'label', '—') if weekly else '—'}· "
        f"ADX {getattr(adx_r, 'label', '—') if adx_r else '—'}· "
        f"1H {getattr(h1_r, 'label', '—') if h1_r else '—'}· "
        f"跟势 {trend_lab or '—'}· 假突破 {getattr(fbo, 'label', '—') if fbo else '—'}"
    )

    stab, stab_label = _stability_score(risk, trend, bias.score)
    # 综合风控分仍计算；最终「做不做」以短线 0–2 周波段结论为准（你的持仓周期）
    _, enter_score, _ = _enter_decision(
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
        price_above_zone=price_above_zone,
        price_far_chase=price_far_chase,
        sector_rs_score=getattr(sector_rs, "score", None) if sector_rs else None,
        volume_confirm_score=getattr(vol_cf, "score", None) if vol_cf else None,
        iv_score=getattr(iv_reg, "score", None) if iv_reg else None,
        iv_high_event=bool(getattr(iv_reg, "high_event_risk", False)) if iv_reg else False,
    )
    # 主周期：决定最终做不做、出场卡、滑点 R:R
    ph = (primary_horizon or "h1").lower().strip()
    if ph not in ("h1", "h2"):
        ph = "h1"
    primary = swing_h1 if ph == "h1" else swing_h2
    if primary is None:
        primary = swing_h1

    _map_ok = {
        "可以入場": ("适合入场", "做多"),
        "可以試倉": ("谨慎试仓", "做多"),
        "暫緩觀望": ("观望", "观望"),
        "不做多": ("回避", "偏空"),
    }
    enter_ok, side = _map_ok.get(primary.verdict, ("观望", "观望"))
    # 净 R:R 过差时再降级（可执行赔率）
    if enter_ok in ("适合入场", "谨慎试仓") and primary.rr_net is not None:
        if primary.rr_net < 0.95:
            enter_ok, side = "观望", "观望"
            primary.verdict = "暫緩觀望"
        elif enter_ok == "适合入场" and primary.rr_net < 1.15:
            enter_ok, side = "谨慎试仓", "做多"
            if primary.verdict == "可以入場":
                primary.verdict = "可以試倉"

    # 分数：用波段可操作性映射，方便扫描排序
    enter_score = {
        "适合入场": max(enter_score, 78.0),
        "谨慎试仓": max(min(enter_score, 72.0), 58.0),
        "观望": min(enter_score, 52.0),
        "回避": min(enter_score, 35.0),
    }.get(enter_ok, enter_score)

    # 出场纪律（主周期）
    exit_pl = build_exit_plan(
        horizon_key=primary.key,
        horizon_label=primary.label,
        max_hold_days=primary.bars,
        entry=primary.entry_plan or entry_plan,
        stop=primary.stop_loss or stop,
        t1=primary.target,
        t2=(swing_h2.target if primary.key == "h1" else None),
        scale_out_pct=0.50,
    )
    slip_main = apply_long_slippage(
        primary.entry_plan or entry_plan,
        primary.stop_loss or stop,
        primary.target,
        win_rate_pct=primary.win_rate_pct,
        slip_pct=DEFAULT_SLIP_PCT,
    )

    # Position: 谨慎试仓强制 0.5R；观望/回避仍给「若强行」参考仓但标注
    lot = suggest_lot_size(sym)
    # 仓位按结构中位；若已在区内可用 display_limit 作为更贴近的限价参考
    plan_entry = float(display_limit if in_zone else (entry_plan or last))
    plan_stop = float(stop or last * 0.97)
    if enter_ok == "适合入场":
        eff_risk_pct = float(risk_pct)
        risk_tag = "1R"
    elif enter_ok == "谨慎试仓":
        eff_risk_pct = float(risk_pct) * 0.5
        risk_tag = "0.5R"
    else:
        eff_risk_pct = float(risk_pct)
        risk_tag = "参考1R(SOP不建议开)"
    pos = calc_position(
        capital=capital,
        risk_pct=eff_risk_pct,
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
            f"{risk_tag} · 约 {pos.shares} 股 · 仓位市值 ${pos.position_value:,.0f} "
            f"· 占本金 {pos.position_pct_of_capital:.1f}%（本金按 HKD→USD 折算）"
        )
        if enter_ok == "谨慎试仓":
            pos_note += " · 半仓试错，到 T1 先减半"

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
                "status": "pass"
                if rr >= MIN_RR_FULL
                else ("warn" if rr >= MIN_RR_CAUTIOUS else "fail"),
                "detail": (
                    f"T1 R:R ≈ {rr:.2f} · 满仓需≥{MIN_RR_FULL} · 试仓需≥{MIN_RR_CAUTIOUS}"
                    + (" · 不足则禁止开多" if rr < MIN_RR_CAUTIOUS else "")
                ),
            }
        )
    if price_above_zone or price_far_chase:
        checklist.append(
            {
                "name": "价格 vs 入场区",
                "status": "fail" if price_far_chase else "warn",
                "detail": (
                    "现价明显高于入场区上沿：追高会压缩赔率"
                    + (" · 已禁止试仓" if price_far_chase else " · 不得满仓")
                ),
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
    if sector_rs is not None and getattr(sector_rs, "available", False):
        sc = sector_rs.score
        checklist.append(
            {
                "name": "板块相对强弱",
                "status": "pass"
                if sc is not None and sc >= 58
                else ("warn" if sc is not None and sc >= 42 else "fail"),
                "detail": sector_rs.summary,
            }
        )
    if vol_cf is not None and getattr(vol_cf, "available", False):
        checklist.append(
            {
                "name": "量能确认",
                "status": "pass"
                if vol_cf.score >= 62
                else ("warn" if vol_cf.score >= 40 else "fail"),
                "detail": vol_cf.summary,
            }
        )
    if iv_reg is not None and getattr(iv_reg, "available", False):
        checklist.append(
            {
                "name": "IV 环境",
                "status": "fail"
                if iv_reg.high_event_risk
                else ("pass" if iv_reg.score >= 55 else "warn"),
                "detail": iv_reg.summary,
            }
        )
    if fbo is not None and getattr(fbo, "available", False):
        checklist.append(
            {
                "name": "假突破过滤",
                "status": "fail"
                if fbo.false_break_risk
                else (
                    "warn"
                    if fbo.block_breakout_chase
                    else ("pass" if fbo.score >= 60 else "warn")
                ),
                "detail": fbo.summary,
            }
        )
    if trend_al is not None and getattr(trend_al, "available", False):
        checklist.append(
            {
                "name": "跟势(SPY/板块)",
                "status": "fail"
                if trend_al.against_trend
                else ("pass" if trend_al.score >= 60 else "warn"),
                "detail": trend_al.summary,
            }
        )
    if weekly is not None and getattr(weekly, "available", False):
        checklist.append(
            {
                "name": "周线过滤",
                "status": "pass"
                if weekly.allow_long
                else "fail",
                "detail": weekly.summary,
            }
        )
    if adx_r is not None and getattr(adx_r, "available", False):
        checklist.append(
            {
                "name": "ADX趋势强度",
                "status": "pass"
                if adx_r.trending
                else "warn",
                "detail": adx_r.summary,
            }
        )
    if fib_r is not None and getattr(fib_r, "available", False):
        checklist.append(
            {
                "name": "Fib回撤区",
                "status": "pass" if "Fib回撤区" in (fib_r.label or "") else "warn",
                "detail": fib_r.summary + (f" · {fib_note}" if fib_note else ""),
            }
        )
    if h1_r is not None and getattr(h1_r, "available", False):
        checklist.append(
            {
                "name": "1H触发",
                "status": "pass" if h1_r.ready else "warn",
                "detail": h1_r.summary,
            }
        )

    # Actions
    actions_now: list[str] = []
    actions_wait: list[str] = []

    if enter_ok == "适合入场":
        actions_now.append(
            f"限价只在 {e_low:.2f}–{e_high:.2f} 成交"
            if e_low and e_high
            else f"参考限价 ≈ {entry_plan:.2f}（勿市价追）"
        )
        if pos.shares > 0:
            actions_now.append(
                f"按 1R={risk_pct:.1f}% 下单约 {pos.shares} 股（本金 USD {capital:,.0f}）"
            )
        else:
            actions_now.append("先调本金/风险% 或收紧止损，使股数 ≥ 1")
        actions_now.append(f"止损硬挂 {stop:.2f}" if stop else "必须先设止损")
        if t1:
            actions_now.append(f"T1={t1:.2f}：到价减仓 ≥50%，把止损移到成本附近")
        if t2:
            actions_now.append(f"剩余仓位看 T2={t2:.2f}；结构破坏则清")
        if exp_r is not None and rr is not None:
            actions_now.append(f"本结构期望约 {exp_r:+.2f}R · R:R {rr:.2f}（符合满仓门槛）")
        actions_now.append("成交后写日志：理由 / 止损 / 目标 / 是否按计划")
    elif enter_ok == "谨慎试仓":
        actions_now.append(
            f"只用 0.5R（约 {pos.shares} 股）" if pos.shares > 0 else "0.5R 预算不足 1 股：缩小止损距或加本金"
        )
        if e_low and e_high:
            actions_now.append(f"限价只挂 {e_low:.2f}–{e_high:.2f}，上方绝不追")
        actions_now.append(f"止损 {stop:.2f} 必须预设" if stop else "先定止损再考虑")
        if exp_r is not None and rr is not None:
            actions_now.append(f"期望 {exp_r:+.2f}R · R:R {rr:.2f}（未达满仓线，故半仓）")
        if earnings_soon:
            actions_now.append("临近财报：半仓已是上限，隔夜风险自负")
        actions_wait.append("跌破止损或变「回避」→ 立刻停手，不加仓摊平")
        actions_wait.append("只有回到入场区且评分升到「适合」才考虑加到 1R")
    elif enter_ok == "观望":
        actions_now.append("今天不开新仓（赔率/期望/位置未过赚钱门槛）")
        if rr is not None and rr < MIN_RR_CAUTIOUS:
            actions_wait.append(
                f"等更好买点把 R:R 做到 ≥{MIN_RR_CAUTIOUS}（现约 {rr:.2f}）"
            )
        if exp_r is not None and exp_r < 0:
            actions_wait.append(f"现期望 {exp_r:+.2f}R 为负：禁止用「感觉」开仓")
        if e_low and e_high:
            actions_wait.append(f"价格回到 {e_low:.2f}–{e_high:.2f} 再评估")
        if regime is not None and regime.score < 40:
            actions_wait.append("等待大盘环境改善（VIX/SPY 结构）")
        actions_wait.append("不预判抄底、不追已经离开入场区的票")
    else:
        actions_now.append("回避做多；不加仓、不摊平")
        actions_now.append("已有多单：优先检查止损是否仍有效 / 考虑减仓")
        actions_wait.append("等空头结构结束且 R:R/期望重新过线后再做多")

    if vol and getattr(vol, "trend", "") == "放量" and bias.score < 0:
        actions_wait.append("放量偏空：避免在恐慌杀跌中接刀")
    if chase_high or price_above_zone:
        actions_wait.append("靠近高位/在入场区上方：等回踩，FOMO 是亏钱主因之一")
    if liq.score < 40:
        actions_now.append("流动性偏弱：限价、减小单笔、避免市价扫单")
    if vol_cf is not None and vol_cf.label == "放量下跌":
        actions_now.append("放量下跌：今天不做多，等抛压缓和")
    if vol_cf is not None and vol_cf.label == "缩量回踩" and enter_ok in ("适合入场", "谨慎试仓"):
        actions_now.append("缩量回踩：优先限价等回踩区成交，不追阳线")
    if sector_rs is not None and sector_rs.score is not None and sector_rs.score < 40:
        actions_wait.append("弱于板块：优先换同板块强势股，或等个股重夺板块相对强度")
    if iv_reg is not None and iv_reg.high_event_risk:
        actions_now.append("IV 偏高/极高：减仓或等波动回落，警惕事件跳空")
    if fbo is not None and fbo.false_break_risk:
        actions_now.append("假突破信号：今天不追多，等回到入场区再评估")
    elif fbo is not None and fbo.block_breakout_chase:
        actions_wait.append("突破未确认：禁止追高，只允许回踩入场区限价")
    if trend_al is not None and trend_al.against_trend:
        actions_now.append("逆势（大盘/板块偏空）：短线优先空手，不做逆势追多")
    elif trend_al is not None and trend_al.label in ("强跟势", "跟势"):
        if enter_ok in ("适合入场", "谨慎试仓"):
            actions_now.append("跟势环境OK：顺大盘/板块方向在区内做多更稳")
    if weekly is not None and not weekly.allow_long:
        actions_now.append("周线空头：最多试仓/观望，不按强趋势满仓做多")
    if adx_r is not None and not adx_r.trending:
        actions_wait.append("ADX震荡：少追突破，优先回踩入场区限价")
    if h1_r is not None:
        if h1_r.ready and enter_ok in ("适合入场", "谨慎试仓"):
            actions_now.append("1H已转强且在区内：可挂限价（勿市价追）")
        elif h1_r.label == "已遠離":
            actions_wait.append("1H已远离入场区：等回踩再挂，不追高")
        elif h1_r.label == "1H偏空":
            actions_wait.append("1H偏空：等EMA9重新站上EMA21再考虑")
        elif not h1_r.ready:
            actions_wait.append(f"1H触发「{h1_r.label}」：先设好限价在入场区，等1H配合")

    invalidation = entry.invalidation or (
        f"收盘跌破止损 {stop:.2f} 则本计划作废" if stop else "结构破坏则作废"
    )

    reg_bit = ""
    if regime is not None:
        reg_bit = f"市场 {regime.label}（{regime.score:.0f}）；"
    exp_bit = f"期望 {exp_r:+.2f}R；" if exp_r is not None else ""
    # 摘要以「主周期」为准
    h1v = swing_h1.verdict
    h2v = swing_h2.verdict
    summary = (
        f"{sym} 现价 **{last:.2f}** · 主周期 **{primary.label}** → **{primary.verdict}**。\n\n"
        f"{trend_note}\n\n"
        f"**主计划**：入场 {primary.entry_low}–{primary.entry_high}，"
        f"止蚀 {primary.stop_loss}，目标 {primary.target}，"
        f"胜率 {primary.win_rate_pct}% · 纸面R:R {primary.rr} · "
        f"净R:R {primary.rr_net}。\n"
        f"出场：{exit_pl.summary}\n"
        f"（对照 0–2周={h1v} / 2–4周={h2v}）{reg_bit}"
    )

    notes = [
        f"主周期={primary.label}：决定做不做、出场纪律、滑点后R:R",
        "短线波段：0–2周≈10交易日；2–4周≈20交易日",
        "胜率=历史路径（先到目标再触止蚀），非实盘保证",
        f"滑点假设单边 {DEFAULT_SLIP_PCT * 100:.2f}%：净R:R更接近可成交",
        "出场硬规则：T1减半 → 止蚀保本 → 时间止损 → 破止蚀全出",
        "限价入场区内，不追高；成交后写交易日志对照真胜率",
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
        entry_plan=display_limit,  # 区内用现价作限价参考；R:R 仍按区中位算
        stop_loss=round(float(stop), 4) if stop else None,
        target_t1=round(float(t1), 4) if t1 else None,
        target_t2=round(float(t2), 4) if t2 else None,
        win_rate_pct=round(primary.win_rate_pct, 1)
        if primary.win_rate_pct is not None
        else (round(win_rate, 1) if win_rate is not None else None),
        win_rate_label=wr_label,
        stability_score=stab,
        stability_label=stab_label,
        side=side,
        risk_per_share=round(primary.risk_per_share, 4)
        if primary.risk_per_share
        else (round(risk_ps, 4) if risk_ps else None),
        rr_t1=primary.rr if primary.rr is not None else (round(rr, 2) if rr else None),
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
        expectancy_r=primary.expectancy_r
        if primary.expectancy_r is not None
        else exp_r,
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
        sector_rs_summary=getattr(sector_rs, "summary", "") if sector_rs else "",
        sector_rs_score=getattr(sector_rs, "score", None) if sector_rs else None,
        sector_rs_label=getattr(sector_rs, "label", "—") if sector_rs else "—",
        volume_confirm_summary=getattr(vol_cf, "summary", "") if vol_cf else "",
        volume_confirm_score=getattr(vol_cf, "score", None) if vol_cf else None,
        volume_confirm_label=getattr(vol_cf, "label", "—") if vol_cf else "—",
        iv_summary=getattr(iv_reg, "summary", "") if iv_reg else "",
        iv_score=getattr(iv_reg, "score", None) if iv_reg else None,
        iv_label=getattr(iv_reg, "label", "—") if iv_reg else "—",
        iv_high_event=bool(getattr(iv_reg, "high_event_risk", False)) if iv_reg else False,
        swing_h1=swing_h1,
        swing_h2=swing_h2,
        trend_note=trend_note,
        primary_horizon=ph,
        primary_plan=primary,
        exit_plan=exit_pl,
        slip_rr=slip_main,
        false_break_summary=getattr(fbo, "summary", "") if fbo else "",
        false_break_score=getattr(fbo, "score", None) if fbo else None,
        false_break_label=getattr(fbo, "label", "—") if fbo else "—",
        false_break_risk=bool(getattr(fbo, "false_break_risk", False)) if fbo else False,
        trend_align_summary=getattr(trend_al, "summary", "") if trend_al else "",
        trend_align_score=getattr(trend_al, "score", None) if trend_al else None,
        trend_align_label=getattr(trend_al, "label", "—") if trend_al else "—",
        against_trend=bool(getattr(trend_al, "against_trend", False)) if trend_al else False,
        weekly_label=getattr(weekly, "label", "—") if weekly else "—",
        weekly_summary=getattr(weekly, "summary", "") if weekly else "",
        weekly_allow_long=weekly_allow,
        adx_label=getattr(adx_r, "label", "—") if adx_r else "—",
        adx_value=adx_val,
        adx_summary=getattr(adx_r, "summary", "") if adx_r else "",
        adx_trending=bool(adx_trending) if adx_trending is not None else False,
        fib_summary=(getattr(fib_r, "summary", "") if fib_r else "")
        + (f" · {fib_note}" if fib_note else ""),
        h1_label=getattr(h1_r, "label", "—") if h1_r else "—",
        h1_summary=getattr(h1_r, "summary", "") if h1_r else "",
        h1_ready=bool(h1_ready) if h1_ready is not None else False,
    )
