"""Vertical spread builders (bull/bear put/call)."""
from __future__ import annotations

import pandas as pd

from options_chain import (
    HARD_CREDIT_FILL_HI,
    HARD_DEBIT_FILL_HI,
    HARD_MIN_CREDIT,
    _leg_from_row,
    _liquid,
    _nearest_strike,
    _pricing_mode_for_legs,
    _row_nearest_strike,
    _width_ok,
)
from options_models import SpreadIdea

def build_bull_put(
    puts: pd.DataFrame,
    spot: float,
    width: float,
    otm_pct: float = 0.03,
    expiry: str = "",
    dte: int = 0,
) -> SpreadIdea | None:
    """Credit: sell higher put, buy lower put."""
    liq = _liquid(puts)
    if liq.empty:
        return None
    otm = liq[liq["strike"] < spot * 0.999]
    if otm.empty:
        return None
    target = spot * (1 - otm_pct)
    short_row = _nearest_strike(otm, target, "below")
    if short_row is None:
        return None
    short_k = float(short_row["strike"])
    lower = liq[liq["strike"] < short_k]
    if lower.empty:
        return None
    long_row = _row_nearest_strike(lower, short_k - width)
    long_k = float(long_row["strike"])
    if long_k >= short_k:
        return None

    short_leg = _leg_from_row(short_row, "put", "sell", spot)
    long_leg = _leg_from_row(long_row, "put", "buy", spot)
    # 自然成交：卖腿 bid − 买腿 ask（比 mid 更保守）
    credit = short_leg.fill - long_leg.fill
    if credit <= HARD_MIN_CREDIT:
        return None
    w = short_k - long_k
    if not _width_ok(w, width) or credit / w > HARD_CREDIT_FILL_HI:
        return None
    max_profit = credit * 100
    max_loss = (w - credit) * 100
    if max_loss <= 0:
        return None
    be = short_k - credit
    fill_ratio = credit / w
    rr = max_profit / max_loss
    score = min(100.0, fill_ratio * 140 + min(short_leg.oi or 0, 3000) / 3000 * 12 + 40)
    actual_otm = (spot - short_k) / spot * 100
    credit_r = round(credit, 2)
    half_bb = round(credit_r * 0.5, 2)
    max_profit_r = round(max_profit, 2)
    return SpreadIdea(
        name=f"Bull Put Credit · 卖{short_k:.0f}/买{long_k:.0f}",
        code="bull_put",
        structure="Credit Vertical",
        thesis="看多/偏多",
        net_credit=credit_r,
        net_debit=None,
        max_profit=max_profit_r,
        max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)],
        width=round(w, 2),
        pop_est=None,
        score=round(score, 1),
        dte=dte,
        expiry=expiry,
        legs=[short_leg, long_leg],
        notes=[
            f"开仓：卖 {short_k:.0f} Put，买 {long_k:.0f} Put（同一到期）",
            f"净收约 ${credit_r:.2f}/股（卖腿按买价bid、买腿按卖价ask）→ 最大盈 ${max_profit_r:.0f}，最大亏 ${max_loss:.0f}",
            f"打和点 {be:.2f}；到期收盘 > {be:.2f} 有利",
            f"短腿约 {actual_otm:.1f}% OTM",
            f"50%止盈：价差买回约 ${half_bb:.2f}/股（约赚 ${max_profit_r * 0.5:.0f}/张）",
        ],
        risk_reward=round(rr, 2),
        otm_label=f"短腿约 {actual_otm:.1f}% OTM",
        pricing_mode=_pricing_mode_for_legs([short_leg, long_leg]),
        metric_half_buyback=half_bb,
        metric_half_profit=round(max_profit_r * 0.5, 1),
    )


def build_bear_call(
    calls: pd.DataFrame,
    spot: float,
    width: float,
    otm_pct: float = 0.03,
    expiry: str = "",
    dte: int = 0,
) -> SpreadIdea | None:
    """Credit: sell lower call, buy higher call."""
    liq = _liquid(calls)
    if liq.empty:
        return None
    otm = liq[liq["strike"] > spot * 1.001]
    if otm.empty:
        return None
    target = spot * (1 + otm_pct)
    short_row = _nearest_strike(otm, target, "above")
    if short_row is None:
        return None
    short_k = float(short_row["strike"])
    upper = liq[liq["strike"] > short_k]
    if upper.empty:
        return None
    long_row = _row_nearest_strike(upper, short_k + width)
    long_k = float(long_row["strike"])
    if long_k <= short_k:
        return None

    short_leg = _leg_from_row(short_row, "call", "sell", spot)
    long_leg = _leg_from_row(long_row, "call", "buy", spot)
    credit = short_leg.fill - long_leg.fill
    if credit <= HARD_MIN_CREDIT:
        return None
    w = long_k - short_k
    if not _width_ok(w, width) or credit / w > HARD_CREDIT_FILL_HI:
        return None
    max_profit = credit * 100
    max_loss = (w - credit) * 100
    if max_loss <= 0:
        return None
    be = short_k + credit
    fill_ratio = credit / w
    rr = max_profit / max_loss
    score = min(100.0, fill_ratio * 140 + min(short_leg.oi or 0, 3000) / 3000 * 12 + 40)
    actual_otm = (short_k - spot) / spot * 100
    credit_r = round(credit, 2)
    half_bb = round(credit_r * 0.5, 2)
    max_profit_r = round(max_profit, 2)
    return SpreadIdea(
        name=f"Bear Call Credit · 卖{short_k:.0f}/买{long_k:.0f}",
        code="bear_call",
        structure="Credit Vertical",
        thesis="看空/偏空",
        net_credit=credit_r,
        net_debit=None,
        max_profit=max_profit_r,
        max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)],
        width=round(w, 2),
        pop_est=None,
        score=round(score, 1),
        dte=dte,
        expiry=expiry,
        legs=[short_leg, long_leg],
        notes=[
            f"开仓：卖 {short_k:.0f} Call，买 {long_k:.0f} Call",
            f"净收约 ${credit_r:.2f}/股（卖腿bid / 买腿ask）→ 最大盈 ${max_profit_r:.0f}，最大亏 ${max_loss:.0f}",
            f"打和点 {be:.2f}；到期收盘 < {be:.2f} 有利",
            f"短腿约 {actual_otm:.1f}% OTM",
            f"50%止盈：价差买回约 ${half_bb:.2f}/股（约赚 ${max_profit_r * 0.5:.0f}/张）",
        ],
        risk_reward=round(rr, 2),
        otm_label=f"短腿约 {actual_otm:.1f}% OTM",
        pricing_mode=_pricing_mode_for_legs([short_leg, long_leg]),
        metric_half_buyback=half_bb,
        metric_half_profit=round(max_profit_r * 0.5, 1),
    )


def build_bull_call(
    calls: pd.DataFrame,
    spot: float,
    width: float,
    long_offset_pct: float = 0.0,
    expiry: str = "",
    dte: int = 0,
) -> SpreadIdea | None:
    """Debit: buy lower call, sell higher call."""
    liq = _liquid(calls)
    if liq.empty:
        return None
    target = spot * (1 + long_offset_pct)
    long_row = _nearest_strike(liq, target, "any")
    if long_row is None:
        return None
    long_k = float(long_row["strike"])
    upper = liq[liq["strike"] > long_k]
    if upper.empty:
        return None
    short_row = _row_nearest_strike(upper, long_k + width)
    short_k = float(short_row["strike"])
    if short_k <= long_k:
        return None

    long_leg = _leg_from_row(long_row, "call", "buy", spot)
    short_leg = _leg_from_row(short_row, "call", "sell", spot)
    # 自然成交：买腿 ask − 卖腿 bid（借方更贵、更保守）
    debit = long_leg.fill - short_leg.fill
    if debit <= 0.05:
        return None
    w = short_k - long_k
    if not _width_ok(w, width) or debit / w > HARD_DEBIT_FILL_HI:
        return None
    max_profit = (w - debit) * 100
    max_loss = debit * 100
    if max_profit <= 0:
        return None
    be = long_k + debit
    rr = max_profit / max_loss
    fill = 1 - debit / w
    score = min(100.0, fill * 90 + rr * 35 + min(long_leg.oi or 0, 2000) / 2000 * 10)
    debit_r = round(debit, 2)
    w_r = round(w, 2)
    half_bb = round(debit_r + 0.5 * (w_r - debit_r), 2)
    max_profit_r = round(max_profit, 2)
    return SpreadIdea(
        name=f"Bull Call Debit · 买{long_k:.0f}/卖{short_k:.0f}",
        code="bull_call",
        structure="Debit Vertical",
        thesis="看多",
        net_credit=None,
        net_debit=debit_r,
        max_profit=max_profit_r,
        max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)],
        width=w_r,
        pop_est=None,
        score=round(score, 1),
        dte=dte,
        expiry=expiry,
        legs=[long_leg, short_leg],
        pricing_mode=_pricing_mode_for_legs([long_leg, short_leg]),
        metric_half_buyback=half_bb,
        metric_half_profit=round(max_profit_r * 0.5, 1),
        notes=[
            f"开仓：买 {long_k:.0f} Call，卖 {short_k:.0f} Call",
            f"净付约 ${debit_r:.2f}/股（买腿ask / 卖腿bid）→ 最大盈 ${max_profit_r:.0f}，最大亏 ${max_loss:.0f}",
            f"打和点 {be:.2f}；越接近/超过 {short_k:.0f} 利润越大",
            "适合明确看多；比单买 Call 更便宜但封顶盈利",
            f"50%止盈：价差卖出约 ${half_bb:.2f}/股（约赚 ${max_profit_r * 0.5:.0f}/张）",
        ],
        risk_reward=round(rr, 2),
        otm_label=f"长腿行权价 {long_k:.0f}",
    )


def build_bear_put(
    puts: pd.DataFrame,
    spot: float,
    width: float,
    long_offset_pct: float = 0.0,
    expiry: str = "",
    dte: int = 0,
) -> SpreadIdea | None:
    """Debit: buy higher put, sell lower put."""
    liq = _liquid(puts)
    if liq.empty:
        return None
    target = spot * (1 + long_offset_pct)
    long_row = _nearest_strike(liq, target, "any")
    if long_row is None:
        return None
    long_k = float(long_row["strike"])
    lower = liq[liq["strike"] < long_k]
    if lower.empty:
        return None
    short_row = _row_nearest_strike(lower, long_k - width)
    short_k = float(short_row["strike"])
    if short_k >= long_k:
        return None

    long_leg = _leg_from_row(long_row, "put", "buy", spot)
    short_leg = _leg_from_row(short_row, "put", "sell", spot)
    debit = long_leg.fill - short_leg.fill
    if debit <= 0.05:
        return None
    w = long_k - short_k
    if not _width_ok(w, width) or debit / w > HARD_DEBIT_FILL_HI:
        return None
    max_profit = (w - debit) * 100
    max_loss = debit * 100
    if max_profit <= 0:
        return None
    be = long_k - debit
    rr = max_profit / max_loss
    fill = 1 - debit / w
    score = min(100.0, fill * 90 + rr * 35 + min(long_leg.oi or 0, 2000) / 2000 * 10)
    debit_r = round(debit, 2)
    w_r = round(w, 2)
    half_bb = round(debit_r + 0.5 * (w_r - debit_r), 2)
    max_profit_r = round(max_profit, 2)
    return SpreadIdea(
        name=f"Bear Put Debit · 买{long_k:.0f}/卖{short_k:.0f}",
        code="bear_put",
        structure="Debit Vertical",
        thesis="看空",
        net_credit=None,
        net_debit=debit_r,
        max_profit=max_profit_r,
        max_loss=round(max_loss, 2),
        breakevens=[round(be, 2)],
        width=w_r,
        pop_est=None,
        score=round(score, 1),
        dte=dte,
        expiry=expiry,
        legs=[long_leg, short_leg],
        pricing_mode=_pricing_mode_for_legs([long_leg, short_leg]),
        metric_half_buyback=half_bb,
        metric_half_profit=round(max_profit_r * 0.5, 1),
        notes=[
            f"开仓：买 {long_k:.0f} Put，卖 {short_k:.0f} Put",
            f"净付约 ${debit_r:.2f}/股（买腿ask / 卖腿bid）→ 最大盈 ${max_profit_r:.0f}，最大亏 ${max_loss:.0f}",
            f"打和点 {be:.2f}；跌破 {short_k:.0f} 附近接近满盈",
            "适合明确看空；风险有限",
            f"50%止盈：价差卖出约 ${half_bb:.2f}/股（约赚 ${max_profit_r * 0.5:.0f}/张）",
        ],
        risk_reward=round(rr, 2),
        otm_label=f"长腿行权价 {long_k:.0f}",
    )
