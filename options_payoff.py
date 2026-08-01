"""
Vertical 盈亏表 + 按日标价日历。

独立模块，避免 Streamlit 缓存旧 options_spreads 导致 ImportError。
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import pandas as pd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _leg_entry_price(leg: Any) -> float:
    """
    Entry premium per share used for P&L.

    Prefer natural fill (sell@bid / buy@ask) so ladder matches KPI max profit/loss;
    fall back to mid when fill is missing or zero.
    """
    fill = getattr(leg, "fill", None)
    try:
        if fill is not None and float(fill) > 0:
            return float(fill)
    except (TypeError, ValueError):
        pass
    try:
        mid = float(getattr(leg, "mid", 0) or 0)
    except (TypeError, ValueError):
        mid = 0.0
    return mid


def payoff_per_share(idea: Any, underlying: float) -> float:
    """到期结算盈亏（$/股，未×100）。不行权也适用。"""
    pnl = 0.0
    for leg in idea.legs:
        entry = _leg_entry_price(leg)
        if leg.side == "buy":
            pnl -= entry
        else:
            pnl += entry
        if leg.right == "call":
            intrinsic = max(underlying - leg.strike, 0.0)
        else:
            intrinsic = max(leg.strike - underlying, 0.0)
        if leg.side == "buy":
            pnl += intrinsic
        else:
            pnl -= intrinsic
    return float(pnl)


def payoff_per_contract(idea: Any, underlying: float) -> float:
    return payoff_per_share(idea, underlying) * 100.0


def build_payoff_ladder(
    idea: Any,
    spot: float,
    step: float | None = None,
    pad: float | None = None,
) -> pd.DataFrame:
    strikes = [lg.strike for lg in idea.legs]
    be_list = list(idea.breakevens or [])
    key_levels = sorted(set(strikes + be_list + [spot]))

    if step is None:
        w = idea.width or 5.0
        if w >= 10:
            step = 5.0
        elif w >= 5:
            step = 2.0
        elif w >= 2:
            step = 1.0
        else:
            step = 0.5

    lo = min(key_levels) - (pad if pad is not None else max(idea.width * 1.5, step * 3))
    hi = max(key_levels) + (pad if pad is not None else max(idea.width * 1.5, step * 3))

    prices: set[float] = set()
    x = lo
    while x <= hi + 1e-9:
        prices.add(round(x / step) * step)
        x += step
    for k in key_levels:
        prices.add(round(float(k), 2))
    prices.add(round(float(spot), 2))

    rows = []
    for px in sorted(prices):
        pnl = payoff_per_contract(idea, px)
        tags = []
        tol = max(step * 0.51, 0.02)
        if any(abs(px - s) <= tol for s in strikes):
            tags.append("行权价")
        if any(abs(px - b) <= tol for b in be_list):
            tags.append("打和点")
        if abs(px - spot) <= tol:
            tags.append("现价")

        if pnl > 0.5:
            result = "賺"
        elif pnl < -0.5:
            result = "蝕"
        else:
            result = "打和"

        zone = ""
        if abs(pnl - idea.max_profit) < 1.0:
            zone = "最大利润区"
        elif abs(pnl + idea.max_loss) < 1.0:
            zone = "最大亏损区"

        rows.append(
            {
                "标的价": round(px, 2),
                "相对现价%": round((px / spot - 1) * 100, 2) if spot else None,
                "到期盈亏$/张": round(pnl, 2),
                "结果": result,
                "区间": zone,
                "标记": " · ".join(tags) if tags else "",
            }
        )
    return pd.DataFrame(rows)


def payoff_zones_summary(idea: Any, spot: float) -> dict[str, Any]:
    strikes = sorted({lg.strike for lg in idea.legs})
    be = idea.breakevens[0] if idea.breakevens else None
    code = idea.code

    low_px = min(strikes) - idea.width
    high_px = max(strikes) + idea.width
    pnl_low = payoff_per_contract(idea, low_px)
    pnl_high = payoff_per_contract(idea, high_px)

    lines: list[str] = []
    lines.append("说明：一般不需要手动行权；到期结算或提前平仓，盈亏看标的价与打和点。")

    if code == "bull_put":
        short_k = max(lg.strike for lg in idea.legs if lg.side == "sell")
        long_k = min(lg.strike for lg in idea.legs if lg.side == "buy")
        lines.append(f"结构：Bull Put Credit（卖 {short_k:.0f} Put / 买 {long_k:.0f} Put）")
        if be is not None:
            lines.append(f"**賺**：到期标的 **> {be:.2f}**（打和点上方）")
            lines.append(f"**蝕**：到期标的 **< {be:.2f}**")
        lines.append(f"**最大利潤**：标的 ≥ {short_k:.0f} ≈ ${idea.max_profit:.0f}/张")
        lines.append(f"**最大亏损**：标的 ≤ {long_k:.0f} ≈ ${idea.max_loss:.0f}/张")
    elif code == "bear_call":
        short_k = min(lg.strike for lg in idea.legs if lg.side == "sell")
        long_k = max(lg.strike for lg in idea.legs if lg.side == "buy")
        lines.append(f"结构：Bear Call Credit（卖 {short_k:.0f} Call / 买 {long_k:.0f} Call）")
        if be is not None:
            lines.append(f"**賺**：到期标的 **< {be:.2f}**（打和点下方）")
            lines.append(f"**蝕**：到期标的 **> {be:.2f}**")
        lines.append(f"**最大利潤**：标的 ≤ {short_k:.0f} ≈ ${idea.max_profit:.0f}/张")
        lines.append(f"**最大亏损**：标的 ≥ {long_k:.0f} ≈ ${idea.max_loss:.0f}/张")
    elif code == "bull_call":
        long_k = min(lg.strike for lg in idea.legs if lg.side == "buy")
        short_k = max(lg.strike for lg in idea.legs if lg.side == "sell")
        lines.append(f"结构：Bull Call Debit（买 {long_k:.0f} Call / 卖 {short_k:.0f} Call）")
        if be is not None:
            lines.append(f"**賺**：到期标的 **> {be:.2f}**")
            lines.append(f"**蝕**：到期标的 **< {be:.2f}**")
        lines.append(f"**最大利潤**：标的 ≥ {short_k:.0f} ≈ ${idea.max_profit:.0f}/张")
        lines.append(f"**最大亏损**：标的 ≤ {long_k:.0f} ≈ ${idea.max_loss:.0f}/张")
    elif code == "bear_put":
        long_k = max(lg.strike for lg in idea.legs if lg.side == "buy")
        short_k = min(lg.strike for lg in idea.legs if lg.side == "sell")
        lines.append(f"结构：Bear Put Debit（买 {long_k:.0f} Put / 卖 {short_k:.0f} Put）")
        if be is not None:
            lines.append(f"**賺**：到期标的 **< {be:.2f}**")
            lines.append(f"**蝕**：到期标的 **> {be:.2f}**")
        lines.append(f"**最大利潤**：标的 ≤ {short_k:.0f} ≈ ${idea.max_profit:.0f}/张")
        lines.append(f"**最大亏损**：标的 ≥ {long_k:.0f} ≈ ${idea.max_loss:.0f}/张")
    else:
        if be is not None:
            if pnl_high > pnl_low:
                lines.append(f"**賺**倾向标的走高；打和点 ≈ {be:.2f}")
            else:
                lines.append(f"**賺**倾向标的走低；打和点 ≈ {be:.2f}")

    lines.append(
        f"现价 {spot:.2f} 若到期不变 → 盈亏约 ${payoff_per_contract(idea, spot):.0f}/张"
    )
    return {
        "lines": lines,
        "breakeven": be,
        "spot_pnl": round(payoff_per_contract(idea, spot), 2),
        "max_profit": idea.max_profit,
        "max_loss": idea.max_loss,
    }


def bs_option_price(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    right: str,
    r: float = 0.04,
) -> float:
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return 0.0
    if t_years <= 1e-8:
        if right == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    vol = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / vol
    d2 = d1 - vol
    if right == "call":
        return float(
            spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
        )
    return float(
        strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    )


def spread_mark_value(
    idea: Any,
    spot: float,
    dte_left: int,
    sigma: float,
    r: float = 0.04,
) -> float:
    """价差市值标价 $/股（未×100）。"""
    t = max(dte_left, 0) / 365.0
    sig = max(0.08, min(float(sigma), 0.9))
    total = 0.0
    for leg in idea.legs:
        px = bs_option_price(spot, leg.strike, t, sig, leg.right, r=r)
        if leg.side == "buy":
            total += px
        else:
            total -= px
    if getattr(idea, "net_credit", None) is not None:
        return float(-total)
    return float(total)


def build_daily_mark_calendar(
    idea: Any,
    spot: float,
    sigma: float | None,
    dte_total: int | None = None,
    hold_days: int | None = None,
    spot_path: str = "flat",
    r: float = 0.04,
) -> pd.DataFrame:
    """
    从今天(第0日)起逐日估算价差标价与相对入场盈亏。
    spot_path: flat | +1% | -1% | +2% | -2% | ...
    """
    dte0 = int(dte_total if dte_total is not None else getattr(idea, "dte", None) or 30)
    dte0 = max(dte0, 1)
    n = int(hold_days if hold_days is not None else dte0)
    n = max(1, min(n, dte0))

    sig = sigma if sigma and sigma > 0 else 0.20
    sig = max(0.08, min(float(sig), 0.9))

    if getattr(idea, "net_credit", None) is not None:
        entry_mark = float(idea.net_credit)
    elif getattr(idea, "net_debit", None) is not None:
        entry_mark = float(idea.net_debit)
    else:
        entry_mark = abs(spread_mark_value(idea, spot, dte0, sig, r=r))

    if spot_path == "flat":
        end_spot = spot
    elif spot_path.startswith("+"):
        end_spot = spot * (1 + float(spot_path[1:].replace("%", "")) / 100.0)
    elif spot_path.startswith("-"):
        end_spot = spot * (1 - float(spot_path[1:].replace("%", "")) / 100.0)
    else:
        end_spot = spot

    today = date.today()
    rows = []
    for day in range(0, n + 1):
        dte_left = max(dte0 - day, 0)
        s_t = spot + (end_spot - spot) * (day / n) if n > 0 else spot
        mark = max(spread_mark_value(idea, s_t, dte_left, sig, r=r), 0.0)

        is_credit = getattr(idea, "net_credit", None) is not None
        if is_credit:
            # 卖出价差：今天收 entry，第N日买回付 mark
            pnl = (entry_mark - mark) * 100.0
            sell_today = entry_mark
            buy_back = mark
            action = "卖出后再买回"
        else:
            # 买进价差：今天付 entry，第N日卖出收 mark
            pnl = (mark - entry_mark) * 100.0
            sell_today = entry_mark  # 其实是买入成本
            buy_back = mark
            action = "买进后再卖出"

        if pnl > 0.5:
            result = "賺"
        elif pnl < -0.5:
            result = "蝕"
        else:
            result = "打和"

        rows.append(
            {
                "第几天后": day,
                "日期": (today + timedelta(days=day)).isoformat(),
                "还剩几天到期": dte_left,
                "股票大约价": round(s_t, 2),
                "比今天高/低%": round((s_t / spot - 1) * 100, 2),
                "今天卖出价差收$/股" if is_credit else "今天买进价差付$/股": round(
                    entry_mark, 2
                ),
                "这天买回要付$/股" if is_credit else "这天卖出收回$/股": round(mark, 2),
                "差价$/股": round(
                    (entry_mark - mark) if is_credit else (mark - entry_mark), 2
                ),
                "若这天平仓赚亏$/张": round(pnl, 2),
                "赚或蚀": result,
                "做法": action,
            }
        )
    return pd.DataFrame(rows)


# 兼容：也从 options_plain 再导出（避免旧缓存）
try:
    from options_plain import plain_spread_steps  # noqa: F401
except Exception:  # pragma: no cover
    def plain_spread_steps(idea: Any) -> list[str]:
        return ["今天卖出价差先收钱，几天后买回平仓。"]
