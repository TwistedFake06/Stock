"""
After you bought: given fill price + current plan, suggest Hold / Take-profit / Stop.

Uses short-term swing structure (stop, T1, T2, max hold days) and live last price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class PositionAdvice:
    action: str  # 持有 | 持有·上移止蚀 | 止盈减仓 | 止盈清仓 | 止蚀离场 | 观望/无效
    color: str  # green | amber | red | gray
    headline: str
    pnl_pct: float | None
    pnl_r: float | None  # vs plan risk (entry-stop) if stop known
    dist_to_stop_pct: float | None
    dist_to_t1_pct: float | None
    dist_to_t2_pct: float | None
    suggested_stop: float | None  # where to place stop now
    bullets: list[str] = field(default_factory=list)
    summary: str = ""


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


def _pct(a: float, b: float) -> float:
    """(a/b - 1) * 100"""
    if b == 0:
        return 0.0
    return (a / b - 1.0) * 100.0


def advise_open_position(
    *,
    buy_price: float,
    last_price: float,
    plan_stop: float | None = None,
    plan_t1: float | None = None,
    plan_t2: float | None = None,
    plan_entry: float | None = None,
    max_hold_days: int | None = None,
    buy_date: str | None = None,
    shares: int | None = None,
    bias_label: str = "",
    bias_score: float = 0.0,
) -> PositionAdvice:
    """
    Long-only coach for an already-filled position.
    """
    buy = _f(buy_price)
    last = _f(last_price)
    stop = _f(plan_stop)
    t1 = _f(plan_t1)
    t2 = _f(plan_t2)
    plan_e = _f(plan_entry)

    if buy is None or buy <= 0 or last is None or last <= 0:
        return PositionAdvice(
            action="观望/无效",
            color="gray",
            headline="请输入有效买入价与当前价",
            pnl_pct=None,
            pnl_r=None,
            dist_to_stop_pct=None,
            dist_to_t1_pct=None,
            dist_to_t2_pct=None,
            suggested_stop=None,
            bullets=["买入价/现价无效"],
            summary="无法评估。",
        )

    pnl_pct = _pct(last, buy)
    risk = None
    if stop is not None and buy > stop:
        risk = buy - stop
    elif plan_e is not None and stop is not None and plan_e > stop:
        # fallback risk from plan structure
        risk = plan_e - stop
    pnl_r = (last - buy) / risk if risk and risk > 0 else None

    dist_stop = _pct(last, stop) if stop else None  # how far above stop (positive = room)
    dist_t1 = _pct(t1, last) if t1 else None  # upside to t1 from last
    dist_t2 = _pct(t2, last) if t2 else None

    bullets: list[str] = []
    bullets.append(f"买入 {buy:.2f} · 现价 {last:.2f} · 浮盈 {pnl_pct:+.2f}%")
    if pnl_r is not None:
        bullets.append(f"相对计划风险约 {pnl_r:+.2f}R")
    if stop is not None:
        bullets.append(f"计划止蚀 {stop:.2f}（现价距止蚀 {dist_stop:+.2f}%）" if dist_stop is not None else f"计划止蚀 {stop:.2f}")
    if t1 is not None:
        bullets.append(f"T1 {t1:.2f}" + (f"（距 T1 还有 {dist_t1:+.2f}%）" if dist_t1 is not None else ""))
    if t2 is not None:
        bullets.append(f"T2 {t2:.2f}" + (f"（距 T2 还有 {dist_t2:+.2f}%）" if dist_t2 is not None else ""))

    # Holding days
    held_days = None
    if buy_date:
        try:
            d0 = date.fromisoformat(str(buy_date)[:10])
            held_days = (date.today() - d0).days
            bullets.append(f"已持有约 {held_days} 自然日")
        except Exception:
            held_days = None

    time_stop_hit = False
    if max_hold_days and held_days is not None:
        # rough: calendar days ~ trading days * 1.4
        approx_trading = held_days * 5 / 7
        if approx_trading >= max_hold_days:
            time_stop_hit = True
            bullets.append(f"已超过时间止损窗口（计划最多约 {max_hold_days} 交易日）")

    # Bias pressure
    bias_bear = "看空" in (bias_label or "") or (bias_score is not None and bias_score <= -18)
    if bias_label:
        bullets.append(f"当前多空：{bias_label}（{bias_score:+.0f}）")

    # --- Decision tree (long) ---
    action = "持有"
    color = "green"
    headline = ""
    suggested_stop = stop

    # 1) Hard stop: price at/below stop
    if stop is not None and last <= stop * 1.002:
        action = "止蚀离场"
        color = "red"
        headline = f"现价已触及/跌破止蚀 {stop:.2f} → 按纪律离场，不加仓摊平"
        suggested_stop = stop

    # 2) Time stop
    elif time_stop_hit and t1 is not None and last < t1:
        action = "止盈减仓" if pnl_pct > 0 else "止蚀离场"
        if pnl_pct > 1:
            action = "止盈减仓"
            color = "amber"
            headline = f"时间到且未稳站 T1：建议至少减半或清仓锁定（浮盈 {pnl_pct:+.1f}%）"
        elif pnl_pct > 0:
            action = "止盈清仓"
            color = "amber"
            headline = "时间止损触发且利润有限：建议清仓，把资金留给更好结构"
        else:
            action = "止蚀离场"
            color = "red"
            headline = "时间止损触发且浮亏：建议离场，避免死扛"

    # 3) At/above T2
    elif t2 is not None and last >= t2 * 0.998:
        action = "止盈清仓"
        color = "amber"
        headline = f"已到/超过 T2 {t2:.2f} → 建议清仓或留极小仓 + 紧止蚀"
        suggested_stop = max(buy * 1.001, last * 0.97) if buy else last * 0.97

    # 4) At/above T1
    elif t1 is not None and last >= t1 * 0.998:
        action = "止盈减仓"
        color = "amber"
        headline = f"已到/超过 T1 {t1:.2f} → 减仓约 50%，止蚀上移至保本"
        suggested_stop = round(buy * 1.001, 4)  # break-even+
        bullets.append(f"建议新止蚀 ≈ {suggested_stop:.2f}（保本）")
        if t2 is not None:
            bullets.append(f"剩仓看 T2 {t2:.2f}；未到则保本止蚀保护")

    # 5) Between BE and T1 with open profit — trail slightly
    elif pnl_pct >= 3.0 and (t1 is None or last < t1 * 0.99):
        action = "持有·上移止蚀"
        color = "green"
        # trail stop: max(plan stop, buy, last - 40% of open profit distance)
        trail = buy
        if stop is not None:
            trail = max(stop, buy)
        # lock some profit: stop at 40% of the way from buy to last
        lock = buy + 0.4 * (last - buy)
        suggested_stop = round(max(trail, lock), 4)
        headline = f"已有浮盈 {pnl_pct:+.1f}%：继续持有，止蚀上移锁利"
        bullets.append(f"建议止蚀上移至 ≈ {suggested_stop:.2f}")

    # 6) Near stop but not hit — tighten alert
    elif stop is not None and dist_stop is not None and 0 < dist_stop < 1.5:
        action = "持有"
        color = "amber"
        headline = f"接近止蚀（仅剩 {dist_stop:.1f}% 空间）：准备执行，勿幻想反转摊平"
        suggested_stop = stop

    # 7) Deep red without hitting stop yet — bias matters
    elif pnl_pct <= -5.0 or (pnl_r is not None and pnl_r <= -0.7):
        if bias_bear:
            action = "止蚀离场"
            color = "red"
            headline = f"浮亏 {pnl_pct:+.1f}% 且方向转空 → 建议提前离场，不必等打止蚀"
            suggested_stop = last  # market exit
        else:
            action = "持有"
            color = "amber"
            headline = f"浮亏 {pnl_pct:+.1f}%：方向未完全转空，仍守计划止蚀 {stop if stop else '—'}"
            suggested_stop = stop

    # 8) Default hold
    else:
        action = "持有"
        color = "green"
        if pnl_pct >= 0:
            headline = f"浮盈 {pnl_pct:+.1f}%：按计划持有，止蚀不要下移"
        else:
            headline = f"浮亏 {pnl_pct:+.1f}%：未破止蚀则持有，禁止摊平"
        suggested_stop = stop if stop is not None else round(buy * 0.97, 4)

    if shares and shares > 0 and pnl_pct is not None:
        pnl_usd = shares * (last - buy)
        bullets.append(f"约 {shares} 股 · 浮动盈亏 ≈ ${pnl_usd:+,.2f}")

    summary = f"**{action}** · {headline}"
    return PositionAdvice(
        action=action,
        color=color,
        headline=headline,
        pnl_pct=round(pnl_pct, 2),
        pnl_r=round(pnl_r, 3) if pnl_r is not None else None,
        dist_to_stop_pct=round(dist_stop, 2) if dist_stop is not None else None,
        dist_to_t1_pct=round(dist_t1, 2) if dist_t1 is not None else None,
        dist_to_t2_pct=round(dist_t2, 2) if dist_t2 is not None else None,
        suggested_stop=suggested_stop,
        bullets=bullets,
        summary=summary,
    )
