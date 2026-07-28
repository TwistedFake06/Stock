"""
实用交易工具：
- 风险仓位计算
- 一键交易计划卡
- 财报 / 除息等事件提醒

仅供学习参考，不构成投资建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from entry_targets import EntryReport, TargetReport


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

@dataclass
class PositionPlan:
    capital: float
    risk_pct: float  # e.g. 1.0 means 1% of capital
    entry_price: float
    stop_price: float
    risk_amount: float  # money at risk
    risk_per_share: float
    shares: int
    position_value: float
    position_pct_of_capital: float
    max_loss_pct: float  # same as risk if filled at entry
    short_target: float | None = None
    medium_target: float | None = None
    reward_short: float | None = None
    reward_medium: float | None = None
    rr_short: float | None = None
    rr_medium: float | None = None
    notes: list[str] = field(default_factory=list)
    valid: bool = True
    error: str = ""


def calc_position(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    short_target: float | None = None,
    medium_target: float | None = None,
    lot_size: int = 1,
) -> PositionPlan:
    """
    Risk-based position size:
      shares = floor( (capital * risk_pct/100) / (entry - stop) / lot_size ) * lot_size
    """
    if capital <= 0 or risk_pct <= 0:
        return PositionPlan(
            capital, risk_pct, entry_price, stop_price, 0, 0, 0, 0, 0, 0,
            valid=False, error="本金与单笔风险%需为正数",
        )
    if entry_price <= 0 or stop_price <= 0:
        return PositionPlan(
            capital, risk_pct, entry_price, stop_price, 0, 0, 0, 0, 0, 0,
            valid=False, error="入场价/止损价无效",
        )
    if stop_price >= entry_price:
        return PositionPlan(
            capital, risk_pct, entry_price, stop_price, 0, 0, 0, 0, 0, 0,
            valid=False, error="做多止损须低于入场价",
        )

    risk_amount = capital * (risk_pct / 100.0)
    risk_per_share = entry_price - stop_price
    raw_shares = risk_amount / risk_per_share
    lot = max(int(lot_size), 1)
    shares = int(raw_shares // lot) * lot

    notes: list[str] = []
    if shares <= 0:
        # Still report theoretical fractional for education
        notes.append(
            f"按风险预算仅可买 {raw_shares:.2f} 股，不足 1 手/1 股："
            f"可提高本金、提高风险%、或收紧止损距离。"
        )
        shares = 0

    position_value = shares * entry_price
    pos_pct = (position_value / capital * 100) if capital else 0
    max_loss_pct = (shares * risk_per_share / capital * 100) if capital and shares else risk_pct

    # Cap warning: if position > 100% capital, reduce to max affordable
    if position_value > capital and entry_price > 0:
        max_shares = int((capital / entry_price) // lot) * lot
        if max_shares < shares:
            notes.append(
                f"按风险可买 {shares} 股，但超过本金，已按本金上限调整为 {max_shares} 股。"
            )
            shares = max_shares
            position_value = shares * entry_price
            pos_pct = position_value / capital * 100
            max_loss_pct = shares * risk_per_share / capital * 100

    reward_short = reward_medium = rr_s = rr_m = None
    if shares > 0 and short_target and short_target > entry_price:
        reward_short = shares * (short_target - entry_price)
        rr_s = (short_target - entry_price) / risk_per_share
    if shares > 0 and medium_target and medium_target > entry_price:
        reward_medium = shares * (medium_target - entry_price)
        rr_m = (medium_target - entry_price) / risk_per_share

    if pos_pct > 40:
        notes.append("仓位占本金偏高（>40%），注意分散与单票风险。")
    if risk_per_share / entry_price > 0.08:
        notes.append("止损距离 >8%，单笔波动大，可考虑缩小风险%或等待更好入场。")
    if rr_s is not None and rr_s < 1.0:
        notes.append(f"短线盈亏比约 {rr_s:.2f} < 1，性价比一般，宜等更好买点或更近止损。")
    if rr_m is not None and rr_m >= 1.5:
        notes.append(f"中线盈亏比约 {rr_m:.2f}，若趋势延续较有吸引力。")

    return PositionPlan(
        capital=capital,
        risk_pct=risk_pct,
        entry_price=entry_price,
        stop_price=stop_price,
        risk_amount=risk_amount,
        risk_per_share=risk_per_share,
        shares=shares,
        position_value=position_value,
        position_pct_of_capital=pos_pct,
        max_loss_pct=max_loss_pct,
        short_target=short_target,
        medium_target=medium_target,
        reward_short=reward_short,
        reward_medium=reward_medium,
        rr_short=rr_s,
        rr_medium=rr_m,
        notes=notes,
        valid=True,
    )


# ---------------------------------------------------------------------------
# Events (earnings / dividend)
# ---------------------------------------------------------------------------

@dataclass
class EventItem:
    name: str
    when: date | None
    days_left: int | None
    level: str  # 高 | 中 | 低 | 信息
    detail: str


@dataclass
class EventReport:
    items: list[EventItem] = field(default_factory=list)
    caution: str = ""
    near_earnings: bool = False
    summary: str = ""


def _to_date(val: Any) -> date | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, (int, float)):
        # unix timestamp
        try:
            ts = float(val)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except Exception:
            return None
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
        except Exception:
            return None
    return None


def analyze_events(info: dict[str, Any], calendar: dict[str, Any] | None = None) -> EventReport:
    today = date.today()
    items: list[EventItem] = []
    calendar = calendar or {}

    # Earnings from calendar list or info timestamps
    earn_dates: list[date] = []
    cal_earn = calendar.get("Earnings Date")
    if isinstance(cal_earn, (list, tuple)):
        for x in cal_earn:
            d = _to_date(x)
            if d:
                earn_dates.append(d)
    elif cal_earn is not None:
        d = _to_date(cal_earn)
        if d:
            earn_dates.append(d)

    for key in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
        d = _to_date(info.get(key))
        if d and d not in earn_dates:
            earn_dates.append(d)

    earn_dates = sorted({d for d in earn_dates if d >= today}) or sorted(
        {d for d in earn_dates if d}
    )
    next_earn = earn_dates[0] if earn_dates else None
    if next_earn:
        days = (next_earn - today).days
        is_est = bool(info.get("isEarningsDateEstimate"))
        level = "高" if 0 <= days <= 14 else "中" if days <= 30 else "信息"
        avg_eps = calendar.get("Earnings Average")
        detail = f"下一次财报约 {next_earn.isoformat()}"
        if is_est:
            detail += "（估计）"
        if avg_eps is not None:
            detail += f"；预期 EPS 约 {avg_eps}"
        if 0 <= days <= 7:
            detail += " · 一周内公布，波动可能放大"
        elif 0 <= days <= 14:
            detail += " · 两周内公布，注意事件风险"
        items.append(
            EventItem("财报日", next_earn, days if days >= 0 else None, level, detail)
        )

    # Dividend
    for label, key in (
        ("除息日", "Ex-Dividend Date"),
        ("派息日", "Dividend Date"),
    ):
        d = _to_date(calendar.get(key) or info.get("exDividendDate" if "除" in label else "dividendDate"))
        if d and d >= today:
            days = (d - today).days
            items.append(
                EventItem(
                    label,
                    d,
                    days,
                    "低" if days > 14 else "中",
                    f"{label} {d.isoformat()}（{days} 天后）",
                )
            )

    near = any(
        it.name == "财报日" and it.days_left is not None and 0 <= it.days_left <= 14
        for it in items
    )
    caution = ""
    if near:
        caution = (
            "⚠️ 财报临近（≤14天）：短线目标不确定性升高。"
            "可减小仓位、避开财报隔夜，或财报后再执行计划。"
        )

    if not items:
        summary = "暂无明确的近期财报/除息日期（部分市场数据不全）。"
    else:
        parts = [f"{it.name} {it.when}" for it in items if it.when]
        summary = "近期事件：" + "；".join(parts) + "。"

    return EventReport(items=items, caution=caution, near_earnings=near, summary=summary)


# ---------------------------------------------------------------------------
# Trade plan card (text)
# ---------------------------------------------------------------------------

@dataclass
class TradePlanCard:
    symbol: str
    title: str
    opportunity: str
    text: str
    bullets: list[str] = field(default_factory=list)


def build_trade_plan_card(
    symbol: str,
    entry: EntryReport,
    targets: TargetReport,
    position: PositionPlan | None = None,
    events: EventReport | None = None,
    name: str = "",
) -> TradePlanCard:
    price = entry.current_price or targets.current_price
    e_lo = entry.suggested_entry_low
    e_hi = entry.suggested_entry_high
    stop = entry.stop_loss or targets.entry_stop
    u_bull = targets.ultra.bull_target if targets.ultra else None
    u_base = targets.ultra.base_target if targets.ultra else None
    u_bear = targets.ultra.bear_target if targets.ultra else None
    s_bull = targets.short.bull_target
    s_base = targets.short.base_target
    s_bear = targets.short.bear_target
    m_bull = targets.medium.bull_target
    m_base = targets.medium.base_target
    m_bear = targets.medium.bear_target

    lines = [
        "======== 交易计划卡 ========",
        f"标的：{symbol}" + (f"（{name}）" if name else ""),
        f"生成日期：{date.today().isoformat()}",
        f"入场评级：{entry.opportunity}（{entry.score:.0f}分 · {entry.side_bias}）",
        f"现价：{price:.2f}" if price else "现价：—",
        "",
        "【入场】",
        f"  关注买区：{e_lo:.2f} ~ {e_hi:.2f}" if e_lo and e_hi else "  买区：—",
        f"  参考入场：{((e_lo or 0)+(e_hi or 0))/2:.2f}" if e_lo and e_hi else "",
        f"  止损参考：{stop:.2f}" if stop else "  止损：—",
        f"  失效条件：{entry.invalidation}" if entry.invalidation else "",
        "",
        "【超短目标 · 约1周内】",
        (
            f"  看多：{u_bull:.2f}（{targets.ultra.upside_pct:+.1f}%）"
            if u_bull and targets.ultra and targets.ultra.upside_pct is not None
            else f"  看多：{u_bull}"
        ),
        f"  中性：{u_base:.2f}" if u_base else "",
        (
            f"  下看：{u_bear:.2f}（{targets.ultra.downside_pct:+.1f}%）"
            if u_bear and targets.ultra and targets.ultra.downside_pct is not None
            else ""
        ),
        "",
        "【短期目标 · 约2周-1月】",
        f"  看多：{s_bull:.2f}（{targets.short.upside_pct:+.1f}%）" if s_bull and targets.short.upside_pct is not None else f"  看多：{s_bull}",
        f"  中性：{s_base:.2f}" if s_base else "",
        f"  下看：{s_bear:.2f}（{targets.short.downside_pct:+.1f}%）" if s_bear and targets.short.downside_pct is not None else "",
        "",
        "【中期目标 · 约1-2月】",
        f"  看多：{m_bull:.2f}（{targets.medium.upside_pct:+.1f}%）" if m_bull and targets.medium.upside_pct is not None else f"  看多：{m_bull}",
        f"  中性：{m_base:.2f}" if m_base else "",
        f"  下看：{m_bear:.2f}（{targets.medium.downside_pct:+.1f}%）" if m_bear and targets.medium.downside_pct is not None else "",
    ]

    if any(x is not None for x in (targets.rr_ultra, targets.rr_short, targets.rr_medium)):
        lines += [
            "",
            "【盈亏比】",
            f"  超短 R:R ≈ {targets.rr_ultra:.2f}" if targets.rr_ultra is not None else "",
            f"  短线 R:R ≈ {targets.rr_short:.2f}" if targets.rr_short is not None else "",
            f"  中线 R:R ≈ {targets.rr_medium:.2f}" if targets.rr_medium is not None else "",
        ]

    if position and position.valid:
        lines += [
            "",
            "【仓位（按风险）】",
            f"  本金：{position.capital:,.0f}",
            f"  单笔风险：{position.risk_pct:.2f}% ≈ {position.risk_amount:,.2f}",
            f"  建议股数：{position.shares}",
            f"  仓位市值：{position.position_value:,.2f}（占本金 {position.position_pct_of_capital:.1f}%）",
            f"  触及止损约亏：{position.max_loss_pct:.2f}%",
        ]
        if position.reward_short is not None:
            lines.append(f"  到短线目标约盈：{position.reward_short:,.2f}")
        if position.reward_medium is not None:
            lines.append(f"  到中线目标约盈：{position.reward_medium:,.2f}")

    if events and events.items:
        lines += ["", "【事件】"]
        for it in events.items:
            extra = f"（{it.days_left}天后）" if it.days_left is not None else ""
            lines.append(f"  {it.name}：{it.when}{extra} · {it.detail}")
        if events.caution:
            lines.append(f"  {events.caution}")

    lines += [
        "",
        "【执行纪律】",
        "  1. 只在买区内分批；不追高突破失控行情",
        "  2. 止损触发坚决离场，不随意下移止损",
        "  3. 一周目标(T1)先减30–40%，短期(T2)再减，中期(T3)清/跟踪",
        "  4. 财报前评估是否持仓过夜",
        "",
        "※ 仅供学习研究，不构成投资建议",
        "============================",
    ]
    text = "\n".join(x for x in lines if x is not None)

    bullets = [
        f"评级 {entry.opportunity}",
        f"买区 {e_lo}–{e_hi}" if e_lo and e_hi else "买区 —",
        f"止损 {stop}" if stop else "止损 —",
        f"一周 {u_bull}" if u_bull else "一周 —",
        f"短目标 {s_bull}" if s_bull else "短目标 —",
        f"中目标 {m_bull}" if m_bull else "中目标 —",
    ]
    if position and position.shares:
        bullets.append(f"建议 {position.shares} 股")

    return TradePlanCard(
        symbol=symbol,
        title=f"{symbol} 交易计划",
        opportunity=entry.opportunity,
        text=text,
        bullets=bullets,
    )


def suggest_lot_size(symbol: str) -> int:
    """A-shares often trade in lots of 100; HK sometimes 100/500; US 1."""
    s = symbol.upper()
    if s.endswith(".SS") or s.endswith(".SZ"):
        return 100
    if s.endswith(".HK"):
        return 100
    return 1
