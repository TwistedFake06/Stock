"""
实用辅助分析：下一周观察点、关键价位合流、乖离/节奏提示、分批止盈建议。

仅供学习参考，不构成投资建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from analysis import BiasReport
from entry_targets import EntryReport, TargetReport
from indicators import enrich
from trade_plan import EventReport


def _safe(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class WatchItem:
    kind: str  # 价格 | 指标 | 事件 | 纪律
    level: str  # 高 | 中 | 低
    title: str
    detail: str


@dataclass
class TakeProfitStep:
    label: str
    price: float | None
    action: str
    pct_from_entry: float | None = None


@dataclass
class HelperReport:
    one_liner: str
    playbook: list[str] = field(default_factory=list)
    watchlist: list[WatchItem] = field(default_factory=list)
    key_levels: list[dict[str, Any]] = field(default_factory=list)
    take_profits: list[TakeProfitStep] = field(default_factory=list)
    extension_note: str = ""
    week_focus: str = ""
    summary: str = ""


def analyze_helpers(
    df: pd.DataFrame,
    entry: EntryReport,
    targets: TargetReport,
    bias: BiasReport | None = None,
    events: EventReport | None = None,
) -> HelperReport:
    if df is None or df.empty or "Close" not in df.columns:
        return HelperReport(one_liner="数据不足", summary="无法生成辅助分析。")

    data = enrich(df)
    close = data["Close"].astype(float)
    last = float(close.iloc[-1])
    high = data["High"].astype(float) if "High" in data.columns else close
    low = data["Low"].astype(float) if "Low" in data.columns else close

    atr = None
    if len(data) >= 15:
        prev = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
            axis=1,
        ).max(axis=1)
        atr = float(tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1])
    atr = atr or last * 0.02

    sma5 = _safe(data["SMA5"].iloc[-1]) if "SMA5" in data.columns else None
    sma20 = _safe(data["SMA20"].iloc[-1]) if "SMA20" in data.columns else None
    rsi = _safe(data["RSI"].iloc[-1]) if "RSI" in data.columns else None
    h5 = float(high.tail(5).max())
    l5 = float(low.tail(5).min())
    h20 = float(high.tail(20).max())
    l20 = float(low.tail(20).min())

    # Extension from SMA20 in ATR units
    ext_note = ""
    if sma20 and atr > 0:
        dist = (last - sma20) / atr
        if dist >= 2.5:
            ext_note = f"价格高于 SMA20 约 {dist:.1f}×ATR，短线偏「伸展」，回调概率升高，不宜追涨。"
        elif dist >= 1.2:
            ext_note = f"价格高于 SMA20 约 {dist:.1f}×ATR，趋势偏强但仍可等回踩再加。"
        elif dist <= -2.0:
            ext_note = f"价格低于 SMA20 约 {abs(dist):.1f}×ATR，超跌反弹可期，但需确认止跌。"
        elif dist <= -1.0:
            ext_note = f"价格低于 SMA20 约 {abs(dist):.1f}×ATR，偏弱，反弹先当反抽。"
        else:
            ext_note = f"价格贴近 SMA20（{dist:+.1f}×ATR），位置中性，适合按买区纪律操作。"

    # Key levels near price (confluence helper)
    candidates: list[tuple[str, float, str]] = []
    if entry.suggested_entry_low:
        candidates.append(("买区下沿", entry.suggested_entry_low, "入场"))
    if entry.suggested_entry_high:
        candidates.append(("买区上沿", entry.suggested_entry_high, "入场"))
    if entry.stop_loss:
        candidates.append(("止损", entry.stop_loss, "风控"))
    if sma5:
        candidates.append(("SMA5", sma5, "均线"))
    if sma20:
        candidates.append(("SMA20", sma20, "均线"))
    candidates += [
        ("近5日高", h5, "波段"),
        ("近5日低", l5, "波段"),
        ("近20日高", h20, "波段"),
        ("近20日低", l20, "波段"),
    ]
    if targets.ultra and targets.ultra.bull_target:
        candidates.append(("一周上看", targets.ultra.bull_target, "目标"))
    if targets.ultra and targets.ultra.bear_target:
        candidates.append(("一周下看", targets.ultra.bear_target, "目标"))
    if targets.short.bull_target:
        candidates.append(("短期上看", targets.short.bull_target, "目标"))
    if targets.medium.bull_target:
        candidates.append(("中期上看", targets.medium.bull_target, "目标"))

    key_levels = []
    for name, px, cat in candidates:
        if px is None or px <= 0:
            continue
        dist_pct = (px / last - 1) * 100
        if abs(dist_pct) > 25:
            continue
        key_levels.append(
            {
                "名称": name,
                "类别": cat,
                "价格": round(px, 2),
                "距现价%": round(dist_pct, 2),
                "距离(ATR)": round(abs(px - last) / atr, 2) if atr else None,
            }
        )
    key_levels.sort(key=lambda x: abs(x["距现价%"]))

    # Watch items for next week
    watch: list[WatchItem] = []
    if entry.suggested_entry_low and entry.suggested_entry_high:
        watch.append(
            WatchItem(
                "价格",
                "高",
                "是否回踩买区",
                f"关注 {entry.suggested_entry_low:.2f}–{entry.suggested_entry_high:.2f}；"
                "触及且不破止损再考虑分批。",
            )
        )
    watch.append(
        WatchItem(
            "价格",
            "高",
            "一周高低点",
            f"上破近5日高 {h5:.2f} 可看作超短动能；跌破近5日低 {l5:.2f} 宜减仓/观望。",
        )
    )
    if targets.ultra and targets.ultra.bull_target:
        watch.append(
            WatchItem(
                "价格",
                "中",
                "一周目标",
                f"上看约 {targets.ultra.bull_target:.2f}"
                f"（{targets.ultra.upside_pct:+.1f}%）可作第一止盈参考；"
                f"下看约 {targets.ultra.bear_target}。",
            )
        )
    if rsi is not None:
        if rsi >= 70:
            watch.append(
                WatchItem("指标", "高", "RSI 超买", f"RSI={rsi:.1f}，一周内优先止盈/不追高。")
            )
        elif rsi <= 30:
            watch.append(
                WatchItem("指标", "高", "RSI 超卖", f"RSI={rsi:.1f}，关注止跌阳线再动手。")
            )
        else:
            watch.append(
                WatchItem("指标", "低", "RSI 中性", f"RSI={rsi:.1f}，以价格结构与买区为主。")
            )
    if events and events.near_earnings:
        watch.append(
            WatchItem(
                "事件",
                "高",
                "财报窗口",
                events.caution or "一周内可能有财报，控制隔夜仓位。",
            )
        )
    elif events and events.items:
        for it in events.items[:2]:
            if it.days_left is not None and it.days_left <= 21:
                watch.append(
                    WatchItem(
                        "事件",
                        it.level,
                        it.name,
                        f"{it.when}（{it.days_left}天后）· {it.detail}",
                    )
                )
    watch.append(
        WatchItem(
            "纪律",
            "中",
            "止损不挪",
            f"参考止损 {entry.stop_loss}；收盘有效跌破则离场，不因「再等一天」下移。",
        )
    )

    # Take profit ladder
    entry_px = None
    if entry.suggested_entry_low and entry.suggested_entry_high:
        entry_px = (entry.suggested_entry_low + entry.suggested_entry_high) / 2
    entry_px = entry_px or last

    def _step(label, price, action) -> TakeProfitStep:
        pct = (price / entry_px - 1) * 100 if price and entry_px else None
        return TakeProfitStep(label, price, action, round(pct, 2) if pct is not None else None)

    tps: list[TakeProfitStep] = []
    if targets.ultra and targets.ultra.bull_target:
        tps.append(
            _step(
                "T1 超短（≤1周）",
                targets.ultra.bull_target,
                "减仓约 30%–40%，锁定部分利润",
            )
        )
    if targets.short.bull_target:
        tps.append(
            _step(
                "T2 短期（2周–1月）",
                targets.short.bull_target,
                "再减约 30%–40%，止损可上移至成本附近",
            )
        )
    if targets.medium.bull_target:
        tps.append(
            _step(
                "T3 中期（1–2月）",
                targets.medium.bull_target,
                "剩余仓位止盈或改跟踪止损",
            )
        )
    if entry.stop_loss:
        tps.append(
            _step("止损", entry.stop_loss, "触发则清仓，本笔计划结束")
        )

    # Playbook by opportunity
    opp = entry.opportunity
    if opp == "较佳入场":
        playbook = [
            "可在买区内分 2 批建仓（例如 50%+50%）。",
            "优先看一周目标 T1 作第一止盈，避免贪心扛波动。",
            "若放量突破近5日高且站稳，第二批可在回踩时补。",
            "严格按风险%算股数，单笔亏损预设可接受。",
        ]
        one_liner = "较佳窗口：买区内分批，一周先看 T1，设好止损。"
        week_focus = (
            f"本周焦点：回踩买区后的反弹能否挑战 {targets.ultra.bull_target if targets.ultra else h5}；"
            f"守住 {entry.stop_loss}。"
        )
    elif opp == "可关注":
        playbook = [
            "只轻仓试探（风险预算可砍半）。",
            "必须等价格进入买区或出现明确止跌K线。",
            "一周内若到 T1 优先减仓，不急于加到满仓。",
            "信号转弱（跌破5日低/死叉）立即退出试探仓。",
        ]
        one_liner = "可关注：轻仓等买区，一周目标见好就收。"
        week_focus = "本周焦点：是否出现回踩+止跌；未进买区不追。"
    elif opp == "不宜追高":
        playbook = [
            "不做追涨；把一周下看/SMA20 当作观察回踩位。",
            "若已有仓位，可借反弹减仓至舒适仓。",
            "等 RSI 回落或价格回到买区再重新评估。",
            "突破新高但量能不足时，优先观望。",
        ]
        one_liner = "不宜追高：等回踩或冷却，一周以风控为主。"
        week_focus = f"本周焦点：能否回落到买区/SMA20；上方压力 {h5:.2f}–{h20:.2f}。"
    elif opp == "偏空回避":
        playbook = [
            "默认不做多；反弹当作减仓/观望机会。",
            "只有极轻仓反弹短线且严格止损才可试错（进阶）。",
            "关注是否放量跌破近20日低，若跌破继续观望。",
            "等均线重新多排或结构转升再开新计划。",
        ]
        one_liner = "偏空：回避做多，本周以观望与保护资金为主。"
        week_focus = f"本周焦点：{l5:.2f} / {l20:.2f} 支撑是否失守。"
    else:
        playbook = [
            "方向不明：不急开仓，缩小观察列表即可。",
            "写下触发条件：进入买区 或 放量突破5日高 才动手。",
            "一周内以学习盘感和记录关键价位为主。",
            "有事件（财报）则直接跳过开仓。",
        ]
        one_liner = "观望：等买区或突破确认，本周不硬做。"
        week_focus = f"本周焦点：{entry.suggested_entry_low}–{entry.suggested_entry_high} 买区，或站上 {h5:.2f}。"

    if bias:
        playbook.append(
            f"技术多空参考：{bias.bias}（{bias.score:+.0f}分，置信度{bias.confidence}），"
            "与入场评级交叉验证。"
        )

    summary = (
        f"{one_liner} {ext_note} "
        f"超短目标 "
        f"{targets.ultra.bull_target if targets.ultra else '—'} / "
        f"短期 {targets.short.bull_target} / "
        f"中期 {targets.medium.bull_target}。"
    )

    return HelperReport(
        one_liner=one_liner,
        playbook=playbook,
        watchlist=watch,
        key_levels=key_levels[:12],
        take_profits=tps,
        extension_note=ext_note,
        week_focus=week_focus,
        summary=summary,
    )
