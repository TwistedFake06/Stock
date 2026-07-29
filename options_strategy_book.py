# -*- coding: utf-8 -*-
"""
参考市面常见 vertical 做法（教学归纳，非荐股）：
- 高胜率收权利金（约 16–30 delta 短腿、30–45 天）
- 权利金约占宽度 1/3（许多零售/教学常用规则）
- 方向顺势：看多卖 put 价差、看空卖 call 价差
- 强趋势才考虑买进 debit 价差

给每个候选打「实战贴合分」，并标出像哪一种常见策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyStyle:
    id: str
    name_zh: str
    plain: str  # 白话
    source_note: str  # 参考来源风格（非背书）
    prefer_codes: tuple[str, ...]
    min_dte: int
    max_dte: int
    # short leg OTM% 目标区间（约等于常见 delta 区）
    otm_lo: float
    otm_hi: float
    # 信用 / 宽度 目标
    credit_fill_lo: float
    credit_fill_hi: float
    want_credit: bool
    min_win_rate: float  # 希望有利润胜率至少
    weight: float = 1.0


# 归纳自常见教学/社区规则（Tastytrade 风格高 POP、零售 1/3 规则等）
STRATEGY_BOOK: list[StrategyStyle] = [
    StrategyStyle(
        id="high_pop_credit",
        name_zh="高胜率收钱价差",
        plain="卖出价差先收钱，短腿离现价远一点，赢面高、单笔赚少",
        source_note="常见于高概率垂直价差教学（约 16–25Δ 短腿、30–45 天）",
        prefer_codes=("bull_put", "bear_call"),
        min_dte=21,
        max_dte=60,
        otm_lo=0.02,
        otm_hi=0.06,
        credit_fill_lo=0.15,
        credit_fill_hi=0.40,
        want_credit=True,
        min_win_rate=65.0,
        weight=1.2,
    ),
    StrategyStyle(
        id="balanced_credit",
        name_zh="平衡收钱价差",
        plain="收钱适中：赢面和权利金取中间，宽度里大约收到 1/4～1/3",
        source_note="零售常用「收到宽度约 1/3」规则",
        prefer_codes=("bull_put", "bear_call"),
        min_dte=14,
        max_dte=45,
        otm_lo=0.01,
        otm_hi=0.04,
        credit_fill_lo=0.22,
        credit_fill_hi=0.38,
        want_credit=True,
        min_win_rate=55.0,
        weight=1.15,
    ),
    StrategyStyle(
        id="income_bull_put",
        name_zh="偏多收租（卖 Put 价差）",
        plain="觉得大盘不会大跌时：卖出看跌价差，先收钱，几天后买回",
        source_note="Bull Put Credit，指数 ETF 最常用方向性收入结构",
        prefer_codes=("bull_put",),
        min_dte=14,
        max_dte=45,
        otm_lo=0.015,
        otm_hi=0.05,
        credit_fill_lo=0.18,
        credit_fill_hi=0.40,
        want_credit=True,
        min_win_rate=60.0,
        weight=1.1,
    ),
    StrategyStyle(
        id="income_bear_call",
        name_zh="偏空收租（卖 Call 价差）",
        plain="觉得不会大涨时：卖出看涨价差，先收钱，几天后买回",
        source_note="Bear Call Credit，温和看空/横盘常用",
        prefer_codes=("bear_call",),
        min_dte=14,
        max_dte=45,
        otm_lo=0.015,
        otm_hi=0.05,
        credit_fill_lo=0.18,
        credit_fill_hi=0.40,
        want_credit=True,
        min_win_rate=60.0,
        weight=1.1,
    ),
    StrategyStyle(
        id="trend_debit",
        name_zh="顺势买进价差",
        plain="方向很明确时才买进价差：先付钱，涨/跌到位再卖出",
        source_note="Debit vertical：Bull Call / Bear Put，强趋势教学常用",
        prefer_codes=("bull_call", "bear_put"),
        min_dte=14,
        max_dte=45,
        otm_lo=-0.02,
        otm_hi=0.03,
        credit_fill_lo=0.0,
        credit_fill_hi=1.0,
        want_credit=False,
        min_win_rate=35.0,
        weight=0.95,
    ),
]


@dataclass
class StyleMatch:
    style: StrategyStyle
    fit_score: float  # 0-100 贴合该策略规则
    reasons: list[str] = field(default_factory=list)


def _short_otm_pct(idea: Any, spot: float) -> float | None:
    """短腿相对现价的 OTM 比例（正数=虚值）。"""
    short = None
    for leg in idea.legs:
        if leg.side == "sell":
            short = leg
            break
    if short is None or spot <= 0:
        return None
    if short.right == "put":
        return (spot - short.strike) / spot
    return (short.strike - spot) / spot


def _credit_fill(idea: Any) -> float | None:
    w = float(getattr(idea, "width", 0) or 0)
    if w <= 0:
        return None
    if idea.net_credit is not None:
        return float(idea.net_credit) / w
    if idea.net_debit is not None:
        # debit：付了宽度的多少（越小越好，用 1 - debit/w 当“性价比”）
        return 1.0 - float(idea.net_debit) / w
    return None


def match_styles(
    idea: Any,
    spot: float,
    dte: int,
    direction: str,
) -> list[StyleMatch]:
    """返回该候选贴合哪些实战策略风格（按 fit 降序）。"""
    out: list[StyleMatch] = []
    wr = getattr(idea, "win_rate_profit", None) or getattr(idea, "pop_est", None)
    wr = float(wr) if wr is not None else None
    otm = _short_otm_pct(idea, spot)
    fill = _credit_fill(idea)
    code = idea.code
    is_credit = idea.net_credit is not None

    for st in STRATEGY_BOOK:
        if code not in st.prefer_codes:
            continue
        if st.want_credit and not is_credit:
            continue
        if not st.want_credit and is_credit:
            continue

        score = 40.0
        reasons: list[str] = []

        # DTE
        if st.min_dte <= dte <= st.max_dte:
            score += 18
            reasons.append(f"天数 {dte} 落在常用 {st.min_dte}–{st.max_dte} 天")
        else:
            score -= 12
            reasons.append(f"天数 {dte} 略偏离常用区间")

        # OTM
        if otm is not None:
            if st.otm_lo <= otm <= st.otm_hi:
                score += 20
                reasons.append(f"短腿离现价约 {otm * 100:.1f}%（常见虚值区）")
            else:
                dist = min(abs(otm - st.otm_lo), abs(otm - st.otm_hi))
                score -= min(15, dist * 200)
                reasons.append(f"短腿离现价约 {otm * 100:.1f}%（略偏常用区）")

        # credit fill
        if fill is not None and st.want_credit:
            if st.credit_fill_lo <= fill <= st.credit_fill_hi:
                score += 18
                reasons.append(f"收到宽度的约 {fill * 100:.0f}%（接近 1/4～1/3 规则）")
            else:
                score -= 8
                reasons.append(f"收到宽度的约 {fill * 100:.0f}%")

        # win rate
        if wr is not None:
            if wr >= st.min_win_rate:
                score += 15
                reasons.append(f"估算赢面 {wr:.0f}% ≥ 该类常见要求")
            else:
                score -= 10
                reasons.append(f"估算赢面 {wr:.0f}% 偏低")

        # direction alignment
        if direction == "看多" and code in ("bull_put", "bull_call"):
            score += 12
            reasons.append("方向看多，顺势")
        elif direction == "看空" and code in ("bear_call", "bear_put"):
            score += 12
            reasons.append("方向看空，顺势")
        elif direction == "中性" and is_credit:
            score += 8
            reasons.append("中性时偏收钱结构")
        elif direction == "看多" and code in ("bear_call", "bear_put"):
            score -= 15
            reasons.append("与看多方向相反")
        elif direction == "看空" and code in ("bull_put", "bull_call"):
            score -= 15
            reasons.append("与看空方向相反")

        score = max(0.0, min(100.0, score * st.weight))
        if score >= 35:
            out.append(StyleMatch(style=st, fit_score=round(score, 1), reasons=reasons))

    out.sort(key=lambda m: m.fit_score, reverse=True)
    return out


def apply_playbook_ranking(
    ideas: list[Any],
    spot: float,
    dte: int,
    direction: str,
) -> tuple[Any | None, Any | None, list[dict[str, Any]]]:
    """
    按「实战策略贴合 + 赢面」重排。
    返回 (最佳实战推荐, 最高赢面且贴合, 说明表)
    """
    if not ideas:
        return None, None, []

    scored: list[tuple[float, Any, StyleMatch | None]] = []
    for idea in ideas:
        matches = match_styles(idea, spot, dte, direction)
        best_m = matches[0] if matches else None
        wr = getattr(idea, "win_rate_profit", None) or getattr(idea, "pop_est", None) or 0.0
        base = float(getattr(idea, "score", 50) or 50)
        fit = best_m.fit_score if best_m else 25.0
        liq = float(getattr(idea, "liquidity_score", None) or 40.0)
        method_c = float(getattr(idea, "method_composite", None) or 0.0)
        # 综合：热门规则 28% + 策略风格 25% + 赢面 22% + 流动性 15% + 原分 10%
        if method_c > 0:
            combo = (
                0.28 * method_c
                + 0.25 * fit
                + 0.22 * float(wr)
                + 0.15 * liq
                + 0.10 * base
            )
        else:
            combo = 0.38 * fit + 0.18 * base + 0.27 * float(wr) + 0.17 * liq
        if liq < 25:
            combo *= 0.75
        if best_m:
            idea.playbook_style = best_m.style.name_zh
            idea.playbook_plain = best_m.style.plain
            idea.playbook_source = best_m.style.source_note
            idea.playbook_fit = best_m.fit_score
            idea.playbook_reasons = list(best_m.reasons)
            liq_label = getattr(idea, "liquidity_label", "") or ""
            if liq_label:
                idea.playbook_reasons.append(
                    f"流动性{liq_label}（{liq:.0f}分）"
                )
            if method_c > 0:
                idea.playbook_reasons.append(f"热门规则综合分 {method_c:.0f}")
            idea.rank_reason = (
                f"像「{best_m.style.name_zh}」· 贴合 {best_m.fit_score:.0f} · "
                f"赢面 {float(wr):.0f}% · 流动性{liq_label or '—'} · "
                f"规则分{method_c:.0f}"
            )
        else:
            idea.playbook_style = "一般扫描"
            idea.playbook_plain = "未强匹配常见规则，仅作比较"
            idea.playbook_source = ""
            idea.playbook_fit = 0.0
            idea.playbook_reasons = []
        idea.playbook_combo = round(combo, 1)
        scored.append((combo, idea, best_m))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_play = scored[0][1]

    def _wr(i: Any) -> float:
        v = getattr(i, "win_rate_profit", None) or getattr(i, "pop_est", None)
        return float(v) if v is not None else -1.0

    def _liq(i: Any) -> float:
        return float(getattr(i, "liquidity_score", None) or 0.0)

    # 最高赢面：贴合够 + 流动性别太差
    aligned = []
    for combo, idea, m in scored:
        fit = getattr(idea, "playbook_fit", 0) or 0
        if fit >= 48 and _wr(idea) >= 0 and _liq(idea) >= 28:
            aligned.append(idea)
    if not aligned:
        aligned = [
            i for _, i, _ in scored if _wr(i) >= 0 and _liq(i) >= 20
        ] or [i for _, i, _ in scored if _wr(i) >= 0]
    # 赢面优先，同赢面选流动性更好
    best_wr = max(aligned, key=lambda i: (_wr(i), _liq(i))) if aligned else best_play

    table = []
    for combo, idea, m in scored[:12]:
        table.append(
            {
                "实战分": round(combo, 1),
                "像哪种做法": getattr(idea, "playbook_style", "—"),
                "白话": getattr(idea, "playbook_plain", ""),
                "名称": idea.name,
                "赢面%": _wr(idea) if _wr(idea) >= 0 else None,
                "热门规则分": getattr(idea, "method_composite", None),
                "权利金/宽度%": getattr(idea, "metric_credit_width", None),
                "赚亏比ROC": getattr(idea, "metric_roc", None),
                "50%止盈约$/张": getattr(idea, "metric_half_profit", None),
                "50%买回价$/股": getattr(idea, "metric_half_buyback", None),
                "流动性": getattr(idea, "liquidity_label", "—"),
                "流动性分": getattr(idea, "liquidity_score", None),
                "卖出收$" if idea.net_credit is not None else "买进付$": (
                    idea.net_credit if idea.net_credit is not None else idea.net_debit
                ),
                "最多赚$": idea.max_profit,
                "最多亏$": idea.max_loss,
                "流动性说明": getattr(idea, "liquidity_detail", ""),
                "参考": getattr(idea, "playbook_source", ""),
            }
        )
    return best_play, best_wr, table
