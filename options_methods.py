# -*- coding: utf-8 -*-
"""
真实规则说明 + 热门评分方式（教学归纳，非荐股、非保证收益）。

数据：Yahoo 实时/延迟期权链（真实挂牌行权价、报价、OI、量）
规则：归纳自常见零售/教育内容（如高概率垂直价差、权利金约宽度1/3、
      约30–45天、约16–30Δ短腿、盈利约50%可提前平仓等）

注意：不是某一家券商的正式 API 策略，也不是历史回测保证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MethodScore:
    id: str
    name: str
    plain: str
    score: float  # 0-100 该规则下得分
    value_text: str  # 算出的关键数字
    pass_ok: bool
    weight: float


@dataclass
class OpsPlan:
    """热门操作方式：进场、管理、出场（白话）。"""
    entry: str
    manage: str
    exit_rules: list[str] = field(default_factory=list)


def _wr(idea: Any) -> float | None:
    v = getattr(idea, "win_rate_profit", None) or getattr(idea, "pop_est", None)
    return float(v) if v is not None else None


def _otm_pct(idea: Any, spot: float) -> float | None:
    short = next((lg for lg in idea.legs if lg.side == "sell"), None)
    if short is None or spot <= 0:
        return None
    if short.right == "put":
        return (spot - short.strike) / spot
    return (short.strike - spot) / spot


def _fill_ratio(idea: Any) -> float | None:
    w = float(getattr(idea, "width", 0) or 0)
    if w <= 0:
        return None
    if idea.net_credit is not None:
        return float(idea.net_credit) / w
    if idea.net_debit is not None:
        return float(idea.net_debit) / w
    return None


def _roc(idea: Any) -> float | None:
    """风险收益率：最多赚 / 最多亏（热门看 ROC / R:R）。"""
    mx = float(idea.max_profit or 0)
    ml = float(idea.max_loss or 0)
    if ml <= 0:
        return None
    return mx / ml


def score_popular_methods(idea: Any, spot: float, dte: int) -> list[MethodScore]:
    """对单个价差套用多条热门规则，给出分项分。"""
    out: list[MethodScore] = []
    wr = _wr(idea)
    otm = _otm_pct(idea, spot)
    fill = _fill_ratio(idea)
    roc = _roc(idea)
    liq = getattr(idea, "liquidity_score", None)
    is_credit = idea.net_credit is not None

    # 1) 天数 30–45（也接受 21–60 稍宽）
    if 30 <= dte <= 45:
        s, ok, txt = 95.0, True, f"{dte} 天（黄金区 30–45）"
    elif 21 <= dte <= 60:
        s, ok, txt = 75.0, True, f"{dte} 天（可接受 21–60）"
    else:
        s, ok, txt = 40.0, False, f"{dte} 天（偏短或偏长）"
    out.append(
        MethodScore(
            "dte_30_45",
            "持有天数（DTE）",
            "很多人挑大约 1 个月左右的期权，别太短也别太长",
            s,
            txt,
            ok,
            1.0,
        )
    )

    # 2) 短腿虚值约 2%–5%（近似 16–30Δ 区，指数 ETF）
    if otm is None:
        out.append(
            MethodScore(
                "otm_delta_zone",
                "短腿离现价（近似 delta 区）",
                "短腿太近=危险，太远=收太少；常见约 2%–5% 虚值",
                40.0,
                "无法计算",
                False,
                1.1,
            )
        )
    else:
        pct = otm * 100
        if 2.0 <= pct <= 5.5:
            s, ok = 92.0, True
            txt = f"短腿约 {pct:.1f}% 虚值（常见高概率区）"
        elif 1.0 <= pct < 2.0 or 5.5 < pct <= 7.5:
            s, ok = 70.0, True
            txt = f"短腿约 {pct:.1f}% 虚值（略偏）"
        else:
            s, ok = 45.0, False
            txt = f"短腿约 {pct:.1f}% 虚值（偏近或偏远）"
        out.append(
            MethodScore(
                "otm_delta_zone",
                "短腿离现价（近似 16–30Δ）",
                "教学里常说卖约 16–30 delta；这里用虚值%近似（无真实希腊值时）",
                s,
                txt,
                ok,
                1.15,
            )
        )

    # 3) 权利金 / 宽度 ≈ 1/4～1/3（信用）
    if is_credit and fill is not None:
        pct = fill * 100
        if 22 <= pct <= 38:
            s, ok = 95.0, True
            txt = f"收到宽度的 {pct:.0f}%（贴近 1/3 规则）"
        elif 15 <= pct < 22 or 38 < pct <= 45:
            s, ok = 72.0, True
            txt = f"收到宽度的 {pct:.0f}%（尚可）"
        else:
            s, ok = 48.0, False
            txt = f"收到宽度的 {pct:.0f}%（偏少或偏多）"
        out.append(
            MethodScore(
                "credit_one_third",
                "权利金 ÷ 宽度",
                "很多人要求：卖出价差收到的钱大约是两腿间距的 1/4～1/3",
                s,
                txt,
                ok,
                1.2,
            )
        )
    elif not is_credit and fill is not None:
        # debit：付得越少（相对宽度）越有空间
        pct = fill * 100
        if pct <= 55:
            s, ok = 80.0, True
            txt = f"买入成本约占宽度 {pct:.0f}%（还有上行空间）"
        else:
            s, ok = 50.0, False
            txt = f"买入成本约占宽度 {pct:.0f}%（偏贵）"
        out.append(
            MethodScore(
                "debit_cost_width",
                "买入成本 ÷ 宽度",
                "买进价差时，付得越少相对宽度，潜在报酬空间越大",
                s,
                txt,
                ok,
                1.0,
            )
        )

    # 4) 估算 POP / 赢面
    if wr is not None:
        if wr >= 70:
            s, ok = 90.0, True
        elif wr >= 60:
            s, ok = 78.0, True
        elif wr >= 50:
            s, ok = 65.0, True
        else:
            s, ok = 45.0, False
        out.append(
            MethodScore(
                "pop_estimate",
                "有利润概率（POP）",
                "用波动率+剩余天数估算到期落在赚钱区的概率（不是历史胜率保证）",
                s,
                f"约 {wr:.0f}%",
                ok,
                1.25,
            )
        )

    # 5) ROC = max profit / max loss
    if roc is not None:
        if roc >= 0.45:
            s, ok = 88.0, True
            txt = f"最多赚/最多亏 = {roc:.2f}（报酬不错）"
        elif roc >= 0.25:
            s, ok = 72.0, True
            txt = f"最多赚/最多亏 = {roc:.2f}（一般）"
        else:
            s, ok = 50.0, False
            txt = f"最多赚/最多亏 = {roc:.2f}（偏低，常见于高胜率）"
        out.append(
            MethodScore(
                "roc",
                "风险收益比 ROC",
                "最多能赚 ÷ 最多会亏；高胜率策略往往 ROC 较低（赚少亏可控）",
                s,
                txt,
                ok,
                0.9,
            )
        )

    # 6) 流动性
    if liq is not None:
        lab = getattr(idea, "liquidity_label", "") or ""
        if liq >= 65:
            s, ok = 92.0, True
        elif liq >= 42:
            s, ok = 75.0, True
        elif liq >= 25:
            s, ok = 55.0, True
        else:
            s, ok = 30.0, False
        out.append(
            MethodScore(
                "liquidity",
                "流动性",
                "看未平仓、成交量、买卖价差——滑点小才容易按计划进出",
                s,
                f"{lab}（{liq:.0f}分）",
                ok,
                1.1,
            )
        )

    # 7) 定义风险（vertical 本身）
    out.append(
        MethodScore(
            "defined_risk",
            "风险有限（Vertical）",
            "垂直价差两边都有腿，最多亏是算得出来的（比裸卖期权安全）",
            100.0,
            f"最多亏约 ${float(idea.max_loss):.0f}/张",
            True,
            0.8,
        )
    )

    return out


def composite_method_score(methods: list[MethodScore]) -> float:
    if not methods:
        return 0.0
    num = sum(m.score * m.weight for m in methods)
    den = sum(m.weight for m in methods) or 1.0
    return round(num / den, 1)


def build_ops_plan(idea: Any) -> OpsPlan:
    """热门操作方式：50% 止盈、接近短腿风控等。"""
    is_credit = idea.net_credit is not None
    half = float(idea.max_profit) * 0.5
    be = idea.breakevens[0] if idea.breakevens else None

    if is_credit:
        entry = (
            f"今天卖出这组价差，目标大约收到 ${idea.net_credit:.2f}/股"
            f"（一张约 ${idea.net_credit * 100:.0f}），用限价单，别市价乱扫。"
        )
        manage = (
            f"很多人会在赚到最大利润约一半（约 ${half:.0f}/张）时提前买回离场，"
            "不必拿到到期——时间衰减对卖方有利，但别贪。"
        )
        exits = [
            f"止盈：浮盈达到约 ${half:.0f}/张（最大利润的一半）可买回",
            f"到期：若股票价仍在有利一侧，可拿到接近最大利润约 ${idea.max_profit:.0f}/张",
            f"风控：浮亏变大或股价靠近危险侧时考虑提前买回；最多亏约 ${idea.max_loss:.0f}/张",
        ]
        if be is not None:
            exits.append(f"记住不赚不亏价大约 {be:.2f}（到期结算参考）")
    else:
        entry = (
            f"今天买进这组价差，大约付 ${idea.net_debit:.2f}/股"
            f"（一张约 ${idea.net_debit * 100:.0f}），限价单。"
        )
        manage = (
            "买进价差靠方向；很多人在趋势走顺、浮盈不错时分批卖出，"
            "或设好最多亏完权利金的心理准备。"
        )
        exits = [
            f"止盈：接近最大利润 ${idea.max_profit:.0f}/张 附近可分批卖出",
            f"止损：最多就是亏掉成本约 ${idea.max_loss:.0f}/张（别再加仓摊平）",
            "时间：越接近到期，方向不对会亏得快，别拖",
        ]
        if be is not None:
            exits.append(f"不赚不亏价大约 {be:.2f}")

    return OpsPlan(entry=entry, manage=manage, exit_rules=exits)


def enrich_idea_with_methods(idea: Any, spot: float, dte: int) -> Any:
    """写入 idea 上的热门规则评分与操作计划。"""
    methods = score_popular_methods(idea, spot, dte)
    idea.method_scores = methods
    idea.method_composite = composite_method_score(methods)
    idea.ops_plan = build_ops_plan(idea)
    # 权利金/宽度、ROC 明文
    fill = _fill_ratio(idea)
    roc = _roc(idea)
    idea.metric_credit_width = round(fill * 100, 1) if fill is not None else None
    idea.metric_roc = round(roc, 2) if roc is not None else None
    half = float(idea.max_profit) * 0.5
    idea.metric_half_profit = round(half, 1)
    return idea


def methods_to_rows(idea: Any) -> list[dict[str, Any]]:
    methods = getattr(idea, "method_scores", None) or []
    rows = []
    for m in methods:
        rows.append(
            {
                "规则": m.name,
                "白话": m.plain,
                "算出什么": m.value_text,
                "这项分": m.score,
                "过关": "是" if m.pass_ok else "偏弱",
            }
        )
    return rows
