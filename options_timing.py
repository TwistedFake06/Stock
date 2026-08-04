"""
Assess whether market conditions are suitable for opening a vertical spread.

Real-trading assist checklist — not investment advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpreadTimingReport:
    """是否适合现在开垂直价差（实盘辅助）。"""

    verdict: str  # 适合开仓 | 谨慎可做 | 暂不建议 | 数据不足
    score: float  # 0-100 综合适合度
    color: str  # green | amber | red | gray
    headline: str  # 一句话结论
    bullets: list[str] = field(default_factory=list)  # 依据（正/负）
    checklist: list[dict[str, str]] = field(default_factory=list)
    # each: {name, status: pass|warn|fail, detail}
    preferred_style: str = ""  # 偏信用 / 偏借方 / 观望
    action: str = ""  # 建议动作


def _iv_regime(iv_atm: float | None) -> tuple[str, str]:
    if iv_atm is None or iv_atm <= 0:
        return "未知", "无 ATM IV"
    pct = iv_atm * 100
    if pct < 12:
        return "偏低", f"IV≈{pct:.0f}%（权利金偏薄，卖方优势弱）"
    if pct < 22:
        return "中等", f"IV≈{pct:.0f}%（常见区间）"
    if pct < 35:
        return "偏高", f"IV≈{pct:.0f}%（卖方权利金较厚，波动风险也大）"
    return "很高", f"IV≈{pct:.0f}%（极端波动，仓位宜更小）"


def assess_spread_timing(
    *,
    direction: Any | None,
    best: Any | None,
    after_hours: bool,
    dte: int | None,
    iv_atm: float | None,
    ideas_count: int,
    quote_warning: str = "",
    pricing_note: str = "",
) -> SpreadTimingReport:
    """
    Score suitability to open a vertical *now*.

    Weights (approx):
      session 15 · direction 20 · liquidity 20 · POP/EV 25 · DTE 10 · IV 10
    """
    checklist: list[dict[str, str]] = []
    bullets: list[str] = []
    score = 50.0  # start neutral

    # ---- 1) Session / quote quality ----
    if after_hours:
        score -= 18
        checklist.append(
            {
                "name": "交易时段",
                "status": "warn",
                "detail": "当前非美股常规交易时段（RTH 09:30–16:00 ET），盘后 bid/ask 易失真",
            }
        )
        bullets.append("⚠️ 盘后：报价参考性下降，不宜急着市价开仓")
    else:
        score += 12
        checklist.append(
            {
                "name": "交易时段",
                "status": "pass",
                "detail": "美股常规交易时段内（或系统判定为盘中）",
            }
        )
        bullets.append("✅ 盘中：更适合按自然成交（bid/ask）下单")

    if quote_warning and after_hours:
        score -= 5
    if "覆盖偏低" in (quote_warning or "") or "覆盖偏低" in (pricing_note or ""):
        score -= 8
        checklist.append(
            {
                "name": "报价质量",
                "status": "warn",
                "detail": "期权链买卖价覆盖偏低，部分腿可能用中间价估算",
            }
        )

    # ---- 2) Candidates available ----
    if ideas_count <= 0 or best is None:
        checklist.append(
            {
                "name": "可用结构",
                "status": "fail",
                "detail": "无通过硬过滤的 vertical（流动性/权利金/宽度）",
            }
        )
        return SpreadTimingReport(
            verdict="暂不建议",
            score=max(0.0, min(35.0, score - 30)),
            color="red",
            headline="现在不适合开仓：没有合格价差候选",
            bullets=bullets
            + ["❌ 链上无符合流动性与权利金门槛的结构，或数据拉取失败"],
            checklist=checklist,
            preferred_style="观望",
            action="等待盘中刷新链，或放宽到期天数后再扫描",
        )

    checklist.append(
        {
            "name": "可用结构",
            "status": "pass",
            "detail": f"扫描后约有 {ideas_count} 个合格结构",
        }
    )
    score += 8

    # ---- 3) Direction ----
    dir_label = "中性"
    dir_score = 0.0
    strength = "弱"
    style_hint = ""
    if direction is not None:
        dir_label = getattr(direction, "direction", "中性") or "中性"
        dir_score = float(getattr(direction, "score", 0) or 0)
        strength = getattr(direction, "strength", "弱") or "弱"
        style_hint = getattr(direction, "style_hint", "") or ""

    abs_d = abs(dir_score)
    best_code = getattr(best, "code", "") or ""
    is_credit = getattr(best, "net_credit", None) is not None
    bull_codes = {"bull_put", "bull_call"}
    bear_codes = {"bear_call", "bear_put"}

    aligned = True
    if dir_label == "看多" and best_code in bear_codes:
        aligned = False
    if dir_label == "看空" and best_code in bull_codes:
        aligned = False

    if abs_d >= 25 and aligned:
        score += 15
        checklist.append(
            {
                "name": "方向清晰度",
                "status": "pass",
                "detail": f"{dir_label}（{strength}，得分 {dir_score:+.0f}），与推荐结构一致",
            }
        )
        bullets.append(f"✅ 方向 {dir_label} 较清晰，且与 {best_code or '推荐'} 同向")
    elif abs_d >= 25 and not aligned:
        score -= 12
        checklist.append(
            {
                "name": "方向清晰度",
                "status": "warn",
                "detail": f"技术面 {dir_label}，但推荐结构偏反向（{best_code}）",
            }
        )
        bullets.append("⚠️ 方向与推荐结构不一致，需确认是否故意做对冲/高胜率侧")
    elif abs_d < 18:
        score += 2
        checklist.append(
            {
                "name": "方向清晰度",
                "status": "warn",
                "detail": f"方向偏中性（得分 {dir_score:+.0f}）——更适合高 POP 信用价差，不宜重仓 debit",
            }
        )
        bullets.append("⚠️ 方向不明：若做，优先小仓信用价差")
    else:
        score += 6
        checklist.append(
            {
                "name": "方向清晰度",
                "status": "pass",
                "detail": f"温和方向 {dir_label}（{dir_score:+.0f}）",
            }
        )

    # ---- 4) Liquidity ----
    liq = getattr(best, "liquidity_label", "") or ""
    liq_score = getattr(best, "liquidity_score", None)
    if liq == "高" or (liq_score is not None and float(liq_score) >= 65):
        score += 14
        checklist.append(
            {
                "name": "流动性",
                "status": "pass",
                "detail": getattr(best, "liquidity_detail", "") or f"流动性 {liq}",
            }
        )
        bullets.append("✅ 推荐腿流动性较好")
    elif liq == "中" or (liq_score is not None and float(liq_score) >= 42):
        score += 4
        checklist.append(
            {
                "name": "流动性",
                "status": "warn",
                "detail": getattr(best, "liquidity_detail", "") or f"流动性 {liq}",
            }
        )
        bullets.append("⚠️ 流动性一般：用限价、分笔，避免市价扫穿")
    else:
        score -= 15
        checklist.append(
            {
                "name": "流动性",
                "status": "fail",
                "detail": getattr(best, "liquidity_detail", "") or f"流动性 {liq or '差'}",
            }
        )
        bullets.append("❌ 流动性偏差：滑点与成交风险高")

    pricing = getattr(best, "pricing_mode", "") or ""
    if pricing == "natural":
        score += 5
        checklist.append(
            {
                "name": "定价假设",
                "status": "pass",
                "detail": "自然成交（卖=bid / 买=ask），估算偏保守",
            }
        )
    elif pricing:
        score -= 4
        checklist.append(
            {
                "name": "定价假设",
                "status": "warn",
                "detail": f"定价模式 {pricing}（部分/全部用中间价）",
            }
        )

    # ---- 5) POP / EV ----
    pop = getattr(best, "win_rate_profit", None)
    if pop is None:
        pop = getattr(best, "pop_est", None)
    ev = getattr(best, "expected_value", None)
    ev_m = getattr(best, "expected_value_managed", None)

    if pop is not None:
        pop = float(pop)
        if is_credit:
            if pop >= 65:
                score += 12
                checklist.append(
                    {
                        "name": "POP 胜率",
                        "status": "pass",
                        "detail": f"信用结构 POP≈{pop:.0f}%（偏高胜率区）",
                    }
                )
            elif pop >= 52:
                score += 5
                checklist.append(
                    {
                        "name": "POP 胜率",
                        "status": "warn",
                        "detail": f"信用结构 POP≈{pop:.0f}%（中等）",
                    }
                )
            else:
                score -= 8
                checklist.append(
                    {
                        "name": "POP 胜率",
                        "status": "warn",
                        "detail": f"信用结构 POP≈{pop:.0f}%（偏低，风险回报需更挑剔）",
                    }
                )
        else:
            if pop >= 45:
                score += 8
                checklist.append(
                    {
                        "name": "POP 胜率",
                        "status": "pass",
                        "detail": f"借方结构 POP≈{pop:.0f}%",
                    }
                )
            else:
                score -= 4
                checklist.append(
                    {
                        "name": "POP 胜率",
                        "status": "warn",
                        "detail": f"借方结构 POP≈{pop:.0f}%（方向赌性更强）",
                    }
                )

    # Prefer managed EV for "should I open" decision
    ev_use = ev_m if ev_m is not None else ev
    if ev_use is not None:
        ev_use = float(ev_use)
        if ev_use > 15:
            score += 12
            checklist.append(
                {
                    "name": "期望值 EV",
                    "status": "pass",
                    "detail": f"{'管理' if ev_m is not None else '到期'}EV≈${ev_use:.0f}/张（为正）",
                }
            )
            bullets.append(f"✅ 模型期望为正（≈${ev_use:.0f}/张）")
        elif ev_use > 0:
            score += 5
            checklist.append(
                {
                    "name": "期望值 EV",
                    "status": "warn",
                    "detail": f"EV≈${ev_use:.0f}/张（略正，边际）",
                }
            )
        else:
            score -= 12
            checklist.append(
                {
                    "name": "期望值 EV",
                    "status": "fail",
                    "detail": f"EV≈${ev_use:.0f}/张（为负：模型不支持开仓）",
                }
            )
            bullets.append(f"❌ 模型期望为负（≈${ev_use:.0f}/张）")

    # ---- 6) DTE ----
    if dte is not None:
        dte = int(dte)
        if 21 <= dte <= 45:
            score += 8
            checklist.append(
                {
                    "name": "到期天数",
                    "status": "pass",
                    "detail": f"DTE={dte}（常见 3–6 周甜蜜区）",
                }
            )
        elif 14 <= dte < 21 or 45 < dte <= 60:
            score += 2
            checklist.append(
                {
                    "name": "到期天数",
                    "status": "warn",
                    "detail": f"DTE={dte}（可用，但非最优区间）",
                }
            )
        else:
            score -= 6
            checklist.append(
                {
                    "name": "到期天数",
                    "status": "warn",
                    "detail": f"DTE={dte}（过近 theta 急 / 过远资金效率差）",
                }
            )

    # ---- 7) IV ----
    regime, iv_detail = _iv_regime(iv_atm)
    if regime in ("中等", "偏高") and is_credit:
        score += 6
        checklist.append({"name": "波动率 IV", "status": "pass", "detail": iv_detail})
        bullets.append(f"✅ IV {regime}：信用卖方权利金尚可")
    elif regime == "偏低" and is_credit:
        score -= 6
        checklist.append({"name": "波动率 IV", "status": "warn", "detail": iv_detail})
        bullets.append("⚠️ IV 偏低：卖方收息薄，需更高胜率/更紧风控")
    elif regime == "很高":
        score -= 4
        checklist.append({"name": "波动率 IV", "status": "warn", "detail": iv_detail})
        bullets.append("⚠️ IV 很高：权利金厚但跳空/扫损风险大")
    elif not is_credit and regime in ("偏低", "中等"):
        score += 4
        checklist.append({"name": "波动率 IV", "status": "pass", "detail": iv_detail + " · 借方买波动不过贵"})
    else:
        checklist.append({"name": "波动率 IV", "status": "warn", "detail": iv_detail})

    # ---- Final verdict ----
    score = float(max(0.0, min(100.0, score)))

    if score >= 72 and not after_hours and (ev_use is None or ev_use > 0):
        verdict, color = "适合开仓", "green"
        headline = "综合判断：当前条件较适合开垂直价差（仍须限价与 1R 风控）"
        action = "可用限价按推荐结构小仓试探；设好 50% 止盈 / 2R 止损"
    elif score >= 55:
        verdict, color = "谨慎可做", "amber"
        headline = "综合判断：可以做，但有扣分项——减小仓位或等更好报价"
        action = "若开仓：严格限价、半仓或以下、必须写清出场条件"
    elif score >= 40:
        verdict, color = "暂不建议", "red"
        headline = "综合判断：现在开仓性价比一般，优先观望或只做模拟"
        action = "建议等待：盘中更好报价 / 方向更清晰 / EV 转正"
    else:
        verdict, color = "暂不建议", "red"
        headline = "综合判断：条件较差，不建议现在开新仓"
        action = "不要开仓；检查时段、流动性与 EV 后再评估"

    # After-hours never "适合开仓"
    if after_hours and verdict == "适合开仓":
        verdict, color = "谨慎可做", "amber"
        headline = "盘后仅作计划：条件尚可，但请等 RTH 确认报价后再下单"
        action = "先记下结构与限价，开盘后核验 bid/ask 再下单"

    preferred = "观望"
    if is_credit:
        preferred = "偏信用价差（收权利金）"
    else:
        preferred = "偏借方价差（付权利金）"
    if style_hint:
        preferred = style_hint

    if not bullets:
        bullets.append(f"适合度评分 {score:.0f}/100 · {verdict}")

    return SpreadTimingReport(
        verdict=verdict,
        score=round(score, 1),
        color=color,
        headline=headline,
        bullets=bullets,
        checklist=checklist,
        preferred_style=preferred,
        action=action,
    )
