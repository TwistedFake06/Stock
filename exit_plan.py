"""
Short-term exit rules for swing plans (0–2w / 2–4w).

Hard rules (not suggestions only):
  - T1 scale-out 50%
  - Move stop to break-even (or BE + buffer) after T1
  - Time stop: max hold = horizon trading days
  - Full exit on structure stop / invalidation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Round-trip friction assumptions for "executable" R:R (US equities retail-ish)
DEFAULT_SLIP_PCT = 0.0030  # 0.30% per side（更贴近零售成交；原 0.15% 偏乐观）


@dataclass
class ExitPlan:
    horizon_key: str  # h1 | h2
    horizon_label: str
    max_hold_days: int
    scale_out_pct: float  # 0.5 = 50% at T1
    t1_price: float | None
    t2_price: float | None  # optional runner target
    stop_initial: float | None
    stop_after_t1: float | None  # typically break-even entry
    entry_ref: float | None
    bullets: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class SlippageRR:
    """Paper R:R vs after-slippage executable R:R."""

    slip_pct: float
    entry_paper: float | None
    stop_paper: float | None
    target_paper: float | None
    entry_fill: float | None  # worse fill for long buy
    stop_fill: float | None
    target_fill: float | None
    rr_paper: float | None
    rr_net: float | None
    exp_paper: float | None
    exp_net: float | None
    note: str = ""


def apply_long_slippage(
    entry: float | None,
    stop: float | None,
    target: float | None,
    win_rate_pct: float | None = None,
    slip_pct: float = DEFAULT_SLIP_PCT,
) -> SlippageRR:
    """
    Long: buy pays up, sell (stop/target) receives less.
    entry_fill = entry * (1+s), stop_fill = stop * (1-s), target_fill = target * (1-s)
    """
    s = max(0.0, float(slip_pct))

    def _f(x: float | None) -> float | None:
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    e, st, tg = _f(entry), _f(stop), _f(target)
    e_f = e * (1 + s) if e is not None else None
    st_f = st * (1 - s) if st is not None else None
    tg_f = tg * (1 - s) if tg is not None else None

    def _rr(en: float | None, sp: float | None, ta: float | None) -> float | None:
        if en is None or sp is None or ta is None:
            return None
        risk = en - sp
        reward = ta - en
        if risk <= 0 or reward <= 0:
            return None
        return round(reward / risk, 3)

    rr_p = _rr(e, st, tg)
    rr_n = _rr(e_f, st_f, tg_f)

    def _exp(rr: float | None, wr: float | None) -> float | None:
        if rr is None or wr is None:
            return None
        p = float(wr) / 100.0
        return round(p * rr - (1 - p) * 1.0, 3)

    note = f"滑点假设单边 {s * 100:.2f}%（买卖各计）· 净R:R更接近可成交"
    if rr_p is not None and rr_n is not None and rr_n < rr_p:
        note += f" · 纸面 {rr_p:.2f} → 净 {rr_n:.2f}"

    return SlippageRR(
        slip_pct=s,
        entry_paper=round(e, 4) if e is not None else None,
        stop_paper=round(st, 4) if st is not None else None,
        target_paper=round(tg, 4) if tg is not None else None,
        entry_fill=round(e_f, 4) if e_f is not None else None,
        stop_fill=round(st_f, 4) if st_f is not None else None,
        target_fill=round(tg_f, 4) if tg_f is not None else None,
        rr_paper=rr_p,
        rr_net=rr_n,
        exp_paper=_exp(rr_p, win_rate_pct),
        exp_net=_exp(rr_n, win_rate_pct),
        note=note,
    )


def build_exit_plan(
    *,
    horizon_key: str,
    horizon_label: str,
    max_hold_days: int,
    entry: float | None,
    stop: float | None,
    t1: float | None,
    t2: float | None = None,
    scale_out_pct: float = 0.50,
) -> ExitPlan:
    """Construct concrete exit rules for one horizon."""
    bullets: list[str] = []
    be = None
    if entry is not None:
        # break-even after T1 hit (slightly above entry to cover fees)
        be = round(float(entry) * 1.001, 4)

    bullets.append(
        f"① **T1 减仓 {scale_out_pct * 100:.0f}%**"
        + (f" @ {t1:.2f}" if t1 is not None else "（到目标价）")
    )
    if be is not None:
        bullets.append(
            f"② **止蚀上移至保本** ≈ {be:.2f}（T1 成交后立即改单，禁止下移）"
        )
    else:
        bullets.append("② T1 后止蚀上移至开仓成本（保本）")

    if t2 is not None and t1 is not None and float(t2) > float(t1):
        bullets.append(
            f"③ **剩余仓位看 T2** {t2:.2f}；未到则用保本止蚀保护"
        )
    else:
        bullets.append("③ 剩余仓位：保本止蚀或再设小 trailing，禁止摊平亏损")

    bullets.append(
        f"④ **时间止损**：持仓满 **{max_hold_days} 个交易日** 未到 T1 → 市价/限价清仓离场"
    )
    bullets.append("⑤ **结构止蚀**：收盘跌破初始止蚀 → 全部离场，不加仓")
    bullets.append("⑥ 禁止：摊平、扩大止蚀、T1 未到就提前搬止蚀远离成本")

    summary = (
        f"{horizon_label} 出场纪律：T1 减 {scale_out_pct * 100:.0f}% → 止蚀保本 → "
        f"最多持有 {max_hold_days} 交易日 → 破止蚀全出。"
    )
    return ExitPlan(
        horizon_key=horizon_key,
        horizon_label=horizon_label,
        max_hold_days=int(max_hold_days),
        scale_out_pct=float(scale_out_pct),
        t1_price=round(float(t1), 4) if t1 is not None else None,
        t2_price=round(float(t2), 4) if t2 is not None else None,
        stop_initial=round(float(stop), 4) if stop is not None else None,
        stop_after_t1=be,
        entry_ref=round(float(entry), 4) if entry is not None else None,
        bullets=bullets,
        summary=summary,
    )


def exit_plan_to_dict(plan: ExitPlan) -> dict[str, Any]:
    return {
        "horizon_key": plan.horizon_key,
        "horizon_label": plan.horizon_label,
        "max_hold_days": plan.max_hold_days,
        "scale_out_pct": plan.scale_out_pct,
        "t1_price": plan.t1_price,
        "t2_price": plan.t2_price,
        "stop_initial": plan.stop_initial,
        "stop_after_t1": plan.stop_after_t1,
        "entry_ref": plan.entry_ref,
        "bullets": list(plan.bullets),
        "summary": plan.summary,
    }
