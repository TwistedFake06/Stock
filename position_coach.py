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


def _level_status(last: float, level: float | None, *, kind: str) -> str:
    """kind: target (above=hit) | stop (below=hit) | resistance | support."""
    if level is None:
        return "—"
    if kind == "target":
        if last >= level * 0.998:
            return f"已到/超过 {level:.2f}"
        return f"未到 {level:.2f}（还差 {(level / last - 1) * 100:+.1f}%）"
    if kind == "stop":
        if last <= level * 1.002:
            return f"已触及 {level:.2f}"
        return f"未破 {level:.2f}（上方还有 {(last / level - 1) * 100:+.1f}%）"
    if kind == "resistance":
        if last >= level * 0.998:
            return f"已过/贴着 {level:.2f}"
        return f"下方距 {level:.2f} 还有 {(level / last - 1) * 100:+.1f}%"
    if kind == "support":
        if last <= level * 1.002:
            return f"已失 {level:.2f}"
        return f"仍在 {level:.2f} 上方（{(last / level - 1) * 100:+.1f}%）"
    return f"{level:.2f}"


def analyze_entry_vs_live(
    *,
    buy_price: float,
    last_price: float,
    entry_stop: float | None = None,
    entry_t1: float | None = None,
    entry_t2: float | None = None,
    entry_e: float | None = None,
    entry_res: float | None = None,
    entry_sup: float | None = None,
    entry_close: float | None = None,
    live_stop: float | None = None,
    live_t1: float | None = None,
    live_t2: float | None = None,
    live_e: float | None = None,
    live_res: float | None = None,
    live_sup: float | None = None,
    entry_as_of: str = "",
    live_bias: str = "",
    live_bias_score: float = 0.0,
) -> list[str]:
    """
    Plain-language compare of entry-day plan vs live plan for hold coaching.
    No look-ahead claims — just relative levels vs buy/last.
    """
    buy = _f(buy_price)
    last = _f(last_price)
    lines: list[str] = []
    if buy is None or last is None:
        return ["无法对比：买入价或现价无效"]

    asof = f"（{entry_as_of}）" if entry_as_of else ""
    lines.append(
        f"【入场日{asof}】相对你的买入 {buy:.2f}："
        + (
            f"E={entry_e:.2f}"
            if entry_e is not None
            else "E=—"
        )
        + (
            f" · 你比E{'便宜' if entry_e and buy < entry_e * 0.995 else ('偏贵' if entry_e and buy > entry_e * 1.005 else '接近')}"
            if entry_e is not None
            else ""
        )
        + (f" · 当日收≈{entry_close:.2f}" if entry_close is not None else "")
    )
    lines.append(
        f"【入场日目标进度】S {_level_status(last, entry_stop, kind='stop')} · "
        f"T1 {_level_status(last, entry_t1, kind='target')} · "
        f"T2 {_level_status(last, entry_t2, kind='target')}"
    )
    if entry_res is not None:
        lines.append(f"【入场日阻力】{_level_status(last, entry_res, kind='resistance')}")
    if entry_sup is not None:
        lines.append(f"【入场日支撑】{_level_status(last, entry_sup, kind='support')}")

    lines.append(
        f"【即时结构】S={live_stop:.2f}" if live_stop is not None else "【即时结构】S=—"
    )
    if live_stop is not None and entry_stop is not None:
        if live_stop > entry_stop * 1.01:
            lines.append(
                f"即时止蚀 {live_stop:.2f} 高于入场日止蚀 {entry_stop:.2f}："
                "结构随涨抬升；**硬纪律仍以入场日 S 为底**，建议止蚀可跟更高"
            )
        elif live_stop < entry_stop * 0.99:
            lines.append(
                f"即时止蚀 {live_stop:.2f} 低于入场日 {entry_stop:.2f}："
                "勿把止蚀下移到即时更宽的位置"
            )
    t1_bits = []
    if live_t1 is not None:
        t1_bits.append(f"即时T1 {_level_status(last, live_t1, kind='target')}")
    if live_t2 is not None:
        t1_bits.append(f"即时T2 {_level_status(last, live_t2, kind='target')}")
    if t1_bits:
        lines.append("【即时目标】" + " · ".join(t1_bits))
    if live_res is not None:
        lines.append(f"【即时阻力】{_level_status(last, live_res, kind='resistance')}")
    if live_sup is not None:
        lines.append(f"【即时支撑】{_level_status(last, live_sup, kind='support')}")

    # Synthesis
    hit_e_t1 = entry_t1 is not None and last >= entry_t1 * 0.998
    hit_e_t2 = entry_t2 is not None and last >= entry_t2 * 0.998
    near_live_res = (
        live_res is not None and abs(last - live_res) / max(last, 1e-9) < 0.015
    )
    past_e_res = entry_res is not None and last >= entry_res * 0.998

    if hit_e_t2:
        lines.append(
            "【综合】现价已完成入场日 T2 一带 → 优先锁定利润，剩仓仅极小 + 紧止蚀"
        )
    elif hit_e_t1:
        ext = ""
        if live_t1 is not None and live_t1 > (entry_t1 or 0) * 1.01:
            ext = f"；剩仓可看即时更高目标 T1≈{live_t1:.2f}"
        elif entry_t2 is not None and last < entry_t2:
            ext = f"；剩仓看入场日 T2≈{entry_t2:.2f}"
        lines.append(
            "【综合】现价已完成入场日 T1 → 建议已减仓或现在减半，止蚀抬到保本"
            + ext
        )
    elif past_e_res and not hit_e_t1:
        lines.append(
            "【综合】已过入场日阻力、尚未到入场日 T1 → 可小减或持有，止蚀上移锁利，勿追高加仓"
        )
    elif near_live_res:
        lines.append(
            "【综合】现价贴近即时阻力 → 不宜新开/加仓；持仓者可分批减或收紧止蚀"
        )
    else:
        lines.append(
            "【综合】入场日目标未完成且未破止蚀 → 按入场日纪律持有，用即时阻力/支撑做节奏"
        )

    if live_bias:
        lines.append(f"【即时多空】{live_bias}（{live_bias_score:+.0f}）")
    return lines


@dataclass
class FollowLevels:
    """
    Single set of numbers the user should act on — no dual ambiguity.

    Split carefully:
    - now_do = what to do *immediately* (may already be triggered; not a future price)
    - wait_price = price still ahead if you are waiting; None if action is now
    - remain_target = next target for shares you still hold *after* now_do
    - far_target = level beyond remain_target (must differ from remain_target)
    """

    stage: str  # before_t1 | past_t1 | past_t2 | at_stop
    rule_one_liner: str
    hard_stop: float | None
    now_stop: float | None
    # Immediate action (not the same field as remain target)
    now_do: str
    now_do_detail: str
    # Future ladder for remaining position
    wait_price: float | None  # only if still waiting to act
    wait_label: str  # e.g. 等T1减半 | 剩仓看T2
    remain_target: float | None
    remain_label: str
    far_target: float | None
    far_label: str
    resistance: float | None
    next_why: str
    follow_note: str
    # Back-compat aliases used by older UI/tests
    next_price: float | None = None
    next_action: str = ""
    extend_target: float | None = None


def _levels_above(last: float | None, *levels: float | None) -> list[float]:
    if last is None:
        return []
    out: list[float] = []
    for x in levels:
        v = _f(x)
        if v is not None and v > last * 1.002:
            out.append(v)
    out.sort()
    return out


def build_follow_levels(
    *,
    buy_price: float,
    last_price: float,
    advice: PositionAdvice,
    entry_stop: float | None = None,
    entry_t1: float | None = None,
    entry_t2: float | None = None,
    entry_res: float | None = None,
    live_t1: float | None = None,
    live_t2: float | None = None,
    live_res: float | None = None,
) -> FollowLevels:
    """
    Collapse entry+live into one ladder.

    Critical UX: when T1 is already hit,「减半」is *now*, not at T2 price.
    T2/live targets are only for the *remaining* size after scale-out.
    """
    last = _f(last_price)
    e_s = _f(entry_stop)
    e_t1 = _f(entry_t1)
    e_t2 = _f(entry_t2)
    e_r = _f(entry_res)
    l_t1 = _f(live_t1)
    l_t2 = _f(live_t2)
    l_r = _f(live_res)
    now_stop = _f(advice.suggested_stop) or e_s

    rule = (
        "只跟这一套："
        "①现在立刻做的动作 ②现在止蚀 ③剩仓下一目标。"
        "T1 已到就立刻减半，不要等到 T2 才减。"
    )

    hit_t1 = e_t1 is not None and last is not None and last >= e_t1 * 0.998
    hit_t2 = e_t2 is not None and last is not None and last >= e_t2 * 0.998
    at_stop = (
        e_s is not None and last is not None and last <= e_s * 1.002
    ) or advice.action == "止蚀离场"

    # Upside ladder above last (for remaining / waiting)
    above = _levels_above(last, e_t1, e_t2, l_t1, l_t2)
    # Prefer entry T2 as first remain target when past T1
    remain: float | None = None
    far: float | None = None

    res_watch = None
    for r in (l_r, e_r):
        if r is not None and last is not None and r >= last * 0.995:
            if res_watch is None or r < res_watch:
                res_watch = r

    if at_stop:
        stage = "at_stop"
        now_do = "立刻离场"
        now_do_detail = f"已触/破硬止蚀 {_fmt_px(e_s)}，不要等反弹"
        wait_price, wait_label = None, "—"
        remain, remain_label = None, "—"
        far, far_label = None, "—"
        why = "破止蚀优先于任何目标"

    elif hit_t2 or advice.action == "止盈清仓":
        stage = "past_t2"
        now_do = "清仓或留极小仓"
        now_do_detail = (
            f"入场日 T2≈{_fmt_px(e_t2)} 已完成（或建议清仓）→ 现在锁定利润"
        )
        wait_price, wait_label = None, "—"
        remain, remain_label = None, "—"
        # only show far if live still has higher
        ups = _levels_above(last, l_t1, l_t2)
        far = ups[0] if ups else None
        far_label = "若留极小仓可看" if far else "—"
        why = "主目标已完成，剩仓仅用紧止蚀"

    elif hit_t1 or advice.action == "止盈减仓":
        # ★ 减半 = 现在做；1026 等是剩仓目标，绝不能写成「到1026才减半」
        stage = "past_t1"
        now_do = "现在减半（T1已到）"
        now_do_detail = (
            f"入场日 T1≈{_fmt_px(e_t1)} 已到/超过 → "
            f"**现在**减约一半，止蚀改到保本 {_fmt_px(now_stop)}；"
            f"不要等更高价才减"
        )
        wait_price, wait_label = None, "已触发·勿再等"
        # remain = nearest above last among entry T2 / live targets
        prefer: list[float] = []
        if e_t2 is not None and last is not None and e_t2 > last * 1.002:
            prefer.append(e_t2)
        prefer.extend(_levels_above(last, l_t1, l_t2))
        # unique sorted
        seen: set[float] = set()
        ladder: list[float] = []
        for p in prefer:
            key = round(p, 2)
            if key not in seen:
                seen.add(key)
                ladder.append(p)
        ladder.sort()
        remain = ladder[0] if ladder else None
        far = ladder[1] if len(ladder) > 1 else None
        if remain is not None and e_t2 is not None and abs(remain - e_t2) < 0.02:
            remain_label = "剩仓下一目标(入场T2)"
        elif remain is not None:
            remain_label = "剩仓下一目标"
        else:
            remain_label = "无更高目标·收紧止蚀"
        far_label = "更远目标" if far is not None else "—"
        why = (
            f"步骤拆开：①先减半（现在）②止蚀→{_fmt_px(now_stop)} "
            f"③剩仓拿到 {_fmt_px(remain) if remain else '—'}"
            + (f" ④更远 {_fmt_px(far)}" if far else "")
        )

    else:
        stage = "before_t1"
        now_do = "持有·守止蚀"
        now_do_detail = "入场日 T1 未到：先拿着，止蚀不降"
        # Wait for nearer of resistance (soft) or T1 (main)
        wait_cands: list[tuple[float, str]] = []
        if e_t1 is not None and last is not None and e_t1 > last:
            wait_cands.append((e_t1, "到T1再减半"))
        for r in (l_r, e_r):
            if (
                r is not None
                and last is not None
                and e_t1 is not None
                and last < r < e_t1
            ):
                wait_cands.append((r, "近阻力可小减"))
                break
        wait_cands.sort(key=lambda x: x[0])
        if wait_cands:
            wait_price, wait_label = wait_cands[0][0], wait_cands[0][1]
        else:
            wait_price, wait_label = e_t1, "到T1再减半"
        # remain after that event = T2 or live
        remain = None
        if e_t2 is not None and (wait_price is None or e_t2 > (wait_price or 0)):
            remain = e_t2
        remain_label = "T1之后的T2" if remain else "—"
        far_ups = _levels_above(remain or last, l_t1, l_t2)
        far = far_ups[0] if far_ups else None
        far_label = "更远" if far else "—"
        why = (
            f"未到 T1：等到 {_fmt_px(wait_price)}（{wait_label}）；"
            f"T2={_fmt_px(remain)}"
        )
        now_do = "持有"
        if wait_label.startswith("近阻力"):
            now_do_detail = (
                f"可先盯阻力 {_fmt_px(wait_price)} 小减；"
                f"主减仓仍是 T1≈{_fmt_px(e_t1)}"
            )

    note = (
        "「现在减半」和「剩仓目标」是两步："
        "T1 已到 → 立刻减；剩下的仓位才看 T2/更高价。"
        "不会出现「到 T2 才减半」这种混在一起的提示。"
    )

    # Back-compat for UI that still reads next_price / next_action / extend_target
    if stage == "past_t1":
        next_price = remain  # future only
        next_action = now_do
        extend_target = far if far is not None else remain
    elif stage == "before_t1":
        next_price = wait_price
        next_action = wait_label
        extend_target = remain
    else:
        next_price = wait_price or last
        next_action = now_do
        extend_target = far

    return FollowLevels(
        stage=stage,
        rule_one_liner=rule,
        hard_stop=e_s,
        now_stop=now_stop,
        now_do=now_do,
        now_do_detail=now_do_detail,
        wait_price=wait_price,
        wait_label=wait_label,
        remain_target=remain,
        remain_label=remain_label,
        far_target=far,
        far_label=far_label,
        resistance=res_watch,
        next_why=why,
        follow_note=note,
        next_price=next_price,
        next_action=next_action,
        extend_target=extend_target,
    )


def _fmt_px(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}"


def advise_dual_hold(
    *,
    buy_price: float,
    last_price: float,
    buy_date: str | None = None,
    shares: int | None = None,
    max_hold_days: int | None = None,
    bias_label: str = "",
    bias_score: float = 0.0,
    # entry-day
    entry_stop: float | None = None,
    entry_t1: float | None = None,
    entry_t2: float | None = None,
    entry_e: float | None = None,
    entry_res: float | None = None,
    entry_sup: float | None = None,
    entry_close: float | None = None,
    entry_as_of: str = "",
    # live
    live_stop: float | None = None,
    live_t1: float | None = None,
    live_t2: float | None = None,
    live_e: float | None = None,
    live_res: float | None = None,
    live_sup: float | None = None,
) -> tuple[PositionAdvice, list[str]]:
    """
    Hold coach using **entry-day** structure as primary contract,
    **live** levels as extension / trailing context.

    Returns (action advice, dual-analysis lines).
    """
    buy = _f(buy_price)
    last = _f(last_price)
    e_stop = _f(entry_stop)
    e_t1 = _f(entry_t1)
    e_t2 = _f(entry_t2)
    l_t1 = _f(live_t1)
    l_t2 = _f(live_t2)
    l_stop = _f(live_stop)

    # Hard stop: entry-day (never widen to a lower live stop)
    hard_stop = e_stop
    if hard_stop is None:
        hard_stop = l_stop
    elif l_stop is not None and l_stop > hard_stop:
        # live higher stop can inform trailing later; hard floor stays entry
        pass

    # Primary milestone = entry T1; extension target = farther of entry T2 / live T1
    coach_t1 = e_t1
    coach_t2 = e_t2
    if e_t2 is not None and l_t1 is not None:
        coach_t2 = max(e_t2, l_t1)
    elif l_t1 is not None and e_t2 is None:
        coach_t2 = l_t1
    if coach_t2 is not None and l_t2 is not None:
        coach_t2 = max(coach_t2, l_t2)

    dual_lines = analyze_entry_vs_live(
        buy_price=buy_price,
        last_price=last_price,
        entry_stop=e_stop,
        entry_t1=e_t1,
        entry_t2=e_t2,
        entry_e=_f(entry_e),
        entry_res=_f(entry_res),
        entry_sup=_f(entry_sup),
        entry_close=_f(entry_close),
        live_stop=l_stop,
        live_t1=l_t1,
        live_t2=l_t2,
        live_e=_f(live_e),
        live_res=_f(live_res),
        live_sup=_f(live_sup),
        entry_as_of=entry_as_of or "",
        live_bias=bias_label,
        live_bias_score=bias_score,
    )

    advice = advise_open_position(
        buy_price=buy_price,
        last_price=last_price,
        plan_stop=hard_stop,
        plan_t1=coach_t1,
        plan_t2=coach_t2,
        plan_entry=_f(entry_e) or _f(live_e) or buy,
        max_hold_days=max_hold_days,
        buy_date=buy_date,
        shares=shares,
        bias_label=bias_label,
        bias_score=bias_score,
    )

    # Enrich: resistance-aware scale-out nudge (doesn't override hard stop exit)
    extra: list[str] = []
    if buy is not None and last is not None and advice.action not in ("止蚀离场",):
        e_res = _f(entry_res)
        l_res = _f(live_res)
        if (
            e_t1 is not None
            and last >= e_t1 * 0.998
            and l_t1 is not None
            and l_t1 > e_t1 * 1.02
        ):
            extra.append(
                f"入场日 T1 已完成；即时给出更高 T1≈{l_t1:.2f} → 减仓后剩仓可看此延伸目标"
            )
        if l_res is not None and abs(last - l_res) / last < 0.012:
            if advice.action == "持有":
                advice = PositionAdvice(
                    action="持有·注意阻力",
                    color="amber",
                    headline=(
                        f"贴近即时阻力 {l_res:.2f}：可小减或收紧止蚀，"
                        f"勿在阻力位加仓"
                    ),
                    pnl_pct=advice.pnl_pct,
                    pnl_r=advice.pnl_r,
                    dist_to_stop_pct=advice.dist_to_stop_pct,
                    dist_to_t1_pct=advice.dist_to_t1_pct,
                    dist_to_t2_pct=advice.dist_to_t2_pct,
                    suggested_stop=advice.suggested_stop,
                    bullets=list(advice.bullets),
                    summary=f"**持有·注意阻力** · 贴近即时阻力 {l_res:.2f}",
                )
            extra.append(f"即时阻力 {l_res:.2f} 近在眼前，分批优先于一把清/一把加")
        if e_res is not None and last >= e_res * 0.998 and (
            e_t1 is None or last < e_t1 * 0.995
        ):
            extra.append(
                f"已过入场日阻力 {e_res:.2f}、未到 T1 → 浮盈宜上移止蚀，不追价加仓"
            )

        # If live stop is higher and we're in profit, bump suggested trail toward live structure
        if (
            advice.suggested_stop is not None
            and l_stop is not None
            and e_stop is not None
            and last > buy
            and l_stop > advice.suggested_stop
            and l_stop < last * 0.995
            and advice.action in ("持有", "持有·上移止蚀", "持有·注意阻力")
        ):
            # don't jump above lock-in trail from coach; take max of suggested and a fraction toward live
            bumped = min(l_stop, buy + 0.5 * (last - buy))
            if bumped > advice.suggested_stop:
                advice.suggested_stop = round(bumped, 4)
                extra.append(f"结合即时结构，建议止蚀可抬到 ≈ {advice.suggested_stop:.2f}")

    if extra:
        advice.bullets = list(advice.bullets) + extra
        advice.summary = f"**{advice.action}** · {advice.headline}"

    # Prepend dual summary line into bullets for single-block UI
    top = [ln for ln in dual_lines if ln.startswith("【综合】")]
    rest_dual = [ln for ln in dual_lines if not ln.startswith("【综合】")]
    advice.bullets = top + list(advice.bullets)

    return advice, rest_dual + top
