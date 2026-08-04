"""Main vertical-spread analysis entrypoint."""
from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from options_builders import (
    build_bear_call,
    build_bear_put,
    build_bull_call,
    build_bull_put,
)
from options_chain import (
    _atm_iv,
    _dte,
    _get_cached_chain,
    _hist_vol,
    _is_us_rth,
    _parse_chain,
    _pick_expiry,
    half_profit_close_price,
    passes_hard_filters,
    suggest_width,
)
from options_direction import analyze_direction
from options_models import (
    INDEX_ETF_WHITELIST,
    OptionsReport,
    is_options_eligible,
    options_symbol,
)
from options_scoring import _attach_win_rates, score_liquidity


def analyze_options_spreads(
    symbol: str,
    target_dte: int = 30,
    width: float | None = None,
    hist_df: pd.DataFrame | None = None,
    bias_label: str | None = None,
) -> OptionsReport:
    """Vertical-only analysis with direction + best spread recommendation."""
    del bias_label  # direction computed from hist_df
    sym = options_symbol(symbol)
    label = INDEX_ETF_WHITELIST.get(sym, "")
    if not is_options_eligible(sym):
        return OptionsReport(
            symbol=sym,
            label=label or "非白名单",
            spot=0.0,
            eligible=False,
            message=(
                f"`{sym}` 不在白名单。请用：{', '.join(sorted(INDEX_ETF_WHITELIST))}。"
            ),
            summary="标的不适合。",
        )

    direction = analyze_direction(hist_df) if hist_df is not None else analyze_direction(pd.DataFrame())

    ticker = yf.Ticker(sym)
    spot = None
    try:
        fi = ticker.fast_info
        spot = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
    except Exception:
        spot = None
    if not spot:
        try:
            h = ticker.history(period="5d")
            if h is not None and not h.empty:
                spot = float(h["Close"].iloc[-1])
        except Exception:
            pass
    if not spot:
        return OptionsReport(
            symbol=sym,
            label=label,
            spot=0.0,
            eligible=True,
            message="无法获取现价。",
            direction=direction,
            summary="无现价。",
        )
    spot = float(spot)

    try:
        exps = list(ticker.options or [])
    except Exception:
        exps = []
    if not exps:
        return OptionsReport(
            symbol=sym,
            label=label,
            spot=spot,
            eligible=True,
            message="无期权到期日。",
            direction=direction,
            summary="无期权链。",
        )

    expiry = _pick_expiry(exps, target_dte=target_dte)
    if not expiry:
        return OptionsReport(
            symbol=sym,
            label=label,
            spot=spot,
            eligible=True,
            expiries=exps[:16],
            message="无合适到期日。",
            direction=direction,
            summary="无到期日。",
        )
    dte = _dte(expiry)
    after_hours = not _is_us_rth()

    try:
        raw = _get_cached_chain(sym, expiry)
        calls = _parse_chain(raw.get("calls"), spot, "call")
        puts = _parse_chain(raw.get("puts"), spot, "put")
    except Exception as exc:
        return OptionsReport(
            symbol=sym,
            label=label,
            spot=spot,
            eligible=True,
            expiries=exps[:16],
            selected_expiry=expiry,
            dte=dte,
            direction=direction,
            message=f"期权链失败：{exc}",
            summary="期权链失败。",
            after_hours=after_hours,
        )

    # 报价质量：有多少腿有有效 bid/ask
    def _quote_coverage(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 0.0
        has = (df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0)
        return float(has.mean()) if len(df) else 0.0

    cov = 0.5 * (_quote_coverage(calls) + _quote_coverage(puts))
    quote_warning = ""
    pricing_note = "开仓按自然成交估算：卖腿=买价(bid)，买腿=卖价(ask)；比中间价更保守。"
    if after_hours:
        quote_warning = (
            "当前可能不在美股常规交易时段（美东 09:30–16:00）。"
            "盘后 bid/ask 常为空或失真，Yahoo lastPrice 也易过期——"
            "权利金仅供参考，请在盘中用限价单核验。"
        )
        pricing_note += " 盘后若无买卖价则回退中间价/最新价。"
    elif cov < 0.35:
        quote_warning = (
            "当前期权链买卖价覆盖偏低，部分腿用中间价估算，成交价可能偏差较大。"
        )
        pricing_note += " 部分腿无有效 bid/ask，已回退 mid。"

    iv_atm = _atm_iv(calls, puts, spot)
    if iv_atm is None or iv_atm < 0.05:
        hv = _hist_vol(sym, 30)
        if hv:
            iv_atm = hv

    w = width or suggest_width(spot)
    ideas: list[SpreadIdea] = []

    # Scan several OTM / offsets for credit & debit verticals
    for otm in (0.015, 0.025, 0.035, 0.05):
        for builder, args in (
            (build_bull_put, dict(puts=puts, spot=spot, width=w, otm_pct=otm, expiry=expiry, dte=dte)),
            (build_bear_call, dict(calls=calls, spot=spot, width=w, otm_pct=otm, expiry=expiry, dte=dte)),
        ):
            try:
                idea = builder(**args)
                if idea:
                    ideas.append(idea)
            except Exception:
                pass

    for off in (-0.01, 0.0, 0.01, 0.02):
        for builder, args in (
            (build_bull_call, dict(calls=calls, spot=spot, width=w, long_offset_pct=off, expiry=expiry, dte=dte)),
            (build_bear_put, dict(puts=puts, spot=spot, width=w, long_offset_pct=off, expiry=expiry, dte=dte)),
        ):
            try:
                idea = builder(**args)
                if idea:
                    ideas.append(idea)
            except Exception:
                pass

    # Deduplicate by code+strikes
    uniq: dict[str, SpreadIdea] = {}
    for idea in ideas:
        key = (
            f"{idea.code}:"
            + "-".join(f"{lg.side[0]}{lg.right[0]}{lg.strike:.0f}" for lg in idea.legs)
        )
        prev = uniq.get(key)
        if prev is None or idea.score > prev.score:
            uniq[key] = idea
    ideas = list(uniq.values())

    # Attach win rates + liquidity + 热门规则评分
    ideas = [_attach_win_rates(i, spot, iv_atm) for i in ideas]
    ideas = [score_liquidity(i) for i in ideas]
    try:
        from options_methods import enrich_idea_with_methods

        ideas = [enrich_idea_with_methods(i, spot, dte) for i in ideas]
    except Exception:
        pass

    # 补全 50% 止盈买回价（enrich 可能覆盖 metric_half_profit）
    for idea in ideas:
        bb = half_profit_close_price(idea)
        if bb is not None:
            idea.metric_half_buyback = bb
        if idea.metric_half_profit is None:
            idea.metric_half_profit = round(float(idea.max_profit) * 0.5, 1)

    # 硬过滤：流动性 + 权利金/宽度
    before_n = len(ideas)
    ideas = [i for i in ideas if passes_hard_filters(i)]
    filtered_out = before_n - len(ideas)

    # Align score with direction
    preferred = set(direction.preferred_verticals)
    for idea in ideas:
        boost = 0.0
        if idea.code in preferred:
            boost += 18
            if preferred and idea.code == direction.preferred_verticals[0]:
                boost += 10
        # Penalize opposite side
        bull_codes = {"bull_put", "bull_call"}
        bear_codes = {"bear_call", "bear_put"}
        if direction.direction == "看多" and idea.code in bear_codes:
            boost -= 20
        if direction.direction == "看空" and idea.code in bull_codes:
            boost -= 20
        if direction.direction == "中性":
            if idea.structure == "Credit Vertical":
                boost += 8
            else:
                boost -= 5
        # Mild bull: prefer credit over expensive debit when RSI high
        if direction.direction == "看多" and direction.strength != "强" and idea.code == "bull_put":
            boost += 6
        if direction.direction == "看空" and direction.strength != "强" and idea.code == "bear_call":
            boost += 6

        idea.score = float(max(0.0, min(100.0, idea.score + boost)))
        idea.rank_reason = (
            f"方向={direction.direction}；结构={idea.structure}；"
            f"{'契合首选' if idea.code in preferred else '次选/对冲'}"
        )

    ideas.sort(key=lambda x: x.score, reverse=True)

    best = ideas[0] if ideas else None
    best_alt = None
    if best and len(ideas) > 1:
        for alt in ideas[1:]:
            if alt.code != best.code or alt.structure != best.structure:
                best_alt = alt
                break
        if best_alt is None:
            best_alt = ideas[1]

    # 最高胜率（有利润）
    def _wr(i: SpreadIdea) -> float:
        v = getattr(i, "win_rate_profit", None)
        if v is None:
            v = getattr(i, "pop_est", None)
        return float(v) if v is not None else -1.0

    with_wr = [i for i in ideas if _wr(i) >= 0]
    best_winrate = max(with_wr, key=_wr) if with_wr else None

    # 与当前方向契合的最高胜率
    bull_codes = {"bull_put", "bull_call"}
    bear_codes = {"bear_call", "bear_put"}
    if direction.direction == "看多":
        aligned_pool = [i for i in with_wr if i.code in bull_codes]
    elif direction.direction == "看空":
        aligned_pool = [i for i in with_wr if i.code in bear_codes]
    else:
        aligned_pool = [i for i in with_wr if i.structure == "Credit Vertical"] or with_wr
    best_winrate_aligned = max(aligned_pool, key=_wr) if aligned_pool else best_winrate

    # 参考市面常见策略规则 → 最佳 / 高赢面
    best_playbook = best_playbook_wr = None
    playbook_table: list = []
    try:
        from options_strategy_book import apply_playbook_ranking

        best_playbook, best_playbook_wr, playbook_table = apply_playbook_ranking(
            ideas, spot, dte, direction.direction
        )
        # 用实战分重排展示顺序（保留原 score）
        ideas = sorted(
            ideas,
            key=lambda x: getattr(x, "playbook_combo", x.score),
            reverse=True,
        )
        if best_playbook is not None:
            best = best_playbook
    except Exception:
        playbook_table = []

    if best_winrate:
        for i, idea in enumerate(sorted(with_wr, key=_wr, reverse=True), start=1):
            idea.rank_reason = (
                (idea.rank_reason + "；" if idea.rank_reason else "")
                + f"赢面排名#{i}（{_wr(idea):.1f}%）"
            )

    if iv_atm is not None:
        if iv_atm >= 0.28:
            regime = "波动偏高 · 信用价差权利金厚但尾部风险大"
        elif iv_atm >= 0.16:
            regime = "波动适中 · 适合标准 vertical"
        else:
            regime = "波动偏低 · 信用价差收租薄，debit 也不宜追贵"
    else:
        regime = "波动数据有限"

    action_plan: list[str] = []
    if best and direction:
        action_plan.append(f"1. 方向结论：{direction.direction}（{direction.strength}）")
        action_plan.append(f"2. 首选 vertical：{best.name}")
        if best.net_credit is not None:
            action_plan.append(
                f"3. 开仓方式：信用价差，目标净收 ≥ ${best.net_credit * 0.9:.2f}（限价单）"
            )
        else:
            action_plan.append(
                f"3. 开仓方式：借方价差，目标净付 ≤ ${best.net_debit * 1.1:.2f}（限价单）"
            )
        if best.win_rate_profit is not None:
            action_plan.append(
                f"4. 预估胜率：有利润 ≈ {best.win_rate_profit:.1f}%"
                + (
                    f"，满盈 ≈ {best.win_rate_max:.1f}%"
                    if best.win_rate_max is not None
                    else ""
                )
                + (
                    f"；粗期望 ≈ ${best.expected_value:.0f}/张"
                    if best.expected_value is not None
                    else ""
                )
            )
            action_plan.append(
                f"5. 风控：最大亏损约 ${best.max_loss:.0f}/张；建议风险不超过账户 1%–2%"
            )
            action_plan.append(
                "6. 管理：盈利达最大利润 50%–70% 可提前平仓；靠近短腿考虑止损/调仓"
            )
            n = 7
        else:
            action_plan.append(
                f"4. 风控：最大亏损约 ${best.max_loss:.0f}/张；建议风险不超过账户 1%–2%"
            )
            action_plan.append(
                "5. 管理：盈利达最大利润 50%–70% 可提前平仓；靠近短腿考虑止损/调仓"
            )
            n = 6
        if best_alt:
            alt_wr = (
                f"，胜率≈{best_alt.win_rate_profit:.1f}%"
                if best_alt.win_rate_profit is not None
                else ""
            )
            action_plan.append(f"{n}. 备选：{best_alt.name}{alt_wr}")
            n += 1
        if best_winrate:
            action_plan.append(
                f"{n}. 最高胜率（全体）：{best_winrate.name}"
                f" ≈ {best_winrate.win_rate_profit:.1f}%"
                + (
                    f"（满盈 {best_winrate.win_rate_max:.1f}%）"
                    if best_winrate.win_rate_max is not None
                    else ""
                )
            )
            n += 1
        if (
            best_winrate_aligned
            and best_winrate
            and best_winrate_aligned is not best_winrate
        ):
            action_plan.append(
                f"{n}. 最高胜率（贴合{direction.direction}）："
                f"{best_winrate_aligned.name} ≈ {best_winrate_aligned.win_rate_profit:.1f}%"
            )

    if best:
        wr_txt = (
            f"，赢面≈{best.win_rate_profit:.0f}%"
            if best.win_rate_profit is not None
            else ""
        )
        style = getattr(best, "playbook_style", "") or ""
        style_txt = f"参考「{style}」。" if style else ""
        hi_txt = ""
        use_wr = best_playbook_wr or best_winrate_aligned or best_winrate
        if use_wr and getattr(use_wr, "win_rate_profit", None) is not None:
            hi_txt = (
                f" 最高赢面：**{use_wr.name}**（{use_wr.win_rate_profit:.0f}%"
                f"{' · ' + getattr(use_wr, 'playbook_style', '') if getattr(use_wr, 'playbook_style', '') else ''}）"
            )
        summary = (
            f"{sym} @ {spot:.2f} → 方向 **{direction.direction}**。"
            f"实战推荐：**{best.name}**"
            f"（{wr_txt.lstrip('，')}"
            + (
                f"，收 ${best.net_credit:.2f}"
                if best.net_credit is not None
                else f"，付 ${best.net_debit:.2f}"
            )
            + f"，到期 {expiry}）。"
            + style_txt
            + hi_txt
        )
        # 白话执行
        half_bb = getattr(best, "metric_half_buyback", None)
        half_p = getattr(best, "metric_half_profit", None)
        if best.net_credit is not None:
            open_line = f"3. 今天：卖出价差，大约收 ${best.net_credit:.2f}/股（卖=bid/买=ask）"
            if half_bb is not None:
                close_line = (
                    f"4. 50%止盈：价差买回约 ${half_bb:.2f}/股"
                    + (f"（约赚 ${half_p:.0f}/张）" if half_p is not None else "")
                )
            else:
                close_line = "4. 几天后：买回平仓"
        else:
            open_line = f"3. 今天：买进价差，大约付 ${best.net_debit:.2f}/股（买=ask/卖=bid）"
            if half_bb is not None:
                close_line = (
                    f"4. 50%止盈：价差卖出约 ${half_bb:.2f}/股"
                    + (f"（约赚 ${half_p:.0f}/张）" if half_p is not None else "")
                )
            else:
                close_line = "4. 几天后：卖出平仓"
        if best.win_rate_profit is not None:
            close_line += f"；估算赢面约 {best.win_rate_profit:.0f}%"
        action_plan = [
            f"1. 方向：{direction.direction}（{direction.strength}）",
            f"2. 实战推荐：{best.name}"
            + (f" —— 像「{style}」" if style else ""),
            open_line,
            close_line,
            f"5. 最多赚约 ${best.max_profit:.0f}/张，最多亏约 ${best.max_loss:.0f}/张",
        ]
        if use_wr and use_wr is not best:
            action_plan.append(
                f"6. 若只要最高赢面：改用 {use_wr.name}"
                f"（约 {use_wr.win_rate_profit:.0f}%）"
            )
        plain = getattr(best, "playbook_plain", "")
        src = getattr(best, "playbook_source", "")
        if plain:
            action_plan.append(f"7. 白话：{plain}")
        if src:
            action_plan.append(f"8. 规则参考：{src}（实盘规则筛，非荐股）")
    else:
        summary = f"{sym} @ {spot:.2f}，方向 {direction.direction}，未能生成可用 vertical。"

    msg = "仅分析 Vertical Spread；" + pricing_note
    if filtered_out > 0:
        msg += f" 已硬过滤剔除 {filtered_out} 个（流动性差或权利金/宽度不合规）。"

    from options_timing import assess_spread_timing

    timing = assess_spread_timing(
        direction=direction,
        best=best,
        after_hours=after_hours,
        dte=dte,
        iv_atm=iv_atm,
        ideas_count=len(ideas),
        quote_warning=quote_warning,
        pricing_note=pricing_note,
    )
    action_plan = list(action_plan)
    action_plan.insert(
        0,
        f"0. 适合做 spread？→ {timing.verdict}（{timing.score:.0f}/100）· {timing.action}",
    )

    return OptionsReport(
        symbol=sym,
        label=label,
        spot=spot,
        eligible=True,
        message=msg,
        direction=direction,
        expiries=exps[:16],
        selected_expiry=expiry,
        dte=dte,
        iv_atm=iv_atm,
        ideas=ideas,
        best=best,
        best_alt=best_alt,
        best_winrate=best_winrate,
        best_winrate_aligned=best_winrate_aligned,
        best_playbook=best_playbook,
        best_playbook_wr=best_playbook_wr,
        playbook_table=playbook_table,
        regime=regime,
        summary=summary,
        action_plan=action_plan,
        after_hours=after_hours,
        pricing_note=pricing_note,
        quote_warning=quote_warning,
        filtered_out=filtered_out,
        timing=timing,
    )
