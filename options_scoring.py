"""Win-rate / expected-value estimates and liquidity scoring for verticals."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from options_models import SpreadIdea

# Default assumptions for live-trading assist (documented in UI / method string)
DEFAULT_RF = 0.04  # risk-free for discounting EV
DEFAULT_SLIP_PER_SHARE = 0.03  # round-trip friction buffer $/share (2-leg)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _prob_above(
    spot: float,
    strike: float,
    sigma: float,
    t_years: float,
    mu: float = 0.0,
) -> float:
    """P(S_T > K) under lognormal with drift mu."""
    if spot <= 0 or strike <= 0 or sigma <= 0 or t_years <= 0:
        return 0.5
    vol_sq = sigma * math.sqrt(t_years)
    if vol_sq < 1e-9:
        return 1.0 if spot > strike else 0.0
    d2 = (math.log(spot / strike) + (mu - 0.5 * sigma * sigma) * t_years) / vol_sq
    return float(_norm_cdf(d2))


def _prob_below(
    spot: float,
    strike: float,
    sigma: float,
    t_years: float,
    mu: float = 0.0,
) -> float:
    return 1.0 - _prob_above(spot, strike, sigma, t_years, mu)


def _payoff_per_share_vertical(
    code: str,
    s: float,
    short_k: float,
    long_k: float,
    credit: float | None,
    debit: float | None,
) -> float:
    """
    Expiry P&L per share for a standard vertical (not ×100).

    Credit: received premium, short higher risk; Debit: paid premium.
    """
    if code == "bull_put":
        # short put Ks > long put Kl, credit C
        c = float(credit or 0.0)
        ks, kl = float(short_k), float(long_k)
        if s >= ks:
            return c
        if s <= kl:
            return c - (ks - kl)
        return c - (ks - s)
    if code == "bear_call":
        c = float(credit or 0.0)
        ks, kl = float(short_k), float(long_k)  # ks < kl
        if s <= ks:
            return c
        if s >= kl:
            return c - (kl - ks)
        return c - (s - ks)
    if code == "bull_call":
        d = float(debit or 0.0)
        kl, ks = float(long_k), float(short_k)  # long low, short high
        if s <= kl:
            return -d
        if s >= ks:
            return (ks - kl) - d
        return (s - kl) - d
    if code == "bear_put":
        d = float(debit or 0.0)
        kl, ks = float(long_k), float(short_k)  # long high put, short low
        if s >= kl:
            return -d
        if s <= ks:
            return (kl - ks) - d
        return (kl - s) - d
    return 0.0


def _expected_payoff_lognormal(
    spot: float,
    sigma: float,
    t_years: float,
    mu: float,
    payoff_fn: Callable[[float], float],
    n_points: int = 96,
) -> float:
    """
    E[payoff(S_T)] under S_T = spot * exp((mu - 0.5 σ²)t + σ√t Z), Z~N(0,1).

    Gauss-Hermite style grid on standard normal (deterministic, no RNG).
    """
    if spot <= 0 or sigma <= 0 or t_years <= 0:
        return float(payoff_fn(spot))

    # z in [-4.5, 4.5]
    zs = np.linspace(-4.5, 4.5, n_points)
    # Discrete distribution proportional to normal density on the grid
    pdf = np.array([_norm_pdf(float(z)) for z in zs], dtype=float)
    mass = float(np.sum(pdf))
    if mass <= 0:
        return float(payoff_fn(spot))
    pdf = pdf / mass

    drift = (mu - 0.5 * sigma * sigma) * t_years
    vol = sigma * math.sqrt(t_years)
    total = 0.0
    for z, p in zip(zs, pdf):
        s = spot * math.exp(drift + vol * float(z))
        total += float(payoff_fn(s)) * float(p)
    return float(total)


def estimate_vertical_win_rates(
    code: str,
    spot: float,
    short_strike: float | None,
    long_strike: float | None,
    breakeven: float | None,
    sigma: float | None,
    dte: int,
    max_profit: float,
    max_loss: float,
    *,
    credit: float | None = None,
    debit: float | None = None,
    mu: float = 0.0,
    r: float = DEFAULT_RF,
    slip_per_share: float = DEFAULT_SLIP_PER_SHARE,
) -> tuple[float | None, float | None, str, float | None]:
    """
    Live-assist POP / max-profit probability / expected value.

    Returns (win_rate_profit%, win_rate_max%, method, expected_value$/contract).

    Model (documented for traders):
    - Terminal price lognormal with drift μ (default 0 = path-neutral)
    - POP = P(expiry P&L > 0) via breakeven threshold
    - EV = e^{-rT} · E[expiry P&L] under full piecewise vertical payoff
      (not the old win*0.55*maxP - lose*maxL shortcut)
    - Slip buffer subtracted from EV (default $0.03/share round-trip)
    """
    if not sigma or sigma <= 0 or dte <= 0 or spot <= 0:
        return None, None, "波动率/天数不足，无法估胜率", None
    if short_strike is None or long_strike is None:
        return None, None, "缺少行权价，无法估胜率", None

    t = max(dte, 1) / 365.0
    sig = max(0.08, min(float(sigma), 0.80))
    ks, kl = float(short_strike), float(long_strike)

    # Infer credit/debit from max P/L if not passed
    width = abs(ks - kl)
    if credit is None and debit is None:
        if code in ("bull_put", "bear_call"):
            credit = float(max_profit) / 100.0 if max_profit else None
        else:
            debit = float(max_loss) / 100.0 if max_loss else None

    method = (
        f"实盘模型 · 对数正态到期 · σ≈{sig * 100:.1f}% · {dte}日 · "
        f"μ={mu:.0%}路径 · EV按分段盈亏积分并折现r={r:.0%} · "
        f"已扣摩擦≈${slip_per_share * 100:.0f}/张"
    )

    wr_p = wr_m = None
    if code == "bull_put":
        if breakeven:
            wr_p = _prob_above(spot, float(breakeven), sig, t, mu=mu) * 100
        wr_m = _prob_above(spot, ks, sig, t, mu=mu) * 100
    elif code == "bear_call":
        if breakeven:
            wr_p = _prob_below(spot, float(breakeven), sig, t, mu=mu) * 100
        wr_m = _prob_below(spot, ks, sig, t, mu=mu) * 100
    elif code == "bull_call":
        if breakeven:
            wr_p = _prob_above(spot, float(breakeven), sig, t, mu=mu) * 100
        wr_m = _prob_above(spot, ks, sig, t, mu=mu) * 100
    elif code == "bear_put":
        if breakeven:
            wr_p = _prob_below(spot, float(breakeven), sig, t, mu=mu) * 100
        wr_m = _prob_below(spot, ks, sig, t, mu=mu) * 100

    if wr_p is not None:
        wr_p = float(np.clip(wr_p, 0.5, 99.5))
    if wr_m is not None:
        wr_m = float(np.clip(wr_m, 0.5, 99.5))

    def _pf(s: float) -> float:
        return _payoff_per_share_vertical(code, s, ks, kl, credit, debit)

    # Expected $/share at expiry, then discount + friction, then ×100 contract
    ev_share = _expected_payoff_lognormal(spot, sig, t, mu, _pf)
    disc = math.exp(-max(r, 0.0) * t)
    ev_contract = disc * ev_share * 100.0 - abs(slip_per_share) * 100.0

    # Sanity: clamp EV inside [-max_loss, max_profit] ± slip
    if max_profit > 0 and max_loss > 0:
        lo = -float(max_loss) - abs(slip_per_share) * 100.0
        hi = float(max_profit)
        ev_contract = float(min(hi, max(lo, ev_contract)))

    return (
        round(wr_p, 1) if wr_p is not None else None,
        round(wr_m, 1) if wr_m is not None else None,
        method,
        round(ev_contract, 1),
    )


def estimate_managed_ev(
    code: str,
    spot: float,
    short_strike: float,
    long_strike: float,
    sigma: float,
    dte: int,
    max_profit: float,
    max_loss: float,
    *,
    credit: float | None = None,
    debit: float | None = None,
    tp_pct: float = 0.5,
    stop_r: float = 2.0,
    slip_per_share: float = DEFAULT_SLIP_PER_SHARE,
    commission_rt: float = 2.6,
    n_paths: int = 41,
) -> tuple[float | None, str]:
    """
    Path EV under active management (实盘常用):

    - Credit: daily BS mark; exit at +tp_pct·max_profit or −stop_r·max_loss
    - Debit: same using debit mark (long value)
    - Paths: linear spot paths to lognormal terminals (deterministic grid, no RNG)

    Returns (EV$/contract, method_note).
    """
    if sigma <= 0 or dte <= 0 or spot <= 0 or max_loss <= 0:
        return None, "参数不足，无法估管理期望"

    from options_payoff import bs_option_price

    sig = max(0.08, min(float(sigma), 0.80))
    ks, kl = float(short_strike), float(long_strike)
    is_credit = code in ("bull_put", "bear_call")
    opt = "put" if code in ("bull_put", "bear_put") else "call"

    if is_credit:
        entry = float(credit if credit is not None else max_profit / 100.0)
    else:
        entry = float(debit if debit is not None else max_loss / 100.0)

    tp = tp_pct * float(max_profit)
    sl = stop_r * float(max_loss)
    slip = abs(slip_per_share)
    comm = abs(commission_rt)
    days = max(int(dte), 1)

    def mark_value(s: float, dte_left: int) -> float:
        t = max(dte_left, 0) / 365.0
        if t <= 1e-8:
            # intrinsic vertical width consumption
            if opt == "put":
                hi, lo = max(ks, kl), min(ks, kl)
                if s >= hi:
                    width_used = 0.0
                elif s <= lo:
                    width_used = hi - lo
                else:
                    width_used = hi - s
            else:
                lo, hi = min(ks, kl), max(ks, kl)
                if s <= lo:
                    width_used = 0.0
                elif s >= hi:
                    width_used = hi - lo
                else:
                    width_used = s - lo
            if is_credit:
                return max(width_used, 0.0)  # buyback cost
            return max((hi - lo) - width_used if opt == "call" else (hi - lo) - width_used, 0.0)

        if is_credit:
            # short premium - long premium
            if code == "bull_put":
                return max(
                    bs_option_price(s, ks, t, sig, "put")
                    - bs_option_price(s, kl, t, sig, "put"),
                    0.0,
                )
            # bear_call: short lower call ks, long higher kl
            return max(
                bs_option_price(s, ks, t, sig, "call")
                - bs_option_price(s, kl, t, sig, "call"),
                0.0,
            )
        # debit mark = long - short
        if code == "bull_call":
            return max(
                bs_option_price(s, kl, t, sig, "call")
                - bs_option_price(s, ks, t, sig, "call"),
                0.0,
            )
        return max(
            bs_option_price(s, kl, t, sig, "put")
            - bs_option_price(s, ks, t, sig, "put"),
            0.0,
        )

    def path_pnl(s_end: float) -> float:
        pnl = 0.0
        for day in range(1, days + 1):
            frac = day / days
            s_t = spot + (s_end - spot) * frac
            dte_left = max(days - day, 0)
            m = mark_value(s_t, dte_left)
            if is_credit:
                pnl = (entry - m - slip) * 100.0 - comm
            else:
                pnl = (m - entry - slip) * 100.0 - comm
            if pnl >= tp:
                return pnl
            if pnl <= -sl:
                return pnl
        return pnl

    zs = np.linspace(-3.5, 3.5, n_paths)
    pdf = np.array([_norm_pdf(float(z)) for z in zs], dtype=float)
    pdf = pdf / float(np.sum(pdf))
    t_years = days / 365.0
    drift = -0.5 * sig * sig * t_years  # μ=0
    vol = sig * math.sqrt(t_years)

    total = 0.0
    for z, p in zip(zs, pdf):
        s_end = spot * math.exp(drift + vol * float(z))
        total += path_pnl(s_end) * float(p)

    method = (
        f"管理期望 · 线性现货路径至对数正态终点 · 日估BS盯市 · "
        f"止盈{tp_pct:.0%}满盈 / 止损{stop_r:.0f}R · 摩擦+佣金≈"
        f"${slip * 100 + comm:.0f}/张"
    )
    return round(float(total), 1), method


def score_liquidity(idea: SpreadIdea) -> SpreadIdea:
    """
    Score liquidity from OI, volume, bid/ask.
    After hours often lacks NBBO — falls back to OI/volume with lower confidence.
    """
    if not idea.legs:
        idea.liquidity_score = 0.0
        idea.liquidity_label = "未知"
        idea.liquidity_detail = "没有腿数据"
        return idea

    oi_sum = 0.0
    vol_sum = 0.0
    spread_pcts: list[float] = []
    has_quote = 0
    n = len(idea.legs)

    for leg in idea.legs:
        oi_sum += float(leg.oi or 0)
        vol_sum += float(leg.volume or 0)
        bid = float(leg.bid or 0)
        ask = float(leg.ask or 0)
        mid = float(leg.mid or 0)
        if bid > 0 and ask > 0 and mid > 0:
            has_quote += 1
            spread_pcts.append((ask - bid) / mid)

    avg_oi = oi_sum / max(n, 1)
    oi_score = min(100.0, avg_oi / 8.0)
    avg_vol = vol_sum / max(n, 1)
    vol_score = min(100.0, avg_vol / 1.5)
    if spread_pcts:
        avg_sp = sum(spread_pcts) / len(spread_pcts)
        ba_score = float(max(0.0, min(100.0, 100.0 - avg_sp * 400.0)))
    else:
        ba_score = 40.0
        avg_sp = None

    if has_quote == n:
        score = 0.45 * ba_score + 0.30 * oi_score + 0.25 * vol_score
    elif has_quote > 0:
        score = 0.30 * ba_score + 0.40 * oi_score + 0.30 * vol_score
    else:
        score = 0.55 * oi_score + 0.45 * vol_score
        if oi_sum < 50 and vol_sum < 20:
            score = min(score, 28.0)

    score = float(np.clip(score, 0, 100))
    if score >= 65:
        label = "高"
    elif score >= 42:
        label = "中"
    elif score >= 25:
        label = "低"
    else:
        label = "很差"

    parts = [
        f"未平仓合计约 {oi_sum:.0f}",
        f"成交量合计约 {vol_sum:.0f}",
    ]
    if avg_sp is not None:
        parts.append(f"买卖价差约中间价的 {avg_sp * 100:.0f}%")
    else:
        parts.append("盘后可能无买卖价（用未平仓/量估算）")
    parts.append(f"流动性：{label}（{score:.0f}分）")

    idea.liquidity_score = round(score, 1)
    idea.liquidity_label = label
    idea.liquidity_detail = "；".join(parts)
    idea.score = round(float(np.clip(idea.score * 0.85 + score * 0.15, 0, 100)), 1)
    return idea


def _attach_win_rates(idea: SpreadIdea, spot: float, sigma: float | None) -> SpreadIdea:
    short_k = long_k = None
    for leg in idea.legs:
        if leg.side == "sell":
            short_k = leg.strike
        else:
            long_k = leg.strike
    be = idea.breakevens[0] if idea.breakevens else None
    wr_p, wr_m, method, ev = estimate_vertical_win_rates(
        idea.code,
        spot,
        short_k,
        long_k,
        be,
        sigma,
        idea.dte,
        idea.max_profit,
        idea.max_loss,
        credit=idea.net_credit,
        debit=idea.net_debit,
    )
    idea.win_rate_profit = wr_p
    idea.win_rate_max = wr_m
    idea.win_rate_method = method
    idea.expected_value = ev
    idea.pop_est = wr_p

    ev_m = None
    if short_k is not None and long_k is not None and sigma and sigma > 0:
        ev_m, m_method = estimate_managed_ev(
            idea.code,
            spot,
            float(short_k),
            float(long_k),
            float(sigma),
            int(idea.dte or 30),
            float(idea.max_profit),
            float(idea.max_loss),
            credit=idea.net_credit,
            debit=idea.net_debit,
        )
        idea.expected_value_managed = ev_m
        if m_method:
            idea.win_rate_method = f"{method} ｜ {m_method}"

    if wr_p is not None:
        ev_boost = 0.0
        if ev is not None and idea.max_loss:
            ev_boost += max(-6.0, min(6.0, (ev / max(idea.max_loss, 1.0)) * 10.0))
        if ev_m is not None and idea.max_loss:
            # Prefer structures that still look good under active management
            ev_boost += max(-8.0, min(8.0, (ev_m / max(idea.max_loss, 1.0)) * 14.0))
        idea.score = round(
            float(np.clip(idea.score * 0.62 + wr_p * 0.26 + ev_boost, 0, 100)),
            1,
        )
        notes = [
            f"POP（到期有利润）≈ {wr_p:.1f}%"
            + (f"；满盈概率≈ {wr_m:.1f}%" if wr_m is not None else ""),
        ]
        if ev is not None:
            notes.append(f"到期EV ≈ ${ev:.0f}/张（分段积分）")
        if ev_m is not None:
            notes.append(f"管理EV ≈ ${ev_m:.0f}/张（50%止盈/2R止损路径）")
        idea.notes = list(idea.notes) + notes
    return idea
