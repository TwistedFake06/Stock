"""
大盘 ETF · 仅 Vertical Spread 分析。

Facade module — implementation lives in:
  options_models, options_direction, options_chain, options_scoring,
  options_builders, options_analyze, options_frames, options_payoff.

Public import path stays ``from options_spreads import ...`` for Streamlit Cloud
and existing call sites.
"""

from __future__ import annotations

from options_analyze import analyze_options_spreads, select_credit_candidates
from options_builders import (
    build_bear_call,
    build_bear_put,
    build_bull_call,
    build_bull_put,
)
from options_chain import (
    HARD_CREDIT_FILL_HI,
    HARD_CREDIT_FILL_LO,
    HARD_DEBIT_FILL_HI,
    HARD_MIN_AVG_OI,
    HARD_MIN_CREDIT,
    HARD_MIN_LIQ_SCORE,
    _atm_iv,
    _dte,
    _fill_price,
    _get_cached_chain,
    _hist_vol,
    _is_us_rth,
    _leg_from_row,
    _liquid,
    _mid,
    _nearest_strike,
    _parse_chain,
    _pick_expiry,
    _pricing_mode_for_legs,
    _row_nearest_strike,
    _sane_option_mid,
    _width_ok,
    half_profit_close_price,
    passes_hard_filters,
    suggest_width,
)
from options_direction import analyze_direction
from options_frames import ideas_to_frame, legs_to_frame
from options_models import (
    INDEX_ETF_WHITELIST,
    LEVERAGED,
    DirectionReport,
    Leg,
    OptionsReport,
    SpreadIdea,
    is_options_eligible,
    options_symbol,
)
from options_payoff import (
    bs_option_price,
    build_daily_mark_calendar,
    build_payoff_ladder,
    payoff_per_contract,
    payoff_per_share,
    payoff_zones_summary,
    spread_mark_value,
)
from options_scoring import (
    _attach_win_rates,
    _norm_cdf,
    _prob_above,
    _prob_below,
    estimate_vertical_win_rates,
    score_liquidity,
)
from options_strategy_book import build_strategy_comparison
from options_timing import SpreadTimingReport, assess_spread_timing

__all__ = [
    "INDEX_ETF_WHITELIST",
    "LEVERAGED",
    "DirectionReport",
    "Leg",
    "SpreadIdea",
    "OptionsReport",
    "is_options_eligible",
    "options_symbol",
    "analyze_direction",
    "analyze_options_spreads",
    "select_credit_candidates",
    "build_strategy_comparison",
    "assess_spread_timing",
    "SpreadTimingReport",
    "build_bull_put",
    "build_bear_call",
    "build_bull_call",
    "build_bear_put",
    "estimate_vertical_win_rates",
    "score_liquidity",
    "half_profit_close_price",
    "passes_hard_filters",
    "suggest_width",
    "legs_to_frame",
    "ideas_to_frame",
    "payoff_per_share",
    "payoff_per_contract",
    "build_payoff_ladder",
    "payoff_zones_summary",
    "build_daily_mark_calendar",
    "bs_option_price",
    "spread_mark_value",
    # test/private helpers re-exported for unit tests
    "_row_nearest_strike",
    "_nearest_strike",
    "_liquid",
    "_leg_from_row",
    "_attach_win_rates",
    "_norm_cdf",
    "_prob_above",
    "_prob_below",
    "HARD_MIN_LIQ_SCORE",
    "HARD_MIN_CREDIT",
    "HARD_CREDIT_FILL_HI",
    "HARD_DEBIT_FILL_HI",
]
