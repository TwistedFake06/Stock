"""Vertical spread pure-function tests (no network)."""
from __future__ import annotations

import unittest

import pandas as pd

from options_payoff import payoff_per_contract
from options_spreads import (
    Leg,
    SpreadIdea,
    _row_nearest_strike,
    build_bear_call,
    build_bear_put,
    build_bull_call,
    build_bull_put,
    estimate_vertical_win_rates,
    half_profit_close_price,
    passes_hard_filters,
)


def _quote_row(strike: float, mid: float, oi: int = 2000) -> dict:
    """Build a chain row with tight bid/ask around mid (natural fill still positive)."""
    mid = max(float(mid), 0.15)
    # Tight book so sell@bid - buy@ask stays positive for verticals
    bid = round(mid - 0.02, 4)
    ask = round(mid + 0.02, 4)
    if bid <= 0:
        bid = 0.01
        ask = mid + 0.03
    return {
        "strike": float(strike),
        "bid": bid,
        "ask": ask,
        "lastPrice": mid,
        "mid": mid,
        "spread": ask - bid,
        "spread_pct": (ask - bid) / mid,
        "oi": oi,
        "vol": 500,
        "impliedVolatility": 0.20,
    }


def _chain_puts(spot: float = 100.0) -> pd.DataFrame:
    # Non-contiguous index; put mid rises with strike (intrinsic + time)
    strikes = [90, 92, 95, 97, 100, 102]
    rows = []
    for k in strikes:
        # Clear vertical credit: short 97 ~1.40, long 92 ~0.55 → credit ~0.81
        mid = 0.40 + max(k - 88, 0) * 0.12
        rows.append(_quote_row(k, mid))
    return pd.DataFrame(rows, index=[100 + i * 3 for i in range(len(rows))])


def _chain_calls(spot: float = 100.0) -> pd.DataFrame:
    # Call mid falls as strike rises
    strikes = [95, 98, 100, 102, 105, 108, 110]
    rows = []
    for k in strikes:
        mid = 0.40 + max(112 - k, 0) * 0.12
        rows.append(_quote_row(k, mid))
    return pd.DataFrame(rows, index=[200 + i * 3 for i in range(len(rows))])


class TestNearestRowIndexSafe(unittest.TestCase):
    def test_noncontiguous_index(self):
        df = pd.DataFrame({"strike": [90.0, 95.0, 100.0]}, index=[7, 21, 99])
        row = _row_nearest_strike(df, 96.0)
        self.assertEqual(float(row["strike"]), 95.0)


class TestVerticalBuilders(unittest.TestCase):
    def test_bull_put_metrics_match_fill_payoff(self):
        puts = _chain_puts(100)
        idea = build_bull_put(puts, spot=100, width=5, otm_pct=0.03, expiry="2099-01-01", dte=30)
        self.assertIsNotNone(idea)
        assert idea is not None
        self.assertEqual(idea.code, "bull_put")
        self.assertIsNotNone(idea.net_credit)
        self.assertGreater(idea.max_profit, 0)
        self.assertGreater(idea.max_loss, 0)
        # Deep OTM-ish high price → near max profit
        high = max(lg.strike for lg in idea.legs) + 20
        pnl_hi = payoff_per_contract(idea, high)
        self.assertAlmostEqual(pnl_hi, idea.max_profit, delta=1.5)
        # Far below long strike → near max loss
        low = min(lg.strike for lg in idea.legs) - 20
        pnl_lo = payoff_per_contract(idea, low)
        self.assertAlmostEqual(pnl_lo, -idea.max_loss, delta=1.5)

    def test_bear_call_builds(self):
        calls = _chain_calls(100)
        idea = build_bear_call(calls, spot=100, width=5, otm_pct=0.03, expiry="2099-01-01", dte=30)
        self.assertIsNotNone(idea)
        assert idea is not None
        self.assertEqual(idea.code, "bear_call")
        self.assertEqual(idea.legs[0].side, "sell")
        self.assertEqual(idea.legs[1].side, "buy")

    def test_bull_call_debit_leg_order(self):
        calls = _chain_calls(100)
        idea = build_bull_call(
            calls, spot=100, width=5, long_offset_pct=0.0, expiry="2099-01-01", dte=30
        )
        self.assertIsNotNone(idea)
        assert idea is not None
        self.assertEqual(idea.code, "bull_call")
        # Debit builders store [long, short]
        self.assertEqual(idea.legs[0].side, "buy")
        self.assertEqual(idea.legs[1].side, "sell")
        self.assertIsNotNone(idea.net_debit)

    def test_bear_put_debit(self):
        puts = _chain_puts(100)
        idea = build_bear_put(
            puts, spot=100, width=5, long_offset_pct=0.0, expiry="2099-01-01", dte=30
        )
        self.assertIsNotNone(idea)
        assert idea is not None
        self.assertEqual(idea.code, "bear_put")
        self.assertEqual(idea.legs[0].side, "buy")


class TestWinRatesAndFilters(unittest.TestCase):
    def test_bull_put_win_rate_reasonable(self):
        wr_p, wr_m, method, ev = estimate_vertical_win_rates(
            code="bull_put",
            spot=100,
            short_strike=97,
            long_strike=92,
            breakeven=96,
            sigma=0.20,
            dte=30,
            max_profit=100,
            max_loss=400,
        )
        self.assertIsNotNone(wr_p)
        assert wr_p is not None
        self.assertTrue(1.0 <= wr_p <= 99.0)
        self.assertIn("对数正态", method)
        self.assertIsNotNone(ev)

    def test_half_profit_credit(self):
        idea = SpreadIdea(
            name="t",
            code="bull_put",
            structure="Credit Vertical",
            thesis="t",
            net_credit=1.20,
            net_debit=None,
            max_profit=120.0,
            max_loss=380.0,
            breakevens=[98.8],
            width=5.0,
            pop_est=None,
            score=50,
            dte=30,
            expiry="2099-01-01",
            legs=[],
        )
        half = half_profit_close_price(idea)
        self.assertEqual(half, 0.60)

    def test_hard_filter_rejects_poor_liquidity(self):
        legs = [
            Leg(
                right="put",
                strike=100,
                side="sell",
                mid=1.0,
                bid=0.9,
                ask=1.1,
                iv=0.2,
                oi=1,
                volume=0,
                fill=0.9,
                fill_source="bid",
            ),
            Leg(
                right="put",
                strike=95,
                side="buy",
                mid=0.4,
                bid=0.3,
                ask=0.5,
                iv=0.2,
                oi=1,
                volume=0,
                fill=0.5,
                fill_source="ask",
            ),
        ]
        idea = SpreadIdea(
            name="t",
            code="bull_put",
            structure="Credit Vertical",
            thesis="t",
            net_credit=0.40,
            net_debit=None,
            max_profit=40,
            max_loss=460,
            breakevens=[99.6],
            width=5.0,
            pop_est=None,
            score=10,
            dte=30,
            expiry="2099-01-01",
            legs=legs,
            liquidity_score=10,
            liquidity_label="很差",
        )
        self.assertFalse(passes_hard_filters(idea))


if __name__ == "__main__":
    unittest.main()
