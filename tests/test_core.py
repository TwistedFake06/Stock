"""
Core unit tests — pure functions only (no network).

Run locally or in CI:
  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from analysis import (
    BIAS_MILD_THRESHOLD,
    BIAS_STRONG_THRESHOLD,
    _bias_from_score,
)
from indicators import add_rsi
from options_greeks import calc_spread_greeks
from options_payoff import payoff_per_contract, payoff_per_share
from options_position import calc_options_position
from stock_service import normalize_symbol


def _leg(
    right: str,
    strike: float,
    side: str,
    *,
    mid: float,
    fill: float | None = None,
    iv: float = 0.20,
):
    return SimpleNamespace(
        right=right,
        strike=strike,
        side=side,
        mid=mid,
        fill=fill if fill is not None else mid,
        iv=iv,
        bid=mid,
        ask=mid,
    )


class TestNormalizeSymbol(unittest.TestCase):
    def test_us(self):
        self.assertEqual(normalize_symbol("aapl"), "AAPL")

    def test_hk(self):
        self.assertEqual(normalize_symbol("0700.hk"), "0700.HK")

    def test_shanghai(self):
        self.assertEqual(normalize_symbol("600519"), "600519.SS")

    def test_shenzhen(self):
        self.assertEqual(normalize_symbol("000001"), "000001.SZ")
        self.assertEqual(normalize_symbol("300750"), "300750.SZ")

    def test_beijing(self):
        self.assertEqual(normalize_symbol("830799"), "830799.BJ")
        self.assertEqual(normalize_symbol("430047"), "430047.BJ")


class TestPayoffFillVsMid(unittest.TestCase):
    """Payoff must use fill (natural) so it matches KPI max profit/loss."""

    def test_bull_put_credit_uses_fill(self):
        # Sell 100 put fill 2.00, buy 95 put fill 0.80 → credit 1.20
        legs = [
            _leg("put", 100, "sell", mid=2.50, fill=2.00),
            _leg("put", 95, "buy", mid=0.50, fill=0.80),
        ]
        idea = SimpleNamespace(
            legs=legs,
            net_credit=1.20,
            net_debit=None,
            max_profit=120.0,
            max_loss=380.0,
            width=5.0,
            breakevens=[98.8],
            code="bull_put",
        )
        # Spot well above short strike → max profit ≈ credit * 100
        pnl = payoff_per_contract(idea, 110.0)
        self.assertAlmostEqual(pnl, 120.0, places=1)

        # Spot well below long strike → max loss ≈ (width - credit) * 100
        pnl_loss = payoff_per_contract(idea, 80.0)
        self.assertAlmostEqual(pnl_loss, -380.0, places=1)

    def test_mid_would_diverge_but_fill_wins(self):
        legs = [
            _leg("put", 100, "sell", mid=3.0, fill=2.0),
            _leg("put", 95, "buy", mid=0.2, fill=0.8),
        ]
        idea = SimpleNamespace(legs=legs)
        # With fill: credit 1.2; with mid would be 2.8
        self.assertAlmostEqual(payoff_per_share(idea, 120.0), 1.2, places=4)


class TestGreeksBySide(unittest.TestCase):
    def test_credit_short_first_order(self):
        legs = [
            _leg("put", 100, "sell", mid=2.0, iv=0.20),
            _leg("put", 95, "buy", mid=0.8, iv=0.20),
        ]
        g = calc_spread_greeks(legs, spot=105.0, dte=30)
        # Bull put credit: short higher put → more negative short delta; net delta > 0
        self.assertLess(g.short_delta, 0)
        self.assertGreater(g.net_delta, 0)

    def test_debit_long_first_order_still_correct(self):
        # Debit builders store [long, short]
        legs = [
            _leg("call", 100, "buy", mid=3.0, iv=0.20),
            _leg("call", 105, "sell", mid=1.5, iv=0.20),
        ]
        g = calc_spread_greeks(legs, spot=100.0, dte=30)
        # Bull call debit: net delta should be positive
        self.assertGreater(g.net_delta, 0)
        self.assertGreater(g.long_delta, g.short_delta)


class TestVerticalEV(unittest.TestCase):
    def test_bull_put_ev_in_range(self):
        from options_scoring import estimate_vertical_win_rates

        wr_p, wr_m, method, ev = estimate_vertical_win_rates(
            code="bull_put",
            spot=100.0,
            short_strike=97.0,
            long_strike=92.0,
            breakeven=96.0,
            sigma=0.20,
            dte=30,
            max_profit=100.0,
            max_loss=400.0,
            credit=1.0,
            slip_per_share=0.0,
            r=0.0,
        )
        self.assertIsNotNone(wr_p)
        self.assertIsNotNone(ev)
        assert wr_p is not None and ev is not None
        self.assertTrue(1.0 <= wr_p <= 99.0)
        # Deep OTM-ish short put → EV should not equal full max loss
        self.assertGreater(ev, -400.0)
        self.assertLess(ev, 100.0)
        self.assertIn("分段盈亏", method)

    def test_ev_better_than_binary_approx_when_high_pop(self):
        """Piecewise EV should not force full max_loss on every loss path."""
        from options_scoring import estimate_vertical_win_rates

        _, _, _, ev = estimate_vertical_win_rates(
            code="bull_put",
            spot=100.0,
            short_strike=95.0,
            long_strike=90.0,
            breakeven=94.0,
            sigma=0.15,
            dte=45,
            max_profit=120.0,
            max_loss=380.0,
            credit=1.20,
            slip_per_share=0.0,
            r=0.0,
        )
        self.assertIsNotNone(ev)
        assert ev is not None
        self.assertGreater(ev, -200.0)

    def test_managed_ev_runs(self):
        from options_scoring import estimate_managed_ev

        ev_m, method = estimate_managed_ev(
            code="bull_put",
            spot=100.0,
            short_strike=97.0,
            long_strike=92.0,
            sigma=0.20,
            dte=21,
            max_profit=100.0,
            max_loss=400.0,
            credit=1.0,
            slip_per_share=0.0,
            commission_rt=0.0,
            n_paths=21,
        )
        self.assertIsNotNone(ev_m)
        self.assertIn("管理期望", method)
        assert ev_m is not None
        self.assertGreater(ev_m, -500.0)
        self.assertLess(ev_m, 150.0)


class TestBiasThresholds(unittest.TestCase):
    def test_cutoffs(self):
        self.assertEqual(BIAS_STRONG_THRESHOLD, 45)
        self.assertEqual(BIAS_MILD_THRESHOLD, 18)
        self.assertEqual(_bias_from_score(45), "强烈看多")
        self.assertEqual(_bias_from_score(18), "看多")
        self.assertEqual(_bias_from_score(0), "中性")
        self.assertEqual(_bias_from_score(-18), "看空")
        self.assertEqual(_bias_from_score(-45), "强烈看空")


class TestRSIEdgeCases(unittest.TestCase):
    def test_monotonic_and_flat_series(self):
        rising = add_rsi(pd.DataFrame({"Close": list(range(1, 31))}))
        falling = add_rsi(pd.DataFrame({"Close": list(range(30, 0, -1))}))
        flat = add_rsi(pd.DataFrame({"Close": [10.0] * 30}))

        self.assertEqual(float(rising["RSI"].iloc[-1]), 100.0)
        self.assertEqual(float(falling["RSI"].iloc[-1]), 0.0)
        self.assertEqual(float(flat["RSI"].iloc[-1]), 50.0)


class TestOptionsPosition(unittest.TestCase):
    def test_floor_contracts(self):
        plan = calc_options_position(
            max_loss_per_contract=250,
            max_profit_per_contract=100,
            account_size=50_000,
            risk_per_trade=500,
        )
        self.assertEqual(plan.contracts, 2)
        self.assertEqual(plan.total_max_loss, 500.0)

    def test_refuse_oversized(self):
        plan = calc_options_position(
            max_loss_per_contract=600,
            max_profit_per_contract=200,
            account_size=10_000,
            risk_per_trade=500,
        )
        self.assertEqual(plan.contracts, 0)


if __name__ == "__main__":
    unittest.main()
