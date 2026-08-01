"""Backtest engine unit tests (no network)."""
from __future__ import annotations

import unittest

from backtest.engine.simulator import simulate_trade
from backtest.engine.spread_sim import estimate_bull_put


class TestSpreadSim(unittest.TestCase):
    def test_bull_put_positive_credit(self):
        out = estimate_bull_put(
            spot=100.0,
            short_strike=97.0,
            long_strike=92.0,
            dte=30,
            volatility=0.20,
        )
        self.assertGreater(out["credit"], 0)
        self.assertGreater(out["max_profit"], 0)
        self.assertGreater(out["max_loss"], 0)
        self.assertAlmostEqual(out["width"], 5.0)
        self.assertLess(out["breakeven"], out["short_strike"] if "short_strike" in out else 97.0)
        self.assertAlmostEqual(out["breakeven"], 97.0 - out["credit"], places=2)


class TestSimulator(unittest.TestCase):
    def _spread(self) -> dict:
        return {
            "credit": 1.0,
            "max_profit": 100.0,
            "max_loss": 400.0,
            "breakeven": 96.0,
            "short_strike": 97.0,
            "long_strike": 92.0,
            "dte": 14,
            "hist_vol": 0.18,
        }

    def test_mtm_exit_produces_pnl(self):
        held, pnl, reason = simulate_trade(
            [100.0] * 5,
            self._spread(),
            hold_days=5,
            use_mtm=True,
            entry_vol=0.18,
            slip_per_share=0.0,
            commission_round_trip=0.0,
            take_profit_pct=0.5,
            stop_loss_pct=2.0,
        )
        self.assertGreaterEqual(held, 1)
        self.assertIn(
            reason,
            ("50%_mtm_tp", "mtm_stop", "time_exit_mtm", "expiry_settle"),
        )
        self.assertIsInstance(pnl, float)

    def test_bear_call_estimate(self):
        from backtest.engine.spread_sim import estimate_bear_call

        out = estimate_bear_call(100.0, 103.0, 108.0, 30, 0.20)
        self.assertEqual(out["strategy"], "bear_call")
        self.assertGreater(out["credit"], 0)
        self.assertEqual(out["option_type"], "call")

    def test_legacy_take_profit(self):
        spread = {
            "max_profit": 100.0,
            "max_loss": 400.0,
            "breakeven": 96.0,
        }
        held, pnl, reason = simulate_trade(
            [100.0, 101.0], spread, hold_days=14, use_mtm=False
        )
        self.assertEqual(reason, "50%_profit")
        self.assertEqual(pnl, 50.0)
        self.assertEqual(held, 1)

    def test_legacy_stop_loss(self):
        spread = {
            "max_profit": 100.0,
            "max_loss": 400.0,
            "breakeven": 96.0,
        }
        held, pnl, reason = simulate_trade(
            [90.0], spread, hold_days=14, use_mtm=False
        )
        self.assertEqual(reason, "stop_loss")
        self.assertEqual(pnl, -300.0)


class TestNonOverlapLogic(unittest.TestCase):
    def test_next_entry_advances(self):
        """Mirror run_backtest single-position index advance."""
        index = 50
        days_held = 7
        next_entry_index = index + days_held + 1
        self.assertEqual(next_entry_index, 58)
        # A candidate on day 55 is skipped
        self.assertTrue(55 < next_entry_index)
        self.assertFalse(58 < next_entry_index)


if __name__ == "__main__":
    unittest.main()
