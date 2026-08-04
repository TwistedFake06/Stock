"""Spread timing suitability tests."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from options_timing import assess_spread_timing


class TestSpreadTiming(unittest.TestCase):
    def test_no_ideas_not_suitable(self):
        rep = assess_spread_timing(
            direction=SimpleNamespace(direction="中性", score=0, strength="弱", style_hint=""),
            best=None,
            after_hours=False,
            dte=30,
            iv_atm=0.18,
            ideas_count=0,
        )
        self.assertEqual(rep.verdict, "暂不建议")
        self.assertLess(rep.score, 50)

    def test_good_credit_setup(self):
        best = SimpleNamespace(
            code="bull_put",
            net_credit=1.2,
            net_debit=None,
            liquidity_label="高",
            liquidity_score=75,
            liquidity_detail="OK",
            pricing_mode="natural",
            win_rate_profit=72.0,
            expected_value=25.0,
            expected_value_managed=30.0,
        )
        rep = assess_spread_timing(
            direction=SimpleNamespace(
                direction="看多", score=30, strength="中", style_hint="偏信用"
            ),
            best=best,
            after_hours=False,
            dte=30,
            iv_atm=0.18,
            ideas_count=8,
        )
        self.assertIn(rep.verdict, ("适合开仓", "谨慎可做"))
        self.assertGreaterEqual(rep.score, 55)

    def test_after_hours_caps_verdict(self):
        best = SimpleNamespace(
            code="bull_put",
            net_credit=1.2,
            net_debit=None,
            liquidity_label="高",
            liquidity_score=80,
            liquidity_detail="OK",
            pricing_mode="natural",
            win_rate_profit=75.0,
            expected_value=40.0,
            expected_value_managed=35.0,
        )
        rep = assess_spread_timing(
            direction=SimpleNamespace(
                direction="看多", score=40, strength="强", style_hint=""
            ),
            best=best,
            after_hours=True,
            dte=30,
            iv_atm=0.20,
            ideas_count=10,
        )
        self.assertNotEqual(rep.verdict, "适合开仓")
        self.assertIn(rep.verdict, ("谨慎可做", "暂不建议"))


if __name__ == "__main__":
    unittest.main()
