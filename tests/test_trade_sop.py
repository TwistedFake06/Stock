"""Trade SOP unit tests (may use network for full build; pure helpers offline)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from free_data import analyze_liquidity, multi_horizon_rs
from trade_sop import _enter_decision, _expectancy_r, _path_win_rate, _stability_score


class TestTradeSOPHelpers(unittest.TestCase):
    def test_path_win_rate_runs(self):
        rng = np.random.default_rng(0)
        # mild uptrend noise
        rets = rng.normal(0.001, 0.015, 200)
        close = 100 * np.cumprod(1 + rets)
        s = pd.Series(close)
        wr = _path_win_rate(s, risk_per_share=2.0, reward_per_share=3.0, lookback=80, horizon=15)
        self.assertIsNotNone(wr)
        assert wr is not None
        self.assertTrue(0 <= wr <= 100)

    def test_enter_decision_avoid_bearish(self):
        ok, score, side = _enter_decision(
            "偏空回避", 30, "强烈看空", -50, 40, 45, 1.0, "高"
        )
        self.assertEqual(ok, "回避")

    def test_enter_decision_good_setup(self):
        ok, score, side = _enter_decision(
            "较佳入场",
            85,
            "看多",
            35,
            70,
            60,
            2.0,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.4,
        )
        self.assertIn(ok, ("适合入场", "谨慎试仓"))
        self.assertEqual(side, "做多")

    def test_enter_decision_regime_blocks_full_entry(self):
        ok, score, side = _enter_decision(
            "较佳入场",
            85,
            "看多",
            40,
            70,
            60,
            2.0,
            "中",
            regime_score=25,
            liquidity_score=80,
            expectancy_r=0.4,
        )
        self.assertNotEqual(ok, "适合入场")

    def test_enter_decision_earnings_caps(self):
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            62,
            2.2,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.5,
            earnings_soon=True,
        )
        self.assertNotEqual(ok, "适合入场")

    def test_expectancy(self):
        # 55% win, 2R reward → 0.55*2 - 0.45*1 = 0.65
        self.assertAlmostEqual(_expectancy_r(55, 2.0) or 0, 0.65, places=2)
        self.assertIsNone(_expectancy_r(None, 2.0))

    def test_stability_labels(self):
        risk = type("R", (), {"ann_vol_pct": 12.0, "max_drawdown_pct": -8.0, "sharpe": 1.5, "win_rate_pct": 55})()
        trend = type("T", (), {"strength_label": "强趋势"})()
        score, label = _stability_score(risk, trend, 20)
        self.assertIn(label, ("高", "中", "低"))
        self.assertTrue(0 <= score <= 100)

    def test_liquidity_offline(self):
        n = 40
        df = pd.DataFrame(
            {
                "Close": np.linspace(100, 110, n),
                "High": np.linspace(101, 111, n),
                "Low": np.linspace(99, 109, n),
                "Volume": np.full(n, 5_000_000.0),
            }
        )
        liq = analyze_liquidity(df, {"averageVolume": 5_000_000, "currentPrice": 110})
        self.assertTrue(0 <= liq.score <= 100)
        self.assertIn(liq.label, ("高", "中", "低"))

    def test_multi_horizon_rs_offline(self):
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        stock = pd.DataFrame(
            {"Date": dates, "Close": 100 * np.cumprod(1 + np.full(n, 0.002))}
        )
        bench = pd.DataFrame(
            {"Date": dates, "Close": 100 * np.cumprod(1 + np.full(n, 0.0005))}
        )
        out = multi_horizon_rs(stock, bench)
        self.assertIsNotNone(out.get("score"))
        self.assertIn(out.get("label"), ("强于大盘", "同步", "弱于大盘", "—"))


if __name__ == "__main__":
    unittest.main()
