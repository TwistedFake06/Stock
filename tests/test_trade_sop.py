"""Trade SOP unit tests (may use network for full build; pure helpers offline)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trade_sop import _enter_decision, _path_win_rate, _stability_score


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
            "较佳入场", 85, "看多", 35, 70, 60, 2.0, "中"
        )
        self.assertIn(ok, ("适合入场", "谨慎试仓"))
        self.assertEqual(side, "做多")

    def test_stability_labels(self):
        risk = type("R", (), {"ann_vol_pct": 12.0, "max_drawdown_pct": -8.0, "sharpe": 1.5, "win_rate_pct": 55})()
        trend = type("T", (), {"strength_label": "强趋势"})()
        score, label = _stability_score(risk, trend, 20)
        self.assertIn(label, ("高", "中", "低"))
        self.assertTrue(0 <= score <= 100)


if __name__ == "__main__":
    unittest.main()
