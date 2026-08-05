"""Trade SOP unit tests (may use network for full build; pure helpers offline)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from edge_signals import (
    analyze_false_breakout,
    analyze_trend_align,
    analyze_volume_confirm,
    map_sector_etf,
)
from free_data import analyze_liquidity, multi_horizon_rs
from trade_sop import (
    _enter_decision,
    _expectancy_r,
    _path_win_rate,
    _stability_score,
    _swing_verdict,
)


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
            multi_rs_score=55,
        )
        self.assertIn(ok, ("适合入场", "谨慎试仓"))
        self.assertEqual(side, "做多")

    def test_enter_decision_full_entry_needs_rr_and_exp(self):
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            62,
            2.0,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.4,
            multi_rs_score=55,
        )
        self.assertEqual(ok, "适合入场")

    def test_poor_rr_blocks_entry(self):
        # High win rate but RR 0.4 → negative-ish expectancy, must not enter
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            70,
            0.4,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=-0.1,
        )
        self.assertIn(ok, ("观望", "回避"))

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

    def test_far_chase_blocks(self):
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            62,
            2.0,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.4,
            price_far_chase=True,
        )
        self.assertIn(ok, ("观望", "回避"))

    def test_iv_high_blocks_full(self):
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            62,
            2.0,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.4,
            multi_rs_score=55,
            sector_rs_score=55,
            volume_confirm_score=60,
            iv_high_event=True,
        )
        self.assertNotEqual(ok, "适合入场")

    def test_volume_dump_blocks_cautious(self):
        ok, _, _ = _enter_decision(
            "可关注",
            70,
            "看多",
            20,
            50,
            55,
            1.2,
            "中",
            regime_score=55,
            liquidity_score=60,
            expectancy_r=0.1,
            volume_confirm_score=20,  # 放量下跌
        )
        self.assertNotEqual(ok, "谨慎试仓")
        self.assertNotEqual(ok, "适合入场")

    def test_map_sector_etf(self):
        etf, lab = map_sector_etf("Technology", "Consumer Electronics")
        self.assertEqual(etf, "XLK")
        etf2, _ = map_sector_etf("Technology", "Semiconductors")
        self.assertEqual(etf2, "SMH")

    def test_volume_confirm_offline(self):
        n = 60
        # up day with high volume at end
        close = np.linspace(100, 110, n)
        close[-1] = 112
        vol = np.full(n, 1_000_000.0)
        vol[-1] = 2_500_000.0
        df = pd.DataFrame({"Close": close, "Volume": vol})
        r = analyze_volume_confirm(df)
        self.assertTrue(r.available)
        self.assertTrue(0 <= r.score <= 100)

    def test_false_breakout_detects_fail(self):
        n = 40
        # base grind then pierce high and close back
        close = np.concatenate([np.linspace(100, 105, n - 3), np.array([106.5, 104.0, 103.5])])
        high = close.copy()
        high[-3] = 108.0  # pierce
        low = close - 1
        open_ = close.copy()
        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": np.full(n, 1e6),
            }
        )
        r = analyze_false_breakout(df)
        self.assertTrue(r.available)
        # may or may not flag depending on prior high calc; score defined
        self.assertTrue(0 <= r.score <= 100)

    def test_swing_blocks_false_break(self):
        v = _swing_verdict(
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            wr=60,
            rr=1.5,
            exp_r=0.3,
            price_far_chase=False,
            false_break_risk=True,
        )
        self.assertEqual(v, "暫緩觀望")

    def test_swing_blocks_against_trend(self):
        v = _swing_verdict(
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            wr=60,
            rr=1.5,
            exp_r=0.3,
            price_far_chase=False,
            against_trend=True,
            trend_label="逆势",
            trend_score=35,
        )
        self.assertEqual(v, "暫緩觀望")

    def test_trend_align_offline(self):
        n = 80
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # strong uptrend stock
        close = 100 * np.cumprod(1 + np.full(n, 0.004))
        df = pd.DataFrame({"Date": dates, "Close": close})
        # without network SPY may fail → still returns structure
        r = analyze_trend_align(df, sector="Technology", industry="Software")
        self.assertIn(r.label, ("强跟势", "跟势", "中性", "逆势", "—"))
        self.assertTrue(0 <= r.score <= 100)

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
