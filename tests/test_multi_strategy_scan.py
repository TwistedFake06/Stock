"""Tests for multi_strategy_scan helpers (no network)."""
from __future__ import annotations

import unittest

import pandas as pd

from multi_strategy_scan import (
    BEAR_TIER_NONE,
    BEAR_TIER_STRONG,
    BEAR_TIER_WATCH,
    SymbolMultiResult,
    _above_entry_zone,
    _in_entry_zone,
    _macd_turn_down,
    _macd_turn_up,
)


class TestMultiHelpers(unittest.TestCase):
    def test_in_zone(self):
        self.assertTrue(_in_entry_zone(100.0, 98.0, 102.0))
        self.assertFalse(_in_entry_zone(110.0, 98.0, 102.0))

    def test_above_zone(self):
        self.assertTrue(_above_entry_zone(110.0, 98.0, 102.0))
        self.assertFalse(_above_entry_zone(100.0, 98.0, 102.0))

    def test_macd_cross_up(self):
        df = pd.DataFrame({"MACD_HIST": [-0.2, -0.1, 0.05]})
        fired, reason, strength = _macd_turn_up(df)
        self.assertTrue(fired)
        self.assertGreater(strength, 0)

    def test_macd_cross_down(self):
        df = pd.DataFrame({"MACD_HIST": [0.2, 0.1, -0.05]})
        fired, reason, strength = _macd_turn_down(df)
        self.assertTrue(fired)
        self.assertGreater(strength, 0)
        self.assertIn("轉負", reason)

    def test_macd_falling_negative(self):
        df = pd.DataFrame({"MACD_HIST": [-0.1, -0.2, -0.35]})
        fired, _, strength = _macd_turn_down(df)
        self.assertTrue(fired)
        self.assertGreaterEqual(strength, 60.0)

    def test_bear_fields_default(self):
        r = SymbolMultiResult(symbol="X", name="X", last_price=1.0)
        self.assertEqual(r.bear_hit_count, 0)
        self.assertEqual(r.bear_tier, BEAR_TIER_NONE)
        self.assertEqual(r.suggest_tier, "不入場")
        self.assertIn(BEAR_TIER_STRONG, (BEAR_TIER_STRONG, BEAR_TIER_WATCH, BEAR_TIER_NONE))


if __name__ == "__main__":
    unittest.main()
