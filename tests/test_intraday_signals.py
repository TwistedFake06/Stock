"""Tests for the pure 5-minute opening-range setup rules."""

from __future__ import annotations

import unittest

import pandas as pd

from intraday_signals import analyze_opening_range_setup


def _bars(*, final_close: float = 100.8, final_volume: float = 4_000) -> pd.DataFrame:
    dates = pd.date_range("2026-08-25 09:30", periods=24, freq="5min", tz="America/New_York")
    # Opening range high is 100.50, followed by consolidation and a fresh breakout.
    close = [100.35, 100.38, 100.40] + [100.15 + (index % 3) * 0.03 for index in range(20)] + [final_close]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [value - 0.05 for value in close],
            "High": [value + 0.12 for value in close],
            "Low": [value - 0.15 for value in close],
            "Close": close,
            "Volume": [1_000] * 23 + [final_volume],
        }
    )


class TestOpeningRangeSetup(unittest.TestCase):
    def test_breakout_builds_entry_stop_and_targets(self):
        setup = analyze_opening_range_setup("NVDA", _bars())
        self.assertEqual(setup.verdict, "可做")
        self.assertIsNotNone(setup.entry)
        self.assertIsNotNone(setup.stop)
        self.assertIsNotNone(setup.target_1)
        self.assertIsNotNone(setup.target_2)
        assert setup.entry is not None and setup.stop is not None
        assert setup.target_1 is not None and setup.target_2 is not None
        self.assertGreater(setup.entry, setup.stop)
        self.assertAlmostEqual(setup.target_1 - setup.entry, setup.entry - setup.stop, places=6)
        self.assertAlmostEqual(setup.target_2 - setup.entry, 2 * (setup.entry - setup.stop), places=6)

    def test_extended_price_is_not_chased(self):
        setup = analyze_opening_range_setup("NVDA", _bars(final_close=102.0))
        self.assertNotEqual(setup.verdict, "可做")
        self.assertTrue(any("远离计划入场" in item for item in setup.reasons))


if __name__ == "__main__":
    unittest.main()