"""Tests for the pure 5-minute opening-range setup rules."""

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from intraday_signals import analyze_opening_range_setup, is_intraday_alert_window
from scripts.watchlist_alert import _format_intraday_alert
from views.intraday_scan_page import _load_default_symbols


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


def _hk_bars() -> pd.DataFrame:
    bars = _bars()
    bars["Date"] = pd.date_range("2026-08-25 09:30", periods=24, freq="5min", tz="Asia/Hong_Kong")
    return bars


def _with_prior_strong_session() -> pd.DataFrame:
    prior = _bars(final_close=104.0)
    prior["Date"] = pd.date_range("2026-08-21 09:30", periods=24, freq="5min", tz="America/New_York")
    current = _bars()
    return pd.concat([prior, current], ignore_index=True)


class TestOpeningRangeSetup(unittest.TestCase):
    def test_warmup_reports_remaining_bars_and_eta(self):
        short = _bars().iloc[:2].copy()
        setup = analyze_opening_range_setup("NVDA", short)
        self.assertEqual(setup.verdict, "暖机中")
        self.assertIn("2/3", setup.reasons[0])
        self.assertIn("09:40", setup.reasons[0])

    def test_prior_session_seeds_indicators_after_opening_range(self):
        data = _with_prior_strong_session()
        first_opening_range = pd.concat([data.iloc[:24], data.iloc[24:27]], ignore_index=True)
        setup = analyze_opening_range_setup("NVDA", first_opening_range)
        self.assertNotEqual(setup.verdict, "暖机中")
        self.assertNotEqual(setup.verdict, "资料不足")

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

    def test_notification_includes_execution_levels(self):
        text = _format_intraday_alert(analyze_opening_range_setup("NVDA", _with_prior_strong_session()))
        self.assertIn("入场 E:", text)
        self.assertIn("止蚀 S:", text)
        self.assertIn("减仓 T1", text)
        self.assertIn("前日K线背景: 前日强势收市", text)

    def test_prior_strong_session_is_shown_as_trend_context(self):
        setup = analyze_opening_range_setup("NVDA", _with_prior_strong_session())
        self.assertEqual(setup.prior_session_label, "前日强势收市")
        self.assertTrue(any("顺势背景加分" in item for item in setup.reasons))

    def test_hong_kong_bars_use_hong_kong_market_session(self):
        setup = analyze_opening_range_setup("0700.HK", _hk_bars())
        self.assertEqual(setup.market, "HK")
        self.assertEqual(setup.session_label, "上午开市")
        self.assertEqual(setup.verdict, "可做")

    def test_hong_kong_alert_window_excludes_lunch(self):
        hkt = ZoneInfo("Asia/Hong_Kong")
        self.assertTrue(is_intraday_alert_window("0700.HK", datetime(2026, 8, 25, 10, 0, tzinfo=hkt)))
        self.assertFalse(is_intraday_alert_window("0700.HK", datetime(2026, 8, 25, 12, 30, tzinfo=hkt)))
        self.assertTrue(is_intraday_alert_window("0700.HK", datetime(2026, 8, 25, 13, 30, tzinfo=hkt)))
        self.assertTrue(is_intraday_alert_window("0700.HK", datetime(2026, 8, 25, 15, 59, tzinfo=hkt)))
        self.assertFalse(is_intraday_alert_window("0700.HK", datetime(2026, 8, 25, 16, 0, tzinfo=hkt)))

    def test_default_intraday_list_includes_hong_kong_symbols(self):
        symbols = _load_default_symbols()
        self.assertIn("0700.HK", symbols)
        self.assertEqual(len([symbol for symbol in symbols if symbol.endswith(".HK")]), 20)


if __name__ == "__main__":
    unittest.main()