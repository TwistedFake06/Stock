"""Tests for walk-forward rule-score validation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategy_validation import build_validation_samples, score_bin, summarize_score_bins


def _history(rows: int = 100) -> pd.DataFrame:
    close = np.linspace(100.0, 130.0, rows) + np.sin(np.arange(rows) / 4.0)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "Open": close - 0.2,
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Volume": np.linspace(1_000_000, 1_500_000, rows),
        }
    )


class TestStrategyValidation(unittest.TestCase):
    def test_score_bins_match_live_thresholds(self):
        self.assertEqual(score_bin(-45), "强烈看空")
        self.assertEqual(score_bin(-18), "看空")
        self.assertEqual(score_bin(17.9), "中性")
        self.assertEqual(score_bin(18), "看多")
        self.assertEqual(score_bin(45), "强烈看多")

    def test_future_changes_do_not_change_earlier_score(self):
        original = _history()
        changed = original.copy()
        changed.loc[80:, ["Open", "High", "Low", "Close"]] *= 2.0

        first = build_validation_samples(original, horizon=5, warmup=60)
        second = build_validation_samples(changed, horizon=5, warmup=60)
        cutoff = pd.Timestamp(original["Date"].iloc[74]).normalize()
        left = first[first["date"] <= cutoff][["date", "score"]].reset_index(drop=True)
        right = second[second["date"] <= cutoff][["date", "score"]].reset_index(drop=True)

        pd.testing.assert_frame_equal(left, right)

    def test_summary_keeps_timeout_and_ambiguous_separate(self):
        samples = pd.DataFrame(
            {
                "score_bin": ["看多"] * 4,
                "forward_return_pct": [1.0, -1.0, 0.5, -0.5],
                "directional_return_pct": [1.0, -1.0, 0.5, -0.5],
                "excess_return_pct": [0.5, -1.5, 0.0, -1.0],
                "path_outcome": ["目标先触", "止损先触", "到期未触", "歧义"],
            }
        )

        summary = summarize_score_bins(samples).iloc[0]
        self.assertEqual(summary["目标先触%"], 25.0)
        self.assertEqual(summary["止损先触%"], 25.0)
        self.assertEqual(summary["到期未触%"], 25.0)
        self.assertEqual(summary["歧义%"], 25.0)


if __name__ == "__main__":
    unittest.main()