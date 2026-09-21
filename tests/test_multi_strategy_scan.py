"""Tests for multi_strategy_scan helpers (no network)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from multi_strategy_scan import _macd_turn_up, _in_entry_zone


class TestMultiHelpers(unittest.TestCase):
    def test_in_zone(self):
        self.assertTrue(_in_entry_zone(100.0, 98.0, 102.0))
        self.assertFalse(_in_entry_zone(110.0, 98.0, 102.0))

    def test_macd_cross_up(self):
        df = pd.DataFrame({"MACD_HIST": [-0.2, -0.1, 0.05]})
        fired, reason, strength = _macd_turn_up(df)
        self.assertTrue(fired)
        self.assertGreater(strength, 0)


if __name__ == "__main__":
    unittest.main()
