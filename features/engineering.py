"""僅以當日及之前資料計算的日線特徵。"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ma5_dist", "ma10_dist", "ma20_dist", "ma60_dist",
    "high_20_dist", "low_20_dist", "vol_ratio_20", "dollar_vol", "vol_chg_5", "atr_14", "hv_20",
    "range_20", "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pos",
]


def _add_features(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.sort_values("Date").copy()
    close, high, low, volume = frame["Close"], frame["High"], frame["Low"], frame["Volume"].replace(0, np.nan)
    for days in (1, 3, 5, 10, 20):
        frame[f"ret_{days}"] = close.pct_change(days)
    for days in (5, 10, 20, 60):
        frame[f"ma{days}_dist"] = close / close.rolling(days).mean() - 1
    frame["high_20_dist"] = close / high.rolling(20).max() - 1
    frame["low_20_dist"] = close / low.rolling(20).min() - 1
    frame["vol_ratio_20"] = volume / volume.rolling(20).mean()
    frame["dollar_vol"] = close * volume
    frame["vol_chg_5"] = volume.pct_change(5)
    previous_close = close.shift(1)
    true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    frame["atr_14"] = true_range.rolling(14).mean()
    frame["hv_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    frame["range_20"] = (high.rolling(20).max() - low.rolling(20).min()) / close
    delta = close.diff()
    gain, loss = delta.clip(lower=0).rolling(14).mean(), -delta.clip(upper=0).rolling(14).mean()
    frame["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    fast, slow = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    frame["macd"] = fast - slow
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
    middle, std = close.rolling(20).mean(), close.rolling(20).std()
    frame["bb_pos"] = ((close - (middle - 2 * std)) / (4 * std)).replace([np.inf, -np.inf], np.nan)
    frame["fwd_ret_5"] = close.shift(-5) / close - 1
    frame["y"] = (frame["fwd_ret_5"] > 0).astype(float).where(frame["fwd_ret_5"].notna())
    return frame


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    return prices.groupby("Ticker", group_keys=False).apply(_add_features).reset_index(drop=True)