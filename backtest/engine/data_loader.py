"""Historical daily data loading."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data and return a clean, chronological DataFrame."""
    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    return data.sort_values("Date").reset_index(drop=True)