"""Technical indicators for stock charts."""

from __future__ import annotations

import pandas as pd


def add_sma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [5, 20, 60]
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(window=w, min_periods=1).mean()
    return out


def add_ema(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [12, 26]
    out = df.copy()
    for w in windows:
        out[f"EMA{w}"] = out["Close"].ewm(span=w, adjust=False).mean()
    return out


def add_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    out = df.copy()
    mid = out["Close"].rolling(window=window, min_periods=1).mean()
    std = out["Close"].rolling(window=window, min_periods=1).std()
    out["BB_MID"] = mid
    out["BB_UPPER"] = mid + num_std * std
    out["BB_LOWER"] = mid - num_std * std
    return out


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out["RSI"] = 100 - (100 / (1 + rs))
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    return out


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a standard set of indicators."""
    if df is None or df.empty:
        return df
    out = add_sma(df)
    out = add_ema(out)
    out = add_bollinger(out)
    out = add_rsi(out)
    out = add_macd(out)
    return out
