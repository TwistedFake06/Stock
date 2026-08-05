"""Technical indicators for stock charts."""

from __future__ import annotations

import pandas as pd


def add_sma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or [5, 20, 60]
    out = df.copy()
    for w in windows:
        # Require full window so SMA60 is a true 60-bar average (NaN until warm)
        out[f"SMA{w}"] = out["Close"].rolling(window=w, min_periods=w).mean()
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
    mid = out["Close"].rolling(window=window, min_periods=window).mean()
    std = out["Close"].rolling(window=window, min_periods=window).std()
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


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    out = df.copy()
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    close = out["Close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    out["ATR"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return out


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ADX + DI+/DI- (Wilder-style smoothing via ewm alpha=1/period).
    ADX high → trending; low → choppy (avoid breakout chase).
    """
    out = df.copy()
    if not {"High", "Low", "Close"}.issubset(out.columns) or len(out) < period + 5:
        out["PLUS_DI"] = pd.NA
        out["MINUS_DI"] = pd.NA
        out["ADX"] = pd.NA
        return out

    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    close = out["Close"].astype(float)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    )
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)).fillna(0)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    out["PLUS_DI"] = plus_di
    out["MINUS_DI"] = minus_di
    out["ADX"] = adx
    return out


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a standard set of indicators."""
    if df is None or df.empty:
        return df
    out = add_sma(df)
    out = add_ema(out)
    # short-term EMAs for 1H / swing timing
    for w in (9, 21):
        if f"EMA{w}" not in out.columns:
            out[f"EMA{w}"] = out["Close"].ewm(span=w, adjust=False).mean()
    out = add_bollinger(out)
    out = add_rsi(out)
    out = add_macd(out)
    if {"High", "Low"}.issubset(out.columns):
        out = add_atr(out)
        out = add_adx(out)
    return out
