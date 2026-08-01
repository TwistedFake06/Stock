"""Stock data service using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf

# Common A-share / HK / US examples for the UI
DEFAULT_WATCHLIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "0700.HK",
    "9988.HK",
    "600519.SS",  # 贵州茅台
    "000001.SZ",  # 平安银行
]

PERIOD_MAP = {
    "1月": "1mo",
    "3月": "3mo",
    "6月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "5年": "5y",
    "最大": "max",
}

INTERVAL_MAP = {
    "日线": "1d",
    "周线": "1wk",
    "月线": "1mo",
}

# Lazy Streamlit cache wrapper (works on Cloud + local streamlit run)
_st_history_cached = None


def normalize_symbol(symbol: str) -> str:
    """Normalize user input to a Yahoo Finance ticker."""
    s = symbol.strip().upper()
    if not s:
        return s

    # Pure 6-digit China A-share / Beijing codes
    if s.isdigit() and len(s) == 6:
        # Shanghai: 6xxxxx (incl. STAR 688xxx)
        if s.startswith("6"):
            return f"{s}.SS"
        # Shenzhen: 0xxxxx / 3xxxxx
        if s.startswith(("0", "3")):
            return f"{s}.SZ"
        # Beijing Stock Exchange: 8xxxxx / 4xxxxx
        if s.startswith(("8", "4")):
            return f"{s}.BJ"
        return s

    # Allow 600519.SS / 000001.SZ / 830799.BJ as-is
    return s


def get_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(normalize_symbol(symbol))


def _fetch_history_uncached(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV history for a symbol (no cache)."""
    ticker = get_ticker(symbol)
    try:
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Normalize datetime column name
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@lru_cache(maxsize=64)
def _fetch_history_lru(
    symbol: str,
    period: str,
    interval: str,
    _bucket: str,
) -> pd.DataFrame:
    """Process-local fallback when Streamlit cache is unavailable (CLI/backtest)."""
    return _fetch_history_uncached(symbol, period, interval)


def _running_under_streamlit() -> bool:
    """True only inside an active Streamlit script run (Cloud or local)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def fetch_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV history.

    Caching strategy (Cloud + local safe):
    - Active Streamlit session: ``st.cache_data`` TTL 5 minutes
    - CLI / backtest / unit tests: time-bucketed ``lru_cache``

    Always returns a copy so callers can mutate safely.
    """
    global _st_history_cached
    sym = normalize_symbol(symbol)
    result: pd.DataFrame | None = None

    if _running_under_streamlit():
        try:
            import streamlit as st

            if _st_history_cached is None:
                _st_history_cached = st.cache_data(ttl=300, show_spinner=False)(
                    _fetch_history_uncached
                )
            result = _st_history_cached(sym, period, interval)
        except Exception:
            result = None

    if result is None:
        result = _fetch_history_lru(sym, period, interval, cache_bucket(5))

    if result is None or (isinstance(result, pd.DataFrame) and result.empty):
        return pd.DataFrame()
    return result.copy()


def fetch_info(symbol: str) -> dict[str, Any]:
    """Fetch basic quote / company info. Best-effort; fields vary by market."""
    ticker = get_ticker(symbol)
    info: dict[str, Any] = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    # Prefer fast_info for live-ish price when available
    try:
        fast = ticker.fast_info
        if fast is not None:
            for key in (
                "last_price",
                "previous_close",
                "open",
                "day_high",
                "day_low",
                "year_high",
                "year_low",
                "market_cap",
                "currency",
                "exchange",
            ):
                val = getattr(fast, key, None)
                if val is not None and key not in info:
                    info[key] = val
            # Map last_price -> currentPrice for consistent UI
            if "last_price" in info and "currentPrice" not in info:
                info["currentPrice"] = info["last_price"]
    except Exception:
        pass

    info["_symbol"] = normalize_symbol(symbol)
    return info


def search_suggestions(query: str, limit: int = 8) -> list[str]:
    """Simple local suggestion helper (watchlist + normalized input)."""
    q = query.strip().upper()
    if not q:
        return DEFAULT_WATCHLIST[:limit]
    hits = [s for s in DEFAULT_WATCHLIST if q in s.upper()]
    norm = normalize_symbol(query)
    if norm and norm not in hits:
        hits.insert(0, norm)
    return hits[:limit]


def compute_returns(df: pd.DataFrame) -> dict[str, float | None]:
    """Compute simple period returns from a history dataframe."""
    if df is None or df.empty or "Close" not in df.columns:
        return {"total_return_pct": None, "volatility_pct": None}

    close = df["Close"].dropna()
    if len(close) < 2:
        return {"total_return_pct": None, "volatility_pct": None}

    total = (close.iloc[-1] / close.iloc[0] - 1) * 100
    daily_ret = close.pct_change().dropna()
    vol = daily_ret.std() * (252**0.5) * 100 if len(daily_ret) > 1 else None
    return {
        "total_return_pct": float(total),
        "volatility_pct": float(vol) if vol is not None else None,
    }


def compare_symbols(
    symbols: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Return normalized close prices (base=100) for multiple symbols."""
    frames: list[pd.Series] = []
    for sym in symbols:
        hist = fetch_history(sym, period=period, interval=interval)
        if hist.empty:
            continue
        s = hist.set_index("Date")["Close"].rename(normalize_symbol(sym))
        # Normalize to 100 at first valid point
        s = s / s.iloc[0] * 100
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    return out.ffill()


@lru_cache(maxsize=32)
def cached_info(symbol: str, _bucket: str) -> dict[str, Any]:
    """Cache info for a few minutes via time bucket key."""
    return fetch_info(symbol)


def fetch_calendar(symbol: str) -> dict[str, Any]:
    """Best-effort corporate calendar (earnings, dividends)."""
    ticker = get_ticker(symbol)
    try:
        cal = ticker.calendar
        if cal is None:
            return {}
        # yfinance may return dict or DataFrame depending on version
        if isinstance(cal, dict):
            return cal
        if hasattr(cal, "to_dict"):
            # DataFrame shaped calendars
            try:
                return {str(k): v for k, v in cal.to_dict().items()}
            except Exception:
                return {}
        return {}
    except Exception:
        return {}


@lru_cache(maxsize=32)
def cached_calendar(symbol: str, _bucket: str) -> dict[str, Any]:
    return fetch_calendar(symbol)


def cache_bucket(minutes: int = 5) -> str:
    now = datetime.utcnow()
    bucket = now - timedelta(
        minutes=now.minute % minutes,
        seconds=now.second,
        microseconds=now.microsecond,
    )
    return bucket.isoformat()
