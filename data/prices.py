"""可快取的 Yahoo 日線下載。"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers: tuple[str, ...], period: str = "2y", interval: str = "1d") -> tuple[pd.DataFrame, tuple[str, ...]]:
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for ticker in tickers:
        try:
            raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if raw is None or raw.empty:
                failed.append(ticker)
                continue
            raw = raw.reset_index().rename(columns={"Datetime": "Date"})
            cols = [column for column in ["Date", "Open", "High", "Low", "Close", "Volume"] if column in raw]
            frame = raw[cols].copy()
            if len(frame) < 2 or frame[["Open", "High", "Low", "Close"]].isna().all().all():
                failed.append(ticker)
                continue
            frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
            frame["Ticker"] = ticker
            frames.append(frame.dropna(subset=["Close"]))
        except Exception:
            failed.append(ticker)
    if not frames:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]), tuple(failed)
    return pd.concat(frames, ignore_index=True), tuple(failed)


def refresh_prices_cache() -> None:
    download_prices.clear()