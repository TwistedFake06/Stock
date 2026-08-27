"""股票池篩選與代碼解析。"""
from __future__ import annotations

from config import DEFAULT_UNIVERSE, HK_UNIVERSE, US_UNIVERSE


def selected_universe(market: str, selected: list[str], custom: str) -> dict[str, str]:
    source = HK_UNIVERSE if market == "只港股" else US_UNIVERSE if market == "只美股" else DEFAULT_UNIVERSE
    result = {ticker: source.get(ticker, ticker) for ticker in selected if ticker in source}
    for ticker in custom.split(","):
        ticker = ticker.strip().upper()
        if ticker:
            result[ticker] = DEFAULT_UNIVERSE.get(ticker, ticker)
    return result or source.copy()