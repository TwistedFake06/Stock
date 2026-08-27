"""回測交易成本設定。"""
from __future__ import annotations


def default_cost_rate(tickers: list[str]) -> float:
    return 0.0015 if any(ticker.endswith(".HK") for ticker in tickers) else 0.0005