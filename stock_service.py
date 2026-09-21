"""Stock data service using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf

# US-only quick list (no HK / A-share)
# 常用置顶：MU / SNDK；名单=已选热门票（无板块龙头、无 ETF）
QUICK_PIN = ["MU", "SNDK"]
DEFAULT_WATCHLIST = [
    "GOOGL",
    "NVDA",
    "MSFT",
    "TSLA",
    "META",
    "SNDK",
    "MU",
    "INTC",
    "ORCL",
    "AMD",
    "AAPL",
    "AMZN",
    "SMCI",
    "IONQ",
    "RGTI",
    "QUBT",
    "ONDS",
    "QCOM",
    "WDC",
    "LITE",
    "VRT",
    "MUR",
    "NFLX",
    "NBIS",
    "UNH",
    "PLTR",
    "RXRX",
    "SMR",
    "BE",
    "COHR",
    "QQQ",
    "SOXX",
    "SPCX",
    "RKLB",
    "WMT",
    "DELL",
]
