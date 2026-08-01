"""Build same-origin market data cache for GitHub Pages static app."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

SYMBOLS = [
    "AAPL",
    "NVDA",
    "MSFT",
    "TSLA",
    "AMZN",
    "GOOGL",
    "META",
    "SPY",
    "QQQ",
    "0700.HK",
    "9988.HK",
    "3690.HK",
    "0005.HK",
    "600519.SS",
    "000001.SZ",
    "300750.SZ",
    "601318.SS",
]

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "web" / "data" / "quotes.json"
OUT_FILE_ROOT = ROOT / "quotes.json"


def to_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if df is None or df.empty:
        return rows

    df = df.dropna(subset=["Close"]).tail(520)
    for idx, row in df.iterrows():
        rows.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(row.get("Open", 0.0) or 0.0),
                "high": float(row.get("High", 0.0) or 0.0),
                "low": float(row.get("Low", 0.0) or 0.0),
                "close": float(row.get("Close", 0.0) or 0.0),
                "volume": float(row.get("Volume", 0.0) or 0.0),
            }
        )
    return rows


def fetch_symbol(symbol: str) -> list[dict]:
    ticker = yf.Ticker(symbol)
    # Match Streamlit stock_service (auto_adjust=True) so Lite vs full prices align
    hist = ticker.history(period="2y", interval="1d", auto_adjust=True)
    return to_rows(hist)


def main() -> None:
    data: dict[str, list[dict]] = {}
    failed: dict[str, str] = {}

    for symbol in SYMBOLS:
        try:
            rows = fetch_symbol(symbol)
            if len(rows) >= 30:
                data[symbol.upper()] = rows
            else:
                failed[symbol] = "insufficient rows"
        except Exception as exc:  # noqa: BLE001
            failed[symbol] = str(exc)

    payload = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "count": len(data),
        "symbols": sorted(data.keys()),
        "quotes": data,
        "failed": failed,
    }

    serialized = json.dumps(payload, ensure_ascii=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(serialized, encoding="utf-8")
    OUT_FILE_ROOT.write_text(serialized, encoding="utf-8")

    print(f"Wrote {OUT_FILE} and {OUT_FILE_ROOT} with {len(data)} symbols")
    if failed:
        print(f"Failed symbols: {len(failed)}")


if __name__ == "__main__":
    main()
