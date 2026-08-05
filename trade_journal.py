"""
Mini trade journal (local JSON) — compare model win rate vs your real results.

File: data/trade_journal.json (gitignored recommended)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
JOURNAL_PATH = ROOT / "data" / "trade_journal.json"


@dataclass
class JournalTrade:
    id: str
    symbol: str
    name: str = ""
    horizon: str = "0–2周"  # 0–2周 | 2–4周
    opened: str = ""  # ISO date
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    shares: int = 0
    model_wr: float | None = None
    model_rr: float | None = None
    model_verdict: str = ""
    status: str = "open"  # open | closed
    exit_price: float | None = None
    exit_date: str | None = None
    result_r: float | None = None  # realized R multiples
    pnl_usd: float | None = None
    notes: str = ""
    exit_reason: str = ""  # t1 | t2 | stop | time | manual


def _ensure_parent() -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_trades() -> list[dict[str, Any]]:
    if not JOURNAL_PATH.exists():
        return []
    try:
        data = json.loads(JOURNAL_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("trades"), list):
            return data["trades"]
    except Exception:
        pass
    return []


def save_trades(trades: list[dict[str, Any]]) -> None:
    _ensure_parent()
    JOURNAL_PATH.write_text(
        json.dumps(trades, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_trade(
    *,
    symbol: str,
    name: str = "",
    horizon: str = "0–2周",
    entry: float | None,
    stop: float | None,
    target: float | None,
    shares: int = 0,
    model_wr: float | None = None,
    model_rr: float | None = None,
    model_verdict: str = "",
    notes: str = "",
    opened: str | None = None,
) -> dict[str, Any]:
    t = JournalTrade(
        id=str(uuid.uuid4())[:8],
        symbol=str(symbol).upper(),
        name=name or symbol,
        horizon=horizon,
        opened=opened or date.today().isoformat(),
        entry=float(entry) if entry is not None else None,
        stop=float(stop) if stop is not None else None,
        target=float(target) if target is not None else None,
        shares=int(shares),
        model_wr=model_wr,
        model_rr=model_rr,
        model_verdict=model_verdict,
        notes=notes,
        status="open",
    )
    trades = load_trades()
    row = asdict(t)
    trades.insert(0, row)
    save_trades(trades)
    return row


def close_trade(
    trade_id: str,
    *,
    exit_price: float,
    exit_reason: str = "manual",
    exit_date: str | None = None,
) -> dict[str, Any] | None:
    trades = load_trades()
    found = None
    for t in trades:
        if t.get("id") == trade_id and t.get("status") == "open":
            entry = t.get("entry")
            stop = t.get("stop")
            t["status"] = "closed"
            t["exit_price"] = float(exit_price)
            t["exit_date"] = exit_date or date.today().isoformat()
            t["exit_reason"] = exit_reason
            # R multiple: (exit - entry) / (entry - stop)
            if entry and stop and float(entry) > float(stop):
                risk = float(entry) - float(stop)
                t["result_r"] = round((float(exit_price) - float(entry)) / risk, 3)
            shares = int(t.get("shares") or 0)
            if shares and entry is not None:
                t["pnl_usd"] = round(shares * (float(exit_price) - float(entry)), 2)
            found = t
            break
    if found:
        save_trades(trades)
    return found


def journal_stats(trades: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    trades = trades if trades is not None else load_trades()
    closed = [t for t in trades if t.get("status") == "closed" and t.get("result_r") is not None]
    open_n = sum(1 for t in trades if t.get("status") == "open")
    if not closed:
        return {
            "closed": 0,
            "open": open_n,
            "win_rate": None,
            "avg_r": None,
            "total_pnl": None,
            "expectancy_r": None,
        }
    wins = [t for t in closed if float(t["result_r"]) > 0]
    avg_r = sum(float(t["result_r"]) for t in closed) / len(closed)
    wr = 100.0 * len(wins) / len(closed)
    pnls = [float(t["pnl_usd"]) for t in closed if t.get("pnl_usd") is not None]
    return {
        "closed": len(closed),
        "open": open_n,
        "win_rate": round(wr, 1),
        "avg_r": round(avg_r, 3),
        "total_pnl": round(sum(pnls), 2) if pnls else None,
        "expectancy_r": round(avg_r, 3),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
    }
