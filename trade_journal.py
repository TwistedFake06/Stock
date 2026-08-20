"""
Persistent trade journal — accumulate real trades as long-term samples.

Primary file: data/trade_journal.json
Daily backups: data/journal_backups/trade_journal_YYYYMMDD.json
Optional override: env TRADE_JOURNAL_PATH (absolute path for always-on disk)

Never auto-deletes closed trades (samples for stats).
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_JOURNAL = ROOT / "data" / "trade_journal.json"
BACKUP_DIR = ROOT / "data" / "journal_backups"


def resolve_journal_path() -> Path:
    """Prefer env path so Cloud/local can point to a durable location."""
    env = (os.environ.get("TRADE_JOURNAL_PATH") or "").strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_JOURNAL


def journal_path_info() -> dict[str, Any]:
    p = resolve_journal_path()
    exists = p.exists()
    n = len(load_trades())
    size = p.stat().st_size if exists else 0
    return {
        "path": str(p.resolve()) if exists or p.parent.exists() else str(p),
        "exists": exists,
        "n_trades": n,
        "bytes": size,
        "backup_dir": str(BACKUP_DIR),
    }


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
    mode: str = ""  # defensive | aggressive
    mode_label: str = ""
    status: str = "open"  # open | closed
    exit_price: float | None = None
    exit_date: str | None = None
    result_r: float | None = None
    pnl_usd: float | None = None
    notes: str = ""
    exit_reason: str = ""  # t1 | t2 | stop | time | manual
    sample: bool = True  # always kept for sample stats unless user forces purge


def _ensure_parent(path: Path | None = None) -> None:
    path = path or resolve_journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_trade(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure required keys for old rows."""
    out = dict(row)
    out.setdefault("id", str(uuid.uuid4())[:8])
    out.setdefault("symbol", "")
    out.setdefault("status", "open")
    out.setdefault("sample", True)
    out.setdefault("shares", 0)
    out.setdefault("notes", "")
    return out


def _valid_model_wr(value: Any) -> float | None:
    try:
        model_wr = float(value)
    except (TypeError, ValueError):
        return None
    return model_wr if 0 <= model_wr <= 100 else None


def load_trades() -> list[dict[str, Any]]:
    path = resolve_journal_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("trades"), list):
            rows = data["trades"]
        else:
            return []
        return [_normalize_trade(t) for t in rows if isinstance(t, dict)]
    except Exception:
        # try latest backup
        try:
            backups = sorted(BACKUP_DIR.glob("trade_journal_*.json"), reverse=True)
            for b in backups[:3]:
                try:
                    data = json.loads(b.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        return [_normalize_trade(t) for t in data if isinstance(t, dict)]
                    if isinstance(data, dict) and isinstance(data.get("trades"), list):
                        return [
                            _normalize_trade(t)
                            for t in data["trades"]
                            if isinstance(t, dict)
                        ]
                except Exception:
                    continue
        except Exception:
            pass
    return []


def save_trades(trades: list[dict[str, Any]]) -> Path:
    """
    Atomic write + daily backup. Never drops closed sample rows here.
    Returns path written.
    """
    path = resolve_journal_path()
    _ensure_parent(path)
    # only keep dict rows
    clean = [_normalize_trade(t) for t in trades if isinstance(t, dict)]
    payload = {
        "version": 2,
        "updated": datetime.utcnow().isoformat() + "Z",
        "n": len(clean),
        "trades": clean,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

    # daily backup (one file per day, overwrite same day = latest)
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        day = date.today().isoformat().replace("-", "")
        bak = BACKUP_DIR / f"trade_journal_{day}.json"
        shutil.copy2(path, bak)
    except Exception:
        pass
    return path


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
    mode: str = "",
    mode_label: str = "",
    notes: str = "",
    opened: str | None = None,
) -> dict[str, Any]:
    mode_key = (mode or "").strip().lower()
    if mode_key in ("a", "防守", "防守版", "def", "defence", "defense"):
        mode_key = "defensive"
    elif mode_key in ("b", "进攻", "进攻版", "進攻", "進攻版", "agg"):
        mode_key = "aggressive"
    if not mode_label and mode_key == "defensive":
        mode_label = "A 防守版"
    elif not mode_label and mode_key == "aggressive":
        mode_label = "B 进攻版"
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
        mode=mode_key,
        mode_label=mode_label or mode_key,
        notes=notes,
        status="open",
        sample=True,
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
            t["sample"] = True  # keep forever for samples
            t["exit_price"] = float(exit_price)
            t["exit_date"] = exit_date or date.today().isoformat()
            t["exit_reason"] = exit_reason
            if entry is not None and stop is not None and float(entry) > float(stop):
                risk = float(entry) - float(stop)
                if risk > 0:
                    t["result_r"] = round(
                        (float(exit_price) - float(entry)) / risk, 3
                    )
            shares = int(t.get("shares") or 0)
            if shares and entry is not None:
                t["pnl_usd"] = round(
                    shares * (float(exit_price) - float(entry)), 2
                )
            found = t
            break
    if found:
        save_trades(trades)
    return found


def merge_trades(
    incoming: list[dict[str, Any]],
    *,
    prefer_incoming_closed: bool = True,
) -> dict[str, Any]:
    """
    Merge imported trades into store by id.
    Closed sample rows are never dropped. Returns merge stats.
    """
    existing = {_normalize_trade(t)["id"]: _normalize_trade(t) for t in load_trades()}
    added = updated = skipped = 0
    for raw in incoming:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        row = _normalize_trade(raw)
        tid = str(row.get("id") or "").strip() or str(uuid.uuid4())[:8]
        row["id"] = tid
        row["sample"] = True
        if tid not in existing:
            existing[tid] = row
            added += 1
            continue
        old = existing[tid]
        # never overwrite a closed sample with an open stub
        if old.get("status") == "closed" and row.get("status") != "closed":
            skipped += 1
            continue
        if old.get("status") == "closed" and row.get("status") == "closed":
            if prefer_incoming_closed:
                existing[tid] = {**old, **row, "sample": True}
                updated += 1
            else:
                skipped += 1
            continue
        # open → closed or open refresh
        existing[tid] = {**old, **row, "sample": True}
        updated += 1

    merged = list(existing.values())
    # stable-ish: closed last activity first by opened desc
    merged.sort(key=lambda t: str(t.get("opened") or ""), reverse=True)
    save_trades(merged)
    return {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "total": len(merged),
    }


def export_json_text() -> str:
    trades = load_trades()
    return json.dumps(
        {
            "version": 2,
            "exported": datetime.utcnow().isoformat() + "Z",
            "n": len(trades),
            "trades": trades,
        },
        ensure_ascii=False,
        indent=2,
    )


def export_csv_text() -> str:
    trades = load_trades()
    cols = [
        "id",
        "symbol",
        "name",
        "horizon",
        "opened",
        "entry",
        "stop",
        "target",
        "shares",
        "model_wr",
        "model_rr",
        "model_verdict",
        "mode",
        "mode_label",
        "status",
        "exit_price",
        "exit_date",
        "result_r",
        "pnl_usd",
        "exit_reason",
        "notes",
        "sample",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for t in trades:
        w.writerow({c: t.get(c, "") for c in cols})
    return buf.getvalue()


def parse_import_bytes(raw: bytes | str) -> list[dict[str, Any]]:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig")
    else:
        text = raw
    text = text.strip()
    if not text:
        return []
    # JSON
    if text[0] in "{[":
        data = json.loads(text)
        if isinstance(data, list):
            return [t for t in data if isinstance(t, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("trades"), list):
                return [t for t in data["trades"] if isinstance(t, dict)]
            # single trade
            if data.get("symbol") or data.get("id"):
                return [data]
        return []
    # CSV
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader if row.get("symbol") or row.get("id")]


def journal_stats(trades: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    trades = trades if trades is not None else load_trades()
    # samples = all trades kept; closed with R for win-rate
    samples = [t for t in trades if t.get("sample", True)]
    closed = [
        t
        for t in samples
        if t.get("status") == "closed" and t.get("result_r") is not None
    ]
    open_n = sum(1 for t in samples if t.get("status") == "open")
    if not closed:
        return {
            "closed": 0,
            "open": open_n,
            "samples": len(samples),
            "win_rate": None,
            "avg_r": None,
            "total_pnl": None,
            "expectancy_r": None,
            "wins": 0,
            "losses": 0,
            "total_r": None,
            "profit_factor": None,
            "payoff_ratio": None,
            "calibration_gap": None,
            "calibration_samples": 0,
            "path": str(resolve_journal_path()),
        }
    wins = [t for t in closed if float(t["result_r"]) > 0]
    losses = [t for t in closed if float(t["result_r"]) <= 0]
    total_r = sum(float(t["result_r"]) for t in closed)
    avg_r = sum(float(t["result_r"]) for t in closed) / len(closed)
    wr = 100.0 * len(wins) / len(closed)
    gross_win_r = sum(float(t["result_r"]) for t in wins)
    gross_loss_r = abs(sum(float(t["result_r"]) for t in losses))
    avg_win_r = gross_win_r / len(wins) if wins else None
    avg_loss_r = gross_loss_r / len(losses) if losses else None
    calibrated = [
        (t, model_wr)
        for t in closed
        if (model_wr := _valid_model_wr(t.get("model_wr"))) is not None
    ]
    predicted_wr = (
        sum(model_wr for _, model_wr in calibrated) / len(calibrated)
        if calibrated
        else None
    )
    actual_calibrated_wr = (
        100.0
        * sum(1 for t, _ in calibrated if float(t["result_r"]) > 0)
        / len(calibrated)
        if calibrated
        else None
    )
    pnls = [float(t["pnl_usd"]) for t in closed if t.get("pnl_usd") is not None]
    return {
        "closed": len(closed),
        "open": open_n,
        "samples": len(samples),
        "win_rate": round(wr, 1),
        "avg_r": round(avg_r, 3),
        "total_pnl": round(sum(pnls), 2) if pnls else None,
        "expectancy_r": round(avg_r, 3),
        "wins": len(wins),
        "losses": len(losses),
        "total_r": round(total_r, 3),
        "profit_factor": round(gross_win_r / gross_loss_r, 2) if gross_loss_r else None,
        "payoff_ratio": (
            round(avg_win_r / avg_loss_r, 2)
            if avg_win_r is not None and avg_loss_r
            else None
        ),
        "calibration_gap": (
            round(predicted_wr - actual_calibrated_wr, 1)
            if predicted_wr is not None and actual_calibrated_wr is not None
            else None
        ),
        "calibration_samples": len(calibrated),
        "path": str(resolve_journal_path()),
    }
