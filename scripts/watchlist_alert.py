"""
Auto-scan watchlist and notify when short-term entry is OK.

Default: every 5 minutes loop, or one-shot with --once.

Notifications:
  - Telegram Bot (recommended, free): TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  - Generic webhook (Make.com / n8n / Zapier → WhatsApp etc.): ALERT_WEBHOOK_URL

Usage:
  # one scan
  .venv\\Scripts\\python.exe scripts\\watchlist_alert.py --once

  # loop every 5 minutes (keep PC awake)
  .venv\\Scripts\\python.exe scripts\\watchlist_alert.py --interval 300

  # only alert 可以入場 (stricter)
  .venv\\Scripts\\python.exe scripts\\watchlist_alert.py --min enter

Alert state is saved so the same symbol won't spam every 5 minutes
until it leaves "enterable" and becomes enterable again (or cooldown expires).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env if present (same keys as free_data)
try:
    from free_data import _load_local_env_once

    _load_local_env_once()
except Exception:
    env_path = ROOT / ".env"
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not (os.environ.get(k) or "").strip():
                os.environ[k] = v

from stock_service import DEFAULT_WATCHLIST, normalize_symbol
from trade_sop import build_trade_sop

DEFAULT_LIST = ROOT / "watchlist_scan.txt"
STATE_FILE = ROOT / "data" / "alert_state.json"


def _load_symbols(path: Path | None) -> list[str]:
    p = path or DEFAULT_LIST
    raw: list[str] = []
    if p and p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                raw.append(line)
    else:
        raw = list(DEFAULT_WATCHLIST)
    out: list[str] = []
    seen: set[str] = set()
    for s in raw:
        n = normalize_symbol(s)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"alerts": {}}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"alerts": {}}


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_enterable(sop, min_level: str) -> bool:
    """
    min_level:
      enter  → 适合入场 / 可以入場 only
      try    → also 谨慎试仓 / 可以試倉
      all    → any non-avoid (not used for notify usually)
    """
    ok = sop.enter_ok
    h1 = getattr(sop, "swing_h1", None)
    v = getattr(h1, "verdict", "") if h1 else ""
    if min_level == "enter":
        return ok == "适合入场" or v == "可以入場"
    # try (default)
    return ok in ("适合入场", "谨慎试仓") or v in ("可以入場", "可以試倉")


def _format_alert(sop) -> str:
    h1 = getattr(sop, "swing_h1", None)
    h2 = getattr(sop, "swing_h2", None)
    lines = [
        f"📣 短线信号 · {sop.symbol} ({sop.name})",
        f"结论: {sop.enter_ok}"
        + (f" · 0–2周: {h1.verdict}" if h1 else ""),
        f"现价: {sop.last_price}",
        f"入場: {sop.entry_low} – {sop.entry_high}  掛單≈{sop.entry_plan}",
        f"止蝕: {sop.stop_loss}",
        f"目标0–2周: {getattr(h1, 'target', None) or sop.target_t1}  "
        f"胜率: {getattr(h1, 'win_rate_pct', None) or sop.win_rate_pct}%  "
        f"R:R: {getattr(h1, 'rr', None) or sop.rr_t1}",
    ]
    if h2:
        lines.append(
            f"目标2–4周: {h2.target}  胜率: {h2.win_rate_pct}%  结论: {h2.verdict}"
        )
    if getattr(sop, "trend_align_label", None):
        lines.append(f"跟势: {sop.trend_align_label}")
    if getattr(sop, "false_break_label", None):
        lines.append(f"假突破: {sop.false_break_label}")
    lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("（模型辅助，非投资建议）")
    return "\n".join(str(x) for x in lines)


def send_telegram(text: str) -> tuple[bool, str]:
    token = (
        os.environ.get("TELEGRAM_BOT_TOKEN")
        or os.environ.get("TG_BOT_TOKEN")
        or ""
    ).strip()
    chat = (
        os.environ.get("TELEGRAM_CHAT_ID")
        or os.environ.get("TG_CHAT_ID")
        or ""
    ).strip()
    if not token or not chat:
        return False, "未配置 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat, "text": text},
            timeout=20,
        )
        if r.status_code == 200:
            return True, "telegram ok"
        return False, f"telegram HTTP {r.status_code}: {r.text[:200]}"
    except Exception as exc:
        return False, f"telegram error: {exc}"


def send_webhook(text: str, payload: dict[str, Any]) -> tuple[bool, str]:
    url = (os.environ.get("ALERT_WEBHOOK_URL") or "").strip()
    if not url:
        return False, "未配置 ALERT_WEBHOOK_URL"
    try:
        r = requests.post(
            url,
            json={"text": text, **payload},
            timeout=20,
        )
        if 200 <= r.status_code < 300:
            return True, "webhook ok"
        return False, f"webhook HTTP {r.status_code}: {r.text[:200]}"
    except Exception as exc:
        return False, f"webhook error: {exc}"


def notify(text: str, payload: dict[str, Any]) -> list[str]:
    logs: list[str] = []
    ok_t, msg_t = send_telegram(text)
    logs.append(msg_t)
    ok_w, msg_w = send_webhook(text, payload)
    if "未配置" not in msg_w:
        logs.append(msg_w)
    elif not ok_t and "未配置" in msg_t:
        logs.append("没有可用通知渠道：请配置 Telegram 或 ALERT_WEBHOOK_URL")
    return logs


def should_alert(
    state: dict[str, Any],
    symbol: str,
    verdict_key: str,
    cooldown_sec: int,
) -> bool:
    """
    Alert only on NEW enterable status, or after cooldown if still enterable
    (default: only on transition to enterable).
    """
    alerts: dict = state.setdefault("alerts", {})
    prev = alerts.get(symbol) or {}
    prev_key = prev.get("verdict")
    prev_ts = float(prev.get("ts") or 0)
    now = time.time()
    # New enterable (was not enterable)
    if prev_key != verdict_key:
        return True
    # Same status: only re-alert if cooldown expired and --repeat
    if cooldown_sec > 0 and (now - prev_ts) >= cooldown_sec:
        return True
    return False


def mark_alerted(state: dict[str, Any], symbol: str, verdict_key: str) -> None:
    state.setdefault("alerts", {})[symbol] = {
        "verdict": verdict_key,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(),
    }


def clear_if_not_enterable(state: dict[str, Any], symbol: str) -> None:
    """Reset state when symbol leaves enterable so next enter triggers again."""
    alerts = state.setdefault("alerts", {})
    if symbol in alerts:
        alerts[symbol]["verdict"] = "out"


def run_scan(
    *,
    symbols: list[str],
    period: str,
    capital_usd: float,
    risk_pct: float,
    min_level: str,
    cooldown_sec: int,
    repeat: bool,
    dry_run: bool,
) -> dict[str, Any]:
    state = _load_state()
    hits = []
    errors = []
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 扫描 {len(symbols)} 只 · "
        f"min={min_level} · period={period}"
    )
    for sym in symbols:
        try:
            sop = build_trade_sop(
                sym,
                period=period,
                capital=capital_usd,
                risk_pct=risk_pct,
            )
            ok = _is_enterable(sop, min_level)
            h1v = getattr(getattr(sop, "swing_h1", None), "verdict", "") or ""
            print(
                f"  {sym:8} {sop.enter_ok:8} h1={h1v or '—':8} "
                f"last={sop.last_price} wr={sop.win_rate_pct}"
            )
            if not ok:
                clear_if_not_enterable(state, sym)
                continue
            verdict_key = f"{sop.enter_ok}|{h1v}"
            # Without --repeat: only transition; with repeat: also cooldown
            if not repeat:
                # transition only: force cooldown_sec=0 path means only when key changes
                if not should_alert(state, sym, verdict_key, cooldown_sec=0):
                    # same as last alerted key
                    if (state.get("alerts") or {}).get(sym, {}).get("verdict") == verdict_key:
                        print(f"    (已通知过，跳过) {sym}")
                        continue
            else:
                if not should_alert(state, sym, verdict_key, cooldown_sec):
                    print(f"    (冷却中，跳过) {sym}")
                    continue

            text = _format_alert(sop)
            payload = {
                "symbol": sop.symbol,
                "name": sop.name,
                "enter_ok": sop.enter_ok,
                "swing_h1": h1v,
                "last": sop.last_price,
                "entry_low": sop.entry_low,
                "entry_high": sop.entry_high,
                "stop": sop.stop_loss,
                "target": sop.target_t1,
                "win_rate": sop.win_rate_pct,
            }
            if dry_run:
                print("    [dry-run] 不发送:\n" + text)
                logs = ["dry-run"]
            else:
                logs = notify(text, payload)
                for lg in logs:
                    print(f"    notify: {lg}")
            mark_alerted(state, sym, verdict_key)
            hits.append(sym)
        except Exception as exc:
            err = f"{sym}: {type(exc).__name__}: {exc}"
            errors.append(err)
            print(f"  ERROR {err}")
            traceback.print_exc()
    _save_state(state)
    print(f"完成 · 新通知 {len(hits)} · 错误 {len(errors)}")
    return {"hits": hits, "errors": errors}


def main() -> None:
    p = argparse.ArgumentParser(description="Watchlist auto-scan + Telegram/webhook alert")
    p.add_argument("--file", "-f", default=str(DEFAULT_LIST), help="symbol list file")
    p.add_argument("--period", default="1y")
    p.add_argument("--capital-hkd", type=float, default=50_000.0)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--hkd-per-usd", type=float, default=7.8)
    p.add_argument(
        "--min",
        dest="min_level",
        choices=["enter", "try"],
        default="try",
        help="enter=仅适合入场; try=适合+谨慎 (default)",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=300,
        help="loop seconds (default 300 = 5 min). Ignored with --once",
    )
    p.add_argument("--once", action="store_true", help="run one scan and exit")
    p.add_argument(
        "--repeat",
        action="store_true",
        help="re-alert same symbol after --cooldown (default: only on new signal)",
    )
    p.add_argument(
        "--cooldown",
        type=int,
        default=3600,
        help="seconds before re-alert same symbol if --repeat (default 3600)",
    )
    p.add_argument("--dry-run", action="store_true", help="scan only, no notify")
    args = p.parse_args()

    symbols = _load_symbols(Path(args.file) if args.file else None)
    if not symbols:
        print("清单为空")
        sys.exit(1)
    capital_usd = float(args.capital_hkd) / float(args.hkd_per_usd)

    # Channel check
    has_tg = bool(
        (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        and (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    )
    has_wh = bool((os.environ.get("ALERT_WEBHOOK_URL") or "").strip())
    print(f"通知渠道: Telegram={'ON' if has_tg else 'OFF'}  Webhook={'ON' if has_wh else 'OFF'}")
    if not has_tg and not has_wh and not args.dry_run:
        print(
            "警告: 未配置通知。请在 .env 设置 TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID\n"
            "或 ALERT_WEBHOOK_URL（可接 Make/n8n 发 WhatsApp）\n"
            "也可先用 --dry-run 测试扫描。"
        )

    def once() -> None:
        run_scan(
            symbols=symbols,
            period=args.period,
            capital_usd=capital_usd,
            risk_pct=float(args.risk_pct),
            min_level=args.min_level,
            cooldown_sec=int(args.cooldown) if args.repeat else 0,
            repeat=bool(args.repeat),
            dry_run=bool(args.dry_run),
        )

    if args.once:
        once()
        return

    interval = max(60, int(args.interval))
    print(f"循环模式: 每 {interval}s 扫描一次（Ctrl+C 停止）")
    while True:
        try:
            once()
        except KeyboardInterrupt:
            print("停止")
            break
        except Exception:
            traceback.print_exc()
        time.sleep(interval)


if __name__ == "__main__":
    main()
