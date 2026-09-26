# -*- coding: utf-8 -*-
"""Weekday Confirm + multi-strategy digest for Stock routine.

Soft-Confirm paper experiment (A-tier only, RR≥0.9 → soft 可入場):
  start 2026-09-26 → end ~2026-10-10. Research / paper only — no orders.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from entry_labels import (
    ENTER_YES,
    ENTER_SOFT,
    ENTER_MAYBE,
    ENTER_NO,
    HARD_RR_MIN,
    SOFT_RR_MIN,
    screen_layer,
    label_enter_ok,
    digest_bucket,
)
from config import CORE_WATCHLIST, CORE_A_TIER, CORE_A_TIER_SET, CORE_B_TIER
from trade_sop import build_trade_sop

try:
    from views.scan_page import _block_reasons, _distance_to_entry
except Exception:
    _block_reasons = None
    _distance_to_entry = None

# Soft-Confirm paper experiment window (HKT calendar)
SOFT_EXPERIMENT = {
    "name": "soft-Confirm A-tier",
    "start": "2026-09-26",
    "end": "2026-10-10",
    "mode": "paper",
    "hard_rr_min": HARD_RR_MIN,
    "soft_rr_min": SOFT_RR_MIN,
    "a_tier_n": len(CORE_A_TIER),
    "note": "A-tier soft 可入場 RR≥0.9 when 可考慮/close; B never soft; no orders",
}


def _light_zh(x: str) -> str:
    return {"green": "綠", "yellow": "黃", "red": "紅"}.get(x or "", "—")


def _fallback_block(sop) -> str:
    reasons = []
    for light, label in (
        (getattr(sop, "position_light", ""), "位置紅"),
        (getattr(sop, "wr_light", ""), "勝率紅"),
        (getattr(sop, "rr_light", ""), "劃算紅"),
        (getattr(sop, "position_light", ""), "位置黃"),
        (getattr(sop, "wr_light", ""), "勝率黃"),
        (getattr(sop, "rr_light", ""), "劃算黃"),
    ):
        if light == "red" and "紅" in label:
            reasons.append(label)
        if light == "yellow" and "黃" in label:
            reasons.append(label)
    # dedupe preserving order
    seen = set()
    out = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return " · ".join(out)


def _row_payload(r: dict, *, include_enter_ok: bool = False) -> dict:
    d = {
        "symbol": r["symbol"],
        "line": r["line"],
        "lights": r["lights"],
        "last": r["last"],
        "rr": r["rr"],
        "tier": r.get("core_tier"),
    }
    if include_enter_ok:
        d["enter_ok"] = r["enter_ok"]
    if r.get("bucket") == ENTER_SOFT:
        d["label"] = ENTER_SOFT
        d["paper"] = True
    return d


def main() -> int:
    tz = ZoneInfo("Asia/Taipei")
    now = datetime.now(tz)
    et_str = None
    us_session = None
    try:
        from zoneinfo import ZoneInfo as ZI
        et_str = datetime.now(ZI("America/New_York")).strftime("%Y-%m-%d %H:%M ET")
    except Exception:
        pass
    try:
        from market_session import us_session_clock
        clock = us_session_clock()
        us_session = getattr(clock, "session", None)
        if getattr(clock, "et_now", None):
            et_str = str(clock.et_now)
    except Exception:
        pass

    CORE = list(CORE_WATCHLIST)
    capital = 50000.0 / 7.8
    yes_rows, soft_rows, maybe_rows, all_rows = [], [], [], []
    no_count = 0
    errors = []

    print(f"HKT={now.isoformat()} ET={et_str} session={us_session} CORE={len(CORE)}")
    print(
        f"soft-Confirm paper: {SOFT_EXPERIMENT['start']}→{SOFT_EXPERIMENT['end']} "
        f"A={len(CORE_A_TIER)} B={len(CORE_B_TIER)} soft_rr>={SOFT_RR_MIN} (no orders)"
    )
    print("-" * 72)

    for sym in CORE:
        try:
            sop = build_trade_sop(
                sym,
                period="1y",
                interval="1d",
                capital=capital,
                risk_pct=1.0,
                primary_horizon="h1",
                mode="defensive",
                include_h1=False,
            )
            pos_l = getattr(sop, "position_light", "") or ""
            wr_l = getattr(sop, "wr_light", "") or ""
            rr_l = getattr(sop, "rr_light", "") or ""
            any_red = "red" in (pos_l, wr_l, rr_l)
            any_yellow = "yellow" in (pos_l, wr_l, rr_l)
            layer = screen_layer(sop.enter_ok, any_red=any_red, any_yellow=any_yellow)
            rr_val = getattr(sop, "rr_t1", None)
            bucket = digest_bucket(
                sop.symbol,
                layer=layer,
                rr=rr_val,
                enter_ok=sop.enter_ok,
                a_tier=CORE_A_TIER_SET,
            )
            core_tier = "A" if sop.symbol in CORE_A_TIER_SET else "B"

            block = ""
            dist = ""
            if _block_reasons is not None:
                try:
                    block = _block_reasons(sop, None) or ""
                except Exception:
                    block = _fallback_block(sop)
            else:
                block = _fallback_block(sop)
            if _distance_to_entry is not None:
                try:
                    _pct, dist = _distance_to_entry(sop.last_price, sop.entry_low, sop.entry_high)
                except Exception:
                    dist = ""

            one = (getattr(sop, "one_liner_reason", "") or "").strip() or block or label_enter_ok(sop.enter_ok)
            line = one
            if dist and dist not in ("—", ""):
                line = f"{one} · {dist}"
            lights = f"位{_light_zh(pos_l)}/勝{_light_zh(wr_l)}/劃{_light_zh(rr_l)}"
            row = {
                "symbol": sop.symbol,
                "layer": layer,
                "bucket": bucket,
                "core_tier": core_tier,
                "enter_ok": sop.enter_ok,
                "line": line,
                "lights": lights,
                "block": block,
                "last": sop.last_price,
                "rr": rr_val,
            }
            all_rows.append(row)
            tag = bucket if bucket != layer else layer
            print(f"  {sym:6} {tag} [{sop.enter_ok}] {core_tier} {lights} rr={rr_val} | {line[:90]}")
            if bucket == ENTER_YES:
                yes_rows.append(row)
            elif bucket == ENTER_SOFT:
                soft_rows.append(row)
            elif bucket == ENTER_MAYBE:
                maybe_rows.append(row)
            else:
                no_count += 1
        except Exception as exc:
            errors.append({"symbol": sym, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {sym:6} ERROR {type(exc).__name__}: {exc}")

    ms_top = []
    ms_bear_top = []
    try:
        from multi_strategy_scan import scan_symbols
        ms_rows = scan_symbols(list(CORE), period="1y", interval="1d", core_only=True)
        for r in ms_rows or []:
            if isinstance(r, dict):
                tier = r.get("suggest_tier") or r.get("tier")
                hc = r.get("hit_count") or 0
                labels = r.get("hit_labels") or r.get("labels") or []
                sym = r.get("symbol") or r.get("code")
                hits = r.get("hits")
            else:
                tier = getattr(r, "suggest_tier", None)
                hc = getattr(r, "hit_count", 0)
                labels = getattr(r, "hit_labels", None) or getattr(r, "labels", "")
                sym = getattr(r, "symbol", None)
                hits = getattr(r, "hits", None)
            try:
                hc_i = int(hc or 0)
            except Exception:
                hc_i = 0
            if hc_i < 2 and tier not in (ENTER_YES, ENTER_MAYBE, "可入場", "可考慮"):
                continue
            if isinstance(labels, str) and labels:
                lab = [x.strip() for x in labels.replace("、", ",").split(",") if x.strip()]
            elif isinstance(labels, (list, tuple)):
                lab = []
                for x in labels:
                    if hasattr(x, "label") and getattr(x, "fired", False):
                        lab.append(getattr(x, "label"))
                    else:
                        lab.append(str(x))
            elif hits:
                lab = [h.label for h in hits if getattr(h, "fired", False)]
            else:
                lab = []
            if hc_i >= 2 or tier in (ENTER_YES, ENTER_MAYBE, "可入場", "可考慮"):
                ms_top.append({"symbol": sym, "tier": tier, "n": hc_i, "labels": lab})
        ms_top.sort(key=lambda x: (-int(x["n"] or 0), str(x.get("symbol") or "")))
        ms_top = ms_top[:10]

        # Optional bearish multi top (separate counters; does not affect 多策略_top)
        ms_bear_top = []
        for r in ms_rows or []:
            if isinstance(r, dict):
                bt = r.get("bear_tier")
                bhc = r.get("bear_hit_count") or 0
                blabels = r.get("bear_hit_labels") or ""
                sym = r.get("symbol") or r.get("code")
                bhits = r.get("bear_hits")
            else:
                bt = getattr(r, "bear_tier", None)
                bhc = getattr(r, "bear_hit_count", 0)
                blabels = getattr(r, "bear_hit_labels", "") or ""
                sym = getattr(r, "symbol", None)
                bhits = getattr(r, "bear_hits", None)
            try:
                bhc_i = int(bhc or 0)
            except Exception:
                bhc_i = 0
            if bhc_i < 2 and bt not in ("偏空強", "偏空留意"):
                continue
            if isinstance(blabels, str) and blabels and blabels != "—":
                blab = [x.strip() for x in blabels.replace("、", ",").split(",") if x.strip()]
            elif bhits:
                blab = [h.label for h in bhits if getattr(h, "fired", False)]
            else:
                blab = []
            ms_bear_top.append({"symbol": sym, "tier": bt, "n": bhc_i, "labels": blab})
        ms_bear_top.sort(key=lambda x: (-int(x["n"] or 0), str(x.get("symbol") or "")))
        ms_bear_top = ms_bear_top[:8]
    except Exception as exc:
        errors.append({"symbol": "multi_strategy", "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        ms_bear_top = []

    out = {
        "hkt": now.strftime("%Y-%m-%d %H:%M HKT"),
        "et": et_str,
        "us_session": us_session,
        "source": "windows-DESKTOP-I6GFS6B",
        "core_n": len(CORE),
        "experiment": SOFT_EXPERIMENT,
        "可入場": [_row_payload(r) for r in yes_rows],
        "soft_可入場": [_row_payload(r) for r in soft_rows],
        "可考慮": [_row_payload(r, include_enter_ok=True) for r in maybe_rows],
        "不入場_count": no_count,
        "多策略_top": ms_top,
        "多策略_看空_top": ms_bear_top,
        "errors": errors,
        "yes_symbols": [r["symbol"] for r in yes_rows],
        "soft_symbols": [r["symbol"] for r in soft_rows],
        "counts": {
            "可入場": len(yes_rows),
            "soft_可入場": len(soft_rows),
            "可考慮": len(maybe_rows),
            "不入場": no_count,
        },
    }
    out_path = ROOT / "data" / "confirm_digest_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("-" * 72)
    print(json.dumps({
        "hkt": out["hkt"],
        "experiment": out["experiment"],
        "counts": out["counts"],
        "可入場": out["可入場"],
        "soft_可入場": out["soft_可入場"],
        "可考慮": [{"symbol": r["symbol"], "line": r["line"], "lights": r["lights"], "rr": r.get("rr")} for r in out["可考慮"]],
        "不入場_count": out["不入場_count"],
        "多策略_top": out["多策略_top"],
        "多策略_看空_top": out.get("多策略_看空_top", []),
        "errors": out["errors"],
        "yes_symbols": out["yes_symbols"],
        "soft_symbols": out["soft_symbols"],
    }, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")
    print(
        f"SECTIONS hard={len(yes_rows)} soft={len(soft_rows)} maybe={len(maybe_rows)} no={no_count} "
        f"| paper experiment — no orders"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
