"""One-off: scan watchlist for today's enterable setups."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_watchlist_swing import load_symbols
from trade_sop import ENTRY_BY_WR_ONLY, build_trade_sop


def main() -> None:
    symbols = load_symbols(ROOT / "watchlist_scan.txt")
    print("ENTRY_BY_WR_ONLY", ENTRY_BY_WR_ONLY)
    print("symbols", len(symbols))
    print("mode=A defensive · horizon=h1 (0–2周)")
    print("scanning...")

    rows: list[dict] = []
    errors: list[tuple[str, str]] = []

    for sym in symbols:
        try:
            sop = build_trade_sop(
                sym,
                period="1y",
                interval="1d",
                primary_horizon="h1",
                mode="defensive",
            )
            h1 = sop.swing_h1
            prim = sop.primary_plan or h1
            verdict = getattr(prim, "verdict", None) or "—"
            row = {
                "sym": sop.symbol,
                "name": (sop.name or "")[:18],
                "enter": sop.enter_ok,
                "verdict": verdict,
                "last": sop.last_price,
                "wr": getattr(prim, "win_rate_display", None) or sop.win_rate_display,
                "wr_pct": getattr(prim, "win_rate_pct", None),
                "plan": getattr(prim, "entry_plan", None) or sop.entry_plan,
                "stop": getattr(prim, "stop_loss", None) or sop.stop_loss,
                "tgt": getattr(prim, "target", None) or sop.target_t1,
                "lo": getattr(prim, "entry_low", None) or sop.entry_low,
                "hi": getattr(prim, "entry_high", None) or sop.entry_high,
                "rr": getattr(prim, "rr", None),
                "rr_net": getattr(prim, "rr_net", None),
                "exp": getattr(prim, "expectancy_r", None),
                "bias": sop.bias,
            }
            rows.append(row)
            print(f"  {sym}: {verdict} | {sop.enter_ok} | wr={row['wr']}")
        except Exception as exc:
            errors.append((sym, f"{type(exc).__name__}: {exc}"))
            print(f"  {sym} ERR {type(exc).__name__}: {exc}")

    def is_full(r: dict) -> bool:
        return r["verdict"] == "可以入場" or r["enter"] == "适合入场"

    def is_half(r: dict) -> bool:
        return r["verdict"] == "可以試倉" or r["enter"] == "谨慎试仓"

    def rank(r: dict) -> tuple:
        if is_full(r):
            return (0, -(r["wr_pct"] or 0))
        if is_half(r):
            return (1, -(r["wr_pct"] or 0))
        return (2, -(r["wr_pct"] or 0))

    rows.sort(key=rank)
    ok = [r for r in rows if is_full(r)]
    half = [r for r in rows if is_half(r)]

    def line(r: dict) -> str:
        exp = f"{r['exp']:+.2f}" if r["exp"] is not None else "—"
        return (
            f"{r['sym']:6} {r['name']:<18} 现价={r['last']}  "
            f"胜率={r['wr']}  "
            f"区={r['lo']}-{r['hi']} 挂={r['plan']} 止={r['stop']} 标={r['tgt']}  "
            f"R:R={r['rr']} 净={r['rr_net']} E[R]={exp}  {r['bias']}"
        )

    print()
    print("=" * 100)
    print("可入場（满仓线）")
    print("=" * 100)
    if not ok:
        print("(无)")
    for r in ok:
        print(line(r))

    print()
    print("=" * 100)
    print("可以試倉")
    print("=" * 100)
    if not half:
        print("(无)")
    for r in half:
        print(line(r))

    print()
    print("=" * 100)
    print("其余（观望/不做多）")
    print("=" * 100)
    for r in rows:
        if is_full(r) or is_half(r):
            continue
        print(
            f"{r['sym']:6} {r['verdict']}/{r['enter']}  胜率={r['wr']}  bias={r['bias']}"
        )

    if errors:
        print()
        print("errors:")
        for s, e in errors:
            print(f"  {s}: {e}")

    print()
    print(
        f"合计: 满仓 {len(ok)} · 试仓 {len(half)} · "
        f"其他 {len(rows) - len(ok) - len(half)} · 失败 {len(errors)}"
    )
    print("说明: 实时快照·非投资建议·请再对投资SOP一屏确认挂单/止损")


if __name__ == "__main__":
    main()
