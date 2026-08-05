"""
Scan a symbol list with Trade SOP and print who can enter.

Usage:
  python scripts/scan_watchlist_sop.py
  python scripts/scan_watchlist_sop.py AAPL NVDA 0700.HK
  python scripts/scan_watchlist_sop.py --file my_list.txt

Default file: watchlist_scan.txt (one symbol per line, # comments ok)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_service import DEFAULT_WATCHLIST, normalize_symbol
from trade_sop import build_trade_sop

DEFAULT_LIST = ROOT / "watchlist_scan.txt"


def load_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        raw = args.symbols
    elif args.file and Path(args.file).exists():
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        raw = []
        for line in lines:
            line = line.split("#")[0].strip()
            if line:
                raw.append(line)
    elif DEFAULT_LIST.exists():
        lines = DEFAULT_LIST.read_text(encoding="utf-8").splitlines()
        raw = []
        for line in lines:
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


def main() -> None:
    p = argparse.ArgumentParser(description="SOP entry scan for a watchlist")
    p.add_argument("symbols", nargs="*", help="Symbols e.g. AAPL NVDA 0700.HK")
    p.add_argument("--file", "-f", default="", help="Text file, one symbol per line")
    p.add_argument("--period", default="1y", help="History period (default 1y)")
    p.add_argument(
        "--capital-hkd",
        type=float,
        default=50_000.0,
        help="Account capital in HKD (default 50000); converted at 7.8 for USD stocks",
    )
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--hkd-per-usd", type=float, default=7.8)
    p.add_argument(
        "--min",
        dest="min_level",
        choices=["suitable", "cautious", "all"],
        default="cautious",
        help="suitable=适合 only; cautious=适合+谨慎; all=print all",
    )
    args = p.parse_args()
    symbols = load_symbols(args)
    capital_usd = float(args.capital_hkd) / float(args.hkd_per_usd)
    print(
        f"Scanning {len(symbols)} symbols | period={args.period} | "
        f"capital=HKD {args.capital_hkd:.0f} (~USD {capital_usd:.0f}) 1R={args.risk_pct}%"
    )
    print("-" * 88)

    rows = []
    for sym in symbols:
        try:
            r = build_trade_sop(
                sym,
                period=args.period,
                capital=capital_usd,
                risk_pct=float(args.risk_pct),
            )
            rows.append(r)
            print(
                f"{r.symbol:12} {r.enter_ok:8} {r.enter_score:5.1f}  "
                f"last={r.last_price}  wr={r.win_rate_pct}  stab={r.stability_score}  "
                f"bias={r.bias}"
            )
        except Exception as exc:
            print(f"{sym:12} ERROR {type(exc).__name__}: {exc}")

    def keep(ok: str) -> bool:
        if args.min_level == "all":
            return True
        if args.min_level == "suitable":
            return ok == "适合入场"
        return ok in ("适合入场", "谨慎试仓")

    good = [r for r in rows if keep(r.enter_ok)]
    print("-" * 88)
    print(f"CAN ENTER ({args.min_level}): {len(good)} / {len(rows)}")
    for r in sorted(good, key=lambda x: -x.enter_score):
        print()
        print(f"### {r.symbol}  {r.name}  →  {r.enter_ok} ({r.enter_score:.0f}/100)")
        print(f"  现价 {r.last_price}  |  入场区 {r.entry_low} ~ {r.entry_high}  |  挂单≈{r.entry_plan}")
        print(f"  止损 {r.stop_loss}  |  T1 {r.target_t1}  T2 {r.target_t2}  |  R:R {r.rr_t1}")
        print(f"  胜率 {r.win_rate_pct}%({r.win_rate_label})  |  稳定度 {r.stability_score}({r.stability_label})")
        print(f"  多空 {r.bias}({r.bias_score:+.0f})  |  建议股数 {r.position_shares}  |  {r.position_note}")
        if r.actions_now:
            print("  立刻:")
            for a in r.actions_now[:4]:
                print(f"    - {a}")
        if r.invalidation:
            print(f"  失效: {r.invalidation}")
    print()
    print("Not investment advice. Re-check in App before trading.")


if __name__ == "__main__":
    main()
