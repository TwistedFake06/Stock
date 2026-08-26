"""Opening-hours 5-minute scan for the active app watchlist."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from intraday_signals import IntradaySetup, analyze_opening_range_setup, is_intraday_alert_window
from stock_service import DEFAULT_WATCHLIST, fetch_history_extended, normalize_symbol
from views.common import fmt_number

ROOT = Path(__file__).resolve().parents[1]
SCAN_FILE = ROOT / "watchlist_scan.txt"


def _load_default_symbols() -> list[str]:
    if not SCAN_FILE.exists():
        return list(st.session_state.get("watchlist") or DEFAULT_WATCHLIST)
    symbols: list[str] = []
    for line in SCAN_FILE.read_text(encoding="utf-8").splitlines():
        symbol = line.split("#", 1)[0].strip()
        if symbol:
            symbols.append(symbol)
    return symbols or list(st.session_state.get("watchlist") or DEFAULT_WATCHLIST)


def _parse_symbols(raw: str) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for part in raw.replace(",", "\n").replace(";", "\n").splitlines():
        symbol = normalize_symbol(part.strip())
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _format_price(value: float | None) -> str:
    return fmt_number(value, digits=2) if value is not None else "—"


def _row(setup: IntradaySetup) -> dict[str, object]:
    verdict = setup.verdict
    if not is_intraday_alert_window(setup.symbol) and verdict == "可做":
        verdict = "等待下次开市"
    return {
        "状态": verdict,
        "代码": setup.symbol,
        "现价": _format_price(setup.last_price),
        "入场 E": _format_price(setup.entry),
        "止蚀 S": _format_price(setup.stop),
        "T1 (1R)": _format_price(setup.target_1),
        "T2 (2R)": _format_price(setup.target_2),
        "每股风险": _format_price(setup.risk_per_share),
        "分数": setup.score,
        "量比": f"{setup.relative_volume:.1f}x" if setup.relative_volume is not None else "—",
        "前日背景": setup.prior_session_label,
        "背离": "警报" if setup.divergence_warning else "—",
        "执行条件": "；".join(setup.reasons[:3]),
    }


def render_intraday_scan() -> None:
    st.header("盘中超短扫描")
    st.caption(
        "美股 / 港股 · 5分钟线 · 首15分钟区间 + VWAP + Boll + MACD + RSI + KDJ + 量能 + 背离警报"
    )
    st.caption("E 为回踩/突破触发价；S 为硬止损；到 T1 先减仓，T2 再减仓。仅作交易计划辅助，非投资建议。")
    st.caption("美股扫描窗口：09:45–12:00 ET；港股全日扫描：09:45–12:00、13:15–16:00 HKT（午休后重算 VWAP/开盘区间）。")

    default_symbols = "\n".join(_load_default_symbols())
    if "intraday_symbols_text" not in st.session_state:
        st.session_state.intraday_symbols_text = default_symbols
    with st.expander("超短扫描清单（可加入港股）", expanded=False):
        if st.button("载入美股 + 20 港股默认清单", key="load_intraday_defaults"):
            st.session_state.intraday_symbols_text = default_symbols
            st.rerun()
        st.text_area(
            "股票代码（一行一个）",
            key="intraday_symbols_text",
            height=150,
            help="例如 NVDA、0700.HK、9988.HK；此页清单独立于美股快速自选。",
        )
    symbols = _parse_symbols(str(st.session_state.get("intraday_symbols_text") or ""))
    if not symbols:
        st.info("清单为空，请输入美股或港股代码。")
        return

    st.caption(f"扫描 {len(symbols)} 只：{', '.join(symbols[:12])}{'…' if len(symbols) > 12 else ''}")
    rows: list[dict[str, object]] = []
    setups: list[IntradaySetup] = []
    progress = st.progress(0, text="下载 5 分钟 RTH 数据并筛选…")
    for index, symbol in enumerate(symbols):
        try:
            history = fetch_history_extended(symbol, period="5d", interval="5m")
            setup = analyze_opening_range_setup(symbol, history)
        except Exception as exc:
            setup = IntradaySetup(symbol=symbol, reasons=[f"资料读取失败：{type(exc).__name__}"])
        setups.append(setup)
        rows.append(_row(setup))
        progress.progress((index + 1) / len(symbols), text=f"分析 {symbol}…")
    progress.empty()

    table = pd.DataFrame(rows)
    rank = {"可做": 3, "等待": 2, "等待下次开市": 1, "不做": 0, "资料不足": -1}
    table["_rank"] = table["状态"].map(rank).fillna(-1)
    table = table.sort_values(["_rank", "分数"], ascending=False).drop(columns="_rank")

    tradeable = table[table["状态"] == "可做"]
    st.subheader(f"可做 · {len(tradeable)}")
    if tradeable.empty:
        st.caption("目前没有同时满足突破、VWAP、动能、量能及不追价条件的标的。空手是有效结果。")
    else:
        st.dataframe(tradeable, width="stretch", hide_index=True)

    waiting = table[table["状态"].isin(["等待", "等待下次开市"])]
    st.subheader(f"等待条件 · {len(waiting)}")
    if waiting.empty:
        st.caption("目前没有等待确认的标的。")
    else:
        st.dataframe(waiting, width="stretch", hide_index=True)

    rejected = table[table["状态"].isin(["不做", "资料不足"])]
    with st.expander(f"暂不做 / 资料不足 · {len(rejected)}", expanded=False):
        if rejected.empty:
            st.caption("没有。")
        else:
            st.dataframe(rejected, width="stretch", hide_index=True)

    with st.expander("逐只理由与执行规则", expanded=False):
        for setup in sorted(setups, key=lambda item: item.score, reverse=True):
            st.markdown(f"**{setup.symbol} · {setup.verdict} · {setup.score}/100**")
            if setup.entry is not None:
                st.caption(
                    f"E {_format_price(setup.entry)} · S {_format_price(setup.stop)} · "
                    f"T1 {_format_price(setup.target_1)} · T2 {_format_price(setup.target_2)}"
                )
            for reason in setup.reasons:
                st.write(f"- {reason}")
