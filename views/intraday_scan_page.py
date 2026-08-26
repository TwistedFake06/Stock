"""Opening-hours 5-minute scan for the active app watchlist."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from intraday_signals import IntradaySetup, analyze_opening_range_setup
from market_session import us_session_clock
from stock_service import DEFAULT_WATCHLIST, fetch_history_extended
from views.common import fmt_number


def _is_opening_window() -> bool:
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/New_York"))
        return now.weekday() < 5 and (9, 45) <= (now.hour, now.minute) < (12, 0)
    except Exception:
        return False


def _format_price(value: float | None) -> str:
    return fmt_number(value, digits=2) if value is not None else "—"


def _row(setup: IntradaySetup, live_window: bool) -> dict[str, object]:
    verdict = setup.verdict
    if not live_window and verdict == "可做":
        verdict = "等待下次RTH"
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
        "背离": "警报" if setup.divergence_warning else "—",
        "执行条件": "；".join(setup.reasons[:3]),
    }


def render_intraday_scan() -> None:
    st.header("开市超短扫描")
    st.caption(
        "自选股 · 美股 RTH 5分钟线 · 首15分钟区间 + VWAP + Boll + MACD + RSI + KDJ + 量能 + 背离警报"
    )
    st.caption("E 为回踩/突破触发价；S 为硬止损；到 T1 先减仓，T2 再减仓。仅作交易计划辅助，非投资建议。")

    clock = us_session_clock()
    live_window = _is_opening_window()
    if live_window:
        st.success(f"{clock.label_zh} · {clock.et_now} · 正在开市首三小时，允许检查即时 setup。")
    else:
        st.warning(
            f"{clock.label_zh} · {clock.et_now} · 这页的即时入场窗口为 09:45–12:00 ET；"
            "当前结果只供下一次 RTH 准备，不应据此下单。"
        )

    symbols = list(st.session_state.get("watchlist") or DEFAULT_WATCHLIST)
    if not symbols:
        st.info("自选股为空，请先到「自选股」添加美股代码。")
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
        rows.append(_row(setup, live_window))
        progress.progress((index + 1) / len(symbols), text=f"分析 {symbol}…")
    progress.empty()

    table = pd.DataFrame(rows)
    rank = {"可做": 3, "等待": 2, "等待下次RTH": 1, "不做": 0, "资料不足": -1}
    table["_rank"] = table["状态"].map(rank).fillna(-1)
    table = table.sort_values(["_rank", "分数"], ascending=False).drop(columns="_rank")

    tradeable = table[table["状态"] == "可做"]
    st.subheader(f"可做 · {len(tradeable)}")
    if tradeable.empty:
        st.caption("目前没有同时满足突破、VWAP、动能、量能及不追价条件的标的。空手是有效结果。")
    else:
        st.dataframe(tradeable, width="stretch", hide_index=True)

    waiting = table[table["状态"].isin(["等待", "等待下次RTH"])]
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
