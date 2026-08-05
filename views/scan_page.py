"""Watchlist SOP scan page — one screen to find enterable names."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from stock_service import DEFAULT_WATCHLIST, normalize_symbol
from trade_sop import build_trade_sop

ROOT = Path(__file__).resolve().parents[1]
SCAN_FILE = ROOT / "watchlist_scan.txt"


def _load_scan_file() -> list[str]:
    if not SCAN_FILE.exists():
        return []
    out: list[str] = []
    for line in SCAN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(line)
    return out


def _save_scan_file(symbols: list[str]) -> None:
    body = (
        "# Watchlist for SOP scan page\n"
        "# One symbol per line\n"
        + "\n".join(symbols)
        + "\n"
    )
    try:
        SCAN_FILE.write_text(body, encoding="utf-8")
    except OSError:
        pass


def _parse_text(raw: str) -> list[str]:
    raw = raw.replace(",", "\n").replace(";", "\n").replace(" ", "\n")
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.splitlines():
        part = part.split("#")[0].strip()
        if not part:
            continue
        n = normalize_symbol(part)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def render_scan(period: str, interval: str, period_label: str) -> None:
    st.markdown("## Watchlist 掃描（投资SOP）")
    st.caption(
        f"一鍵掃清單 → 适合入场 / 谨慎试仓 · {period_label} · "
        "實盤輔助，非投資建議"
    )

    # ---- list source ----
    file_syms = _load_scan_file()
    session_wl = list(st.session_state.get("watchlist") or DEFAULT_WATCHLIST)

    default_text = "\n".join(file_syms) if file_syms else "\n".join(session_wl[:20])

    st.markdown("### 掃描清單")
    src = st.radio(
        "清單來源",
        ["編輯下方文字", "App 自選股", "預設 DEFAULT_WATCHLIST"],
        horizontal=True,
        index=0,
    )

    text = st.text_area(
        "股票代碼（一行一個，或逗號分隔）",
        value=default_text,
        height=180,
        help="例：AAPL / NVDA / 0700.HK / SPY",
        key="scan_list_text",
    )

    HKD_PER_USD = 7.8
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        capital_hkd = st.number_input(
            "本金 HKD",
            min_value=1000.0,
            value=50000.0,
            step=1000.0,
            help="默认 50,000 HKD",
        )
    with c2:
        risk_pct = st.number_input("1R %", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    with c3:
        min_level = st.selectbox(
            "顯示級別",
            ["适合+谨慎", "仅适合入场", "全部"],
            index=0,
        )
    with c4:
        st.write("")
        st.write("")
        save_list = st.checkbox("掃完寫入 watchlist_scan.txt", value=True)
    capital = float(capital_hkd) / HKD_PER_USD
    st.caption(f"仓位按 USD 计价：约 ${capital:,.0f}（{HKD_PER_USD} HKD/USD）")

    if src == "App 自選股":
        symbols = []
        seen: set[str] = set()
        for s in session_wl:
            n = normalize_symbol(s)
            if n and n not in seen:
                seen.add(n)
                symbols.append(n)
    elif src == "預設 DEFAULT_WATCHLIST":
        symbols = list(DEFAULT_WATCHLIST)
    else:
        symbols = _parse_text(text)

    st.caption(f"將掃描 **{len(symbols)}** 隻：{', '.join(symbols[:12])}{'…' if len(symbols) > 12 else ''}")

    run = st.button("開始掃描", type="primary", use_container_width=True)
    if not run:
        st.info("編好 list 後按「開始掃描」。亦可在側欄切換標的後去「投资SOP」睇單隻詳情。")
        return

    if not symbols:
        st.error("清單係空嘅。")
        return

    if save_list and src == "編輯下方文字":
        _save_scan_file(symbols)

    progress = st.progress(0, text="掃描中…")
    rows_all = []
    errors = []
    n = len(symbols)
    for i, sym in enumerate(symbols):
        try:
            sop = build_trade_sop(
                sym,
                period=period,
                interval=interval,
                capital=float(capital),
                risk_pct=float(risk_pct),
            )
            rows_all.append(
                {
                    "代码": sop.symbol,
                    "名称": sop.name,
                    "结论": sop.enter_ok,
                    "适合度": sop.enter_score,
                    "现价": sop.last_price,
                    "入场低": sop.entry_low,
                    "入场高": sop.entry_high,
                    "挂单价": sop.entry_plan,
                    "止损": sop.stop_loss,
                    "T1": sop.target_t1,
                    "T2": sop.target_t2,
                    "R:R": sop.rr_t1,
                    "胜率%": sop.win_rate_pct,
                    "胜率档": sop.win_rate_label,
                    "稳定度": sop.stability_score,
                    "稳定档": sop.stability_label,
                    "多空": sop.bias,
                    "多空分": sop.bias_score,
                    "机会": sop.opportunity,
                    "建议股数": sop.position_shares,
                    "立刻动作": "；".join(sop.actions_now[:2]) if sop.actions_now else "",
                    "失效": sop.invalidation,
                }
            )
        except Exception as exc:
            errors.append(f"{sym}: {type(exc).__name__}: {exc}")
        progress.progress((i + 1) / n, text=f"掃描 {sym}… ({i+1}/{n})")
    progress.empty()

    if errors:
        with st.expander(f"錯誤 {len(errors)}", expanded=False):
            for e in errors:
                st.text(e)

    if not rows_all:
        st.warning("無結果。")
        return

    df = pd.DataFrame(rows_all)

    # filter
    if min_level == "仅适合入场":
        show = df[df["结论"] == "适合入场"]
    elif min_level == "适合+谨慎":
        show = df[df["结论"].isin(["适合入场", "谨慎试仓"])]
    else:
        show = df

    # sort: 适合 first, then score
    order_map = {"适合入场": 0, "谨慎试仓": 1, "观望": 2, "回避": 3}
    df["_ord"] = df["结论"].map(lambda x: order_map.get(x, 9))
    df = df.sort_values(["_ord", "适合度"], ascending=[True, False]).drop(columns=["_ord"])
    show = show.copy()
    if not show.empty:
        show["_ord"] = show["结论"].map(lambda x: order_map.get(x, 9))
        show = show.sort_values(["_ord", "适合度"], ascending=[True, False]).drop(columns=["_ord"])

    # summary metrics
    n_suit = int((df["结论"] == "适合入场").sum())
    n_caut = int((df["结论"] == "谨慎试仓").sum())
    n_wait = int((df["结论"] == "观望").sum())
    n_avoid = int((df["结论"] == "回避").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("适合入场", n_suit)
    m2.metric("谨慎试仓", n_caut)
    m3.metric("观望", n_wait)
    m4.metric("回避", n_avoid)

    st.markdown("### 可入場 / 掃描結果")
    if show.empty:
        st.warning("以目前過濾條件，沒有可顯示標的。可改「顯示級別」為「全部」。")
    else:
        # highlight table
        display_cols = [
            "代码",
            "名称",
            "结论",
            "适合度",
            "现价",
            "入场低",
            "入场高",
            "止损",
            "T1",
            "T2",
            "R:R",
            "胜率%",
            "稳定度",
            "多空",
            "建议股数",
        ]
        st.dataframe(
            show[[c for c in display_cols if c in show.columns]],
            use_container_width=True,
            hide_index=True,
            height=min(480, 48 + 36 * len(show)),
        )

    # detail cards for enterable
    enterable = df[df["结论"].isin(["适合入场", "谨慎试仓"])]
    if not enterable.empty:
        st.markdown("### 詳情卡（适合 / 谨慎）")
        for _, row in enterable.iterrows():
            title = f"{row['代码']} · {row['结论']}（{row['适合度']:.0f}）"
            if row["结论"] == "适合入场":
                box = st.success
            else:
                box = st.warning
            with st.container(border=True):
                box(f"**{title}** · {row['名称']}")
                c1, c2, c3 = st.columns(3)
                c1.markdown(
                    f"現價 **{row['现价']}**  \n"
                    f"入場 **{row['入场低']}–{row['入场高']}**  \n"
                    f"掛單 ≈ **{row['挂单价']}**"
                )
                c2.markdown(
                    f"止損 **{row['止损']}**  \n"
                    f"T1 **{row['T1']}** · T2 **{row['T2']}**  \n"
                    f"R:R **{row['R:R']}**"
                )
                c3.markdown(
                    f"勝率 **{row['胜率%']}%**（{row['胜率档']}）  \n"
                    f"穩定 **{row['稳定度']}**（{row['稳定档']}）  \n"
                    f"股數 **{row['建议股数']}**"
                )
                if row.get("立刻动作"):
                    st.caption(f"立刻：{row['立刻动作']}")
                if row.get("失效"):
                    st.caption(f"失效：{row['失效']}")
                if st.button(f"開投资SOP詳情 · {row['代码']}", key=f"goto_{row['代码']}"):
                    st.session_state.symbol = row["代码"]
                    st.info(f"已選 `{row['代码']}` → 側欄切去「投资SOP」睇完整卡。")

    with st.expander("完整結果表（含回避/观望）", expanded=False):
        st.dataframe(df.drop(columns=["立刻动作", "失效"], errors="ignore"), use_container_width=True, hide_index=True)

    st.caption("數據 Yahoo 可能延遲 · 非投資建議 · 下單前請再確認報價與風控")
