"""Watchlist SOP scan page — one screen to find enterable names."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from stock_service import DEFAULT_WATCHLIST, normalize_symbol
from trade_journal import journal_stats
from trade_sop import build_trade_sop, format_win_rate

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
    st.markdown("## Watchlist 掃描（短线 0–2周 / 2–4周）")
    st.caption(
        f"極簡預設：只看 **結論 + 三灯 + 一句話 + 掛單/止蝕** · {period_label} · "
        "與投資SOP同源 · 非投資建議"
    )
    scan_simple = st.toggle(
        "掃描極簡表（推荐）",
        value=bool(st.session_state.get("scan_simple_mode", True)),
        key="scan_simple_mode",
        help="开启只显示决策列；关闭显示更多栏位",
    )

    # ---- list source ----
    file_syms = _load_scan_file()
    session_wl = list(st.session_state.get("watchlist") or DEFAULT_WATCHLIST)

    default_text = "\n".join(file_syms) if file_syms else "\n".join(session_wl[:20])

    # 清单默认用档案/自选（避免一开页就一大坨编辑区）
    with st.expander("編輯掃描清單 / 参数", expanded=False):
        src = st.radio(
            "清單來源",
            ["編輯下方文字", "App 自選股", "預設 DEFAULT_WATCHLIST"],
            horizontal=True,
            index=0,
            key="scan_list_src",
        )
        text = st.text_area(
            "股票代碼（一行一個，或逗號分隔）",
            value=default_text,
            height=140,
            help="例：AAPL / NVDA / SPY",
            key="scan_list_text",
        )
        HKD_PER_USD = 7.8
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            capital_hkd = st.number_input(
                "本金 HKD",
                min_value=1000.0,
                value=50000.0,
                step=1000.0,
                key="scan_capital_hkd",
            )
        with c2:
            risk_pct = st.number_input(
                "1R %", min_value=0.25, max_value=5.0, value=1.0, step=0.25, key="scan_risk"
            )
        with c3:
            horizon_ui = st.selectbox(
                "主周期",
                ["0–2周", "2–4周"],
                index=0,
                key="scan_horizon",
            )
        with c4:
            mode_ui = st.selectbox(
                "模式",
                ["A 防守版", "B 进攻版"],
                index=0
                if st.session_state.get("sop_mode", "defensive") != "aggressive"
                else 1,
                key="scan_mode_ui",
            )
        with c5:
            min_level = st.selectbox(
                "顯示級別",
                ["适合+谨慎", "仅适合入场", "全部"],
                index=0,
                key="scan_min_level",
            )
        save_list = st.checkbox("寫入 scan 檔", value=True, key="scan_save_list")

    # 未打开 expander 时用 session / 默认，避免 NameError
    src = st.session_state.get("scan_list_src", "編輯下方文字")
    text = st.session_state.get("scan_list_text", default_text)
    capital_hkd = float(st.session_state.get("scan_capital_hkd", 50000.0))
    risk_pct = float(st.session_state.get("scan_risk", 1.0))
    horizon_ui = st.session_state.get("scan_horizon", "0–2周")
    mode_ui = st.session_state.get("scan_mode_ui", "A 防守版")
    min_level = st.session_state.get("scan_min_level", "适合+谨慎")
    save_list = bool(st.session_state.get("scan_save_list", True))
    HKD_PER_USD = 7.8
    capital = float(capital_hkd) / HKD_PER_USD
    primary_horizon = "h1" if horizon_ui == "0–2周" else "h2"
    mode_key = "aggressive" if str(mode_ui).startswith("B") else "defensive"
    st.session_state["sop_mode"] = mode_key
    st.caption(
        f"主周期 **{horizon_ui}** · 模式 **{mode_ui}** · "
        f"{'極簡表' if scan_simple else '完整表'}"
    )

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

    import hashlib
    import time as _time

    list_key = hashlib.md5(
        (
            ",".join(symbols)
            + f"|{period}|{interval}|{capital:.0f}|{risk_pct}|{primary_horizon}|{mode_key}"
        ).encode()
    ).hexdigest()[:12]
    cache_key = f"scan_cache_{list_key}"
    cache_ttl_sec = 15 * 60  # 15 分钟内同清单结果固定，减少「10分钟三样」

    b1, b2 = st.columns([2, 1])
    with b1:
        run = st.button("開始掃描", type="primary", use_container_width=True)
    with b2:
        force = st.button("強制刷新", use_container_width=True, help="忽略快取，重新拉數")

    cached = st.session_state.get(cache_key)
    cache_age = None
    if cached and isinstance(cached, dict):
        cache_age = _time.time() - float(cached.get("ts", 0))
        if cache_age > cache_ttl_sec:
            cached = None

    if not run and not force:
        if cached and cached.get("rows"):
            st.info(
                f"顯示 **{int(cache_age // 60)} 分鐘前** 的掃描快取（約 {cache_ttl_sec // 60} 分鐘內不變）。"
                "要最新結果請按「強制刷新」。"
            )
            rows_all = cached["rows"]
            errors = cached.get("errors") or []
        else:
            st.info(
                "編好 list 後按「開始掃描」。結果會快取約 15 分鐘，避免盤中每掃一次可開倉名單都不同。"
            )
            return
    else:
        if not symbols:
            st.error("清單係空嘅。")
            return

        if save_list and src == "編輯下方文字":
            _save_scan_file(symbols)

        # 有快取且非強制 → 直接用
        if cached and cached.get("rows") and not force:
            rows_all = cached["rows"]
            errors = cached.get("errors") or []
            st.caption(f"使用快取（{int((cache_age or 0) // 60)} 分鐘前）")
        else:
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
                        primary_horizon=primary_horizon,
                        mode=mode_key,
                        include_h1=False,
                    )
                    if sop.enter_ok in ("适合入场", "谨慎试仓"):
                        sop = build_trade_sop(
                            sym,
                            period=period,
                            interval=interval,
                            capital=float(capital),
                            risk_pct=float(risk_pct),
                            primary_horizon=primary_horizon,
                            mode=mode_key,
                            include_h1=True,
                        )
                    h1 = getattr(sop, "swing_h1", None)
                    h2 = getattr(sop, "swing_h2", None)
                    prim = getattr(sop, "primary_plan", None) or h1
                    exit_pl = getattr(sop, "exit_plan", None)
                    wr_h1 = (
                        getattr(h1, "win_rate_display", None)
                        or format_win_rate(
                            getattr(h1, "win_rate_pct", None) if h1 else None,
                            getattr(h1, "win_rate_samples", None) if h1 else None,
                        )
                    )
                    wr_h2 = (
                        getattr(h2, "win_rate_display", None)
                        or format_win_rate(
                            getattr(h2, "win_rate_pct", None) if h2 else None,
                            getattr(h2, "win_rate_samples", None) if h2 else None,
                        )
                    )
                    def _lg(x: str) -> str:
                        return {"green": "绿", "yellow": "黄", "red": "红"}.get(
                            x or "", "—"
                        )

                    rows_all.append(
                        {
                            "代码": sop.symbol,
                            "名称": sop.name,
                            "模式": getattr(sop, "mode_label", mode_ui),
                            "结论": sop.enter_ok,
                            "主结论": getattr(prim, "verdict", "—") if prim else "—",
                            "位置灯": _lg(getattr(sop, "position_light", "")),
                            "胜率灯": _lg(getattr(sop, "wr_light", "")),
                            "划算灯": _lg(getattr(sop, "rr_light", "")),
                            "一句话": getattr(sop, "one_liner_reason", "") or "",
                            "赚到目标HKD": getattr(sop, "pnl_if_win_hkd", None),
                            "止损亏HKD": getattr(sop, "pnl_if_loss_hkd", None),
                            "财报天": getattr(sop, "earnings_days_left", None),
                            "财报": (
                                "是"
                                if getattr(sop, "earnings_soon", False)
                                else ""
                            ),
                            "阻力": getattr(sop, "nearest_resistance", None),
                            "支撑": getattr(sop, "nearest_support", None),
                            "阻力%": getattr(sop, "resistance_pct", None),
                            "主周期": getattr(prim, "label", horizon_ui) if prim else horizon_ui,
                            "0-2周": getattr(h1, "verdict", "—") if h1 else "—",
                            "2-4周": getattr(h2, "verdict", "—") if h2 else "—",
                            "适合度": sop.enter_score,
                            "现价": sop.last_price,
                            "入場低": sop.entry_low,
                            "入場高": sop.entry_high,
                            "掛單": sop.entry_plan,
                            "止蝕": sop.stop_loss,
                            "目标0-2周": getattr(h1, "target", None) if h1 else sop.target_t1,
                            "目标2-4周": getattr(h2, "target", None) if h2 else sop.target_t2,
                            "胜率0-2周": wr_h1,
                            "胜率2-4周": wr_h2,
                            "胜率0-2周%": getattr(h1, "win_rate_pct", None) if h1 else None,
                            "胜率2-4周%": getattr(h2, "win_rate_pct", None) if h2 else None,
                            "R:R_0-2周": getattr(h1, "rr", None) if h1 else sop.rr_t1,
                            "净R:R": getattr(prim, "rr_net", None) if prim else None,
                            "E[R]_0-2周": getattr(h1, "expectancy_r", None) if h1 else None,
                            "净E[R]": getattr(prim, "expectancy_net", None) if prim else None,
                            "时间止损日": getattr(exit_pl, "max_hold_days", None)
                            if exit_pl
                            else None,
                            # legacy keys for rest of page
                            "入场低": sop.entry_low,
                            "入场高": sop.entry_high,
                            "挂单价": sop.entry_plan,
                            "止损": sop.stop_loss,
                            "T1": sop.target_t1,
                            "T2": sop.target_t2,
                            "R:R": getattr(h1, "rr", None) if h1 else sop.rr_t1,
                            "期望E[R]": getattr(h1, "expectancy_r", None) if h1 else getattr(sop, "expectancy_r", None),
                            "胜率%": getattr(h1, "win_rate_pct", None) if h1 else sop.win_rate_pct,
                            "胜率档": sop.win_rate_label,
                            "稳定度": sop.stability_score,
                            "稳定档": sop.stability_label,
                            "多空": sop.bias,
                            "多空分": sop.bias_score,
                            "机会": sop.opportunity,
                            "板块RS": getattr(sop, "sector_rs_label", "—") or "—",
                            "板块分": getattr(sop, "sector_rs_score", None),
                            "量能": getattr(sop, "volume_confirm_label", "—") or "—",
                            "量能分": getattr(sop, "volume_confirm_score", None),
                            "IV": getattr(sop, "iv_label", "—") or "—",
                            "IV分": getattr(sop, "iv_score", None),
                            "IV事件": "是" if getattr(sop, "iv_high_event", False) else "",
                            "假突破": getattr(sop, "false_break_label", "—") or "—",
                            "假突破风险": "是" if getattr(sop, "false_break_risk", False) else "",
                            "跟势": getattr(sop, "trend_align_label", "—") or "—",
                            "跟势分": getattr(sop, "trend_align_score", None),
                            "逆势": "是" if getattr(sop, "against_trend", False) else "",
                            "周线": getattr(sop, "weekly_label", "—") or "—",
                            "1H": getattr(sop, "h1_label", "—") or "—",
                            "1H可掛": "是" if getattr(sop, "h1_ready", False) else "",
                            "ADX": getattr(sop, "adx_label", "—") or "—",
                            "ADX值": getattr(sop, "adx_value", None),
                            "市场环境": getattr(sop, "regime_label", "—") or "—",
                            "建议股数": sop.position_shares,
                            "立刻动作": "；".join(sop.actions_now[:2]) if sop.actions_now else "",
                            "失效": sop.invalidation,
                            "板块说明": getattr(sop, "sector_rs_summary", "") or "",
                            "量能说明": getattr(sop, "volume_confirm_summary", "") or "",
                            "IV说明": getattr(sop, "iv_summary", "") or "",
                            "走势": getattr(sop, "trend_note", "") or "",
                        }
                    )
                except Exception as exc:
                    errors.append(f"{sym}: {type(exc).__name__}: {exc}")
                progress.progress((i + 1) / n, text=f"掃描 {sym}… ({i+1}/{n})")
            progress.empty()
            st.session_state[cache_key] = {
                "ts": _time.time(),
                "rows": rows_all,
                "errors": errors,
            }

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
    n_all = len(df)
    n_suit = int((df["结论"] == "适合入场").sum())
    n_caut = int((df["结论"] == "谨慎试仓").sum())
    n_wait = int((df["结论"] == "观望").sum())
    n_avoid = int((df["结论"] == "回避").sum())
    n_enter = n_suit + n_caut
    wr_all = pd.to_numeric(df["胜率%"], errors="coerce").dropna()
    wr_enter = pd.to_numeric(
        df.loc[df["结论"].isin(["适合入场", "谨慎试仓"]), "胜率%"], errors="coerce"
    ).dropna()
    exp_enter = pd.to_numeric(
        df.loc[df["结论"].isin(["适合入场", "谨慎试仓"]), "期望E[R]"], errors="coerce"
    ).dropna()
    n_iv_event = int((df["IV事件"] == "是").sum()) if "IV事件" in df.columns else 0
    n_vol_dump = int((df["量能"] == "放量下跌").sum()) if "量能" in df.columns else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("适合入场", n_suit)
    m2.metric("谨慎试仓", n_caut)
    m3.metric(
        "入場率",
        f"{100 * n_enter / n_all:.0f}%" if n_all else "—",
        f"{n_enter}/{n_all}",
    )
    m4.metric(
        "平均勝率(全)",
        f"{wr_all.mean():.0f}%" if len(wr_all) else "—",
    )
    m5.metric(
        "平均勝率(可開)",
        f"{wr_enter.mean():.0f}%" if len(wr_enter) else "—",
    )
    m6.metric(
        "平均E[R](可開)",
        f"{exp_enter.mean():+.2f}" if len(exp_enter) else "—",
    )
    st.caption(
        f"观望 {n_wait} · 回避 {n_avoid} · "
        f"IV事件風險標誌 {n_iv_event} 隻 · 放量下跌 {n_vol_dump} 隻 "
        f"（後兩者會壓低入場評級）"
    )
    realized = journal_stats()
    if realized.get("closed"):
        pf = realized.get("profit_factor")
        gap = realized.get("calibration_gap")
        pf_text = f"{pf:.2f}" if pf is not None else "—"
        st.caption(
            f"实盘对照（{realized['closed']} 笔已平仓）：胜率 "
            f"{realized['win_rate']:.1f}% · 平均 {realized['avg_r']:+.2f}R · "
            f"Profit Factor {pf_text}"
        )
        if gap is not None:
            st.caption(
                f"模型校准差 {gap:+.1f}%（预测减真实，样本 "
                f"{realized['calibration_samples']}）；接近 0 较理想。"
            )

    st.markdown("### 可入場 / 掃描結果")
    if show.empty:
        st.warning("以目前過濾條件，沒有可顯示標的。可改「顯示級別」為「全部」。")
    else:
        if scan_simple:
            display_cols = [
                "代码",
                "主结论",
                "位置灯",
                "胜率灯",
                "划算灯",
                "一句话",
                "阻力",
                "支撑",
                "财报天",
                "现价",
                "掛單",
                "止蝕",
                "目标0-2周",
            ]
        else:
            display_cols = [
                "代码",
                "名称",
                "主结论",
                "结论",
                "位置灯",
                "胜率灯",
                "划算灯",
                "一句话",
                "阻力",
                "支撑",
                "阻力%",
                "财报",
                "财报天",
                "现价",
                "掛單",
                "止蝕",
                "目标0-2周",
                "胜率0-2周",
                "净R:R",
                "赚到目标HKD",
                "止损亏HKD",
                "周线",
                "1H",
            ]
        st.dataframe(
            show[[c for c in display_cols if c in show.columns]],
            use_container_width=True,
            hide_index=True,
            height=min(520, 48 + 36 * len(show)),
        )

    # detail cards for enterable
    enterable = df[df["结论"].isin(["适合入场", "谨慎试仓"])]
    if not enterable.empty:
        st.markdown("### 詳情卡（适合 / 谨慎）")
        for _, row in enterable.iterrows():
            title = f"{row['代码']} · {row.get('主结论', row['结论'])}（{row['适合度']:.0f}）"
            if row["结论"] == "适合入场":
                box = st.success
            else:
                box = st.warning
            with st.container(border=True):
                box(f"**{title}** · {row['名称']}")
                lights = (
                    f"位置 **{row.get('位置灯', '—')}** · "
                    f"胜率 **{row.get('胜率灯', '—')}** · "
                    f"划算 **{row.get('划算灯', '—')}**"
                )
                st.caption(lights)
                if row.get("一句话"):
                    st.markdown(f"**主因：** {row['一句话']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(
                    f"現價 **{row['现价']}**  \n"
                    f"入場 **{row.get('入場低', row.get('入场低'))}–{row.get('入場高', row.get('入场高'))}**  \n"
                    f"掛單 ≈ **{row.get('掛單', row.get('挂单价'))}**"
                )
                c2.markdown(
                    f"止蝕 **{row.get('止蝕', row.get('止损'))}**  \n"
                    f"目标0–2周 **{row.get('目标0-2周', row.get('T1'))}**  \n"
                    f"目标2–4周 **{row.get('目标2-4周', row.get('T2'))}**"
                )
                wr_show = row.get("胜率0-2周") or row.get("胜率0-2周%") or row.get("胜率%")
                c3.markdown(
                    f"0–2周 **{row.get('0-2周', row.get('结论'))}**  \n"
                    f"勝率 **{wr_show}**  \n"
                    f"净R:R **{row.get('净R:R', row.get('R:R'))}**"
                )
                win_h = row.get("赚到目标HKD")
                loss_h = row.get("止损亏HKD")
                c4.markdown(
                    f"賺到目標 ≈ **{win_h if win_h is not None else '—'} HKD**  \n"
                    f"止損約虧 ≈ **{loss_h if loss_h is not None else '—'} HKD**  \n"
                    f"（按每筆 5000 HKD）  \n"
                    f"股數 **{row['建议股数']}**"
                )
                if row.get("立刻动作"):
                    st.caption(f"立刻：{row['立刻动作']}")
                if row.get("失效"):
                    st.caption(f"失效：{row['失效']}")
                if st.button(f"開投资SOP詳情 · {row['代码']}", key=f"goto_{row['代码']}"):
                    # 下一轮 app 开头处理：换代码 + 自动打开投资SOP（勿在此改 nav_page，widget 已创建）
                    sym = str(row["代码"])
                    st.session_state.symbol = sym
                    st.session_state._pending_symbol = sym
                    st.session_state._goto_sop = True
                    st.rerun()

    with st.expander("完整結果表（含回避/观望 · 含板塊/量能/IV）", expanded=False):
        drop_cols = ["立刻动作", "失效", "板块说明", "量能说明", "IV说明"]
        st.dataframe(
            df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "與投資SOP同一套 **三灯裁决**（位置 · 胜率 · 划算）· 每筆按 5000 HKD 估算賺蝕空間 · "
        "數據 Yahoo 可能延遲 · 非投資建議 · 改規則後請按「強制刷新」"
    )
