"""Streamlit page: 回测复盘（实盘辅助 MTM 引擎）."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def render_backtest() -> None:
    st.header("回测复盘 · 信用垂直价差")
    st.caption("引擎 bs_mtm_v3 · 单仓不重叠 · BS 盯市 + 佣金/摩擦")
    st.info(
        "**实盘辅助回测（非券商成交回报）**\n\n"
        "- **策略**：Bull Put（上升趋势）/ Bear Call（下降趋势）  \n"
        "- **入场**：均线过滤 + BS 估算权利金 + credit/width 门槛  \n"
        "- **持仓**：逐日 BS **盯市**；50% 最大利润止盈 / 2R 止损  \n"
        "- **成本**：默认佣金 round-trip **$2.6/张** + 摩擦 **$0.03/股**  \n"
        "- **仓位**：单仓不重叠  \n"
        "- **局限**：无真实历史期权链、无指派模型  \n\n"
        "用于验证规则与期望数量级；下单以实时链 + 限价为准。"
    )

    results_dir = Path(__file__).resolve().parent.parent / "backtest" / "results"
    strategies = {
        "Bull Put（偏多）": "bull_put",
        "Bear Call（偏空）": "bear_call",
    }
    strat_label = st.selectbox("策略", list(strategies.keys()), index=0)
    strategy = strategies[strat_label]

    file_map: dict[str, Path] = {}
    for sym in ("SPY", "VOO", "QQQ"):
        p = results_dir / f"{sym.lower()}_{strategy}_trades.csv"
        # fallback legacy bull_put filename
        if not p.exists() and strategy == "bull_put":
            legacy = results_dir / f"{sym.lower()}_bull_put_trades.csv"
            if legacy.exists():
                p = legacy
        if p.exists():
            file_map[sym] = p

    c_comm, c_slip = st.columns(2)
    with c_comm:
        commission_rt = st.number_input(
            "往返佣金 $/张",
            min_value=0.0,
            max_value=20.0,
            value=2.6,
            step=0.1,
            help="约 $0.65/腿 × 开平 × 2 腿",
        )
    with c_slip:
        slip = st.number_input(
            "摩擦 $/股",
            min_value=0.0,
            max_value=0.5,
            value=0.03,
            step=0.01,
        )

    if not file_map:
        st.warning("尚未找到该策略的回测结果文件。")
        st.info(
            f"请点击下方重跑，或命令行：\n\n"
            f"`py -m backtest.run_backtest --symbol SPY --strategy {strategy}`"
        )

    b1, b2 = st.columns(2)
    with b1:
        sel = st.selectbox(
            "标的",
            ["SPY", "VOO", "QQQ"],
            index=0,
        )
        rerun_one = st.button(f"重跑 {sel} · {strategy}", type="primary", use_container_width=True)
    with b2:
        st.write("")
        st.write("")
        rerun_all = st.button("重跑 SPY/VOO/QQQ", use_container_width=True)

    if rerun_one or rerun_all:
        targets = [sel] if rerun_one else ["SPY", "VOO", "QQQ"]
        with st.spinner(f"回测中：{', '.join(targets)} · {strategy}..."):
            try:
                from backtest.run_backtest import run_backtest as run_vertical_backtest

                for t in targets:
                    run_vertical_backtest(
                        symbol=t,
                        strategy=strategy,
                        commission_rt=float(commission_rt),
                        slip_per_share=float(slip),
                    )
                st.success(f"完成：{', '.join(targets)} · {strategy}")
                st.rerun()
            except Exception as exc:
                st.error(f"回测失败：{exc}")

    csv_path = file_map.get(sel) if file_map else None
    if csv_path is None:
        # after rerun, refresh map
        p = results_dir / f"{sel.lower()}_{strategy}_trades.csv"
        if p.exists():
            csv_path = p
        elif strategy == "bull_put":
            legacy = results_dir / f"{sel.lower()}_bull_put_trades.csv"
            csv_path = legacy if legacy.exists() else None

    if csv_path is None or not csv_path.exists():
        return

    try:
        trades = pd.read_csv(csv_path)
    except Exception as exc:
        st.error(f"读取失败：{exc}")
        return

    st.caption(f"文件：`{csv_path.name}`")
    if "engine" in trades.columns and len(trades):
        st.caption(f"引擎：{trades['engine'].iloc[-1]}")
    if trades.empty:
        st.warning("该文件没有交易记录。")
        return

    if "r_multiple" not in trades.columns or "pnl" not in trades.columns:
        st.error("CSV 缺少 r_multiple / pnl 列。")
        st.dataframe(trades, use_container_width=True, hide_index=True)
        return

    if "entry_date" in trades.columns:
        trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
        trades = trades.sort_values("entry_date").reset_index(drop=True)

    r = trades["r_multiple"].astype(float)
    pnl = trades["pnl"].astype(float)
    wins = int((r > 0).sum())
    losses = int((r < 0).sum())
    win_rate = (wins / len(trades) * 100) if len(trades) else 0.0
    avg_r = float(r.mean()) if len(r) else 0.0
    neg_sum = abs(float(r[r < 0].sum()))
    pos_sum = float(r[r > 0].sum())
    profit_factor = (pos_sum / neg_sum) if neg_sum > 0 else np.inf

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("笔数", f"{len(trades)}")
    m2.metric("胜率", f"{win_rate:.1f}%")
    m3.metric("平均 R", f"{avg_r:.3f}")
    m4.metric("期望", f"{avg_r:.3f}R")
    m5.metric("盈亏因子", "inf" if np.isinf(profit_factor) else f"{profit_factor:.2f}")
    st.caption(f"平均 P&L: ${float(pnl.mean()):.2f} · 赢 {wins} / 亏 {losses}")

    if "exit_reason" in trades.columns:
        st.subheader("出场原因分布")
        st.bar_chart(trades["exit_reason"].value_counts())

    eq_df = trades.copy()
    eq_df["cum_pnl"] = pnl.cumsum()
    st.subheader("累积 P&L")
    if "entry_date" in eq_df.columns and eq_df["entry_date"].notna().any():
        st.line_chart(eq_df[["entry_date", "cum_pnl"]].dropna().set_index("entry_date"))
    else:
        st.line_chart(eq_df[["cum_pnl"]])

    st.subheader("R 分布")
    r_vals = r.dropna().to_numpy()
    if len(r_vals) >= 1:
        low = np.floor(r_vals.min() * 4) / 4
        high = np.ceil(r_vals.max() * 4) / 4
        if low == high:
            low -= 0.25
            high += 0.25
        bins = np.arange(low, high + 0.25, 0.25)
        hist, edges = np.histogram(r_vals, bins=bins)
        labels = [f"{edges[i]:.2f}~{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
        st.bar_chart(pd.DataFrame({"R区间": labels, "笔数": hist}).set_index("R区间"))

    show = trades.copy()
    if "entry_date" in show.columns:
        show["entry_date"] = pd.to_datetime(show["entry_date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    st.subheader("交易明细")
    st.dataframe(show.iloc[::-1], use_container_width=True, hide_index=True)
