"""Reusable trade journal UI — persistent samples for real win-rate."""
from __future__ import annotations

from typing import Any

import streamlit as st

from trade_journal import (
    add_trade,
    close_trade,
    export_csv_text,
    export_json_text,
    journal_path_info,
    journal_stats,
    load_trades,
    merge_trades,
    parse_import_bytes,
    resolve_journal_path,
)


def render_journal_panel(
    *,
    key_prefix: str = "jr",
    default_symbol: str = "",
    default_name: str = "",
    default_horizon: str = "0–2周",
    default_entry: float | None = None,
    default_stop: float | None = None,
    default_target: float | None = None,
    default_shares: int = 0,
    default_wr: float | None = None,
    default_rr: float | None = None,
    default_verdict: str = "",
    default_mode: str = "",
    default_mode_label: str = "",
    default_exit_px: float | None = None,
    expanded: bool = True,
) -> None:
    """Full journal: stats, write, close with price, export/import for permanent samples."""
    with st.expander("📒 交易日志（永久样本 · 实盘对照）", expanded=expanded):
        info = journal_path_info()
        stats = journal_stats()
        st.caption(
            f"样本永久保存在本机文件（平仓后不删）。路径：`{info['path']}` · "
            f"共 **{stats.get('samples') or 0}** 笔 · "
            f"每日备份 `data/journal_backups/`"
        )
        st.caption(
            "⚠️ Streamlit Cloud 重部署可能清空磁盘 → 请定期 **导出 JSON** 备份；"
            "本地用 `TRADE_JOURNAL_PATH` 可指定固定路径。"
        )

        j1, j2, j3, j4, j5 = st.columns(5)
        j1.metric("样本总数", stats.get("samples") or 0)
        j2.metric("已平仓", stats.get("closed") or 0)
        j3.metric(
            "实盘胜率",
            f"{stats['win_rate']:.0f}%" if stats.get("win_rate") is not None else "—",
        )
        j4.metric(
            "平均R",
            f"{stats['avg_r']:+.2f}" if stats.get("avg_r") is not None else "—",
        )
        j5.metric("持仓中", stats.get("open") or 0)

        # ---- write ----
        st.markdown("##### 写入一笔（开仓）")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ent = st.number_input(
                "成交价",
                min_value=0.0,
                value=float(default_entry or 0.0),
                step=0.01,
                format="%.4f",
                key=f"{key_prefix}_entry",
            )
        with c2:
            stp = st.number_input(
                "止蚀",
                min_value=0.0,
                value=float(default_stop or 0.0),
                step=0.01,
                format="%.4f",
                key=f"{key_prefix}_stop",
            )
        with c3:
            tgt = st.number_input(
                "目标",
                min_value=0.0,
                value=float(default_target or 0.0),
                step=0.01,
                format="%.4f",
                key=f"{key_prefix}_tgt",
            )
        with c4:
            sh = st.number_input(
                "股数",
                min_value=0,
                value=int(default_shares or 0),
                step=1,
                key=f"{key_prefix}_sh",
            )
        if st.button("写入日志（永久样本）", type="primary", key=f"{key_prefix}_add"):
            if ent <= 0 or stp <= 0:
                st.error("请填写有效成交价与止蚀")
            else:
                add_trade(
                    symbol=default_symbol or "UNKNOWN",
                    name=default_name or default_symbol,
                    horizon=default_horizon,
                    entry=float(ent),
                    stop=float(stp) if stp > 0 else None,
                    target=float(tgt) if tgt > 0 else None,
                    shares=int(sh),
                    model_wr=default_wr,
                    model_rr=default_rr,
                    model_verdict=default_verdict,
                    mode=default_mode,
                    mode_label=default_mode_label,
                    notes=f"sample=1 · 模型={default_verdict}",
                )
                st.success("已写入并备份（计入样本）")
                st.rerun()

        # ---- open positions close ----
        trades = load_trades()
        open_t = [t for t in trades if t.get("status") == "open"]
        if open_t:
            st.markdown("##### 持仓中 → 平仓（填真实出场价）")
            for t in open_t[:12]:
                tid = t.get("id")
                cols = st.columns([2.2, 1.2, 1.2, 1])
                with cols[0]:
                    st.caption(
                        f"#{tid} **{t.get('symbol')}** "
                        f"[{t.get('mode_label') or t.get('mode') or '?'}] "
                        f"入{t.get('entry')} 止{t.get('stop')} 标{t.get('target')} "
                        f"股{t.get('shares')}"
                    )
                with cols[1]:
                    ex = st.number_input(
                        "出场价",
                        min_value=0.0,
                        value=float(
                            default_exit_px
                            or t.get("entry")
                            or 0.0
                        ),
                        step=0.01,
                        format="%.4f",
                        key=f"{key_prefix}_ex_{tid}",
                    )
                with cols[2]:
                    reason = st.selectbox(
                        "原因",
                        ["manual", "t1", "t2", "stop", "time"],
                        key=f"{key_prefix}_rs_{tid}",
                    )
                with cols[3]:
                    if st.button("平仓", key=f"{key_prefix}_cl_{tid}"):
                        if ex <= 0:
                            st.error("出场价无效")
                        else:
                            close_trade(str(tid), exit_price=float(ex), exit_reason=reason)
                            st.success("已平仓，样本已保留")
                            st.rerun()

        # ---- export / import ----
        st.markdown("##### 导出 / 导入（保证样本不丢）")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "导出 JSON（推荐备份）",
                data=export_json_text(),
                file_name=f"trade_journal_{len(trades)}samples.json",
                mime="application/json",
                key=f"{key_prefix}_dl_json",
            )
        with e2:
            st.download_button(
                "导出 CSV",
                data=export_csv_text(),
                file_name=f"trade_journal_{len(trades)}samples.csv",
                mime="text/csv",
                key=f"{key_prefix}_dl_csv",
            )
        with e3:
            up = st.file_uploader(
                "导入合并 JSON/CSV",
                type=["json", "csv"],
                key=f"{key_prefix}_up",
                help="按 id 合并；已平仓样本不会被打开仓覆盖删除",
            )
            if up is not None and st.button("执行合并导入", key=f"{key_prefix}_merge"):
                try:
                    rows = parse_import_bytes(up.getvalue())
                    res = merge_trades(rows)
                    st.success(
                        f"合并完成：新增 {res['added']} · 更新 {res['updated']} · "
                        f"跳过 {res['skipped']} · 现总样本 {res['total']}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"导入失败：{exc}")

        closed = [t for t in trades if t.get("status") == "closed"][:15]
        if closed:
            st.markdown("##### 最近已平仓样本")
            for t in closed:
                rr = t.get("result_r")
                st.caption(
                    f"#{t.get('id')} {t.get('symbol')} "
                    f"{t.get('opened')}→{t.get('exit_date')} "
                    f"R={rr if rr is not None else '—'} "
                    f"pnl={t.get('pnl_usd')} "
                    f"[{t.get('exit_reason')}]"
                )
