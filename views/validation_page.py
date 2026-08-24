"""Streamlit page for out-of-sample inspection of the live rule score."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stock_service import fetch_history
from strategy_validation import build_validation_samples, summarize_score_bins


def _fmt_metric(value: float | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.2f}{suffix}"


def render_strategy_validation(symbol: str) -> None:
    st.header(f"策略验证 · `{symbol}`")
    st.caption("逐日 walk-forward · 当日规则分只对照之后的 K 线 · 不把规则分当概率")
    st.info(
        "这里测试目前 **Bias 规则分** 是否和后续走势有关。分数计算只使用当日及以前数据；"
        "目标、止损、到期未触和同 K 歧义分别统计。结果是历史描述，不代表未来保证。"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        history_period = st.selectbox("历史范围", ["2y", "5y", "max"], index=1)
    with c2:
        horizon = st.selectbox("观察期（交易日）", [5, 10, 20, 30], index=1)
    with c3:
        target_atr = st.number_input("目标 ATR", 0.5, 5.0, 1.5, 0.25)
    with c4:
        stop_atr = st.number_input("止损 ATR", 0.5, 5.0, 1.0, 0.25)

    benchmark_symbol = st.selectbox(
        "比较基准",
        ["SPY", "QQQ"],
        index=1 if symbol.upper() == "SPY" else 0,
        help="超额回报 = 标的未来回报 - 同期基准回报",
    )

    with st.spinner("重放历史规则分并计算未来结果..."):
        history = fetch_history(symbol, period=history_period, interval="1d")
        benchmark = fetch_history(benchmark_symbol, period=history_period, interval="1d")
        samples = build_validation_samples(
            history,
            horizon=int(horizon),
            benchmark=benchmark,
            target_atr=float(target_atr),
            stop_atr=float(stop_atr),
        )

    if samples.empty:
        st.warning("数据不足：至少需要约 60 根日 K，再加所选观察期。")
        return

    summary = summarize_score_bins(samples)
    signaled = samples[samples["direction"] != "中性"].copy()
    correlation = samples["score"].corr(samples["forward_return_pct"], method="spearman")
    directional_mean = signaled["directional_return_pct"].mean() if len(signaled) else None
    directional_positive = (
        (signaled["directional_return_pct"] > 0).mean() * 100.0 if len(signaled) else None
    )
    average_excess = samples["excess_return_pct"].mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("总样本", f"{len(samples)}")
    m2.metric("方向信号", f"{len(signaled)}")
    m3.metric("分数/未来相关", _fmt_metric(correlation))
    m4.metric("方向后平均回报", _fmt_metric(directional_mean, "%"))
    m5.metric("方向正确比例", _fmt_metric(directional_positive, "%"))
    st.caption(
        f"全样本平均超额回报：{_fmt_metric(average_excess, '%')} · "
        "相关系数接近 0 表示分数与未来回报缺少单调关系。"
    )

    if len(signaled) < 100:
        st.warning(f"方向信号只有 {len(signaled)} 个，样本偏少；不要据此调整权重或门槛。")

    st.subheader("分数分箱表现")
    display = summary.copy()
    numeric_columns = [column for column in display.columns if column not in ("分数区间", "样本")]
    for column in numeric_columns:
        display[column] = display[column].map(
            lambda value: "—" if pd.isna(value) else f"{float(value):.1f}%"
        )
    st.dataframe(display, width="stretch", hide_index=True)

    if not summary.empty:
        chart = summary.set_index("分数区间")
        st.subheader("未来回报是否随分数改善")
        st.bar_chart(chart[["平均未来回报%", "方向后平均回报%"]])
        st.caption(
            "理想状态是分数由看空到看多时，未来回报大致递增；若没有单调关系，"
            "应减少重复指标或重新校准，而不是增加更多指标。"
        )

        st.subheader("目标 / 止损路径")
        st.bar_chart(chart[["目标先触%", "止损先触%", "到期未触%", "歧义%"]])

    with st.expander("逐日原始样本", expanded=False):
        raw = samples.copy()
        raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(raw.iloc[::-1], width="stretch", hide_index=True)

    st.caption(
        "限制：使用日 K，未计交易成本、隔夜跳空及成交困难；此页先验证方向分是否有信息量，"
        "不是完整资金曲线回测。"
    )