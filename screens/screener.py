"""今日候選畫面。"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from models.predict import model_available, score_features
from models.train import train_classifier
from screens.common import disclaimer, sidebar_controls


def apply_scenario_filter(
    latest: pd.DataFrame,
    scenario: str,
    min_ret: float,
    min_vol: float,
    above_ma: bool,
    rsi_range: tuple[int, int],
) -> pd.DataFrame:
    """Return candidates for the selected, mutually exclusive research setup."""
    if scenario == "A 追勢":
        return latest[
            latest["rsi_14"].between(46, 68)
            & (latest["ma20_dist"] >= 0)
            & (latest["ret_5"] >= 0.02)
            & (latest["vol_ratio_20"] >= 1.3)
        ]
    if scenario == "B 抄跌":
        return latest[
            latest["rsi_14"].between(28, 44)
            & (latest["ma20_dist"] < 0)
            & (latest["ret_5"] < 0)
            & latest["vol_ratio_20"].between(0.80, 1.10)
        ]
    rules = latest[
        (latest["ret_5"] >= min_ret)
        & (latest["vol_ratio_20"] >= min_vol)
        & latest["rsi_14"].between(*rsi_range)
    ]
    return rules[rules["ma20_dist"] >= 0] if above_ma else rules


def render_screener() -> None:
    st.title("今日候選")
    disclaimer()
    universe, _, features, failed = sidebar_controls()
    with st.sidebar:
        scenario = st.selectbox(
            "篩選情境",
            ["自訂條件", "A 追勢", "B 抄跌"],
            key="workbench_scenario",
        )
        min_ret = 0.0
        min_vol = 1.0
        above_ma = False
        rsi_range = (35, 75)
        if scenario == "A 追勢":
            st.caption("RSI 46–68 · 20 日線之上 · 5 日 ≥ 2% · 量比 ≥ 1.30")
        elif scenario == "B 抄跌":
            st.caption("RSI 28–44 · 20 日線之下 · 5 日 < 0% · 量比 0.80–1.10")
        else:
            min_ret = st.slider("近 5 日升幅下限", -20.0, 20.0, 0.0, format="%.1f%%", key="workbench_min_ret") / 100
            min_vol = st.slider("成交量比 20 日均量下限", 0.0, 5.0, 1.0, 0.1, key="workbench_min_vol")
            above_ma = st.checkbox("必須在 20 日線上", key="workbench_above_ma")
            rsi_range = st.slider("RSI 區間", 0, 100, (35, 75), key="workbench_rsi")
        retrain = st.button("重新訓練模型", key="workbench_retrain")
    if retrain:
        try:
            with st.spinner("正在按時間切分訓練 LightGBM..."):
                st.session_state.workbench_metrics = train_classifier(features)
            st.success("模型已儲存至 models/。")
        except (ImportError, ValueError) as exc:
            st.warning(str(exc))
    if failed:
        st.warning("下載失敗：" + ", ".join(failed))
    scored = score_features(features) if model_available() else pd.DataFrame()
    latest = features.sort_values("Date").groupby("Ticker", as_index=False).tail(1).copy()
    if not scored.empty:
        latest = latest.merge(scored[["Ticker", "score", "signal"]], on="Ticker", how="left")
    else:
        latest["score"] = pd.NA
        latest["signal"] = "未訓練"
    latest["名稱"] = latest["Ticker"].map(universe)
    rules = apply_scenario_filter(latest, scenario, min_ret, min_vol, above_ma, rsi_range)
    top = scored.iloc[0]["Ticker"] if not scored.empty else "尚未訓練"
    a, b, c = st.columns(3)
    a.metric("股票池數量", len(universe)); b.metric("成功下載數量", latest["Ticker"].nunique()); c.metric("模型分數最高", top)
    st.subheader("今日股票清單")
    st.caption(f"情境：{scenario} · 成功下載 {latest['Ticker'].nunique()} 隻；目前規則條件符合 {len(rules)} 隻。以下條件只作研究篩選，與模型分數分開呈現。")
    view_mode = st.radio("顯示範圍", ["全部股票", "只看符合規則"], horizontal=True, key="workbench_view_mode")
    shown = latest if view_mode == "全部股票" else rules
    columns = ["Ticker", "名稱", "Close", "ret_1", "ret_5", "ret_20", "vol_ratio_20", "rsi_14", "score", "signal"]
    table = shown.sort_values("score", ascending=False, na_position="last").reindex(columns=columns).rename(columns={"Ticker": "代碼", "Close": "最新收市", "ret_1": "1日%", "ret_5": "5日%", "ret_20": "20日%", "vol_ratio_20": "量比", "rsi_14": "RSI", "score": "LightGBM 分數", "signal": "訊號"})
    for column in ("1日%", "5日%", "20日%"):
        table[column] = table[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "-")
    table["最新收市"] = table["最新收市"].map(lambda value: f"{value:.2f}")
    table["LightGBM 分數"] = table["LightGBM 分數"].map(lambda value: f"{value:.3f}" if pd.notna(value) else "未訓練")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("下載候選 CSV", table.to_csv(index=False).encode("utf-8-sig"), "今日候選.csv", "text/csv")
    if not shown.empty:
        ticker = st.selectbox("查看迷你走勢", shown["Ticker"].tolist(), key="workbench_preview")
        chart_data = features[features["Ticker"] == ticker].tail(90)
        st.plotly_chart(px.line(chart_data, x="Date", y="Close", title=f"{ticker} 近 90 日收市"), width="stretch")
    if scored.empty:
        st.info("尚未有可用模型。可於側欄按「重新訓練模型」建立本機模型。")