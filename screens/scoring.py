"""模型打分畫面。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import FEATURE_FILE, MODEL_FILE
from models.predict import load_metrics, model_available, score_features
from screens.common import disclaimer, sidebar_controls


def render_scoring() -> None:
    st.title("模型打分")
    disclaimer()
    universe, _, features, failed = sidebar_controls()
    if failed:
        st.warning("下載失敗：" + ", ".join(failed))
    if not model_available():
        st.info("尚未找到模型檔。請到「今日候選」按重新訓練模型。")
        return
    metrics = load_metrics()
    a, b, c = st.columns(3)
    a.metric("訓練時間", metrics.get("trained_at", "-")); b.metric("驗證 AUC", f"{metrics.get('auc', 0):.3f}"); c.metric("樣本期", metrics.get("date_range", "-"))
    if metrics.get("auc", 0) < .52:
        st.warning("模型區分度很弱，請當玩具，不要當交易訊號。")
    scores = score_features(features)
    if scores.empty:
        st.warning("目前資料未有足夠完整特徵可打分。")
        return
    scores["名稱"] = scores.Ticker.map(universe)
    st.plotly_chart(px.histogram(scores, x="score", nbins=12, title="全市場分數分布"), width="stretch")
    import joblib
    feature_names = json.loads(Path(FEATURE_FILE).read_text(encoding="utf-8"))
    importance = pd.DataFrame({"特徵": feature_names, "gain": joblib.load(MODEL_FILE).feature_importances_}).sort_values("gain", ascending=False)
    st.plotly_chart(px.bar(importance.head(15), x="gain", y="特徵", orientation="h", title="特徵重要性（LightGBM gain）"), width="stretch")
    top_n = st.slider("只看 Top N", 1, len(scores), min(10, len(scores)), key="workbench_top_n")
    view = scores.head(top_n)[["Ticker", "名稱", "Close", "ret_5", "rsi_14", "vol_ratio_20", "ma20_dist", "score", "signal"]].copy()
    for column in ("ret_5", "ma20_dist"):
        view[column] = view[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "-")
    view["score"] = view.score.map(lambda value: f"{value:.3f}")
    st.dataframe(view.rename(columns={"Ticker": "代碼", "Close": "最新收市", "ret_5": "已實現 5日%", "rsi_14": "RSI", "vol_ratio_20": "量比", "ma20_dist": "20日線距離", "score": "分數", "signal": "訊號"}), width="stretch", hide_index=True)
    st.markdown("模型以歷史日線特徵估計未來 5 個交易日錄得正回報的機率。它以時間切分驗證，並不會看見訓練日之後的資料。分數較高只代表歷史條件較接近，並不保證明天會升。模型會過時，資料品質、成本和突發事件都可能令結果失效。")