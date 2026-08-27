"""載入模型並輸出各股票最新完整特徵列分數。"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from config import FEATURE_FILE, METRICS_FILE, MODEL_FILE


def model_available() -> bool:
    return Path(MODEL_FILE).exists() and Path(FEATURE_FILE).exists()


def load_metrics() -> dict:
    return json.loads(Path(METRICS_FILE).read_text(encoding="utf-8")) if Path(METRICS_FILE).exists() else {}


def score_features(features: pd.DataFrame, latest_only: bool = True) -> pd.DataFrame:
    if not model_available():
        return pd.DataFrame()
    columns = json.loads(Path(FEATURE_FILE).read_text(encoding="utf-8"))
    usable = features.dropna(subset=columns).copy()
    if usable.empty:
        return usable
    usable["score"] = joblib.load(MODEL_FILE).predict_proba(usable[columns])[:, 1]
    if latest_only:
        usable = usable.sort_values("Date").groupby("Ticker", as_index=False).tail(1)
    usable["signal"] = pd.cut(usable["score"], [-1, 0.6, 0.7, 1.01], labels=["觀望", "關注", "偏多"])
    return usable.sort_values("score", ascending=False).reset_index(drop=True)