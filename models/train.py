"""以時間切分訓練未來五日上升機率模型。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from config import FEATURE_FILE, METRICS_FILE, MODEL_DIR, MODEL_FILE
from features.engineering import FEATURE_COLUMNS


def train_classifier(features: pd.DataFrame) -> dict:
    from lightgbm import LGBMClassifier, early_stopping

    usable = features.dropna(subset=FEATURE_COLUMNS + ["y"]).sort_values("Date").copy()
    if len(usable) < 200 or usable["Date"].nunique() < 30:
        raise ValueError("可用訓練樣本不足（至少需要 200 行及 30 個交易日）。")
    split_date = usable["Date"].drop_duplicates().sort_values().iloc[int(usable["Date"].nunique() * 0.8)]
    train, valid = usable[usable["Date"] < split_date], usable[usable["Date"] >= split_date]
    if train["y"].nunique() < 2 or valid["y"].nunique() < 2:
        raise ValueError("時間切分後類別不足，請增加股票池或歷史年期。")
    positive_weight = (train["y"] == 0).sum() / max((train["y"] == 1).sum(), 1)
    model = LGBMClassifier(
        objective="binary", metric="auc", learning_rate=0.05, num_leaves=31, min_child_samples=50,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1, verbosity=-1, n_estimators=300,
        scale_pos_weight=positive_weight, random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train["y"], eval_set=[(valid[FEATURE_COLUMNS], valid["y"])], callbacks=[early_stopping(30, verbose=False)])
    score = model.predict_proba(valid[FEATURE_COLUMNS])[:, 1]
    valid["score"] = score
    deciles = valid.groupby(pd.qcut(score, 10, duplicates="drop"), observed=False)["fwd_ret_5"].mean().round(5).tolist()
    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "auc": round(float(roc_auc_score(valid["y"], score)), 4),
        "accuracy": round(float(accuracy_score(valid["y"], score >= 0.5)), 4),
        "precision": round(float(precision_score(valid["y"], score >= 0.5, zero_division=0)), 4),
        "recall": round(float(recall_score(valid["y"], score >= 0.5, zero_division=0)), 4),
        "samples": len(usable), "date_range": f"{usable['Date'].min():%Y-%m-%d} 至 {usable['Date'].max():%Y-%m-%d}",
        "decile_fwd_returns": deciles,
    }
    Path(MODEL_DIR).mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    Path(FEATURE_FILE).write_text(json.dumps(FEATURE_COLUMNS), encoding="utf-8")
    Path(METRICS_FILE).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics