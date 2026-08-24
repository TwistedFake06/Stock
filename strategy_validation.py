"""Walk-forward validation helpers for the live technical rule score."""

from __future__ import annotations

import math

import pandas as pd

from analysis import analyze_bias
from indicators import enrich

SCORE_BIN_ORDER = ["强烈看空", "看空", "中性", "看多", "强烈看多"]


def score_bin(score: float) -> str:
    if score <= -45:
        return "强烈看空"
    if score <= -18:
        return "看空"
    if score < 18:
        return "中性"
    if score < 45:
        return "看多"
    return "强烈看多"


def _path_outcome(
    future: pd.DataFrame,
    *,
    entry: float,
    atr: float,
    direction: int,
    target_atr: float,
    stop_atr: float,
) -> str | None:
    if direction == 0 or not math.isfinite(atr) or atr <= 0:
        return None

    if direction > 0:
        target = entry + target_atr * atr
        stop = entry - stop_atr * atr
    else:
        target = entry - target_atr * atr
        stop = entry + stop_atr * atr

    for row in future.itertuples(index=False):
        high = float(row.High)
        low = float(row.Low)
        target_hit = high >= target if direction > 0 else low <= target
        stop_hit = low <= stop if direction > 0 else high >= stop
        if target_hit and stop_hit:
            return "歧义"
        if target_hit:
            return "目标先触"
        if stop_hit:
            return "止损先触"
    return "到期未触"


def _benchmark_closes(benchmark: pd.DataFrame | None) -> pd.Series | None:
    if benchmark is None or benchmark.empty or not {"Date", "Close"}.issubset(benchmark.columns):
        return None
    data = benchmark[["Date", "Close"]].copy()
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None).dt.normalize()
    data = data.drop_duplicates("Date", keep="last").set_index("Date")
    return data["Close"].astype(float)


def build_validation_samples(
    history: pd.DataFrame,
    *,
    horizon: int = 10,
    benchmark: pd.DataFrame | None = None,
    target_atr: float = 1.5,
    stop_atr: float = 1.0,
    warmup: int = 60,
) -> pd.DataFrame:
    """Evaluate each historical score only against bars that followed it."""
    required = {"Date", "Open", "High", "Low", "Close"}
    if history is None or history.empty or not required.issubset(history.columns):
        return pd.DataFrame()
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    data = history.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = enrich(data)
    benchmark_close = _benchmark_closes(benchmark)
    rows: list[dict[str, object]] = []

    first_index = max(int(warmup) - 1, 0)
    for index in range(first_index, len(data) - horizon):
        report = analyze_bias(data.iloc[: index + 1])
        entry = float(data["Close"].iloc[index])
        exit_price = float(data["Close"].iloc[index + horizon])
        forward_return = (exit_price / entry - 1.0) * 100.0
        direction = 1 if report.score >= 18 else -1 if report.score <= -18 else 0
        directional_return = forward_return * direction if direction else None

        date = pd.Timestamp(data["Date"].iloc[index]).normalize()
        exit_date = pd.Timestamp(data["Date"].iloc[index + horizon]).normalize()
        benchmark_return = None
        if benchmark_close is not None and date in benchmark_close.index and exit_date in benchmark_close.index:
            start = float(benchmark_close.loc[date])
            end = float(benchmark_close.loc[exit_date])
            if start > 0:
                benchmark_return = (end / start - 1.0) * 100.0

        atr_value = data["ATR"].iloc[index] if "ATR" in data.columns else float("nan")
        atr = float(atr_value) if pd.notna(atr_value) else float("nan")
        outcome = _path_outcome(
            data.iloc[index + 1 : index + horizon + 1],
            entry=entry,
            atr=atr,
            direction=direction,
            target_atr=target_atr,
            stop_atr=stop_atr,
        )

        rows.append(
            {
                "date": date,
                "score": float(report.score),
                "score_bin": score_bin(report.score),
                "direction": "做多" if direction > 0 else "做空" if direction < 0 else "中性",
                "forward_return_pct": forward_return,
                "directional_return_pct": directional_return,
                "benchmark_return_pct": benchmark_return,
                "excess_return_pct": (
                    forward_return - benchmark_return if benchmark_return is not None else None
                ),
                "path_outcome": outcome,
            }
        )

    return pd.DataFrame(rows)


def summarize_score_bins(samples: pd.DataFrame) -> pd.DataFrame:
    """Aggregate score behavior without converting missing outcomes into wins."""
    if samples is None or samples.empty:
        return pd.DataFrame()

    summaries: list[dict[str, object]] = []
    for label in SCORE_BIN_ORDER:
        group = samples[samples["score_bin"] == label]
        if group.empty:
            continue
        path = group["path_outcome"].dropna()
        summaries.append(
            {
                "分数区间": label,
                "样本": len(group),
                "平均未来回报%": group["forward_return_pct"].mean(),
                "上涨比例%": (group["forward_return_pct"] > 0).mean() * 100.0,
                "平均超额回报%": group["excess_return_pct"].mean(),
                "方向后平均回报%": group["directional_return_pct"].mean(),
                "目标先触%": (path == "目标先触").mean() * 100.0 if len(path) else None,
                "止损先触%": (path == "止损先触").mean() * 100.0 if len(path) else None,
                "到期未触%": (path == "到期未触").mean() * 100.0 if len(path) else None,
                "歧义%": (path == "歧义").mean() * 100.0 if len(path) else None,
            }
        )
    return pd.DataFrame(summaries)