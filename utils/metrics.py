"""回測績效計算。"""
from __future__ import annotations

import pandas as pd


def performance(equity: pd.Series) -> dict[str, float]:
    if equity.empty:
        return {"total_return": 0.0, "annual_return": 0.0, "max_drawdown": 0.0}
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(len(equity) / 252, 1 / 252)
    annual = (1 + total) ** (1 / years) - 1
    drawdown = equity / equity.cummax() - 1
    return {"total_return": total, "annual_return": annual, "max_drawdown": drawdown.min()}