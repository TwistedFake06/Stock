"""Trade-level performance summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int | str]:
    """Return headline metrics calculated from a trade log."""
    if trades.empty:
        return {"error": "No trades"}

    r_multiples = trades["r_multiple"]
    losses = r_multiples[r_multiples < 0]
    gains = r_multiples[r_multiples > 0]
    profit_factor = gains.sum() / abs(losses.sum()) if not losses.empty else np.inf
    avg_r = r_multiples.mean()

    return {
        "total_trades": len(trades),
        "win_rate": round(trades["is_win"].mean() * 100, 1),
        "avg_r": round(avg_r, 3),
        "expectancy": round(avg_r, 3),
        "profit_factor": round(profit_factor, 2),
        "avg_pnl": round(trades["pnl"].mean(), 2),
    }