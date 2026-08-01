"""DataFrame export helpers for UI tables."""
from __future__ import annotations

import pandas as pd

from options_models import SpreadIdea

def legs_to_frame(idea: SpreadIdea) -> pd.DataFrame:
    rows = []
    for leg in idea.legs:
        bid = float(leg.bid or 0)
        ask = float(leg.ask or 0)
        mid = float(leg.mid or 0)
        fill = float(getattr(leg, "fill", 0) or mid)
        fill_src = getattr(leg, "fill_source", "") or "mid"
        if bid > 0 and ask > 0 and mid > 0:
            ba = f"{(ask - bid) / mid * 100:.0f}%"
        else:
            ba = "—"
        src_zh = {"bid": "买价bid", "ask": "卖价ask", "mid": "中间价"}.get(
            fill_src, fill_src
        )
        rows.append(
            {
                "方向": "卖出" if leg.side == "sell" else "买入",
                "类型": "看涨" if leg.right == "call" else "看跌",
                "行权价": leg.strike,
                "成交估": round(fill, 2),
                "成交用": src_zh,
                "中间价": round(mid, 2),
                "买价": round(bid, 2),
                "卖价": round(ask, 2),
                "买卖差": ba,
                "未平仓": int(leg.oi or 0),
                "成交量": int(leg.volume or 0),
            }
        )
    return pd.DataFrame(rows)


def ideas_to_frame(ideas: list[SpreadIdea], sort_by: str = "score") -> pd.DataFrame:
    rows = []
    for i in ideas:
        wr_p = getattr(i, "win_rate_profit", None)
        if wr_p is None:
            wr_p = getattr(i, "pop_est", None)
        rows.append(
            {
                "分数": i.score,
                "名称": i.name,
                "赢面%": wr_p,
                "流动性": getattr(i, "liquidity_label", "") or "—",
                "流动性分": getattr(i, "liquidity_score", None),
                "卖出收$": i.net_credit,
                "买进付$": i.net_debit,
                "最多赚$": i.max_profit,
                "最多亏$": i.max_loss,
                "像哪种做法": getattr(i, "playbook_style", ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if sort_by == "winrate":
        df = df.sort_values("赢面%", ascending=False, na_position="last")
    elif sort_by == "liquidity":
        df = df.sort_values("流动性分", ascending=False, na_position="last")
    else:
        df = df.sort_values("分数", ascending=False, na_position="last")
    return df.reset_index(drop=True)
