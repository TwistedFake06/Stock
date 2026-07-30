# options_position.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class OptionsPositionPlan:
    contracts: int
    total_max_loss: float
    total_max_profit: float
    risk_pct: float
    r_multiple: float
    notes: List[str] = field(default_factory=list)


def calc_options_position(
    max_loss_per_contract: float,
    max_profit_per_contract: float,
    account_size: float,
    risk_per_trade: float,
) -> OptionsPositionPlan:
    """
    嚴格用 max_loss 反推張數，確保單筆風險 ≤ 1R。
    適用於 SPY / QQQ / VOO 等垂直價差。
    """
    notes: List[str] = []

    if max_loss_per_contract <= 0 or account_size <= 0 or risk_per_trade <= 0:
        return OptionsPositionPlan(
            contracts=0,
            total_max_loss=0.0,
            total_max_profit=0.0,
            risk_pct=0.0,
            r_multiple=0.0,
            notes=["參數錯誤"]
        )

    # 向下取整，絕不超過設定的 1R
    contracts = int(risk_per_trade // max_loss_per_contract)

    if contracts < 1:
        notes.append(
            f"單張最大虧損 ${max_loss_per_contract:.0f} 已超過你的 1R (${risk_per_trade:.0f})，不建議開倉"
        )
        return OptionsPositionPlan(
            contracts=0,
            total_max_loss=0.0,
            total_max_profit=0.0,
            risk_pct=0.0,
            r_multiple=0.0,
            notes=notes
        )

    total_max_loss = contracts * max_loss_per_contract
    total_max_profit = contracts * max_profit_per_contract
    risk_pct = (total_max_loss / account_size) * 100
    r_multiple = total_max_profit / total_max_loss if total_max_loss > 0 else 0.0

    if total_max_loss < risk_per_trade * 0.7:
        notes.append(f"目前只用了 {total_max_loss / risk_per_trade:.2f}R，可考慮加倉接近 1R")
    if r_multiple < 0.35:
        notes.append("盈虧比偏低，請確認勝率是否足夠高")

    return OptionsPositionPlan(
        contracts=contracts,
        total_max_loss=round(total_max_loss, 2),
        total_max_profit=round(total_max_profit, 2),
        risk_pct=round(risk_pct, 2),
        r_multiple=round(r_multiple, 2),
        notes=notes
    )