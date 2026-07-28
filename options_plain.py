# -*- coding: utf-8 -*-
"""白话说明：卖出价差 / 几天后买回。独立文件避免缓存 ImportError。"""

from __future__ import annotations

from typing import Any


def plain_spread_steps(idea: Any) -> list[str]:
    """用简单字解释：今天怎么做、几天后怎么平仓。"""
    is_credit = getattr(idea, "net_credit", None) is not None
    legs: list[str] = []
    for leg in idea.legs:
        side = "卖出" if leg.side == "sell" else "买入"
        kind = "看涨期权" if leg.right == "call" else "看跌期权"
        legs.append(f"{side} {leg.strike:.0f} 的{kind}")

    lines = ["做法分两步（一般不用行权）："]
    if is_credit:
        credit = float(idea.net_credit)
        lines.append(
            f"第1步（今天）：卖出一组价差，先收钱约 ${credit:.2f}/股"
            f"（一张大约收 ${credit * 100:.0f}）。"
        )
        lines.append("组合：" + " + ".join(legs))
        lines.append(
            "第2步（几天后）：买回同一组价差结束。"
            "买回付得比当初收的少 = 赚；付得多 = 蚀。"
        )
        lines.append(
            f"最坏大约亏 ${float(idea.max_loss):.0f}/张；"
            f"最好大约赚 ${float(idea.max_profit):.0f}/张。"
        )
    else:
        debit = float(idea.net_debit)
        lines.append(
            f"第1步（今天）：买进一组价差，先付钱约 ${debit:.2f}/股"
            f"（一张大约付 ${debit * 100:.0f}）。"
        )
        lines.append("组合：" + " + ".join(legs))
        lines.append(
            "第2步（几天后）：卖出同一组价差结束。"
            "卖得比当初付的多 = 赚；少 = 蚀。"
        )
        lines.append(
            f"最坏大约亏 ${float(idea.max_loss):.0f}/张；"
            f"最好大约赚 ${float(idea.max_profit):.0f}/张。"
        )
    return lines
