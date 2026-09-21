"""User-facing entry wording (simple, direct).

Internal SOP still uses 适合入场 / 谨慎试仓 / 观望 / 回避.
UI and scans show: 可入場 / 可考慮 / 不入場.
"""
from __future__ import annotations

# Display labels
ENTER_YES = "可入場"
ENTER_MAYBE = "可考慮"
ENTER_NO = "不入場"

_ENTER_OK_TO_LABEL = {
    "适合入场": ENTER_YES,
    "可以入場": ENTER_YES,
    "谨慎试仓": ENTER_MAYBE,
    "可以試倉": ENTER_MAYBE,
    "观望": ENTER_NO,
    "回避": ENTER_NO,
    "暫緩觀望": ENTER_NO,
}

_LAYER_TO_LABEL = {
    "Confirm": ENTER_YES,
    "Early": ENTER_MAYBE,
    "觀察": ENTER_NO,
    "观察": ENTER_NO,
    "回避": ENTER_NO,
    ENTER_YES: ENTER_YES,
    ENTER_MAYBE: ENTER_MAYBE,
    ENTER_NO: ENTER_NO,
}

_TIER_TO_LABEL = {
    "優先看": ENTER_YES,
    "关注": ENTER_MAYBE,
    "關注": ENTER_MAYBE,
    "觀察": ENTER_NO,
    "观察": ENTER_NO,
    ENTER_YES: ENTER_YES,
    ENTER_MAYBE: ENTER_MAYBE,
    ENTER_NO: ENTER_NO,
}


def label_enter_ok(enter_ok: str | None) -> str:
    s = (enter_ok or "").strip()
    return _ENTER_OK_TO_LABEL.get(s, ENTER_NO if not s else s)


def label_layer(layer: str | None) -> str:
    s = (layer or "").strip()
    return _LAYER_TO_LABEL.get(s, ENTER_NO if not s else s)


def label_multi_tier(tier: str | None) -> str:
    s = (tier or "").strip()
    return _TIER_TO_LABEL.get(s, ENTER_NO if not s else s)


def screen_layer(enter_ok: str, *, any_red: bool, any_yellow: bool) -> str:
    """Simple layer for watchlist scan: 可入場 / 可考慮 / 不入場."""
    if enter_ok == "适合入场" and not any_red and not any_yellow:
        return ENTER_YES
    if enter_ok in ("适合入场", "谨慎试仓"):
        return ENTER_MAYBE
    return ENTER_NO
