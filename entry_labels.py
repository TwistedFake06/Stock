"""User-facing entry wording (simple, direct).

Internal SOP still uses 适合入场 / 谨慎试仓 / 观望 / 回避.
UI and scans show: 可入場 / soft 可入場 / 可考慮 / 不入場.
"""
from __future__ import annotations

from collections.abc import Collection

# Display labels
ENTER_YES = "可入場"
ENTER_SOFT = "soft 可入場"
ENTER_MAYBE = "可考慮"
ENTER_NO = "不入場"

# Soft-Confirm paper experiment floors (digest RR field = rr_t1)
HARD_RR_MIN = 1.2  # documented Confirm strict context; hard path still via screen_layer
SOFT_RR_MIN = 0.9  # A-tier soft 可入場 only

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
    ENTER_SOFT: ENTER_SOFT,
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
    ENTER_SOFT: ENTER_SOFT,
    ENTER_MAYBE: ENTER_MAYBE,
    ENTER_NO: ENTER_NO,
}

_MAYBE_OR_CLOSE_ENTER_OK = frozenset({"适合入场", "谨慎试仓", "可以入場", "可以試倉"})


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
    """Simple layer for watchlist scan: 可入場 / 可考慮 / 不入場.

    Hard 可入場 = Confirm (适合入场 + no red/yellow). RR≥1.2 is the hist/strict
    Confirm context baked into lights/enter_ok; this helper stays unchanged.
    """
    if enter_ok == "适合入场" and not any_red and not any_yellow:
        return ENTER_YES
    if enter_ok in ("适合入场", "谨慎试仓"):
        return ENTER_MAYBE
    return ENTER_NO


def coerce_rr(rr: object) -> float | None:
    """Parse RR; None / blank / NaN → None (soft must not treat as 0)."""
    if rr is None:
        return None
    if isinstance(rr, str) and not rr.strip():
        return None
    try:
        v = float(rr)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _rr_ok(rr: float | None, floor: float) -> bool:
    v = coerce_rr(rr)
    if v is None:
        return False
    return v >= float(floor)


def soft_rr_from_row(row: dict) -> float | None:
    """RR for soft digest rebuild from a scan cache row.

    If `_rr_soft` was stored at scan time (including explicit None), keep it —
    do not substitute display `R:R` (may be another horizon). Legacy rows
    without the key fall back to `R:R` / `R:R_0-2周`.
    """
    if "_rr_soft" in row:
        return coerce_rr(row.get("_rr_soft"))
    for key in ("R:R", "R:R_0-2周"):
        if key in row:
            v = coerce_rr(row.get(key))
            if v is not None:
                return v
    return None


def is_soft_enter(
    symbol: str,
    *,
    layer: str,
    rr: float | None,
    enter_ok: str | None = None,
    a_tier: Collection[str] | None = None,
    soft_rr_min: float = SOFT_RR_MIN,
) -> bool:
    """A-tier soft 可入場: not hard, RR≥soft_rr_min, 可考慮 or close. B never soft.

    ``layer`` should be the hard screen_layer result (可入場 / 可考慮 / 不入場).
    ``rr`` is the same field digest stores (sop.rr_t1).
    """
    if a_tier is None:
        try:
            from config import CORE_A_TIER_SET as a_tier  # type: ignore
        except Exception:
            return False
    sym = (symbol or "").strip().upper()
    if not sym or sym not in {str(x).strip().upper() for x in a_tier}:
        return False
    if layer == ENTER_YES:
        return False
    if not _rr_ok(rr, soft_rr_min):
        return False
    if layer == ENTER_MAYBE:
        return True
    # "close": enter_ok still trial/full but layered 不入場 by lights (rare)
    eo = (enter_ok or "").strip()
    return eo in _MAYBE_OR_CLOSE_ENTER_OK


def digest_bucket(
    symbol: str,
    *,
    layer: str,
    rr: float | None,
    enter_ok: str | None = None,
    a_tier: Collection[str] | None = None,
) -> str:
    """Bucket for Daily Confirm digest: 可入場 / soft 可入場 / 可考慮 / 不入場."""
    if layer == ENTER_YES:
        return ENTER_YES
    if is_soft_enter(
        symbol, layer=layer, rr=rr, enter_ok=enter_ok, a_tier=a_tier
    ):
        return ENTER_SOFT
    if layer == ENTER_MAYBE:
        return ENTER_MAYBE
    return ENTER_NO


def filter_layers_for_min_level(min_level: str | None) -> frozenset[str] | None:
    """Allowed 层級 labels for scan filter; None means show all.

    Legacy session value「只可入場」maps to hard Confirm + soft (same as 可入場+soft).
    """
    s = (min_level or "").strip()
    if s in (
        "可入場+soft",
        "可入場+soft（預設）",
        "只可入場",
        "硬+soft",
    ):
        return frozenset({ENTER_YES, ENTER_SOFT})
    if s in ("只 soft 可入場", "只 soft"):
        return frozenset({ENTER_SOFT})
    if s in ("可入場+可考慮", "含可考慮"):
        return frozenset({ENTER_YES, ENTER_SOFT, ENTER_MAYBE})
    if s in ("全部", ""):
        return None
    # Unknown / stale UI value → safe default (hard + soft), not empty.
    return frozenset({ENTER_YES, ENTER_SOFT})

