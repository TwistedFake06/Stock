# -*- coding: utf-8 -*-
"""Unit smoke: soft 可入場 A-tier RR≥0.9; B (e.g. NVDA) never soft."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CORE_A_TIER, CORE_A_TIER_SET, CORE_B_TIER, CORE_WATCHLIST
from entry_labels import (
    ENTER_YES,
    ENTER_SOFT,
    ENTER_MAYBE,
    ENTER_NO,
    SOFT_RR_MIN,
    coerce_rr,
    digest_bucket,
    filter_layers_for_min_level,
    is_soft_enter,
    screen_layer,
    soft_rr_from_row,
)


def test_a_tier_membership():
    assert len(CORE_WATCHLIST) == 37
    assert len(CORE_A_TIER) == 14
    assert len(CORE_B_TIER) == 23
    for s in (
        "LITE",
        "UNH",
        "COHR",
        "SMR",
        "SNDK",
        "WDC",
        "MU",
        "GOOGL",
        "VRT",
        "RXRX",
        "AAPL",
        "AMD",
        "RKLB",
        "PLTR",
    ):
        assert s in CORE_A_TIER_SET
    assert "NVDA" not in CORE_A_TIER_SET
    assert "NVDA" in CORE_B_TIER


@pytest.mark.parametrize(
    "sym,rr",
    [
        ("WDC", 0.90),
        ("WDC", 0.95),
        ("COHR", 0.90),
        ("SNDK", 0.88),  # below floor → not soft (strict >=0.9)
        ("SNDK", 0.90),
    ],
)
def test_a_tier_soft_at_rr_floor(sym, rr):
    layer = ENTER_MAYBE  # 可考慮 / close
    soft = is_soft_enter(sym, layer=layer, rr=rr, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET)
    bucket = digest_bucket(sym, layer=layer, rr=rr, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET)
    if rr >= SOFT_RR_MIN:
        assert soft is True
        assert bucket == ENTER_SOFT
    else:
        assert soft is False
        assert bucket == ENTER_MAYBE


def test_wdc_cohr_sndk_near_09_can_soft_when_at_floor():
    """Smoke: WDC/COHR/SNDK with RR~0.9 land soft if A; ~0.88 stays 可考慮."""
    for sym in ("WDC", "COHR", "SNDK"):
        assert digest_bucket(
            sym, layer=ENTER_MAYBE, rr=0.90, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
        ) == ENTER_SOFT
        assert digest_bucket(
            sym, layer=ENTER_MAYBE, rr=0.88, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
        ) == ENTER_MAYBE


def test_nvda_cannot_soft():
    """B-tier NVDA never soft even with strong RR and 可考慮."""
    assert "NVDA" in CORE_B_TIER
    assert is_soft_enter(
        "NVDA", layer=ENTER_MAYBE, rr=1.5, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
    ) is False
    assert (
        digest_bucket(
            "NVDA", layer=ENTER_MAYBE, rr=1.5, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
        )
        == ENTER_MAYBE
    )


def test_hard_yes_not_also_soft():
    assert (
        digest_bucket(
            "WDC", layer=ENTER_YES, rr=1.3, enter_ok="适合入场", a_tier=CORE_A_TIER_SET
        )
        == ENTER_YES
    )
    assert is_soft_enter(
        "WDC", layer=ENTER_YES, rr=1.3, enter_ok="适合入场", a_tier=CORE_A_TIER_SET
    ) is False


def test_hard_screen_layer_unchanged():
    assert screen_layer("适合入场", any_red=False, any_yellow=False) == ENTER_YES
    assert screen_layer("适合入场", any_red=False, any_yellow=True) == ENTER_MAYBE
    assert screen_layer("谨慎试仓", any_red=False, any_yellow=True) == ENTER_MAYBE
    assert screen_layer("观望", any_red=False, any_yellow=False) == ENTER_NO


def test_sample_soft_vs_hard_counts_from_synthetic_digest():
    """Illustrative hard vs soft counts on a fixed synthetic set."""
    rows = [
        ("GOOGL", ENTER_YES, 1.25, "适合入场"),  # hard
        ("WDC", ENTER_MAYBE, 0.90, "谨慎试仓"),  # soft
        ("COHR", ENTER_MAYBE, 0.92, "谨慎试仓"),  # soft
        ("SNDK", ENTER_MAYBE, 0.88, "谨慎试仓"),  # maybe (under floor)
        ("NVDA", ENTER_MAYBE, 1.10, "谨慎试仓"),  # maybe B — not soft
        ("MSFT", ENTER_NO, 0.50, "观望"),  # no
    ]
    hard = soft = maybe = no = 0
    for sym, layer, rr, eo in rows:
        b = digest_bucket(sym, layer=layer, rr=rr, enter_ok=eo, a_tier=CORE_A_TIER_SET)
        if b == ENTER_YES:
            hard += 1
        elif b == ENTER_SOFT:
            soft += 1
        elif b == ENTER_MAYBE:
            maybe += 1
        else:
            no += 1
    assert (hard, soft, maybe, no) == (1, 2, 2, 1)


def test_filter_layers_for_min_level():
    assert filter_layers_for_min_level("可入場+soft") == frozenset({ENTER_YES, ENTER_SOFT})
    assert filter_layers_for_min_level("可入場+soft（預設）") == frozenset(
        {ENTER_YES, ENTER_SOFT}
    )
    # legacy session key
    assert filter_layers_for_min_level("只可入場") == frozenset({ENTER_YES, ENTER_SOFT})
    assert filter_layers_for_min_level("只 soft 可入場") == frozenset({ENTER_SOFT})
    assert filter_layers_for_min_level("可入場+可考慮") == frozenset(
        {ENTER_YES, ENTER_SOFT, ENTER_MAYBE}
    )
    assert filter_layers_for_min_level("全部") is None
    # stale / unknown → safe default hard+soft
    assert filter_layers_for_min_level("???") == frozenset({ENTER_YES, ENTER_SOFT})


def test_cache_rebuild_rr_soft_preferred_over_display_rr():
    """Simulate scan_page cache rebuild: _rr_soft (rr_t1) wins over R:R (h1)."""
    sym = "WDC"
    # hard layer is 可考慮; rr_t1 at floor → soft; display R:R below floor must not demote
    hard = ENTER_MAYBE
    rr_t1 = 0.90
    display_rr = 0.50
    bucket = digest_bucket(
        sym, layer=hard, rr=rr_t1, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
    )
    assert bucket == ENTER_SOFT
    # if rebuild wrongly used display_rr, would fall to 可考慮
    wrong = digest_bucket(
        sym, layer=hard, rr=display_rr, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
    )
    assert wrong == ENTER_MAYBE


def test_b_tier_hard_yes_stays_yes_never_soft():
    """B-tier hard Confirm stays 可入場; never promoted to soft."""
    assert (
        digest_bucket(
            "NVDA", layer=ENTER_YES, rr=2.0, enter_ok="适合入场", a_tier=CORE_A_TIER_SET
        )
        == ENTER_YES
    )
    assert is_soft_enter(
        "NVDA", layer=ENTER_YES, rr=2.0, enter_ok="适合入场", a_tier=CORE_A_TIER_SET
    ) is False


def test_coerce_rr_rejects_empty_and_nan():
    assert coerce_rr(None) is None
    assert coerce_rr("") is None
    assert coerce_rr("  ") is None
    assert coerce_rr("x") is None
    assert coerce_rr(float("nan")) is None
    assert coerce_rr(0.9) == 0.9
    assert coerce_rr("1.25") == 1.25


def test_soft_rr_from_row_keeps_explicit_none():
    """Explicit _rr_soft=None must not fall back to display R:R (wrong soft)."""
    row = {"_rr_soft": None, "R:R": 1.5, "R:R_0-2周": 1.4}
    assert soft_rr_from_row(row) is None
    assert (
        digest_bucket(
            "WDC", layer=ENTER_MAYBE, rr=soft_rr_from_row(row), enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
        )
        == ENTER_MAYBE
    )


def test_soft_rr_from_row_prefers_stored_over_display():
    row = {"_rr_soft": 0.90, "R:R": 0.40}
    assert soft_rr_from_row(row) == 0.90
    assert (
        digest_bucket(
            "WDC", layer=ENTER_MAYBE, rr=soft_rr_from_row(row), enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
        )
        == ENTER_SOFT
    )


def test_soft_rr_from_row_legacy_fallback():
    row = {"R:R": 0.95}
    assert soft_rr_from_row(row) == 0.95
    row2 = {"R:R_0-2周": 1.1}
    assert soft_rr_from_row(row2) == 1.1


def test_empty_rr_never_soft():
    assert is_soft_enter(
        "WDC", layer=ENTER_MAYBE, rr=None, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
    ) is False
    assert (
        digest_bucket(
            "WDC", layer=ENTER_MAYBE, rr=None, enter_ok="谨慎试仓", a_tier=CORE_A_TIER_SET
        )
        == ENTER_MAYBE
    )


def test_empty_a_tier_never_soft():
    """Missing CORE_A (empty set) must not soft-promote anyone."""
    assert is_soft_enter(
        "WDC", layer=ENTER_MAYBE, rr=1.5, enter_ok="谨慎试仓", a_tier=frozenset()
    ) is False
    assert (
        digest_bucket(
            "WDC", layer=ENTER_MAYBE, rr=1.5, enter_ok="谨慎试仓", a_tier=frozenset()
        )
        == ENTER_MAYBE
    )
