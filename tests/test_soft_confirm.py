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
    digest_bucket,
    is_soft_enter,
    screen_layer,
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
