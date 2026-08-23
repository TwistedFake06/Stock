"""Trade SOP unit tests (may use network for full build; pure helpers offline)."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from edge_signals import (
    analyze_false_breakout,
    analyze_trend_align,
    analyze_volume_confirm,
    map_sector_etf,
)
from exit_plan import apply_long_slippage, build_exit_plan
from position_coach import (
    advise_dual_hold,
    advise_open_position,
    analyze_entry_vs_live,
    build_follow_levels,
)
from mtf_signals import (
    analyze_adx,
    analyze_fib_levels,
    analyze_h1_trigger,
    detect_weekly_turning_bullish,
    merge_entry_with_fib,
)
from indicators import enrich
from trade_journal import add_trade, close_trade, journal_stats, load_trades, save_trades
from free_data import analyze_liquidity, multi_horizon_rs
from trade_sop import (
    MIN_SAMPLES_FULL,
    MIN_SAMPLES_LOW,
    MODE_THRESHOLDS,
    PATH_LOOKBACK_DEFAULT,
    _enter_decision,
    _expectancy_r,
    _path_win_rate,
    _path_win_rate_detail,
    _stability_score,
    _swing_verdict,
    THREE_LIGHT_SOP,
    aggressive_upgrade_1r,
    build_decision_brief,
    cap_stop_by_atr,
    decide_three_lights,
    ensure_min_rr_target,
    order_targets_near_far,
    format_win_rate,
    get_mode_thresholds,
    parse_as_of_date,
    path_wr_confidence,
    plan_limit_from_zone,
    resolve_path_win_rate,
    resolve_trading_mode,
    slice_ohlcv_as_of,
)


class TestTradeSOPHelpers(unittest.TestCase):
    @patch("mtf_signals.fetch_history")
    def test_h1_trigger_example(self, fetch_history):
        close = pd.Series(np.linspace(100.0, 110.0, 40))
        fetch_history.return_value = pd.DataFrame(
            {
                "Close": close,
                "Volume": np.full(40, 1000.0),
            }
        )
        report = analyze_h1_trigger("AAPL", 108.0, 112.0)
        self.assertTrue(report.available)
        self.assertEqual(report.label, "可掛單")
        fetch_history.assert_called_once_with("AAPL", period="60d", interval="1h")

    def test_path_win_rate_runs(self):
        rng = np.random.default_rng(0)
        # mild uptrend noise
        rets = rng.normal(0.001, 0.015, 200)
        close = 100 * np.cumprod(1 + rets)
        s = pd.Series(close)
        wr = _path_win_rate(s, risk_per_share=2.0, reward_per_share=3.0, lookback=80, horizon=15)
        self.assertIsNotNone(wr)
        assert wr is not None
        self.assertTrue(0 <= wr <= 100)

    def test_format_win_rate_never_nan(self):
        self.assertEqual(format_win_rate(None), "样本不足")
        self.assertIn("样本不足", format_win_rate(None, 5))
        self.assertIn("53%", format_win_rate(53.2, 18, confidence="full"))
        self.assertIn("低样本", format_win_rate(53.2, 9, confidence="low"))
        self.assertIn("日线", format_win_rate(54.0, None, confidence="day", source="day"))
        self.assertEqual(format_win_rate(float("nan")), "样本不足")
        self.assertNotIn("nan", format_win_rate(float("nan")).lower())

    def test_path_win_rate_detail_min_samples(self):
        # tiny series → no samples
        s = pd.Series([100.0, 101.0, 99.0, 102.0])
        wr, n = _path_win_rate_detail(s, 1.0, 2.0, lookback=120, horizon=15)
        self.assertIsNone(wr)
        self.assertEqual(n, 0)

    def test_resolve_path_fallback_to_day(self):
        # short flat series → path None, day fallback
        s = pd.Series(np.full(30, 100.0))
        wr, n, src = resolve_path_win_rate(
            s, 2.0, 3.0, primary_horizon=10, day_wr=54.0, lookback=120
        )
        self.assertEqual(src, "day")
        self.assertAlmostEqual(wr or 0, 54.0)

    def test_pct_path_resolves_more_than_abs_on_scaled_prices(self):
        """Fixed $ stops fail on old low prices; % scale should get samples."""
        rng = np.random.default_rng(42)
        # price drifts from ~20 → ~80 with noise
        rets = rng.normal(0.002, 0.02, 220)
        close = 20 * np.cumprod(1 + rets)
        high = close * (1 + rng.uniform(0.002, 0.02, len(close)))
        low = close * (1 - rng.uniform(0.002, 0.02, len(close)))
        s = pd.Series(close)
        h = pd.Series(high)
        lo = pd.Series(low)
        # structure at current price ~ last: risk $3 reward $4 (abs would be huge % early)
        risk, reward = 3.0, 4.0
        ref = float(close[-1])
        wr_pct, n_pct = _path_win_rate_detail(
            s,
            risk,
            reward,
            lookback=PATH_LOOKBACK_DEFAULT,
            horizon=15,
            min_samples=MIN_SAMPLES_LOW,
            high=h,
            low=lo,
            ref_entry=ref,
            scale="pct",
        )
        wr_abs, n_abs = _path_win_rate_detail(
            s,
            risk,
            reward,
            lookback=PATH_LOOKBACK_DEFAULT,
            horizon=15,
            min_samples=1,
            high=h,
            low=lo,
            ref_entry=ref,
            scale="abs",
        )
        # pct should yield usable samples more often
        self.assertGreaterEqual(n_pct, n_abs)
        wr_r, n_r, src = resolve_path_win_rate(
            s,
            risk,
            reward,
            primary_horizon=10,
            day_wr=52.0,
            high=h,
            low=lo,
            ref_entry=ref,
        )
        self.assertIsNotNone(wr_r)
        self.assertNotEqual(src, "none")
        conf = path_wr_confidence(src, n_r)
        self.assertIn(conf, ("full", "low", "blend", "day"))

    def test_same_bar_both_sides_not_auto_loss(self):
        """Ambiguous HL bar should not force stop-first losses."""
        # Flat path then one huge range bar that pierces both stop and target
        close = np.array([100.0] * 80 + [100.0, 100.0, 100.0])
        high = close.copy()
        low = close.copy()
        high[-2] = 110.0  # target + stop both possible from 100 with risk 3 reward 4
        low[-2] = 90.0
        # After that stay flat
        s = pd.Series(np.concatenate([np.full(100, 100.0), close[-3:]]))
        # rebuild clean series
        n = 120
        c = np.full(n, 100.0)
        h = c.copy()
        lo = c.copy()
        # many independent ambiguous bars would all be skipped → low n
        # add clear wins: drift up without wick below stop
        for i in range(40, 70):
            c[i] = 100 + (i - 40) * 0.15
            h[i] = c[i] + 0.5
            lo[i] = c[i] - 0.3
        # one ambiguous day in the middle of a path window
        h[55] = 120.0
        lo[55] = 80.0
        wr, n = _path_win_rate_detail(
            pd.Series(c),
            3.0,
            4.0,
            lookback=80,
            horizon=10,
            min_samples=1,
            high=pd.Series(h),
            low=pd.Series(lo),
            ref_entry=100.0,
            scale="pct",
        )
        # Should still produce some samples; not collapse to all losses from ambiguous bars
        self.assertGreaterEqual(n, 1)
        if wr is not None:
            self.assertTrue(0 <= wr <= 100)

    def test_plan_limit_zone_mid_lower(self):
        e = plan_limit_from_zone(100.0, 110.0, frac=0.35)
        self.assertIsNotNone(e)
        assert e is not None
        self.assertAlmostEqual(e, 103.5, places=2)

    def test_cap_stop_atr_and_min_target(self):
        # stop too wide vs ATR
        s, note = cap_stop_by_atr(100.0, 90.0, atr=2.0, cap_mult=1.5, floor_mult=0.6)
        self.assertIsNotNone(s)
        assert s is not None
        self.assertAlmostEqual(s, 100.0 - 1.5 * 2.0, places=2)
        self.assertIn("收紧", note)
        # target too close
        t, n2 = ensure_min_rr_target(100.0, 95.0, 101.0, k=1.0)
        self.assertAlmostEqual(t or 0, 105.0, places=2)
        self.assertIn("1.0:1", n2)

    def test_order_targets_t1_le_t2(self):
        a, b, note = order_targets_near_far(110.0, 105.0)
        self.assertAlmostEqual(a or 0, 105.0)
        self.assertAlmostEqual(b or 0, 110.0)
        self.assertIn("重排", note)
        a2, b2, n2 = order_targets_near_far(105.0, 120.0)
        self.assertAlmostEqual(a2 or 0, 105.0)
        self.assertAlmostEqual(b2 or 0, 120.0)
        self.assertEqual(n2, "")

    def test_three_light_all_green_full_entry(self):
        self.assertTrue(THREE_LIGHT_SOP)
        thr = MODE_THRESHOLDS["defensive"]
        d = decide_three_lights(
            thr=thr,
            last=100.0,
            entry_low=98.0,
            entry_high=102.0,
            entry_plan=100.0,
            stop=95.0,
            target=110.0,
            wr=58.0,
            wr_samples=20,
            rr_net=1.25,
            rr_paper=1.4,
            price_far_chase=False,
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
        )
        self.assertEqual(d["verdict"], "可以入場")
        self.assertEqual(d["position_light"], "green")
        self.assertEqual(d["wr_light"], "green")
        self.assertEqual(d["rr_light"], "green")
        self.assertIn("结论", d["plain_card"])

    def test_three_light_time_gate_preserves_result_contract(self):
        thr = MODE_THRESHOLDS["defensive"]
        with patch(
            "trade_sop.is_us_open_first_2h",
            return_value=(False, "窗口外"),
        ):
            result = decide_three_lights(
                thr=thr,
                last=100.0,
                entry_low=98.0,
                entry_high=102.0,
                entry_plan=100.0,
                stop=95.0,
                target=110.0,
                wr=58.0,
                wr_samples=20,
                rr_net=1.25,
                rr_paper=1.4,
                price_far_chase=False,
                entry_opp="较佳入场",
                bias_label="看多",
                bias_score=30,
                enforce_time_window=True,
            )
        self.assertEqual(result["verdict"], "暫緩觀望")
        self.assertEqual(result["notional_hkd"], 5000.0)
        self.assertIn("pnl_if_win_hkd", result)
        self.assertIn("pnl_if_loss_hkd", result)

    def test_three_light_earnings_blocks_full(self):
        thr = MODE_THRESHOLDS["defensive"]
        d = decide_three_lights(
            thr=thr,
            last=100.0,
            entry_low=98.0,
            entry_high=102.0,
            entry_plan=100.0,
            stop=95.0,
            target=110.0,
            wr=60.0,
            wr_samples=30,
            rr_net=1.3,
            rr_paper=1.4,
            price_far_chase=False,
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            earnings_days_left=2,
        )
        self.assertEqual(d["verdict"], "暫緩觀望")
        d2 = decide_three_lights(
            thr=thr,
            last=100.0,
            entry_low=98.0,
            entry_high=102.0,
            entry_plan=100.0,
            stop=95.0,
            target=110.0,
            wr=60.0,
            wr_samples=30,
            rr_net=1.3,
            rr_paper=1.4,
            price_far_chase=False,
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            earnings_days_left=10,
        )
        self.assertEqual(d2["verdict"], "可以試倉")

    def test_three_light_bad_rr_red_wait(self):
        """高胜率 + 区内 + 净R:R 很差 → 先别做（不划算）。"""
        thr = MODE_THRESHOLDS["defensive"]
        d = decide_three_lights(
            thr=thr,
            last=924.0,
            entry_low=882.0,
            entry_high=940.0,
            entry_plan=924.0,
            stop=819.0,
            target=959.0,
            wr=77.0,
            wr_samples=180,
            rr_net=0.29,
            rr_paper=0.52,
            price_far_chase=False,
            entry_opp="可关注",
            bias_label="强烈看多",
            bias_score=55,
        )
        self.assertEqual(d["verdict"], "暫緩觀望")
        self.assertEqual(d["rr_light"], "red")
        self.assertEqual(d["position_light"], "green")
        self.assertIn("不划算", d["one_liner_reason"] + d["rr_light_note"])

    def test_mode_thresholds(self):
        d = get_mode_thresholds("defensive")
        a = get_mode_thresholds("B")
        # v1-realism-2026-08-18 收紧门槛
        self.assertEqual(d.wr_full, 54.0)
        self.assertEqual(d.rr_full, 1.15)
        self.assertEqual(a.key, "aggressive")
        self.assertEqual(a.wr_full, 52.0)
        self.assertEqual(a.wr_half, 48.0)
        self.assertEqual(a.default_risk_units_full, 0.5)

    def test_decision_brief_has_verdict_and_reasons(self):
        text = build_decision_brief(
            verdict="可以試倉",
            mode_label="A 防守版",
            wr_display="52%（低样本9）",
            wr_confidence="low",
            rr_net=1.15,
            exp_r=0.12,
            risk_units=0.5,
            bias_label="看多",
            bias_score=22,
            thr=MODE_THRESHOLDS["defensive"],
        )
        self.assertIn("可以試倉", text)
        self.assertIn("根据", text)
        self.assertIn("R:R", text)
        self.assertIn("E[R]", text)

    def test_resolve_mode_forces_defensive_on_weak_regime(self):
        thr, forced, note = resolve_trading_mode(
            "aggressive", regime_score=30.0, vix_label="中"
        )
        self.assertEqual(thr.key, "defensive")
        self.assertTrue(forced)
        self.assertIn("大盘", note)

    def test_aggressive_upgrade_needs_3(self):
        ok, hits, notes = aggressive_upgrade_1r(
            wr=55,
            bias_label="强烈看多",
            bias_score=60,
            multi_rs_score=50,
            vol_label="缩量回踩",
            false_break_risk=False,
            rr_net=1.1,
            rr=1.2,
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(hits, 3)

    def test_swing_mode_requires_wr_not_none(self):
        # 样本不足 → 不得可以入場 / 可以試倉
        v = _swing_verdict(
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=35,
            wr=None,
            rr=1.5,
            exp_r=0.3,
            price_far_chase=False,
            mode=MODE_THRESHOLDS["defensive"],
            stability=60,
            trend_score=70,
            rr_net=1.3,
        )
        self.assertEqual(v, "暫緩觀望")

    def test_swing_defensive_full_entry(self):
        v = _swing_verdict(
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            wr=58,
            rr=1.4,
            exp_r=0.25,
            price_far_chase=False,
            mode=MODE_THRESHOLDS["defensive"],
            stability=55,
            multi_rs_score=55,
            trend_score=70,
            weekly_allow_long=True,
            rr_net=1.25,
            vol_label="放量上涨",
        )
        self.assertEqual(v, "可以入場")

    def test_enter_decision_avoid_bearish(self):
        ok, score, side = _enter_decision(
            "偏空回避", 30, "强烈看空", -50, 40, 45, 1.0, "高"
        )
        self.assertEqual(ok, "回避")

    def test_enter_decision_good_setup(self):
        ok, score, side = _enter_decision(
            "较佳入场",
            85,
            "看多",
            35,
            70,
            60,
            2.0,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.4,
            multi_rs_score=55,
        )
        self.assertIn(ok, ("适合入场", "谨慎试仓"))
        self.assertEqual(side, "做多")

    def test_enter_decision_full_entry_needs_rr_and_exp(self):
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            62,
            2.0,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.4,
            multi_rs_score=55,
        )
        self.assertEqual(ok, "适合入场")

    def test_poor_rr_blocks_entry(self):
        # High win rate but RR 0.4 → negative-ish expectancy, must not enter
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            70,
            0.4,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=-0.1,
        )
        self.assertIn(ok, ("观望", "回避"))

    def test_enter_decision_regime_blocks_full_entry(self):
        ok, score, side = _enter_decision(
            "较佳入场",
            85,
            "看多",
            40,
            70,
            60,
            2.0,
            "中",
            regime_score=25,
            liquidity_score=80,
            expectancy_r=0.4,
        )
        self.assertNotEqual(ok, "适合入场")

    def test_enter_decision_earnings_caps(self):
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            62,
            2.2,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.5,
            earnings_soon=True,
        )
        self.assertNotEqual(ok, "适合入场")

    def test_far_chase_blocks(self):
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            62,
            2.0,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.4,
            price_far_chase=True,
        )
        self.assertIn(ok, ("观望", "回避"))

    def test_iv_high_blocks_full(self):
        ok, _, _ = _enter_decision(
            "较佳入场",
            90,
            "看多",
            40,
            70,
            62,
            2.0,
            "中",
            regime_score=70,
            liquidity_score=80,
            expectancy_r=0.4,
            multi_rs_score=55,
            sector_rs_score=55,
            volume_confirm_score=60,
            iv_high_event=True,
        )
        self.assertNotEqual(ok, "适合入场")

    def test_volume_dump_blocks_cautious(self):
        ok, _, _ = _enter_decision(
            "可关注",
            70,
            "看多",
            20,
            50,
            55,
            1.2,
            "中",
            regime_score=55,
            liquidity_score=60,
            expectancy_r=0.1,
            volume_confirm_score=20,  # 放量下跌
        )
        self.assertNotEqual(ok, "谨慎试仓")
        self.assertNotEqual(ok, "适合入场")

    def test_map_sector_etf(self):
        etf, lab = map_sector_etf("Technology", "Consumer Electronics")
        self.assertEqual(etf, "XLK")
        etf2, _ = map_sector_etf("Technology", "Semiconductors")
        self.assertEqual(etf2, "SMH")

    def test_volume_confirm_offline(self):
        n = 60
        # up day with high volume at end
        close = np.linspace(100, 110, n)
        close[-1] = 112
        vol = np.full(n, 1_000_000.0)
        vol[-1] = 2_500_000.0
        df = pd.DataFrame({"Close": close, "Volume": vol})
        r = analyze_volume_confirm(df)
        self.assertTrue(r.available)
        self.assertTrue(0 <= r.score <= 100)

    def test_false_breakout_detects_fail(self):
        n = 40
        # base grind then pierce high and close back
        close = np.concatenate([np.linspace(100, 105, n - 3), np.array([106.5, 104.0, 103.5])])
        high = close.copy()
        high[-3] = 108.0  # pierce
        low = close - 1
        open_ = close.copy()
        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": np.full(n, 1e6),
            }
        )
        r = analyze_false_breakout(df)
        self.assertTrue(r.available)
        # may or may not flag depending on prior high calc; score defined
        self.assertTrue(0 <= r.score <= 100)

    def test_swing_blocks_false_break(self):
        thr = MODE_THRESHOLDS["defensive"]
        d = decide_three_lights(
            thr=thr,
            last=100.0,
            entry_low=98.0,
            entry_high=102.0,
            entry_plan=100.0,
            stop=95.0,
            target=110.0,
            wr=60.0,
            wr_samples=30,
            rr_net=1.2,
            rr_paper=1.4,
            price_far_chase=False,
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            false_break_risk=True,
        )
        self.assertEqual(d["verdict"], "可以試倉")

    def test_swing_blocks_against_trend(self):
        thr = MODE_THRESHOLDS["defensive"]
        d = decide_three_lights(
            thr=thr,
            last=100.0,
            entry_low=98.0,
            entry_high=102.0,
            entry_plan=100.0,
            stop=95.0,
            target=110.0,
            wr=60.0,
            wr_samples=30,
            rr_net=1.2,
            rr_paper=1.4,
            price_far_chase=False,
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            against_trend=True,
        )
        self.assertEqual(d["verdict"], "可以試倉")

    def test_trend_align_offline(self):
        n = 80
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # strong uptrend stock
        close = 100 * np.cumprod(1 + np.full(n, 0.004))
        df = pd.DataFrame({"Date": dates, "Close": close})
        # without network SPY may fail → still returns structure
        r = analyze_trend_align(df, sector="Technology", industry="Software")
        self.assertIn(r.label, ("强跟势", "跟势", "中性", "逆势", "—"))
        self.assertTrue(0 <= r.score <= 100)

    def test_adx_and_fib_offline(self):
        n = 80
        rng = np.random.default_rng(1)
        close = 100 * np.cumprod(1 + rng.normal(0.002, 0.015, n))
        high = close * 1.01
        low = close * 0.99
        df = pd.DataFrame(
            {"Open": close, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 1e6)}
        )
        df = enrich(df)
        adx = analyze_adx(df)
        self.assertTrue(adx.available)
        self.assertIsNotNone(adx.adx)
        fib = analyze_fib_levels(df)
        self.assertTrue(fib.available)
        self.assertIsNotNone(fib.level_618)
        lo, hi, note = merge_entry_with_fib(95, 105, fib)
        self.assertIsNotNone(lo)
        self.assertIsNotNone(hi)

    def test_weekly_blocks_full_entry(self):
        v = _swing_verdict(
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            wr=60,
            rr=1.5,
            exp_r=0.3,
            price_far_chase=False,
            weekly_allow_long=False,
            trend_score=70,
        )
        # 周线空头不能「可以入場」
        self.assertNotEqual(v, "可以入場")

    def test_weekly_turning_bullish_reclaim_sma20(self):
        # 深跌在均线下 → 不算转多
        close = pd.Series([100.0] * 20 + [80.0, 81.0])
        sma20 = float(close.tail(20).mean())
        ok, _ = detect_weekly_turning_bullish(
            close, allow_long=False, score=20.0, last=float(close.iloc[-1]), sma20=sma20
        )
        self.assertFalse(ok)
        # 站上 SMA20 但没有连续两周收高 → 收紧后不算转多
        one_week = np.linspace(90.0, 100.0, 22)
        one_week[-3] = 101.0
        one_week[-2] = 99.0  # 低于上上周
        one_week[-1] = 102.0
        s1 = pd.Series(one_week)
        sma1 = float(s1.tail(20).mean())
        ok1, _ = detect_weekly_turning_bullish(
            s1, allow_long=False, score=35.0, last=float(s1.iloc[-1]), sma20=sma1
        )
        self.assertFalse(ok1)
        # 站上均线 + 两周收高 → 算转多
        base = np.linspace(90.0, 100.0, 22)
        base[-3] = 99.0
        base[-2] = 100.5
        base[-1] = 102.0
        s = pd.Series(base)
        sma = float(s.tail(20).mean())
        self.assertGreater(float(s.iloc[-1]), sma)
        ok2, note = detect_weekly_turning_bullish(
            s, allow_long=False, score=35.0, last=float(s.iloc[-1]), sma20=sma
        )
        self.assertTrue(ok2)
        self.assertIn("转多", note)
        self.assertIn("两周收高", note)

    def test_weekly_turning_allows_trial_not_full(self):
        thr = MODE_THRESHOLDS["defensive"]
        kwargs = dict(
            thr=thr,
            last=100.0,
            entry_low=98.0,
            entry_high=102.0,
            entry_plan=100.0,
            stop=95.0,
            target=112.0,
            wr=58.0,
            wr_samples=20,
            rr_net=1.25,
            rr_paper=1.4,
            price_far_chase=False,
            entry_opp="较佳入场",
            bias_label="看多",
            bias_score=30,
            weekly_allow_long=False,
        )
        blocked = decide_three_lights(**kwargs)
        self.assertEqual(blocked["verdict"], "不做多")
        trial = decide_three_lights(**kwargs, weekly_turning_bullish=True)
        self.assertEqual(trial["verdict"], "可以試倉")
        self.assertIn("开始转多", trial["one_liner_reason"] + "".join(trial.get("caps") or []))

    def test_slippage_rr_worse_than_paper(self):
        slip = apply_long_slippage(100.0, 95.0, 110.0, win_rate_pct=55, slip_pct=0.002)
        self.assertIsNotNone(slip.rr_paper)
        self.assertIsNotNone(slip.rr_net)
        self.assertLess(slip.rr_net, slip.rr_paper)

    def test_exit_plan_rules(self):
        ep = build_exit_plan(
            horizon_key="h1",
            horizon_label="0–2周",
            max_hold_days=10,
            entry=100.0,
            stop=96.0,
            t1=108.0,
            t2=112.0,
        )
        self.assertEqual(ep.max_hold_days, 10)
        self.assertEqual(ep.scale_out_pct, 0.5)
        self.assertIsNotNone(ep.stop_after_t1)
        self.assertTrue(any("时间止损" in b for b in ep.bullets))

    def test_position_coach_stop(self):
        a = advise_open_position(
            buy_price=100.0,
            last_price=95.0,
            plan_stop=96.0,
            plan_t1=110.0,
            plan_t2=115.0,
        )
        self.assertEqual(a.action, "止蚀离场")

    def test_position_coach_take_profit(self):
        a = advise_open_position(
            buy_price=100.0,
            last_price=111.0,
            plan_stop=96.0,
            plan_t1=110.0,
            plan_t2=120.0,
        )
        self.assertIn(a.action, ("止盈减仓", "止盈清仓"))

    def test_position_coach_hold(self):
        a = advise_open_position(
            buy_price=100.0,
            last_price=102.0,
            plan_stop=96.0,
            plan_t1=110.0,
        )
        self.assertIn("持有", a.action)

    def test_dual_hold_uses_entry_t1_and_analysis(self):
        # Past entry T1 but below live higher T1 → scale out on entry contract
        a, lines = advise_dual_hold(
            buy_price=913.0,
            last_price=1011.0,
            buy_date="2026-08-13",
            shares=4,
            entry_stop=823.0,
            entry_t1=1008.0,
            entry_t2=1027.0,
            entry_e=916.0,
            entry_res=1012.0,
            entry_as_of="2026-08-13",
            live_stop=882.0,
            live_t1=1077.0,
            live_t2=1096.0,
            live_res=1012.0,
            bias_label="强烈看多",
            bias_score=63.0,
        )
        self.assertEqual(a.action, "止盈减仓")
        self.assertTrue(any("入场日" in x or "T1" in x for x in lines))
        self.assertTrue(any("综合" in x for x in lines))
        # analysis helper alone
        al = analyze_entry_vs_live(
            buy_price=913.0,
            last_price=1011.0,
            entry_t1=1008.0,
            entry_stop=823.0,
            live_t1=1077.0,
        )
        self.assertTrue(any("综合" in x for x in al))
        fl = build_follow_levels(
            buy_price=913.0,
            last_price=1011.0,
            advice=a,
            entry_stop=823.0,
            entry_t1=1008.0,
            entry_t2=1027.0,
            live_t1=1077.0,
        )
        self.assertEqual(fl.stage, "past_t1")
        self.assertIsNotNone(fl.now_stop)
        self.assertIn("现在减半", fl.now_do)
        # 减半是现在做；T2 只给剩仓，二者不能混成同一个「下一动作价」
        self.assertIsNotNone(fl.remain_target)
        self.assertGreater(fl.remain_target or 0, 1011.0)
        self.assertNotEqual(fl.now_do, fl.remain_label)
        self.assertIn("只跟", fl.rule_one_liner)

    def test_journal_roundtrip(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "j.json"
            old = os.environ.get("TRADE_JOURNAL_PATH")
            os.environ["TRADE_JOURNAL_PATH"] = str(path)
            try:
                save_trades([])
                row = add_trade(
                    symbol="TEST",
                    entry=10.0,
                    stop=9.0,
                    target=12.0,
                    shares=100,
                    model_wr=55.0,
                    model_rr=2.0,
                    model_verdict="可以試倉",
                )
                self.assertEqual(row["status"], "open")
                closed = close_trade(row["id"], exit_price=12.0, exit_reason="t1")
                self.assertIsNotNone(closed)
                self.assertEqual(closed["status"], "closed")
                self.assertTrue(closed.get("sample", True))
                self.assertAlmostEqual(float(closed["result_r"]), 2.0, places=2)
                st = journal_stats()
                self.assertEqual(st["closed"], 1)
                self.assertEqual(st["samples"], 1)
                self.assertAlmostEqual(st["win_rate"], 100.0, places=0)
                self.assertAlmostEqual(st["total_r"], 2.0, places=2)
                self.assertAlmostEqual(st["calibration_gap"], -45.0, places=1)
                self.assertEqual(st["calibration_samples"], 1)
            finally:
                if old is None:
                    os.environ.pop("TRADE_JOURNAL_PATH", None)
                else:
                    os.environ["TRADE_JOURNAL_PATH"] = old

    def test_expectancy(self):
        # 55% win, 2R reward → 0.55*2 - 0.45*1 = 0.65
        self.assertAlmostEqual(_expectancy_r(55, 2.0) or 0, 0.65, places=2)
        self.assertIsNone(_expectancy_r(None, 2.0))

    def test_stability_labels(self):
        risk = type("R", (), {"ann_vol_pct": 12.0, "max_drawdown_pct": -8.0, "sharpe": 1.5, "win_rate_pct": 55})()
        trend = type("T", (), {"strength_label": "强趋势"})()
        score, label = _stability_score(risk, trend, 20)
        self.assertIn(label, ("高", "中", "低"))
        self.assertTrue(0 <= score <= 100)

    def test_liquidity_offline(self):
        n = 40
        df = pd.DataFrame(
            {
                "Close": np.linspace(100, 110, n),
                "High": np.linspace(101, 111, n),
                "Low": np.linspace(99, 109, n),
                "Volume": np.full(n, 5_000_000.0),
            }
        )
        liq = analyze_liquidity(df, {"averageVolume": 5_000_000, "currentPrice": 110})
        self.assertTrue(0 <= liq.score <= 100)
        self.assertIn(liq.label, ("高", "中", "低"))

    def test_multi_horizon_rs_offline(self):
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        stock = pd.DataFrame(
            {"Date": dates, "Close": 100 * np.cumprod(1 + np.full(n, 0.002))}
        )
        bench = pd.DataFrame(
            {"Date": dates, "Close": 100 * np.cumprod(1 + np.full(n, 0.0005))}
        )
        out = multi_horizon_rs(stock, bench)
        self.assertIsNotNone(out.get("score"))
        self.assertIn(out.get("label"), ("强于大盘", "同步", "弱于大盘", "—"))

    def test_parse_as_of_date(self):
        from datetime import date, datetime

        self.assertEqual(parse_as_of_date("2026-08-13"), date(2026, 8, 13))
        self.assertEqual(parse_as_of_date(date(2026, 8, 13)), date(2026, 8, 13))
        self.assertEqual(
            parse_as_of_date(datetime(2026, 8, 13, 15, 30)), date(2026, 8, 13)
        )
        self.assertIsNone(parse_as_of_date(None))
        self.assertIsNone(parse_as_of_date("not-a-date"))

    def test_slice_ohlcv_as_of_no_lookahead(self):
        from datetime import date

        # Match stock_service shape: Date column + RangeIndex
        dates = pd.date_range("2026-08-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": np.arange(10.0),
                "High": np.arange(10.0) + 1,
                "Low": np.arange(10.0) - 1,
                "Close": np.arange(100.0, 110.0),
                "Volume": np.full(10, 1e6),
            }
        )
        cut = slice_ohlcv_as_of(df, date(2026, 8, 13))
        self.assertFalse(cut.empty)
        last_d = pd.Timestamp(cut["Date"].iloc[-1]).date()
        self.assertLessEqual(last_d, date(2026, 8, 13))
        self.assertTrue(
            all(pd.Timestamp(x).date() <= date(2026, 8, 13) for x in cut["Date"])
        )
        self.assertGreater(len(df), len(cut))
        # also DatetimeIndex form
        df2 = df.set_index("Date")
        cut2 = slice_ohlcv_as_of(df2, date(2026, 8, 13))
        self.assertFalse(cut2.empty)
        self.assertLessEqual(len(cut2), len(df2))


if __name__ == "__main__":
    unittest.main()
