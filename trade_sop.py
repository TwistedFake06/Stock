"""
Investment SOP engine — practical trading checklist when a symbol is selected.

Produces: enter or not, entry price zone, stop, targets, win-rate estimate,
stability score, and ordered actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import pandas as pd

from analysis import analyze_bias
from entry_targets import analyze_entry, analyze_targets
from extra_analysis import (
    analyze_fundamentals,
    analyze_relative_strength,
    analyze_risk,
    analyze_support_resistance,
    analyze_trend,
    analyze_volume,
    build_scorecard,
    default_benchmark,
)
from edge_signals import edge_bundle
from exit_plan import (
    DEFAULT_SLIP_PCT,
    ExitPlan,
    SlippageRR,
    apply_long_slippage,
    build_exit_plan,
)
from mtf_signals import mtf_bundle
from free_data import (
    analyze_liquidity,
    extract_quality_extras,
    free_data_status,
    get_market_regime,
    get_news_pulse,
    merge_av_overview_into_info,
    multi_horizon_rs,
)
from indicators import enrich
from stock_service import (
    cached_calendar,
    cached_info,
    cache_bucket,
    compute_returns,
    fetch_history,
    normalize_symbol,
)
from trade_plan import analyze_events, calc_position, suggest_lot_size


@dataclass
class SwingHorizonPlan:
    """短线波段单周期计划：0–2周 或 2–4周。"""

    key: str  # h1 | h2
    label: str  # 0–2周 | 2–4周
    bars: int  # 交易日路径窗口
    verdict: str  # 可以入場 | 可以試倉 | 暫緩觀望 | 不做多
    win_rate_pct: float | None
    entry_low: float | None
    entry_high: float | None
    entry_plan: float | None
    stop_loss: float | None
    target: float | None
    rr: float | None
    expectancy_r: float | None
    risk_per_share: float | None
    reward_per_share: float | None
    note: str = ""
    # 滑点后可执行赔率
    rr_net: float | None = None
    expectancy_net: float | None = None
    slip_note: str = ""
    # 路径胜率元数据（避免 nan% / 说明样本）
    win_rate_samples: int | None = None
    win_rate_source: str = ""  # path_hN | day | none
    win_rate_display: str = "样本不足"


@dataclass
class TradeSOP:
    symbol: str
    name: str
    last_price: float | None

    # Core SOP fields user asked for
    enter_ok: str  # 适合入场 | 谨慎试仓 | 观望 | 回避
    enter_score: float  # 0-100
    entry_low: float | None
    entry_high: float | None
    entry_plan: float | None  # recommended limit / mid of zone
    stop_loss: float | None
    target_t1: float | None
    target_t2: float | None
    win_rate_pct: float | None  # estimated path win rate
    win_rate_label: str  # 高 | 中 | 低
    stability_score: float  # 0-100
    stability_label: str  # 高 | 中 | 低

    side: str  # 做多 | 观望 | 偏空
    risk_per_share: float | None
    rr_t1: float | None
    position_shares: int
    position_note: str

    # 短线波段双周期（主展示）
    swing_h1: SwingHorizonPlan | None = None  # 0–2周
    swing_h2: SwingHorizonPlan | None = None  # 2–4周
    trend_note: str = ""  # 走势一句话
    # 主周期选择：h1 | h2 — 决定 enter_ok / 出场卡
    primary_horizon: str = "h1"
    primary_plan: SwingHorizonPlan | None = None
    exit_plan: ExitPlan | None = None
    slip_rr: SlippageRR | None = None

    # What to do now
    actions_now: list[str] = field(default_factory=list)
    actions_wait: list[str] = field(default_factory=list)
    invalidation: str = ""
    checklist: list[dict[str, str]] = field(default_factory=list)

    # Context
    bias: str = "—"
    bias_score: float = 0.0
    opportunity: str = "—"
    risk_level: str = "—"
    max_dd_pct: float | None = None
    ann_vol_pct: float | None = None
    scorecard_stance: str = "—"
    scorecard_total: float | None = None
    summary: str = ""
    period: str = "1y"
    notes: list[str] = field(default_factory=list)

    # Real-invest extras
    expectancy_r: float | None = None  # expected R per trade from path WR + R:R
    regime_label: str = "—"
    regime_score: float | None = None
    regime_summary: str = ""
    regime_bullets: list[str] = field(default_factory=list)
    liquidity_label: str = "—"
    liquidity_score: float | None = None
    multi_rs_summary: str = ""
    quality_notes: list[str] = field(default_factory=list)
    news_summary: str = ""
    data_sources: list[str] = field(default_factory=list)
    scorecard_bullets: list[str] = field(default_factory=list)
    # Edge signals (sector RS / volume / IV)
    sector_rs_summary: str = ""
    sector_rs_score: float | None = None
    sector_rs_label: str = "—"
    volume_confirm_summary: str = ""
    volume_confirm_score: float | None = None
    volume_confirm_label: str = "—"
    iv_summary: str = ""
    iv_score: float | None = None
    iv_label: str = "—"
    iv_high_event: bool = False
    false_break_summary: str = ""
    false_break_score: float | None = None
    false_break_label: str = "—"
    false_break_risk: bool = False
    trend_align_summary: str = ""
    trend_align_score: float | None = None
    trend_align_label: str = "—"
    against_trend: bool = False
    weekly_label: str = "—"
    weekly_summary: str = ""
    weekly_allow_long: bool = True
    weekly_turning_bullish: bool = False
    weekly_turning_note: str = ""
    adx_label: str = "—"
    adx_value: float | None = None
    adx_summary: str = ""
    adx_trending: bool = False
    fib_summary: str = ""
    h1_label: str = "—"
    h1_summary: str = ""
    h1_ready: bool = False
    # 双模式 SOP（A 防守 / B 进攻）
    mode: str = "defensive"  # defensive | aggressive
    mode_label: str = "A 防守版"
    mode_forced: bool = False
    mode_note: str = ""
    win_rate_samples: int | None = None
    win_rate_source: str = ""
    win_rate_display: str = "样本不足"
    risk_units: float = 0.0  # 允许的 R 数：1.0 / 0.5 / 0
    upgrade_1r: bool = False
    upgrade_hits: int = 0
    upgrade_notes: list[str] = field(default_factory=list)
    # 一屏「决策摘要」：1–2 句说明为什么是这个结论
    decision_brief: str = ""
    # 三灯短线 SOP（位置 · 胜率 · 划算）
    position_light: str = "—"  # green | yellow | red
    wr_light: str = "—"
    rr_light: str = "—"
    position_light_note: str = ""
    wr_light_note: str = ""
    rr_light_note: str = ""
    one_liner_reason: str = ""
    plain_card: str = ""
    notional_hkd: float = 5000.0
    pnl_if_win_hkd: float | None = None
    pnl_if_loss_hkd: float | None = None
    earnings_soon: bool = False
    earnings_days_left: int | None = None
    earnings_note: str = ""
    # 支撑 / 阻力（日线结构）
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    support_pct: float | None = None  # vs last, negative
    resistance_pct: float | None = None  # vs last, positive
    resistance_note: str = ""
    support_note: str = ""
    sr_summary: str = ""
    # 上方阻力列表文案，如 "950(强); 980(中)"
    resistance_levels_txt: str = ""
    support_levels_txt: str = ""
    # 若用历史切片重建计划：YYYY-MM-DD；None/空 = 即时（最新 bar）
    as_of: str | None = None


def parse_as_of_date(as_of: Any) -> date | None:
    """Parse buy / snapshot date from date | datetime | 'YYYY-MM-DD'."""
    if as_of is None or as_of == "":
        return None
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    s = str(as_of).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def slice_ohlcv_as_of(hist: pd.DataFrame, as_of: date | datetime | str) -> pd.DataFrame:
    """
    Keep bars with session date <= as_of (inclusive).
    Used to rebuild entry-day E/S/T1/T2/SR without look-ahead.

    Supports yfinance frames with a ``Date`` / ``Datetime`` column (RangeIndex)
    or a DatetimeIndex.
    """
    if hist is None or getattr(hist, "empty", True):
        return hist if hist is not None else pd.DataFrame()
    d = parse_as_of_date(as_of)
    if d is None:
        return hist
    try:
        ts: pd.Series | None = None
        for col in ("Date", "Datetime", "date", "datetime"):
            if col in hist.columns:
                ts = pd.to_datetime(hist[col], utc=False, errors="coerce")
                break
        if ts is None:
            # DatetimeIndex path
            idx = pd.DatetimeIndex(hist.index)
            if len(idx) == 0 or not isinstance(hist.index, pd.DatetimeIndex):
                # RangeIndex without Date column — cannot slice safely
                if not isinstance(hist.index, pd.DatetimeIndex):
                    return hist
            ts = pd.Series(idx, index=hist.index)

        # Normalize to calendar date (US session if tz-aware)
        def _to_d(x: Any) -> date | None:
            if x is None or (isinstance(x, float) and x != x):
                return None
            t = pd.Timestamp(x)
            if pd.isna(t):
                return None
            if t.tzinfo is not None:
                try:
                    t = t.tz_convert("America/New_York")
                except Exception:
                    t = t.tz_localize(None) if t.tzinfo else t
            return t.date()

        bar_dates = ts.map(_to_d)
        mask = bar_dates.map(lambda x: x is not None and x <= d)
        out = hist.loc[mask].copy()
        # Stable integer index after filter (downstream often uses iloc)
        out = out.reset_index(drop=True)
        return out if out is not None else pd.DataFrame()
    except Exception:
        return hist


# ---------------------------------------------------------------------------
# Dual-mode thresholds (spec v1.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeThresholds:
    key: str
    label: str
    wr_full: float
    wr_half: float
    stab_full: float
    stab_half: float
    rr_full: float  # 净 R:R 优先；无净则用纸面
    rr_half: float
    exp_full: float
    exp_half: float
    max_hold_h1: int
    max_hold_h2: int
    default_risk_units_full: float  # 满仓默认 R（进攻版默认 0.5，达加分再升 1）


# 产品宪法：三灯裁决（位置 · 胜率 · 划算）——默认启用
# 已关闭「只看胜率」实验，避免与净 R:R 后置降级互相打架
THREE_LIGHT_SOP = True
ENTRY_BY_WR_ONLY = False  # 废弃实验；保留开关仅兼容旧测试
MIN_PATH_SAMPLES_FOR_RATE = 6  # 至少 6 笔才显示 %（原 1 过宽，实盘易虚高）
RR_YELLOW_FLOOR = 0.80  # 净 R:R 黄灯下限；低于此 = 红灯不划算
DEFAULT_NOTIONAL_HKD = 5000.0
# E/S/T 可执行结构（Phase 2）
ZONE_ENTRY_FRAC = 0.35  # E_plan = 区下沿 + 35%×区宽（中下挂单，不追区上沿现价）
MIN_RR_TARGET_K = 1.0  # T >= E + k×(E−S)，保证至少 1:1
STOP_ATR_CAP = 1.5  # 止损距离上限（ATR 倍数）
STOP_ATR_FLOOR = 0.6  # 止损距离下限（ATR 倍数）

# 部署指纹：Streamlit Cloud 侧栏应显示同一字串；否则仍是旧代码
# v1-realism-opt：時間窗口自動閘門 + 周線偏空硬擋 + 作息友善
SOP_BUILD = "v1-realism-2026-08-23-wturn2"

# 時間窗口硬規則開關（True = 窗口外強制觀望）
ENFORCE_US_OPEN_FIRST_2H = True

MODE_THRESHOLDS: dict[str, ModeThresholds] = {
    "defensive": ModeThresholds(
        key="defensive",
        label="A 防守版",
        wr_full=54.0,  # 原 52 → 满仓门槛
        wr_half=50.0,  # 原 48 → 试仓也要 ≥50%
        stab_full=48.0,
        stab_half=42.0,
        rr_full=1.15,  # 原 1.10
        rr_half=1.00,  # 原 0.95
        exp_full=0.15,
        exp_half=0.05,
        max_hold_h1=12,
        max_hold_h2=20,
        default_risk_units_full=1.0,
    ),
    "aggressive": ModeThresholds(
        key="aggressive",
        label="B 进攻版",
        wr_full=52.0,  # 原 50
        wr_half=48.0,  # 原 45
        stab_full=30.0,
        stab_half=25.0,
        rr_full=1.05,
        rr_half=0.95,
        exp_full=0.10,
        exp_half=0.02,
        max_hold_h1=10,
        max_hold_h2=15,
        default_risk_units_full=0.5,
    ),
}


def get_mode_thresholds(mode: str) -> ModeThresholds:
    key = (mode or "defensive").strip().lower()
    if key in ("a", "防守", "防守版", "def", "defence", "defense"):
        key = "defensive"
    elif key in ("b", "进攻", "进攻版", "進攻", "進攻版", "agg", "attack"):
        key = "aggressive"
    return MODE_THRESHOLDS.get(key, MODE_THRESHOLDS["defensive"])


def is_us_open_first_2h(now: datetime | None = None) -> tuple[bool, str]:
    """
    判斷當下是否在美股常規交易時段開市後的首 2 小時。

    Returns:
        (is_allowed, note)
    時間換算（香港時間）：
      夏令 EDT：21:30 – 23:30 HKT
      冬令 EST：22:30 – 00:30 HKT
    """
    try:
        hkt = ZoneInfo("Asia/Hong_Kong")
        et = ZoneInfo("US/Eastern")
    except Exception:
        return True, "時區模組不可用，跳過時間窗口檢查"

    if now is None:
        now = datetime.now(hkt)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=hkt)
    else:
        now = now.astimezone(hkt)

    et_now = now.astimezone(et)

    if et_now.weekday() >= 5:
        return False, f"週末（ET {et_now.strftime('%a %H:%M')}），非操作時段"

    open_t = time(9, 30)
    end_t = time(11, 30)

    t = et_now.time()
    if open_t <= t <= end_t:
        return True, f"在允許窗口內（ET {et_now.strftime('%H:%M')}，開市後首2小時）"

    if t < open_t:
        return False, f"尚未開市（ET 現在 {et_now.strftime('%H:%M')}，開市 09:30）"
    return False, f"已過開市首2小時（ET 現在 {et_now.strftime('%H:%M')}，窗口至 11:30）"




# Path WR quality tiers（优化后）
PATH_LOOKBACK_DEFAULT = 180
MIN_SAMPLES_FULL = 15  # 可支撑「可以入場」（原 12）
MIN_SAMPLES_LOW = 10  # 显示胜率，最多试仓（原 8）
MIN_SAMPLES_BLEND = 7  # 与 day_wr 混合的最低路径样本（原 5）


def format_win_rate(
    win_rate: float | None,
    samples: int | None = None,
    *,
    min_samples: int = MIN_SAMPLES_FULL,
    confidence: str | None = None,
    source: str = "",
) -> str:
    """UI-safe win-rate string — never ``nan%``."""
    src = (source or "").lower()
    conf = (confidence or "").lower()
    if conf not in ("full", "low", "day", "blend", "none", ""):
        conf = ""
    # Infer confidence from source/samples when not passed
    if not conf:
        if "day" in src and "blend" not in src and "path" not in src:
            conf = "day"
        elif "blend" in src:
            conf = "blend"
        elif "low" in src or (
            samples is not None and MIN_SAMPLES_LOW <= int(samples) < MIN_SAMPLES_FULL
        ):
            conf = "low"
        elif win_rate is not None:
            conf = "full"

    if win_rate is None:
        floor = MIN_SAMPLES_LOW
        if samples is not None and samples < floor:
            return f"样本不足（<{floor}）"
        if samples is not None and samples < min_samples:
            return f"样本不足（<{min_samples}）"
        return "样本不足"
    try:
        wr = float(win_rate)
    except (TypeError, ValueError):
        return "样本不足"
    if math.isnan(wr) or math.isinf(wr):
        return "样本不足"
    if conf == "day":
        return f"{wr:.0f}%（日线估算）"
    if conf == "blend":
        nbit = f"·n{int(samples)}" if samples else ""
        return f"{wr:.0f}%（路径+日线{nbit}）"
    if conf == "low" or (
        samples is not None and MIN_SAMPLES_LOW <= int(samples) < MIN_SAMPLES_FULL
    ):
        return f"{wr:.0f}%（低样本{int(samples)}）"
    if samples is not None and samples > 0:
        return f"{wr:.0f}%（样本{int(samples)}）"
    return f"{wr:.0f}%"


def _path_win_rate(
    close: pd.Series,
    risk_per_share: float,
    reward_per_share: float,
    lookback: int = PATH_LOOKBACK_DEFAULT,
    horizon: int = 15,
    min_samples: int = MIN_SAMPLES_FULL,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    ref_entry: float | None = None,
    scale: str = "pct",
) -> float | None:
    """
    Historical path win rate: from each past bar, long at close;
    win if +reward hit before -risk within horizon bars.
    Timeout paths (neither hit) are skipped and do not count as samples.
    Default scale=pct uses risk/reward as % of ref_entry on every historical bar.
    """
    wr, _n = _path_win_rate_detail(
        close,
        risk_per_share,
        reward_per_share,
        lookback=lookback,
        horizon=horizon,
        min_samples=min_samples,
        high=high,
        low=low,
        ref_entry=ref_entry,
        scale=scale,
    )
    return wr


def _align_ohlc(
    close: pd.Series,
    high: pd.Series | None,
    low: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    c = close.astype(float).values
    n = len(c)
    h_arr = l_arr = None
    if high is not None and low is not None and len(high) == n and len(low) == n:
        try:
            h_arr = high.astype(float).values
            l_arr = low.astype(float).values
            if np.all(np.isnan(h_arr)) or np.all(np.isnan(l_arr)):
                h_arr = l_arr = None
        except Exception:
            h_arr = l_arr = None
    return c, h_arr, l_arr


def _path_win_rate_detail(
    close: pd.Series,
    risk_per_share: float,
    reward_per_share: float,
    lookback: int = PATH_LOOKBACK_DEFAULT,
    horizon: int = 15,
    min_samples: int = MIN_SAMPLES_LOW,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    ref_entry: float | None = None,
    scale: str = "pct",
) -> tuple[float | None, int]:
    """
    Return (win_rate_pct or None, effective_sample_count).

    scale:
      - ``pct``: risk/reward as fraction of *current* structure entry, applied
        proportionally on each historical close (fixes abs-$ on old prices).
      - ``abs``: legacy fixed $ stop/target distance.
    High/Low used when provided (same-bar both sides → stop first, conservative).
    """
    if close is None or len(close) < max(40, horizon + 25):
        return None, 0
    if risk_per_share <= 0 or reward_per_share <= 0:
        return None, 0

    c, h_arr, l_arr = _align_ohlc(close, high, low)
    n = len(c)
    # need some history; soft lookback if series shorter than lookback+horizon
    lb = min(int(lookback), max(30, n - horizon - 5))
    if lb < 25:
        return None, 0

    # % of current structure (prefer mid entry)
    ref = float(ref_entry) if ref_entry and ref_entry > 0 else float(c[-1])
    if ref <= 0:
        return None, 0
    risk_pct = float(risk_per_share) / ref
    reward_pct = float(reward_per_share) / ref
    # sanity clamps — extreme structure still measurable
    risk_pct = float(min(0.30, max(0.004, risk_pct)))
    reward_pct = float(min(0.50, max(0.004, reward_pct)))

    wins = 0
    total = 0
    start = max(20, n - lb - horizon)
    end = n - horizon
    use_hl = h_arr is not None and l_arr is not None
    use_pct = (scale or "pct").lower() != "abs"

    for i in range(start, end):
        entry = float(c[i])
        if entry <= 0 or math.isnan(entry):
            continue
        if use_pct:
            stop = entry * (1.0 - risk_pct)
            target = entry * (1.0 + reward_pct)
        else:
            stop = entry - float(risk_per_share)
            target = entry + float(reward_per_share)
        if stop >= entry or target <= entry:
            continue

        hit_t = hit_s = False
        ambiguous = False
        for j in range(1, horizon + 1):
            idx = i + j
            if use_hl:
                lo = float(l_arr[idx])
                hi = float(h_arr[idx])
                if math.isnan(lo) or math.isnan(hi):
                    px = float(c[idx])
                    if math.isnan(px):
                        continue
                    if px <= stop:
                        hit_s = True
                        break
                    if px >= target:
                        hit_t = True
                        break
                    continue
                # 日线无法知道先触哪边：同根既触止损又触目标 → 整条路径作 timeout（中性）
                # （旧逻辑「一律算止损」会系统性压低胜率）
                stop_hit = lo <= stop
                tgt_hit = hi >= target
                if stop_hit and tgt_hit:
                    ambiguous = True
                    break
                if stop_hit:
                    hit_s = True
                    break
                if tgt_hit:
                    hit_t = True
                    break
            else:
                px = float(c[idx])
                if math.isnan(px):
                    continue
                if px <= stop:
                    hit_s = True
                    break
                if px >= target:
                    hit_t = True
                    break
        if ambiguous or (not hit_t and not hit_s):
            if ambiguous:
                continue  # 同 bar 歧义不计入（避免武断）
            # timeout：计 0.4 胜（保守折中，避免全丢压低/抬高失真）
            total += 1
            wins += 0.4
            continue
        total += 1
        if hit_t:
            wins += 1

    if total < min_samples:
        return None, total
    return 100.0 * wins / total, total


def resolve_path_win_rate(
    close: pd.Series,
    risk_per_share: float,
    reward_per_share: float,
    *,
    primary_horizon: int = 15,
    day_wr: float | None = None,
    lookback: int = PATH_LOOKBACK_DEFAULT,
    min_samples: int = MIN_PATH_SAMPLES_FOR_RATE,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    ref_entry: float | None = None,
) -> tuple[float | None, int | None, str]:
    """
    技术路径胜率（% 缩放 + High/Low）：
      1) 各 horizon 取有效结算样本最多的路径胜率（默认 n≥1 即给出 %）
      2) 仍无路径时 fallback 日线上涨胜率 day_wr
      3) 都无 → None

    样本数只写入 source/返回值供显示；是否入場由调用方按胜率%门檻决定（可忽略样本）。
    """
    horizons: list[int] = []
    for h in (int(primary_horizon), 15, 20, 30):
        if h > 0 and h not in horizons:
            horizons.append(h)

    best: tuple[float, int, int] | None = None  # wr, n, h
    last_n = 0
    floor = max(1, int(min_samples or MIN_PATH_SAMPLES_FOR_RATE))

    for h in horizons:
        wr, n = _path_win_rate_detail(
            close,
            risk_per_share,
            reward_per_share,
            lookback=lookback,
            horizon=h,
            min_samples=floor,
            high=high,
            low=low,
            ref_entry=ref_entry,
            scale="pct",
        )
        last_n = max(last_n, n)
        if wr is None:
            continue
        # 优先样本更多的窗口；样本相同取胜率更高者
        if best is None or n > best[1] or (n == best[1] and wr > best[0]):
            best = (float(wr), int(n), h)

    if best is not None:
        wr, n, h = best
        tag = f"path_h{h}"
        if n < MIN_SAMPLES_FULL:
            tag = f"{tag}_n{n}"  # 标注样本少，但不阻止用 %
        return round(wr, 1), n, tag

    day_v: float | None = None
    if day_wr is not None:
        try:
            d = float(day_wr)
            if not math.isnan(d) and not math.isinf(d):
                day_v = d
        except (TypeError, ValueError):
            day_v = None

    if day_v is not None:
        return round(day_v, 1), None, "day"

    return None, (last_n if last_n > 0 else None), "none"


def _light_label(light: str) -> str:
    return {"green": "绿", "yellow": "黄", "red": "红"}.get(light, light)


def compute_position_light(
    *,
    last: float | None,
    entry_low: float | None,
    entry_high: float | None,
    price_far_chase: bool,
    entry_opp: str,
) -> tuple[str, str]:
    """位置灯：绿=可挂/区内；黄=略高；红=追高。"""
    if price_far_chase or entry_opp == "不宜追高":
        return "red", "现价远离入場区（追高），先别追"
    if last is None or entry_low is None or entry_high is None:
        return "yellow", "入場区不完整，谨慎"
    lo, hi, px = float(entry_low), float(entry_high), float(last)
    if lo > hi:
        lo, hi = hi, lo
    if lo <= px <= hi:
        return "green", "现价在入場区内，可以挂限价"
    if px < lo:
        return "green", "现价低于入場区，可把限价挂在区内等回补"
    # above zone
    pct = (px / hi - 1.0) * 100.0 if hi > 0 else 0.0
    if pct <= 3.0:
        return "yellow", f"现价略高于区上沿约 {pct:.1f}%，最多试仓、勿追"
    return "red", f"现价高于区上沿约 {pct:.1f}%（追高），先别做"


def compute_wr_light(
    wr: float | None,
    thr: ModeThresholds,
    *,
    samples: int | None = None,
) -> tuple[str, str]:
    """胜率灯。"""
    if wr is None:
        return "red", "算不出路径胜率（样本/结构不足）"
    # 低样本强制保守（实盘勿轻信）
    if samples is not None and samples < MIN_SAMPLES_LOW:
        return "red", f"路径胜率样本不足（仅{int(samples)}笔），暂缓"
    note_n = f"，样本{int(samples)}" if samples else ""
    if wr >= thr.wr_full:
        return "green", f"历史路径胜率约 {wr:.0f}%（达满仓线 {thr.wr_full:.0f}%{note_n}）"
    if wr >= thr.wr_half:
        return "yellow", f"历史路径胜率约 {wr:.0f}%（仅试仓线 {thr.wr_half:.0f}%{note_n}）"
    return "red", f"历史路径胜率约 {wr:.0f}%（低于试仓线 {thr.wr_half:.0f}%{note_n}）"


def compute_rr_light(
    rr_net: float | None,
    thr: ModeThresholds,
    *,
    rr_paper: float | None = None,
) -> tuple[str, str]:
    """划算灯（净 R:R）。"""
    rr = rr_net if rr_net is not None else rr_paper
    if rr is None:
        return "red", "算不出盈亏比（止损/目标不完整）"
    if rr >= thr.rr_full:
        return "green", f"净R:R约 {rr:.2f}（划算，满仓线≥{thr.rr_full:.2f}）"
    if rr >= RR_YELLOW_FLOOR:
        return "yellow", f"净R:R约 {rr:.2f}（偏紧，最多试仓）"
    return "red", f"净R:R约 {rr:.2f}（不划算：目标太近或止损太远）"


def hkd_pnl_space(
    entry: float | None,
    stop: float | None,
    target: float | None,
    notional_hkd: float = DEFAULT_NOTIONAL_HKD,
) -> tuple[float | None, float | None]:
    """按固定落单金额估算：赚到目标 / 打到止损 大约多少 HKD。"""
    if entry is None or entry <= 0:
        return None, None
    e = float(entry)
    win_hkd = loss_hkd = None
    if target is not None and float(target) > e:
        win_hkd = round(notional_hkd * (float(target) - e) / e, 1)
    if stop is not None and float(stop) < e:
        loss_hkd = round(notional_hkd * (e - float(stop)) / e, 1)
    return win_hkd, loss_hkd


def plan_limit_from_zone(
    entry_low: float | None,
    entry_high: float | None,
    *,
    frac: float = ZONE_ENTRY_FRAC,
) -> float | None:
    """计划限价 E_plan：区中下，避免用区上沿现价把 R:R 算爆。"""
    if entry_low is None or entry_high is None:
        return None
    lo, hi = float(entry_low), float(entry_high)
    if lo > hi:
        lo, hi = hi, lo
    if hi <= lo:
        return round(lo, 4)
    f = min(0.9, max(0.05, float(frac)))
    return round(lo + f * (hi - lo), 4)


def cap_stop_by_atr(
    entry: float | None,
    stop: float | None,
    atr: float | None,
    *,
    cap_mult: float = STOP_ATR_CAP,
    floor_mult: float = STOP_ATR_FLOOR,
    prefer_structure: bool = True,  # 只收紧过宽；默认不强制放宽结构止蚀
) -> tuple[float | None, str]:
    """止损距离用 ATR 上下限夹住：太宽收紧、太近略放。

    prefer_structure=True（默认）：只在「止蚀过宽」时收紧，不把结构止蚀强制放宽。
    """
    if entry is None or stop is None:
        return stop, ""
    e, s = float(entry), float(stop)
    if s >= e:
        return stop, ""
    if atr is None or atr <= 0:
        return round(s, 4), ""
    atr = float(atr)
    risk = e - s
    max_risk = cap_mult * atr
    min_risk = floor_mult * atr
    notes: list[str] = []
    if risk > max_risk + 1e-9:
        s = e - max_risk
        notes.append(f"止损过宽→按{cap_mult:.1f}×ATR收紧至{s:.2f}")
    # 过近：仅 prefer_structure=False 才强制放宽（保留结构止蚀）
    elif (not prefer_structure) and (e - s < min_risk - 1e-9):
        s = e - min_risk
        notes.append(f"止损过近→按{floor_mult:.1f}×ATR放宽至{s:.2f}")
    return round(s, 4), "；".join(notes)


def ensure_min_rr_target(
    entry: float | None,
    stop: float | None,
    target: float | None,
    *,
    k: float = MIN_RR_TARGET_K,
) -> tuple[float | None, str]:
    """目标至少 E + k×风险，避免 T 贴 E 导致假高胜率/净R:R≈0。"""
    if entry is None or stop is None:
        return target, ""
    e, s = float(entry), float(stop)
    risk = e - s
    if risk <= 0:
        return target, ""
    min_t = e + float(k) * risk
    if target is None or float(target) < min_t - 1e-9:
        return round(min_t, 4), f"目标过近→抬至至少{k:.1f}:1（{min_t:.2f}）"
    return round(float(target), 4), ""


def order_targets_near_far(
    t1: float | None,
    t2: float | None,
) -> tuple[float | None, float | None, str]:
    """
    保证 T1（近）≤ T2（远）。
    结构/1:1 抬升后，短周期目标可能被抬得比中期还高，需重排。
    """
    a, b = (
        float(t1) if t1 is not None else None,
        float(t2) if t2 is not None else None,
    )
    if a is None and b is None:
        return None, None, ""
    if a is None:
        return round(b, 4), None, ""
    if b is None:
        return round(a, 4), None, ""
    if a <= b + 1e-9:
        return round(a, 4), round(b, 4), ""
    # 颠倒：近的做 T1，远的做 T2
    return round(b, 4), round(a, 4), f"已重排目标：T1={b:.2f}（近）≤ T2={a:.2f}（远）"


def decide_three_lights(
    *,
    thr: ModeThresholds,
    last: float | None,
    entry_low: float | None,
    entry_high: float | None,
    entry_plan: float | None,
    stop: float | None,
    target: float | None,
    wr: float | None,
    wr_samples: int | None,
    rr_net: float | None,
    rr_paper: float | None,
    price_far_chase: bool,
    entry_opp: str,
    bias_label: str,
    bias_score: float,
    vol_label: str = "",
    false_break_risk: bool = False,
    against_trend: bool = False,
    weekly_allow_long: bool = True,
    weekly_turning_bullish: bool = False,
    notional_hkd: float = DEFAULT_NOTIONAL_HKD,
    earnings_days_left: int | None = None,
    enforce_time_window: bool = False,
) -> dict[str, Any]:
    """
    三灯裁决：位置 · 胜率 · 划算 → 可以入場 | 可以試倉 | 暫緩觀望 | 不做多
    财报窗口（Yahoo 日历）：≤3 天强制暂缓新仓；≤14 天最多试仓。
    """

    # ========== 時間窗口硬閘門（最高優先）==========
    if ENFORCE_US_OPEN_FIRST_2H and enforce_time_window:
        allowed, time_note = is_us_open_first_2h()
        if not allowed:
            return {
                "verdict": "暫緩觀望",
                "position_light": "red",
                "wr_light": "red",
                "rr_light": "red",
                "position_light_note": time_note,
                "wr_light_note": "非操作時段",
                "rr_light_note": "非操作時段",
                "one_liner_reason": time_note,
                "plain_card": f"**暫緩觀望**（時間窗口）。{time_note}。窗口外不新開倉，已持倉只執行預設止蝕/T1。",
                "pnl_if_win_hkd": None,
                "pnl_if_loss_hkd": None,
                "notional_hkd": float(notional_hkd),
                "hard_no": time_note,
                "caps": ["非操作時段"],
            }

    pos_l, pos_n = compute_position_light(
        last=last,
        entry_low=entry_low,
        entry_high=entry_high,
        price_far_chase=price_far_chase,
        entry_opp=entry_opp,
    )
    wr_l, wr_n = compute_wr_light(wr, thr, samples=wr_samples)
    rr_l, rr_n = compute_rr_light(rr_net, thr, rr_paper=rr_paper)
    win_hkd, loss_hkd = hkd_pnl_space(
        entry_plan or last, stop, target, notional_hkd=notional_hkd
    )

    # 硬覆盖：方向/量能
    hard_no: str | None = None
    if entry_opp == "偏空回避" or "强烈看空" in (bias_label or ""):
        hard_no = "方向强烈偏空，不做多"
    elif "看空" in (bias_label or "") and bias_score <= -25:
        hard_no = "方向看空且偏弱，不做多"
    elif vol_label == "放量下跌":
        hard_no = "放量下跌，今天不做多"
    elif not weekly_allow_long and not weekly_turning_bullish:
        # 周线空头且未开始转多 → 硬挡
        hard_no = "周线偏空，强制观望"
    elif not weekly_allow_long and weekly_turning_bullish:
        # 仍标空头，但已开始转多 → 不硬挡，最多试仓
        hard_no = None

    # 盖帽：最多试仓（黄）
    caps: list[str] = []
    if false_break_risk:
        caps.append("假突破风险")
    if against_trend:
        caps.append("逆势环境")
    if not weekly_allow_long and weekly_turning_bullish:
        caps.append("周线空头但开始转多")
    # 财报：≤3 天不新开；≤14 天最多试仓（免费 Yahoo 日历，已有）
    earn_block = False
    earn_msg = ""
    if earnings_days_left is not None and 0 <= int(earnings_days_left) <= 14:
        d = int(earnings_days_left)
        if d <= 3:
            earn_block = True
            earn_msg = f"财报约 {d} 天内：跳空风险高，暂缓开新仓"
        else:
            caps.append(f"财报约 {d} 天内")
            earn_msg = f"财报约 {d} 天内：最多试仓，避免扛隔夜"

    if hard_no:
        verdict = "不做多"
        one = hard_no
    elif earn_block:
        verdict = "暫緩觀望"
        one = earn_msg
    else:
        lights = (pos_l, wr_l, rr_l)
        if "red" in lights:
            verdict = "暫緩觀望"
            # 主因：优先红灯
            if pos_l == "red":
                one = pos_n
            elif rr_l == "red":
                one = rr_n
            else:
                one = wr_n
        elif all(x == "green" for x in lights) and not caps:
            verdict = "可以入場"
            one = "位置、胜率、划算三灯都绿，可按计划限价做"
        else:
            verdict = "可以試倉"
            if caps:
                one = "；".join(caps) + " → 最多试仓，勿满仓"
            elif "yellow" in lights:
                bits = []
                if pos_l == "yellow":
                    bits.append(pos_n)
                if wr_l == "yellow":
                    bits.append(wr_n)
                if rr_l == "yellow":
                    bits.append(rr_n)
                one = "；".join(bits) if bits else "条件未全绿，最多试仓"
            else:
                one = "条件未全绿，最多试仓"
        if earn_msg and verdict == "可以試倉" and "财报" not in one:
            one = earn_msg + "；" + one

    # 白话卡
    v_human = {
        "可以入場": "可以入場",
        "可以試倉": "可以試倉（半仓心态）",
        "暫緩觀望": "先别做",
        "不做多": "不做多",
    }.get(verdict, verdict)
    win_s = f"约 +{win_hkd:.0f} HKD" if win_hkd is not None else "—"
    loss_s = f"约 −{loss_hkd:.0f} HKD" if loss_hkd is not None else "—"
    fair = {
        "green": "还算划算",
        "yellow": "勉强",
        "red": "不划算",
    }.get(rr_l, "—")
    plain = (
        f"【结论】{v_human}\n\n"
        f"【三句话】\n"
        f"1. 位置（{_light_label(pos_l)}）：{pos_n}\n"
        f"2. 胜率（{_light_label(wr_l)}）：{wr_n}\n"
        f"3. 划算吗（{_light_label(rr_l)}）：若赚到目标 {win_s}，"
        f"若止损 {loss_s}（按每笔 {notional_hkd:.0f} HKD）→ {fair}\n\n"
        f"【主因】{one}\n\n"
        f"【可以做的】"
    )
    if verdict in ("可以入場", "可以試倉"):
        plain += (
            f" 限价挂在 {entry_low}–{entry_high}；"
            f"止损 {stop}；目标 {target}。"
            + (" 只用试仓量。" if verdict == "可以試倉" else "")
        )
    else:
        plain += " 今天不新开多单；可等价格回到入場区且赔率改善再看。"

    return {
        "verdict": verdict,
        "position_light": pos_l,
        "wr_light": wr_l,
        "rr_light": rr_l,
        "position_light_note": pos_n,
        "wr_light_note": wr_n,
        "rr_light_note": rr_n,
        "one_liner_reason": one,
        "plain_card": plain,
        "pnl_if_win_hkd": win_hkd,
        "pnl_if_loss_hkd": loss_hkd,
        "notional_hkd": float(notional_hkd),
    }


def path_wr_confidence(source: str, samples: int | None = None) -> str:
    """Map source/samples → full | low | day | blend | none."""
    src = (source or "").lower()
    if not src or src == "none":
        return "none"
    if src == "day":
        return "day"
    if "blend" in src:
        return "blend"
    if src.endswith("_low") or "low" in src:
        return "low"
    if samples is not None and samples < MIN_SAMPLES_FULL:
        return "low"
    if src.startswith("path"):
        return "full"
    return "none"


def build_decision_brief(
    *,
    verdict: str,
    mode_label: str,
    wr_display: str,
    wr_confidence: str = "none",
    rr_net: float | None = None,
    rr_paper: float | None = None,
    exp_r: float | None = None,
    risk_units: float = 0.0,
    bias_label: str = "",
    bias_score: float = 0.0,
    price_far_chase: bool = False,
    false_break_risk: bool = False,
    against_trend: bool = False,
    weekly_allow_long: bool = True,
    vol_label: str = "",
    mode_forced: bool = False,
    mode_note: str = "",
    thr: ModeThresholds | None = None,
) -> str:
    """
    一屏决策摘要：结论 + 主要根据 + 主风险（1–3 短句，中文）。
    """
    thr = thr or MODE_THRESHOLDS["defensive"]
    v = verdict or "暫緩觀望"
    conf = (wr_confidence or "none").lower()

    # —— 结论句 ——
    if v in ("可以入場", "适合入场"):
        lead = f"**可以入場**（{mode_label} · 允许约 {risk_units:g}R）"
    elif v in ("可以試倉", "谨慎试仓"):
        lead = f"**可以試倉**（{mode_label} · 固定约 {max(risk_units, 0.5):g}R）"
    elif v in ("不做多", "回避"):
        lead = f"**不做多**（{mode_label}）"
    else:
        lead = f"**暫緩觀望**（{mode_label}）"

    # —— 根据（胜率 / R:R / E[R]）——
    reasons: list[str] = []
    wr_bit = wr_display or "样本不足"
    if conf == "full":
        reasons.append(f"路径胜率 {wr_bit}（样本充足）")
    elif conf == "low":
        reasons.append(f"路径胜率 {wr_bit}（低样本，撑不满仓）")
    elif conf == "blend":
        reasons.append(f"胜率 {wr_bit}（路径+日线混合）")
    elif conf == "day":
        reasons.append(f"胜率 {wr_bit}（仅日线估算）")
    else:
        reasons.append("路径胜率样本不足，不用高胜率背书")

    rr_use = rr_net if rr_net is not None else rr_paper
    if rr_use is not None:
        tag = "净R:R" if rr_net is not None else "纸面R:R"
        ok = "达标" if rr_use >= thr.rr_half else "偏弱"
        if rr_use >= thr.rr_full:
            ok = "良好"
        reasons.append(f"{tag} {rr_use:.2f}（{ok}，满仓线≥{thr.rr_full:.2f}）")
    else:
        reasons.append("R:R 暂不可算")

    if exp_r is not None:
        if exp_r >= thr.exp_full:
            reasons.append(f"E[R] {exp_r:+.2f}（正期望）")
        elif exp_r >= 0:
            reasons.append(f"E[R] {exp_r:+.2f}（弱正/边缘）")
        else:
            reasons.append(f"E[R] {exp_r:+.2f}（负期望，不利开仓）")

    if bias_label:
        reasons.append(f"方向 {bias_label}（{bias_score:+.0f}）")

    why = "根据：" + "；".join(reasons[:4]) + "。"

    # —— 主风险 / 限制 ——
    risks: list[str] = []
    if mode_forced and mode_note:
        risks.append(mode_note)
    if price_far_chase:
        risks.append("现价远离入场区（追高）")
    if false_break_risk:
        risks.append("假突破风险")
    if against_trend:
        risks.append("逆势（大盘/板块偏空）")
    if not weekly_allow_long:
        risks.append("周线偏空")
    if vol_label in ("放量下跌",):
        risks.append("放量下跌")
    if conf in ("low", "day", "blend") and v in ("可以試倉", "谨慎试仓", "可以入場", "适合入场"):
        risks.append("胜率置信度不足，勿加仓到满仓")
    if exp_r is not None and exp_r < 0 and v not in ("不做多", "回避"):
        risks.append("负期望结构")
    if rr_use is not None and rr_use < thr.rr_half:
        risks.append("赔率未过试仓线")

    if v in ("可以入場", "适合入场"):
        tip = "主风险：" + ("；".join(risks[:3]) if risks else "按计划限价，止损硬挂，到 T1 减半。")
    elif v in ("可以試倉", "谨慎试仓"):
        tip = "限制：" + (
            "；".join(risks[:3]) if risks else "半仓试错，不达标不加到 1R。"
        )
    elif v in ("不做多", "回避"):
        tip = "原因偏向：" + (
            "；".join(risks[:3]) if risks else "方向或结构不支持做多。"
        )
    else:
        tip = "暂缓主因：" + (
            "；".join(risks[:3]) if risks else "门檻未齐，等回入場区或环境改善。"
        )

    return f"{lead}。{why}{tip}"


def _stability_score(risk, trend, bias_score: float) -> tuple[float, str]:
    """0-100: higher = calmer / more stable trend quality."""
    score = 50.0
    vol = getattr(risk, "ann_vol_pct", None)
    dd = getattr(risk, "max_drawdown_pct", None)
    sharpe = getattr(risk, "sharpe", None)
    wr = getattr(risk, "win_rate_pct", None)

    if vol is not None:
        # 12% vol ~ +20, 40% vol ~ -15
        score += max(-20.0, min(20.0, (22.0 - float(vol)) * 0.9))
    if dd is not None:
        # dd is negative; -10% good, -40% bad
        score += max(-25.0, min(15.0, (abs(float(dd)) * -0.7) + 12))
    if sharpe is not None:
        score += max(-15.0, min(20.0, float(sharpe) * 8.0))
    if wr is not None:
        score += max(-10.0, min(10.0, (float(wr) - 50.0) * 0.4))

    # choppy trend reduces stability of signal
    st = getattr(trend, "strength_label", "") or ""
    if "强" in st:
        score += 6
    elif "弱" in st or "震荡" in st:
        score -= 6

    # extreme bias can mean unstable chase
    if abs(bias_score) >= 55:
        score -= 4

    score = float(max(0.0, min(100.0, score)))
    if score >= 68:
        label = "高"
    elif score >= 45:
        label = "中"
    else:
        label = "低"
    return round(score, 1), label


def _expectancy_r(win_rate_pct: float | None, rr: float | None) -> float | None:
    """E[R] = p*R - (1-p)*1  assuming 1R risk, R:R = reward/risk."""
    if win_rate_pct is None or rr is None or rr <= 0:
        return None
    p = float(win_rate_pct) / 100.0
    return round(p * float(rr) - (1.0 - p) * 1.0, 3)


def _bias_ok_for_mode(
    mode: ModeThresholds,
    bias_label: str,
    bias_score: float,
    *,
    full: bool,
) -> bool:
    """Mode-specific bias gates (spec §2)."""
    lab = bias_label or ""
    if "强烈看空" in lab:
        return False
    if full:
        if mode.key == "aggressive":
            return ("强烈看多" in lab) or (bias_score >= 50)
        # defensive: 看多 or 强烈看多
        return ("看多" in lab) or bias_score >= 18
    # trial
    if mode.key == "aggressive":
        return bias_score >= 10 and "看空" not in lab
    return bias_score >= -5 and "强烈看空" not in lab


def _swing_verdict(
    *,
    entry_opp: str,
    bias_label: str,
    bias_score: float,
    wr: float | None,
    rr: float | None,
    exp_r: float | None,
    price_far_chase: bool,
    vol_label: str = "",
    false_break_risk: bool = False,
    block_breakout_chase: bool = False,
    against_trend: bool = False,
    trend_label: str = "",
    trend_score: float | None = None,
    weekly_allow_long: bool = True,
    adx_trending: bool | None = None,
    adx_value: float | None = None,
    h1_ready: bool | None = None,
    mode: ModeThresholds | None = None,
    stability: float | None = None,
    multi_rs_score: float | None = None,
    rr_net: float | None = None,
    wr_confidence: str = "full",
) -> str:
    """
    短线波段结论（双模式门檻）:
      可以入場 | 可以試倉 | 暫緩觀望 | 不做多

    wr_confidence: full | low | day | blend | none
      - full: 样本≥12，可满仓
      - low/blend/day: 最多试仓（有胜率数字但不撑满仓）
      - none: 样本不足，不可开
    """
    thr = mode or MODE_THRESHOLDS["defensive"]
    # 门檻用净 R:R（可执行）；无则退回纸面
    rr_gate = rr_net if rr_net is not None else rr
    wr_conf = (wr_confidence or "full").lower()

    if entry_opp == "偏空回避" or "强烈看空" in bias_label:
        return "不做多"
    if "看空" in bias_label and bias_score <= -25:
        return "不做多"
    if not weekly_allow_long:
        # 統一：周線偏空一律不給新開倉機會（高勝率優先）
        return "暫緩觀望"
    if price_far_chase or entry_opp == "不宜追高":
        return "暫緩觀望"
    if vol_label == "放量下跌":
        return "暫緩觀望"
    # 缩量假突破标记（WR-only 时后面最多试仓，不在这里直接毙）
    shrink_fbo = false_break_risk and vol_label in ("缩量", "缩量回踩", "缩量整理")
    if not ENTRY_BY_WR_ONLY:
        if false_break_risk and thr.key == "defensive":
            return "暫緩觀望"
        if block_breakout_chase and price_far_chase:
            return "暫緩觀望"
        if against_trend and trend_label == "逆势":
            return "暫緩觀望"
        if adx_trending is False and adx_value is not None and adx_value < 18:
            if entry_opp in ("较佳入场",) and block_breakout_chase:
                return "暫緩觀望"
        # 绝对地板：赔率/期望过差
        if rr_gate is not None and rr_gate < 0.90:
            return "暫緩觀望"
        if exp_r is not None and exp_r < -0.08:
            return "暫緩觀望"

    # —— 技术胜率门檻（不因样本 conf 降级）——
    # wr_conf / 样本数仅用于展示；ENTRY_BY_WR_ONLY 时入場只看胜率%
    wr_full_ok = wr is not None and wr >= thr.wr_full
    wr_half_ok = wr is not None and wr >= thr.wr_half

    if ENTRY_BY_WR_ONLY:
        # 主裁决：技术路径胜率%；样本不参与。仍挡已在上方的极端风险。
        if wr is None:
            return "暫緩觀望"  # 完全算不出胜率
        if false_break_risk or shrink_fbo:
            # 假突破：最多试仓
            if wr_half_ok:
                return "可以試倉"
            return "暫緩觀望"
        if not weekly_allow_long:
            if wr_half_ok:
                return "可以試倉"
            return "暫緩觀望"
        if against_trend:
            if wr_half_ok and wr >= thr.wr_full:
                return "可以試倉"  # 逆势即使高胜率也不满仓
            return "暫緩觀望"
        if wr_full_ok:
            return "可以入場"
        if wr_half_ok:
            return "可以試倉"
        return "暫緩觀望"

    # —— 以下：旧多因子门檻（ENTRY_BY_WR_ONLY=False 时）——
    good_setup = entry_opp in ("较佳入场", "可关注")
    rs_ok_full = multi_rs_score is None or multi_rs_score >= (
        48 if thr.key == "defensive" else 35
    )
    rs_ok_half = multi_rs_score is None or multi_rs_score >= (
        40 if thr.key == "defensive" else 28
    )
    stab_ok_full = stability is None or stability >= thr.stab_full
    stab_ok_half = stability is None or stability >= thr.stab_half
    rr_good = rr_gate is not None and rr_gate >= thr.rr_full
    rr_ok = rr_gate is not None and rr_gate >= thr.rr_half
    exp_good = exp_r is not None and exp_r >= thr.exp_full
    exp_ok = exp_r is not None and exp_r >= thr.exp_half
    bias_full = _bias_ok_for_mode(thr, bias_label, bias_score, full=True)
    bias_half = _bias_ok_for_mode(thr, bias_label, bias_score, full=False)
    trend_ok = trend_score is None or trend_score >= 48
    trend_good = trend_score is None or trend_score >= 60
    no_fbo = not false_break_risk
    weekly_ok_full = weekly_allow_long
    if thr.key == "aggressive":
        weekly_ok_full = True
    vol_ok_full = vol_label not in ("放量下跌",) and not (
        thr.key == "defensive" and vol_label in ("缩量假突破",)
    )
    if thr.key == "defensive" and vol_label in ("放量下跌", "缩量破位"):
        vol_ok_full = False

    if (
        good_setup
        and bias_full
        and wr_full_ok
        and rr_good
        and exp_good
        and stab_ok_full
        and rs_ok_full
        and vol_ok_full
        and not price_far_chase
        and not against_trend
        and trend_good
        and no_fbo
        and not block_breakout_chase
        and weekly_ok_full
        and weekly_allow_long
        and (adx_trending is not False or (adx_value is not None and adx_value >= 18))
        and not shrink_fbo
    ):
        return "可以入場"

    if bias_half and wr_half_ok and rr_ok and exp_ok and stab_ok_half and rs_ok_half:
        if entry_opp in ("偏空回避",):
            return "暫緩觀望"
        if against_trend:
            return "暫緩觀望"
        if false_break_risk or shrink_fbo:
            if thr.key == "aggressive" and good_setup and wr_half_ok and rr_ok:
                return "可以試倉"
            return "暫緩觀望"
        if block_breakout_chase and not good_setup:
            pass
        if (good_setup or bias_score >= 10) and trend_ok:
            return "可以試倉"
        if entry_opp in ("观望",) and bias_score >= 18 and trend_ok and wr_half_ok:
            return "可以試倉"
    return "暫緩觀望"


def aggressive_upgrade_1r(
    *,
    wr: float | None,
    bias_label: str,
    bias_score: float,
    multi_rs_score: float | None,
    vol_label: str,
    false_break_risk: bool,
    rr_net: float | None,
    rr: float | None,
) -> tuple[bool, int, list[str]]:
    """
    进攻版升至 1R：至少满足 3 项（spec §5）。
    """
    hits: list[str] = []
    if wr is not None and wr >= 50:
        hits.append("路径胜率≥50%")
    if ("强烈看多" in (bias_label or "")) or bias_score >= 50:
        hits.append("Bias强烈看多/≥+50")
    if multi_rs_score is None or multi_rs_score >= 40:
        # 不得极弱：有分数则 ≥40；无数据视为不否决但也不算加分
        if multi_rs_score is not None and multi_rs_score >= 40:
            hits.append("相对强弱非极弱")
        elif multi_rs_score is None:
            pass
    vol_ok = vol_label not in ("放量下跌",) and not (
        false_break_risk and vol_label in ("缩量", "缩量回踩", "缩量整理")
    )
    if vol_ok and vol_label not in ("", "—"):
        hits.append("量能非缩量假突破")
    rr_use = rr_net if rr_net is not None else rr
    if rr_use is not None and rr_use >= 1.0:
        hits.append("净R:R≥1.0")
    return len(hits) >= 3, len(hits), hits


def _build_swing_plan(
    *,
    key: str,
    label: str,
    bars: int,
    close: pd.Series,
    entry_low: float | None,
    entry_high: float | None,
    entry_mid: float | None,
    display_limit: float | None,
    stop: float | None,
    target: float | None,
    entry_opp: str,
    bias_label: str,
    bias_score: float,
    price_far_chase: bool,
    vol_label: str,
    false_break_risk: bool = False,
    block_breakout_chase: bool = False,
    against_trend: bool = False,
    trend_label: str = "",
    trend_score: float | None = None,
    weekly_allow_long: bool = True,
    weekly_turning_bullish: bool = False,
    adx_trending: bool | None = None,
    adx_value: float | None = None,
    h1_ready: bool | None = None,
    mode: ModeThresholds | None = None,
    stability: float | None = None,
    multi_rs_score: float | None = None,
    day_wr: float | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    last_price: float | None = None,
    notional_hkd: float = DEFAULT_NOTIONAL_HKD,
    earnings_days_left: int | None = None,
    enforce_time_window: bool = False,
) -> SwingHorizonPlan:
    thr = mode or MODE_THRESHOLDS["defensive"]
    risk_ps = None
    reward_ps = None
    mid = entry_mid
    if mid is not None and stop is not None and mid > stop:
        risk_ps = float(mid) - float(stop)
    if mid is not None and target is not None and target > mid:
        reward_ps = float(target) - float(mid)
    rr = None
    if risk_ps and reward_ps and risk_ps > 0:
        rr = reward_ps / risk_ps

    wr: float | None = None
    wr_n: int | None = None
    wr_src = "none"
    wr_conf = "none"
    ref_for_pct = float(mid) if mid is not None else (
        float(display_limit) if display_limit else None
    )
    if risk_ps and reward_ps and close is not None and len(close) > bars + 30:
        wr, wr_n, wr_src = resolve_path_win_rate(
            close,
            risk_ps,
            reward_ps,
            primary_horizon=bars,
            day_wr=day_wr,
            lookback=PATH_LOOKBACK_DEFAULT,
            min_samples=MIN_PATH_SAMPLES_FOR_RATE,
            high=high,
            low=low,
            ref_entry=ref_for_pct,
        )
        wr_conf = path_wr_confidence(wr_src, wr_n)
    wr_display = format_win_rate(wr, wr_n, confidence=wr_conf, source=wr_src)

    # 滑点后 R:R / E[R]（可执行）— 先算，门檻优先用净 R:R
    entry_ref = (
        float(display_limit)
        if display_limit
        else (float(mid) if mid is not None else None)
    )
    slip = apply_long_slippage(
        entry_ref, stop, target, win_rate_pct=wr, slip_pct=DEFAULT_SLIP_PCT
    )
    exp_r = _expectancy_r(wr, rr) if wr is not None else None

    plan_px = entry_ref
    if THREE_LIGHT_SOP:
        tl = decide_three_lights(
            thr=thr,
            last=last_price,
            entry_low=entry_low,
            entry_high=entry_high,
            entry_plan=plan_px,
            stop=stop,
            target=target,
            wr=wr,
            wr_samples=wr_n,
            rr_net=slip.rr_net,
            rr_paper=rr,
            price_far_chase=price_far_chase,
            entry_opp=entry_opp,
            bias_label=bias_label,
            bias_score=bias_score,
            vol_label=vol_label,
            false_break_risk=false_break_risk,
            against_trend=against_trend,
            weekly_allow_long=weekly_allow_long,
            weekly_turning_bullish=weekly_turning_bullish,
            notional_hkd=notional_hkd,
            earnings_days_left=earnings_days_left,
            enforce_time_window=enforce_time_window,
        )
        verdict = tl["verdict"]
        note_bits = [
            f"模式{thr.label}",
            f"三灯 位置{_light_label(tl['position_light'])}/"
            f"胜率{_light_label(tl['wr_light'])}/"
            f"划算{_light_label(tl['rr_light'])}",
            tl["one_liner_reason"],
        ]
    else:
        verdict = _swing_verdict(
            entry_opp=entry_opp,
            bias_label=bias_label,
            bias_score=bias_score,
            wr=wr,
            rr=rr,
            exp_r=exp_r,
            price_far_chase=price_far_chase,
            vol_label=vol_label,
            false_break_risk=false_break_risk,
            block_breakout_chase=block_breakout_chase,
            against_trend=against_trend,
            trend_label=trend_label,
            trend_score=trend_score,
            weekly_allow_long=weekly_allow_long,
            adx_trending=adx_trending,
            adx_value=adx_value,
            h1_ready=h1_ready,
            mode=thr,
            stability=stability,
            multi_rs_score=multi_rs_score,
            rr_net=slip.rr_net,
            wr_confidence=wr_conf,
        )
        note_bits = [f"模式{thr.label}"]
        if wr is not None:
            note_bits.append(f"路径胜率{wr_display}（{wr_src}）")
        else:
            note_bits.append(wr_display)
    if wr is not None and THREE_LIGHT_SOP:
        note_bits.append(f"胜率{wr_display}")
    if rr is not None:
        note_bits.append(f"R:R≈{rr:.2f}")
    if exp_r is not None:
        note_bits.append(f"E[R]≈{exp_r:+.2f}")
    if h1_ready is True:
        note_bits.append("1H可掛單")
    elif h1_ready is False:
        note_bits.append("等1H/回踩")
    if slip.rr_net is not None:
        note_bits.append(f"净R:R≈{slip.rr_net:.2f}")

    return SwingHorizonPlan(
        key=key,
        label=label,
        bars=bars,
        verdict=verdict,
        win_rate_pct=round(wr, 1) if wr is not None else None,
        entry_low=round(float(entry_low), 4) if entry_low else None,
        entry_high=round(float(entry_high), 4) if entry_high else None,
        entry_plan=round(float(display_limit), 4)
        if display_limit
        else (round(float(mid), 4) if mid else None),
        stop_loss=round(float(stop), 4) if stop else None,
        target=round(float(target), 4) if target else None,
        rr=round(rr, 2) if rr is not None else None,
        expectancy_r=exp_r,
        risk_per_share=round(risk_ps, 4) if risk_ps else None,
        reward_per_share=round(reward_ps, 4) if reward_ps else None,
        note=" · ".join(note_bits),
        rr_net=slip.rr_net,
        expectancy_net=slip.exp_net,
        slip_note=slip.note,
        win_rate_samples=wr_n,
        win_rate_source=wr_src,
        win_rate_display=wr_display,
    )


# ---- Legacy soft floors for _enter_decision ranking (score only) ----
# 真正做不做以 swing 双模式门檻为准；下列常数仅用于综合分硬挡极端情况
MIN_RR_FULL = 1.10
MIN_RR_CAUTIOUS = 0.90
MIN_EXP_FULL = 0.10
MIN_EXP_CAUTIOUS = 0.0
MIN_WR_FULL = 50.0
MIN_STAB_FULL = 45.0
MIN_STAB_CAUTIOUS = 22.0
MIN_REGIME_FULL = 48.0
MIN_LIQ_FULL = 42.0
MIN_RS_FULL = 42.0


def resolve_trading_mode(
    requested: str | None,
    *,
    regime_score: float | None = None,
    vix_label: str = "",
    force_defensive: bool = False,
) -> tuple[ModeThresholds, bool, str]:
    """
    Pick A/B mode. Force defensive when market weak / VIX elevated / caller says so.
    Returns (thresholds, forced, note).
    """
    thr = get_mode_thresholds(requested or "defensive")
    notes: list[str] = []
    forced = False
    if force_defensive and thr.key != "defensive":
        thr = MODE_THRESHOLDS["defensive"]
        forced = True
        notes.append("连续亏损/日亏达限：强制 A 防守版")
    high_vix = any(k in (vix_label or "") for k in ("高", "极高", "elevated", "high"))
    if thr.key == "aggressive":
        if regime_score is not None and regime_score < 40:
            thr = MODE_THRESHOLDS["defensive"]
            forced = True
            notes.append("大盘转弱：强制切回 A 防守版")
        elif high_vix:
            thr = MODE_THRESHOLDS["defensive"]
            forced = True
            notes.append("VIX 偏高：强制切回 A 防守版")
    note = "；".join(notes)
    return thr, forced, note


def _enter_decision(
    entry_opp: str,
    entry_score: float,
    bias_label: str,
    bias_score: float,
    stability: float,
    win_rate: float | None,
    rr: float | None,
    risk_level: str,
    *,
    regime_score: float | None = None,
    liquidity_score: float | None = None,
    expectancy_r: float | None = None,
    earnings_soon: bool = False,
    chase_high: bool = False,
    multi_rs_score: float | None = None,
    price_above_zone: bool = False,
    price_far_chase: bool = False,
    sector_rs_score: float | None = None,
    volume_confirm_score: float | None = None,
    iv_score: float | None = None,
    iv_high_event: bool = False,
) -> tuple[str, float, str]:
    """
    Map models → 适合入场 / 谨慎试仓 / 观望 / 回避.

    Soft score is for ranking; **hard gates** protect expectancy:
    bad R:R / negative E[R] / chase / earnings / thin liquidity cannot full-size.
    Edge: sector RS, volume confirm, IV event risk.
    """
    s = 38.0 + entry_score * 0.32
    if entry_opp in ("较佳入场",):
        s += 16
    elif entry_opp in ("可关注",):
        s += 8
    elif entry_opp in ("观望",):
        s -= 5
    elif entry_opp in ("不宜追高",):
        s -= 15
    elif entry_opp in ("偏空回避",):
        s -= 25

    if "强烈看多" in bias_label or bias_label == "看多":
        s += 10 if bias_score > 0 else 0
    if "看空" in bias_label:
        s -= 12
    if "强烈看空" in bias_label:
        s -= 20

    s += (stability - 50) * 0.12
    if win_rate is not None:
        s += (win_rate - 50) * 0.18
    if rr is not None:
        if rr >= 2.0:
            s += 10
        elif rr >= 1.35:
            s += 6
        elif rr >= 1.05:
            s += 2
        elif rr < 1.0:
            s -= 14  # 赔率不足，强烈减分
        elif rr < 0.8:
            s -= 18

    if risk_level in ("高", "极高"):
        s -= 8

    if regime_score is not None:
        s += (float(regime_score) - 50.0) * 0.22
        if regime_score < 35:
            s -= 8
    if liquidity_score is not None:
        s += (float(liquidity_score) - 50.0) * 0.10
        if liquidity_score < 35:
            s -= 10
    if multi_rs_score is not None:
        s += (float(multi_rs_score) - 50.0) * 0.10
    if expectancy_r is not None:
        if expectancy_r >= 0.35:
            s += 10
        elif expectancy_r >= 0.12:
            s += 5
        elif expectancy_r >= 0:
            s += 1
        else:
            s -= 16  # 负期望：长期必亏结构
    if earnings_soon:
        s -= 14
    if chase_high or price_above_zone:
        s -= 10
    if price_far_chase:
        s -= 12

    # Edge signals: sector / volume / IV
    if sector_rs_score is not None:
        s += (float(sector_rs_score) - 50.0) * 0.12
        if sector_rs_score < 38:
            s -= 6
    if volume_confirm_score is not None:
        s += (float(volume_confirm_score) - 50.0) * 0.14
        if volume_confirm_score <= 32:
            s -= 10  # 放量下跌等
        elif volume_confirm_score >= 68:
            s += 4
    if iv_score is not None:
        s += (float(iv_score) - 50.0) * 0.08
    if iv_high_event:
        s -= 8

    s = float(max(0.0, min(100.0, s)))

    # ---- Absolute blocks (protect capital / expectancy) ----
    if entry_opp in ("偏空回避",) or ("强烈看空" in bias_label and entry_score < 50):
        return "回避", min(s, 35.0), "偏空"
    if "看空" in bias_label and bias_score <= -18:
        return "回避", min(s, 38.0), "偏空"
    if liquidity_score is not None and liquidity_score < 28:
        return "观望", min(s, 48.0), "观望"
    # 赔率不足 1：做多数学期望很难正 → 最多观望
    if rr is not None and rr < MIN_RR_CAUTIOUS:
        if s >= 40:
            return "观望", min(s, 52.0), "观望"
        return "回避", min(s, 40.0), "观望" if bias_score >= -15 else "偏空"
    # 负期望：禁止开新多
    if expectancy_r is not None and expectancy_r < MIN_EXP_CAUTIOUS:
        if s >= 40:
            return "观望", min(s, 50.0), "观望"
        return "回避", min(s, 38.0), "观望"
    # 远离入场区追高
    if price_far_chase or entry_opp in ("不宜追高",):
        if s >= 45:
            return "观望", min(s, 50.0), "观望"
        return "回避", min(s, 40.0), "观望"
    if earnings_soon:
        # 财报窗口：永不适合入场；期望仍为正才允许极小试仓
        can_half = (
            s >= 58
            and (rr is None or rr >= MIN_RR_CAUTIOUS)
            and (expectancy_r is None or expectancy_r >= 0.05)
            and not price_far_chase
        )
        if can_half:
            return "谨慎试仓", min(s, 62.0), "做多"
        if s >= 40:
            return "观望", min(s, 52.0), "观望"
        return "回避", min(s, 40.0), "观望"
    if regime_score is not None and regime_score < 32:
        if s >= 50:
            return "观望", min(s, 55.0), "观望"
        return "回避", min(s, 40.0), "观望" if bias_score >= -15 else "偏空"

    # ---- 适合入场：全部硬条件 ----
    full_ok = (
        s >= 74
        and entry_opp in ("较佳入场", "可关注")
        and bias_score >= -5
        and "看空" not in bias_label
        and (rr is not None and rr >= MIN_RR_FULL)
        and (expectancy_r is not None and expectancy_r >= MIN_EXP_FULL)
        and (win_rate is None or win_rate >= MIN_WR_FULL)
        and stability >= MIN_STAB_FULL
        and (regime_score is None or regime_score >= MIN_REGIME_FULL)
        and (liquidity_score is None or liquidity_score >= MIN_LIQ_FULL)
        and (multi_rs_score is None or multi_rs_score >= MIN_RS_FULL)
        and (sector_rs_score is None or sector_rs_score >= 42)
        and (volume_confirm_score is None or volume_confirm_score >= 40)
        and not iv_high_event
        and not earnings_soon
        and not chase_high
        and not price_above_zone
        and not price_far_chase
        and risk_level not in ("极高",)
    )
    if full_ok:
        return "适合入场", s, "做多"

    # ---- 谨慎试仓：允许稍差，但赔率/期望不能穿底 ----
    cautious_ok = (
        s >= 55
        and entry_opp not in ("偏空回避", "不宜追高")
        and bias_score >= -12
        and "强烈看空" not in bias_label
        and (rr is None or rr >= MIN_RR_CAUTIOUS)
        and (expectancy_r is None or expectancy_r >= MIN_EXP_CAUTIOUS)
        and stability >= MIN_STAB_CAUTIOUS
        and not price_far_chase
        and (liquidity_score is None or liquidity_score >= 32)
        and (regime_score is None or regime_score >= 38)
        and (volume_confirm_score is None or volume_confirm_score >= 28)
        # 放量下跌禁止试仓
        and not (volume_confirm_score is not None and volume_confirm_score < 28)
    )
    if cautious_ok:
        return "谨慎试仓", min(s, 72.0) if not full_ok else s, "做多"

    if s >= 40:
        return "观望", s, "观望"
    return "回避", s, "观望" if bias_score >= -15 else "偏空"


def build_trade_sop(
    symbol: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    capital: float = 50_000.0,
    risk_pct: float = 1.0,
    primary_horizon: str = "h1",
    mode: str = "defensive",
    force_defensive: bool = False,
    include_h1: bool = True,
    as_of: date | datetime | str | None = None,
) -> TradeSOP:
    """
    Full SOP for one symbol (short-term swing playbook).

    primary_horizon: ``h1`` = 0–2周（默认）, ``h2`` = 2–4周
    — 决定 enter_ok / 出场纪律 / 主展示计划。

    mode: ``defensive`` (A 防守版) | ``aggressive`` (B 进攻版)
    force_defensive: 连续亏损等场景由 UI/日志强制 A
    include_h1: 是否取 1H 数据作短线入场确认；扫描预筛选可关闭。
    as_of: 若提供，仅用该日及之前的日线重建计划（入场日锁定 E/S/T/阻力），
           不含未来 K 线；1H/新闻等「今日实时」信号会跳过。
    """
    sym = normalize_symbol(symbol)
    as_of_d = parse_as_of_date(as_of)
    as_of_str = as_of_d.isoformat() if as_of_d else None
    # 历史切片需要更长窗口，保证 as_of 前仍有足够 lookback
    fetch_period = period
    if as_of_d is not None and period in ("6mo", "1y", "ytd", "3mo"):
        fetch_period = "2y"
    info = cached_info(sym, cache_bucket(5))
    hist = fetch_history(sym, period=fetch_period, interval=interval)
    name = (
        (info or {}).get("shortName")
        or (info or {}).get("longName")
        or sym
    )

    if hist is not None and not hist.empty and as_of_d is not None:
        hist = slice_ohlcv_as_of(hist, as_of_d)

    if hist is None or hist.empty or "Close" not in hist.columns:
        return TradeSOP(
            symbol=sym,
            name=str(name),
            last_price=None,
            enter_ok="观望",
            enter_score=0,
            entry_low=None,
            entry_high=None,
            entry_plan=None,
            stop_loss=None,
            target_t1=None,
            target_t2=None,
            win_rate_pct=None,
            win_rate_label="—",
            stability_score=0,
            stability_label="—",
            side="观望",
            risk_per_share=None,
            rr_t1=None,
            position_shares=0,
            position_note="无行情数据",
            actions_now=["检查代码 / 网络 / 稍后再试"],
            summary=(
                f"无法拉取 {as_of_str} 及以前的历史数据，SOP 不可用。"
                if as_of_str
                else "无法拉取历史数据，SOP 不可用。"
            ),
            period=period,
            mode=get_mode_thresholds(mode).key,
            mode_label=get_mode_thresholds(mode).label,
            win_rate_display="样本不足",
            as_of=as_of_str,
        )

    df = enrich(hist)
    last = float(df["Close"].iloc[-1])
    bias = analyze_bias(df)
    entry = analyze_entry(df)
    risk = analyze_risk(df)
    trend = analyze_trend(df)
    vol = analyze_volume(df)
    sr = analyze_support_resistance(df)
    rets = compute_returns(df)

    # Free market regime + optional Alpha Vantage (funda fill + news sentiment)
    # 历史 as_of：跳过「今日」宏观/新闻，避免用未来信息污染入场日计划
    regime = None
    news = None
    if as_of_d is None:
        try:
            regime = get_market_regime()
        except Exception:
            regime = None

    info_use = dict(info or {})
    av_filled: list[str] = []
    if as_of_d is None:
        try:
            info_use, av_filled = merge_av_overview_into_info(info_use, sym)
        except Exception:
            av_filled = []

    targets = analyze_targets(df, info=info_use, entry=entry)
    funda = analyze_fundamentals(info_use)
    liq = analyze_liquidity(df, info_use)
    quality = extract_quality_extras(info_use, last_price=last)
    if av_filled:
        quality.notes.append(
            f"Alpha Vantage 补全 {len(av_filled)} 项基本面字段（仅填 yfinance 缺失）"
        )
    if as_of_d is None:
        try:
            news = get_news_pulse(sym)
        except Exception:
            news = None

    # Soft note from news sentiment (Alpha Vantage)
    if news and getattr(news, "available", False) and news.sentiment_score is not None:
        sc = float(news.sentiment_score)
        if sc <= -0.20:
            quality.notes.append(f"新闻情绪偏空（AV {sc:+.2f}）：消息面逆风，宜减仓/观望")
        elif sc >= 0.25:
            quality.notes.append(f"新闻情绪偏多（AV {sc:+.2f}）：注意是否已定价/追高")

    bench_sym, bench_label = default_benchmark(sym)
    bench = None
    try:
        bench = fetch_history(bench_sym, period=fetch_period, interval=interval)
        if as_of_d is not None and bench is not None and not bench.empty:
            bench = slice_ohlcv_as_of(bench, as_of_d)
        rs = analyze_relative_strength(
            hist, bench, benchmark=bench_sym, bench_label=bench_label
        )
    except Exception:
        rs = None
    multi_rs = multi_horizon_rs(hist, bench) if bench is not None else {
        "score": None,
        "summary": "",
        "label": "—",
    }
    card = build_scorecard(
        bias.score,
        funda,
        risk,
        rs,
        regime_score=getattr(regime, "score", None) if regime else None,
        liquidity_score=liq.score,
        multi_rs_score=multi_rs.get("score"),
    )

    try:
        cal = cached_calendar(sym, cache_bucket(30))
        events = analyze_events(info_use, cal)
    except Exception:
        events = None

    earnings_soon = bool(getattr(events, "near_earnings", False)) if events else False
    earnings_days_left: int | None = None
    earnings_note = ""
    if events is not None:
        for it in getattr(events, "items", []) or []:
            if getattr(it, "name", "") != "财报日":
                continue
            dleft = getattr(it, "days_left", None)
            if dleft is not None:
                try:
                    earnings_days_left = int(dleft)
                except (TypeError, ValueError):
                    earnings_days_left = None
            earnings_note = str(getattr(it, "detail", "") or "")
            if earnings_days_left is not None and 0 <= earnings_days_left <= 14:
                earnings_soon = True
            break
        if not earnings_note and getattr(events, "caution", None):
            earnings_note = str(events.caution)

    chase_high = bool(
        quality.fifty_two_week_pct is not None and quality.fifty_two_week_pct >= 92
    )

    e_low = entry.suggested_entry_low
    e_high = entry.suggested_entry_high
    stop = entry.stop_loss

    # 多周期：周线过滤 + ADX + Fib  refinement + 1H 触发
    # as_of 历史切片：只用当日已收盘日线做 ADX/Fib，跳过会拉「今日」周线/1H 的接口
    weekly = adx_r = fib_r = h1_r = None
    fib_note = ""
    try:
        if as_of_d is not None:
            from mtf_signals import analyze_adx, analyze_fib_levels, merge_entry_with_fib

            adx_r = analyze_adx(df)
            fib_r = analyze_fib_levels(df)
            e_low, e_high, fib_note = merge_entry_with_fib(e_low, e_high, fib_r)
            fib_note = str(fib_note or "")
        else:
            mtf = mtf_bundle(sym, df, e_low, e_high, include_h1=include_h1)
            weekly = mtf.get("weekly")
            adx_r = mtf.get("adx")
            fib_r = mtf.get("fib")
            h1_r = mtf.get("h1")
            fib_note = str(mtf.get("fib_note") or "")
            if mtf.get("entry_low") is not None and mtf.get("entry_high") is not None:
                e_low = mtf["entry_low"]
                e_high = mtf["entry_high"]
    except Exception:
        pass

    # —— E/S/T 可执行结构 ——
    # E_plan：区中下挂单（不用区上沿现价算 R:R）
    structure_notes: list[str] = []
    atr_v: float | None = None
    try:
        if "ATR" in df.columns and len(df) > 0:
            atr_v = float(df["ATR"].iloc[-1])
            if atr_v != atr_v or atr_v <= 0:  # NaN
                atr_v = None
    except Exception:
        atr_v = None

    if e_low and e_high:
        zone_mid = round((float(e_low) + float(e_high)) / 2.0, 4)
        entry_plan = plan_limit_from_zone(e_low, e_high, frac=ZONE_ENTRY_FRAC)
        if entry_plan is None:
            entry_plan = zone_mid
        in_zone = float(e_low) <= last <= float(e_high)
        # 挂单始终推荐 E_plan；现价仅用于位置灯 / 是否可立刻成交
        display_limit = float(entry_plan)
        structure_notes.append(
            f"计划限价E={entry_plan:.2f}（区中下{ZONE_ENTRY_FRAC:.0%}），"
            f"现价={last:.2f}仅判断追不追"
        )
    else:
        zone_mid = round(last, 4)
        entry_plan = zone_mid
        display_limit = zone_mid
        in_zone = True

    # 现价相对入场区
    price_above_zone = False
    price_far_chase = False
    if e_high is not None and last > 0:
        eh = float(e_high)
        if last > eh * 1.02:
            price_above_zone = True
        if last > eh * 1.045:
            price_far_chase = True

    # S：相对 E_plan 用 ATR 夹住
    stop_raw = stop
    stop, stop_adj = cap_stop_by_atr(
        entry_plan, stop, atr_v, cap_mult=STOP_ATR_CAP, floor_mult=STOP_ATR_FLOOR
    )
    if stop_adj:
        structure_notes.append(stop_adj)

    weekly_allow = True if weekly is None else bool(getattr(weekly, "allow_long", True))
    weekly_turning = bool(getattr(weekly, "turning_bullish", False)) if weekly else False
    weekly_turn_note = str(getattr(weekly, "turning_note", "") or "") if weekly else ""
    adx_trending = getattr(adx_r, "trending", None) if adx_r else None
    adx_val = getattr(adx_r, "adx", None) if adx_r else None
    h1_ready = getattr(h1_r, "ready", None) if h1_r else None

    # 短线目标：0–2周用超短/短期；2–4周用短期/中期
    t_ultra = getattr(getattr(targets, "ultra", None), "bull_target", None)
    t_short = getattr(getattr(targets, "short", None), "bull_target", None)
    t_med = getattr(getattr(targets, "medium", None), "bull_target", None)
    t1 = t_ultra or t_short  # 0–2周目标
    t2 = t_short or t_med  # 2–4周目标
    if t2 is not None and t1 is not None and float(t2) < float(t1):
        t2 = t_med or t2

    # T：至少 E + k×风险（结构目标更远则保留）
    t1, t1_adj = ensure_min_rr_target(
        entry_plan, stop, t1, k=MIN_RR_TARGET_K
    )
    t2, t2_adj = ensure_min_rr_target(
        entry_plan, stop, t2, k=max(MIN_RR_TARGET_K, 1.2)
    )
    if t1_adj:
        structure_notes.append("T1" + t1_adj)
    if t2_adj and t2_adj != t1_adj:
        structure_notes.append("T2" + t2_adj)
    # 强制 T1（近）≤ T2（远）—— 避免 1:1 抬升后短目标反超中期
    t1, t2, t_ord = order_targets_near_far(t1, t2)
    if t_ord:
        structure_notes.append(t_ord)

    # 结构赔率 / 路径一律相对 E_plan（计划限价）
    risk_ps = None
    if entry_plan and stop and entry_plan > stop:
        risk_ps = float(entry_plan) - float(stop)
    reward_ps = None
    if entry_plan and t1 and t1 > entry_plan:
        reward_ps = float(t1) - float(entry_plan)

    rr = None
    if risk_ps and reward_ps and risk_ps > 0:
        rr = reward_ps / risk_ps

    # 路径用已收盘日线（去掉未完成当日 bar）+ High/Low 触价
    # as_of 历史日：最后一根已是当日收盘，不必再丢
    path_df = df
    if as_of_d is None and len(path_df) >= 40:
        try:
            from market_session import us_session_clock

            sess = us_session_clock().session
            if sess in ("pre_market", "rth", "after_hours", "overnight"):
                path_df = path_df.iloc[:-1]
        except Exception:
            pass
    close_for_path = path_df["Close"]
    high_for_path = path_df["High"] if "High" in path_df.columns else None
    low_for_path = path_df["Low"] if "Low" in path_df.columns else None

    # 路径胜率 + fallback（% 缩放 / HL / 低样本分档）— 全局摘要用 0–2 周
    day_wr = getattr(risk, "win_rate_pct", None)
    wr_samples: int | None = None
    wr_source = "none"
    path_wr = None
    if risk_ps and reward_ps:
        path_wr, wr_samples, wr_source = resolve_path_win_rate(
            close_for_path,
            risk_ps,
            reward_ps,
            primary_horizon=10,
            day_wr=day_wr,
            lookback=PATH_LOOKBACK_DEFAULT,
            min_samples=MIN_PATH_SAMPLES_FOR_RATE,
            high=high_for_path,
            low=low_for_path,
            ref_entry=float(entry_plan) if entry_plan else None,
        )
    win_rate = path_wr  # 已含 day / blend fallback
    wr_conf_global = path_wr_confidence(wr_source, wr_samples)
    wr_display = format_win_rate(
        win_rate, wr_samples, confidence=wr_conf_global, source=wr_source
    )

    if win_rate is not None:
        if win_rate >= 58:
            wr_label = "高"
        elif win_rate >= 48:
            wr_label = "中"
        else:
            wr_label = "低"
    else:
        wr_label = "样本不足"

    exp_r = _expectancy_r(win_rate, rr)

    # Edge: sector RS + volume confirm + IV regime
    try:
        edges = edge_bundle(sym, df, info_use, period=period)
        sector_rs = edges["sector_rs"]
        vol_cf = edges["volume"]
        iv_reg = edges["iv"]
        fbo = edges.get("false_break")
        trend_al = edges.get("trend_align")
    except Exception:
        sector_rs = None
        vol_cf = None
        iv_reg = None
        fbo = None
        trend_al = None

    vol_label = getattr(vol_cf, "label", "") if vol_cf else ""
    fbo_risk = bool(getattr(fbo, "false_break_risk", False)) if fbo else False
    fbo_block = bool(getattr(fbo, "block_breakout_chase", False)) if fbo else False
    against_tr = bool(getattr(trend_al, "against_trend", False)) if trend_al else False
    trend_lab = getattr(trend_al, "label", "") if trend_al else ""
    trend_sc = getattr(trend_al, "score", None) if trend_al else None

    stab, stab_label = _stability_score(risk, trend, bias.score)

    # 双模式：A 防守 / B 进攻（大盘弱/VIX 高强制 A）
    mode_thr, mode_forced, mode_note = resolve_trading_mode(
        mode,
        regime_score=getattr(regime, "score", None) if regime else None,
        vix_label=str(getattr(regime, "vix_label", "") or "") if regime else "",
        force_defensive=force_defensive,
    )

    # 双周期短线计划（主输出）— 门檻按 mode
    _mtf_kw = dict(
        false_break_risk=fbo_risk,
        block_breakout_chase=fbo_block,
        against_trend=against_tr,
        trend_label=trend_lab,
        trend_score=trend_sc,
        weekly_allow_long=weekly_allow,
        weekly_turning_bullish=weekly_turning,
        adx_trending=adx_trending,
        adx_value=adx_val,
        h1_ready=h1_ready,
        mode=mode_thr,
        stability=stab,
        multi_rs_score=multi_rs.get("score"),
        day_wr=float(day_wr) if day_wr is not None else None,
        high=high_for_path,
        low=low_for_path,
        last_price=float(last) if last is not None else None,
        notional_hkd=DEFAULT_NOTIONAL_HKD,
        earnings_days_left=earnings_days_left,
        enforce_time_window=as_of is None,
    )
    swing_h1 = _build_swing_plan(
        key="h1",
        label="0–2周",
        bars=10,
        close=close_for_path,
        entry_low=e_low,
        entry_high=e_high,
        entry_mid=entry_plan,
        display_limit=display_limit,
        stop=stop,
        target=t1,
        entry_opp=entry.opportunity,
        bias_label=bias.bias,
        bias_score=bias.score,
        price_far_chase=price_far_chase,
        vol_label=vol_label,
        **_mtf_kw,
    )
    swing_h2 = _build_swing_plan(
        key="h2",
        label="2–4周",
        bars=20,
        close=close_for_path,
        entry_low=e_low,
        entry_high=e_high,
        entry_mid=entry_plan,
        display_limit=display_limit,
        stop=stop,
        target=t2,
        entry_opp=entry.opportunity,
        bias_label=bias.bias,
        bias_score=bias.score,
        price_far_chase=price_far_chase,
        vol_label=vol_label,
        **_mtf_kw,
    )
    trend_note = (
        f"模式 {mode_thr.label}"
        + (f"（强制）" if mode_forced else "")
        + f" · 走势：{bias.bias}（{bias.score:+.0f}）· 入场 {entry.opportunity}· "
        f"周线 {getattr(weekly, 'label', '—') if weekly else '—'}· "
        f"ADX {getattr(adx_r, 'label', '—') if adx_r else '—'}· "
        f"1H {getattr(h1_r, 'label', '—') if h1_r else '—'}· "
        f"跟势 {trend_lab or '—'}· 假突破 {getattr(fbo, 'label', '—') if fbo else '—'}"
    )
    # 综合风控分仍计算；最终「做不做」以短线 0–2 周波段结论为准（你的持仓周期）
    _, enter_score, _ = _enter_decision(
        entry.opportunity,
        entry.score,
        bias.bias,
        bias.score,
        stab,
        win_rate,
        rr,
        getattr(risk, "risk_level", "—") or "—",
        regime_score=getattr(regime, "score", None) if regime else None,
        liquidity_score=liq.score,
        expectancy_r=exp_r,
        earnings_soon=earnings_soon,
        chase_high=chase_high,
        multi_rs_score=multi_rs.get("score"),
        price_above_zone=price_above_zone,
        price_far_chase=price_far_chase,
        sector_rs_score=getattr(sector_rs, "score", None) if sector_rs else None,
        volume_confirm_score=getattr(vol_cf, "score", None) if vol_cf else None,
        iv_score=getattr(iv_reg, "score", None) if iv_reg else None,
        iv_high_event=bool(getattr(iv_reg, "high_event_risk", False)) if iv_reg else False,
    )
    # 主周期：决定最终做不做、出场卡、滑点 R:R
    ph = (primary_horizon or "h1").lower().strip()
    if ph not in ("h1", "h2"):
        ph = "h1"
    primary = swing_h1 if ph == "h1" else swing_h2
    if primary is None:
        primary = swing_h1

    _map_ok = {
        "可以入場": ("适合入场", "做多"),
        "可以試倉": ("谨慎试仓", "做多"),
        "暫緩觀望": ("观望", "观望"),
        "不做多": ("回避", "偏空"),
    }
    enter_ok, side = _map_ok.get(primary.verdict, ("观望", "观望"))
    # 三灯已在 swing 裁决中含赔率；不再二次用 rr_net 偷偷降级（避免规则打架）

    # 主周期三灯（最终结论与白话卡；在仓位计算前定稿）
    tl_main = decide_three_lights(
        thr=mode_thr,
        last=float(last) if last is not None else None,
        entry_low=primary.entry_low or e_low,
        entry_high=primary.entry_high or e_high,
        entry_plan=primary.entry_plan or display_limit or entry_plan,
        stop=primary.stop_loss or stop,
        target=primary.target or t1,
        wr=primary.win_rate_pct,
        wr_samples=getattr(primary, "win_rate_samples", None),
        rr_net=primary.rr_net,
        rr_paper=primary.rr,
        price_far_chase=bool(price_far_chase),
        entry_opp=entry.opportunity,
        bias_label=bias.bias,
        bias_score=float(bias.score),
        vol_label=vol_label or "",
        false_break_risk=bool(fbo_risk),
        against_trend=bool(against_tr),
        weekly_allow_long=bool(weekly_allow),
        weekly_turning_bullish=weekly_turning,
        notional_hkd=DEFAULT_NOTIONAL_HKD,
        earnings_days_left=earnings_days_left,
        enforce_time_window=as_of is None,
    )
    if THREE_LIGHT_SOP:
        primary.verdict = tl_main["verdict"]
        enter_ok, side = _map_ok.get(primary.verdict, ("观望", "观望"))

    # 分数：用波段可操作性映射，方便扫描排序
    enter_score = {
        "适合入场": max(enter_score, 78.0),
        "谨慎试仓": max(min(enter_score, 72.0), 58.0),
        "观望": min(enter_score, 52.0),
        "回避": min(enter_score, 35.0),
    }.get(enter_ok, enter_score)

    # 出场纪律（主周期）— 持有天数按模式
    max_hold = (
        mode_thr.max_hold_h1 if primary.key == "h1" else mode_thr.max_hold_h2
    )
    exit_pl = build_exit_plan(
        horizon_key=primary.key,
        horizon_label=primary.label,
        max_hold_days=max_hold,
        entry=primary.entry_plan or entry_plan,
        stop=primary.stop_loss or stop,
        t1=primary.target,
        t2=(swing_h2.target if primary.key == "h1" else None),
        scale_out_pct=0.50,
    )
    slip_main = apply_long_slippage(
        primary.entry_plan or entry_plan,
        primary.stop_loss or stop,
        primary.target,
        win_rate_pct=primary.win_rate_pct,
        slip_pct=DEFAULT_SLIP_PCT,
    )

    # 仓位 R 数：防守满仓默认 1R；进攻满仓默认 0.5R，≥3 项加分可升 1R
    upgrade_1r = False
    upgrade_hits = 0
    upgrade_notes: list[str] = []
    risk_units = 0.0
    if enter_ok == "适合入场":
        if mode_thr.key == "aggressive":
            upgrade_1r, upgrade_hits, upgrade_notes = aggressive_upgrade_1r(
                wr=primary.win_rate_pct,
                bias_label=bias.bias,
                bias_score=bias.score,
                multi_rs_score=multi_rs.get("score"),
                vol_label=vol_label,
                false_break_risk=fbo_risk,
                rr_net=primary.rr_net,
                rr=primary.rr,
            )
            risk_units = 1.0 if upgrade_1r else mode_thr.default_risk_units_full
        else:
            risk_units = mode_thr.default_risk_units_full  # 1.0
    elif enter_ok == "谨慎试仓":
        risk_units = 0.5
    else:
        risk_units = 0.0

    lot = suggest_lot_size(sym)
    plan_entry = float(display_limit if in_zone else (entry_plan or last))
    plan_stop = float(stop or last * 0.97)
    if risk_units >= 0.99:
        eff_risk_pct = float(risk_pct)
        risk_tag = f"1R·{mode_thr.label}"
    elif risk_units >= 0.4:
        eff_risk_pct = float(risk_pct) * 0.5
        risk_tag = f"0.5R·{mode_thr.label}"
    else:
        eff_risk_pct = float(risk_pct)
        risk_tag = f"参考1R(SOP不建议开)·{mode_thr.label}"
    pos = calc_position(
        capital=capital,
        risk_pct=eff_risk_pct,
        entry_price=plan_entry,
        stop_price=plan_stop,
        short_target=float(t1) if t1 else None,
        medium_target=float(t2) if t2 else None,
        lot_size=lot,
    )
    pos_note = ""
    if not pos.valid:
        pos_note = pos.error or "仓位无效"
    elif pos.shares <= 0:
        pos_note = "；".join(pos.notes) if pos.notes else "风险预算不足 1 股"
    else:
        pos_note = (
            f"{risk_tag} · 约 {pos.shares} 股 · 仓位市值 ${pos.position_value:,.0f} "
            f"· 占本金 {pos.position_pct_of_capital:.1f}%（本金按 HKD→USD 折算）"
        )
        if enter_ok == "谨慎试仓":
            pos_note += " · 半仓试错，到 T1 先减半"
        if mode_thr.key == "aggressive" and enter_ok == "适合入场":
            if upgrade_1r:
                pos_note += f" · 进攻升1R（{upgrade_hits}/5：{'、'.join(upgrade_notes)}）"
            else:
                pos_note += (
                    f" · 进攻默认0.5R（加分{upgrade_hits}/5，需≥3才升1R）"
                )
        if mode_forced and mode_note:
            pos_note += f" · {mode_note}"

    # Checklist
    checklist: list[dict[str, str]] = []
    checklist.append(
        {
            "name": "入场评级",
            "status": "pass"
            if entry.opportunity in ("较佳入场", "可关注")
            else ("warn" if entry.opportunity == "观望" else "fail"),
            "detail": f"{entry.opportunity}（{entry.score:.0f}分）· {entry.side_bias}",
        }
    )
    checklist.append(
        {
            "name": "多空方向",
            "status": "pass"
            if bias.score >= 18
            else ("warn" if abs(bias.score) < 18 else "fail"),
            "detail": f"{bias.bias}（{bias.score:+.0f}）置信度 {bias.confidence}",
        }
    )
    checklist.append(
        {
            "name": "稳定度",
            "status": "pass"
            if stab >= 68
            else ("warn" if stab >= 45 else "fail"),
            "detail": f"{stab_label}（{stab:.0f}/100）· 风险等级 {getattr(risk, 'risk_level', '—')}",
        }
    )
    checklist.append(
        {
            "name": f"模式",
            "status": "pass" if not mode_forced else "warn",
            "detail": (
                f"{mode_thr.label}（{mode_thr.key}）"
                + (f" · {mode_note}" if mode_note else "")
                + f" · 满仓胜率≥{mode_thr.wr_full:.0f}% · 试仓≥{mode_thr.wr_half:.0f}%"
                + f" · 净R:R≥{mode_thr.rr_full:.2f}/{mode_thr.rr_half:.2f}"
            ),
        }
    )
    prim_wr = primary.win_rate_pct
    prim_src = getattr(primary, "win_rate_source", "") or wr_source
    prim_n = getattr(primary, "win_rate_samples", None)
    prim_conf = path_wr_confidence(prim_src, prim_n)
    prim_disp = getattr(primary, "win_rate_display", None) or format_win_rate(
        prim_wr, prim_n, confidence=prim_conf, source=prim_src
    )
    if prim_wr is not None:
        if prim_conf == "full" and prim_wr >= mode_thr.wr_full:
            wr_status = "pass"
        elif prim_wr >= mode_thr.wr_half:
            wr_status = "warn"  # 低样本/day/blend 或未达满仓线
        else:
            wr_status = "fail"
        conf_tip = {
            "full": "样本充足可支撑满仓",
            "low": "低样本：最多试仓",
            "blend": "路径+日线混合：最多试仓",
            "day": "日线估算：最多试仓",
        }.get(prim_conf, "")
        checklist.append(
            {
                "name": "路径胜率",
                "status": wr_status,
                "detail": (
                    f"{prim_disp} · 来源 {prim_src}"
                    f" · %缩放+High/Low · {conf_tip}"
                ),
            }
        )
    else:
        checklist.append(
            {
                "name": "路径胜率",
                "status": "fail",
                "detail": f"{prim_disp} · 不得按高胜率开仓",
            }
        )
    rr_show = primary.rr_net if primary.rr_net is not None else (
        primary.rr if primary.rr is not None else rr
    )
    if rr_show is not None:
        checklist.append(
            {
                "name": "盈亏比 R:R（净优先）",
                "status": "pass"
                if rr_show >= mode_thr.rr_full
                else ("warn" if rr_show >= mode_thr.rr_half else "fail"),
                "detail": (
                    f"≈ {rr_show:.2f} · 满仓净R:R≥{mode_thr.rr_full:.2f} · "
                    f"试仓≥{mode_thr.rr_half:.2f}"
                    + (
                        " · 不足则禁止开多"
                        if rr_show < mode_thr.rr_half
                        else ""
                    )
                ),
            }
        )
    checklist.append(
        {
            "name": "稳定度（模式门檻）",
            "status": "pass"
            if stab >= mode_thr.stab_full
            else ("warn" if stab >= mode_thr.stab_half else "fail"),
            "detail": (
                f"{stab_label}（{stab:.0f}/100）· "
                f"满仓≥{mode_thr.stab_full:.0f} · 试仓≥{mode_thr.stab_half:.0f}"
            ),
        }
    )
    if price_above_zone or price_far_chase:
        checklist.append(
            {
                "name": "价格 vs 入场区",
                "status": "fail" if price_far_chase else "warn",
                "detail": (
                    "现价明显高于入场区上沿：追高会压缩赔率"
                    + (" · 已禁止试仓" if price_far_chase else " · 不得满仓")
                ),
            }
        )
    if events is not None and getattr(events, "caution", None):
        checklist.append(
            {
                "name": "事件风险",
                "status": "fail" if earnings_soon else "warn",
                "detail": str(events.caution),
            }
        )
    if regime is not None:
        rstat = (
            "pass"
            if regime.score >= 60
            else ("warn" if regime.score >= 40 else "fail")
        )
        checklist.append(
            {
                "name": "市场环境",
                "status": rstat,
                "detail": f"{regime.label}（{regime.score:.0f}）· VIX {regime.vix_label}"
                + (f" {regime.vix:.1f}" if regime.vix is not None else "")
                + f" · SPY {regime.spy_trend}",
            }
        )
    checklist.append(
        {
            "name": "流动性",
            "status": "pass"
            if liq.score >= 65
            else ("warn" if liq.score >= 40 else "fail"),
            "detail": liq.summary.replace("**", ""),
        }
    )
    if exp_r is not None:
        checklist.append(
            {
                "name": "期望值 E[R]",
                "status": "pass" if exp_r >= 0.15 else ("warn" if exp_r >= 0 else "fail"),
                "detail": f"约 {exp_r:+.2f}R / 笔（用胜率×盈亏比估算，含1R亏损假设）",
            }
        )
    if multi_rs.get("score") is not None:
        checklist.append(
            {
                "name": "多周期强弱",
                "status": "pass"
                if multi_rs["score"] >= 58
                else ("warn" if multi_rs["score"] >= 42 else "fail"),
                "detail": multi_rs.get("summary", ""),
            }
        )
    if sector_rs is not None and getattr(sector_rs, "available", False):
        sc = sector_rs.score
        checklist.append(
            {
                "name": "板块相对强弱",
                "status": "pass"
                if sc is not None and sc >= 58
                else ("warn" if sc is not None and sc >= 42 else "fail"),
                "detail": sector_rs.summary,
            }
        )
    if vol_cf is not None and getattr(vol_cf, "available", False):
        checklist.append(
            {
                "name": "量能确认",
                "status": "pass"
                if vol_cf.score >= 62
                else ("warn" if vol_cf.score >= 40 else "fail"),
                "detail": vol_cf.summary,
            }
        )
    if iv_reg is not None and getattr(iv_reg, "available", False):
        checklist.append(
            {
                "name": "IV 环境",
                "status": "fail"
                if iv_reg.high_event_risk
                else ("pass" if iv_reg.score >= 55 else "warn"),
                "detail": iv_reg.summary,
            }
        )
    if fbo is not None and getattr(fbo, "available", False):
        checklist.append(
            {
                "name": "假突破过滤",
                "status": "fail"
                if fbo.false_break_risk
                else (
                    "warn"
                    if fbo.block_breakout_chase
                    else ("pass" if fbo.score >= 60 else "warn")
                ),
                "detail": fbo.summary,
            }
        )
    if trend_al is not None and getattr(trend_al, "available", False):
        checklist.append(
            {
                "name": "跟势(SPY/板块)",
                "status": "fail"
                if trend_al.against_trend
                else ("pass" if trend_al.score >= 60 else "warn"),
                "detail": trend_al.summary,
            }
        )
    if weekly is not None and getattr(weekly, "available", False):
        checklist.append(
            {
                "name": "周线过滤",
                "status": "pass"
                if weekly.allow_long
                else "fail",
                "detail": weekly.summary,
            }
        )
    if adx_r is not None and getattr(adx_r, "available", False):
        checklist.append(
            {
                "name": "ADX趋势强度",
                "status": "pass"
                if adx_r.trending
                else "warn",
                "detail": adx_r.summary,
            }
        )
    if fib_r is not None and getattr(fib_r, "available", False):
        checklist.append(
            {
                "name": "Fib回撤区",
                "status": "pass" if "Fib回撤区" in (fib_r.label or "") else "warn",
                "detail": fib_r.summary + (f" · {fib_note}" if fib_note else ""),
            }
        )
    if h1_r is not None and getattr(h1_r, "available", False):
        checklist.append(
            {
                "name": "1H触发",
                "status": "pass" if h1_r.ready else "warn",
                "detail": h1_r.summary,
            }
        )

    # Actions
    actions_now: list[str] = []
    actions_wait: list[str] = []

    if enter_ok == "适合入场":
        actions_now.append(
            f"限价只在 {e_low:.2f}–{e_high:.2f} 成交"
            if e_low and e_high
            else f"参考限价 ≈ {entry_plan:.2f}（勿市价追）"
        )
        if pos.shares > 0:
            actions_now.append(
                f"按 1R={risk_pct:.1f}% 下单约 {pos.shares} 股（本金 USD {capital:,.0f}）"
            )
        else:
            actions_now.append("先调本金/风险% 或收紧止损，使股数 ≥ 1")
        actions_now.append(f"止损硬挂 {stop:.2f}" if stop else "必须先设止损")
        if t1:
            actions_now.append(f"T1={t1:.2f}：到价减仓 ≥50%，把止损移到成本附近")
        if t2:
            actions_now.append(f"剩余仓位看 T2={t2:.2f}；结构破坏则清")
        if exp_r is not None and rr is not None:
            actions_now.append(f"本结构期望约 {exp_r:+.2f}R · R:R {rr:.2f}（符合满仓门槛）")
        actions_now.append("成交后写日志：理由 / 止损 / 目标 / 是否按计划")
    elif enter_ok == "谨慎试仓":
        actions_now.append(
            f"只用 0.5R（约 {pos.shares} 股）" if pos.shares > 0 else "0.5R 预算不足 1 股：缩小止损距或加本金"
        )
        if e_low and e_high:
            actions_now.append(f"限价只挂 {e_low:.2f}–{e_high:.2f}，上方绝不追")
        actions_now.append(f"止损 {stop:.2f} 必须预设" if stop else "先定止损再考虑")
        if exp_r is not None and rr is not None:
            actions_now.append(f"期望 {exp_r:+.2f}R · R:R {rr:.2f}（未达满仓线，故半仓）")
        if earnings_soon:
            actions_now.append("临近财报：半仓已是上限，隔夜风险自负")
        actions_wait.append("跌破止损或变「回避」→ 立刻停手，不加仓摊平")
        actions_wait.append("只有回到入场区且评分升到「适合」才考虑加到 1R")
    elif enter_ok == "观望":
        actions_now.append("今天不开新仓（赔率/期望/位置未过赚钱门槛）")
        if rr is not None and rr < MIN_RR_CAUTIOUS:
            actions_wait.append(
                f"等更好买点把 R:R 做到 ≥{MIN_RR_CAUTIOUS}（现约 {rr:.2f}）"
            )
        if exp_r is not None and exp_r < 0:
            actions_wait.append(f"现期望 {exp_r:+.2f}R 为负：禁止用「感觉」开仓")
        if e_low and e_high:
            actions_wait.append(f"价格回到 {e_low:.2f}–{e_high:.2f} 再评估")
        if regime is not None and regime.score < 40:
            actions_wait.append("等待大盘环境改善（VIX/SPY 结构）")
        actions_wait.append("不预判抄底、不追已经离开入场区的票")
    else:
        actions_now.append("回避做多；不加仓、不摊平")
        actions_now.append("已有多单：优先检查止损是否仍有效 / 考虑减仓")
        actions_wait.append("等空头结构结束且 R:R/期望重新过线后再做多")

    if vol and getattr(vol, "trend", "") == "放量" and bias.score < 0:
        actions_wait.append("放量偏空：避免在恐慌杀跌中接刀")
    if chase_high or price_above_zone:
        actions_wait.append("靠近高位/在入场区上方：等回踩，FOMO 是亏钱主因之一")
    if liq.score < 40:
        actions_now.append("流动性偏弱：限价、减小单笔、避免市价扫单")
    if vol_cf is not None and vol_cf.label == "放量下跌":
        actions_now.append("放量下跌：今天不做多，等抛压缓和")
    if vol_cf is not None and vol_cf.label == "缩量回踩" and enter_ok in ("适合入场", "谨慎试仓"):
        actions_now.append("缩量回踩：优先限价等回踩区成交，不追阳线")
    if sector_rs is not None and sector_rs.score is not None and sector_rs.score < 40:
        actions_wait.append("弱于板块：优先换同板块强势股，或等个股重夺板块相对强度")
    if iv_reg is not None and iv_reg.high_event_risk:
        actions_now.append("IV 偏高/极高：减仓或等波动回落，警惕事件跳空")
    if fbo is not None and fbo.false_break_risk:
        actions_now.append("假突破信号：今天不追多，等回到入场区再评估")
    elif fbo is not None and fbo.block_breakout_chase:
        actions_wait.append("突破未确认：禁止追高，只允许回踩入场区限价")
    if trend_al is not None and trend_al.against_trend:
        actions_now.append("逆势（大盘/板块偏空）：短线优先空手，不做逆势追多")
    elif trend_al is not None and trend_al.label in ("强跟势", "跟势"):
        if enter_ok in ("适合入场", "谨慎试仓"):
            actions_now.append("跟势环境OK：顺大盘/板块方向在区内做多更稳")
    if weekly is not None and not weekly.allow_long:
        if getattr(weekly, "turning_bullish", False):
            actions_now.append(
                "周线空头但开始转多：最多试仓，不按满仓做多"
                + (
                    f"（{weekly.turning_note}）"
                    if getattr(weekly, "turning_note", "")
                    else ""
                )
            )
        else:
            actions_now.append("周线空头：不做新开多，等开始转多或周线转中性")
    if adx_r is not None and not adx_r.trending:
        actions_wait.append("ADX震荡：少追突破，优先回踩入场区限价")
    if h1_r is not None:
        if h1_r.ready and enter_ok in ("适合入场", "谨慎试仓"):
            actions_now.append("1H已转强且在区内：可挂限价（勿市价追）")
        elif h1_r.label == "已遠離":
            actions_wait.append("1H已远离入场区：等回踩再挂，不追高")
        elif h1_r.label == "1H偏空":
            actions_wait.append("1H偏空：等EMA9重新站上EMA21再考虑")
        elif not h1_r.ready:
            actions_wait.append(f"1H触发「{h1_r.label}」：先设好限价在入场区，等1H配合")

    invalidation = entry.invalidation or (
        f"收盘跌破止损 {stop:.2f} 则本计划作废" if stop else "结构破坏则作废"
    )

    reg_bit = ""
    if regime is not None:
        reg_bit = f"市场 {regime.label}（{regime.score:.0f}）；"
    exp_bit = f"期望 {exp_r:+.2f}R；" if exp_r is not None else ""
    # 摘要以「主周期」为准
    h1v = swing_h1.verdict
    h2v = swing_h2.verdict
    wr_txt = getattr(primary, "win_rate_display", None) or format_win_rate(
        primary.win_rate_pct, getattr(primary, "win_rate_samples", None)
    )
    summary = (
        f"{sym} 现价 **{last:.2f}** · **{mode_thr.label}** · "
        f"主周期 **{primary.label}** → **{primary.verdict}**"
        f"（{risk_tag}）。\n\n"
        f"{trend_note}\n\n"
        f"**主计划**：入场 {primary.entry_low}–{primary.entry_high}，"
        f"止蚀 {primary.stop_loss}，目标 {primary.target}，"
        f"胜率 {wr_txt} · 纸面R:R {primary.rr} · "
        f"净R:R {primary.rr_net}。\n"
        f"出场：{exit_pl.summary}\n"
        f"（对照 0–2周={h1v} / 2–4周={h2v}）{reg_bit}"
    )

    notes = [
        f"模式={mode_thr.label}（日志必须记录）"
        + (f" · {mode_note}" if mode_note else ""),
        f"主周期={primary.label}：决定做不做、出场纪律、滑点后R:R",
        f"时间止损：本模式最多 {max_hold} 个交易日",
        "三灯=位置·胜率·划算；R:R/路径按计划限价E_plan（区中下），非追现价",
        f"滑点假设单边 {DEFAULT_SLIP_PCT * 100:.2f}%：净R:R更接近可成交",
        f"目标至少{MIN_RR_TARGET_K:.1f}:1；止损风险约{STOP_ATR_FLOOR:.1f}–{STOP_ATR_CAP:.1f}×ATR",
        "出场硬规则：T1减半 → 止蚀保本 → 时间止损 → 破止蚀全出",
        "禁止：追高、摊平、扩大止损、无止损进场、逆势硬做",
        "限价挂在计划价附近；成交后写交易日志（含模式 A/B）",
        f"区间报酬 {rets.get('total_return_pct'):.1f}%，年化波动 {rets.get('volatility_pct'):.1f}%"
        if rets.get("total_return_pct") is not None and rets.get("volatility_pct") is not None
        else "数据仅供实盘辅助",
    ]
    notes.extend(structure_notes[:6])
    # 支撑/阻力文案
    resist_lvls = [
        lv
        for lv in (getattr(sr, "levels", None) or [])
        if getattr(lv, "kind", "") == "阻力"
        and last is not None
        and getattr(lv, "price", 0) > float(last)
    ]
    support_lvls = [
        lv
        for lv in (getattr(sr, "levels", None) or [])
        if getattr(lv, "kind", "") == "支撑"
        and last is not None
        and getattr(lv, "price", 0) < float(last)
    ]
    resist_lvls = sorted(resist_lvls, key=lambda x: float(x.price))[:4]
    support_lvls = sorted(support_lvls, key=lambda x: float(x.price), reverse=True)[:4]
    resistance_levels_txt = "；".join(
        f"{float(lv.price):.2f}({lv.strength})" for lv in resist_lvls
    )
    support_levels_txt = "；".join(
        f"{float(lv.price):.2f}({lv.strength})" for lv in support_lvls
    )
    nr = getattr(sr, "nearest_resistance", None)
    ns = getattr(sr, "nearest_support", None)
    up_pct = getattr(sr, "upside_pct", None)
    dn_pct = getattr(sr, "downside_pct", None)
    resistance_note = ""
    support_note = ""
    if nr is not None and last:
        resistance_note = f"最近阻力 ≈ {float(nr):.2f}" + (
            f"（现价上方约 {up_pct:+.1f}%）" if up_pct is not None else ""
        )
    if ns is not None and last:
        support_note = f"最近支撑 ≈ {float(ns):.2f}" + (
            f"（现价下方约 {dn_pct:+.1f}%）" if dn_pct is not None else ""
        )
    if ns is not None or nr is not None:
        notes.append(
            f"近支撑 ≈ {ns} · 近阻力 ≈ {nr}"
            if ns is not None and nr is not None
            else (support_note or resistance_note)
        )
    # T1 与阻力关系提示（不自动改 T，只提醒）
    if (
        nr is not None
        and entry_plan is not None
        and t1 is not None
        and float(entry_plan) < float(nr) < float(t1)
    ):
        structure_notes.append(
            f"T1={float(t1):.2f} 上方先遇阻力 {float(nr):.2f}，可考虑阻力位先减仓"
        )
        notes.append(
            f"阻力提示：计划目标前有阻力 {float(nr):.2f}，短线可先看阻力"
        )
    notes.extend(quality.notes[:4])

    status = free_data_status()
    data_sources = list(getattr(regime, "sources", None) or ["yfinance"])
    if status.get("fred"):
        data_sources.append("FRED")
    if status.get("alphavantage") and (av_filled or (news and getattr(news, "source", "") == "AlphaVantage")):
        data_sources.append("AlphaVantage")
    if status.get("finnhub") and news and getattr(news, "source", "") == "Finnhub":
        data_sources.append("Finnhub")
    elif status.get("finnhub") and news and getattr(news, "available", False) and "AlphaVantage" not in data_sources:
        data_sources.append("Finnhub")

    prim_src_brief = getattr(primary, "win_rate_source", "") or wr_source
    prim_n_brief = getattr(primary, "win_rate_samples", None)
    if prim_n_brief is None:
        prim_n_brief = wr_samples
    if THREE_LIGHT_SOP:
        decision_brief = tl_main["plain_card"]
        if structure_notes:
            decision_brief += "\n\n【结构优化】" + "；".join(structure_notes[:4])
        if resistance_note or support_note:
            decision_brief += (
                "\n\n【支撑/阻力】"
                + (resistance_note or "")
                + ("；" if resistance_note and support_note else "")
                + (support_note or "")
            )
            if resistance_levels_txt:
                decision_brief += f"\n上方阻力：{resistance_levels_txt}"
        # 现价 vs 计划限价提示
        if (
            last is not None
            and entry_plan is not None
            and float(last) > float(entry_plan) * 1.005
        ):
            decision_brief += (
                f"\n现价 {float(last):.2f} 高于计划限价 {float(entry_plan):.2f}："
                f"请挂限价等回，勿市价追。"
            )
        tl_main["plain_card"] = decision_brief
    else:
        decision_brief = build_decision_brief(
            verdict=primary.verdict,
            mode_label=mode_thr.label,
            wr_display=wr_txt,
            wr_confidence=path_wr_confidence(prim_src_brief, prim_n_brief),
            rr_net=primary.rr_net,
            rr_paper=primary.rr,
            exp_r=primary.expectancy_net
            if primary.expectancy_net is not None
            else primary.expectancy_r,
            risk_units=float(risk_units),
            bias_label=bias.bias,
            bias_score=float(bias.score),
            price_far_chase=bool(price_far_chase),
            false_break_risk=bool(fbo_risk),
            against_trend=bool(against_tr),
            weekly_allow_long=bool(weekly_allow),
            vol_label=vol_label or "",
            mode_forced=bool(mode_forced),
            mode_note=mode_note or "",
            thr=mode_thr,
        )

    return TradeSOP(
        symbol=sym,
        name=str(name),
        last_price=round(last, 4),
        enter_ok=enter_ok,
        enter_score=round(enter_score, 1),
        entry_low=round(float(e_low), 4) if e_low else None,
        entry_high=round(float(e_high), 4) if e_high else None,
        entry_plan=display_limit,  # 区内用现价作限价参考；R:R 仍按区中位算
        stop_loss=round(float(stop), 4) if stop else None,
        target_t1=round(float(t1), 4) if t1 else None,
        target_t2=round(float(t2), 4) if t2 else None,
        win_rate_pct=round(primary.win_rate_pct, 1)
        if primary.win_rate_pct is not None
        else (round(win_rate, 1) if win_rate is not None else None),
        win_rate_label=wr_label,
        mode=mode_thr.key,
        mode_label=mode_thr.label,
        mode_forced=mode_forced,
        mode_note=mode_note,
        win_rate_samples=getattr(primary, "win_rate_samples", None)
        if getattr(primary, "win_rate_samples", None) is not None
        else wr_samples,
        win_rate_source=getattr(primary, "win_rate_source", "") or wr_source,
        win_rate_display=wr_txt,
        risk_units=float(risk_units),
        upgrade_1r=bool(upgrade_1r),
        upgrade_hits=int(upgrade_hits),
        upgrade_notes=list(upgrade_notes),
        decision_brief=decision_brief,
        position_light=tl_main["position_light"],
        wr_light=tl_main["wr_light"],
        rr_light=tl_main["rr_light"],
        position_light_note=tl_main["position_light_note"],
        wr_light_note=tl_main["wr_light_note"],
        rr_light_note=tl_main["rr_light_note"],
        one_liner_reason=tl_main["one_liner_reason"],
        plain_card=tl_main["plain_card"],
        notional_hkd=float(tl_main["notional_hkd"]),
        pnl_if_win_hkd=tl_main["pnl_if_win_hkd"],
        pnl_if_loss_hkd=tl_main["pnl_if_loss_hkd"],
        earnings_soon=bool(earnings_soon),
        earnings_days_left=earnings_days_left,
        earnings_note=earnings_note or "",
        nearest_support=round(float(ns), 4) if ns is not None else None,
        nearest_resistance=round(float(nr), 4) if nr is not None else None,
        support_pct=round(float(dn_pct), 2) if dn_pct is not None else None,
        resistance_pct=round(float(up_pct), 2) if up_pct is not None else None,
        resistance_note=resistance_note,
        support_note=support_note,
        sr_summary=str(getattr(sr, "summary", "") or ""),
        resistance_levels_txt=resistance_levels_txt,
        support_levels_txt=support_levels_txt,
        as_of=as_of_str,
        stability_score=stab,
        stability_label=stab_label,
        side=side,
        risk_per_share=round(primary.risk_per_share, 4)
        if primary.risk_per_share
        else (round(risk_ps, 4) if risk_ps else None),
        rr_t1=primary.rr if primary.rr is not None else (round(rr, 2) if rr else None),
        position_shares=int(pos.shares),
        position_note=pos_note,
        actions_now=actions_now,
        actions_wait=actions_wait,
        invalidation=invalidation,
        checklist=checklist,
        bias=bias.bias,
        bias_score=bias.score,
        opportunity=entry.opportunity,
        risk_level=getattr(risk, "risk_level", "—") or "—",
        max_dd_pct=getattr(risk, "max_drawdown_pct", None),
        ann_vol_pct=getattr(risk, "ann_vol_pct", None),
        scorecard_stance=card.stance,
        scorecard_total=card.total_score,
        summary=summary,
        period=period,
        notes=notes,
        expectancy_r=primary.expectancy_r
        if primary.expectancy_r is not None
        else exp_r,
        regime_label=getattr(regime, "label", "—") if regime else "—",
        regime_score=getattr(regime, "score", None) if regime else None,
        regime_summary=getattr(regime, "summary", "") if regime else "",
        regime_bullets=list(getattr(regime, "bullets", []) or []) if regime else [],
        liquidity_label=liq.label,
        liquidity_score=liq.score,
        multi_rs_summary=str(multi_rs.get("summary") or ""),
        quality_notes=list(quality.notes or []),
        news_summary=getattr(news, "summary", "") if news else "",
        data_sources=data_sources,
        scorecard_bullets=list(card.bullets or []),
        sector_rs_summary=getattr(sector_rs, "summary", "") if sector_rs else "",
        sector_rs_score=getattr(sector_rs, "score", None) if sector_rs else None,
        sector_rs_label=getattr(sector_rs, "label", "—") if sector_rs else "—",
        volume_confirm_summary=getattr(vol_cf, "summary", "") if vol_cf else "",
        volume_confirm_score=getattr(vol_cf, "score", None) if vol_cf else None,
        volume_confirm_label=getattr(vol_cf, "label", "—") if vol_cf else "—",
        iv_summary=getattr(iv_reg, "summary", "") if iv_reg else "",
        iv_score=getattr(iv_reg, "score", None) if iv_reg else None,
        iv_label=getattr(iv_reg, "label", "—") if iv_reg else "—",
        iv_high_event=bool(getattr(iv_reg, "high_event_risk", False)) if iv_reg else False,
        swing_h1=swing_h1,
        swing_h2=swing_h2,
        trend_note=trend_note,
        primary_horizon=ph,
        primary_plan=primary,
        exit_plan=exit_pl,
        slip_rr=slip_main,
        false_break_summary=getattr(fbo, "summary", "") if fbo else "",
        false_break_score=getattr(fbo, "score", None) if fbo else None,
        false_break_label=getattr(fbo, "label", "—") if fbo else "—",
        false_break_risk=bool(getattr(fbo, "false_break_risk", False)) if fbo else False,
        trend_align_summary=getattr(trend_al, "summary", "") if trend_al else "",
        trend_align_score=getattr(trend_al, "score", None) if trend_al else None,
        trend_align_label=getattr(trend_al, "label", "—") if trend_al else "—",
        against_trend=bool(getattr(trend_al, "against_trend", False)) if trend_al else False,
        weekly_label=getattr(weekly, "label", "—") if weekly else "—",
        weekly_summary=getattr(weekly, "summary", "") if weekly else "",
        weekly_allow_long=weekly_allow,
        weekly_turning_bullish=weekly_turning,
        weekly_turning_note=weekly_turn_note,
        adx_label=getattr(adx_r, "label", "—") if adx_r else "—",
        adx_value=adx_val,
        adx_summary=getattr(adx_r, "summary", "") if adx_r else "",
        adx_trending=bool(adx_trending) if adx_trending is not None else False,
        fib_summary=(getattr(fib_r, "summary", "") if fib_r else "")
        + (f" · {fib_note}" if fib_note else ""),
        h1_label=getattr(h1_r, "label", "—") if h1_r else "—",
        h1_summary=getattr(h1_r, "summary", "") if h1_r else "",
        h1_ready=bool(h1_ready) if h1_ready is not None else False,
    )
