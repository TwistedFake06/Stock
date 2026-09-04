"""
Free / freemium market data helpers for real-trading context.

Primary (no key): yfinance market proxies
  - ^VIX  fear / vol regime
  - ^TNX  10Y yield level & trend
  - SPY   risk-on trend (price vs SMA50/SMA200)
  - HYG   credit risk-on proxy (optional)

Optional free API keys (env):
  - FRED_API_KEY   https://fred.stlouisfed.org/docs/api/api_key.html
  - FINNHUB_API_KEY https://finnhub.io/register  (60 calls/min free)
  - ALPHAVANTAGE_API_KEY (or ALPHA_VANTAGE_API_KEY / AV_API_KEY)
      https://www.alphavantage.co/support/#api-key
      Free tier is rate-limited (often ~25 req/day) — we cache heavily.

All functions degrade gracefully when offline / key missing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from stock_service import cache_bucket, fetch_history, normalize_symbol

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_ENV_LOADED = False


def _load_local_env_once() -> None:
    """
    Load API keys into os.environ if not already set.

    Order:
      1) process env (Cloud Secrets often inject as env, or local shell)
      2) Streamlit secrets (share.streamlit.io → App settings → Secrets)
      3) project .env (local only, gitignored)
    """
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED:
        return
    _LOCAL_ENV_LOADED = True

    secret_keys = (
        "ALPHAVANTAGE_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "AV_API_KEY",
        "FRED_API_KEY",
        "FRED_KEY",
        "FINNHUB_API_KEY",
        "FINNHUB_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TG_BOT_TOKEN",
        "TG_CHAT_ID",
    )

    # Streamlit Cloud / local streamlit secrets.toml
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            for key in secret_keys:
                if (os.environ.get(key) or "").strip():
                    continue
                try:
                    val = secrets.get(key)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        val = secrets[key]  # type: ignore[index]
                    except Exception:
                        val = None
                if val is not None and str(val).strip():
                    os.environ[key] = str(val).strip()
    except Exception:
        pass

    env_path = _PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key or not val:
            continue
        # Do not overwrite explicit process env / secrets
        if not (os.environ.get(key) or "").strip():
            os.environ[key] = val


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MarketRegime:
    """Cross-asset regime used to gate long entries."""

    label: str = "—"  # 风险偏好 | 中性 | 避险 | 数据不足
    score: float = 50.0  # 0-100 higher = better for long risk assets
    vix: float | None = None
    vix_label: str = "—"
    tnx: float | None = None  # 10Y yield %
    spy_vs_sma200_pct: float | None = None
    spy_trend: str = "—"  # 多头 | 震荡 | 空头
    credit_ok: bool | None = None
    bullets: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class LiquidityReport:
    """Can you actually trade this size without too much friction?"""

    score: float = 50.0  # 0-100
    label: str = "—"  # 高 | 中 | 低
    avg_dollar_vol: float | None = None
    avg_volume: float | None = None
    last_volume: float | None = None
    bid_ask_proxy_bps: float | None = None  # rough from daily range / price
    notes: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class StockQualityExtras:
    """Extra yfinance fields useful for real equity selection."""

    short_pct_float: float | None = None
    short_ratio: float | None = None
    inst_own_pct: float | None = None
    insider_own_pct: float | None = None
    beta: float | None = None
    avg_volume: float | None = None
    market_cap: float | None = None
    free_cashflow: float | None = None
    fcf_yield_pct: float | None = None
    operating_margins: float | None = None
    current_ratio: float | None = None
    fifty_two_week_pct: float | None = None  # 0-100 position in 52w range
    sector: str = ""
    industry: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class OptionalNewsPulse:
    """Company news buzz (Finnhub and/or Alpha Vantage)."""

    available: bool = False
    article_count_7d: int = 0
    headline_sample: list[str] = field(default_factory=list)
    summary: str = ""
    sentiment_score: float | None = None  # Alpha Vantage: typically ~-1..+1
    sentiment_label: str = ""
    source: str = ""


@dataclass
class AlphaVantageEnrichment:
    """Result of optional Alpha Vantage free-key pull."""

    available: bool = False
    overview_merged: int = 0
    filled_keys: list[str] = field(default_factory=list)
    news: OptionalNewsPulse = field(default_factory=OptionalNewsPulse)
    summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_close(symbol: str, period: str = "6mo") -> float | None:
    try:
        df = fetch_history(symbol, period=period, interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return None


def _hist_close(symbol: str, period: str = "1y") -> pd.Series:
    try:
        df = fetch_history(symbol, period=period, interval="1d")
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df.set_index("Date")["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)


def _sma(series: pd.Series, n: int) -> float | None:
    if series is None or len(series) < n:
        return None
    return float(series.tail(n).mean())


# ---------------------------------------------------------------------------
# Market regime (yfinance free)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def fetch_market_regime(_bucket: str = "") -> MarketRegime:
    """
    Build a simple risk-on / risk-off score from free tickers.

    Weights (approx):
      VIX 40 · SPY trend 40 · 10Y shock 10 · HYG 10
    """
    _ = _bucket  # time-bucket cache key from caller
    bullets: list[str] = []
    sources: list[str] = ["yfinance:^VIX", "yfinance:SPY", "yfinance:^TNX"]
    score = 50.0

    # --- VIX ---
    vix = _last_close("^VIX", period="3mo")
    vix_label = "—"
    if vix is not None:
        if vix < 14:
            vix_label = "低波动"
            score += 12
            bullets.append(f"✅ VIX {vix:.1f} 低：风险偏好环境，利于做多")
        elif vix < 20:
            vix_label = "正常"
            score += 5
            bullets.append(f"✅ VIX {vix:.1f} 正常区间")
        elif vix < 28:
            vix_label = "偏高"
            score -= 10
            bullets.append(f"⚠️ VIX {vix:.1f} 偏高：仓位宜缩小、止损严格执行")
        else:
            vix_label = "恐慌"
            score -= 22
            bullets.append(f"❌ VIX {vix:.1f} 恐慌区：新开多单需极高门槛")
    else:
        bullets.append("— 无法获取 VIX")

    # --- SPY trend ---
    spy = _hist_close("SPY", period="2y")
    spy_vs = None
    spy_trend = "—"
    if len(spy) >= 60:
        last = float(spy.iloc[-1])
        sma50 = _sma(spy, 50)
        sma200 = _sma(spy, 200) if len(spy) >= 200 else _sma(spy, min(120, len(spy)))
        if sma200 and sma200 > 0:
            spy_vs = (last / sma200 - 1.0) * 100
        if sma50 and sma200:
            if last > sma50 > sma200:
                spy_trend = "多头"
                score += 14
                bullets.append(
                    f"✅ SPY 多头结构（价>{sma50:.0f}>{sma200:.0f}）"
                    + (f" · 距SMA200 {spy_vs:+.1f}%" if spy_vs is not None else "")
                )
            elif last < sma50 < sma200:
                spy_trend = "空头"
                score -= 16
                bullets.append(
                    f"❌ SPY 空头结构（价<{sma50:.0f}<{sma200:.0f}）· 大盘逆风做多"
                )
            else:
                spy_trend = "震荡"
                score -= 2
                bullets.append("⚠️ SPY 均线纠缠：大盘中性/震荡")
        elif sma50:
            if last > sma50:
                spy_trend = "偏多"
                score += 6
            else:
                spy_trend = "偏空"
                score -= 6
    else:
        bullets.append("— SPY 历史不足，跳过大盘趋势")

    # --- 10Y yield level (rate regime) ---
    tnx = _last_close("^TNX", period="6mo")
    tnx_hist = _hist_close("^TNX", period="6mo")
    if tnx is not None:
        # Rising yields often pressure growth multiples
        if len(tnx_hist) >= 22:
            prev = float(tnx_hist.iloc[-22])
            chg = tnx - prev
            if chg >= 0.35:
                score -= 8
                bullets.append(f"⚠️ 10Y 收益率近月上行 {chg:+.2f}pt → 估值承压")
            elif chg <= -0.35:
                score += 5
                bullets.append(f"✅ 10Y 收益率近月回落 {chg:+.2f}pt → 利多风险资产")
            else:
                bullets.append(f"· 10Y ≈ {tnx:.2f}%（近月变化有限）")
        else:
            bullets.append(f"· 10Y ≈ {tnx:.2f}%")
    else:
        bullets.append("— 无法获取 ^TNX")

    # --- HYG credit proxy (optional) ---
    hyg = _hist_close("HYG", period="6mo")
    credit_ok: bool | None = None
    if len(hyg) >= 50:
        sources.append("yfinance:HYG")
        h_last = float(hyg.iloc[-1])
        h_sma = _sma(hyg, 50)
        if h_sma:
            credit_ok = h_last >= h_sma
            if credit_ok:
                score += 4
                bullets.append("✅ HYG 高于 SMA50：信用环境偏稳")
            else:
                score -= 6
                bullets.append("⚠️ HYG 低于 SMA50：信用压力↑，高收益债风险偏好下降")

    # Optional FRED enrichment
    fred = _fred_snapshot()
    if fred:
        sources.append("FRED")
        if fred.get("vix") is not None and vix is None:
            vix = float(fred["vix"])
        if fred.get("dgs10") is not None and tnx is None:
            tnx = float(fred["dgs10"])
        if fred.get("t10y2y") is not None:
            curve = float(fred["t10y2y"])
            if curve < 0:
                score -= 4
                bullets.append(f"⚠️ 美债利差倒挂 T10Y2Y={curve:.2f}（衰退定价信号）")
            else:
                bullets.append(f"· 美债利差 T10Y2Y={curve:.2f}")

    score = float(max(0.0, min(100.0, score)))
    if score >= 65:
        label = "风险偏好"
    elif score >= 45:
        label = "中性"
    elif score >= 0 and (vix is not None or spy_trend != "—"):
        label = "避险"
    else:
        label = "数据不足"

    summary = (
        f"市场环境 **{label}**（{score:.0f}/100）。"
        f"VIX {vix_label}"
        + (f" {vix:.1f}" if vix is not None else "")
        + f" · SPY {spy_trend}"
        + (f" · 10Y {tnx:.2f}%" if tnx is not None else "")
        + "。"
    )
    return MarketRegime(
        label=label,
        score=round(score, 1),
        vix=round(vix, 2) if vix is not None else None,
        vix_label=vix_label,
        tnx=round(tnx, 3) if tnx is not None else None,
        spy_vs_sma200_pct=round(spy_vs, 2) if spy_vs is not None else None,
        spy_trend=spy_trend,
        credit_ok=credit_ok,
        bullets=bullets,
        sources=sources,
        summary=summary,
    )


def get_market_regime() -> MarketRegime:
    """Cached ~15 minutes via time bucket."""
    return fetch_market_regime(cache_bucket(15))


# ---------------------------------------------------------------------------
# Liquidity (from OHLCV — free)
# ---------------------------------------------------------------------------


def analyze_liquidity(df: pd.DataFrame, info: dict[str, Any] | None = None) -> LiquidityReport:
    """Score tradability from volume + daily range proxy."""
    notes: list[str] = []
    score = 50.0
    avg_vol = last_vol = avg_dollar = spread_bps = None

    if df is not None and not df.empty and "Close" in df.columns:
        close = df["Close"].astype(float)
        last_px = float(close.iloc[-1]) if len(close) else None
        if "Volume" in df.columns:
            vol = df["Volume"].astype(float).dropna()
            if len(vol):
                avg_vol = float(vol.tail(20).mean()) if len(vol) >= 5 else float(vol.mean())
                last_vol = float(vol.iloc[-1])
                if last_px and avg_vol:
                    avg_dollar = avg_vol * last_px
        # range proxy for spread friction
        if last_px and last_px > 0 and "High" in df.columns and "Low" in df.columns:
            rng = (df["High"].astype(float) - df["Low"].astype(float)).tail(20)
            mid = float(rng.mean()) if len(rng) else None
            if mid is not None:
                spread_bps = (mid / last_px) * 10000 / 4  # rough: 1/4 of range as friction
                if spread_bps < 15:
                    score += 10
                elif spread_bps > 60:
                    score -= 12
                    notes.append(f"日振幅折算摩擦偏大（~{spread_bps:.0f}bp）")

    # Prefer yfinance averageVolume if present
    info = info or {}
    try:
        av = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        if av is not None:
            avg_vol = float(av)
            px = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("last_price")
            if px and avg_vol:
                avg_dollar = float(avg_vol) * float(px)
    except Exception:
        pass

    if avg_dollar is not None:
        # US liquid names often > $20M–$50M ADV
        if avg_dollar >= 50_000_000:
            score += 25
            notes.append(f"日均成交额约 ${avg_dollar/1e6:.0f}M · 流动性优秀")
        elif avg_dollar >= 10_000_000:
            score += 15
            notes.append(f"日均成交额约 ${avg_dollar/1e6:.1f}M · 流动性良好")
        elif avg_dollar >= 2_000_000:
            score += 0
            notes.append(f"日均成交额约 ${avg_dollar/1e6:.2f}M · 流动性一般")
        else:
            score -= 20
            notes.append(f"日均成交额约 ${avg_dollar/1e6:.2f}M · 偏薄，滑点/冲击成本风险")
    else:
        notes.append("无可靠成交额，流动性评分中性")

    score = float(max(0.0, min(100.0, score)))
    if score >= 70:
        label = "高"
    elif score >= 45:
        label = "中"
    else:
        label = "低"

    summary = f"流动性 **{label}**（{score:.0f}/100）"
    if avg_dollar is not None:
        summary += f" · 约日均 ${avg_dollar/1e6:.1f}M"
    summary += "。"
    return LiquidityReport(
        score=round(score, 1),
        label=label,
        avg_dollar_vol=avg_dollar,
        avg_volume=avg_vol,
        last_volume=last_vol,
        bid_ask_proxy_bps=round(spread_bps, 1) if spread_bps is not None else None,
        notes=notes,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Quality extras from yfinance info (free)
# ---------------------------------------------------------------------------


def extract_quality_extras(info: dict[str, Any] | None, last_price: float | None = None) -> StockQualityExtras:
    info = info or {}
    notes: list[str] = []

    def f(key: str) -> float | None:
        v = info.get(key)
        if v is None:
            return None
        try:
            x = float(v)
            if x != x:  # NaN
                return None
            return x
        except (TypeError, ValueError):
            return None

    short_pct = f("shortPercentOfFloat")
    if short_pct is not None and short_pct <= 1.5:
        short_pct = short_pct * 100  # sometimes 0.05 = 5%
    short_ratio = f("shortRatio")
    inst = f("heldPercentInstitutions")
    if inst is not None and inst <= 1.5:
        inst = inst * 100
    insider = f("heldPercentInsiders")
    if insider is not None and insider <= 1.5:
        insider = insider * 100
    beta = f("beta")
    avg_vol = f("averageVolume") or f("averageDailyVolume10Day")
    mcap = f("marketCap")
    fcf = f("freeCashflow")
    op_m = f("operatingMargins")
    if op_m is not None and abs(op_m) <= 1.5:
        op_m = op_m * 100
    cr = f("currentRatio")

    price = last_price or f("currentPrice") or f("regularMarketPrice") or f("last_price")
    hi = f("fiftyTwoWeekHigh") or f("year_high")
    lo = f("fiftyTwoWeekLow") or f("year_low")
    pct52 = None
    if price and hi and lo and hi > lo:
        pct52 = (float(price) - float(lo)) / (float(hi) - float(lo)) * 100
        if pct52 >= 90:
            notes.append("靠近52周高位：追高风险↑，更要严格入场区")
        elif pct52 <= 20:
            notes.append("靠近52周低位：可能超跌或基本面恶化，需分清")

    fcf_yield = None
    if fcf is not None and mcap and mcap > 0:
        fcf_yield = fcf / mcap * 100
        if fcf_yield >= 5:
            notes.append(f"FCF 收益率约 {fcf_yield:.1f}%（偏吸引）")
        elif fcf_yield < 0:
            notes.append("自由现金流为负：扩张期或盈利质量需警惕")

    if short_pct is not None and short_pct >= 15:
        notes.append(f"空头占流通盘约 {short_pct:.1f}%：挤空/波动放大风险")
    if beta is not None and beta >= 1.6:
        notes.append(f"Beta≈{beta:.2f}：弹性大，仓位宜更小")

    return StockQualityExtras(
        short_pct_float=round(short_pct, 2) if short_pct is not None else None,
        short_ratio=round(short_ratio, 2) if short_ratio is not None else None,
        inst_own_pct=round(inst, 1) if inst is not None else None,
        insider_own_pct=round(insider, 1) if insider is not None else None,
        beta=round(beta, 2) if beta is not None else None,
        avg_volume=avg_vol,
        market_cap=mcap,
        free_cashflow=fcf,
        fcf_yield_pct=round(fcf_yield, 2) if fcf_yield is not None else None,
        operating_margins=round(op_m, 1) if op_m is not None else None,
        current_ratio=round(cr, 2) if cr is not None else None,
        fifty_two_week_pct=round(pct52, 1) if pct52 is not None else None,
        sector=str(info.get("sector") or ""),
        industry=str(info.get("industry") or ""),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-horizon relative strength (free OHLC)
# ---------------------------------------------------------------------------


def multi_horizon_rs(
    stock_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    windows: tuple[int, ...] = (21, 63, 126),
) -> dict[str, Any]:
    """
    Excess return vs benchmark over ~1m / 3m / 6m trading days.
    Returns score 0-100 and per-window alphas.
    """
    out: dict[str, Any] = {
        "alphas": {},
        "score": None,
        "label": "—",
        "summary": "",
    }
    if stock_df is None or bench_df is None or stock_df.empty or bench_df.empty:
        out["summary"] = "相对强弱数据不足"
        return out
    try:
        s = stock_df.set_index("Date")["Close"].astype(float)
        b = bench_df.set_index("Date")["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        b.index = pd.to_datetime(b.index).tz_localize(None)
        j = pd.concat([s.rename("s"), b.rename("b")], axis=1).dropna()
    except Exception:
        out["summary"] = "相对强弱对齐失败"
        return out

    if len(j) < 30:
        out["summary"] = "重叠交易日过少"
        return out

    alphas: list[float] = []
    weights = {21: 0.45, 63: 0.35, 126: 0.20}
    for w in windows:
        if len(j) <= w:
            continue
        seg = j.iloc[-(w + 1) :]
        s_ret = float(seg["s"].iloc[-1] / seg["s"].iloc[0] - 1) * 100
        b_ret = float(seg["b"].iloc[-1] / seg["b"].iloc[0] - 1) * 100
        a = s_ret - b_ret
        out["alphas"][f"{w}d"] = round(a, 2)
        alphas.append(a * weights.get(w, 0.3))

    if not alphas:
        out["summary"] = "窗口不足"
        return out

    # Weighted excess → score (0 excess = 50)
    w_alpha = sum(alphas)
    score = max(0.0, min(100.0, 50.0 + w_alpha * 1.8))
    if score >= 62:
        label = "强于大盘"
    elif score <= 40:
        label = "弱于大盘"
    else:
        label = "同步"
    parts = [f"{k}超额{v:+.1f}%" for k, v in out["alphas"].items()]
    out["score"] = round(score, 1)
    out["label"] = label
    out["summary"] = f"多周期RS **{label}**（{score:.0f}）· " + " · ".join(parts)
    return out


# ---------------------------------------------------------------------------
# Optional: FRED free API
# ---------------------------------------------------------------------------


def _fred_api_key() -> str:
    _load_local_env_once()
    return (os.environ.get("FRED_API_KEY") or os.environ.get("FRED_KEY") or "").strip()


@lru_cache(maxsize=4)
def _fred_snapshot(_bucket: str = "") -> dict[str, float]:
    """Latest few FRED series if free API key present."""
    _ = _bucket
    key = _fred_api_key()
    if not key:
        return {}
    series_map = {
        "vix": "VIXCLS",
        "dgs10": "DGS10",
        "t10y2y": "T10Y2Y",
    }
    out: dict[str, float] = {}
    for name, sid in series_map.items():
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            r = requests.get(
                url,
                params={
                    "series_id": sid,
                    "api_key": key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                },
                timeout=8,
            )
            if r.status_code != 200:
                continue
            obs = r.json().get("observations") or []
            for o in obs:
                v = o.get("value")
                if v in (None, "."):
                    continue
                out[name] = float(v)
                break
        except Exception:
            continue
        time.sleep(0.05)
    return out


def fred_enabled() -> bool:
    return bool(_fred_api_key())


# ---------------------------------------------------------------------------
# Optional: Finnhub free API (news pulse)
# ---------------------------------------------------------------------------


def _finnhub_key() -> str:
    _load_local_env_once()
    return (os.environ.get("FINNHUB_API_KEY") or os.environ.get("FINNHUB_KEY") or "").strip()


def finnhub_enabled() -> bool:
    return bool(_finnhub_key())


@lru_cache(maxsize=32)
def fetch_finnhub_news_pulse(symbol: str, _bucket: str = "") -> OptionalNewsPulse:
    """Last 7 days company news count (free tier)."""
    _ = _bucket
    key = _finnhub_key()
    sym = normalize_symbol(symbol)
    if not key or not sym or not _is_us_simple(sym):
        return OptionalNewsPulse(
            available=False,
            summary="未配置 FINNHUB_API_KEY 或非美股，跳过新闻脉冲（免费注册即可用）。",
        )
    try:
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=7)
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": sym,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "token": key,
            },
            timeout=8,
        )
        if r.status_code != 200:
            return OptionalNewsPulse(
                available=False,
                summary=f"Finnhub 新闻请求失败 HTTP {r.status_code}",
            )
        data = r.json()
        if not isinstance(data, list):
            data = []
        heads = []
        for item in data[:5]:
            h = (item or {}).get("headline") or ""
            if h:
                heads.append(str(h)[:120])
        n = len(data)
        if n >= 12:
            tone = "媒体关注度高（波动/消息驱动概率↑）"
        elif n >= 3:
            tone = "有一定新闻覆盖"
        else:
            tone = "近7日新闻较少"
        return OptionalNewsPulse(
            available=True,
            article_count_7d=n,
            headline_sample=heads,
            summary=f"Finnhub 近7日新闻 {n} 条 · {tone}",
        )
    except Exception as exc:
        return OptionalNewsPulse(available=False, summary=f"Finnhub 不可用：{exc}")


def get_finnhub_news(symbol: str) -> OptionalNewsPulse:
    return fetch_finnhub_news_pulse(normalize_symbol(symbol), cache_bucket(30))


def _is_us_simple(sym: str) -> bool:
    s = (sym or "").upper()
    if not s:
        return False
    if s.endswith((".HK", ".SS", ".SZ", ".BJ", ".SH")):
        return False
    if s.isdigit():
        return False
    return True


# ---------------------------------------------------------------------------
# Optional: Alpha Vantage free API
# ---------------------------------------------------------------------------


def _av_api_key() -> str:
    _load_local_env_once()
    return (
        os.environ.get("ALPHAVANTAGE_API_KEY")
        or os.environ.get("ALPHA_VANTAGE_API_KEY")
        or os.environ.get("AV_API_KEY")
        or ""
    ).strip()


def alphavantage_enabled() -> bool:
    return bool(_av_api_key())


def _av_get(params: dict[str, Any], timeout: float = 12.0) -> dict[str, Any] | None:
    """GET alphavantage.co/query; return JSON dict or None on rate-limit/error."""
    key = _av_api_key()
    if not key:
        return None
    try:
        q = dict(params)
        q["apikey"] = key
        r = requests.get("https://www.alphavantage.co/query", params=q, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, dict):
            return None
        # Free-tier throttle / premium upsell messages
        if data.get("Note") or data.get("Information") or data.get("Error Message"):
            return None
        return data
    except Exception:
        return None


# Map Alpha Vantage OVERVIEW fields → yfinance-like keys used by analyze_fundamentals
_AV_OVERVIEW_MAP: dict[str, str] = {
    "PERatio": "trailingPE",
    "ForwardPE": "forwardPE",
    "PEGRatio": "pegRatio",
    "PriceToBookRatio": "priceToBook",
    "PriceToSalesRatioTTM": "priceToSalesTrailing12Months",
    "ReturnOnEquityTTM": "returnOnEquity",
    "ProfitMargin": "profitMargins",
    "OperatingMarginTTM": "operatingMargins",
    "QuarterlyRevenueGrowthYOY": "revenueGrowth",
    "QuarterlyEarningsGrowthYOY": "earningsQuarterlyGrowth",
    "DividendYield": "dividendYield",
    "Beta": "beta",
    "MarketCapitalization": "marketCap",
    "52WeekHigh": "fiftyTwoWeekHigh",
    "52WeekLow": "fiftyTwoWeekLow",
    "AnalystTargetPrice": "targetMeanPrice",
    "TrailingPE": "trailingPE",
    "EVToEBITDA": "enterpriseToEbitda",
    "BookValue": "bookValue",
    "DilutedEPSTTM": "trailingEps",
    "RevenueTTM": "totalRevenue",
    "GrossProfitTTM": "grossProfits",
    "EBITDA": "ebitda",
    "SharesOutstanding": "sharesOutstanding",
    "PercentInsiders": "heldPercentInsiders",
    "PercentInstitutions": "heldPercentInstitutions",
}


def _av_num(val: Any) -> float | None:
    if val is None or val == "" or val == "None" or val == "-":
        return None
    try:
        x = float(val)
        if x != x:  # NaN
            return None
        return x
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=48)
def fetch_av_overview(symbol: str, _bucket: str = "") -> dict[str, Any]:
    """
    COMPANY_OVERVIEW → dict of yfinance-compatible keys (only numeric/useful fields).
    Cached ~60–120 min via caller bucket (free tier is tight).
    """
    _ = _bucket
    sym = normalize_symbol(symbol)
    if not _av_api_key() or not _is_us_simple(sym):
        return {}
    data = _av_get({"function": "OVERVIEW", "symbol": sym})
    if not data or not data.get("Symbol"):
        return {}
    out: dict[str, Any] = {"_av_source": "AlphaVantage", "_av_symbol": data.get("Symbol")}
    sector = data.get("Sector")
    industry = data.get("Industry")
    name = data.get("Name")
    if sector:
        out["sector"] = sector
    if industry:
        out["industry"] = industry
    if name:
        out["longName"] = name
        out["shortName"] = name
    for av_key, yf_key in _AV_OVERVIEW_MAP.items():
        num = _av_num(data.get(av_key))
        if num is None:
            continue
        # AV often returns percent-like ratios already as decimals (0.15) or percents
        if yf_key in ("heldPercentInsiders", "heldPercentInstitutions") and num > 1.5:
            num = num / 100.0
        out[yf_key] = num
    # Free cash flow not always on OVERVIEW; skip if absent
    return out


@lru_cache(maxsize=48)
def fetch_av_news_sentiment(symbol: str, _bucket: str = "") -> OptionalNewsPulse:
    """
    NEWS_SENTIMENT for ticker (free key). Gives overall sentiment + headlines.
    1 request per symbol per cache bucket — protect free daily quota.
    """
    _ = _bucket
    sym = normalize_symbol(symbol)
    if not _av_api_key() or not _is_us_simple(sym):
        return OptionalNewsPulse(
            available=False,
            summary="未配置 ALPHAVANTAGE_API_KEY 或非美股，跳过 Alpha Vantage 新闻情绪。",
            source="",
        )
    data = _av_get(
        {
            "function": "NEWS_SENTIMENT",
            "tickers": sym,
            "limit": 20,
            "sort": "LATEST",
        }
    )
    if not data:
        return OptionalNewsPulse(
            available=False,
            summary="Alpha Vantage 新闻情绪不可用（限流/无数据/key 无效）。",
            source="AlphaVantage",
        )

    feed = data.get("feed") or []
    if not isinstance(feed, list):
        feed = []

    # Aggregate ticker-specific sentiment when present
    scores: list[float] = []
    heads: list[str] = []
    for item in feed[:20]:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or ""
        if title:
            heads.append(str(title)[:120])
        tickers = item.get("ticker_sentiment") or []
        hit = False
        if isinstance(tickers, list):
            for t in tickers:
                if not isinstance(t, dict):
                    continue
                if str(t.get("ticker", "")).upper() != sym.upper():
                    continue
                sc = _av_num(t.get("ticker_sentiment_score"))
                if sc is not None:
                    scores.append(sc)
                    hit = True
        if not hit:
            sc = _av_num(item.get("overall_sentiment_score"))
            if sc is not None:
                scores.append(sc)

    n = len(feed)
    avg = float(sum(scores) / len(scores)) if scores else None
    if avg is None:
        label = "中性"
    elif avg >= 0.25:
        label = "偏多"
    elif avg >= 0.10:
        label = "略偏多"
    elif avg <= -0.25:
        label = "偏空"
    elif avg <= -0.10:
        label = "略偏空"
    else:
        label = "中性"

    tone = f"情绪 {label}"
    if avg is not None:
        tone += f"（score {avg:+.2f}）"
    if n >= 12:
        tone += " · 报道量大"
    elif n <= 2:
        tone += " · 报道偏少"

    return OptionalNewsPulse(
        available=True,
        article_count_7d=n,
        headline_sample=heads[:5],
        summary=f"Alpha Vantage 新闻情绪：{tone} · 样本 {n} 条",
        sentiment_score=round(avg, 3) if avg is not None else None,
        sentiment_label=label,
        source="AlphaVantage",
    )


def merge_av_overview_into_info(
    info: dict[str, Any] | None,
    symbol: str,
) -> tuple[dict[str, Any], list[str]]:
    """
    Fill missing fundamental fields from Alpha Vantage OVERVIEW.
    Does not overwrite existing yfinance values (prefer live Yahoo when present).
    """
    base = dict(info or {})
    if not alphavantage_enabled():
        return base, []
    av = fetch_av_overview(normalize_symbol(symbol), cache_bucket(90))
    if not av:
        return base, []
    filled: list[str] = []
    for k, v in av.items():
        if k.startswith("_"):
            continue
        cur = base.get(k)
        empty = cur is None or cur == "" or cur == 0
        if empty:
            base[k] = v
            filled.append(k)
    if filled:
        base["_av_filled"] = filled
    return base, filled


def get_alphavantage_enrichment(symbol: str) -> AlphaVantageEnrichment:
    """Overview merge metadata + news sentiment (0–2 free API calls, cached)."""
    if not alphavantage_enabled():
        return AlphaVantageEnrichment(
            available=False,
            summary="未设置 ALPHAVANTAGE_API_KEY（免费注册：https://www.alphavantage.co/support/#api-key）",
        )
    sym = normalize_symbol(symbol)
    overview = fetch_av_overview(sym, cache_bucket(90))
    news = fetch_av_news_sentiment(sym, cache_bucket(60))
    n_ov = len([k for k in overview if not k.startswith("_")])
    parts = []
    if n_ov:
        parts.append(f"OVERVIEW {n_ov} 字段")
    if news.available:
        parts.append(news.summary)
    elif _av_api_key():
        parts.append(news.summary or "新闻情绪未取到")
    return AlphaVantageEnrichment(
        available=bool(n_ov or news.available),
        overview_merged=n_ov,
        filled_keys=[k for k in overview if not k.startswith("_")],
        news=news,
        summary="Alpha Vantage：" + ("；".join(parts) if parts else "无数据/可能触达免费限额"),
    )


def get_news_pulse(symbol: str) -> OptionalNewsPulse:
    """
    Prefer Alpha Vantage sentiment (richer), else Finnhub article count.
    """
    if alphavantage_enabled():
        av = fetch_av_news_sentiment(normalize_symbol(symbol), cache_bucket(60))
        if av.available:
            return av
    if finnhub_enabled():
        return get_finnhub_news(symbol)
    return OptionalNewsPulse(
        available=False,
        summary="未配置新闻 API。可选：ALPHAVANTAGE_API_KEY（情绪）或 FINNHUB_API_KEY（条数）。",
    )


def free_data_status() -> dict[str, Any]:
    """UI helper: which free sources are active."""
    return {
        "yfinance_regime": True,
        "fred": fred_enabled(),
        "finnhub": finnhub_enabled(),
        "alphavantage": alphavantage_enabled(),
        "hints": [
            "默认用 yfinance 免费拉 VIX / SPY / 10Y / HYG（无需 key）",
            "可选：环境变量 FRED_API_KEY → 美债利差等宏观（https://fred.stlouisfed.org/docs/api/api_key.html）",
            "可选：环境变量 FINNHUB_API_KEY → 公司新闻条数（https://finnhub.io/register）",
            "可选：环境变量 ALPHAVANTAGE_API_KEY → 基本面 OVERVIEW + 新闻情绪（https://www.alphavantage.co/support/#api-key）",
            "Alpha Vantage 免费额度很紧（常见约 25 次/天）· 本应用已按小时缓存",
        ],
    }
