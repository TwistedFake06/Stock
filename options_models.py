"""Options domain models and ETF whitelist."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

INDEX_ETF_WHITELIST = {
    "QQQ": "纳斯达克100 ETF",
    "VOO": "标普500 ETF (Vanguard)",
    "SPY": "标普500 ETF (SPDR)",
    "IVV": "标普500 ETF (iShares)",
    "DIA": "道琼斯 ETF",
    "IWM": "罗素2000 ETF",
    "VTI": "全美股市 ETF",
    "VT": "全球股市 ETF",
    "EFA": "发达市场 ETF",
}

LEVERAGED = {"QLD", "SSO", "TQQQ", "UPRO", "SPXL"}


def is_options_eligible(symbol: str) -> bool:
    return options_symbol(symbol) in INDEX_ETF_WHITELIST


def options_symbol(symbol: str) -> str:
    return symbol.strip().upper().split(".")[0]


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

@dataclass
class DirectionReport:
    direction: str  # 看多 | 看空 | 中性
    strength: str  # 强 | 中 | 弱
    score: float  # -100 .. +100
    preferred_verticals: list[str]  # codes ordered
    reasons: list[str] = field(default_factory=list)
    summary: str = ""
    style_hint: str = ""  # 偏信用价差 | 偏借方价差 | 观望


@dataclass
class Leg:
    right: str
    strike: float
    side: str
    mid: float
    bid: float
    ask: float
    iv: float | None
    oi: float | None
    volume: float | None
    delta_proxy: float | None = None
    # 成交假设：卖用 bid、买用 ask（自然成交）；无报价时回退 mid
    fill: float = 0.0
    fill_source: str = "mid"  # bid | ask | mid


@dataclass
class SpreadIdea:
    name: str
    code: str  # bull_put | bull_call | bear_call | bear_put
    structure: str  # Credit Vertical | Debit Vertical
    thesis: str
    net_credit: float | None
    net_debit: float | None
    max_profit: float
    max_loss: float
    breakevens: list[float]
    width: float
    pop_est: float | None  # = win_rate_profit（兼容旧字段）
    score: float
    dte: int
    expiry: str
    legs: list[Leg]
    notes: list[str] = field(default_factory=list)
    risk_reward: float | None = None
    otm_label: str = ""  # e.g. 约2% OTM
    rank_reason: str = ""
    # 胜率（对数正态 + IV/HV 估算）
    win_rate_profit: float | None = None  # 到期有利润的概率 %
    win_rate_max: float | None = None  # 拿到最大利润的概率 %（信用短腿OTM）
    win_rate_method: str = ""  # 说明
    expected_value: float | None = None  # 到期期望 $/张（分段积分）
    expected_value_managed: float | None = None  # 50%止盈/2R止损路径期望 $/张
    # 流动性
    liquidity_score: float | None = None  # 0-100
    liquidity_label: str = ""  # 高 / 中 / 低 / 未知
    liquidity_detail: str = ""  # 白话说明
    # 定价：自然成交（bid/ask）还是 mid 回退
    pricing_mode: str = "natural"  # natural | mid_mixed | mid_only
    # 50% 止盈：买回/卖出目标价（$/股）与浮盈（$/张）
    metric_half_buyback: float | None = None
    metric_half_profit: float | None = None

    def __post_init__(self) -> None:
        # 兼容旧缓存 / 不完整构造，保证属性始终存在
        if not hasattr(self, "win_rate_profit"):
            self.win_rate_profit = None
        if not hasattr(self, "win_rate_max"):
            self.win_rate_max = None
        if not hasattr(self, "win_rate_method"):
            self.win_rate_method = ""
        if not hasattr(self, "expected_value"):
            self.expected_value = None
        if not hasattr(self, "expected_value_managed"):
            self.expected_value_managed = None
        if not hasattr(self, "liquidity_score"):
            self.liquidity_score = None
        if not hasattr(self, "liquidity_label"):
            self.liquidity_label = ""
        if not hasattr(self, "liquidity_detail"):
            self.liquidity_detail = ""
        if not hasattr(self, "pricing_mode"):
            self.pricing_mode = "natural"
        if not hasattr(self, "metric_half_buyback"):
            self.metric_half_buyback = None
        if not hasattr(self, "metric_half_profit"):
            self.metric_half_profit = None
        if self.pop_est is None and self.win_rate_profit is not None:
            self.pop_est = self.win_rate_profit


@dataclass
class OptionsReport:
    symbol: str
    label: str
    spot: float
    eligible: bool
    message: str
    direction: DirectionReport | None = None
    expiries: list[str] = field(default_factory=list)
    selected_expiry: str | None = None
    dte: int | None = None
    iv_atm: float | None = None
    ideas: list[SpreadIdea] = field(default_factory=list)
    best: SpreadIdea | None = None
    best_alt: SpreadIdea | None = None
    best_winrate: SpreadIdea | None = None  # 全体最高「有利润」胜率
    best_winrate_aligned: SpreadIdea | None = None  # 与方向契合的最高胜率
    best_playbook: SpreadIdea | None = None  # 贴合常见实战策略
    best_playbook_wr: SpreadIdea | None = None  # 实战规则下最高赢面
    playbook_table: list = field(default_factory=list)
    regime: str = "—"
    summary: str = ""
    action_plan: list[str] = field(default_factory=list)
    # 报价质量
    after_hours: bool = False
    pricing_note: str = ""
    quote_warning: str = ""
    filtered_out: int = 0  # 硬过滤剔除数
    # 是否适合现在做 spread（实盘辅助）
    timing: Any | None = None
