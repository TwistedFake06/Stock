"""
股票分析助手 — Streamlit 桌面/浏览器应用
快速选择 / 自选：美股为主。资金默认 50,000 HKD。

Streamlit Cloud / 本地入口固定为本文件；页面实现在 views/（勿改名 pages/，
以免触发 Streamlit multipage 自动导航）。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure project root is on sys.path (fixes odd launch cwd / IDE issues)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st


def _load_stock_service():
    """Load stock_service; force reload if a stale process-level module is missing helpers."""
    import stock_service as _ss

    if not hasattr(_ss, "filter_us_only") or not hasattr(_ss, "is_us_symbol"):
        _ss = importlib.reload(_ss)
    return _ss


try:
    _ss = _load_stock_service()
    INTERVAL_MAP = _ss.INTERVAL_MAP
    PERIOD_MAP = _ss.PERIOD_MAP
    filter_us_only = _ss.filter_us_only
    is_us_symbol = _ss.is_us_symbol
    normalize_symbol = _ss.normalize_symbol
    from ui_mobile import inject_ios_safari_support, inject_mobile_css
    from views.bias_page import render_bias_page
    from views.common import load_watchlist
    from views.compare_page import render_compare
    from views.dashboard import render_dashboard
    from views.entry_page import render_entry
    from views.extra_page import render_extra
    from views.scan_page import render_scan
    from views.intraday_scan_page import render_intraday_scan
    from views.hold_page import render_hold_page
    from views.options_page import render_options
    from views.sop_page import render_sop
    from views.technical_page import render_technical
    from views.validation_page import render_strategy_validation
    from views.watchlist_page import render_watchlist
    from screens.workbench import render_workbench
except Exception as exc:
    # Friendly page when deps/modules fail (e.g. ran without venv / missing plotly)
    try:
        st.set_page_config(page_title="启动失败", page_icon="⚠️")
    except Exception:
        pass
    st.error("启动失败：模块导入错误")
    st.code(f"{type(exc).__name__}: {exc}")
    st.markdown(
        """
**推荐启动方式（请用项目虚拟环境）：**

**Windows（本地）**

1. 双击 `run.bat`，或在项目根目录执行：

```bat
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m streamlit run app.py
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Streamlit Community Cloud**

- Main file path: `app.py`
- Python 3.11 · 依赖来自 `requirements.txt`
- 无需本地路径；重新 Deploy 即可

不要用未安装依赖的系统 Python 直接 `streamlit run`。
"""
    )
    st.stop()

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",  # better for iPhone
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Dashboard · 实盘辅助",
    },
)

inject_ios_safari_support()
inject_mobile_css()

try:
    from trade_sop import SOP_BUILD as _SOP_BUILD
except Exception:
    _SOP_BUILD = "legacy-no-build"

# ---- session state ----
def _ensure_quick_pins(wl: list) -> list:
    """Keep MU / SNDK (QUICK_PIN) at front of watchlist for one-tap select."""
    try:
        from stock_service import QUICK_PIN as _pins
    except Exception:
        _pins = ["MU", "SNDK"]
    pins = [normalize_symbol(p) for p in _pins if p]
    rest = [s for s in filter_us_only(list(wl or [])) if s not in pins]
    return filter_us_only(pins + rest)


if "watchlist" not in st.session_state:
    st.session_state.watchlist = _ensure_quick_pins(load_watchlist())
else:
    # Drop non-US from quick-select / saved list only (manual symbol stays free)
    st.session_state.watchlist = _ensure_quick_pins(st.session_state.watchlist)

if "symbol" not in st.session_state or not str(st.session_state.symbol).strip():
    st.session_state.symbol = st.session_state.watchlist[0]


def _on_symbol_box_change() -> None:
    """Commit analysis symbol when the text box value changes (Enter / blur).

    Do NOT write back to ``symbol_box`` here — Streamlit forbids mutating a
    widget's own key after it is bound (and on_change is still that key).
    """
    raw = str(st.session_state.get("symbol_box") or "").strip()
    ns = normalize_symbol(raw)
    if ns:
        st.session_state.symbol = ns


# 主流程（精简）：日常计划、持仓、扫描与规则验证；其余收进「更多」
PAGE_MAIN = [
    "投资SOP",
    "短炒工作台",
    "我已买入",
    "Watchlist扫描",
    "开市超短扫描",
    "策略验证",
    "期权价差",
    "更多…",
]
PAGE_MORE = [
    "行情看板",
    "多空分析",
    "入场与目标价",
    "综合分析",
    "技术分析",
    "多股对比",
    "自选股",
]
PAGE_OPTIONS = PAGE_MAIN + PAGE_MORE  # 兼容旧 session 值


def _request_symbol(raw: str, *, open_sop: bool = False) -> None:
    """
    Queue a symbol change for the *next* run (before text_input is created).

    After a widget with key=symbol_box exists, we must not assign
    session_state.symbol_box in the same run — use _pending_symbol instead.
    If open_sop=True, next run switches nav to 投资SOP for that symbol.
    """
    ns = normalize_symbol(str(raw or "").strip())
    if not ns:
        return
    st.session_state.symbol = ns
    st.session_state._pending_symbol = ns
    if open_sop:
        st.session_state._goto_sop = True


# Apply pending symbol → input box BEFORE the widget is instantiated
if "_pending_symbol" in st.session_state:
    _ps = normalize_symbol(str(st.session_state.pop("_pending_symbol") or ""))
    if _ps:
        st.session_state.symbol = _ps
        st.session_state.symbol_box = _ps

# Jump to 投资SOP / 我已买入 (set by buttons; must run BEFORE selectbox)
if st.session_state.pop("_goto_sop", False):
    st.session_state.nav_page = "投资SOP"
if st.session_state.pop("_goto_hold", False):
    st.session_state.nav_page = "我已买入"

# Seed text box / nav once
if "symbol_box" not in st.session_state:
    st.session_state.symbol_box = str(st.session_state.symbol)
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "投资SOP"
if "nav_more" not in st.session_state:
    st.session_state.nav_more = PAGE_MORE[0]

# 旧 session 若停在「更多」子页，主选择框映射为「更多…」
# 「期权价差」已升到主菜单：从更多里点过来的旧状态直接打开该页
if st.session_state.get("nav_more") == "期权价差":
    st.session_state.nav_page = "期权价差"
    st.session_state.nav_more = PAGE_MORE[0]
_raw_nav = st.session_state.get("nav_page", "投资SOP")
if _raw_nav in PAGE_MORE:
    st.session_state.nav_more = _raw_nav
    st.session_state.nav_page = "更多…"
elif _raw_nav not in PAGE_MAIN:
    st.session_state.nav_page = "投资SOP"

# ---- sidebar ----
with st.sidebar:
    st.markdown("### 📊 短线助手")
    st.caption("主流程：计划 → 已买入 → 扫描 · 美股")

    page_main = st.selectbox(
        "功能页面",
        PAGE_MAIN,
        key="nav_page",
    )
    page = page_main
    if page_main == "更多…":
        page = st.selectbox(
            "更多工具",
            PAGE_MORE,
            key="nav_more",
            help="进阶/旧页，日常可不用",
        )

    if page_main != "短炒工作台":
        st.divider()
        st.text_input(
            "股票代码",
            key="symbol_box",
            placeholder="AAPL / NVDA / SPY",
            help="Enter 或「应用」；快速选择会打开投资SOP。",
            on_change=_on_symbol_box_change,
        )
        apply_col, tip_col = st.columns([1, 1.2])
        with apply_col:
            if st.button("应用代码", width="stretch", key="btn_apply_symbol"):
                box = str(st.session_state.get("symbol_box") or "").strip()
                if box:
                    ns = normalize_symbol(box)
                    if ns:
                        st.session_state.symbol = ns
                        st.session_state._goto_sop = True
                    st.rerun()
                else:
                    st.warning("请输入代码")
        with tip_col:
            cur = str(st.session_state.get("symbol") or "")
            if cur and not is_us_symbol(cur):
                st.caption(f"当前 `{cur}` 非美股")
            else:
                st.caption(f"当前：`{cur or '—'}`")

        period_label = st.selectbox("时间范围", list(PERIOD_MAP.keys()), index=3)
        interval_label = st.selectbox("K线周期", list(INTERVAL_MAP.keys()), index=0)
        period = PERIOD_MAP[period_label]
        interval = INTERVAL_MAP[interval_label]

        st.caption(f"SOP build: `{_SOP_BUILD}`")
        st.caption("Cloud 须与本地同一 build；不同=未部署最新代码")
        with st.expander("v1 定版 · 怎么用（别再加指标）", expanded=False):
            st.markdown(
                """
**已定版，暂停加功能。** 目标：少乱做，不是保证暴利。

**未买（投资SOP / 扫描）**  
1. 开着「极简」  
2. 只看 **三灯 + 主因 + 挂单E / 止蚀S / 目标T**  
3. 有红灯 → 不做；三绿可做；有黄最多试仓  
4. **限价挂 E，勿市价追**

**已买**  
1. 对照计划止蚀 / T1  
2. 破止蚀走；到 T1 减半保本  
3. 不摊平、不止蚀下移

**每周**  
- 写交易日志  
- 只复盘执行，不换系统  

**财报（Yahoo 日历，免费）**  
- ≤3 天 → 暂缓新仓  
- 4–14 天 → 最多试仓  

**数据**  
- 够用：K线 + 财报日；可选 FRED / Finnhub key  
- 不接社交情绪等噪音源  

**期望**  
- 可提升纪律与过滤；**不保证赚钱**  
- 多数日子空手 = 正常
"""
                )

        st.divider()
        st.markdown("**快速选择**")
        try:
            from stock_service import QUICK_PIN
        except Exception:
            QUICK_PIN = ["MU", "SNDK"]
        pin_syms = [normalize_symbol(p) for p in QUICK_PIN]
        # 常用：MU / SNDK 固定第一行（一眼点进 SOP）
        st.caption("常用")
        pin_cols = st.columns(len(pin_syms) if pin_syms else 2)
        for i, s in enumerate(pin_syms):
            if pin_cols[i].button(
                s,
                key=f"quick_pin_{s}",
                width="stretch",
                type="primary",
            ):
                # 确保在自选里
                if s not in st.session_state.watchlist:
                    st.session_state.watchlist = _ensure_quick_pins(
                        list(st.session_state.watchlist) + [s]
                    )
                    try:
                        from views.common import save_watchlist

                        save_watchlist(st.session_state.watchlist)
                    except Exception:
                        pass
                _request_symbol(s, open_sop=True)
                st.rerun()
        # 其余自选（不含已显示的 pin，避免重复 key）
        rest = [s for s in st.session_state.watchlist if s not in pin_syms][:10]
        if rest:
            st.caption("自选")
            cols = st.columns(2)
            for i, s in enumerate(rest):
                if cols[i % 2].button(s, key=f"quick_{s}", width="stretch"):
                    _request_symbol(s, open_sop=True)
                    st.rerun()

        st.divider()
        st.caption("本金默认 5万 HKD · 非投资建议")


symbol = str(st.session_state.get("symbol") or "").strip()
if not symbol:
    box = str(st.session_state.get("symbol_box") or "").strip()
    symbol = normalize_symbol(box) if box else st.session_state.watchlist[0]
    st.session_state.symbol = symbol

# ---- router ----
if page == "投资SOP":
    render_sop(symbol, period, interval, period_label, interval_label)
elif page == "短炒工作台":
    render_workbench()
elif page == "我已买入":
    render_hold_page(symbol, period=period, interval=interval)
elif page == "Watchlist扫描":
    render_scan(period, interval, period_label)
elif page == "开市超短扫描":
    render_intraday_scan()
elif page == "策略验证":
    render_strategy_validation(symbol)
elif page == "行情看板":
    render_dashboard(symbol, period, interval, period_label, interval_label)
elif page == "多空分析":
    render_bias_page(symbol, period, interval, period_label, interval_label)
elif page == "综合分析":
    render_extra(symbol, period, interval, period_label, interval_label)
elif page == "入场与目标价":
    render_entry(symbol, period, interval, period_label, interval_label)
elif page == "技术分析":
    render_technical(symbol, period, interval, period_label, interval_label)
elif page == "多股对比":
    render_compare(period, interval, period_label)
elif page == "自选股":
    render_watchlist(period, interval)
elif page == "期权价差":
    render_options(symbol)
