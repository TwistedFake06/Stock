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
    from views.options_page import render_options
    from views.scan_page import render_scan
    from views.sop_page import render_sop
    from views.technical_page import render_technical
    from views.watchlist_page import render_watchlist
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
    page_title="投资SOP · 股票分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",  # better for iPhone
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "投资SOP · 实盘辅助 · Streamlit Cloud",
    },
)

inject_ios_safari_support()
inject_mobile_css()

# ---- session state ----
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()
else:
    # Drop non-US from quick-select / saved list
    st.session_state.watchlist = filter_us_only(list(st.session_state.watchlist))
if "symbol" not in st.session_state:
    st.session_state.symbol = st.session_state.watchlist[0]
else:
    # If current symbol is non-US, snap to first US watch item
    if not is_us_symbol(str(st.session_state.symbol)):
        st.session_state.symbol = st.session_state.watchlist[0]

# ---- sidebar ----
with st.sidebar:
    st.markdown("### 📊 投资 SOP")
    st.caption("首页 = 投资SOP · 快速选择仅美股 · 资金 HKD")

    page = st.selectbox(
        "功能页面",
        [
            "投资SOP",
            "Watchlist扫描",
            "期权价差",
            "行情看板",
            "多空分析",
            "入场与目标价",
            "综合分析",
            "技术分析",
            "多股对比",
            "自选股",
        ],
        index=0,
    )

    st.divider()
    symbol_input = st.text_input(
        "股票代码",
        value=st.session_state.symbol,
        placeholder="AAPL / NVDA / SPY",
        help="快速选择仅美股；手动可输入其他 Yahoo 代码",
    )
    if symbol_input:
        st.session_state.symbol = normalize_symbol(symbol_input)

    period_label = st.selectbox("时间范围", list(PERIOD_MAP.keys()), index=3)
    interval_label = st.selectbox("K线周期", list(INTERVAL_MAP.keys()), index=0)
    period = PERIOD_MAP[period_label]
    interval = INTERVAL_MAP[interval_label]

    st.divider()
    st.markdown("**快速选择（美股）**")
    cols = st.columns(2)
    for i, s in enumerate(st.session_state.watchlist[:12]):
        if cols[i % 2].button(s, key=f"quick_{s}", use_container_width=True):
            st.session_state.symbol = s
            st.rerun()

    st.divider()
    st.caption(
        "数据: Yahoo Finance · 本金默认 50,000 HKD · 不构成投资建议"
    )


symbol = st.session_state.symbol

# ---- router (thin shell for Cloud + local) ----
if page == "投资SOP":
    render_sop(symbol, period, interval, period_label, interval_label)
elif page == "Watchlist扫描":
    render_scan(period, interval, period_label)
elif page == "行情看板":
    render_dashboard(symbol, period, interval, period_label, interval_label)
elif page == "多空分析":
    render_bias_page(symbol, period, interval, period_label, interval_label)
elif page == "综合分析":
    render_extra(symbol, period, interval, period_label, interval_label)
elif page == "入场与目标价":
    render_entry(symbol, period, interval, period_label, interval_label)
elif page == "期权价差":
    render_options(symbol)
elif page == "技术分析":
    render_technical(symbol, period, interval, period_label, interval_label)
elif page == "多股对比":
    render_compare(period, interval, period_label)
elif page == "自选股":
    render_watchlist(period, interval)
