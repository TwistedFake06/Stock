# -*- coding: utf-8 -*-
"""iPhone / mobile-friendly Streamlit helpers + dark theme CSS."""

from __future__ import annotations

from typing import Any

import streamlit as st
import streamlit.components.v1 as components

PLOTLY_CONFIG = {
    "responsive": True,
    "displayModeBar": False,
    "scrollZoom": False,
}

# Dark chart defaults (match Streamlit dark theme)
PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.6)",
    font=dict(size=11, color="#e6edf3"),
    title_font=dict(color="#e6edf3"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        x=0,
        font=dict(size=10, color="#c9d1d9"),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        color="#c9d1d9",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        color="#c9d1d9",
    ),
    margin=dict(l=24, r=12, t=40, b=24),
    autosize=True,
)


def inject_mobile_css() -> None:
    """Global CSS: dark UI + iPhone Safari layout."""
    st.markdown(
        """
<style>
/* ---- Base ---- */
html, body, [class*="css"] {
  -webkit-text-size-adjust: 100%;
  text-size-adjust: 100%;
}
html, body {
  min-height: 100%;
  min-height: -webkit-fill-available;
  overflow-x: hidden;
  overscroll-behavior-y: contain;
}
.stApp {
  min-height: calc(var(--app-vh, 1vh) * 100);
}
.block-container {
  padding-top: 0.8rem !important;
  padding-bottom: 5rem !important;
  padding-left: 0.9rem !important;
  padding-right: 0.9rem !important;
  max-width: 100% !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  min-width: min(100vw, 18rem);
}
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] input {
  min-height: 2.6rem;
  font-size: 1rem !important;
}

/* Metrics — dark cards */
div[data-testid="stMetric"] {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 0.55rem 0.65rem;
  margin-bottom: 0.25rem;
}
div[data-testid="stMetricValue"] {
  font-size: 1.15rem !important;
  word-break: break-word;
  color: #e6edf3 !important;
}
div[data-testid="stMetricLabel"] {
  font-size: 0.78rem !important;
  opacity: 0.8;
  color: #8b949e !important;
}
div[data-testid="stMetricDelta"] {
  font-size: 0.85rem !important;
}

/* Bias / info cards */
.bias-card {
  border-radius: 12px;
  padding: 0.85rem 1rem;
  border: 1px solid #30363d;
  background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
  margin-bottom: 0.65rem;
  box-shadow: 0 1px 0 rgba(255,255,255,0.04);
}
.bias-title {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0;
  line-height: 1.3;
  color: #e6edf3;
}
.bias-sub {
  color: #8b949e;
  margin-top: 0.35rem;
  font-size: 0.9rem;
  line-height: 1.45;
}
.sig-bull { color: #f85149; font-weight: 600; }
.sig-bear { color: #3fb950; font-weight: 600; }
.sig-flat { color: #8b949e; font-weight: 600; }

/* ---- Beauty: options home hero ---- */
.hero-wrap {
  position: relative;
  border-radius: 18px;
  padding: 1.25rem 1.2rem 1.15rem;
  margin: 0 0 1rem 0;
  overflow: hidden;
  border: 1px solid rgba(100,181,246,0.22);
  background:
    radial-gradient(120% 80% at 0% 0%, rgba(100,181,246,0.18) 0%, transparent 55%),
    radial-gradient(90% 70% at 100% 20%, rgba(187,134,252,0.14) 0%, transparent 50%),
    linear-gradient(160deg, #12171f 0%, #0d1117 55%, #0a0e14 100%);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.hero-kicker {
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #64b5f6;
  font-weight: 600;
  margin: 0 0 0.35rem 0;
}
.hero-title {
  font-size: 1.55rem;
  font-weight: 800;
  margin: 0;
  line-height: 1.25;
  color: #f0f6fc;
  letter-spacing: -0.02em;
}
.hero-desc {
  margin: 0.45rem 0 0 0;
  color: #8b949e;
  font-size: 0.92rem;
  line-height: 1.5;
}
.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.85rem;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  border: 1px solid #30363d;
  background: rgba(22,27,34,0.9);
  color: #c9d1d9;
}
.pill-blue { border-color: rgba(100,181,246,0.45); color: #90caf9; background: rgba(100,181,246,0.12); }
.pill-green { border-color: rgba(63,185,80,0.45); color: #7ee787; background: rgba(63,185,80,0.12); }
.pill-red { border-color: rgba(248,81,73,0.45); color: #ff7b72; background: rgba(248,81,73,0.12); }
.pill-purple { border-color: rgba(187,134,252,0.45); color: #d2a8ff; background: rgba(187,134,252,0.12); }
.pill-amber { border-color: rgba(210,153,34,0.5); color: #e3b341; background: rgba(210,153,34,0.12); }

.glass-card {
  border-radius: 16px;
  padding: 1rem 1.05rem;
  margin-bottom: 0.85rem;
  border: 1px solid #30363d;
  background: linear-gradient(165deg, rgba(22,27,34,0.95) 0%, rgba(13,17,23,0.98) 100%);
  box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.glass-card-accent {
  border-color: rgba(100,181,246,0.35);
  box-shadow: 0 4px 24px rgba(100,181,246,0.08), 0 4px 20px rgba(0,0,0,0.25);
}
.section-label {
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #64b5f6;
  font-weight: 700;
  margin: 0 0 0.4rem 0;
}
.step-row {
  display: flex;
  gap: 0.65rem;
  margin: 0.55rem 0;
  align-items: flex-start;
}
.step-num {
  flex-shrink: 0;
  width: 1.65rem;
  height: 1.65rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 800;
  background: linear-gradient(135deg, #64b5f6, #7c4dff);
  color: #0d1117;
}
.step-body {
  color: #c9d1d9;
  font-size: 0.9rem;
  line-height: 1.45;
  padding-top: 0.15rem;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  margin-top: 0.75rem;
}
@media (min-width: 640px) {
  .kpi-grid { grid-template-columns: repeat(3, 1fr); }
}
.kpi {
  border-radius: 12px;
  padding: 0.65rem 0.7rem;
  background: #0d1117;
  border: 1px solid #30363d;
}
.kpi-label {
  font-size: 0.7rem;
  color: #8b949e;
  margin: 0 0 0.2rem 0;
}
.kpi-value {
  font-size: 1.05rem;
  font-weight: 700;
  color: #e6edf3;
  margin: 0;
  word-break: break-word;
}
.kpi-value.up { color: #ff7b72; }
.kpi-value.down { color: #7ee787; }
.kpi-value.accent { color: #90caf9; }
.divider-soft {
  height: 1px;
  background: linear-gradient(90deg, transparent, #30363d, transparent);
  margin: 0.85rem 0;
}
.mini-note {
  font-size: 0.8rem;
  color: #8b949e;
  line-height: 1.4;
  margin: 0.4rem 0 0 0;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
  overflow-x: auto !important;
  -webkit-overflow-scrolling: touch;
  max-width: 100%;
  border: 1px solid #30363d;
  border-radius: 8px;
}

/* Buttons */
.stButton > button {
  border-radius: 10px;
  min-height: 2.7rem;
  width: 100%;
  border-color: #30363d !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input {
  border-radius: 8px !important;
}
input, select, textarea {
  font-size: 16px !important; /* prevent iOS auto-zoom on focus */
}

/* Radio / tabs */
div[role="radiogroup"] label {
  padding: 0.35rem 0.2rem !important;
  line-height: 1.35;
}
button[data-baseweb="tab"] {
  font-size: 0.85rem !important;
  padding: 0.4rem 0.55rem !important;
  white-space: nowrap;
}
.streamlit-expanderHeader {
  font-size: 0.95rem !important;
  min-height: 2.5rem;
}

/* Plotly */
.js-plotly-plot, .plotly {
  max-width: 100% !important;
}

footer { visibility: hidden; height: 0; }
.block-container::after {
  content: "";
  display: block;
  height: calc(env(safe-area-inset-bottom, 0) + 0.35rem);
}

/* Alerts slightly softer on dark */
div[data-testid="stAlert"] {
  border-radius: 10px;
}

/* Progress bars */
div[data-testid="stProgress"] > div {
  border-radius: 6px;
}

/* Phone */
@media (max-width: 768px) {
  .block-container {
    padding-top: calc(env(safe-area-inset-top, 0) + 0.35rem) !important;
    padding-left: calc(env(safe-area-inset-left, 0) + 0.6rem) !important;
    padding-right: calc(env(safe-area-inset-right, 0) + 0.6rem) !important;
  }
  h1 { font-size: 1.35rem !important; color: #e6edf3 !important; }
  h2 { font-size: 1.15rem !important; color: #e6edf3 !important; }
  h3 { font-size: 1.05rem !important; color: #e6edf3 !important; }
  .bias-title { font-size: 1.1rem !important; }
  .bias-sub { font-size: 0.85rem !important; }
  div[data-testid="stMetricValue"] { font-size: 1.05rem !important; }
  div[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 0.35rem !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    min-width: min(100%, 46%) !important;
    flex: 1 1 46% !important;
  }
  .mobile-stack div[data-testid="column"] {
    min-width: 100% !important;
    flex: 1 1 100% !important;
  }
  section[data-testid="stSidebar"] > div {
    width: 100% !important;
  }
  .element-container iframe {
    max-width: 100% !important;
  }
  .stCaption, small { font-size: 0.8rem !important; color: #8b949e !important; }
}

@media (max-width: 400px) {
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    min-width: 100% !important;
    flex: 1 1 100% !important;
  }
  div[data-testid="stMetricValue"] { font-size: 1rem !important; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def inject_ios_safari_support() -> None:
    """Inject viewport/meta tweaks for iPhone Safari rendering behavior."""
    components.html(
        """
<script>
(() => {
  const ensureMeta = (name, content, key = 'name') => {
    let tag = document.head.querySelector(`meta[${key}="${name}"]`);
    if (!tag) {
      tag = document.createElement('meta');
      tag.setAttribute(key, name);
      document.head.appendChild(tag);
    }
    tag.setAttribute('content', content);
  };

  ensureMeta('viewport', 'width=device-width, initial-scale=1, viewport-fit=cover');
  ensureMeta('apple-mobile-web-app-capable', 'yes');
  ensureMeta('apple-mobile-web-app-status-bar-style', 'black-translucent');
  ensureMeta('theme-color', '#0d1117');

  const setVh = () => {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--app-vh', `${vh}px`);
  };
  setVh();
  window.addEventListener('resize', setVh, { passive: true });
})();
</script>
        """,
        height=0,
        width=0,
    )


def apply_dark_plotly(fig: Any) -> Any:
    """Force dark template + transparent paper for Streamlit dark bg."""
    if fig is None:
        return fig
    try:
        fig.update_layout(**PLOTLY_DARK_LAYOUT)
        # gauge / polar etc.
        if hasattr(fig.layout, "polar") and fig.layout.polar is not None:
            fig.update_polars(
                bgcolor="rgba(14,17,23,0.6)",
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#c9d1d9"),
                radialaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#c9d1d9"),
            )
    except Exception:
        try:
            fig.update_layout(template="plotly_dark")
        except Exception:
            pass
    return fig


def plotly_chart(fig: Any, **kwargs: Any) -> None:
    """Mobile-safe + dark plotly chart wrapper."""
    fig = apply_dark_plotly(fig)
    kwargs.setdefault("use_container_width", True)
    kwargs.setdefault("config", PLOTLY_CONFIG)
    st.plotly_chart(fig, **kwargs)


def metric_grid(items: list[tuple[str, str]], per_row: int = 2) -> None:
    """Phone-friendly metric grid (default 2 per row)."""
    if not items:
        return
    n = max(1, min(per_row, 3))
    for i in range(0, len(items), n):
        chunk = items[i : i + n]
        cols = st.columns(len(chunk))
        for c, it in zip(cols, chunk):
            with c:
                if len(it) >= 3 and it[2] is not None:
                    st.metric(it[0], it[1], it[2])
                else:
                    st.metric(it[0], it[1])


def section_title(text: str) -> None:
    st.markdown(f"### {text}")
