# -*- coding: utf-8 -*-
"""iPhone / mobile-friendly Streamlit helpers + dark theme CSS."""

from __future__ import annotations

from typing import Any

import streamlit as st

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
  height: env(safe-area-inset-bottom, 0);
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
    padding-top: 0.5rem !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
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
