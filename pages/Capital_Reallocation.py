# Capital_Reallocation.py
import streamlit as st
import yfinance as yf
import os
import yfinance.cache as _yfc
os.makedirs(os.path.join(os.path.expanduser('~'), '.cache', 'py-yfinance'), exist_ok=True)
_yfc._TzCacheManager._tz_cache = _yfc._TzCacheDummy()
_yfc._CookieCacheManager._Cookie_cache = _yfc._CookieCacheDummy()
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Capital Reallocation", layout="wide")
st.title("Capital Reallocation")
st.info("**Purpose:** Track where money is flowing across asset classes and sectors using ETF price action as a proxy. Used periodically to read broad market sentiment and identify rotation trends.")

# ── Constants ─────────────────────────────────────────────────────────────────

ASSET_CLASSES = [
    # (ticker, label, asset_class, color)
    ("SPY",  "S&P 500",       "Stocks",      "#1f77b4"),
    ("QQQ",  "Nasdaq 100",    "Stocks",      "#aec7e8"),
    ("TLT",  "Long-Term Tsy", "Bonds",       "#2ca02c"),
    ("AGG",  "Total Bond",    "Bonds",       "#98df8a"),
    ("IBIT", "Bitcoin ETF",   "Crypto",      "#9467bd"),
    ("ETHA", "Ethereum ETF",  "Crypto",      "#c5b0d5"),
    ("GLD",  "Gold",          "Commodities", "#d62728"),
    ("USO",  "Oil",           "Commodities", "#ffbb78"),
]

SECTORS = [
    ("XLK",  "Technology"),
    ("XLF",  "Financials"),
    ("XLV",  "Health Care"),
    ("XLY",  "Cons. Discretionary"),
    ("XLP",  "Cons. Staples"),
    ("XLI",  "Industrials"),
    ("XLE",  "Energy"),
    ("XLC",  "Comm. Services"),
    ("XLB",  "Materials"),
    ("XLRE", "Real Estate"),
    ("XLU",  "Utilities"),
]

PERIOD_LABELS   = ["YTD", "1M", "3M", "6M", "1Y", "2Y"]
HEATMAP_PERIODS = ["1W", "1M", "3M", "6M", "1Y"]

# ── Session State ─────────────────────────────────────────────────────────────

if "cr_period" not in st.session_state:
    st.session_state.cr_period = "YTD"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Period Selector ───────────────────────────────────────────────────────────

btn_cols = st.columns(len(PERIOD_LABELS))
for i, label in enumerate(PERIOD_LABELS):
    with btn_cols[i]:
        btn_type = "primary" if st.session_state.cr_period == label else "secondary"
        if st.button(label, key=f"cr_btn_{label}", use_container_width=True, type=btn_type):
            st.session_state.cr_period = label
            st.rerun()

selected_period = st.session_state.cr_period

# ── Helpers ───────────────────────────────────────────────────────────────────

def _period_start(label: str) -> date:
    today = date.today()
    if label == "YTD":
        return date(today.year, 1, 1)
    days_map = {"1W": 7, "1M": 30, "3M": 91, "6M": 182, "1Y": 365, "2Y": 730}
    return today - timedelta(days=days_map[label])

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_close(tkr: str) -> pd.Series:
    try:
        hist = yf.Ticker(tkr).history(period="2y", interval="1d", auto_adjust=True)
        if hist.empty:
            return pd.Series(dtype=float, name=tkr)
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        s = hist["Close"].copy()
        s.name = tkr
        return s
    except Exception:
        return pd.Series(dtype=float, name=tkr)

def _get_prices(tickers: list) -> pd.DataFrame:
    return pd.concat([_fetch_close(t) for t in tickers], axis=1)

def _pct_return(series: pd.Series, period_label: str):
    s = series.dropna()
    start = pd.Timestamp(_period_start(period_label))
    s = s[s.index >= start]
    if len(s) < 2:
        return None
    return (s.iloc[-1] / s.iloc[0] - 1) * 100

# ── Fetch All Data ────────────────────────────────────────────────────────────

ac_tickers  = [r[0] for r in ASSET_CLASSES]
sec_tickers = [r[0] for r in SECTORS]

with st.spinner("Loading market data..."):
    ac_prices  = _get_prices(ac_tickers)
    sec_prices = _get_prices(sec_tickers)

# ── Section 1: Asset Class Normalized Chart ───────────────────────────────────

st.subheader("Asset Class Performance")

period_start_ts = pd.Timestamp(_period_start(selected_period))
ac_sliced = ac_prices[ac_prices.index >= period_start_ts]

fig_ac = go.Figure()
for tkr, label, asset_class, color in ASSET_CLASSES:
    if tkr not in ac_sliced.columns:
        continue
    s = ac_sliced[tkr].dropna()
    if len(s) < 2:
        continue
    norm = (s / s.iloc[0]) * 100
    fig_ac.add_trace(go.Scatter(
        x=norm.index,
        y=norm.values,
        name=f"{tkr} – {label}",
        line=dict(color=color, width=2),
        hovertemplate=f"<b>{tkr}</b> ({asset_class}): %{{y:.1f}}<extra></extra>",
    ))

fig_ac.add_hline(y=100, line_dash="dot", line_color="gray", line_width=1, opacity=0.4)
fig_ac.update_layout(
    height=420,
    margin=dict(l=0, r=0, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    yaxis_title="Indexed to 100 at period start",
    xaxis_title=None,
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_ac, use_container_width=True)

# Returns summary table
table_rows = []
for tkr, label, asset_class, _ in ASSET_CLASSES:
    if tkr not in ac_prices.columns:
        continue
    row = {"Ticker": tkr, "Name": label, "Class": asset_class}
    for p in ["YTD", "1M", "3M", "6M", "1Y"]:
        v = _pct_return(ac_prices[tkr], p)
        row[p] = f"{v:+.1f}%" if v is not None else "—"
    table_rows.append(row)

st.dataframe(
    pd.DataFrame(table_rows).set_index("Ticker"),
    use_container_width=True,
)

st.divider()

# ── Section 2: Sector Rotation Heatmap + Bar Chart ───────────────────────────

st.subheader("Sector Rotation")

sec_name_map = {r[0]: r[1] for r in SECTORS}
valid_sec    = [t for t in sec_tickers if t in sec_prices.columns]
sec_labels   = [sec_name_map[t] for t in valid_sec]

# Heatmap matrix: sectors × HEATMAP_PERIODS
hmap_z = []
for tkr in valid_sec:
    row = []
    for p in HEATMAP_PERIODS:
        v = _pct_return(sec_prices[tkr], p)
        row.append(round(v, 2) if v is not None else None)
    hmap_z.append(row)

text_labels = [
    [f"{v:+.1f}%" if v is not None else "—" for v in row]
    for row in hmap_z
]
z_filled = [[v if v is not None else 0.0 for v in row] for row in hmap_z]

col1, col2 = st.columns([3, 2])

with col1:
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_filled,
        x=HEATMAP_PERIODS,
        y=sec_labels,
        colorscale="RdYlGn",
        zmid=0,
        text=text_labels,
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate="<b>%{y}</b> – %{x}: %{z:.2f}%<extra></extra>",
        showscale=True,
        colorbar=dict(title="% Chg", thickness=14, len=0.8),
    ))
    fig_heat.update_layout(
        height=420,
        margin=dict(l=0, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col2:
    bar_pairs = [
        (f"{sec_name_map[t]} ({t})", _pct_return(sec_prices[t], selected_period))
        for t in valid_sec
    ]
    bar_pairs = [(n, v) for n, v in bar_pairs if v is not None]
    bar_pairs.sort(key=lambda x: x[1])

    if bar_pairs:
        names  = [p[0] for p in bar_pairs]
        vals   = [p[1] for p in bar_pairs]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]

        fig_bar = go.Figure(go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
        ))
        fig_bar.update_layout(
            title=dict(text=f"Sector Returns ({selected_period})", font=dict(size=14)),
            height=420,
            margin=dict(l=0, r=80, t=40, b=10),
            xaxis_title="% Change",
            yaxis_title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No sector data available for this period.")
