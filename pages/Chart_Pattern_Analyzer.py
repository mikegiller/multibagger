import streamlit as st
import yfinance as yf
import os
import yfinance.cache as _yfc
os.makedirs(os.path.join(os.path.expanduser('~'), '.cache', 'py-yfinance'), exist_ok=True)
_yfc._TzCacheManager._tz_cache = _yfc._TzCacheDummy()
_yfc._CookieCacheManager._Cookie_cache = _yfc._CookieCacheDummy()
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

st.set_page_config(page_title="Chart Pattern Analyzer", layout="wide")
st.title("Chart Pattern Analyzer")

# === Favorites ===
def _load_favorites():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "favorites.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return ["SPY", "QQQ", "VOO", "XLK", "NVDA"]

_favs = _load_favorites()

# === Session State ===
for k, v in [("cpa_ticker", "SPY"), ("cpa_period", "1y"), ("cpa_horizon", "3 Months")]:
    if k not in st.session_state:
        st.session_state[k] = v

if "fav" in st.query_params and st.query_params["fav"] in _favs:
    st.session_state.cpa_ticker = st.query_params["fav"]
    del st.query_params["fav"]

# === Sidebar — data settings only ===
with st.sidebar:
    st.header("Settings")

    period_map = {
        "1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months",
        "1y": "1 Year", "2y": "2 Years", "5y": "5 Years",
    }
    period = st.selectbox(
        "History Period",
        options=list(period_map.keys()),
        format_func=lambda x: period_map[x],
        index=3, key="cpa_period",
    )

    horizon_map = {"1 Month": 21, "3 Months": 65, "6 Months": 130}
    horizon_label = st.selectbox(
        "Projection Horizon",
        options=list(horizon_map.keys()),
        index=1, key="cpa_horizon",
    )
    horizon_bars = horizon_map[horizon_label]

    st.markdown("---")
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

# === Ticker input ===
_fav_placeholder = st.empty()
ticker = st.text_input("Ticker", key="cpa_ticker", max_chars=12).upper().strip()
if not ticker:
    st.stop()

_fav_links = ", ".join([
    f'<a href="?fav={f}" target="_self" style="text-decoration:none">{f}</a>'
    for f in _favs
])
_fav_placeholder.markdown(f"**Favorites:** {_fav_links}", unsafe_allow_html=True)


# === Data fetch ===
@st.cache_data(ttl=900, show_spinner="Fetching market data...")
def fetch_data(ticker, period):
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


df = fetch_data(ticker, period)
if df is None or df.empty:
    st.error(f"No data found for **{ticker}**. Check the ticker symbol.")
    st.stop()

df["_bar"] = np.arange(len(df))
current_price = float(df["Close"].iloc[-1])
n_bars = len(df)

future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon_bars)
future_bar_idx = np.arange(n_bars, n_bars + horizon_bars)


# === Swing detection ===
def find_swings(df, window=5):
    roll = window * 2 + 1
    high_max = df["High"].rolling(roll, center=True, min_periods=roll // 2).max()
    low_min  = df["Low"].rolling(roll, center=True, min_periods=roll // 2).min()
    highs = df[df["High"] == high_max][["High", "_bar"]].rename(
        columns={"High": "price", "_bar": "bar_idx"}).tail(4)
    lows  = df[df["Low"] == low_min][["Low", "_bar"]].rename(
        columns={"Low": "price", "_bar": "bar_idx"}).tail(4)
    return highs, lows


swing_highs, swing_lows = find_swings(df)


# === Pattern functions ===

def calc_linreg(df, future_bar_idx):
    x = df["_bar"].values
    y = df["Close"].values
    valid = ~np.isnan(y)
    if valid.sum() < 10:
        return None
    coeffs = np.polyfit(x[valid], y[valid], 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    y_hist = np.polyval(coeffs, x)
    std = float((y[valid] - y_hist[valid]).std())
    y_proj = slope * future_bar_idx + intercept
    return {
        "slope": slope, "intercept": intercept, "std": std,
        "y_hist": y_hist, "y_proj": y_proj,
        "last_center": float(y_hist[-1]),
    }


def calc_trendlines(df, swing_highs, swing_lows, future_dates, future_bar_idx):
    result = {}
    for key, pts in [("resist", swing_highs), ("support", swing_lows)]:
        if pts is None or len(pts) < 2:
            result[key] = None
            continue
        i1, p1 = int(pts.iloc[-2]["bar_idx"]), float(pts.iloc[-2]["price"])
        i2, p2 = int(pts.iloc[-1]["bar_idx"]), float(pts.iloc[-1]["price"])
        if i2 == i1:
            result[key] = None
            continue
        slope = (p2 - p1) / (i2 - i1)
        intercept = p2 - slope * i2
        bars_hist = df["_bar"].iloc[i1:].values
        result[key] = {
            "slope": slope, "intercept": intercept,
            "x_hist": df.index[i1:],
            "y_hist": slope * bars_hist + intercept,
            "x_proj": future_dates,
            "y_proj": slope * future_bar_idx + intercept,
            "last_val": float(slope * (n_bars - 1) + intercept),
        }
    return result


def calc_moving_averages(df, future_dates, future_bar_idx, periods):
    colors = {20: "#ff9800", 50: "#29b6f6", 100: "#ce93d8", 200: "#a5d6a7"}
    result = {}
    for p in periods:
        sma = df["Close"].rolling(p, min_periods=max(p // 4, 5)).mean()
        non_nan = sma.dropna()
        if non_nan.empty:
            continue
        last_val = float(non_nan.iloc[-1])
        recent = non_nan.iloc[-10:].values
        ma_slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 2 else 0.0
        result[p] = {
            "slope": ma_slope, "last_val": last_val,
            "color": colors.get(p, "#ffffff"),
            "x_hist": df.index, "y_hist": sma.values,
            "x_proj": future_dates,
            "y_proj": last_val + ma_slope * np.arange(1, len(future_bar_idx) + 1),
        }
    return result


def calc_fibonacci(swing_highs, swing_lows):
    if any(x is None or x.empty for x in [swing_highs, swing_lows]):
        return None
    hi = float(swing_highs["price"].max())
    lo = float(swing_lows["price"].min())
    if hi <= lo:
        return None
    rng = hi - lo
    return {
        "0.0%":   hi,
        "23.6%":  hi - 0.236 * rng,
        "38.2%":  hi - 0.382 * rng,
        "50.0%":  hi - 0.500 * rng,
        "61.8%":  hi - 0.618 * rng,
        "78.6%":  hi - 0.786 * rng,
        "100.0%": lo,
    }


def calc_bollinger(df, future_dates, future_bar_idx, period=20):
    sma   = df["Close"].rolling(period, min_periods=period // 2).mean()
    std_s = df["Close"].rolling(period, min_periods=period // 2).std()
    non_nan_sma = sma.dropna()
    non_nan_std = std_s.dropna()
    if non_nan_sma.empty or non_nan_std.empty:
        return None
    last_sma = float(non_nan_sma.iloc[-1])
    last_std = float(non_nan_std.iloc[-1])
    recent = non_nan_sma.iloc[-10:].values
    sma_slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 2 else 0.0
    proj_sma = last_sma + sma_slope * np.arange(1, len(future_bar_idx) + 1)
    return {
        "slope": sma_slope, "last_sma": last_sma, "last_std": last_std,
        "last_upper": last_sma + 2 * last_std,
        "last_lower": last_sma - 2 * last_std,
        "x_hist": df.index,
        "y_sma": sma.values,
        "y_upper": (sma + 2 * std_s).values,
        "y_lower": (sma - 2 * std_s).values,
        "x_proj": future_dates,
        "y_proj_sma": proj_sma,
        "y_proj_upper": proj_sma + 2 * last_std,
        "y_proj_lower": proj_sma - 2 * last_std,
    }


def calc_sr(swing_highs, swing_lows):
    resist  = swing_highs["price"].tolist() if swing_highs is not None and not swing_highs.empty else []
    support = swing_lows["price"].tolist()  if swing_lows  is not None and not swing_lows.empty  else []
    return {"resistance": resist, "support": support}


# === Overlay checkboxes (inline, above chart) ===
st.markdown("**Overlays**")

_oc = st.columns(7)
with _oc[0]:
    st.caption("Linear Regression")
    show_lr_center = st.checkbox("LR Center",    value=True,  key="cpa_lr_center")
    show_lr_1std   = st.checkbox("LR ±1σ",       value=False, key="cpa_lr_1std")
    show_lr_2std   = st.checkbox("LR ±2σ",       value=False, key="cpa_lr_2std")
with _oc[1]:
    st.caption("Trend Lines")
    show_tl_resist  = st.checkbox("Resistance",  value=False, key="cpa_tl_resist")
    show_tl_support = st.checkbox("Support",     value=False, key="cpa_tl_support")
with _oc[2]:
    st.caption("Moving Averages")
    ma_20  = st.checkbox("SMA 20",  value=False, key="cpa_ma20")
    ma_50  = st.checkbox("SMA 50",  value=False, key="cpa_ma50")
    ma_100 = st.checkbox("SMA 100", value=False, key="cpa_ma100")
    ma_200 = st.checkbox("SMA 200", value=False, key="cpa_ma200")
with _oc[3]:
    st.caption("Fibonacci")
    show_fib = st.checkbox("Fib Retracement", value=False, key="cpa_fib")
with _oc[4]:
    st.caption("Bollinger (20)")
    show_bb_bands  = st.checkbox("BB Upper/Lower", value=False, key="cpa_bb_bands")
    show_bb_center = st.checkbox("BB Center",      value=False, key="cpa_bb_center")
with _oc[5]:
    st.caption("S/R Horizontal")
    show_sr_resist  = st.checkbox("S/R Resistance", value=False, key="cpa_sr_resist")
    show_sr_support = st.checkbox("S/R Support",    value=False, key="cpa_sr_support")
with _oc[6]:
    st.write("")  # spacer

st.markdown("---")

# === Compute patterns ===
_need_linreg = show_lr_center or show_lr_1std or show_lr_2std
_need_tl     = show_tl_resist or show_tl_support
_need_bb     = show_bb_bands or show_bb_center
_need_sr     = show_sr_resist or show_sr_support

ma_periods = [p for p, on in [(20, ma_20), (50, ma_50), (100, ma_100), (200, ma_200)] if on]

linreg_data = calc_linreg(df, future_bar_idx) if _need_linreg else None
tl_data     = calc_trendlines(df, swing_highs, swing_lows, future_dates, future_bar_idx) if _need_tl else None
ma_data     = calc_moving_averages(df, future_dates, future_bar_idx, ma_periods) if ma_periods else None
fib_data    = calc_fibonacci(swing_highs, swing_lows) if show_fib else None
bb_data     = calc_bollinger(df, future_dates, future_bar_idx) if _need_bb else None
sr_data     = calc_sr(swing_highs, swing_lows) if _need_sr else None


# === Build chart ===
fig = make_subplots(
    rows=2, cols=1, row_heights=[0.75, 0.25],
    vertical_spacing=0.04, shared_xaxes=True,
)

# ── Band fills (added first so they sit behind everything) ──────────────
if linreg_data and show_lr_2std:
    std = linreg_data["std"]
    all_x = df.index.append(pd.DatetimeIndex(future_dates))
    u2 = np.concatenate([linreg_data["y_hist"] + 2 * std, linreg_data["y_proj"] + 2 * std])
    l2 = np.concatenate([linreg_data["y_hist"] - 2 * std, linreg_data["y_proj"] - 2 * std])
    fig.add_trace(go.Scatter(x=all_x, y=u2, line=dict(width=0),
                             showlegend=False, hoverinfo="skip", name="_lr_u"), row=1, col=1)
    fig.add_trace(go.Scatter(x=all_x, y=l2, line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(126,207,255,0.05)",
                             showlegend=False, hoverinfo="skip", name="_lr_l"), row=1, col=1)

if bb_data and show_bb_bands:
    all_x_bb = df.index.append(pd.DatetimeIndex(future_dates))
    bb_u = np.concatenate([bb_data["y_upper"], bb_data["y_proj_upper"]])
    bb_l = np.concatenate([bb_data["y_lower"], bb_data["y_proj_lower"]])
    fig.add_trace(go.Scatter(x=all_x_bb, y=bb_u, line=dict(width=0),
                             showlegend=False, hoverinfo="skip", name="_bb_u"), row=1, col=1)
    fig.add_trace(go.Scatter(x=all_x_bb, y=bb_l, line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(120,144,156,0.08)",
                             showlegend=False, hoverinfo="skip", name="_bb_l"), row=1, col=1)

# ── Candlestick ─────────────────────────────────────────────────────────
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    increasing_line_color="#00c853", decreasing_line_color="#d50000",
    increasing_fillcolor="#00c853", decreasing_fillcolor="#d50000",
    line=dict(width=1.5), name="Price",
), row=1, col=1)

fig.add_hline(
    y=current_price, line_dash="dash", line_color="#448aff", line_width=1,
    annotation_text=f"${current_price:,.2f}",
    annotation_position="top right", annotation_font_size=10,
    row=1, col=1,
)

# ── Linear Regression Channel ────────────────────────────────────────────
if linreg_data:
    std = linreg_data["std"]

    if show_lr_center:
        fig.add_trace(go.Scatter(
            x=df.index, y=linreg_data["y_hist"],
            line=dict(color="#7ecfff", width=1.5), name="LinReg Center", opacity=0.75,
            hovertemplate="LinReg: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=future_dates, y=linreg_data["y_proj"],
            line=dict(color="#7ecfff", width=1.5, dash="dot"),
            showlegend=False, hovertemplate="LinReg Proj: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    if show_lr_1std:
        for y_hist, y_proj in [(linreg_data["y_hist"] + std, linreg_data["y_proj"] + std),
                                (linreg_data["y_hist"] - std, linreg_data["y_proj"] - std)]:
            fig.add_trace(go.Scatter(
                x=df.index, y=y_hist,
                line=dict(color="rgba(126,207,255,0.4)", width=0.8, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=future_dates, y=y_proj,
                line=dict(color="rgba(126,207,255,0.4)", width=0.8, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

    if show_lr_2std:
        for y_hist, y_proj in [(linreg_data["y_hist"] + 2*std, linreg_data["y_proj"] + 2*std),
                                (linreg_data["y_hist"] - 2*std, linreg_data["y_proj"] - 2*std)]:
            fig.add_trace(go.Scatter(
                x=df.index, y=y_hist,
                line=dict(color="rgba(126,207,255,0.2)", width=0.6, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=future_dates, y=y_proj,
                line=dict(color="rgba(126,207,255,0.2)", width=0.6, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

# ── Bollinger Band lines ─────────────────────────────────────────────────
if bb_data:
    if show_bb_bands:
        for y_hist, y_proj, lbl in [
            (bb_data["y_upper"], bb_data["y_proj_upper"], "BB Upper"),
            (bb_data["y_lower"], bb_data["y_proj_lower"], "BB Lower"),
        ]:
            fig.add_trace(go.Scatter(
                x=df.index, y=y_hist,
                line=dict(color="#78909c", width=1),
                name=lbl, opacity=0.85,
                hovertemplate=f"{lbl}: $%{{y:,.2f}}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=future_dates, y=y_proj,
                line=dict(color="#78909c", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

    if show_bb_center:
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_data["y_sma"],
            line=dict(color="#78909c", width=1, dash="dash"),
            name="BB SMA", opacity=0.85,
            hovertemplate="BB SMA: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=future_dates, y=bb_data["y_proj_sma"],
            line=dict(color="#78909c", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

# ── Moving Averages ──────────────────────────────────────────────────────
if ma_data:
    for p, md in sorted(ma_data.items()):
        fig.add_trace(go.Scatter(
            x=md["x_hist"], y=md["y_hist"],
            line=dict(color=md["color"], width=1.3),
            name=f"SMA {p}", opacity=0.9,
            hovertemplate=f"SMA {p}: $%{{y:,.2f}}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=md["x_proj"], y=md["y_proj"],
            line=dict(color=md["color"], width=1.3, dash="dot"),
            showlegend=False,
            hovertemplate=f"SMA {p} Proj: $%{{y:,.2f}}<extra></extra>",
        ), row=1, col=1)

# ── Swing trend lines ────────────────────────────────────────────────────
if tl_data:
    for key, flag, color, lbl in [
        ("resist",  show_tl_resist,  "#ff4d4d", "Trend Resistance"),
        ("support", show_tl_support, "#4ecdc4", "Trend Support"),
    ]:
        td = tl_data.get(key)
        if td and flag:
            fig.add_trace(go.Scatter(
                x=td["x_hist"], y=td["y_hist"],
                line=dict(color=color, width=1.5), name=lbl,
                hovertemplate=f"{lbl}: $%{{y:,.2f}}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=td["x_proj"], y=td["y_proj"],
                line=dict(color=color, width=1.5, dash="dot"),
                showlegend=False,
                hovertemplate=f"{lbl} Proj: $%{{y:,.2f}}<extra></extra>",
            ), row=1, col=1)

# ── S/R horizontal lines ─────────────────────────────────────────────────
if sr_data:
    if show_sr_resist:
        for level in sr_data.get("resistance", []):
            fig.add_hline(y=level, line_width=1, line_color="#ff4d4d", line_dash="dash",
                          annotation_text=f"R ${level:,.2f}", annotation_position="top right",
                          annotation_font_size=9, row=1, col=1)
    if show_sr_support:
        for level in sr_data.get("support", []):
            fig.add_hline(y=level, line_width=1, line_color="#4ecdc4", line_dash="dash",
                          annotation_text=f"S ${level:,.2f}", annotation_position="bottom right",
                          annotation_font_size=9, row=1, col=1)

# ── Fibonacci lines ──────────────────────────────────────────────────────
if fib_data:
    fib_style = {
        "0.0%":   "rgba(192,192,192,0.35)",
        "23.6%":  "rgba(192,192,192,0.55)",
        "38.2%":  "rgba(192,192,192,0.55)",
        "50.0%":  "rgba(192,192,192,0.55)",
        "61.8%":  "#ffd700",
        "78.6%":  "rgba(192,192,192,0.55)",
        "100.0%": "rgba(192,192,192,0.35)",
    }
    for lbl, level in fib_data.items():
        fig.add_hline(
            y=level, line_width=0.8,
            line_color=fib_style.get(lbl, "rgba(192,192,192,0.5)"),
            line_dash="dot",
            annotation_text=f"Fib {lbl}  ${level:,.2f}",
            annotation_position="top left",
            annotation_font_size=8,
            row=1, col=1,
        )

# ── Volume ───────────────────────────────────────────────────────────────
fig.add_trace(go.Bar(
    x=df.index, y=df["Volume"],
    marker_color="rgba(100,181,246,0.4)",
    name="Volume", showlegend=False,
), row=2, col=1)

avg_vol = df["Volume"].rolling(20, min_periods=5).mean()
spike_mask = df["Volume"] > 1.8 * avg_vol
if spike_mask.any():
    fig.add_trace(go.Scatter(
        x=df.index[spike_mask],
        y=df.loc[spike_mask, "Volume"] * 1.05,
        mode="markers",
        marker=dict(symbol="diamond", size=10, color="yellow",
                    line=dict(width=1, color="black")),
        name="Vol Spike", showlegend=False,
        hovertemplate="Spike: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

# ── Projection zone ──────────────────────────────────────────────────────
last_ts  = df.index[-1]
last_proj = future_dates[-1]
fig.add_vrect(x0=last_ts, x1=last_proj,
              fillcolor="rgba(180,220,255,0.04)", layer="below", line_width=0)
# Draw the boundary as a thin Scatter line instead of add_vline to avoid
# Plotly's annotation positioning bug with datetime string x values.
fig.add_trace(go.Scatter(
    x=[last_ts, last_ts],
    y=[df[["Low", "High"]].min().min(), df[["Low", "High"]].max().max()],
    mode="lines",
    line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
    showlegend=False, hoverinfo="skip", name="_proj_vline",
), row=1, col=1)
fig.add_annotation(
    x=last_ts, y=1, yref="paper",
    text="  Projection →",
    showarrow=False,
    font=dict(color="rgba(255,255,255,0.4)", size=10),
    xanchor="left", yanchor="top",
)

# ── Layout ───────────────────────────────────────────────────────────────
fig.update_layout(
    height=820,
    template="plotly_dark",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10)),
    hovermode="x unified",
    margin=dict(t=100, b=50, l=60, r=60),
    dragmode=False,
    xaxis_rangeslider_visible=False,
    title=dict(
        text=f"{ticker} — Chart Pattern Analyzer  |  Current: ${current_price:,.2f}",
        x=0.02, xanchor="left", font=dict(size=16),
    ),
)
fig.update_xaxes(
    fixedrange=True,
    rangebreaks=[dict(bounds=["sat", "mon"])],
    showgrid=True, gridcolor="rgba(255,255,255,0.05)",
)
fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="rgba(255,255,255,0.05)")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# === Projection Table ====================================================

def build_projection_table(current_price, horizon_bars, linreg, tl, ma, fib, bb, sr,
                           show_lr_center, show_lr_2std,
                           show_tl_resist, show_tl_support,
                           show_bb_bands, show_bb_center,
                           show_sr_resist, show_sr_support):
    TARGETS = {"+1 Week": 5, "+2 Weeks": 10, "+1 Month": 21, "+3 Months": 65}
    rows = []

    def proj(last_val, slope, offset):
        return None if offset > horizon_bars else last_val + slope * offset

    def fmt(v):
        return "N/A" if v is None else f"${v:,.2f}"

    def direction(slope):
        if slope > 0.05:   return "Rising ↑"
        if slope < -0.05:  return "Falling ↓"
        return "Flat →"

    def vs_curr(last_val, slope, offset=21):
        v = proj(last_val, slope, offset)
        if v is None:
            return "N/A"
        pct = (v - current_price) / current_price * 100
        return f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"

    def add_row(pattern, last_val, slope):
        rows.append({
            "Pattern": pattern,
            "Current": fmt(last_val),
            **{t: fmt(proj(last_val, slope, off)) for t, off in TARGETS.items()},
            "Direction": direction(slope),
            "vs Current (+1M)": vs_curr(last_val, slope),
        })

    if linreg:
        lc, sl, std = linreg["last_center"], linreg["slope"], linreg["std"]
        if show_lr_center: add_row("LinReg Center", lc, sl)
        if show_lr_2std:
            add_row("LinReg +2σ", lc + 2 * std, sl)
            add_row("LinReg -2σ", lc - 2 * std, sl)

    if tl:
        for key, flag, lbl in [("resist", show_tl_resist, "Trend Resistance"),
                                 ("support", show_tl_support, "Trend Support")]:
            td = tl.get(key)
            if td and flag:
                add_row(lbl, td["last_val"], td["slope"])

    if ma:
        for p, md in sorted(ma.items()):
            add_row(f"SMA {p}", md["last_val"], md["slope"])

    if bb:
        if show_bb_bands:
            add_row("BB Upper", bb["last_upper"], bb["slope"])
            add_row("BB Lower", bb["last_lower"], bb["slope"])
        if show_bb_center:
            add_row("BB SMA", bb["last_sma"], bb["slope"])

    if fib:
        for lbl, level in fib.items():
            add_row(f"Fib {lbl}", level, 0.0)

    if sr:
        if show_sr_resist:
            for level in sr.get("resistance", []):
                add_row("Resistance", level, 0.0)
        if show_sr_support:
            for level in sr.get("support", []):
                add_row("Support", level, 0.0)

    if not rows:
        return pd.DataFrame()

    cols = ["Pattern", "Current", "+1 Week", "+2 Weeks", "+1 Month", "+3 Months",
            "Direction", "vs Current (+1M)"]
    return pd.DataFrame(rows, columns=cols)


proj_df = build_projection_table(
    current_price, horizon_bars,
    linreg_data, tl_data, ma_data, fib_data, bb_data, sr_data,
    show_lr_center, show_lr_2std,
    show_tl_resist, show_tl_support,
    show_bb_bands, show_bb_center,
    show_sr_resist, show_sr_support,
)

if not proj_df.empty:
    st.subheader(f"Price Projection Table — {ticker}")
    st.caption(
        f"Projections from {df.index[-1].strftime('%b %d, %Y')} "
        f"over {horizon_label} horizon  |  "
        f"Rising/Falling = slope >±$0.05/bar  |  vs Current uses +1 Month projection"
    )

    def _style_direction(v):
        if "Rising" in str(v):   return "color: #00c853"
        if "Falling" in str(v):  return "color: #d50000"
        return "color: #888888"

    def _style_vs(v):
        try:
            pct = float(str(v).replace("%", "").replace("+", ""))
            return "color: #00c853" if pct > 0 else ("color: #d50000" if pct < 0 else "")
        except Exception:
            return ""

    styled = (
        proj_df.style
        .map(_style_direction, subset=["Direction"])
        .map(_style_vs, subset=["vs Current (+1M)"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
