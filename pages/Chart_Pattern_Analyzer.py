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
_defaults = {
    "cpa_ticker": "SPY",
    "cpa_period": "1y",
    "cpa_horizon": "3 Months",
    "cpa_lt_period": "2y",
    "cpa_lt_horizon": "1 Year",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "fav" in st.query_params and st.query_params["fav"] in _favs:
    st.session_state.cpa_ticker = st.query_params["fav"]
    del st.query_params["fav"]

# === Sidebar — Clear Cache only ===
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

# === Shared Ticker Input ===
_fav_placeholder = st.empty()
ticker = st.text_input("Ticker", key="cpa_ticker", max_chars=12).upper().strip()
if not ticker:
    st.stop()

_fav_links = ", ".join([
    f'<a href="?fav={f}" target="_self" style="text-decoration:none">{f}</a>'
    for f in _favs
])
_fav_placeholder.markdown(f"**Favorites:** {_fav_links}", unsafe_allow_html=True)


# === Period % Change Helpers ===
@st.cache_data(ttl=900, show_spinner=False)
def get_period_changes(tkr):
    changes = {}
    try:
        hist = yf.Ticker(tkr).history(period="5y", interval="1d")
        if not hist.empty:
            last = float(hist.Close.iloc[-1])
            for p, bars in [("1mo", 21), ("3mo", 63), ("6mo", 126),
                             ("1y", 252), ("2y", 504), ("5y", len(hist))]:
                idx = min(bars, len(hist))
                if idx > 0:
                    start = float(hist.Close.iloc[-idx])
                    if start > 0:
                        changes[p] = (last - start) / start * 100
    except Exception:
        pass
    return changes


@st.cache_data(ttl=900, show_spinner=False)
def get_lt_period_changes(tkr):
    changes = {}
    try:
        hist = yf.Ticker(tkr).history(period="max", interval="1wk")
        if not hist.empty:
            last = float(hist.Close.iloc[-1])
            for p, bars in [("2y", 104), ("3y", 156), ("5y", 260),
                             ("10y", 520), ("max", len(hist))]:
                idx = min(bars, len(hist))
                if idx > 0:
                    start = float(hist.Close.iloc[-idx])
                    if start > 0:
                        changes[p] = (last - start) / start * 100
    except Exception:
        pass
    return changes


period_changes    = get_period_changes(ticker)
lt_period_changes = get_lt_period_changes(ticker)


# === Data Fetch ===
@st.cache_data(ttl=900, show_spinner="Fetching daily data...")
def fetch_data(tkr, per):
    df = yf.Ticker(tkr).history(period=per, interval="1d", auto_adjust=True)
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


@st.cache_data(ttl=900, show_spinner="Fetching weekly data...")
def fetch_data_lt(tkr, per):
    df = yf.Ticker(tkr).history(period=per, interval="1wk", auto_adjust=True)
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


# === Shared Pattern Functions ===
def find_swings(df, window=5):
    roll = window * 2 + 1
    high_max = df["High"].rolling(roll, center=True, min_periods=roll // 2).max()
    low_min  = df["Low"].rolling(roll, center=True, min_periods=roll // 2).min()
    highs = df[df["High"] == high_max][["High", "_bar"]].rename(
        columns={"High": "price", "_bar": "bar_idx"}).tail(4)
    lows  = df[df["Low"] == low_min][["Low", "_bar"]].rename(
        columns={"Low": "price", "_bar": "bar_idx"}).tail(4)
    return highs, lows


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
    n = len(df)
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
            "last_val": float(slope * (n - 1) + intercept),
        }
    return result


def calc_moving_averages(df, future_dates, future_bar_idx, periods):
    ma_colors = {
        20: "#ff9800",  50: "#29b6f6",  100: "#ce93d8", 200: "#a5d6a7",
        13: "#ff9800",  26: "#29b6f6",  52:  "#ce93d8", 104: "#a5d6a7",
    }
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
            "color": ma_colors.get(p, "#ffffff"),
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
    if sma.dropna().empty or std_s.dropna().empty:
        return None
    last_sma = float(sma.dropna().iloc[-1])
    last_std = float(std_s.dropna().iloc[-1])
    recent = sma.dropna().iloc[-10:].values
    sma_slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 2 else 0.0
    proj_sma = last_sma + sma_slope * np.arange(1, len(future_bar_idx) + 1)
    return {
        "slope": sma_slope, "last_sma": last_sma, "last_std": last_std,
        "last_upper": last_sma + 2 * last_std,
        "last_lower": last_sma - 2 * last_std,
        "x_hist": df.index,
        "y_sma":   sma.values,
        "y_upper": (sma + 2 * std_s).values,
        "y_lower": (sma - 2 * std_s).values,
        "x_proj": future_dates,
        "y_proj_sma":   proj_sma,
        "y_proj_upper": proj_sma + 2 * last_std,
        "y_proj_lower": proj_sma - 2 * last_std,
    }


def calc_sr(swing_highs, swing_lows):
    resist  = swing_highs["price"].tolist() if swing_highs is not None and not swing_highs.empty else []
    support = swing_lows["price"].tolist()  if swing_lows  is not None and not swing_lows.empty  else []
    return {"resistance": resist, "support": support}


# === Chart Builder ===
def build_chart(df, current_price, future_dates, future_bar_idx,
                linreg_data, tl_data, ma_data, fib_data, bb_data, sr_data,
                show, ticker, title_suffix="", is_weekly=False):
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.75, 0.25],
        vertical_spacing=0.04, shared_xaxes=True,
    )

    # Band fills (bottom layer)
    if linreg_data and show.get("lr_2std"):
        std = linreg_data["std"]
        all_x = df.index.append(pd.DatetimeIndex(future_dates))
        u2 = np.concatenate([linreg_data["y_hist"] + 2 * std, linreg_data["y_proj"] + 2 * std])
        l2 = np.concatenate([linreg_data["y_hist"] - 2 * std, linreg_data["y_proj"] - 2 * std])
        fig.add_trace(go.Scatter(x=all_x, y=u2, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip", name="_lr_u"), row=1, col=1)
        fig.add_trace(go.Scatter(x=all_x, y=l2, line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(126,207,255,0.05)",
                                 showlegend=False, hoverinfo="skip", name="_lr_l"), row=1, col=1)

    if bb_data and show.get("bb_bands"):
        all_x_bb = df.index.append(pd.DatetimeIndex(future_dates))
        bb_u = np.concatenate([bb_data["y_upper"], bb_data["y_proj_upper"]])
        bb_l = np.concatenate([bb_data["y_lower"], bb_data["y_proj_lower"]])
        fig.add_trace(go.Scatter(x=all_x_bb, y=bb_u, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip", name="_bb_u"), row=1, col=1)
        fig.add_trace(go.Scatter(x=all_x_bb, y=bb_l, line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(120,144,156,0.08)",
                                 showlegend=False, hoverinfo="skip", name="_bb_l"), row=1, col=1)

    # Candlestick
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

    # Linear Regression Channel
    if linreg_data:
        std = linreg_data["std"]
        if show.get("lr_center"):
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
        if show.get("lr_1std"):
            for yh, yp in [(linreg_data["y_hist"] + std, linreg_data["y_proj"] + std),
                           (linreg_data["y_hist"] - std, linreg_data["y_proj"] - std)]:
                fig.add_trace(go.Scatter(x=df.index, y=yh,
                    line=dict(color="rgba(126,207,255,0.4)", width=0.8, dash="dash"),
                    showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates, y=yp,
                    line=dict(color="rgba(126,207,255,0.4)", width=0.8, dash="dot"),
                    showlegend=False, hoverinfo="skip"), row=1, col=1)
        if show.get("lr_2std"):
            for yh, yp in [(linreg_data["y_hist"] + 2*std, linreg_data["y_proj"] + 2*std),
                           (linreg_data["y_hist"] - 2*std, linreg_data["y_proj"] - 2*std)]:
                fig.add_trace(go.Scatter(x=df.index, y=yh,
                    line=dict(color="rgba(126,207,255,0.2)", width=0.6, dash="dot"),
                    showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates, y=yp,
                    line=dict(color="rgba(126,207,255,0.2)", width=0.6, dash="dot"),
                    showlegend=False, hoverinfo="skip"), row=1, col=1)

    # Bollinger Bands
    if bb_data:
        if show.get("bb_bands"):
            for yh, yp, lbl in [
                (bb_data["y_upper"], bb_data["y_proj_upper"], "BB Upper"),
                (bb_data["y_lower"], bb_data["y_proj_lower"], "BB Lower"),
            ]:
                fig.add_trace(go.Scatter(x=df.index, y=yh,
                    line=dict(color="#78909c", width=1), name=lbl, opacity=0.85,
                    hovertemplate=f"{lbl}: $%{{y:,.2f}}<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=future_dates, y=yp,
                    line=dict(color="#78909c", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip"), row=1, col=1)
        if show.get("bb_center"):
            fig.add_trace(go.Scatter(x=df.index, y=bb_data["y_sma"],
                line=dict(color="#78909c", width=1, dash="dash"),
                name="BB SMA", opacity=0.85,
                hovertemplate="BB SMA: $%{y:,.2f}<extra></extra>"), row=1, col=1)
            fig.add_trace(go.Scatter(x=future_dates, y=bb_data["y_proj_sma"],
                line=dict(color="#78909c", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip"), row=1, col=1)

    # Moving Averages
    if ma_data:
        for p, md in sorted(ma_data.items()):
            lbl = f"SMA {p}w" if is_weekly else f"SMA {p}"
            fig.add_trace(go.Scatter(x=md["x_hist"], y=md["y_hist"],
                line=dict(color=md["color"], width=1.3),
                name=lbl, opacity=0.9,
                hovertemplate=f"{lbl}: $%{{y:,.2f}}<extra></extra>"), row=1, col=1)
            fig.add_trace(go.Scatter(x=md["x_proj"], y=md["y_proj"],
                line=dict(color=md["color"], width=1.3, dash="dot"),
                showlegend=False,
                hovertemplate=f"{lbl} Proj: $%{{y:,.2f}}<extra></extra>"), row=1, col=1)

    # Swing trend lines
    if tl_data:
        for key, flag_key, color, lbl in [
            ("resist",  "tl_resist",  "#ff4d4d", "Trend Resistance"),
            ("support", "tl_support", "#4ecdc4", "Trend Support"),
        ]:
            td = tl_data.get(key)
            if td and show.get(flag_key):
                fig.add_trace(go.Scatter(x=td["x_hist"], y=td["y_hist"],
                    line=dict(color=color, width=1.5), name=lbl,
                    hovertemplate=f"{lbl}: $%{{y:,.2f}}<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=td["x_proj"], y=td["y_proj"],
                    line=dict(color=color, width=1.5, dash="dot"),
                    showlegend=False,
                    hovertemplate=f"{lbl} Proj: $%{{y:,.2f}}<extra></extra>"), row=1, col=1)

    # S/R horizontal lines
    if sr_data:
        if show.get("sr_resist"):
            for level in sr_data.get("resistance", []):
                fig.add_hline(y=level, line_width=1, line_color="#ff4d4d", line_dash="dash",
                              annotation_text=f"R ${level:,.2f}", annotation_position="top right",
                              annotation_font_size=9, row=1, col=1)
        if show.get("sr_support"):
            for level in sr_data.get("support", []):
                fig.add_hline(y=level, line_width=1, line_color="#4ecdc4", line_dash="dash",
                              annotation_text=f"S ${level:,.2f}", annotation_position="bottom right",
                              annotation_font_size=9, row=1, col=1)

    # Fibonacci lines
    if fib_data:
        fib_style = {
            "0.0%":   "rgba(192,192,192,0.35)", "23.6%":  "rgba(192,192,192,0.55)",
            "38.2%":  "rgba(192,192,192,0.55)", "50.0%":  "rgba(192,192,192,0.55)",
            "61.8%":  "#ffd700",                "78.6%":  "rgba(192,192,192,0.55)",
            "100.0%": "rgba(192,192,192,0.35)",
        }
        for lbl, level in fib_data.items():
            fig.add_hline(y=level, line_width=0.8,
                          line_color=fib_style.get(lbl, "rgba(192,192,192,0.5)"),
                          line_dash="dot",
                          annotation_text=f"Fib {lbl}  ${level:,.2f}",
                          annotation_position="top left", annotation_font_size=8,
                          row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        marker_color="rgba(100,181,246,0.4)", name="Volume", showlegend=False), row=2, col=1)

    avg_vol = df["Volume"].rolling(20, min_periods=5).mean()
    spike_mask = df["Volume"] > 1.8 * avg_vol
    if spike_mask.any():
        fig.add_trace(go.Scatter(
            x=df.index[spike_mask], y=df.loc[spike_mask, "Volume"] * 1.05,
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="yellow", line=dict(width=1, color="black")),
            name="Vol Spike", showlegend=False,
            hovertemplate="Spike: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # Projection separator and zone
    last_ts   = df.index[-1]
    last_proj = future_dates[-1]
    fig.add_vrect(x0=last_ts, x1=last_proj,
                  fillcolor="rgba(180,220,255,0.04)", layer="below", line_width=0)
    # Scatter line avoids Plotly's datetime annotation bug with add_vline
    fig.add_trace(go.Scatter(
        x=[last_ts, last_ts],
        y=[df[["Low", "High"]].min().min(), df[["Low", "High"]].max().max()],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
        showlegend=False, hoverinfo="skip", name="_proj_vline",
    ), row=1, col=1)
    fig.add_annotation(
        x=last_ts, y=1, yref="paper", text="  Projection →",
        showarrow=False, font=dict(color="rgba(255,255,255,0.4)", size=10),
        xanchor="left", yanchor="top",
    )

    # Layout
    xaxis_cfg = dict(fixedrange=True, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    if not is_weekly:
        xaxis_cfg["rangebreaks"] = [dict(bounds=["sat", "mon"])]

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
            text=f"{ticker} — Chart Pattern Analyzer{title_suffix}  |  Current: ${current_price:,.2f}",
            x=0.02, xanchor="left", font=dict(size=16),
        ),
    )
    fig.update_xaxes(**xaxis_cfg)
    fig.update_yaxes(fixedrange=True, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    return fig


# === Projection Table Builder ===
def build_projection_table(current_price, horizon_bars, linreg, tl, ma, fib, bb, sr,
                            show, targets, vs_col, is_weekly=False):
    rows = []
    slope_threshold = 0.01 if is_weekly else 0.05

    def proj(last_val, slope, offset):
        return None if offset > horizon_bars else last_val + slope * offset

    def fmt(v):
        return "N/A" if v is None else f"${v:,.2f}"

    def direction(slope):
        if slope > slope_threshold:  return "Rising ↑"
        if slope < -slope_threshold: return "Falling ↓"
        return "Flat →"

    def vs_curr(last_val, slope):
        offset = targets.get(vs_col, 21)
        v = proj(last_val, slope, offset)
        if v is None:
            return "N/A"
        pct = (v - current_price) / current_price * 100
        return f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"

    def add_row(pattern, last_val, slope):
        rows.append({
            "Pattern": pattern,
            "Current": fmt(last_val),
            **{t: fmt(proj(last_val, slope, off)) for t, off in targets.items()},
            "Direction": direction(slope),
            f"vs Current ({vs_col})": vs_curr(last_val, slope),
        })

    if linreg:
        lc, sl, std = linreg["last_center"], linreg["slope"], linreg["std"]
        if show.get("lr_center"): add_row("LinReg Center", lc, sl)
        if show.get("lr_2std"):
            add_row("LinReg +2σ", lc + 2 * std, sl)
            add_row("LinReg -2σ", lc - 2 * std, sl)

    if tl:
        for key, flag_key, lbl in [("resist", "tl_resist", "Trend Resistance"),
                                    ("support", "tl_support", "Trend Support")]:
            td = tl.get(key)
            if td and show.get(flag_key):
                add_row(lbl, td["last_val"], td["slope"])

    if ma:
        for p, md in sorted(ma.items()):
            lbl = f"SMA {p}w" if is_weekly else f"SMA {p}"
            add_row(lbl, md["last_val"], md["slope"])

    if bb:
        if show.get("bb_bands"):
            add_row("BB Upper", bb["last_upper"], bb["slope"])
            add_row("BB Lower", bb["last_lower"], bb["slope"])
        if show.get("bb_center"):
            add_row("BB SMA", bb["last_sma"], bb["slope"])

    if fib:
        for lbl, level in fib.items():
            add_row(f"Fib {lbl}", level, 0.0)

    if sr:
        if show.get("sr_resist"):
            for level in sr.get("resistance", []):
                add_row("Resistance", level, 0.0)
        if show.get("sr_support"):
            for level in sr.get("support", []):
                add_row("Support", level, 0.0)

    if not rows:
        return pd.DataFrame()

    col_order = ["Pattern", "Current"] + list(targets.keys()) + ["Direction", f"vs Current ({vs_col})"]
    return pd.DataFrame(rows, columns=col_order)


def _render_projection_table(proj_df, vs_col):
    def _style_direction(v):
        if "Rising"  in str(v): return "color: #00c853"
        if "Falling" in str(v): return "color: #d50000"
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
        .map(_style_vs, subset=[f"vs Current ({vs_col})"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _pct_label(pct):
    color = "green" if pct >= 0 else "red"
    sign = "+" if pct >= 0 else ""
    return f"<div style='text-align:center;color:{color};font-size:0.85em;margin-top:-6px'>{sign}{pct:.2f}%</div>"


# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_st, tab_lt = st.tabs(["Short Term (Daily)", "Long Term (Weekly / LEAPS)"])


# ─── Short Term Tab ──────────────────────────────────────────────────────────
with tab_st:
    # --- History Period ---
    ST_PERIOD_OPTS = {
        "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
        "1y": "1Y", "2y": "2Y", "5y": "5Y",
    }
    st.write("**Select history period:**")
    _st_per_cols = st.columns(len(ST_PERIOD_OPTS))
    for i, (k, lbl) in enumerate(ST_PERIOD_OPTS.items()):
        with _st_per_cols[i]:
            st.button(lbl, key=f"cpa_stper_{k}", use_container_width=True,
                      on_click=lambda _k=k: st.session_state.update({"cpa_period": _k}))
            pct = period_changes.get(k)
            if pct is not None:
                st.markdown(_pct_label(pct), unsafe_allow_html=True)
    period = st.session_state.cpa_period

    # --- Projection Horizon ---
    ST_HORIZON_OPTS = {"1 Month": 21, "3 Months": 65, "6 Months": 130}
    st.write("**Select projection horizon:**")
    _st_hor_cols = st.columns(len(ST_HORIZON_OPTS))
    for i, (k, bars) in enumerate(ST_HORIZON_OPTS.items()):
        with _st_hor_cols[i]:
            st.button(k, key=f"cpa_sthor_{k.replace(' ','_')}", use_container_width=True,
                      on_click=lambda _k=k: st.session_state.update({"cpa_horizon": _k}))
    horizon_label = st.session_state.cpa_horizon
    horizon_bars  = ST_HORIZON_OPTS[horizon_label]

    st.info(f"Showing: **{ST_PERIOD_OPTS.get(period, period)}** history  |  **{horizon_label}** projection", icon="📊")

    # --- Data ---
    df = fetch_data(ticker, period)
    if df is None or df.empty:
        st.error(f"No daily data found for **{ticker}**.")
    else:
        df["_bar"] = np.arange(len(df))
        current_price = float(df["Close"].iloc[-1])
        n_bars = len(df)
        future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon_bars)
        future_bar_idx = np.arange(n_bars, n_bars + horizon_bars)
        swing_highs, swing_lows = find_swings(df, window=5)

        # Overlay checkboxes
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
            st.write("")

        st.markdown("---")

        show = {
            "lr_center": show_lr_center, "lr_1std": show_lr_1std, "lr_2std": show_lr_2std,
            "tl_resist": show_tl_resist, "tl_support": show_tl_support,
            "bb_bands": show_bb_bands, "bb_center": show_bb_center,
            "sr_resist": show_sr_resist, "sr_support": show_sr_support,
        }

        ma_periods  = [p for p, on in [(20, ma_20), (50, ma_50), (100, ma_100), (200, ma_200)] if on]
        linreg_data = calc_linreg(df, future_bar_idx) if (show_lr_center or show_lr_1std or show_lr_2std) else None
        tl_data     = calc_trendlines(df, swing_highs, swing_lows, future_dates, future_bar_idx) if (show_tl_resist or show_tl_support) else None
        ma_data     = calc_moving_averages(df, future_dates, future_bar_idx, ma_periods) if ma_periods else None
        fib_data    = calc_fibonacci(swing_highs, swing_lows) if show_fib else None
        bb_data     = calc_bollinger(df, future_dates, future_bar_idx, period=20) if (show_bb_bands or show_bb_center) else None
        sr_data     = calc_sr(swing_highs, swing_lows) if (show_sr_resist or show_sr_support) else None

        fig = build_chart(
            df, current_price, future_dates, future_bar_idx,
            linreg_data, tl_data, ma_data, fib_data, bb_data, sr_data, show,
            ticker, title_suffix=" (Daily)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        ST_TARGETS = {"+1 Week": 5, "+2 Weeks": 10, "+1 Month": 21, "+3 Months": 65}
        proj_df = build_projection_table(
            current_price, horizon_bars, linreg_data, tl_data, ma_data, fib_data, bb_data, sr_data,
            show, ST_TARGETS, vs_col="+1 Month",
        )
        if not proj_df.empty:
            st.subheader(f"Price Projection Table — {ticker} (Daily)")
            st.caption(
                f"Projections from {df.index[-1].strftime('%b %d, %Y')} "
                f"over {horizon_label} horizon  |  "
                f"Rising/Falling = slope >±$0.05/day  |  vs Current uses +1 Month"
            )
            _render_projection_table(proj_df, "+1 Month")


# ─── Long Term Tab ───────────────────────────────────────────────────────────
with tab_lt:
    # --- History Period ---
    LT_PERIOD_OPTS = {
        "2y": "2Y", "3y": "3Y", "5y": "5Y", "10y": "10Y", "max": "Max",
    }
    st.write("**Select history period:**")
    _lt_per_cols = st.columns(len(LT_PERIOD_OPTS))
    for i, (k, lbl) in enumerate(LT_PERIOD_OPTS.items()):
        with _lt_per_cols[i]:
            st.button(lbl, key=f"cpa_ltper_{k}", use_container_width=True,
                      on_click=lambda _k=k: st.session_state.update({"cpa_lt_period": _k}))
            pct = lt_period_changes.get(k)
            if pct is not None:
                st.markdown(_pct_label(pct), unsafe_allow_html=True)
    lt_period = st.session_state.cpa_lt_period

    # --- Projection Horizon ---
    LT_HORIZON_OPTS = {"6 Months": 26, "1 Year": 52, "2 Years": 104}
    st.write("**Select projection horizon:**")
    _lt_hor_cols = st.columns(len(LT_HORIZON_OPTS))
    for i, (k, bars) in enumerate(LT_HORIZON_OPTS.items()):
        with _lt_hor_cols[i]:
            st.button(k, key=f"cpa_lthor_{k.replace(' ','_')}", use_container_width=True,
                      on_click=lambda _k=k: st.session_state.update({"cpa_lt_horizon": _k}))
    lt_horizon_label = st.session_state.cpa_lt_horizon
    lt_horizon_bars  = LT_HORIZON_OPTS[lt_horizon_label]

    st.info(f"Showing: **{LT_PERIOD_OPTS.get(lt_period, lt_period)}** history  |  **{lt_horizon_label}** projection", icon="📊")

    # --- Data ---
    df_lt = fetch_data_lt(ticker, lt_period)
    if df_lt is None or df_lt.empty:
        st.error(f"No weekly data found for **{ticker}**.")
    else:
        df_lt["_bar"] = np.arange(len(df_lt))
        current_price_lt = float(df_lt["Close"].iloc[-1])
        n_bars_lt = len(df_lt)
        future_dates_lt = pd.date_range(
            start=df_lt.index[-1] + pd.Timedelta(weeks=1),
            periods=lt_horizon_bars, freq="W-MON",
        )
        future_bar_idx_lt = np.arange(n_bars_lt, n_bars_lt + lt_horizon_bars)
        swing_highs_lt, swing_lows_lt = find_swings(df_lt, window=4)

        # Overlay checkboxes
        st.markdown("**Overlays**")
        _oc_lt = st.columns(7)
        with _oc_lt[0]:
            st.caption("Linear Regression")
            lt_lr_center = st.checkbox("LR Center",    value=True,  key="cpa_lt_lr_center")
            lt_lr_1std   = st.checkbox("LR ±1σ",       value=False, key="cpa_lt_lr_1std")
            lt_lr_2std   = st.checkbox("LR ±2σ",       value=False, key="cpa_lt_lr_2std")
        with _oc_lt[1]:
            st.caption("Trend Lines")
            lt_tl_resist  = st.checkbox("Resistance",  value=False, key="cpa_lt_tl_resist")
            lt_tl_support = st.checkbox("Support",     value=False, key="cpa_lt_tl_support")
        with _oc_lt[2]:
            st.caption("Weekly MAs")
            lt_ma13  = st.checkbox("SMA 13w  (~1Q)",  value=False, key="cpa_lt_ma13")
            lt_ma26  = st.checkbox("SMA 26w  (~6M)",  value=False, key="cpa_lt_ma26")
            lt_ma52  = st.checkbox("SMA 52w  (~1Y)",  value=False, key="cpa_lt_ma52")
            lt_ma104 = st.checkbox("SMA 104w (~2Y)",  value=False, key="cpa_lt_ma104")
        with _oc_lt[3]:
            st.caption("Fibonacci")
            lt_fib = st.checkbox("Fib Retracement", value=False, key="cpa_lt_fib")
        with _oc_lt[4]:
            st.caption("Bollinger (26w)")
            lt_bb_bands  = st.checkbox("BB Upper/Lower", value=False, key="cpa_lt_bb_bands")
            lt_bb_center = st.checkbox("BB Center",      value=False, key="cpa_lt_bb_center")
        with _oc_lt[5]:
            st.caption("S/R Horizontal")
            lt_sr_resist  = st.checkbox("S/R Resistance", value=False, key="cpa_lt_sr_resist")
            lt_sr_support = st.checkbox("S/R Support",    value=False, key="cpa_lt_sr_support")
        with _oc_lt[6]:
            st.write("")

        st.markdown("---")

        show_lt = {
            "lr_center": lt_lr_center, "lr_1std": lt_lr_1std, "lr_2std": lt_lr_2std,
            "tl_resist": lt_tl_resist, "tl_support": lt_tl_support,
            "bb_bands": lt_bb_bands, "bb_center": lt_bb_center,
            "sr_resist": lt_sr_resist, "sr_support": lt_sr_support,
        }

        lt_ma_periods = [p for p, on in [(13, lt_ma13), (26, lt_ma26), (52, lt_ma52), (104, lt_ma104)] if on]
        lt_linreg = calc_linreg(df_lt, future_bar_idx_lt) if (lt_lr_center or lt_lr_1std or lt_lr_2std) else None
        lt_tl     = calc_trendlines(df_lt, swing_highs_lt, swing_lows_lt, future_dates_lt, future_bar_idx_lt) if (lt_tl_resist or lt_tl_support) else None
        lt_ma     = calc_moving_averages(df_lt, future_dates_lt, future_bar_idx_lt, lt_ma_periods) if lt_ma_periods else None
        lt_fib_d  = calc_fibonacci(swing_highs_lt, swing_lows_lt) if lt_fib else None
        lt_bb     = calc_bollinger(df_lt, future_dates_lt, future_bar_idx_lt, period=26) if (lt_bb_bands or lt_bb_center) else None
        lt_sr     = calc_sr(swing_highs_lt, swing_lows_lt) if (lt_sr_resist or lt_sr_support) else None

        fig_lt = build_chart(
            df_lt, current_price_lt, future_dates_lt, future_bar_idx_lt,
            lt_linreg, lt_tl, lt_ma, lt_fib_d, lt_bb, lt_sr, show_lt,
            ticker, title_suffix=" (Weekly)",
            is_weekly=True,
        )
        st.plotly_chart(fig_lt, use_container_width=True, config={"displayModeBar": False})

        LT_TARGETS = {"+3 Months": 13, "+6 Months": 26, "+1 Year": 52, "+2 Years": 104}
        proj_df_lt = build_projection_table(
            current_price_lt, lt_horizon_bars, lt_linreg, lt_tl, lt_ma, lt_fib_d, lt_bb, lt_sr,
            show_lt, LT_TARGETS, vs_col="+1 Year", is_weekly=True,
        )
        if not proj_df_lt.empty:
            st.subheader(f"Price Projection Table — {ticker} (Weekly / LEAPS)")
            st.caption(
                f"Weekly projections from {df_lt.index[-1].strftime('%b %d, %Y')} "
                f"over {lt_horizon_label} horizon  |  "
                f"Rising/Falling = slope >±$0.01/week  |  vs Current uses +1 Year"
            )
            _render_projection_table(proj_df_lt, "+1 Year")
