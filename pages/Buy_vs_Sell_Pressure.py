# app.py — ULTIMATE BUYING vs SELLING PRESSURE DASHBOARD (Dec 2025)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import json, os
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _ET = _ZoneInfo("America/New_York")
except Exception:
    _ET = None

# Optional: Google Gemini (only needed for AI analysis)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# === Page Config ===
st.set_page_config(page_title="Pressure Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("Buying vs Selling Pressure")
st.write("")

# === Favorites ===
def _load_favorites():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "favorites.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return ["SPY", "QQQ", "VOO", "XLK", "NVDA"]

_favs = _load_favorites()

if "bvs_ticker" not in st.session_state:
    st.session_state.bvs_ticker = "SPY"
if "bvs_period" not in st.session_state:
    st.session_state.bvs_period = "1d"
if "_ext_pref" not in st.session_state:
    st.session_state._ext_pref = False

if "fav" in st.query_params and st.query_params["fav"] in _favs:
    st.session_state.bvs_ticker = st.query_params["fav"]
    if "period" in st.query_params:
        st.session_state.bvs_period = st.query_params["period"]
        del st.query_params["period"]
    if "ext" in st.query_params:
        st.session_state._ext_pref = (st.query_params["ext"] == "1")
        del st.query_params["ext"]
    del st.query_params["fav"]

_fav_placeholder = st.empty()

ticker = st.text_input("Ticker", key="bvs_ticker", max_chars=12).upper().strip()
if not ticker:
    st.stop()

# --- Gemini API Key Setup (in sidebar) ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI Analysis Settings")

if not GEMINI_AVAILABLE:
    st.sidebar.warning("⚠️ Google Gemini not installed")
    st.sidebar.code("pip install google-generativeai", language="bash")
    gemini_api_key = None
else:
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key", 
        type="password",
        help="Get free API key at: https://aistudio.google.com/app/apikey",
        value=""
    )
    
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            st.sidebar.success("✅ Gemini API configured")
        except Exception as e:
            st.sidebar.error(f"❌ API configuration failed: {str(e)}")
    else:
        st.sidebar.info("💡 Add API key to enable AI analysis")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("**Free Tier**: 1,500 requests/day")
    st.sidebar.caption("[Get API Key →](https://aistudio.google.com/app/apikey)")

if st.sidebar.button("Reset Zoom & Settings"):
    st.session_state.clear()
    st.rerun()

# === Extended Hours Helpers ===
@st.cache_data(ttl=60, show_spinner=False)
def get_extended_data_1d(t):
    try:
        return yf.Ticker(t).history(period="1d", interval="1m", prepost=True)
    except Exception:
        return pd.DataFrame()

def _split_extended(ext_df):
    """Split into (pre_market, regular, after_hours) DataFrames by ET market hours."""
    empty = pd.DataFrame()
    if ext_df is None or ext_df.empty or ext_df.index.tz is None or _ET is None:
        return empty, ext_df if ext_df is not None else empty, empty
    try:
        et_idx = ext_df.index.tz_convert(_ET)
        today = datetime.now(_ET).date()
        today_mask = [ts.date() == today for ts in et_idx]
        if not any(today_mask):
            most_recent = et_idx[-1].date()
            today_mask = [ts.date() == most_recent for ts in et_idx]
        if not any(today_mask):
            return empty, ext_df, empty
        sub = ext_df[today_mask].copy()
        sub_et = et_idx[today_mask]
        mins = sub_et.hour * 60 + sub_et.minute
        OPEN, CLOSE = 9 * 60 + 30, 16 * 60
        return sub[mins.values < OPEN], sub[(mins.values >= OPEN) & (mins.values < CLOSE)], sub[mins.values >= CLOSE]
    except Exception:
        return empty, ext_df, empty

def _get_ext_hours_metric(t, prev_close, regular_close):
    """Return (label, price, pct_change) for current extended session, or (None, None, None)."""
    if _ET is None:
        return None, None, None
    try:
        ext = get_extended_data_1d(t)
        if ext is None or ext.empty or ext.index.tz is None:
            return None, None, None
        et_idx = ext.index.tz_convert(_ET)
        today = datetime.now(_ET).date()
        if et_idx[-1].date() != today:
            return None, None, None
        pre, regular, after = _split_extended(ext)
        now_et = datetime.now(_ET)
        now_min = now_et.hour * 60 + now_et.minute
        OPEN, CLOSE = 9 * 60 + 30, 16 * 60
        if now_min < OPEN and not pre.empty:
            price = float(pre.Close.iloc[-1])
            ref = prev_close
            chg = ((price - ref) / ref * 100) if ref else None
            return "Pre-market", price, chg
        if now_min >= CLOSE and not after.empty:
            price = float(after.Close.iloc[-1])
            ref = float(regular.Close.iloc[-1]) if not regular.empty else (regular_close or prev_close)
            chg = ((price - ref) / ref * 100) if ref else None
            return "After-hours", price, chg
    except Exception:
        pass
    return None, None, None

# === Live Price ===
stock = yf.Ticker(ticker)
try:
    fast = stock.fast_info
    current_price = fast.last_price or fast.previous_close or None
    prev_close = fast.previous_close or None
except Exception:
    current_price = None
    prev_close = None

if isinstance(current_price, (int, float)):
    price_display = f"${current_price:,.2f}"
    if isinstance(prev_close, (int, float)) and prev_close > 0:
        day_change = current_price - prev_close
        day_pct = (day_change / prev_close) * 100
        arrow = "▲" if day_change >= 0 else "▼"
        sign = "+" if day_change >= 0 else ""
        change_color = "green" if day_change >= 0 else "red"
        change_str = f"<span style='color:{change_color}'>{arrow} {sign}${day_change:,.2f} ({sign}{day_pct:.2f}%)</span>"
    else:
        change_str = ""
    st.markdown(f"<h3>{ticker} &mdash; {price_display} &nbsp; {change_str}</h3>", unsafe_allow_html=True)
else:
    st.markdown(f"<h3>{ticker} &mdash; Current Price: N/A</h3>", unsafe_allow_html=True)

_ext_label, _ext_price, _ext_chg = _get_ext_hours_metric(ticker, prev_close, current_price)
if _ext_label and isinstance(_ext_price, float):
    _ext_color = "green" if (_ext_chg or 0) >= 0 else "red"
    _ext_sign = "+" if (_ext_chg or 0) >= 0 else ""
    _ext_chg_str = f" ({_ext_sign}{_ext_chg:.2f}%)" if _ext_chg is not None else ""
    st.markdown(
        f"<div style='color:{_ext_color}; font-size:0.95em'><b>{_ext_label}:</b> ${_ext_price:,.2f}{_ext_chg_str}</div>",
        unsafe_allow_html=True
    )

# === Period Selection ===
period_options = {
    "1d": "1 Day", "5d": "1 Week", "1mo": "1 Month", "3mo": "3 Months",
    "6mo": "6 Months", "1y": "1 Year", "2y": "2 Years", "5y": "5 Years"
}

@st.cache_data(ttl=900, show_spinner=False)
def get_period_changes(t, cur_price, prev_close):
    changes = {}
    if isinstance(cur_price, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
        changes["1d"] = ((cur_price - prev_close) / prev_close) * 100
    try:
        hist = yf.Ticker(t).history(period="5y", interval="1d")
        if not hist.empty:
            last = hist.Close.iloc[-1]
            for p, bars in [("5d", 5), ("1mo", 21), ("3mo", 63), ("6mo", 126), ("1y", 252), ("2y", 504), ("5y", len(hist))]:
                idx = min(bars, len(hist))
                start = hist.Close.iloc[-idx]
                if start > 0:
                    changes[p] = ((last - start) / start) * 100
    except Exception:
        pass
    return changes

period_changes = get_period_changes(ticker, current_price, prev_close)

st.write("**Select time range:**")
cols = st.columns(len(period_options))
selected_period = None
for i, (key, label) in enumerate(period_options.items()):
    with cols[i]:
        if st.button(label, key=f"btn_{key}", use_container_width=True):
            selected_period = key
        pct = period_changes.get(key)
        if pct is not None:
            color = "green" if pct >= 0 else "red"
            sign = "+" if pct >= 0 else ""
            st.markdown(
                f"<div style='text-align:center; color:{color}; font-size:0.85em; margin-top:-6px'>{sign}{pct:.2f}%</div>",
                unsafe_allow_html=True
            )

if selected_period is not None:
    st.session_state.bvs_period = selected_period

period = st.session_state.bvs_period
st.info(f"Showing: **{period_options[period]}**", icon="📊")

_ext_url = "&ext=1" if st.session_state._ext_pref else ""
_fav_links = ", ".join([f'<a href="?fav={f}&period={period}{_ext_url}" target="_self" style="text-decoration:none">{f}</a>' for f in _favs])
_fav_placeholder.markdown(f"**Favorites:** {_fav_links}", unsafe_allow_html=True)

# === Data Fetching ===
@st.cache_data(ttl=900, show_spinner="Fetching data...")
def get_data(t, p):
    stock = yf.Ticker(t)
    interval = "1m" if p in ["1d","5d"] else "1d"
    hist = stock.history(period=p, interval=interval)
    if hist.empty:
        return None, None, None, None, None, None

    # Get last data timestamp
    last_timestamp = hist.index[-1] if not hist.empty else None

    # Long-dated unusual options activity (>=6 months out)
    today = datetime.now().date()
    six_months = today + timedelta(days=180)
    long_dates = [d for d in stock.options if datetime.strptime(d, "%Y-%m-%d").date() >= six_months]

    spikes = []
    total_call = total_put = 0
    for date in long_dates[:20]:
        try:
            chain = stock.option_chain(date)
            calls = chain.calls[['strike','volume']].assign(type='CALL', expiry=date)
            puts  = chain.puts[['strike','volume']].assign(type='PUT', expiry=date)
            frame = pd.concat([calls, puts])
            frame['total_vol'] = frame['volume']
            spikes.append(frame[['strike','volume','type','expiry','total_vol']])
            total_call += calls['volume'].sum()
            total_put  += puts['volume'].sum()
        except:
            continue

    if spikes:
        opts = pd.concat(spikes)
        avg = opts['total_vol'].replace(0, pd.NA).mean() or 1
        unusual = opts[opts['total_vol'] > 5*avg].copy()
        unusual['x'] = (unusual['total_vol']/avg).round(1)
        unusual = unusual.sort_values('total_vol', ascending=False).head(12)
    else:
        unusual = pd.DataFrame()
        avg = 0

    pcr = total_put / (total_call + 1e-6) if total_call else 10
    opt_vol = total_call + total_put

    return hist, pcr, opt_vol, unusual, avg, last_timestamp

df, pcr, opt_vol, spikes, avg_opt, last_data_time = get_data(ticker, period)
if df is None or df.empty:
    st.error("No data available for this ticker/period.")
    st.stop()

# === Indicators ===
df["OBV"] = (df.Volume * ((df.Close > df.Open).astype(int)*2 - 1)).cumsum()

df["TP"] = (df.High + df.Low + df.Close) / 3
df["TPV"] = df["TP"] * df.Volume
df["CumVol"] = df.Volume.cumsum()
df["CumTPV"] = df["TPV"].cumsum()
df["VWAP"] = df["CumTPV"] / df["CumVol"]

lookback = min(120, len(df))
recent = df.tail(lookback)
bins = pd.cut(recent.Close, bins=50)
poc_price = recent.groupby(bins).Volume.sum().idxmax().mid

df["PCT_Change"] = df.Close.pct_change()
df["PVT"] = (df["PCT_Change"] * df.Volume).cumsum()

exp1 = df.Close.ewm(span=12, adjust=False).mean()
exp2 = df.Close.ewm(span=26, adjust=False).mean()
df["MACD"] = exp1 - exp2
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"] = df["MACD"] - df["Signal"]

avg_vol = df.Volume.tail(20).mean()

# Get latest non-zero volume (handle incomplete minute bars)
latest_vol = df.Volume.iloc[-1]
if latest_vol == 0 and len(df) > 1:
    # Look back up to 5 bars for last non-zero volume
    for i in range(1, min(6, len(df))):
        if df.Volume.iloc[-i] > 0:
            latest_vol = df.Volume.iloc[-i]
            break

# === IMPROVED Pressure Score (Gradient-based) ===
def calculate_pressure_score(df, pcr, poc_price, period):
    """
    Calculate a nuanced pressure score (0-100) using gradients instead of binary jumps
    """
    score = 0
    components = {}
    
    # 1. VWAP Distance (0-25 points) - Gradient based on % distance
    vwap_diff_pct = ((df.Close.iloc[-1] - df.VWAP.iloc[-1]) / df.VWAP.iloc[-1]) * 100
    vwap_score = np.clip(vwap_diff_pct * 2.5, -25, 25)  # ±1% = ±2.5 pts, caps at ±10%
    score += vwap_score
    components['VWAP Distance'] = round(vwap_score, 1)
    
    # 2. OBV Momentum (0-25 points) - Gradient based on % change
    if len(df) > 20:
        obv_change_pct = ((df.OBV.iloc[-1] - df.OBV.iloc[-20]) / abs(df.OBV.iloc[-20] + 1)) * 100
        obv_score = np.clip(obv_change_pct * 0.5, -25, 25)  # ±50% = ±25 pts
        score += obv_score
        components['OBV Momentum'] = round(obv_score, 1)
    else:
        components['OBV Momentum'] = 0
    
    # 3. POC Position (0-20 points) - Gradient based on % distance
    poc_diff_pct = ((df.Close.iloc[-1] - poc_price) / poc_price) * 100
    poc_score = np.clip(poc_diff_pct * 2, -20, 20)  # ±1% = ±2 pts, caps at ±10%
    score += poc_score
    components['POC Position'] = round(poc_score, 1)
    
    # 4. Volume Surge (0-15 points) - Gradient based on volume vs average
    avg_vol = df.Volume.tail(20).mean()
    vol_ratio = df.Volume.iloc[-1] / (avg_vol + 1)
    vol_score = np.clip((vol_ratio - 1) * 7.5, -5, 15)  # Above avg = positive, caps at 3x
    score += vol_score
    components['Volume Surge'] = round(vol_score, 1)
    
    # 5. Options P/C Ratio (0-15 points) - Gradient for non-intraday only
    if period not in ["1d", "5d"]:
        # Bullish: PCR < 0.7, Neutral: 0.7-1.3, Bearish: > 1.3
        if pcr < 0.7:
            pcr_score = np.clip((0.7 - pcr) * 30, 0, 15)  # More bullish = higher
        elif pcr > 1.3:
            pcr_score = np.clip((1.3 - pcr) * 10, -15, 0)  # More bearish = lower
        else:
            pcr_score = 0  # Neutral zone
        score += pcr_score
        components['P/C Ratio'] = round(pcr_score, 1)
    else:
        components['P/C Ratio'] = 0
    
    # Normalize to 0-100 scale (score can range roughly -90 to +90)
    normalized_score = ((score + 90) / 180) * 100
    normalized_score = np.clip(normalized_score, 0, 100)
    
    return round(normalized_score, 1), components

score, score_components = calculate_pressure_score(df, pcr, poc_price, period)

# === Shared Zoom State ===
if "zoom_range" not in st.session_state:
    st.session_state.zoom_range = None
# Restore widget key from persistent pref (handles Streamlit clearing widget keys for non-rendered widgets)
st.session_state.show_extended_hours = st.session_state._ext_pref

# === Plotly Config (Disable zoom/select) ===
config = {
    "displayModeBar": False,
    "scrollZoom": False,
    "displaylogo": False
}

# === Rangebreaks — hide non-trading hours for intraday periods ===
intraday = period in ["1d", "5d"]
rangebreaks = [
    dict(bounds=["sat", "mon"]),               # hide weekends
    dict(bounds=[16, 9.5], pattern="hour"),    # hide overnight (4 PM – 9:30 AM)
] if intraday else []

_show_ext = period == "1d" and st.session_state._ext_pref
rangebreaks_price = [
    dict(bounds=["sat", "mon"]),
    dict(bounds=[20, 4], pattern="hour"),      # show 4 AM – 8 PM window
] if _show_ext else rangebreaks

# === Universal Chart Function (ensures perfect alignment) ===
def plot(fig, title=None, height=380):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=16), x=0.02, xanchor="left"),
        legend=dict(orientation="h", y=1.05, x=0),
        uirevision="constant",
        margin=dict(t=90, l=60, r=60, b=50),
        hovermode="x",
        dragmode=False
    )
    fig.update_xaxes(fixedrange=True, rangebreaks=rangebreaks)
    fig.update_yaxes(fixedrange=True)
    if st.session_state.zoom_range:
        fig.update_xaxes(range=st.session_state.zoom_range)
    st.plotly_chart(fig, use_container_width=True, config=config)

# === Layout ===
col_left, col_right = st.columns([3.4, 1])

with col_left:
    # 1. Price Chart with Volume Subplot
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("", "")
    )
    
    # Chart controls
    col_vwap, col_poc, col_ext = st.columns(3)
    with col_vwap:
        show_vwap = st.checkbox("VWAP", True, key="vwap", help="Volume-Weighted Average Price - average price weighted by volume, used to identify support/resistance levels")
    with col_poc:
        show_poc = st.checkbox("POC", True, key="poc", help="Point of Control - price level with highest traded volume, acts as a magnet for price action")
    with col_ext:
        if period == "1d":
            def _on_ext_change():
                st.session_state._ext_pref = st.session_state.show_extended_hours
            st.checkbox("Extended Hours", key="show_extended_hours",
                        on_change=_on_ext_change,
                        help="Show pre-market (4–9:30 AM ET) and after-hours (4–8 PM ET) candles.")

    # Price candlestick(s)
    if _show_ext:
        _ext_raw = get_extended_data_1d(ticker)
        _pre_df, _reg_df, _aft_df = _split_extended(_ext_raw)
        if not _pre_df.empty:
            fig1.add_trace(go.Candlestick(
                x=_pre_df.index, open=_pre_df.Open, high=_pre_df.High, low=_pre_df.Low, close=_pre_df.Close,
                name="Pre-market",
                increasing=dict(line=dict(color="rgba(0,180,0,0.5)"), fillcolor="rgba(0,180,0,0.18)"),
                decreasing=dict(line=dict(color="rgba(200,0,0,0.5)"), fillcolor="rgba(200,0,0,0.18)"),
            ), row=1, col=1)
        _chart_df = _reg_df if not _reg_df.empty else df
        fig1.add_trace(go.Candlestick(
            x=_chart_df.index, open=_chart_df.Open, high=_chart_df.High,
            low=_chart_df.Low, close=_chart_df.Close, name="Regular Hours",
        ), row=1, col=1)
        if not _aft_df.empty:
            fig1.add_trace(go.Candlestick(
                x=_aft_df.index, open=_aft_df.Open, high=_aft_df.High, low=_aft_df.Low, close=_aft_df.Close,
                name="After-hours",
                increasing=dict(line=dict(color="rgba(80,120,220,0.5)"), fillcolor="rgba(80,120,220,0.18)"),
                decreasing=dict(line=dict(color="rgba(150,80,200,0.5)"), fillcolor="rgba(150,80,200,0.18)"),
            ), row=1, col=1)
        if not _ext_raw.empty and _ET is not None:
            _et_for_vline = _ext_raw.index.tz_convert(_ET)
            _open_mask = (_et_for_vline.hour == 9) & (_et_for_vline.minute == 30)
            _close_mask = (_et_for_vline.hour == 16) & (_et_for_vline.minute == 0)
            if _open_mask.any():
                fig1.add_vline(x=_ext_raw.index[_open_mask][0].value // 1_000_000, line_dash="dot",
                               line_color="rgba(128,128,128,0.6)",
                               annotation_text="Open", annotation_position="top left",
                               annotation_font_size=10, annotation_font_color="gray")
            if _close_mask.any():
                fig1.add_vline(x=_ext_raw.index[_close_mask][0].value // 1_000_000, line_dash="dot",
                               line_color="rgba(128,128,128,0.6)",
                               annotation_text="Close", annotation_position="top left",
                               annotation_font_size=10, annotation_font_color="gray")
    else:
        _ext_raw = pd.DataFrame()
        fig1.add_trace(
            go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"),
            row=1, col=1
        )
    
    if show_vwap:
        fig1.add_trace(
            go.Scatter(x=df.index, y=df.VWAP, name="VWAP (Volume-Weighted Avg Price)", line=dict(color="orange", width=3)),
            row=1, col=1
        )
    
    if show_poc:
        fig1.add_trace(
            go.Scatter(x=df.index, y=[poc_price]*len(df), name="POC (Point of Control)", line=dict(color="purple", dash="dot", width=3)),
            row=1, col=1
        )
    
    # Volume bars
    if _show_ext and not _ext_raw.empty and _ET is not None:
        _et_vol_idx = _ext_raw.index.tz_convert(_ET)
        _vol_colors = [
            "rgba(100,150,200,0.55)" if (ts.hour * 60 + ts.minute) < 9 * 60 + 30
            else "rgba(150,100,200,0.55)" if (ts.hour * 60 + ts.minute) >= 16 * 60
            else "rgba(255,100,100,0.7)"
            for ts in _et_vol_idx
        ]
        fig1.add_trace(
            go.Bar(x=_ext_raw.index, y=_ext_raw.Volume, name="Volume", marker_color=_vol_colors, showlegend=False),
            row=2, col=1
        )
    else:
        fig1.add_trace(
            go.Bar(x=df.index, y=df.Volume, name="Volume", marker_color="rgba(255,100,100,0.7)", showlegend=False),
            row=2, col=1
        )
    
    fig1.update_layout(
        height=600,
        title=dict(text="Price Chart", font=dict(size=16), x=0.02, xanchor="left"),
        legend=dict(orientation="h", y=1.02, x=0),
        uirevision="constant",
        margin=dict(t=90, l=60, r=60, b=50),
        hovermode="x",
        dragmode=False,
        xaxis_rangeslider_visible=False
    )
    
    fig1.update_xaxes(fixedrange=True, rangebreaks=rangebreaks_price)
    fig1.update_yaxes(fixedrange=True)

    if st.session_state.zoom_range:
        fig1.update_xaxes(range=st.session_state.zoom_range)

    st.plotly_chart(fig1, use_container_width=True, config=config)

    # 2. OBV & PVT — Toggleable
    st.subheader("OBV & PVT")

    col_cb1, col_cb2 = st.columns(2)
    show_obv = col_cb1.checkbox("OBV (On-Balance Volume)", value=True, key="obv_toggle")
    show_pvt = col_cb2.checkbox("PVT (Price-Volume Trend)", value=False, key="pvt_toggle")

    fig3 = go.Figure()

    if show_obv:
        fig3.add_scatter(x=df.index, y=df.OBV, name="OBV", line=dict(color="#00ff88", width=3), yaxis="y")
    if show_pvt:
        fig3.add_scatter(x=df.index, y=df.PVT, name="PVT", line=dict(color="#ff00ff", width=3), yaxis="y2")

    if show_obv or show_pvt:
        fig3.update_layout(
            yaxis=dict(title=dict(text="OBV", font=dict(color="#00ff88")), tickfont=dict(color="#00ff88"), side="left", showgrid=False, fixedrange=True),
            yaxis2=dict(title=dict(text="PVT", font=dict(color="#ff00ff")), tickfont=dict(color="#ff00ff"), overlaying="y", side="right", showgrid=False, fixedrange=True),
            legend=dict(orientation="h", y=1.12, x=0.25)
        )
        plot(fig3, "OBV & PVT", height=500)
    else:
        st.info("Enable at least one indicator to display OBV & PVT")

    # 3. MACD with descriptions
    fig4 = go.Figure()
    fig4.add_scatter(x=df.index, y=df.MACD, name="MACD (Moving Avg Convergence Divergence)", line=dict(color="blue"))
    fig4.add_scatter(x=df.index, y=df.Signal, name="Signal Line (9-day EMA)", line=dict(color="red"))
    fig4.add_bar(x=df.index, y=df.MACD_Hist, name="Histogram (MACD - Signal)", marker_color="gray", opacity=0.6)
    plot(fig4, "MACD")

# === Right Panel ===
with col_right:
    st.metric("Long-Dated P/C Ratio", f"{pcr:.2f}")
    st.metric("Options Vol (6m+)", f"{opt_vol:,.0f}")
    st.metric("Latest Volume", f"{latest_vol:,.0f}")
    st.metric("20d Avg Vol", f"{avg_vol:,.0f}")

    color = "green" if score >= 65 else "orange" if score >= 45 else "red"
    st.markdown(f"### Pressure Score\n<h1 style='color:{color}; text-align:center; font-size:3rem'>{score}/100</h1>", unsafe_allow_html=True)
    
    # Show score breakdown
    with st.expander("📊 Score Breakdown"):
        for component, value in score_components.items():
            st.markdown(f"**{component}:** {value:+.1f}")

    if not spikes.empty:
        st.markdown("### Long-Dated Unusual Activity")
        for _, r in spikes.iterrows():
            exp = datetime.strptime(r['expiry'],"%Y-%m-%d").strftime("%b %d %Y")
            st.markdown(f"**{r['type']} ${r['strike']:,}** → **{r['volume']:,.0f}** ({r['x']}× avg) | {exp}")
    else:
        st.markdown("#### No long-dated unusual activity")

# === Footer with Timestamp and Refresh ===
if last_data_time:
    timestamp_str = last_data_time.strftime("%B %d, %Y at %I:%M:%S %p %Z")
    st.caption(f"📊 Data last updated: {timestamp_str}")
else:
    st.caption("📊 Data timestamp unavailable")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ─── AI ANALYSIS SECTION ───────────────────────────────────────────
st.markdown("---")
st.header("🤖 AI Pressure & Momentum Analysis")

# Initialize chat history in session state
if "chat_history_pressure" not in st.session_state:
    st.session_state.chat_history_pressure = []
if "initial_context_pressure" not in st.session_state:
    st.session_state.initial_context_pressure = None

if not GEMINI_AVAILABLE:
    st.error("❌ Google Gemini package not installed")
    st.code("pip install google-generativeai", language="bash")
    st.info("Install the package and restart the app to enable AI analysis")
elif not gemini_api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to enable AI analysis")
    st.info("Get a free API key at: https://aistudio.google.com/app/apikey")
else:
    # Initial Analysis Button
    if st.button("🔍 Generate AI Pressure Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing buying/selling pressure with Gemini AI..."):
            try:
                # Prepare context for AI
                latest_close = df.Close.iloc[-1]
                price_change = ((latest_close - df.Close.iloc[0]) / df.Close.iloc[0]) * 100
                
                # VWAP analysis
                vwap_current = df.VWAP.iloc[-1]
                vwap_diff_pct = ((latest_close - vwap_current) / vwap_current) * 100
                vwap_position = "ABOVE" if latest_close > vwap_current else "BELOW"
                
                # OBV trend
                obv_current = df.OBV.iloc[-1]
                obv_20_ago = df.OBV.iloc[-20] if len(df) > 20 else df.OBV.iloc[0]
                obv_change_pct = ((obv_current - obv_20_ago) / abs(obv_20_ago + 1)) * 100
                obv_trend = "RISING" if obv_change_pct > 0 else "FALLING"
                
                # MACD analysis
                macd_current = df.MACD.iloc[-1]
                signal_current = df.Signal.iloc[-1]
                macd_hist = df.MACD_Hist.iloc[-1]
                macd_signal = "BULLISH CROSS" if macd_current > signal_current else "BEARISH CROSS"
                
                # Volume analysis
                vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1
                vol_status = "HIGH" if vol_ratio > 1.5 else "NORMAL" if vol_ratio > 0.5 else "LOW"
                
                # POC analysis
                poc_diff_pct = ((latest_close - poc_price) / poc_price) * 100
                poc_position = "ABOVE" if latest_close > poc_price else "BELOW"
                
                # Unusual options activity
                unusual_activity = ""
                if not spikes.empty:
                    unusual_activity = "\n".join([
                        f"- {r['type']} ${r['strike']:,}: {r['volume']:,.0f} volume ({r['x']}x avg) exp {r['expiry']}"
                        for _, r in spikes.head(5).iterrows()
                    ])
                else:
                    unusual_activity = "No significant unusual activity detected"
                
                # Get recent price action (last 10 bars)
                recent_bars = df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume', 'OBV', 'MACD']].to_string()
                
                context = f"""
Analyze this stock's buying/selling pressure and momentum to provide trading insights.

TICKER: {ticker}
TIMEFRAME: {period_options[period]}
CURRENT PRICE: ${latest_close:.2f}
PRICE CHANGE ({period_options[period]}): {price_change:+.2f}%

PRESSURE SCORE: {score}/100 ({"BULLISH" if score >= 65 else "NEUTRAL" if score >= 45 else "BEARISH"})

TECHNICAL INDICATORS:
- VWAP: ${vwap_current:.2f} (Price is {vwap_position} by {abs(vwap_diff_pct):.2f}%)
- POC (Point of Control): ${poc_price:.2f} (Price is {poc_position} by {abs(poc_diff_pct):.2f}%)
- OBV: {obv_current:,.0f} ({obv_trend}, {obv_change_pct:+.1f}% change)
- MACD: {macd_current:.2f} (Signal: {signal_current:.2f}, Histogram: {macd_hist:.2f})
- MACD Status: {macd_signal}

VOLUME ANALYSIS:
- Latest Volume: {latest_vol:,.0f}
- 20-day Average: {avg_vol:,.0f}
- Volume Status: {vol_status} ({vol_ratio:.1f}x average)

OPTIONS FLOW (Long-dated):
- Put/Call Ratio: {pcr:.2f} ({"Bullish" if pcr < 0.7 else "Bearish" if pcr > 1.3 else "Neutral"})
- Total Options Volume (6m+): {opt_vol:,.0f}

UNUSUAL OPTIONS ACTIVITY:
{unusual_activity}

RECENT PRICE ACTION (Last 10 bars):
{recent_bars}

SCORE BREAKDOWN:
{chr(10).join([f"- {k}: {v:+.1f}" for k, v in score_components.items()])}

Provide analysis in this format:

1. PRESSURE ASSESSMENT: (2-3 sentences on dominant pressure - buying or selling - and confidence level)

2. KEY TECHNICAL SIGNALS:
   - VWAP Significance: [what VWAP position tells us]
   - OBV Interpretation: [what OBV trend indicates about accumulation/distribution]
   - MACD Status: [momentum confirmation or divergence]

3. VOLUME PROFILE:
   - Current activity level: [interpretation of volume vs average]
   - What this suggests: [institutional interest, retail activity, etc.]

4. OPTIONS MARKET SENTIMENT:
   - P/C Ratio interpretation: [what options traders are positioning for]
   - Unusual activity significance: [if any large bets stand out]

5. RECOMMENDATION: **BUY** / **SELL** / **HOLD** / **WATCH**
   Reasoning: (2-3 sentences explaining the call based on pressure analysis)

6. KEY LEVELS TO WATCH:
   - Immediate support: [price level]
   - Immediate resistance: [price level]
   - Critical level: [VWAP, POC, or other significant level]

7. TRADE SETUP (if actionable):
   - Entry: [specific condition or price]
   - Stop Loss: [price level based on technicals]
   - Target: [price level]
   - Risk/Reward: [ratio]

8. RISK FACTORS:
   - [List 2-3 key risks to the thesis]

Be specific, actionable, and focused on pressure dynamics and momentum. This is for short-term to medium-term trading.
"""

                # Store initial context for follow-ups
                st.session_state.initial_context_pressure = context
                
                # Call Gemini API
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(context)
                
                # Clear previous chat and add initial exchange
                st.session_state.chat_history_pressure = [
                    {"role": "user", "content": "Analyze this pressure and momentum data"},
                    {"role": "assistant", "content": response.text}
                ]
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
                st.info("Check your API key or try again. Error details above.")
    
    # Display chat history
    if st.session_state.chat_history_pressure:
        st.markdown("### 💬 AI Conversation")
        
        # Display all messages
        for i, msg in enumerate(st.session_state.chat_history_pressure):
            if msg["role"] == "user" and i > 0:  # Skip first generic prompt
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
        
        # Follow-up question input
        st.markdown("---")
        follow_up = st.text_input(
            "💭 Ask a follow-up question:",
            placeholder="e.g., What if VWAP is breached? How does unusual options activity affect this?",
            key="follow_up_pressure"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            send_button = st.button("Send", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.chat_history_pressure = []
                st.session_state.initial_context_pressure = None
                st.rerun()
        
        if send_button and follow_up:
            with st.spinner("Thinking..."):
                try:
                    # Build conversation history for context
                    conversation = [{"role": "user", "parts": [st.session_state.initial_context_pressure]}]
                    
                    for msg in st.session_state.chat_history_pressure:
                        conversation.append({
                            "role": "user" if msg["role"] == "user" else "model",
                            "parts": [msg["content"]]
                        })
                    
                    # Add new question
                    conversation.append({"role": "user", "parts": [follow_up]})
                    
                    # Call Gemini with full conversation
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    chat = model.start_chat(history=conversation[:-1])
                    response = chat.send_message(follow_up)
                    
                    # Add to chat history
                    st.session_state.chat_history_pressure.append({"role": "user", "content": follow_up})
                    st.session_state.chat_history_pressure.append({"role": "assistant", "content": response.text})
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Disclaimer
        st.warning("⚠️ **Disclaimer**: This is AI-generated analysis for educational purposes only. Not financial advice. Always do your own research and consult with a financial advisor.")