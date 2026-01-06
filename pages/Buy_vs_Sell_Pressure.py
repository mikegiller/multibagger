# app.py — ULTIMATE BUYING vs SELLING PRESSURE DASHBOARD (Dec 2025)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# === Page Config ===
st.set_page_config(page_title="Pressure Dashboard", layout="wide", initial_sidebar_state="expanded")

# === Sidebar ===
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Ticker", "SPY").upper()

period_options = {
    "1 Day": "1d", "1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo",
    "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"
}
period_name = st.sidebar.selectbox("Period", list(period_options.keys()), index=0)
period = period_options[period_name]

if st.sidebar.button("Reset Zoom & Settings"):
    st.session_state.clear()
    st.rerun()

# === Live Price ===
stock = yf.Ticker(ticker)
info = stock.info
current_price = info.get("regularMarketPrice") or info.get("previousClose") or "N/A"
st.title(f"{ticker} — Buying vs Selling Pressure")
st.markdown(f"### Current Price: **${current_price:,}**")

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

# === Plotly Config (Disable zoom/select) ===
config = {
    "displayModeBar": False,
    "scrollZoom": False,
    "displaylogo": False
}

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
    fig.update_xaxes(fixedrange=True)
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
    
    # Price candlestick
    fig1.add_trace(
        go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"),
        row=1, col=1
    )
    
    # Checkbox with description for VWAP
    col_vwap, col_poc = st.columns(2)
    with col_vwap:
        show_vwap = st.checkbox("VWAP", True, key="vwap", help="Volume-Weighted Average Price - average price weighted by volume, used to identify support/resistance levels")
    with col_poc:
        show_poc = st.checkbox("POC", True, key="poc", help="Point of Control - price level with highest traded volume, acts as a magnet for price action")
    
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
    
    fig1.update_xaxes(fixedrange=True)
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