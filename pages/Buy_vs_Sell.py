# app.py — ULTIMATE BUYING vs SELLING PRESSURE DASHBOARD (Dec 2025)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# === Page Config ===
st.set_page_config(page_title="Pressure Dashboard", layout="wide", initial_sidebar_state="expanded")

# === Sidebar ===
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Ticker", "SPY").upper()

period_options = {
    "1 Day": "1d", "1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo",
    "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"
}
period_name = st.sidebar.selectbox("Period", list(period_options.keys()), index=3)
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
        return None, None, None, None, None

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

    return hist, pcr, opt_vol, unusual, avg

df, pcr, opt_vol, spikes, avg_opt = get_data(ticker, period)
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

# === Pressure Score ===
score = 50
if df.Close.iloc[-1] > df.VWAP.iloc[-1]: score += 25
else: score -= 25
if len(df)>20 and df.OBV.iloc[-1] > df.OBV.iloc[-20]: score += 20
else: score -= 20
if df.Close.iloc[-1] > poc_price: score += 25
else: score -= 25
if period not in ["1d","5d"]:
    if pcr < 0.8: score += 15
    if pcr > 1.3: score -= 15
score = max(0, min(105, score))

# === Shared Zoom State ===
if "zoom_range" not in st.session_state:
    st.session_state.zoom_range = None

# === Plotly Config ===
config = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}

# === Universal Chart Function (ensures perfect alignment) ===
def plot(fig, title=None, height=380):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=16), x=0.02, xanchor="left"),
        legend=dict(orientation="h", y=1.05, x=0),
        uirevision="constant",
        margin=dict(t=90, l=60, r=60, b=50),
        hovermode="x"
    )
    if st.session_state.zoom_range:
        fig.update_xaxes(range=st.session_state.zoom_range)
    st.plotly_chart(fig, use_container_width=True, config=config)

# === Layout ===
col_left, col_right = st.columns([3.4, 1])

with col_left:
    # 1. Price Chart
    fig1 = go.Figure()
    fig1.add_candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price")
    if st.checkbox("VWAP", True, key="vwap"):
        fig1.add_scatter(x=df.index, y=df.VWAP, name="VWAP", line=dict(color="orange", width=3))
    if st.checkbox("POC", True, key="poc"):
        fig1.add_scatter(x=df.index, y=[poc_price]*len(df), name="POC", line=dict(color="purple", dash="dot", width=3))
    plot(fig1, height=500)

    # 2. Volume
    fig2 = go.Figure()
    fig2.add_bar(x=df.index, y=df.Volume, name="Volume", marker_color="rgba(255,100,100,0.7)")
    fig2.add_hline(y=avg_vol, line_dash="dash", line_color="cyan", annotation_text="20d Avg Vol")
    plot(fig2, "Stock Volume")

    # 3. OBV & PVT — Toggleable
    st.subheader("OBV & PVT")

    col_cb1, col_cb2 = st.columns(2)
    show_obv = col_cb1.checkbox("OBV (On-Balance Volume)", value=True, key="obv_toggle")
    show_pvt = col_cb2.checkbox("PVT (Price-Volume Trend)", value=True, key="pvt_toggle")

    fig3 = go.Figure()

    if show_obv:
        fig3.add_scatter(x=df.index, y=df.OBV, name="OBV", line=dict(color="#00ff88", width=3), yaxis="y")
    if show_pvt:
        fig3.add_scatter(x=df.index, y=df.PVT, name="PVT", line=dict(color="#ff00ff", width=3), yaxis="y2")

    if show_obv or show_pvt:
        fig3.update_layout(
            yaxis=dict(title=dict(text="OBV", font=dict(color="#00ff88")), tickfont=dict(color="#00ff88"), side="left", showgrid=False),
            yaxis2=dict(title=dict(text="PVT", font=dict(color="#ff00ff")), tickfont=dict(color="#ff00ff"), overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0.25)
        )
        plot(fig3, "OBV & PVT", height=500)
    else:
        st.info("Enable at least one indicator to display OBV & PVT")

    # 4. MACD
    fig4 = go.Figure()
    fig4.add_scatter(x=df.index, y=df.MACD, name="MACD", line=dict(color="blue"))
    fig4.add_scatter(x=df.index, y=df.Signal, name="Signal", line=dict(color="red"))
    fig4.add_bar(x=df.index, y=df.MACD_Hist, name="Histogram", marker_color="gray", opacity=0.6)
    plot(fig4, "MACD")

# === Right Panel ===
with col_right:
    st.metric("Long-Dated P/C Ratio", f"{pcr:.2f}")
    st.metric("Options Vol (6m+)", f"{opt_vol:,.0f}")
    st.metric("Latest Volume", f"{df.Volume.iloc[-1]:,.0f}")
    st.metric("20d Avg Vol", f"{avg_vol:,.0f}")

    color = "green" if score >= 70 else "orange" if score >= 45 else "red"
    st.markdown(f"### Pressure Score\n<h1 style='color:{color}; text-align:center; font-size:3rem'>{int(score)}/100</h1>", unsafe_allow_html=True)

    if not spikes.empty:
        st.markdown("### Long-Dated Unusual Activity")
        for _, r in spikes.iterrows():
            exp = datetime.strptime(r['expiry'],"%Y-%m-%d").strftime("%b %d %Y")
            st.markdown(f"**{r['type']} ${r['strike']:,}** → **{r['volume']:,.0f}** ({r['x']}× avg) | {exp}")
    else:
        st.markdown("#### No long-dated unusual activity")

st.caption("All charts zoom together • Toggle indicators • Real-time data from Yahoo Finance")
