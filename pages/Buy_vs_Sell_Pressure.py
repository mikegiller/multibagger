# app.py — ULTIMATE BUYING vs SELLING PRESSURE DASHBOARD (Dec 2025)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# SSL fix for Mac
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

# Optional: Google Gemini (only needed for AI analysis)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
TIMEFRAME: {period_name}
CURRENT PRICE: ${latest_close:.2f}
PRICE CHANGE ({period_name}): {price_change:+.2f}%

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