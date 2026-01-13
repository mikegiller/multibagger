# strike_expiration_analyzer.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
st.set_page_config(page_title="Strike Expiration Analyzer", layout="wide")

# === Sidebar ===
with st.sidebar:
    st.header("📊 Controls")
    
    # Ticker input
    ticker = st.text_input("Ticker Symbol", value="SPY", max_chars=12).upper().strip()
    
    st.markdown("---")
    
    # Calls/Puts toggle
    option_type = st.radio(
        "Option Type",
        ["Calls", "Puts"],
        index=0,
        horizontal=True
    )
    
    st.markdown("---")
    
    # AI API Key
    st.header("🤖 AI Analysis")
    
    if not GEMINI_AVAILABLE:
        st.warning("⚠️ Google Gemini not installed")
        st.code("pip install google-generativeai", language="bash")
        gemini_api_key = None
    else:
        gemini_api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            help="Get free API key at: https://aistudio.google.com/app/apikey",
            value=""
        )
        
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                st.success("✅ API configured")
            except Exception as e:
                st.error(f"❌ Failed: {str(e)}")
        else:
            st.info("💡 Add API key for AI analysis")
        
        st.caption("[Get API Key →](https://aistudio.google.com/app/apikey)")

if not ticker:
    st.warning("Please enter a ticker symbol")
    st.stop()

st.title(f"📈 Strike Expiration Analyzer – {ticker}")

# Period selection (same as Swing Scanner)
periods = {
    "1D":   "Today only (15-min)",
    "5D":   "5 days",
    "10D":  "10 days",
    "1M":   "1 month",
    "3M":   "3 months",
    "6M":   "6 months",
    "1Y":   "1 year",
    "All":  "all available"
}

st.write("**Select time range:**")
cols = st.columns(len(periods))
selected_key = None
for i, (key, label) in enumerate(periods.items()):
    with cols[i]:
        if st.button(label, key=f"btn_{key}", width="stretch"):
            selected_key = key

if "period" not in st.session_state:
    st.session_state.period = "5D"
if selected_key is not None:
    st.session_state.period = selected_key

period = st.session_state.period
st.info(f"Showing: **{periods[period]}**", icon="📊")

show_resistance = st.checkbox("Show Resistance", value=True)
show_support = st.checkbox("Show Support", value=True)

# ─── Load data ─────────────────────────────────────────────────────
with st.spinner("Loading market data..."):
    try:
        stock = yf.Ticker(ticker)

        if period == "1D":
            # Fetch 5 days of 15-min data to ensure RSI has enough history
            df_full = stock.history(period="5d", interval="15m", prepost=True)
            
            if df_full.empty:
                st.error(f"No data for {ticker}")
                st.stop()
            
            # After calculating RSI, we'll filter to today only
            today = date.today()
            today_start = pd.Timestamp(today).tz_localize('America/New_York')
            today_end = today_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        else:
            # For daily intervals, fetch all available data
            df_full = stock.history(period="max", interval="1d", auto_adjust=True)

        if df_full.empty:
            st.error(f"No data for {ticker}")
            st.stop()

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# ─── Add RSI(14) on FULL dataset ───────────────────────────────────
delta = df_full['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14, min_periods=14).mean()
avg_loss = loss.rolling(window=14, min_periods=14).mean()
rs = avg_gain / avg_loss
df_full['RSI'] = 100 - (100 / (1 + rs))

# ─── NOW filter to selected display period ────────────────────────
if period == "1D":
    df = df_full[(df_full.index >= today_start) & (df_full.index <= today_end)].copy()
    
    if df.empty and len(df_full) > 0:
        last_day = df_full.index.normalize().max()
        df = df_full[df_full.index.normalize() == last_day].copy()
        st.info("Today has no data yet → showing last trading day", icon="ℹ️")
else:
    today = df_full.index[-1]
    if period == "5D": 
        start = today - pd.Timedelta(days=7)
    elif period == "10D": 
        start = today - pd.Timedelta(days=14)
    elif period == "1M": 
        start = today - pd.Timedelta(days=40)
    elif period == "3M": 
        start = today - pd.Timedelta(days=100)
    elif period == "6M": 
        start = today - pd.Timedelta(days=200)
    elif period == "1Y": 
        start = today - pd.Timedelta(days=370)
    else: 
        start = df_full.index[0]
    
    df = df_full[df_full.index >= start].copy()

current_price = float(df['Close'].iloc[-1])

# ─── Swing detection ───────────────────────────────────────────────
def find_swings(df, window=5):
    df = df.copy()
    roll_window = window * 2 + 1
    high_max = df['High'].rolling(window=roll_window, center=True, min_periods=roll_window//2).max()
    df['swing_high'] = df['High'][df['High'] == high_max]
    low_min = df['Low'].rolling(window=roll_window, center=True, min_periods=roll_window//2).min()
    df['swing_low'] = df['Low'][df['Low'] == low_min]
    recent_highs = df['swing_high'].dropna().tail(4)
    recent_lows = df['swing_low'].dropna().tail(4)
    return recent_highs, recent_lows

recent_highs, recent_lows = find_swings(df, window=5)

# ─── Volume spikes ─────────────────────────────────────────────────
avg_vol = df['Volume'].rolling(window=20, min_periods=5).mean()
df['volume_spike'] = df['Volume'] > 1.8 * avg_vol
spike_dates = df[df['volume_spike']].index

# ─── Chart ─────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.70, 0.30],
    vertical_spacing=0.06,
    shared_xaxes=True,
    subplot_titles=(
        f"{ticker} – Current ≈ ${current_price:,.2f}",
        "Volume"
    ),
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
)

# Candlestick (prominent styling)
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    increasing_line_color='#00c853',
    decreasing_line_color='#d50000',
    increasing_fillcolor='#00c853',
    decreasing_fillcolor='#d50000',
    line=dict(width=1.5),
    name='Price'
), row=1, col=1, secondary_y=False)

fig.add_hline(y=current_price, line_dash="dash", line_color="#448aff",
              annotation_text=f"Current ${current_price:,.2f}",
              annotation_position="top right", row=1, col=1)

# Resistance & Support
if show_resistance:
    for level in recent_highs:
        fig.add_hline(y=level, line_width=1.5, line_color="#ff4d4d",
                      annotation_text=f"R ▼▼▼ ${level:,.2f}",
                      annotation_position="top right",
                      annotation_font_size=10, row=1, col=1)

if show_support:
    for level in recent_lows:
        fig.add_hline(y=level, line_width=1.5, line_color="#4ecdc4",
                      annotation_text=f"S ▲▲▲ ${level:,.2f}",
                      annotation_position="bottom right",
                      annotation_font_size=10, row=1, col=1)

# RSI on secondary y-axis (subtle styling)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['RSI'],
    line=dict(color='rgba(255, 193, 7, 1)', width=3, dash='dot'),
    name='RSI(14)',
    opacity=1
), row=1, col=1, secondary_y=True)

# RSI reference lines on secondary y-axis (very subtle)
fig.add_hline(y=70, line_dash="dot", line_color="rgba(255, 82, 82, 0.2)", 
              line_width=0.5, row=1, col=1, secondary_y=True)
fig.add_hline(y=30, line_dash="dot", line_color="rgba(76, 175, 80, 0.2)",
              line_width=0.5, row=1, col=1, secondary_y=True)
fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.1)", 
              line_width=0.5, row=1, col=1, secondary_y=True)

# Volume
fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                     marker_color='rgba(100,181,246,0.4)',
                     name='Volume'),
              row=2, col=1)

if not spike_dates.empty:
    fig.add_trace(go.Scatter(
        x=spike_dates,
        y=df.loc[spike_dates, 'Volume'] * 1.05,
        mode='markers',
        marker=dict(symbol='diamond', size=12, color='yellow', line=dict(width=1, color='black')),
        name='Volume Spike',
        hovertemplate='Spike<br>%{x}<br>Vol: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)

# Update layout
fig.update_layout(
    height=780,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode="x unified",
    template="plotly_dark",
    margin=dict(t=100, b=50, l=60, r=60),
    dragmode=False,
    xaxis_rangeslider_visible=False,
    xaxis_fixedrange=True,
    xaxis2_fixedrange=True,
    yaxis_fixedrange=True,
    yaxis2_fixedrange=True
)

# Update y-axes
fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Volume", tickformat="~s", row=2, col=1)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.caption("**Red ▼▼▼** = Resistance | **Cyan ▲▲▲** = Support | **Yellow diamonds** = Volume spikes (>1.8× avg) | **RSI >70** overbought, **<30** oversold")

# ─── OPTIONS ANALYSIS ──────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📋 {option_type} Options Analysis")

# Calculate strike range (90% to 110% of current price)
min_strike = current_price * 0.9
max_strike = current_price * 1.1

try:
    # Get all expirations in next 6 months
    expirations = stock.options
    today_dt = datetime.today()
    six_months = today_dt + pd.Timedelta(days=180)
    
    valid_expirations = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        if today_dt <= exp_date <= six_months:
            valid_expirations.append(exp_str)
    
    if not valid_expirations:
        st.warning("No options expirations found in the next 6 months")
        st.stop()
    
    # Get all options data for the first expiration to find available strikes
    first_chain = stock.option_chain(valid_expirations[0])
    if option_type == "Calls":
        options_df = first_chain.calls
    else:
        options_df = first_chain.puts
    
    # Filter strikes in range and sort DESCENDING
    available_strikes = sorted(options_df[
        (options_df['strike'] >= min_strike) & 
        (options_df['strike'] <= max_strike)
    ]['strike'].unique(), reverse=True)
    
    if not available_strikes:
        st.warning(f"No strikes found between ${min_strike:.2f} and ${max_strike:.2f}")
        st.stop()
    
    # Find strike closest to current price
    closest_strike = min(available_strikes, key=lambda x: abs(x - current_price))
    closest_index = available_strikes.index(closest_strike)
    
    widget_key = f"strike_select_{option_type}"
    
    # Determine which index to use
    if widget_key in st.session_state and st.session_state[widget_key] in available_strikes:
        # Use the stored value's index
        use_index = available_strikes.index(st.session_state[widget_key])
    else:
        # Use the closest strike's index
        use_index = closest_index
    
    # Strike selection dropdown
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_strike = st.selectbox(
            f"Select Strike (${min_strike:.2f} - ${max_strike:.2f})",
            available_strikes,
            index=use_index,
            format_func=lambda x: f"${x:.2f} ({((x/current_price - 1) * 100):+.1f}%)",
            key=widget_key
        )
    
    with col2:
        show_active_only = st.checkbox(
            "Show only active contracts",
            value=True,
            help="Volume > 0 or Open Interest > 10"
        )
    
    st.markdown(f"### Strike: ${selected_strike:.2f} ({option_type})")
    st.caption(f"Current Price: ${current_price:.2f} | Distance: {((selected_strike/current_price - 1) * 100):+.1f}%")
    
    # Build table for selected strike across all expirations
    rows = []
    for exp_str in valid_expirations:
        try:
            chain = stock.option_chain(exp_str)
            if option_type == "Calls":
                opts = chain.calls
            else:
                opts = chain.puts
            
            # Find the selected strike
            strike_data = opts[opts['strike'] == selected_strike]
            
            if strike_data.empty:
                continue
            
            row = strike_data.iloc[0]
            
            # Calculate metrics
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            days_to_exp = (exp_date - today_dt).days
            
            if days_to_exp <= 0:
                continue
            
            bid = row['bid']
            ask = row['ask']
            last = row['lastPrice']
            volume = row['volume']
            oi = row['openInterest']
            
            mid = (bid + ask) / 2
            mid_pct = (mid / current_price) * 100
            mid_pct_per_day = mid_pct / days_to_exp if days_to_exp > 0 else 0
            
            # Apply active filter
            is_active = volume > 0 or oi > 10
            if show_active_only and not is_active:
                continue
            
            rows.append({
                'Expiration': exp_str,
                'Days': days_to_exp,
                'Bid': f"${bid:.2f}",
                'Ask': f"${ask:.2f}",
                'Last': f"${last:.2f}",
                'Mid': f"${mid:.2f}",
                'Mid %': f"{mid_pct:.2f}%",
                'Mid % / Day': f"{mid_pct_per_day:.4f}%",
                'Volume': int(volume),
                'OI': int(oi),
                '_mid_raw': mid,
                '_mid_pct_raw': mid_pct,
                '_mid_per_day_raw': mid_pct_per_day
            })
            
        except Exception as e:
            continue
    
    if not rows:
        st.warning("No data available for this strike (try disabling 'active only' filter)")
    else:
        # Create DataFrame
        table_df = pd.DataFrame(rows)
        
        # Display table
        st.dataframe(
            table_df[['Expiration', 'Days', 'Bid', 'Ask', 'Last', 'Mid', 'Mid %', 'Mid % / Day', 'Volume', 'OI']],
            width='stretch',
            hide_index=True,
            height=400
        )
        
        st.caption(f"**Mid %**: Premium as % of stock price | **Mid % / Day**: Daily income rate from premium")
        
except Exception as e:
    st.error(f"Error loading options data: {str(e)}")
    st.info("Market may be closed or options data unavailable")

# ─── AI ANALYSIS SECTION ───────────────────────────────────────────
st.markdown("---")
st.header("🤖 AI Covered Call Strategy Analysis")

# Initialize chat history in session state
if "chat_history_strike" not in st.session_state:
    st.session_state.chat_history_strike = []
if "initial_context_strike" not in st.session_state:
    st.session_state.initial_context_strike = None

if not GEMINI_AVAILABLE:
    st.error("❌ Google Gemini package not installed")
    st.code("pip install google-generativeai", language="bash")
    st.info("Install the package and restart the app to enable AI analysis")
elif not gemini_api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to enable AI analysis")
    st.info("Get a free API key at: https://aistudio.google.com/app/apikey")
else:
    # Initial Analysis Button
    if st.button("🔍 Analyze Best Strike/Expiration Combo", type="primary", width="stretch"):
        if not rows:
            st.error("No options data available. Please select a valid strike with data.")
        else:
            with st.spinner("Analyzing covered call opportunities with Gemini AI..."):
                try:
                    # Prepare context for AI
                    latest_rsi = df['RSI'].iloc[-1]
                    rsi_5_ago = df['RSI'].iloc[-5] if len(df) >= 5 else df['RSI'].iloc[0]
                    rsi_trend = "rising" if latest_rsi > rsi_5_ago else "falling"
                    
                    # Determine RSI condition
                    if latest_rsi > 70:
                        rsi_status = "OVERBOUGHT (>70)"
                    elif latest_rsi < 30:
                        rsi_status = "OVERSOLD (<30)"
                    else:
                        rsi_status = "NEUTRAL (30-70)"
                    
                    support_levels = [f"${x:.2f}" for x in recent_lows]
                    resistance_levels = [f"${x:.2f}" for x in recent_highs]
                    
                    # Get top 5 by Mid % / Day
                    analysis_df = pd.DataFrame(rows).sort_values('_mid_per_day_raw', ascending=False)
                    top_5 = analysis_df.head(5)[['Expiration', 'Days', 'Mid', 'Mid %', 'Mid % / Day', 'Volume', 'OI']].to_string(index=False)
                    
                    # Get all data for context
                    all_data = analysis_df[['Expiration', 'Days', 'Mid', 'Mid %', 'Mid % / Day']].to_string(index=False)
                    
                    context = f"""
Analyze these covered call opportunities for {ticker} and recommend the best strike/expiration combination.

TICKER: {ticker}
CURRENT PRICE: ${current_price:.2f}
SELECTED STRIKE: ${selected_strike:.2f} ({option_type})
STRIKE DISTANCE: {((selected_strike/current_price - 1) * 100):+.1f}%

TECHNICAL ANALYSIS:
- RSI(14): {latest_rsi:.1f} - {rsi_status} (trend: {rsi_trend})
- Support Levels: {', '.join(support_levels)}
- Resistance Levels: {', '.join(resistance_levels)}

TOP 5 EXPIRATIONS BY DAILY INCOME RATE:
{top_5}

ALL AVAILABLE EXPIRATIONS FOR THIS STRIKE:
{all_data}

STRATEGY CONTEXT:
This is for selling covered {option_type.lower()} - the investor owns the stock at ${current_price:.2f} and wants to generate premium income while managing risk.

Provide analysis in this format:

1. STRIKE ASSESSMENT: (2-3 sentences on whether ${selected_strike:.2f} is a good strike given current price, RSI, and support/resistance levels)

2. TOP 3 RECOMMENDED EXPIRATIONS:
   For each, provide:
   - Expiration date & days
   - Premium (Mid): $[amount]
   - Daily income rate: [Mid % / Day]
   - Why this one: [1-2 sentences on pros/cons]

3. INCOME ANALYSIS:
   - Best for maximum daily income: [which expiration and why]
   - Best for balance of premium vs time: [which expiration and why]
   - Best for conservative approach: [which expiration and why]

4. RISK CONSIDERATIONS:
   - Assignment risk: [based on strike vs current price and RSI]
   - Technical levels: [how support/resistance affects this trade]
   - What could go wrong: [2-3 key risks]

5. OVERALL RECOMMENDATION:
   - Recommended expiration: [specific date]
   - Expected premium: $[amount] ([Mid %]%)
   - Annualized return: [calculate]
   - Action: SELL NOW / WAIT FOR BETTER ENTRY / AVOID

6. ALTERNATIVE SUGGESTIONS:
   - If not this strike, what strike would be better and why?
   - Should they consider further out in time (>6 months)?

Be specific with numbers and dates. Focus on practical execution for a covered call seller.
"""

                    # Store initial context for follow-ups
                    st.session_state.initial_context_strike = context
                    
                    # Call Gemini API
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(context)
                    
                    # Clear previous chat and add initial exchange
                    st.session_state.chat_history_strike = [
                        {"role": "user", "content": "Analyze the best covered call strategy for this strike"},
                        {"role": "assistant", "content": response.text}
                    ]
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")
                    st.info("Check your API key or try again. Error details above.")
    
    # Display chat history
    if st.session_state.chat_history_strike:
        st.markdown("### 💬 AI Conversation")
        
        # Display all messages
        for i, msg in enumerate(st.session_state.chat_history_strike):
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
            placeholder="e.g., What if I want a longer expiration? Should I wait for a pullback?",
            key="follow_up_strike"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            send_button = st.button("Send", type="primary", width="stretch")
        with col2:
            if st.button("🗑️ Clear Conversation", width="stretch"):
                st.session_state.chat_history_strike = []
                st.session_state.initial_context_strike = None
                st.rerun()
        
        if send_button and follow_up:
            with st.spinner("Thinking..."):
                try:
                    # Build conversation history for context
                    conversation = [{"role": "user", "parts": [st.session_state.initial_context_strike]}]
                    
                    for msg in st.session_state.chat_history_strike:
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
                    st.session_state.chat_history_strike.append({"role": "user", "content": follow_up})
                    st.session_state.chat_history_strike.append({"role": "assistant", "content": response.text})
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Disclaimer
        st.warning("⚠️ **Disclaimer**: This is AI-generated analysis for educational purposes only. Not financial advice. Options trading involves significant risk. Always do your own research and consult with a financial advisor.")