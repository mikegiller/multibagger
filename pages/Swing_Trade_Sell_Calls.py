import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional: Google Gemini (only needed for AI analysis)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="Swing Trade / Covered Call Scanner", layout="wide")

st.title("Short Term Swing + Premium Selling Helper")

# ─── Gemini API Key Setup ──────────────────────────────────────────
with st.sidebar:
    st.header("🤖 AI Analysis Settings")
    
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
                st.success("✅ Gemini API configured")
            except Exception as e:
                st.error(f"❌ API configuration failed: {str(e)}")
        else:
            st.info("💡 Add API key to enable AI analysis")
        
        st.markdown("---")
        st.caption("**Free Tier**: 1,500 requests/day")
        st.caption("[Get API Key →](https://aistudio.google.com/app/apikey)")

st.write("")  # Add spacing

ticker = st.text_input("Ticker", value="NVDA", max_chars=12).upper().strip()
if not ticker:
    st.stop()

# Period selection
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
        if st.button(label, key=f"btn_{key}", use_container_width=True):
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
    line=dict(color='rgba(255, 193, 7, 0.5)', width=1, dash='dot'),
    name='RSI(14)',
    opacity=0.6
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


# ─── OPTIONS TABLES ────────────────────────────────────────────────────
st.subheader(f"Options Chain – {ticker}  |  Expiration: {st.session_state.get('selected_exp', 'Not loaded')}")

try:
    expirations = stock.options
    today_dt = datetime.today()

    exp_list = []
    exp_map = {}
    for d in expirations:
        try:
            exp_date = datetime.strptime(d, "%Y-%m-%d")
            days = (exp_date - today_dt).days
            if days >= -1:
                label = f"{d}  ({days}d)"
                exp_list.append(label)
                exp_map[label] = d
        except:
            pass

    if not exp_list:
        st.warning("No upcoming expirations found")
    else:
        selected_label = st.selectbox("Select Expiration", exp_list, index=0)
        selected_exp = exp_map[selected_label]
        st.session_state.selected_exp = selected_exp  # save for display

        chain = stock.option_chain(selected_exp)
        calls = chain.calls
        puts = chain.puts

        # ── Shared formatting & highlighting logic ───────────────────────
        res_levels = list(recent_highs)
        sup_levels = list(recent_lows)

        def prepare_options_df(df_opt, is_call=True):
            df = df_opt[['strike', 'bid', 'ask', 'impliedVolatility']].copy()

            # ──────────────── FILTER: ATM → slightly OTM only ────────────
            if is_call:
                # Calls: 100% to 110% of current price
                df = df[(df['strike'] >= current_price) & 
                        (df['strike'] <= current_price * 1.10)]
            else:
                # Puts: 90% to 100% of current price
                df = df[(df['strike'] >= current_price * 0.90) & 
                        (df['strike'] <= current_price)]

            df['Mid'] = ((df['bid'] + df['ask']) / 2).round(2)
            df['Mid %'] = (df['Mid'] / current_price * 100).round(2).astype(str) + ' %'
            df['IV%']  = (df['impliedVolatility'] * 100).round(1).astype(str) + ' %'
            df['Dist%'] = ((df['strike'] - current_price) / current_price * 100).round(1)
            df['Dist%'] = df['Dist%'].apply(lambda x: f"{x:+.1f}%")
            
            # Sort: calls highest → lowest, puts lowest → highest (more natural for each)
            if is_call:
                display_df = df[['strike', 'Dist%', 'bid', 'ask', 'Mid', 'Mid %', 'IV%']]\
                               .sort_values('strike', ascending=False)
            else:
                display_df = df[['strike', 'Dist%', 'bid', 'ask', 'Mid', 'Mid %', 'IV%']]\
                               .sort_values('strike', ascending=False)

            # Find closest strikes to levels (only among the filtered rows)
            indices = []
            for level in (res_levels + sup_levels):
                if len(display_df) == 0:
                    continue
                idx = (display_df['strike'] - level).abs().idxmin()
                if idx not in indices:
                    indices.append(idx)

            return display_df, indices

        def highlight_rows(row, highlight_indices):
            styles = [''] * len(row)
            if row.name in highlight_indices:
                closest = min(res_levels + sup_levels, key=lambda x: abs(x - row['strike']))
                if closest in res_levels:
                    styles = ['background-color: #4d0000; color: white'] * len(row)  # dark red
                else:
                    styles = ['background-color: #004d4d; color: white'] * len(row)  # dark cyan
            return styles

        # ── CALLS ────────────────────────────────────────────────────────
        st.markdown("### Calls")
        call_df, call_highlight_idx = prepare_options_df(calls, is_call=True)
        styled_calls = call_df.style.apply(
            lambda row: highlight_rows(row, call_highlight_idx), axis=1
        ).format(precision=2, subset=['strike','bid','ask','Mid'])

        st.markdown(f"**Current ≈ ${current_price:,.2f}**")
        st.dataframe(styled_calls, width='stretch', height='auto')

        # ── PUTS ─────────────────────────────────────────────────────────
        st.markdown("### Puts")
        put_df, put_highlight_idx = prepare_options_df(puts, is_call=False)
        styled_puts = put_df.style.apply(
            lambda row: highlight_rows(row, put_highlight_idx), axis=1
        ).format(precision=2, subset=['strike','bid','ask','Mid'])

        st.markdown(f"**Current ≈ ${current_price:,.2f}**")
        st.caption(f"Rows in this table: {len(styled_puts.data)}   ← debug")
        st.dataframe(styled_puts, width='stretch', height='auto')

except Exception as e:
    st.error("Could not load options chain")
    st.info("Market closed / no options / yfinance issue?")

st.markdown("---")
st.caption("**Tip:** Look for strikes near support/resistance with good premium (Mid) and elevated IV%")
st.caption("**Red background** ≈ near resistance   |   **Cyan background** ≈ near support")

# ─── AI ANALYSIS SECTION ───────────────────────────────────────────
st.markdown("---")
st.header("🤖 AI Trading Analysis")

# Initialize chat history in session state
if "chat_history_swing" not in st.session_state:
    st.session_state.chat_history_swing = []
if "initial_context_swing" not in st.session_state:
    st.session_state.initial_context_swing = None

if not GEMINI_AVAILABLE:
    st.error("❌ Google Gemini package not installed")
    st.code("pip install google-generativeai", language="bash")
    st.info("Install the package and restart the app to enable AI analysis")
elif not gemini_api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to enable AI analysis")
    st.info("Get a free API key at: https://aistudio.google.com/app/apikey")
else:
    # Initial Analysis Button
    if st.button("🔍 Generate AI Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing market data with Gemini AI..."):
            try:
                # Prepare context for AI
                latest_rsi = df['RSI'].iloc[-1]
                rsi_5_ago = df['RSI'].iloc[-5] if len(df) >= 5 else df['RSI'].iloc[0]
                rsi_trend = "rising" if latest_rsi > rsi_5_ago else "falling"
                
                price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                
                support_levels = [f"${x:.2f}" for x in recent_lows]
                resistance_levels = [f"${x:.2f}" for x in recent_highs]
                
                spike_count = int(df['volume_spike'].sum())
                
                # Determine RSI condition
                if latest_rsi > 70:
                    rsi_status = "OVERBOUGHT (>70)"
                elif latest_rsi < 30:
                    rsi_status = "OVERSOLD (<30)"
                else:
                    rsi_status = "NEUTRAL (30-70)"
                
                # Get last 10 bars for recent pattern
                recent_bars = df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].to_string()
                
                context = f"""
Analyze this stock and provide a clear trading recommendation.

TICKER: {ticker}
TIMEFRAME: {periods[period]}
CURRENT PRICE: ${current_price:.2f}

TECHNICAL INDICATORS:
- RSI(14): {latest_rsi:.1f} - {rsi_status} (trend: {rsi_trend})
- Support Levels: {', '.join(support_levels)}
- Resistance Levels: {', '.join(resistance_levels)}
- Price Change ({period}): {price_change:+.2f}%
- Volume Spikes: {spike_count} occurrences

PRICE RANGE:
- High: ${df['High'].max():.2f}
- Low: ${df['Low'].min():.2f}
- Average Volume: {df['Volume'].mean():.0f}

RECENT PRICE ACTION (Last 10 bars):
{recent_bars}

Provide analysis in this format:

1. MARKET SENTIMENT: (1-2 sentences about current trend and momentum)

2. KEY LEVELS TO WATCH:
   - Critical support: [price]
   - Critical resistance: [price]

3. RSI INTERPRETATION: (What does current RSI tell us?)

4. RECOMMENDATION: **BUY** / **SELL** / **HOLD**
   Reasoning: (2-3 sentences explaining why)

5. RISK LEVEL: Low / Medium / High
   Why: (1 sentence)

6. ACTION PLAN:
   - Entry: [specific price or condition]
   - Stop Loss: [price]
   - Target: [price]

Be concise, specific, and actionable. Focus on swing trading (days to weeks).
"""

                # Store initial context for follow-ups
                st.session_state.initial_context_swing = context
                
                # Call Gemini API
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(context)
                
                # Clear previous chat and add initial exchange
                st.session_state.chat_history_swing = [
                    {"role": "user", "content": "Analyze this stock data"},
                    {"role": "assistant", "content": response.text}
                ]
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")
                st.info("Check your API key or try again. Error details above.")
    
    # Display chat history
    if st.session_state.chat_history_swing:
        st.markdown("### 💬 AI Conversation")
        
        # Display all messages
        for i, msg in enumerate(st.session_state.chat_history_swing):
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
            placeholder="e.g., What if the price breaks above resistance? Should I use options?",
            key="follow_up_swing"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            send_button = st.button("Send", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.chat_history_swing = []
                st.session_state.initial_context_swing = None
                st.rerun()
        
        if send_button and follow_up:
            with st.spinner("Thinking..."):
                try:
                    # Build conversation history for context
                    conversation = [{"role": "user", "parts": [st.session_state.initial_context_swing]}]
                    
                    for msg in st.session_state.chat_history_swing:
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
                    st.session_state.chat_history_swing.append({"role": "user", "content": follow_up})
                    st.session_state.chat_history_swing.append({"role": "assistant", "content": response.text})
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Disclaimer
        st.warning("⚠️ **Disclaimer**: This is AI-generated analysis for educational purposes only. Not financial advice. Always do your own research and consult with a financial advisor.")