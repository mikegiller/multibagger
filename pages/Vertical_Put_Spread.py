# VerticalPutSpread.py
import streamlit as st
import yfinance as yf
import utils  # applies the yfinance cache workaround on import
import pandas as pd
from datetime import datetime, timedelta, timezone


st.set_page_config(page_title="Vertical Put Spread Finder", layout="wide")

st.title("📉 Vertical Put Spread Finder")
st.info("**Purpose:** Model vertical put spread P&L for event-driven moves like earnings. Built for bearish directional plays with capped risk and defined max loss.")

# --- REFRESH BUTTON ---
if st.button("🔄 Refresh / Clear All"):
    # Only clear this page's keys so other pages keep their state
    for key in ["vps_ticker", "vps_downside", "vps_expiration"]:
        st.session_state.pop(key, None)
    st.rerun()

# --- USER INPUTS ---
ticker_symbol = utils.ticker_input("vps_ticker", label="Enter stock ticker (e.g., AAPL):")

if ticker_symbol:
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            st.error("Could not fetch price data for that ticker.")
        else:
            current_price = hist["Close"].iloc[-1]
            st.markdown(f"**Current {ticker_symbol} Price:** ${current_price:.2f}")

            # Downside % and expiration
            downside_pct = st.number_input(
                "Enter expected downside percentage (e.g., 10 for 10% drop):",
                min_value=0.0,
                value=10.0,
                step=1.0,
                key="vps_downside"
            )

            expirations = ticker.options
            if not expirations:
                st.warning("No options data available for this ticker.")
            else:
                expiration = st.selectbox(
                    "Select option expiration date:",
                    expirations,
                    key="vps_expiration"
                )

                if st.button("Find Best Put Spreads"):
                    # --- FETCH OPTIONS DATA ---
                    opt_chain = ticker.option_chain(expiration)
                    puts = opt_chain.puts.copy()
                    puts["mid"] = (puts["bid"] + puts["ask"]) / 2

                    if "lastTradeDate" not in puts.columns:
                        st.error("No last trade date data found in the option chain.")
                    else:
                        # --- Normalize timezone ---
                        puts["lastTradeDate"] = pd.to_datetime(puts["lastTradeDate"]).dt.tz_localize(None)

                        # Filter to recent trades (<= 3 days old)
                        now = datetime.now(timezone.utc).replace(tzinfo=None)
                        recent_cutoff = now - timedelta(days=3)
                        puts = puts[puts["lastTradeDate"] >= recent_cutoff]

                        if puts.empty:
                            st.warning("No recent option trades within the last 3 days.")
                        else:
                            # --- APPLY STRIKE FILTERS ---
                            target_price = current_price * (1 - downside_pct / 100)
                            otm_puts = puts[
                                (puts["strike"] < current_price)
                                & (puts["strike"] >= target_price)
                            ].copy()

                            otm_puts = otm_puts.sort_values("strike", ascending=False).reset_index(drop=True)

                            spreads = []
                            for i in range(len(otm_puts)):
                                for j in range(i + 1, len(otm_puts)):
                                    high_strike = otm_puts.loc[i, "strike"]  # higher strike (buy)
                                    low_strike = otm_puts.loc[j, "strike"]   # lower strike (sell)
                                    cost_high = otm_puts.loc[i, "mid"]
                                    cost_low = otm_puts.loc[j, "mid"]

                                    strike_spread = high_strike - low_strike
                                    cost_of_spread = cost_high - cost_low
                                    if cost_of_spread <= 0:
                                        continue

                                    max_profit = strike_spread - cost_of_spread
                                    return_pct = (max_profit / cost_of_spread) * 100

                                    # --- Calculate % difference from current price ---
                                    high_diff = ((high_strike / current_price) - 1) * 100
                                    low_diff = ((low_strike / current_price) - 1) * 100

                                    # --- Format for display ---
                                    high_strike_display = f"${high_strike:.2f} ({high_diff:+.2f}%)"
                                    low_strike_display = f"${low_strike:.2f} ({low_diff:+.2f}%)"

                                    spreads.append({
                                        "High Strike": high_strike_display,
                                        "Low Strike": low_strike_display,
                                        "Cost of Spread": round(cost_of_spread, 2),
                                        "Max Profit": round(max_profit, 2),
                                        "Return %": round(return_pct, 2),
                                    })

                            if not spreads:
                                st.warning("No valid spreads found for your filters.")
                            else:
                                df = pd.DataFrame(spreads)
                                df = df.sort_values("Return %", ascending=False).reset_index(drop=True)

                                # --- STYLE FUNCTION TO HIGHLIGHT RETURN RANGE ---
                                def highlight_returns(row):
                                    color = "background-color: #8B8000; color: white;" if 190 <= row["Return %"] <= 220 else ""
                                    return [color] * len(row)

                                styled_df = (
                                    df.style
                                    .apply(highlight_returns, axis=1)
                                    .format({
                                        "Cost of Spread": "${:.2f}",
                                        "Max Profit": "${:.2f}",
                                        "Return %": "{:.2f}%"
                                    })
                                )

                                st.dataframe(styled_df, width='stretch')

    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
