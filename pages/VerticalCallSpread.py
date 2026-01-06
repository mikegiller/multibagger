import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Vertical Call Spread Finder", layout="wide")

st.title("📈 Vertical Call Spread Finder")

# --- REFRESH BUTTON ---
if st.button("🔄 Refresh / Clear All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- USER INPUTS ---
ticker_symbol = st.text_input("Enter stock ticker (e.g., AAPL):", key="ticker_input").upper()

if ticker_symbol:
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            st.error("Could not fetch price data for that ticker.")
        else:
            current_price = hist["Close"].iloc[-1]
            st.markdown(f"**Current {ticker_symbol} Price:** ${current_price:.2f}")

            # Upside % and expiration
            upside_pct = st.number_input(
                "Enter expected upside percentage (e.g., 10 for 10%):",
                min_value=0.0,
                value=10.0,
                step=1.0,
                key="upside_input"
            )

            expirations = ticker.options
            if not expirations:
                st.warning("No options data available for this ticker.")
            else:
                expiration = st.selectbox(
                    "Select option expiration date:",
                    expirations,
                    key="expiration_select"
                )

                if st.button("Find Best Call Spreads"):
                    # --- FETCH OPTIONS DATA ---
                    opt_chain = ticker.option_chain(expiration)
                    calls = opt_chain.calls.copy()
                    calls["mid"] = (calls["bid"] + calls["ask"]) / 2

                    if "lastTradeDate" not in calls.columns:
                        st.error("No last trade date data found in the option chain.")
                    else:
                        # --- FIX: Normalize timezone ---
                        calls["lastTradeDate"] = pd.to_datetime(calls["lastTradeDate"]).dt.tz_localize(None)

                        # Filter to recent trades (<= 3 days old)
                        now = datetime.utcnow()
                        recent_cutoff = now - timedelta(days=3)
                        calls = calls[calls["lastTradeDate"] >= recent_cutoff]

                        if calls.empty:
                            st.warning("No recent option trades within the last 3 days.")
                        else:
                            # --- APPLY STRIKE FILTERS ---
                            target_price = current_price * (1 + upside_pct / 100)
                            otm_calls = calls[
                                (calls["strike"] > current_price)
                                & (calls["strike"] <= target_price)
                            ].copy()

                            otm_calls = otm_calls.sort_values("strike").reset_index(drop=True)

                            spreads = []
                            for i in range(len(otm_calls)):
                                for j in range(i + 1, len(otm_calls)):
                                    low_strike = otm_calls.loc[i, "strike"]
                                    high_strike = otm_calls.loc[j, "strike"]
                                    cost_low = otm_calls.loc[i, "mid"]
                                    cost_high = otm_calls.loc[j, "mid"]

                                    strike_spread = high_strike - low_strike
                                    cost_of_spread = cost_low - cost_high
                                    if cost_of_spread <= 0:
                                        continue

                                    max_profit = strike_spread - cost_of_spread
                                    return_pct = (max_profit / cost_of_spread) * 100

                                    # --- Calculate % difference from current price ---
                                    low_diff = ((low_strike / current_price) - 1) * 100
                                    high_diff = ((high_strike / current_price) - 1) * 100

                                    # --- Format for display ---
                                    low_strike_display = f"${low_strike:.2f} ({low_diff:+.2f}%)"
                                    high_strike_display = f"${high_strike:.2f} ({high_diff:+.2f}%)"

                                    spreads.append({
                                        "Low Strike": low_strike_display,
                                        "High Strike": high_strike_display,
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
                                    # Dark gold background for good white contrast
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

                                st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
