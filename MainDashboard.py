# MainDashboard.py
import streamlit as st

st.set_page_config(page_title="Options Analysis Hub", layout="wide")

st.title("🧭 Options Analysis Hub")

st.markdown("Your personal toolkit for options analysis, sentiment, and market structure. Select a tool below or use the sidebar.")

st.divider()

st.subheader("⭐ Featured")

col_mp1, col_mp2 = st.columns(2)

with col_mp1:
    st.page_link(
        "pages/Master_Plan_Stocks.py",
        label="Master Plan — Stocks & ETFs",
        icon="🗺️",
        width='stretch'
    )
with col_mp2:
    st.page_link(
        "pages/Master_Plan_Index.py",
        label="Master Plan — Index",
        icon="🗺️",
        width='stretch'
    )
st.caption("Project price along its trend, then find the optimum call strike to buy for every expiration based on the projected price. Combines the Chart Pattern Analyzer and Options Data Explorer.")

st.divider()

st.subheader("Core Tools")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.page_link(
        "pages/Options_Data_Explorer_Stocks.py",
        label="Options Data Explorer",
        icon="📈",
        width='stretch'
    )
    st.caption("Find best LEAPS and future options under different annual growth assumptions (5%, 10%, 15%).")

with row1_col2:
    st.page_link(
        "pages/Options_Data_Explorer_Index.py",
        label="Options Data Explorer — Index",
        icon="📈",
        width='stretch'
    )
    st.caption("Same LEAPS/future-return workflow, tuned for index options (SPX, NDX, OEX) — falls back to last-trade price on illiquid strikes with no live bid/ask.")

with row2_col1:
    st.page_link(
        "pages/Buy_vs_Sell_Pressure.py",
        label="Buy vs Sell Pressure",
        icon="📊",
        width='stretch'
    )
    st.caption("Gauge bullish/bearish conviction via OBV, PVT, MACD, options activity, and overall sentiment score.")

with row2_col2:
    st.page_link(
        "pages/Chart_Pattern_Analyzer.py",
        label="Chart Pattern Analyzer",
        icon="📐",
        width='stretch'
    )
    st.caption("Project future price targets using linear regression, moving averages, and other trend overlays.")

st.divider()
st.subheader("Market Analysis")

st.page_link(
    "pages/Capital_Reallocation.py",
    label="Capital Reallocation",
    icon="🔄",
    width='stretch'
)
st.caption("Track where money is flowing across asset classes and sectors. Good for reading broad market rotation.")

st.divider()
st.subheader("Spread Tools")

with st.expander("Less frequently used", expanded=False):
    col6, col7 = st.columns(2)
    with col6:
        st.page_link(
            "pages/Vertical_Call_Spread.py",
            label="Vertical Call Spread Finder",
            icon="📈"
        )
        st.caption("Model bullish vertical call spreads for event-driven moves like earnings.")
    with col7:
        st.page_link(
            "pages/Vertical_Put_Spread.py",
            label="Vertical Put Spread Finder",
            icon="📉"
        )
        st.caption("Model bearish vertical put spreads for event-driven moves like earnings.")