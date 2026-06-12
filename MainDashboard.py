# MainDashboard.py
import streamlit as st

st.set_page_config(page_title="Options Analysis Hub", layout="wide")

st.title("🧭 Options Analysis Hub")

st.markdown("Your personal toolkit for options analysis, sentiment, and market structure. Select a tool below or use the sidebar.")

st.divider()

st.subheader("Core Tools")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link(
        "pages/Options_Data_Explorer.py",
        label="Options Data Explorer",
        icon="📈",
        use_container_width=True
    )
    st.caption("Find best LEAPS and future options under different annual growth assumptions (5%, 10%, 15%).")

with col2:
    st.page_link(
        "pages/Buy_vs_Sell_Pressure.py",
        label="Buy vs Sell Pressure",
        icon="📊",
        use_container_width=True
    )
    st.caption("Gauge bullish/bearish conviction via OBV, PVT, MACD, options activity, and overall sentiment score.")

with col3:
    st.page_link(
        "pages/Chart_Pattern_Analyzer.py",
        label="Chart Pattern Analyzer",
        icon="📐",
        use_container_width=True
    )
    st.caption("Project future price targets using linear regression, moving averages, and other trend overlays.")

st.divider()
st.subheader("Market Analysis")

st.page_link(
    "pages/Capital_Reallocation.py",
    label="Capital Reallocation",
    icon="🔄",
    use_container_width=True
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