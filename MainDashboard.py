# MainDashboard.py
import streamlit as st

st.set_page_config(page_title="Options Analysis Hub", layout="wide")

st.title("🧭 Options Analysis Hub")

st.markdown("""
Welcome to your options trading toolkit!  
Select a tool below to analyze spreads or view option data.
""")

st.divider()

st.subheader("Available Tools")

# Main prominent tools
col1, col2 = st.columns(2)

with col1:
    st.page_link(
        "pages/optionsData_v3.py", 
        label="Options Data Explorer", 
        icon="📈",
        use_container_width=True
    )

with col2:
    st.page_link(
        "pages/Buy_vs_Sell.py", 
        label="Buy vs Sell Pressure", 
        icon="📊",
        use_container_width=True
    )

# Collapsed section for other tools
with st.expander("Other ▼", expanded=False):
    st.page_link(
        "pages/VerticalCallSpread.py", 
        label="Vertical Call Spread Finder", 
        icon="📈"
    )
    
    st.page_link(
        "pages/VerticalPutSpread.py", 
        label="Vertical Put Spread Finder", 
        icon="📉"
    )

st.divider()
st.info("Use the navigation links above or the sidebar to launch each specialized app.")