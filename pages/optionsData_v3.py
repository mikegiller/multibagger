# app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule

# --- Page settings ---
st.set_page_config(page_title="Call Options Viewer", layout="wide")
st.title("📈 Call Options Viewer")
st.write("Enter a ticker to view call options and optionally download as Excel.")

# --- Session state ---
for key in ["ticker_input", "expiry", "calls_data", "summary_data", "column_names"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Reset button ---
if st.button("🔄 Reset"):
    for key in ["ticker_input", "expiry", "calls_data", "summary_data", "column_names"]:
        st.session_state[key] = None

# --- Ticker input ---
ticker_input = st.text_input("Enter ticker symbol (required):", value=st.session_state.ticker_input or "").upper()
st.session_state.ticker_input = ticker_input

# --- Fetch expirations ---
expirations = []
if ticker_input:
    try:
        ticker = yf.Ticker(ticker_input)
        hist = ticker.history(period="1d")
        if hist.empty:
            st.error("No data found for ticker.")
        elif not ticker.options:
            st.error("No options available for this ticker.")
        else:
            expirations = ticker.options
    except Exception as e:
        st.error(f"Error fetching ticker info: {e}")

# --- Expiration dropdown ---
expiry = None
if expirations:
    # Convert tuple to list and prepend placeholder
    expiry_options = ["Select expiration date..."] + list(expirations)
    selected_expiry = st.selectbox(
        "Select expiration date:",
        expiry_options,
        index=0 if st.session_state.expiry is None else expiry_options.index(st.session_state.expiry)
    )

    # Only use a real expiry if the user has selected it
    expiry = selected_expiry if selected_expiry != "Select expiration date..." else None
    st.session_state.expiry = expiry


# --- Data Refresh button ---
refresh_pressed = st.button("🔄 Data Refresh")

# --- Function to fetch option data ---
def fetch_option_data(ticker_input, expiry):
    ticker = yf.Ticker(ticker_input)
    hist = ticker.history(period="1d")
    last_price = round(hist["Close"].iloc[-1], 2)
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
    today = datetime.today()
    days_to_expiry = (expiry_date - today).days
    years_to_expiry = max(round(days_to_expiry / 365, 2), 1)

    # Summary
    summary_df = pd.DataFrame({
        "Ticker": [ticker_input],
        "Last Price": [last_price],
        "Expiration": [expiry],
        "Days to Expiry": [days_to_expiry],
        "Years to Expiry": [years_to_expiry]
    })

    # Calls
    calls = ticker.option_chain(expiry).calls
    cutoff_date = today - timedelta(days=30)
    if "lastTradeDate" in calls.columns:
        calls["lastTradeDate"] = calls["lastTradeDate"].dt.tz_localize(None)
        calls = calls[calls["lastTradeDate"] >= cutoff_date]

    if calls.empty:
        st.warning("No call options traded in the last 30 days for this expiration.")
        return summary_df, None, []

    # Keep necessary columns
    cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility", "inTheMoney"]
    calls = calls[cols].copy()
    calls["middle"] = ((calls["bid"] + calls["ask"]) / 2).round(2)
    calls["impliedVolatility"] = (calls["impliedVolatility"]*100).round(2)

    # Dynamic percentage columns
    percentages = [5, 10, 15, 20, 25, 30, 35, 40]
    column_names = []
    for pct in percentages:
        target = round(last_price * ((1 + pct/100) ** years_to_expiry), 2)
        col_name = f"{ticker_input}: {pct}% - {target}"
        column_names.append(col_name)
        calls[col_name] = ((target - (calls["strike"] + calls["middle"])) / calls["middle"]) * 100
        calls[col_name] = calls[col_name].round(2).astype(str) + "%"

    # Reorder columns: strike, middle, volume to left
    front_cols = ["strike", "middle", "volume"]
    other_cols = [c for c in calls.columns if c not in front_cols]
    calls = calls[front_cols + other_cols]

    return summary_df, calls, column_names

# --- Fetch data ---
if ticker_input and expiry and (refresh_pressed or st.session_state.calls_data is None):
    summary_df, calls, column_names = fetch_option_data(ticker_input, expiry)
    st.session_state.calls_data = calls
    st.session_state.summary_data = summary_df
    st.session_state.column_names = column_names
else:
    calls = st.session_state.calls_data
    summary_df = st.session_state.summary_data
    column_names = st.session_state.column_names or []

# --- Show summary ---
if summary_df is not None:
    st.subheader("Summary")
    st.dataframe(summary_df.round(2))

# --- Show call options in AgGrid ---
if calls is not None and not calls.empty:
    st.subheader("Call Options")

    # Compute max per percentage column
    pct_max = {}
    for col in column_names:
        if col in calls.columns:
            pct_max[col] = calls[col].str.rstrip('%').astype(float).max()

    # Build AgGrid
    gb = GridOptionsBuilder.from_dataframe(calls)

    # Freeze and bold left columns
    for col in ["strike", "middle", "volume"]:
        gb.configure_column(col, pinned='left', cellStyle={'fontWeight': 'bold'})

    # Conditional formatting for percentage columns
    for col in column_names:
        if col in calls.columns:
            # Compute max safely
            col_numeric = calls[col].str.rstrip('%').astype(float)
            max_val = col_numeric.replace([float('inf'), float('-inf')], pd.NA).max()
            if pd.isna(max_val):
                max_val = 0  # fallback to 0 if no valid max

            js_code = JsCode(f"""
            function(params) {{
                if (!params.value) return {{}};
                let val = parseFloat(params.value.replace('%',''));
                if (isNaN(val)) return {{}};
                if (val < 0) {{
                    return {{'color':'white','backgroundColor':'#5DADE2'}};
                }} else if (val === {max_val}) {{
                    return {{'color':'black','backgroundColor':'#F39C12'}};
                }} else {{
                    return {{}};
                }}
            }}
            """)
            gb.configure_column(col, cellStyle=js_code)


    grid_options = gb.build()

    AgGrid(
        calls,
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        height=600,
        width='100%',
        allow_unsafe_jscode=True
    )

    # --- Excel download with formatting ---
    # Define fills
    blue_fill = PatternFill(start_color="5DADE2", end_color="5DADE2", fill_type="solid")
    orange_fill = PatternFill(start_color="F39C12", end_color="F39C12", fill_type="solid")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Calls", index=False, startrow=0)
        calls_excel = calls.copy()
        for col in column_names:
            if col in calls_excel.columns:
                calls_excel[col] = calls_excel[col].str.rstrip('%').astype(float)
        calls_excel.to_excel(writer, sheet_name="Calls", index=False, startrow=6)
    output.seek(0)

    # Load workbook to apply formatting
    wb = load_workbook(output)
    ws = wb["Calls"]

    # Freeze left columns A:C
    ws.freeze_panes = ws['D7']

    # Header formatting
    for cell in ws[7]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    # Auto-fit columns
    for col in ws.columns:
        max_length = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value is not None:
                max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max_length + 2

    # Conditional formatting
    header_values = [cell.value for cell in ws[7]]
    for col_name in column_names:
        if col_name not in header_values:
            continue
        col_idx = header_values.index(col_name) + 1
        # Negative = blue
        ws.conditional_formatting.add(
            f"{get_column_letter(col_idx)}8:{get_column_letter(col_idx)}{ws.max_row}",
            CellIsRule(operator='lessThan', formula=['0'], stopIfTrue=True, fill=blue_fill)
        )
        # Max = orange
        col_cells = [ws.cell(row=row, column=col_idx).value for row in range(8, ws.max_row + 1)]
        numeric_vals = [v for v in col_cells if isinstance(v, (int, float))]
        if numeric_vals:
            max_val = max(numeric_vals)
            ws.conditional_formatting.add(
                f"{get_column_letter(col_idx)}8:{get_column_letter(col_idx)}{ws.max_row}",
                CellIsRule(operator='equal', formula=[str(max_val)], stopIfTrue=True, fill=orange_fill)
            )

    # Save to BytesIO
    output2 = BytesIO()
    wb.save(output2)
    output2.seek(0)

    # Streamlit download button
    st.download_button(
        label="📥 Download Excel (formatted)",
        data=output2,
        file_name=f"{ticker_input}_CallOptions_{expiry}_formatted.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
