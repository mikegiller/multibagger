import streamlit as st
import yfinance as yf
import utils  # applies the yfinance cache workaround on import
import chart_utils  # shared chart/projection helpers
import pandas as pd
import numpy as np
from chart_utils import (
    find_swings, calc_linreg, calc_trendlines, calc_moving_averages,
    calc_fibonacci, calc_bollinger, calc_sr, build_chart,
    build_projection_table, render_projection_table as _render_projection_table,
    pct_label as _pct_label,
)

st.set_page_config(page_title="Chart Pattern Analyzer", layout="wide")
st.title("Chart Pattern Analyzer")
st.info("**Purpose:** Project future price targets based on existing trends. Applies technical overlays — linear regression channels, moving averages, and more — to identify where price is likely heading.")

# === Favorites ===
_favs = utils.load_favorites()

# === Session State ===
_defaults = {
    "cpa_period": "1y",
    "cpa_horizon": "3 Months",
    "cpa_lt_period": "2y",
    "cpa_lt_horizon": "1 Year",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "fav" in st.query_params and st.query_params["fav"] in _favs:
    st.session_state.cpa_ticker = st.query_params["fav"]
    utils.set_active_ticker(st.query_params["fav"])
    del st.query_params["fav"]

# === Sidebar — Clear Cache only ===
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

# === Shared Ticker Input ===
_fav_placeholder = st.empty()
ticker = utils.ticker_input("cpa_ticker", max_chars=12)
if not ticker:
    st.stop()

_fav_placeholder.markdown(utils.favorites_html(_favs), unsafe_allow_html=True)


# === Period % Change Helpers ===
@st.cache_data(ttl=900, show_spinner=False)
def get_period_changes(tkr):
    changes = {}
    try:
        hist = yf.Ticker(tkr).history(period="5y", interval="1d")
        if not hist.empty:
            last = float(hist.Close.iloc[-1])
            for p, bars in [("1mo", 21), ("3mo", 63), ("6mo", 126),
                             ("1y", 252), ("2y", 504), ("5y", len(hist))]:
                idx = min(bars, len(hist))
                if idx > 0:
                    start = float(hist.Close.iloc[-idx])
                    if start > 0:
                        changes[p] = (last - start) / start * 100
    except Exception:
        pass
    return changes


@st.cache_data(ttl=900, show_spinner=False)
def get_lt_period_changes(tkr):
    changes = {}
    try:
        hist = yf.Ticker(tkr).history(period="max", interval="1wk")
        if not hist.empty:
            last = float(hist.Close.iloc[-1])
            for p, bars in [("2y", 104), ("3y", 156), ("5y", 260),
                             ("10y", 520), ("max", len(hist))]:
                idx = min(bars, len(hist))
                if idx > 0:
                    start = float(hist.Close.iloc[-idx])
                    if start > 0:
                        changes[p] = (last - start) / start * 100
    except Exception:
        pass
    return changes


period_changes    = get_period_changes(ticker)
lt_period_changes = get_lt_period_changes(ticker)


# === Data Fetch ===
@st.cache_data(ttl=900, show_spinner="Fetching daily data...")
def fetch_data(tkr, per):
    df = yf.Ticker(tkr).history(period=per, interval="1d", auto_adjust=True)
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


@st.cache_data(ttl=900, show_spinner="Fetching weekly data...")
def fetch_data_lt(tkr, per):
    df = yf.Ticker(tkr).history(period=per, interval="1wk", auto_adjust=True)
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()



# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_st, tab_lt = st.tabs(["Short Term (Daily)", "Long Term (Weekly / LEAPS)"])


# ─── Short Term Tab ──────────────────────────────────────────────────────────
with tab_st:
    # --- History Period ---
    ST_PERIOD_OPTS = {
        "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
        "1y": "1Y", "2y": "2Y", "5y": "5Y",
    }
    st.write("**Select history period:**")
    _st_per_cols = st.columns(len(ST_PERIOD_OPTS))
    for i, (k, lbl) in enumerate(ST_PERIOD_OPTS.items()):
        with _st_per_cols[i]:
            st.button(lbl, key=f"cpa_stper_{k}", width='stretch',
                      on_click=lambda _k=k: st.session_state.update({"cpa_period": _k}))
            pct = period_changes.get(k)
            if pct is not None:
                st.markdown(_pct_label(pct), unsafe_allow_html=True)
    period = st.session_state.cpa_period

    # --- Projection Horizon ---
    ST_HORIZON_OPTS = {"1 Month": 21, "3 Months": 65, "6 Months": 130}
    st.write("**Select projection horizon:**")
    _st_hor_cols = st.columns(len(ST_HORIZON_OPTS))
    for i, (k, bars) in enumerate(ST_HORIZON_OPTS.items()):
        with _st_hor_cols[i]:
            st.button(k, key=f"cpa_sthor_{k.replace(' ','_')}", width='stretch',
                      on_click=lambda _k=k: st.session_state.update({"cpa_horizon": _k}))
    horizon_label = st.session_state.cpa_horizon
    horizon_bars  = ST_HORIZON_OPTS[horizon_label]

    st.info(f"Showing: **{ST_PERIOD_OPTS.get(period, period)}** history  |  **{horizon_label}** projection", icon="📊")

    # --- Data ---
    df = fetch_data(ticker, period)
    if df is None or df.empty:
        st.error(f"No daily data found for **{ticker}**.")
    else:
        df["_bar"] = np.arange(len(df))
        current_price = float(df["Close"].iloc[-1])
        n_bars = len(df)
        future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon_bars)
        future_bar_idx = np.arange(n_bars, n_bars + horizon_bars)
        swing_highs, swing_lows = find_swings(df, window=5)

        # Overlay checkboxes
        st.markdown("**Overlays**")
        _oc = st.columns(7)
        with _oc[0]:
            st.caption("Linear Regression")
            show_lr_center = st.checkbox("LR Center",    value=True,  key="cpa_lr_center")
            show_lr_1std   = st.checkbox("LR ±1σ",       value=False, key="cpa_lr_1std")
            show_lr_2std   = st.checkbox("LR ±2σ",       value=False, key="cpa_lr_2std")
        with _oc[1]:
            st.caption("Trend Lines")
            show_tl_resist  = st.checkbox("Resistance",  value=False, key="cpa_tl_resist")
            show_tl_support = st.checkbox("Support",     value=False, key="cpa_tl_support")
        with _oc[2]:
            st.caption("Moving Averages")
            ma_20  = st.checkbox("SMA 20",  value=False, key="cpa_ma20")
            ma_50  = st.checkbox("SMA 50",  value=False, key="cpa_ma50")
            ma_100 = st.checkbox("SMA 100", value=False, key="cpa_ma100")
            ma_200 = st.checkbox("SMA 200", value=False, key="cpa_ma200")
        with _oc[3]:
            st.caption("Fibonacci")
            show_fib = st.checkbox("Fib Retracement", value=False, key="cpa_fib")
        with _oc[4]:
            st.caption("Bollinger (20)")
            show_bb_bands  = st.checkbox("BB Upper/Lower", value=False, key="cpa_bb_bands")
            show_bb_center = st.checkbox("BB Center",      value=False, key="cpa_bb_center")
        with _oc[5]:
            st.caption("S/R Horizontal")
            show_sr_resist  = st.checkbox("S/R Resistance", value=False, key="cpa_sr_resist")
            show_sr_support = st.checkbox("S/R Support",    value=False, key="cpa_sr_support")
        with _oc[6]:
            st.write("")

        st.markdown("---")

        show = {
            "lr_center": show_lr_center, "lr_1std": show_lr_1std, "lr_2std": show_lr_2std,
            "tl_resist": show_tl_resist, "tl_support": show_tl_support,
            "bb_bands": show_bb_bands, "bb_center": show_bb_center,
            "sr_resist": show_sr_resist, "sr_support": show_sr_support,
        }

        ma_periods  = [p for p, on in [(20, ma_20), (50, ma_50), (100, ma_100), (200, ma_200)] if on]
        linreg_data = calc_linreg(df, future_bar_idx) if (show_lr_center or show_lr_1std or show_lr_2std) else None
        tl_data     = calc_trendlines(df, swing_highs, swing_lows, future_dates, future_bar_idx) if (show_tl_resist or show_tl_support) else None
        ma_data     = calc_moving_averages(df, future_dates, future_bar_idx, ma_periods) if ma_periods else None
        fib_data    = calc_fibonacci(swing_highs, swing_lows) if show_fib else None
        bb_data     = calc_bollinger(df, future_dates, future_bar_idx, period=20) if (show_bb_bands or show_bb_center) else None
        sr_data     = calc_sr(swing_highs, swing_lows) if (show_sr_resist or show_sr_support) else None

        fig = build_chart(
            df, current_price, future_dates, future_bar_idx,
            linreg_data, tl_data, ma_data, fib_data, bb_data, sr_data, show,
            ticker, title_suffix=" (Daily)",
        )
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        ST_TARGETS = {"+1 Week": 5, "+2 Weeks": 10, "+1 Month": 21, "+3 Months": 65}
        proj_df = build_projection_table(
            current_price, horizon_bars, linreg_data, tl_data, ma_data, fib_data, bb_data, sr_data,
            show, ST_TARGETS, vs_col="+1 Month",
        )
        if not proj_df.empty:
            st.subheader(f"Price Projection Table — {ticker} (Daily)")
            st.caption(
                f"Projections from {df.index[-1].strftime('%b %d, %Y')} "
                f"over {horizon_label} horizon  |  "
                f"Rising/Falling = slope >±$0.05/day  |  vs Current uses +1 Month"
            )
            _render_projection_table(proj_df, "+1 Month")


# ─── Long Term Tab ───────────────────────────────────────────────────────────
with tab_lt:
    # --- History Period ---
    LT_PERIOD_OPTS = {
        "2y": "2Y", "3y": "3Y", "5y": "5Y", "10y": "10Y", "max": "Max",
    }
    st.write("**Select history period:**")
    _lt_per_cols = st.columns(len(LT_PERIOD_OPTS))
    for i, (k, lbl) in enumerate(LT_PERIOD_OPTS.items()):
        with _lt_per_cols[i]:
            st.button(lbl, key=f"cpa_ltper_{k}", width='stretch',
                      on_click=lambda _k=k: st.session_state.update({"cpa_lt_period": _k}))
            pct = lt_period_changes.get(k)
            if pct is not None:
                st.markdown(_pct_label(pct), unsafe_allow_html=True)
    lt_period = st.session_state.cpa_lt_period

    # --- Projection Horizon ---
    LT_HORIZON_OPTS = {"6 Months": 26, "1 Year": 52, "2 Years": 104}
    st.write("**Select projection horizon:**")
    _lt_hor_cols = st.columns(len(LT_HORIZON_OPTS))
    for i, (k, bars) in enumerate(LT_HORIZON_OPTS.items()):
        with _lt_hor_cols[i]:
            st.button(k, key=f"cpa_lthor_{k.replace(' ','_')}", width='stretch',
                      on_click=lambda _k=k: st.session_state.update({"cpa_lt_horizon": _k}))
    lt_horizon_label = st.session_state.cpa_lt_horizon
    lt_horizon_bars  = LT_HORIZON_OPTS[lt_horizon_label]

    st.info(f"Showing: **{LT_PERIOD_OPTS.get(lt_period, lt_period)}** history  |  **{lt_horizon_label}** projection", icon="📊")

    # --- Data ---
    df_lt = fetch_data_lt(ticker, lt_period)
    if df_lt is None or df_lt.empty:
        st.error(f"No weekly data found for **{ticker}**.")
    else:
        df_lt["_bar"] = np.arange(len(df_lt))
        current_price_lt = float(df_lt["Close"].iloc[-1])
        n_bars_lt = len(df_lt)
        future_dates_lt = pd.date_range(
            start=df_lt.index[-1] + pd.Timedelta(weeks=1),
            periods=lt_horizon_bars, freq="W-MON",
        )
        future_bar_idx_lt = np.arange(n_bars_lt, n_bars_lt + lt_horizon_bars)
        swing_highs_lt, swing_lows_lt = find_swings(df_lt, window=4)

        # Overlay checkboxes
        st.markdown("**Overlays**")
        _oc_lt = st.columns(7)
        with _oc_lt[0]:
            st.caption("Linear Regression")
            lt_lr_center = st.checkbox("LR Center",    value=True,  key="cpa_lt_lr_center")
            lt_lr_1std   = st.checkbox("LR ±1σ",       value=False, key="cpa_lt_lr_1std")
            lt_lr_2std   = st.checkbox("LR ±2σ",       value=False, key="cpa_lt_lr_2std")
        with _oc_lt[1]:
            st.caption("Trend Lines")
            lt_tl_resist  = st.checkbox("Resistance",  value=False, key="cpa_lt_tl_resist")
            lt_tl_support = st.checkbox("Support",     value=False, key="cpa_lt_tl_support")
        with _oc_lt[2]:
            st.caption("Weekly MAs")
            lt_ma13  = st.checkbox("SMA 13w  (~1Q)",  value=False, key="cpa_lt_ma13")
            lt_ma26  = st.checkbox("SMA 26w  (~6M)",  value=False, key="cpa_lt_ma26")
            lt_ma52  = st.checkbox("SMA 52w  (~1Y)",  value=False, key="cpa_lt_ma52")
            lt_ma104 = st.checkbox("SMA 104w (~2Y)",  value=False, key="cpa_lt_ma104")
        with _oc_lt[3]:
            st.caption("Fibonacci")
            lt_fib = st.checkbox("Fib Retracement", value=False, key="cpa_lt_fib")
        with _oc_lt[4]:
            st.caption("Bollinger (26w)")
            lt_bb_bands  = st.checkbox("BB Upper/Lower", value=False, key="cpa_lt_bb_bands")
            lt_bb_center = st.checkbox("BB Center",      value=False, key="cpa_lt_bb_center")
        with _oc_lt[5]:
            st.caption("S/R Horizontal")
            lt_sr_resist  = st.checkbox("S/R Resistance", value=False, key="cpa_lt_sr_resist")
            lt_sr_support = st.checkbox("S/R Support",    value=False, key="cpa_lt_sr_support")
        with _oc_lt[6]:
            st.write("")

        st.markdown("---")

        show_lt = {
            "lr_center": lt_lr_center, "lr_1std": lt_lr_1std, "lr_2std": lt_lr_2std,
            "tl_resist": lt_tl_resist, "tl_support": lt_tl_support,
            "bb_bands": lt_bb_bands, "bb_center": lt_bb_center,
            "sr_resist": lt_sr_resist, "sr_support": lt_sr_support,
        }

        lt_ma_periods = [p for p, on in [(13, lt_ma13), (26, lt_ma26), (52, lt_ma52), (104, lt_ma104)] if on]
        lt_linreg = calc_linreg(df_lt, future_bar_idx_lt) if (lt_lr_center or lt_lr_1std or lt_lr_2std) else None
        lt_tl     = calc_trendlines(df_lt, swing_highs_lt, swing_lows_lt, future_dates_lt, future_bar_idx_lt) if (lt_tl_resist or lt_tl_support) else None
        lt_ma     = calc_moving_averages(df_lt, future_dates_lt, future_bar_idx_lt, lt_ma_periods) if lt_ma_periods else None
        lt_fib_d  = calc_fibonacci(swing_highs_lt, swing_lows_lt) if lt_fib else None
        lt_bb     = calc_bollinger(df_lt, future_dates_lt, future_bar_idx_lt, period=26) if (lt_bb_bands or lt_bb_center) else None
        lt_sr     = calc_sr(swing_highs_lt, swing_lows_lt) if (lt_sr_resist or lt_sr_support) else None

        fig_lt = build_chart(
            df_lt, current_price_lt, future_dates_lt, future_bar_idx_lt,
            lt_linreg, lt_tl, lt_ma, lt_fib_d, lt_bb, lt_sr, show_lt,
            ticker, title_suffix=" (Weekly)",
            is_weekly=True,
        )
        st.plotly_chart(fig_lt, width='stretch', config={"displayModeBar": False})

        LT_TARGETS = {"+3 Months": 13, "+6 Months": 26, "+1 Year": 52, "+2 Years": 104}
        proj_df_lt = build_projection_table(
            current_price_lt, lt_horizon_bars, lt_linreg, lt_tl, lt_ma, lt_fib_d, lt_bb, lt_sr,
            show_lt, LT_TARGETS, vs_col="+1 Year", is_weekly=True,
        )
        if not proj_df_lt.empty:
            st.subheader(f"Price Projection Table — {ticker} (Weekly / LEAPS)")
            st.caption(
                f"Weekly projections from {df_lt.index[-1].strftime('%b %d, %Y')} "
                f"over {lt_horizon_label} horizon  |  "
                f"Rising/Falling = slope >±$0.01/week  |  vs Current uses +1 Year"
            )
            _render_projection_table(proj_df_lt, "+1 Year")
