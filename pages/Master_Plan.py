import streamlit as st
import yfinance as yf
import utils  # applies the yfinance cache workaround on import
import chart_utils  # shared chart/projection helpers
import pandas as pd
import numpy as np
from datetime import datetime
from chart_utils import (
    find_swings, calc_linreg, calc_trendlines, calc_moving_averages,
    calc_fibonacci, calc_bollinger, calc_sr, build_chart,
    build_projection_table, render_projection_table, pct_label,
)

TODAY = pd.Timestamp(datetime.now().date())

# Only trust a contract's mid as the "optimum strike" if it traded within the
# last N trading days — option mids go stale within days, and the optimum pick
# (max return %) otherwise gravitates to cheap, stale, illiquid strikes.
RECENCY_TRADING_DAYS = 2

st.set_page_config(page_title="Master Plan", layout="wide")
st.title("Master Plan")
st.info(
    "**Purpose:** Combine the trend projection with the options chain. The chart "
    "projects price along its linear-regression trend; for every available expiration "
    "the table then finds the call strike that maximizes return *assuming the stock "
    "lands on its projected price by that date* — the same return calc as the "
    "yellow-highlighted cell in Options Data Explorer."
)

# === Favorites ===
_favs = utils.load_favorites()

# === Session State ===
_defaults = {
    "mp_period": "1y",
    "mp_horizon": "3 Months",
    "mp_lt_period": "2y",
    "mp_lt_horizon": "1 Year",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "fav" in st.query_params and st.query_params["fav"] in _favs:
    st.session_state.mp_ticker = st.query_params["fav"]
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
ticker = utils.ticker_input("mp_ticker", max_chars=12)
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


# === Options Chains ===
@st.cache_data(ttl=600, show_spinner="Fetching option chains for all expirations...")
def fetch_all_call_chains(tkr):
    """{expiration_str: trimmed calls DataFrame} for every available expiration."""
    t = yf.Ticker(tkr)
    out = {}
    try:
        exps = t.options
    except Exception:
        return out
    for exp in exps:
        try:
            calls = t.option_chain(exp).calls
        except Exception:
            continue
        if calls is None or calls.empty:
            continue
        keep = [c for c in ["strike", "bid", "ask", "lastPrice", "volume",
                            "openInterest", "lastTradeDate"] if c in calls.columns]
        out[exp] = calls[keep].copy()
    return out


# === Master Plan Table ===
def build_master_plan_table(chains, last_date, linreg, is_weekly, active_only=True):
    """For each expiration, find the call strike that maximizes projected return,
    where the target price is the LinReg-center value extrapolated to that date.

    Return % uses the Options Data Explorer formula:
        ((target - (strike + middle)) / middle) * 100
    """
    slope = linreg["slope"]
    last_center = linreg["last_center"]
    cutoff = (TODAY - pd.tseries.offsets.BDay(RECENCY_TRADING_DAYS)).normalize()
    rows = []

    for exp_str, calls in chains.items():
        exp_date = pd.Timestamp(exp_str)
        if exp_date <= last_date:
            continue  # already expired relative to the data window

        # Bars from the last historical bar to the expiration, in the chart's unit
        if is_weekly:
            offset = (exp_date - last_date).days / 7.0
        else:
            offset = float(np.busday_count(last_date.date(), exp_date.date()))
        if offset <= 0:
            continue

        target = last_center + slope * offset

        c = calls.copy()
        if "lastTradeDate" in c.columns:
            ltd = pd.to_datetime(c["lastTradeDate"], errors="coerce")
            if getattr(ltd.dtype, "tz", None) is not None:
                ltd = ltd.dt.tz_localize(None)
            c = c[ltd >= cutoff]
        if active_only and {"volume", "openInterest"}.issubset(c.columns):
            c = c[(c["volume"].fillna(0) > 0) | (c["openInterest"].fillna(0) > 10)]
        # Require a genuine two-sided quote — a one-sided phantom (bid=0) yields
        # a tiny mid and a bogus, untradeable "optimum" return.
        c = c[(c["bid"] > 0) & (c["ask"] > 0)]
        c["middle"] = ((c["bid"] + c["ask"]) / 2).round(2)
        c = c[c["middle"] > 0]
        if c.empty:
            continue

        c["ret"] = ((target - (c["strike"] + c["middle"])) / c["middle"]) * 100
        best = c.loc[c["ret"].idxmax()]

        rows.append({
            "Expiration Date": exp_str,
            "Days to Exp": int((exp_date - TODAY).days),
            "Projected Price": round(float(target), 2),
            "Optimum Strike": float(best["strike"]),
            "Optimum Middle Cost": round(float(best["middle"]), 2),
            "Projected Return": round(float(best["ret"]), 2),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Expiration Date").reset_index(drop=True)


def render_master_plan(chains, last_date, linreg, current_price, is_weekly, active_only):
    if linreg is None:
        st.warning("Not enough price history to fit a regression trend for projections.")
        return
    if not chains:
        st.warning(f"No options chains available for **{ticker}**.")
        return

    mp_df = build_master_plan_table(chains, last_date, linreg, is_weekly, active_only)
    if mp_df.empty:
        st.warning(
            f"No call contracts traded within the last {RECENCY_TRADING_DAYS} trading "
            "days with a two-sided quote. Illiquid names (or after a long market "
            "closure) may have no fresh quotes — try unchecking 'Active only'."
        )
        return

    # --- Recommendation: best projected return row ---
    best = mp_df.loc[mp_df["Projected Return"].idxmax()]
    trend = "rising" if linreg["slope"] > 0 else "falling"
    st.success(
        f"**Recommendation — {ticker}:** Buy the **${best['Optimum Strike']:,.2f}** call "
        f"expiring **{best['Expiration Date']}** at ~**${best['Optimum Middle Cost']:,.2f}** mid. "
        f"If the {trend} trend holds, price projects to **${best['Projected Price']:,.2f}** by then "
        f"→ **{best['Projected Return']:+.1f}%** projected return.",
        icon="🎯",
    )

    # --- Full expiration table, best row highlighted ---
    best_exp = best["Expiration Date"]

    def _highlight_best(row):
        hit = row["Expiration Date"] == best_exp
        return ["background-color: #F39C12; color: black" if hit else "" for _ in row]

    def _style_return(v):
        return "color: #00c853" if v > 0 else ("color: #d50000" if v < 0 else "")

    styled = (
        mp_df.style
        .apply(_highlight_best, axis=1)
        .map(_style_return, subset=["Projected Return"])
        .format({
            "Projected Price": "${:,.2f}",
            "Optimum Strike": "${:,.2f}",
            "Optimum Middle Cost": "${:,.2f}",
            "Projected Return": "{:+.2f}%",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "Projected Price = LinReg-center trend extrapolated to each expiration date. "
        "Optimum Strike = the call maximizing "
        "((Projected Price − (Strike + Mid)) / Mid) on that expiration's chain, "
        f"limited to contracts traded within the last {RECENCY_TRADING_DAYS} trading "
        "days with a two-sided quote. Highlighted row = highest projected return."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_st, tab_lt = st.tabs(["Short Term (Daily)", "Long Term (Weekly / LEAPS)"])

call_chains = fetch_all_call_chains(ticker)


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
            st.button(lbl, key=f"mp_stper_{k}", use_container_width=True,
                      on_click=lambda _k=k: st.session_state.update({"mp_period": _k}))
            pct = period_changes.get(k)
            if pct is not None:
                st.markdown(pct_label(pct), unsafe_allow_html=True)
    period = st.session_state.mp_period

    # --- Projection Horizon ---
    ST_HORIZON_OPTS = {"1 Month": 21, "3 Months": 65, "6 Months": 130}
    st.write("**Select projection horizon:**")
    _st_hor_cols = st.columns(len(ST_HORIZON_OPTS))
    for i, (k, bars) in enumerate(ST_HORIZON_OPTS.items()):
        with _st_hor_cols[i]:
            st.button(k, key=f"mp_sthor_{k.replace(' ','_')}", use_container_width=True,
                      on_click=lambda _k=k: st.session_state.update({"mp_horizon": _k}))
    horizon_label = st.session_state.mp_horizon
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
            show_lr_center = st.checkbox("LR Center",    value=True,  key="mp_lr_center")
            show_lr_1std   = st.checkbox("LR ±1σ",       value=False, key="mp_lr_1std")
            show_lr_2std   = st.checkbox("LR ±2σ",       value=False, key="mp_lr_2std")
        with _oc[1]:
            st.caption("Trend Lines")
            show_tl_resist  = st.checkbox("Resistance",  value=False, key="mp_tl_resist")
            show_tl_support = st.checkbox("Support",     value=False, key="mp_tl_support")
        with _oc[2]:
            st.caption("Moving Averages")
            ma_20  = st.checkbox("SMA 20",  value=False, key="mp_ma20")
            ma_50  = st.checkbox("SMA 50",  value=False, key="mp_ma50")
            ma_100 = st.checkbox("SMA 100", value=False, key="mp_ma100")
            ma_200 = st.checkbox("SMA 200", value=False, key="mp_ma200")
        with _oc[3]:
            st.caption("Fibonacci")
            show_fib = st.checkbox("Fib Retracement", value=False, key="mp_fib")
        with _oc[4]:
            st.caption("Bollinger (20)")
            show_bb_bands  = st.checkbox("BB Upper/Lower", value=False, key="mp_bb_bands")
            show_bb_center = st.checkbox("BB Center",      value=False, key="mp_bb_center")
        with _oc[5]:
            st.caption("S/R Horizontal")
            show_sr_resist  = st.checkbox("S/R Resistance", value=False, key="mp_sr_resist")
            show_sr_support = st.checkbox("S/R Support",    value=False, key="mp_sr_support")
        with _oc[6]:
            st.caption("Options Filter")
            active_only = st.checkbox("Active only", value=True, key="mp_active_only",
                                      help="Volume > 0 or OI > 10, traded in last 30 days")

        st.markdown("---")

        show = {
            "lr_center": show_lr_center, "lr_1std": show_lr_1std, "lr_2std": show_lr_2std,
            "tl_resist": show_tl_resist, "tl_support": show_tl_support,
            "bb_bands": show_bb_bands, "bb_center": show_bb_center,
            "sr_resist": show_sr_resist, "sr_support": show_sr_support,
        }

        ma_periods  = [p for p, on in [(20, ma_20), (50, ma_50), (100, ma_100), (200, ma_200)] if on]
        # LinReg is always computed — it drives the Master Plan projection.
        linreg_data = calc_linreg(df, future_bar_idx)
        tl_data     = calc_trendlines(df, swing_highs, swing_lows, future_dates, future_bar_idx) if (show_tl_resist or show_tl_support) else None
        ma_data     = calc_moving_averages(df, future_dates, future_bar_idx, ma_periods) if ma_periods else None
        fib_data    = calc_fibonacci(swing_highs, swing_lows) if show_fib else None
        bb_data     = calc_bollinger(df, future_dates, future_bar_idx, period=20) if (show_bb_bands or show_bb_center) else None
        sr_data     = calc_sr(swing_highs, swing_lows) if (show_sr_resist or show_sr_support) else None

        fig = build_chart(
            df, current_price, future_dates, future_bar_idx,
            linreg_data, tl_data, ma_data, fib_data, bb_data, sr_data, show,
            ticker, title_suffix=" (Daily)", tool_name="Master Plan",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
            render_projection_table(proj_df, "+1 Month")

        st.markdown("---")
        st.subheader(f"Master Plan — Optimum Call by Expiration ({ticker})")
        render_master_plan(call_chains, df.index[-1], linreg_data, current_price,
                           is_weekly=False, active_only=active_only)


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
            st.button(lbl, key=f"mp_ltper_{k}", use_container_width=True,
                      on_click=lambda _k=k: st.session_state.update({"mp_lt_period": _k}))
            pct = lt_period_changes.get(k)
            if pct is not None:
                st.markdown(pct_label(pct), unsafe_allow_html=True)
    lt_period = st.session_state.mp_lt_period

    # --- Projection Horizon ---
    LT_HORIZON_OPTS = {"6 Months": 26, "1 Year": 52, "2 Years": 104}
    st.write("**Select projection horizon:**")
    _lt_hor_cols = st.columns(len(LT_HORIZON_OPTS))
    for i, (k, bars) in enumerate(LT_HORIZON_OPTS.items()):
        with _lt_hor_cols[i]:
            st.button(k, key=f"mp_lthor_{k.replace(' ','_')}", use_container_width=True,
                      on_click=lambda _k=k: st.session_state.update({"mp_lt_horizon": _k}))
    lt_horizon_label = st.session_state.mp_lt_horizon
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
            lt_lr_center = st.checkbox("LR Center",    value=True,  key="mp_lt_lr_center")
            lt_lr_1std   = st.checkbox("LR ±1σ",       value=False, key="mp_lt_lr_1std")
            lt_lr_2std   = st.checkbox("LR ±2σ",       value=False, key="mp_lt_lr_2std")
        with _oc_lt[1]:
            st.caption("Trend Lines")
            lt_tl_resist  = st.checkbox("Resistance",  value=False, key="mp_lt_tl_resist")
            lt_tl_support = st.checkbox("Support",     value=False, key="mp_lt_tl_support")
        with _oc_lt[2]:
            st.caption("Weekly MAs")
            lt_ma13  = st.checkbox("SMA 13w  (~1Q)",  value=False, key="mp_lt_ma13")
            lt_ma26  = st.checkbox("SMA 26w  (~6M)",  value=False, key="mp_lt_ma26")
            lt_ma52  = st.checkbox("SMA 52w  (~1Y)",  value=False, key="mp_lt_ma52")
            lt_ma104 = st.checkbox("SMA 104w (~2Y)",  value=False, key="mp_lt_ma104")
        with _oc_lt[3]:
            st.caption("Fibonacci")
            lt_fib = st.checkbox("Fib Retracement", value=False, key="mp_lt_fib")
        with _oc_lt[4]:
            st.caption("Bollinger (26w)")
            lt_bb_bands  = st.checkbox("BB Upper/Lower", value=False, key="mp_lt_bb_bands")
            lt_bb_center = st.checkbox("BB Center",      value=False, key="mp_lt_bb_center")
        with _oc_lt[5]:
            st.caption("S/R Horizontal")
            lt_sr_resist  = st.checkbox("S/R Resistance", value=False, key="mp_lt_sr_resist")
            lt_sr_support = st.checkbox("S/R Support",    value=False, key="mp_lt_sr_support")
        with _oc_lt[6]:
            st.caption("Options Filter")
            lt_active_only = st.checkbox("Active only", value=True, key="mp_lt_active_only",
                                         help="Volume > 0 or OI > 10, traded in last 30 days")

        st.markdown("---")

        show_lt = {
            "lr_center": lt_lr_center, "lr_1std": lt_lr_1std, "lr_2std": lt_lr_2std,
            "tl_resist": lt_tl_resist, "tl_support": lt_tl_support,
            "bb_bands": lt_bb_bands, "bb_center": lt_bb_center,
            "sr_resist": lt_sr_resist, "sr_support": lt_sr_support,
        }

        lt_ma_periods = [p for p, on in [(13, lt_ma13), (26, lt_ma26), (52, lt_ma52), (104, lt_ma104)] if on]
        # LinReg is always computed — it drives the Master Plan projection.
        lt_linreg = calc_linreg(df_lt, future_bar_idx_lt)
        lt_tl     = calc_trendlines(df_lt, swing_highs_lt, swing_lows_lt, future_dates_lt, future_bar_idx_lt) if (lt_tl_resist or lt_tl_support) else None
        lt_ma     = calc_moving_averages(df_lt, future_dates_lt, future_bar_idx_lt, lt_ma_periods) if lt_ma_periods else None
        lt_fib_d  = calc_fibonacci(swing_highs_lt, swing_lows_lt) if lt_fib else None
        lt_bb     = calc_bollinger(df_lt, future_dates_lt, future_bar_idx_lt, period=26) if (lt_bb_bands or lt_bb_center) else None
        lt_sr     = calc_sr(swing_highs_lt, swing_lows_lt) if (lt_sr_resist or lt_sr_support) else None

        fig_lt = build_chart(
            df_lt, current_price_lt, future_dates_lt, future_bar_idx_lt,
            lt_linreg, lt_tl, lt_ma, lt_fib_d, lt_bb, lt_sr, show_lt,
            ticker, title_suffix=" (Weekly)", is_weekly=True, tool_name="Master Plan",
        )
        st.plotly_chart(fig_lt, use_container_width=True, config={"displayModeBar": False})

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
            render_projection_table(proj_df_lt, "+1 Year")

        st.markdown("---")
        st.subheader(f"Master Plan — Optimum Call by Expiration ({ticker})")
        render_master_plan(call_chains, df_lt.index[-1], lt_linreg, current_price_lt,
                           is_weekly=True, active_only=lt_active_only)
