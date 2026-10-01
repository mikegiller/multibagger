import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta

import journal
import utils

st.set_page_config(page_title="Trade Journal", layout="wide")
st.title("Trade Journal")
st.info(
    "**Purpose:** Log every trade with your thesis and expected return, then see how "
    "your predictions played out: expected vs actual, and how your contract compared "
    "to the other options available that day. Numbers only, no scores."
)

journal.init_db()
TAGS = journal.load_tags()
TYPE_OPTIONS = journal.TRADE_TYPES
TYPE_LABEL = journal.TRADE_TYPE_LABELS


def fmt_pct(v):
    return "—" if v is None or pd.isna(v) else f"{v:+.1f}%"


def fmt_money(v):
    return "—" if v is None or pd.isna(v) else f"${v:,.2f}"


# === Prefill from "Log this trade" buttons on other pages ===
_prefill = st.session_state.pop(journal.PREFILL_KEY, None)
if _prefill:
    st.session_state["tj_type"] = "call"
    st.session_state["tj_ticker"] = _prefill["ticker"]
    if _prefill.get("expiration"):
        st.session_state["tj_exp"] = _prefill["expiration"]
    if _prefill.get("strike") is not None:
        st.session_state["tj_strike"] = _prefill["strike"]
    st.session_state["tj_context"] = _prefill.get("context") or {}
    st.session_state["tj_source"] = _prefill.get("source_page")

entries = journal.list_entries()
open_entries = [e for e in entries if e["status"] == "open"]
due = [e for e in open_entries if e["target_due"]]
graded = [e for e in entries if journal.result(e) is not None]

if due:
    st.warning(
        f"**{len(due)} target date(s) reached:** "
        + ", ".join(e["description"] for e in due)
        + ". Record the result in the **Open** tab.",
        icon=":material/event_available:",
    )

tab_log, tab_open, tab_closed, tab_insights = st.tabs(
    ["Log trade", f"Open ({len(open_entries)})", f"Results ({len(graded)})", "Insights"],
    default="Log trade" if _prefill else None,
)


# ═══════════════════════════════════════════════════════════════════════
# Dialogs
# ═══════════════════════════════════════════════════════════════════════
def _snapshot(entry, event):
    if entry["trade_type"] == "stock":
        return
    with st.spinner("Saving option chain snapshot (all expirations)..."):
        warn = journal.snapshot_entry(entry["id"], entry["ticker"], event)
    if warn:
        st.session_state["tj_flash_warn"] = warn


@st.dialog("Record a sale", width="medium")
def close_dialog(entry):
    st.write(f"**{entry['description']}**, {entry['open_qty']:g} open at {fmt_money(entry['order_price'])}")
    cv = journal.current_value(entry)
    qty = st.number_input("Quantity sold", min_value=0.0, max_value=float(entry["open_qty"]),
                          value=float(entry["open_qty"]), step=1.0)
    price = st.number_input("Sale price (per share, as quoted)", min_value=0.0,
                            value=float(cv["mid"]) if cv else 0.0, step=0.05, format="%.2f",
                            help="Defaults to the current mid. Enter your actual fill.")
    closed_on = st.date_input("Sale date", value=journal.today_et(),
                              min_value=journal._d(entry["created_at"]), max_value=journal.today_et())
    note = st.text_input("Note (optional)", placeholder="Why sell now?")
    closing_all = qty >= entry["open_qty"] - 1e-9
    reflection = st.text_area("Reflection: what did you get right or wrong?",
                              value=entry["reflection"] or "") if closing_all else None
    if price > 0:
        st.caption(f"Return on this sale: **{fmt_pct(journal.ret_pct(price, entry['order_price']))}**")
    if st.button("Save sale", type="primary", disabled=qty <= 0 or price <= 0):
        journal.add_close(entry["id"], closed_on.isoformat(), qty, price, "sale", note)
        if closing_all and reflection:
            journal.set_reflection(entry["id"], reflection)
        # The exit snapshot only matters for the final sale, and only if there's no
        # target-date result (which already has its own snapshot).
        if closing_all and entry["target_mark"] is None:
            _snapshot(entry, "close")
        st.rerun()


@st.dialog("Target date reached", width="medium")
def target_dialog(entry):
    r_exp = entry["expected_return_pct"]
    st.write(f"**{entry['description']}**: you expected **{fmt_pct(r_exp)}** by "
             f"**{entry['target_date']}**.")
    choice = st.segmented_control("What happened?", ["I sold", "Still holding"], default="I sold")
    cv = journal.current_value(entry)
    if choice == "I sold":
        price = st.number_input("Sale price (per share)", min_value=0.0,
                                value=float(cv["mid"]) if cv else 0.0, step=0.05, format="%.2f")
        closed_on = st.date_input("Sale date", value=journal.today_et(),
                                  min_value=journal._d(entry["created_at"]), max_value=journal.today_et())
        reflection = st.text_area("Reflection: what did you get right or wrong?")
        if st.button("Save sale", type="primary", disabled=price <= 0):
            journal.add_close(entry["id"], closed_on.isoformat(), entry["open_qty"], price, "sale")
            if reflection:
                journal.set_reflection(entry["id"], reflection)
            _snapshot(entry, "close")
            st.rerun()
    elif choice == "Still holding":
        st.caption("Records today's market value as your target-date result. The position "
                   "stays open and keeps tracking until you sell.")
        if cv is None:
            st.warning("No live quote available. Enter the current value yourself.")
        price = st.number_input("Current value (per share)", min_value=0.0,
                                value=float(cv["mid"]) if cv else 0.0, step=0.05, format="%.2f",
                                help="Defaults to the current mid.")
        if st.button("Save target-date result", type="primary", disabled=price <= 0):
            journal.add_close(entry["id"], journal.today_et().isoformat(), entry["open_qty"],
                              price, "target_mark")
            _snapshot(entry, "target")
            st.rerun()


@st.dialog("Revise your view", width="medium")
def revise_dialog(entry):
    st.caption("Your original prediction stays locked and results are still measured against "
               "it. A revision is a dated note of how your view changed.")
    exp_pct = st.number_input("New expected return %", value=float(entry["expected_return_pct"]), step=5.0)
    tgt = st.date_input("New target date", value=journal._d(entry["target_date"]))
    note = st.text_area("What changed?")
    if st.button("Save revision", type="primary", disabled=not note.strip()):
        journal.add_revision(entry["id"], exp_pct, tgt.isoformat(), note.strip())
        st.rerun()


@st.dialog("Fix a typo", width="medium")
def edit_dialog(entry):
    st.caption(f"Editable until {entry['edit_until']:%b %d, %I:%M %p} ET. "
               "To change the contract itself, delete and log it again.")
    qty = st.number_input("Quantity", min_value=0.0, value=float(entry["quantity"]), step=1.0)
    price = st.number_input("Order price", min_value=0.0, value=float(entry["order_price"]),
                            step=0.05, format="%.2f")
    exp_pct = st.number_input("Expected return %", value=float(entry["expected_return_pct"]), step=5.0)
    tgt = st.date_input("Target date", value=journal._d(entry["target_date"]))
    thesis = st.text_area("Thesis", value=entry["thesis"])
    tags = st.pills("Tags", TAGS + [t for t in entry["tags"] if t not in TAGS],
                    selection_mode="multi", default=entry["tags"])
    c1, c2 = st.columns(2)
    if c1.button("Save", type="primary", disabled=qty <= 0 or price <= 0):
        journal.update_entry(entry["id"], {
            "quantity": qty, "order_price": price, "expected_return_pct": exp_pct,
            "target_date": tgt.isoformat(), "thesis": thesis,
        }, tags)
        st.rerun()
    confirm = c2.checkbox("Delete this entry")
    if confirm and c2.button("Confirm delete", icon=":material/delete:"):
        journal.delete_entry(entry["id"])
        st.rerun()


@st.dialog("Reflection", width="medium")
def reflection_dialog(entry):
    text = st.text_area("What did you get right or wrong?", value=entry["reflection"] or "", height=200)
    if st.button("Save", type="primary"):
        journal.set_reflection(entry["id"], text)
        st.rerun()


_flash = st.session_state.pop("tj_flash_warn", None)
if _flash:
    st.warning(_flash)
_flash_ok = st.session_state.pop("tj_flash_ok", None)
if _flash_ok:
    st.success(_flash_ok)


# ═══════════════════════════════════════════════════════════════════════
# Log trade
# ═══════════════════════════════════════════════════════════════════════
def _leg_label(calls, strike):
    row = calls[np.isclose(calls["strike"], strike)]
    q = journal.quote_row(row.iloc[0]) if not row.empty else None
    if q is None:
        return f"{strike:g}  (no quote)"
    tag = "" if q["source"] == "Live" else ", last trade"
    return f"{strike:g}  (mid {q['mid']:.2f}{tag})"


def _clear_log_form():
    for k in list(st.session_state):
        if k in ("tj_qty", "tj_expected", "tj_thesis", "tj_tags", "tj_context", "tj_source") \
                or k.startswith(("tj_price_", "tj_target_")):
            st.session_state.pop(k, None)


with tab_log:
    if "tj_ticker" not in st.session_state:
        st.session_state["tj_ticker"] = st.session_state.get("active_ticker_stocks", "SPY")
    if "tj_type" not in st.session_state:
        st.session_state["tj_type"] = "call"
    if st.session_state.get("tj_context"):
        src = st.session_state.get("tj_source") or "another page"
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.caption(f":material/link: Pre-filled from **{src}**. Its projection and market "
                   "context will be saved with this trade.")
        if c2.button("Discard context", key="tj_discard_ctx"):
            st.session_state.pop("tj_context", None)
            st.session_state.pop("tj_source", None)
            st.rerun()

    c1, c2 = st.columns([1, 2])
    trade_type = c1.segmented_control("Trade type", TYPE_OPTIONS, format_func=TYPE_LABEL.get,
                                      key="tj_type", required=True)
    ticker = c2.text_input("Ticker", key="tj_ticker", max_chars=12).upper().strip()

    legs, ref_price, valid_contract = [], None, bool(ticker)
    spot = journal.fetch_spot(ticker) if ticker else None
    if ticker and spot is None:
        st.error(f"No price data for **{ticker}**.")
        valid_contract = False
    elif ticker:
        st.caption(f"{ticker} last price: **{fmt_money(spot)}**")

    if valid_contract and trade_type in ("call", "spread"):
        exps = journal.fetch_expirations(ticker)
        if not exps:
            st.error(f"No options available for **{ticker}**.")
            valid_contract = False
        else:
            today = journal.today_et()
            if st.session_state.get("tj_exp") not in exps:
                st.session_state["tj_exp"] = exps[-1]
            c1, c2, c3 = st.columns(3)
            exp = c1.selectbox(
                "Expiration", exps, key="tj_exp",
                format_func=lambda x: f"{x}  ({(journal._d(x) - today).days} days)",
            )
            calls = journal.fetch_calls(ticker, exp)
            strikes = calls["strike"].tolist() if not calls.empty else []
            if not strikes:
                st.error("No call strikes for that expiration.")
                valid_contract = False
            else:
                if st.session_state.get("tj_strike") not in strikes:
                    st.session_state["tj_strike"] = min(strikes, key=lambda s: abs(s - (spot or 0)))
                long_strike = c2.selectbox("Strike" if trade_type == "call" else "Long strike (buy)",
                                           strikes, key="tj_strike",
                                           format_func=lambda s: _leg_label(calls, s))
                long_row = calls[np.isclose(calls["strike"], long_strike)].iloc[0]
                legs.append({"side": "long", "strike": float(long_strike), "expiration": exp,
                             "contract_symbol": long_row.get("contractSymbol")})
                lq = journal.quote_row(long_row)
                ref_price = lq["mid"] if lq else None

                if trade_type == "spread":
                    higher = [s for s in strikes if s > long_strike]
                    if not higher:
                        c3.error("No higher strike to sell.")
                        valid_contract = False
                    else:
                        if st.session_state.get("tj_short_strike") not in higher:
                            st.session_state["tj_short_strike"] = higher[min(len(higher) - 1, 4)]
                        short_strike = c3.selectbox("Short strike (sell)", higher, key="tj_short_strike",
                                                    format_func=lambda s: _leg_label(calls, s))
                        short_row = calls[np.isclose(calls["strike"], short_strike)].iloc[0]
                        legs.append({"side": "short", "strike": float(short_strike), "expiration": exp,
                                     "contract_symbol": short_row.get("contractSymbol")})
                        sq = journal.quote_row(short_row)
                        ref_price = round(lq["mid"] - sq["mid"], 2) if lq and sq else None
    elif valid_contract and trade_type == "stock":
        ref_price = spot

    if valid_contract:
        if ref_price is not None:
            st.caption(f"Current mid for this position: **{fmt_money(ref_price)}**"
                       + (" (net debit)" if trade_type == "spread" else ""))
        # Keys include the contract so the price/target defaults reset when you pick another one.
        sig = "_".join(f"{l['strike']:g}{l['expiration']}" for l in legs) or f"{ticker}_stock"
        with st.form("tj_log_form", border=True):
            unit = "Shares" if trade_type == "stock" else "Contracts"
            price_label = {"call": "Order price (per share, as quoted)",
                           "spread": "Net debit paid (per share)",
                           "stock": "Order price (per share)"}[trade_type]
            c1, c2 = st.columns(2)
            qty = c1.number_input(unit, min_value=0.0, value=1.0, step=1.0, key="tj_qty")
            price = c2.number_input(price_label, min_value=0.0, value=float(ref_price or 0.0),
                                    step=0.05, format="%.2f", key=f"tj_price_{sig}")
            default_target = legs[0]["expiration"] if legs else \
                (journal.today_et() + timedelta(days=365)).isoformat()
            c3, c4 = st.columns(2)
            expected = c3.number_input("Expected return on the position (%)", value=100.0, step=5.0,
                                       key="tj_expected",
                                       help="Return on this position itself, e.g. +150% on the call.")
            target = c4.date_input("Target date", value=journal._d(default_target),
                                   min_value=journal.today_et() + timedelta(days=1), key=f"tj_target_{sig}")
            thesis = st.text_area("Thesis: why this trade?", key="tj_thesis", height=120)
            tags = st.pills("Tags", TAGS, selection_mode="multi", key="tj_tags")
            st.caption("The expected return, target date and thesis lock 24 hours after saving. "
                       "After that, changes are added as dated revisions.")
            submitted = st.form_submit_button("Log trade", type="primary", icon=":material/save:")

        if submitted:
            errors = []
            if qty <= 0:
                errors.append("Quantity must be above zero.")
            if price <= 0:
                errors.append("Order price must be above zero.")
            if not thesis.strip():
                errors.append("Write down your thesis. It's the point of the journal.")
            if errors:
                for err in errors:
                    st.error(err)
            else:
                context = {**journal.market_context(), **(st.session_state.get("tj_context") or {})}
                entry_id = journal.create_entry({
                    "ticker": ticker,
                    "asset_class": "index" if utils.is_index_ticker(ticker) else "stocks",
                    "trade_type": trade_type, "quantity": qty, "order_price": price,
                    "underlying_price": spot, "expected_return_pct": expected,
                    "target_date": target.isoformat(), "thesis": thesis.strip(),
                    "context": context, "source_page": st.session_state.get("tj_source"),
                }, legs, tags or [])
                entry = journal.get_entry(entry_id)
                _snapshot(entry, "entry")
                st.session_state["tj_flash_ok"] = f"Logged: {entry['description']}."
                _clear_log_form()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# Open positions
# ═══════════════════════════════════════════════════════════════════════
with tab_open:
    if not open_entries:
        st.info("No open positions. Log a trade to start.")
    else:
        rows = []
        with st.spinner("Fetching live quotes..."):
            values = {e["id"]: journal.current_value(e) for e in open_entries}
        for e in open_entries:
            cv = values[e["id"]]
            mid = cv["mid"] if cv else None
            bid = cv["bid"] if cv else None
            days_left = (journal._d(e["target_date"]) - journal.today_et()).days
            rows.append({
                "Position": e["description"],
                "Open qty": e["open_qty"],
                "Order": e["order_price"],
                "Mid": mid,
                "Return @ mid": journal.ret_pct(mid, e["order_price"]),
                "Return @ bid": journal.ret_pct(bid, e["order_price"]),
                "Expected": e["expected_return_pct"],
                "P&L @ mid": None if mid is None else (mid - e["order_price"]) * e["open_qty"] * journal.multiplier(e),
                "Target date": e["target_date"],
                "Days left": days_left,
                "Quote": ("Expired" if e["expired"] else cv["source"] if cv else "No quote"),
                "Tags": ", ".join(e["tags"]),
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={
                "Order": st.column_config.NumberColumn(format="$%.2f"),
                "Mid": st.column_config.NumberColumn(format="$%.2f"),
                "Return @ mid": st.column_config.NumberColumn(format="%+.1f%%"),
                "Return @ bid": st.column_config.NumberColumn(
                    format="%+.1f%%", help="What selling at the bid right now would return. "
                                           "LEAPS spreads are wide, so this is the honest number."),
                "Expected": st.column_config.NumberColumn(format="%+.1f%%"),
                "P&L @ mid": st.column_config.NumberColumn(format="$%,.0f"),
                "Days left": st.column_config.NumberColumn(help="Negative = target date has passed"),
            },
        )
        st.caption("Return @ mid uses the mid between bid and ask. Last trade = no two-sided "
                   "quote, so the last trade price is used (common on illiquid index LEAPS).")

        st.subheader("Manage a position")
        by_label = {f"#{e['id']}  {e['description']}  (logged {e['created_at'][:10]})": e
                    for e in open_entries}
        sel = by_label[st.selectbox("Position", list(by_label), key="tj_manage_sel")]

        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"**Thesis:** {sel['thesis']}")
                if sel["tags"]:
                    st.caption("Tags: " + ", ".join(sel["tags"]))
                if sel["context"].get("vix"):
                    st.caption(f"VIX at entry: {sel['context']['vix']['level']:.1f}  ·  "
                               f"Underlying at entry: {fmt_money(sel['underlying_price'])}")
            with c2:
                st.markdown(f"**Expected:** {fmt_pct(sel['expected_return_pct'])} by {sel['target_date']}")
                for r in sel["revisions"]:
                    st.caption(f"Revised {r['created_at'][:10]}: {fmt_pct(r['expected_return_pct'])} "
                               f"by {r['target_date']}: {r['note']}")
                if sel["sales"]:
                    for c in sel["sales"]:
                        st.caption(f"Sold {c['quantity']:g} on {c['closed_on']} at {fmt_money(c['price'])} "
                                   f"({fmt_pct(journal.ret_pct(c['price'], sel['order_price']))})")
                if sel["target_mark"]:
                    m = sel["target_mark"]
                    st.caption(f"Target-date result recorded {m['closed_on']}: {fmt_money(m['price'])} "
                               f"({fmt_pct(journal.ret_pct(m['price'], sel['order_price']))})")
            if sel["expired"]:
                st.warning("This contract has expired. Record the final sale or settlement price.",
                           icon=":material/hourglass_bottom:")

            with st.container(horizontal=True):
                if sel["target_due"]:
                    if st.button("Target date reached", type="primary", icon=":material/flag:"):
                        target_dialog(sel)
                if st.button("Record a sale", icon=":material/sell:"):
                    close_dialog(sel)
                if st.button("Revise view", icon=":material/edit_calendar:"):
                    revise_dialog(sel)
                if sel["editable"]:
                    if st.button("Fix a typo / delete", icon=":material/edit:"):
                        edit_dialog(sel)


# ═══════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════
def comparison_view(e):
    df, info = journal.comparison(e)
    if df is None:
        st.caption(info)
        return
    if df.empty:
        st.caption("No contracts had two-sided quotes in both snapshots.")
        return
    exit_on = info
    mine = df[df["yours"]]
    st.markdown(f"**Other options over the same period** ({e['created_at'][:10]} → {exit_on}), "
                "all priced at mid so they compare like for like")
    if not mine.empty:
        m = mine.iloc[0]
        rank = int((df["return_pct"] > m["return_pct"]).sum()) + 1
        st.caption(f"Your contract at mid: **{fmt_pct(m['return_pct'])}**, which ranks "
                   f"**{rank} of {len(df)}** contracts with quotes at both ends.")
    else:
        st.caption("Your contract had no two-sided quote in one of the snapshots, so it isn't on the grid.")

    # Diverging heatmap: red = loss, neutral gray = 0, blue = gain. Color range is
    # clipped to the 95th percentile so one 5000% lottery ticket doesn't wash out the rest.
    dark = getattr(st.context.theme, "type", "light") == "dark"
    mid_gray = "#383835" if dark else "#f0efec"
    lim = float(np.nanpercentile(df["return_pct"].abs(), 95)) or 1.0
    piv = df.pivot_table(index="expiration", columns="strike", values="return_pct")
    entry_px = df.pivot_table(index="expiration", columns="strike", values="entry_mid")
    exit_px = df.pivot_table(index="expiration", columns="strike", values="exit_mid")
    custom = np.dstack([entry_px.reindex_like(piv).values, exit_px.reindex_like(piv).values])
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[f"{s:g}" for s in piv.columns], y=piv.index.tolist(),
        customdata=custom, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorscale=[[0, "#e34948"], [0.5, mid_gray], [1, "#2a78d6"]],
        colorbar=dict(title="Return %", ticksuffix="%"),
        hovertemplate="Strike %{x}<br>Expiration %{y}<br>Entry mid $%{customdata[0]:.2f}"
                      "<br>Exit $%{customdata[1]:.2f}<br>Return %{z:+.1f}%<extra></extra>",
    ))
    if not mine.empty:
        fig.add_trace(go.Scatter(
            x=[f"{m['strike']:g}"], y=[m["expiration"]], mode="markers",
            marker=dict(size=14, symbol="square-open", line=dict(width=3), color="#000" if not dark else "#fff"),
            name="Your contract", hovertemplate="Your contract<extra></extra>",
        ))
    fig.update_layout(
        height=max(300, 26 * len(piv.index) + 120), margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(title="Strike", type="category"), yaxis=dict(title="Expiration", type="category"),
        showlegend=False,
        template="plotly_dark" if dark else "plotly_white",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Expirations that passed before the exit date are valued at intrinsic value on "
               "their expiration day. The square marks your contract.")

    with st.expander("Table view: top 15 by return"):
        top = df.sort_values("return_pct", ascending=False).head(15)
        st.dataframe(
            top[["expiration", "strike", "entry_mid", "exit_mid", "return_pct", "value_source", "yours"]],
            hide_index=True, width="stretch",
            column_config={
                "expiration": "Expiration", "strike": "Strike",
                "entry_mid": st.column_config.NumberColumn("Entry mid", format="$%.2f"),
                "exit_mid": st.column_config.NumberColumn("Exit value", format="$%.2f"),
                "return_pct": st.column_config.NumberColumn("Return", format="%+.1f%%"),
                "value_source": "Exit priced by",
                "yours": st.column_config.CheckboxColumn("Yours"),
            },
        )


with tab_closed:
    if not graded:
        st.info("Results appear here once a position is fully sold or its target date is recorded.")
    else:
        rows = []
        for e in graded:
            r = journal.result(e)
            rows.append({
                "Position": e["description"], "Tags": ", ".join(e["tags"]),
                "Expected": r["expected"], "Actual": r["actual"], "Delta (pp)": r["delta"],
                "Direction": r["direction"], "Days held": r["days_held"],
                "Days planned": r["days_planned"], "Timing": r["timing"],
                "Status": "Closed" if e["status"] == "closed" else "Open (target recorded)",
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={
                "Expected": st.column_config.NumberColumn(format="%+.1f%%"),
                "Actual": st.column_config.NumberColumn(format="%+.1f%%"),
                "Delta (pp)": st.column_config.NumberColumn(
                    format="%+.1f", help="Actual minus expected, in percentage points"),
            },
        )
        st.caption("Overshoot = did better than expected. Undershoot = expected more than you got. "
                   "Actual for a target-date result = sales before the target date plus the "
                   "target-date value for the rest.")

        st.subheader("Trade detail")
        by_label = {f"#{e['id']}  {e['description']}  (logged {e['created_at'][:10]})": e for e in graded}
        sel = by_label[st.selectbox("Trade", list(by_label), key="tj_result_sel")]
        r = journal.result(sel)
        with st.container(border=True):
            with st.container(horizontal=True):
                st.metric("Expected", fmt_pct(r["expected"]))
                st.metric("Actual", fmt_pct(r["actual"]),
                          delta=f"{r['delta']:+.1f} pp, {r['direction'].lower()}")
                st.metric("Held", f"{r['days_held']} of {r['days_planned']} days", help=r["timing"])
                if r["basis"] == "target" and r["final_realized"] is not None:
                    st.metric("Final realized", fmt_pct(r["final_realized"]),
                              help="All sales, including after the target date. Not used for the delta.")
            st.markdown(f"**Thesis:** {sel['thesis']}")
            for rv in sel["revisions"]:
                st.caption(f"Revised {rv['created_at'][:10]}: {fmt_pct(rv['expected_return_pct'])} "
                           f"by {rv['target_date']}: {rv['note']}")
            sales = [f"{c['quantity']:g} on {c['closed_on']} at {fmt_money(c['price'])} "
                     f"({fmt_pct(journal.ret_pct(c['price'], sel['order_price']))})" for c in sel["sales"]]
            if sales:
                st.caption(f"Bought {sel['quantity']:g} at {fmt_money(sel['order_price'])}. Sold: " + "; ".join(sales))
            if sel["reflection"]:
                st.markdown(f"**Reflection:** {sel['reflection']}")
            if st.button("Edit reflection" if sel["reflection"] else "Add reflection",
                         icon=":material/rate_review:", key="tj_refl_btn"):
                reflection_dialog(sel)

        if sel["trade_type"] != "stock":
            with st.container(border=True):
                comparison_view(sel)


# ═══════════════════════════════════════════════════════════════════════
# Insights
# ═══════════════════════════════════════════════════════════════════════
with tab_insights:
    overall, by_tag, by_type = journal.insights(entries)
    if overall is None:
        st.info("Insights appear once you have results.")
    else:
        with st.container(horizontal=True):
            st.metric("Trades with results", int(overall["Trades"]))
            st.metric("Avg expected", fmt_pct(overall["Avg expected %"]))
            st.metric("Avg actual", fmt_pct(overall["Avg actual %"]))
            st.metric("Avg delta", f"{overall['Avg delta (pp)']:+.1f} pp")
            st.metric("Overshot / undershot", f"{overall['Overshot']} / {overall['Undershot']}")
        pct_cols = {
            "Avg expected %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Avg actual %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Avg delta (pp)": st.column_config.NumberColumn(format="%+.1f"),
        }
        st.subheader("By tag")
        st.caption("A trade with several tags counts under each of them.")
        st.dataframe(by_tag, hide_index=True, width="stretch", column_config=pct_cols)
        st.subheader("By trade type")
        st.dataframe(by_type, hide_index=True, width="stretch", column_config=pct_cols)
