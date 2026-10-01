# journal.py — Trade Journal storage, return math, live quotes, and chain snapshots.
#
# See docs/trade_journal_spec.md for the agreed design. Key rules encoded here:
#   - Predictions (expected return %, target date, thesis) are locked after a
#     24-hour typo window; later changes are dated revisions, and results are
#     always measured against the ORIGINAL prediction.
#   - Returns are on the position itself: (price / order_price - 1) * 100. The
#     x100 option multiplier cancels out, so prices are stored per share as quoted.
#   - No scores. A result is expected vs actual, the delta, and its direction.
#   - Nothing is tracked after a position is sold.
#   - Data lives in MULTIBAGGER_JOURNAL_DIR (default ./journal, gitignored).
#     Trade data must never be committed.
import json
import os
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import utils  # applies the yfinance cache workaround on import

ET = ZoneInfo("America/New_York")
_ROOT = os.path.dirname(os.path.abspath(__file__))
JOURNAL_DIR = os.environ.get("MULTIBAGGER_JOURNAL_DIR") or os.path.join(_ROOT, "journal")
DB_PATH = os.path.join(JOURNAL_DIR, "journal.db")
CHAINS_DIR = os.path.join(JOURNAL_DIR, "chains")

EDIT_WINDOW = timedelta(hours=24)
TRADE_TYPES = ["call", "spread", "stock"]
TRADE_TYPE_LABELS = {"call": "Call", "spread": "Call spread", "stock": "Stock"}
DEFAULT_TAGS = ["trend continuation", "dip buy", "earnings", "sector rotation",
                "macro", "breakout", "mean reversion"]
SNAPSHOT_WORKERS = 4  # parallel option_chain requests; higher risks Yahoo rate limits

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,          -- ET ISO timestamp
    ticker              TEXT NOT NULL,
    asset_class         TEXT NOT NULL,          -- stocks | index
    trade_type          TEXT NOT NULL,          -- call | spread | stock
    quantity            REAL NOT NULL,          -- contracts or shares
    order_price         REAL NOT NULL,          -- per share; net debit for spreads
    underlying_price    REAL,
    expected_return_pct REAL NOT NULL,
    target_date         TEXT NOT NULL,          -- YYYY-MM-DD
    thesis              TEXT NOT NULL DEFAULT '',
    context_json        TEXT NOT NULL DEFAULT '{}',
    source_page         TEXT
);
CREATE TABLE IF NOT EXISTS legs (
    entry_id        INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    side            TEXT NOT NULL,              -- long | short
    strike          REAL NOT NULL,
    expiration      TEXT NOT NULL,              -- YYYY-MM-DD
    contract_symbol TEXT
);
CREATE TABLE IF NOT EXISTS entry_tags (
    entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id            INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    created_at          TEXT NOT NULL,
    expected_return_pct REAL NOT NULL,
    target_date         TEXT NOT NULL,
    note                TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS closes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    recorded_at TEXT NOT NULL,                  -- when it was entered (ET)
    closed_on   TEXT NOT NULL,                  -- trade date (YYYY-MM-DD)
    quantity    REAL NOT NULL,
    price       REAL NOT NULL,
    kind        TEXT NOT NULL,                  -- sale | target_mark
    note        TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS reflections (
    entry_id   INTEGER PRIMARY KEY REFERENCES entries(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    text       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    taken_at TEXT NOT NULL,
    event    TEXT NOT NULL,                     -- entry | close | target
    path     TEXT NOT NULL                      -- relative to JOURNAL_DIR
);
"""


# === Time helpers (everything is US Eastern market time) ===
def now_et():
    return datetime.now(ET)


def today_et():
    return now_et().date()


def _ts():
    return now_et().isoformat(timespec="seconds")


def _d(s):
    return date.fromisoformat(s[:10])


# === Config ===
def load_tags():
    """Tag list from journal_config.json (tracked in git, edit freely)."""
    try:
        with open(os.path.join(_ROOT, "journal_config.json")) as f:
            return json.load(f)["tags"]
    except Exception:
        return DEFAULT_TAGS


# === Database ===
@contextmanager
def _conn():
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript(_SCHEMA)
        if con.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def create_entry(entry, legs, tags):
    """entry: dict of entries columns (minus id/created_at); legs: list of leg dicts."""
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO entries (created_at, ticker, asset_class, trade_type, quantity,
               order_price, underlying_price, expected_return_pct, target_date, thesis,
               context_json, source_page)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (_ts(), entry["ticker"], entry["asset_class"], entry["trade_type"],
             entry["quantity"], entry["order_price"], entry.get("underlying_price"),
             entry["expected_return_pct"], entry["target_date"], entry.get("thesis", ""),
             json.dumps(entry.get("context") or {}, default=str), entry.get("source_page")),
        )
        entry_id = cur.lastrowid
        con.executemany(
            "INSERT INTO legs (entry_id, side, strike, expiration, contract_symbol) VALUES (?,?,?,?,?)",
            [(entry_id, l["side"], l["strike"], l["expiration"], l.get("contract_symbol")) for l in legs],
        )
        con.executemany("INSERT INTO entry_tags (entry_id, tag) VALUES (?,?)",
                        [(entry_id, t) for t in tags])
    return entry_id


def update_entry(entry_id, fields, tags):
    """Typo fixes inside the edit window only. fields: subset of the editable columns."""
    allowed = {"quantity", "order_price", "expected_return_pct", "target_date", "thesis"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    entry = get_entry(entry_id)
    if entry is None or not entry["editable"]:
        raise ValueError("This entry is locked; add a revision instead.")
    with _conn() as con:
        if fields:
            sets = ", ".join(f"{k} = ?" for k in fields)
            con.execute(f"UPDATE entries SET {sets} WHERE id = ?", (*fields.values(), entry_id))
        con.execute("DELETE FROM entry_tags WHERE entry_id = ?", (entry_id,))
        con.executemany("INSERT INTO entry_tags (entry_id, tag) VALUES (?,?)",
                        [(entry_id, t) for t in tags])


def delete_entry(entry_id):
    """Only inside the edit window (for mistaken logs)."""
    entry = get_entry(entry_id)
    if entry is None or not entry["editable"]:
        raise ValueError("Only entries inside the 24-hour edit window can be deleted.")
    with _conn() as con:
        con.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    shutil.rmtree(os.path.join(CHAINS_DIR, str(entry_id)), ignore_errors=True)


def add_revision(entry_id, expected_return_pct, target_date, note):
    with _conn() as con:
        con.execute(
            "INSERT INTO revisions (entry_id, created_at, expected_return_pct, target_date, note) VALUES (?,?,?,?,?)",
            (entry_id, _ts(), expected_return_pct, target_date, note),
        )


def add_close(entry_id, closed_on, quantity, price, kind="sale", note=""):
    with _conn() as con:
        con.execute(
            "INSERT INTO closes (entry_id, recorded_at, closed_on, quantity, price, kind, note) VALUES (?,?,?,?,?,?,?)",
            (entry_id, _ts(), closed_on, quantity, price, kind, note),
        )


def set_reflection(entry_id, text):
    with _conn() as con:
        con.execute(
            "INSERT INTO reflections (entry_id, created_at, text) VALUES (?,?,?) "
            "ON CONFLICT(entry_id) DO UPDATE SET text = excluded.text",
            (entry_id, _ts(), text),
        )


def _rows(con, sql, ids):
    marks = ",".join("?" * len(ids))
    out = {}
    for r in con.execute(sql.format(marks=marks), ids):
        out.setdefault(r["entry_id"], []).append(dict(r))
    return out


def list_entries():
    """All entries, newest first, each with legs/tags/revisions/closes/reflection/
    snapshots attached and derived status fields (see _derive)."""
    with _conn() as con:
        entries = [dict(r) for r in con.execute("SELECT * FROM entries ORDER BY created_at DESC")]
        if not entries:
            return []
        ids = [e["id"] for e in entries]
        legs = _rows(con, "SELECT * FROM legs WHERE entry_id IN ({marks}) ORDER BY side", ids)
        tags = _rows(con, "SELECT * FROM entry_tags WHERE entry_id IN ({marks})", ids)
        revs = _rows(con, "SELECT * FROM revisions WHERE entry_id IN ({marks}) ORDER BY created_at", ids)
        closes = _rows(con, "SELECT * FROM closes WHERE entry_id IN ({marks}) ORDER BY closed_on, id", ids)
        refl = _rows(con, "SELECT * FROM reflections WHERE entry_id IN ({marks})", ids)
        snaps = _rows(con, "SELECT * FROM snapshots WHERE entry_id IN ({marks}) ORDER BY taken_at", ids)
    for e in entries:
        i = e["id"]
        e["legs"] = legs.get(i, [])
        e["tags"] = [t["tag"] for t in tags.get(i, [])]
        e["revisions"] = revs.get(i, [])
        e["closes"] = closes.get(i, [])
        e["reflection"] = (refl.get(i) or [{}])[0].get("text")
        e["snapshots"] = snaps.get(i, [])
        e["context"] = json.loads(e.pop("context_json") or "{}")
        _derive(e)
    return entries


def get_entry(entry_id):
    return next((e for e in list_entries() if e["id"] == entry_id), None)


def _derive(e):
    sales = [c for c in e["closes"] if c["kind"] == "sale"]
    sold = sum(c["quantity"] for c in sales)
    e["sales"] = sales
    e["target_mark"] = next((c for c in e["closes"] if c["kind"] == "target_mark"), None)
    e["open_qty"] = max(e["quantity"] - sold, 0.0)
    e["status"] = "closed" if e["open_qty"] <= 1e-9 else "open"
    created = datetime.fromisoformat(e["created_at"])
    e["edit_until"] = created + EDIT_WINDOW
    e["editable"] = now_et() < e["edit_until"]
    e["target_due"] = (e["status"] == "open" and e["target_mark"] is None
                       and _d(e["target_date"]) <= today_et())
    e["long_leg"] = next((l for l in e["legs"] if l["side"] == "long"), None)
    e["short_leg"] = next((l for l in e["legs"] if l["side"] == "short"), None)
    e["expiration"] = e["long_leg"]["expiration"] if e["long_leg"] else None
    e["expired"] = e["expiration"] is not None and _d(e["expiration"]) < today_et()
    e["description"] = describe(e)


def describe(e):
    t = e["ticker"]
    if e["trade_type"] == "stock":
        return f"{t} stock"
    lg, sh = e.get("long_leg"), e.get("short_leg")
    if e["trade_type"] == "spread" and lg and sh:
        return f"{t} {lg['expiration']} {lg['strike']:g}/{sh['strike']:g} call spread"
    if lg:
        return f"{t} {lg['expiration']} {lg['strike']:g} call"
    return t


def multiplier(e):
    return 1 if e["trade_type"] == "stock" else 100


# === Return math ===
def ret_pct(price, order_price):
    if price is None or not order_price:
        return None
    return (price / order_price - 1) * 100


def _blend(lots, order_price):
    """Quantity-weighted return of [(qty, price), ...] against one order price."""
    q = sum(x for x, _ in lots)
    if q <= 0:
        return None
    return ret_pct(sum(x * p for x, p in lots) / q, order_price)


def result(e):
    """Expected vs actual for a graded entry (fully closed, or target-date marked).
    Returns None while an entry is open with no target-date result yet."""
    mark = e["target_mark"]
    if mark is not None:
        # Target-date result: sales on/before the mark, plus the mark for the rest.
        lots = [(c["quantity"], c["price"]) for c in e["sales"] if c["id"] < mark["id"]]
        lots.append((mark["quantity"], mark["price"]))
        basis, exit_on = "target", mark["closed_on"]
    elif e["status"] == "closed":
        lots = [(c["quantity"], c["price"]) for c in e["sales"]]
        basis, exit_on = "closed", max(c["closed_on"] for c in e["sales"])
    else:
        return None

    actual = _blend(lots, e["order_price"])
    expected = e["expected_return_pct"]
    delta = actual - expected
    entry_on, target_on, exit_d = _d(e["created_at"]), _d(e["target_date"]), _d(exit_on)
    days_early = (target_on - exit_d).days
    if basis == "target":
        timing = "held to target date"
    elif days_early > 0:
        timing = f"closed {days_early} days before target"
    elif days_early < 0:
        timing = f"closed {-days_early} days after target"
    else:
        timing = "closed on target date"
    final = _blend([(c["quantity"], c["price"]) for c in e["sales"]], e["order_price"]) \
        if e["status"] == "closed" else None
    return {
        "expected": expected,
        "actual": actual,
        "delta": delta,
        "direction": "Overshoot" if delta > 0 else "Undershoot" if delta < 0 else "On target",
        "basis": basis,
        "exit_on": exit_on,
        "days_held": (exit_d - entry_on).days,
        "days_planned": (target_on - entry_on).days,
        "timing": timing,
        "final_realized": final,
    }


# === Live quotes ===
@st.cache_data(ttl=600, show_spinner=False)
def fetch_expirations(ticker):
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_calls(ticker, expiration):
    try:
        return yf.Ticker(ticker).option_chain(expiration).calls
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_spot(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return None if hist.empty else round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        return None


def quote_row(row):
    """{'mid','bid','ask','source'} for one chain row. Two-sided quote → Live mid;
    otherwise fall back to the last trade (common on illiquid index LEAPS)."""
    bid, ask, last = (float(row.get(c) or 0) for c in ("bid", "ask", "lastPrice"))
    if bid > 0 and ask > 0:
        return {"mid": round((bid + ask) / 2, 2), "bid": bid, "ask": ask, "source": "Live"}
    if last > 0:
        return {"mid": last, "bid": bid or None, "ask": ask or None, "source": "Last trade"}
    return None


def leg_quote(ticker, leg):
    calls = fetch_calls(ticker, leg["expiration"])
    if calls.empty:
        return None
    row = calls[np.isclose(calls["strike"], leg["strike"])]
    return None if row.empty else quote_row(row.iloc[0])


def current_value(e):
    """Current per-share value of the position: {'mid','bid','source'} or None.
    Spreads: mid = long mid − short mid; bid = long bid − short ask (what selling now would get)."""
    if e["trade_type"] == "stock":
        spot = fetch_spot(e["ticker"])
        return None if spot is None else {"mid": spot, "bid": spot, "source": "Live"}
    if e["expired"]:
        return None
    lq = leg_quote(e["ticker"], e["long_leg"])
    if lq is None:
        return None
    if e["trade_type"] == "call":
        return lq
    sq = leg_quote(e["ticker"], e["short_leg"])
    if sq is None:
        return None
    bid = (lq["bid"] - sq["ask"]) if lq["bid"] is not None and sq["ask"] is not None else None
    source = "Live" if lq["source"] == sq["source"] == "Live" else "Last trade"
    return {"mid": round(lq["mid"] - sq["mid"], 2),
            "bid": None if bid is None else round(max(bid, 0.0), 2), "source": source}


# === Chain snapshots ===
_SNAP_COLS = ["contractSymbol", "expiration", "strike", "bid", "ask", "lastPrice",
              "volume", "openInterest", "impliedVolatility", "lastTradeDate"]


def fetch_chain_snapshot(ticker):
    """Full call chain, every expiration. Returns (DataFrame, n_failed_expirations)."""
    exps = list(yf.Ticker(ticker).options)

    def one(exp):
        try:
            c = yf.Ticker(ticker).option_chain(exp).calls.copy()
            c["expiration"] = exp
            return c
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=SNAPSHOT_WORKERS) as pool:
        parts = list(pool.map(one, exps))
    ok = [p for p in parts if p is not None and not p.empty]
    if not ok:
        return pd.DataFrame(columns=_SNAP_COLS), len(exps)
    df = pd.concat(ok, ignore_index=True)
    return df[[c for c in _SNAP_COLS if c in df.columns]], len(exps) - len(ok)


def save_snapshot(entry_id, event, df):
    rel = os.path.join("chains", str(entry_id), f"{now_et():%Y%m%dT%H%M%S}_{event}.csv.gz")
    os.makedirs(os.path.join(JOURNAL_DIR, os.path.dirname(rel)), exist_ok=True)
    df.to_csv(os.path.join(JOURNAL_DIR, rel), index=False, compression="gzip")
    with _conn() as con:
        con.execute("INSERT INTO snapshots (entry_id, taken_at, event, path) VALUES (?,?,?,?)",
                    (entry_id, _ts(), event, rel))


def snapshot_entry(entry_id, ticker, event):
    """Fetch + save a chain snapshot. Returns a warning string, or None if all good."""
    try:
        df, failed = fetch_chain_snapshot(ticker)
    except Exception as ex:
        return f"Option chain snapshot failed: {ex}"
    if df.empty:
        return "Option chain snapshot came back empty, so the other-options comparison won't be available."
    save_snapshot(entry_id, event, df)
    if failed:
        return f"Option chain snapshot saved, but {failed} expiration(s) failed to download."
    return None


def load_snapshot(snap):
    return pd.read_csv(os.path.join(JOURNAL_DIR, snap["path"]))


# === Other-options comparison ===
@st.cache_data(ttl=3600, show_spinner=False)
def underlying_close_on(ticker, day):
    """Underlying close on `day` (or the last trading day before it)."""
    d = date.fromisoformat(day)
    try:
        hist = yf.Ticker(ticker).history(start=d - timedelta(days=10), end=d + timedelta(days=1))
    except Exception:
        return None
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])


def _two_sided_mid(df):
    df = df.copy()
    ok = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(ok, (df["bid"] + df["ask"]) / 2, np.nan)
    return df


def comparison(e):
    """Every call in the entry-time chain, priced entry→exit at mid. Exit = the
    target-date snapshot if there is one, else the close snapshot from the final sale.
    Contracts that expired before the exit date are valued at intrinsic on their
    expiration day. Returns (DataFrame, exit_on) or (None, reason)."""
    if e["trade_type"] == "stock":
        return None, "No options comparison for stock trades."
    res = result(e)
    if res is None:
        return None, "Available once the position is closed or has a target-date result."
    entry_snap = next((s for s in e["snapshots"] if s["event"] == "entry"), None)
    exit_event = "target" if res["basis"] == "target" else "close"
    exit_snap = next((s for s in reversed(e["snapshots"]) if s["event"] == exit_event), None)
    if entry_snap is None or exit_snap is None:
        return None, "Missing an option chain snapshot for this trade."

    a = _two_sided_mid(load_snapshot(entry_snap))[["expiration", "strike", "mid"]]
    a = a.dropna(subset=["mid"]).rename(columns={"mid": "entry_mid"})
    b = _two_sided_mid(load_snapshot(exit_snap))[["expiration", "strike", "mid"]]
    b = b.rename(columns={"mid": "exit_mid"})
    df = a.merge(b, on=["expiration", "strike"], how="left")

    exit_on = res["exit_on"]
    expired = df["expiration"] < exit_on
    for exp in df.loc[expired, "expiration"].unique():
        s = underlying_close_on(e["ticker"], exp)
        m = df["expiration"] == exp
        df.loc[m, "exit_mid"] = (s - df.loc[m, "strike"]).clip(lower=0) if s is not None else np.nan
    df["value_source"] = np.where(expired, "Intrinsic at expiry", "Quote")
    df["return_pct"] = (df["exit_mid"] / df["entry_mid"] - 1) * 100
    lg = e["long_leg"]
    df["yours"] = (df["expiration"] == lg["expiration"]) & np.isclose(df["strike"], lg["strike"])
    return df.dropna(subset=["return_pct"]).reset_index(drop=True), exit_on


# === Insights ===
def insights(entries):
    """(overall dict, by_tag DataFrame, by_type DataFrame) over graded entries.
    An entry with several tags counts once under each of its tags."""
    rows = []
    for e in entries:
        r = result(e)
        if r is None:
            continue
        for tag in e["tags"] or ["(untagged)"]:
            rows.append({"id": e["id"], "tag": tag, "type": TRADE_TYPE_LABELS[e["trade_type"]],
                         "expected": r["expected"], "actual": r["actual"], "delta": r["delta"]})
    if not rows:
        return None, pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    per_entry = df.drop_duplicates("id")

    def agg(g):
        return pd.Series({
            "Trades": len(g),
            "Avg expected %": g["expected"].mean(),
            "Avg actual %": g["actual"].mean(),
            "Avg delta (pp)": g["delta"].mean(),
            "Overshot": f"{(g['delta'] > 0).mean():.0%}",
            "Undershot": f"{(g['delta'] < 0).mean():.0%}",
        })

    overall = agg(per_entry).to_dict()
    by_tag = df.groupby("tag").apply(agg, include_groups=False).reset_index().rename(columns={"tag": "Tag"})
    by_type = per_entry.groupby("type").apply(agg, include_groups=False).reset_index().rename(columns={"type": "Trade type"})
    return overall, by_tag, by_type


# === Prefill from other pages ("Log this trade") ===
PREFILL_KEY = "journal_prefill"
JOURNAL_PAGE = "pages/Trade_Journal.py"


def log_trade_button(label, key, *, ticker, expiration=None, strike=None, context=None, source_page=None):
    """Button that opens the Trade Journal's log form pre-filled with this contract."""
    if st.button(label, key=key, icon=":material/edit_note:"):
        st.session_state[PREFILL_KEY] = {
            "ticker": ticker, "expiration": expiration,
            "strike": None if strike is None else float(strike),
            "context": context or {}, "source_page": source_page,
        }
        st.switch_page(JOURNAL_PAGE)


def market_context():
    """Context captured on every logged trade."""
    vix = utils.fetch_vix()
    return {"vix": vix} if vix else {}
