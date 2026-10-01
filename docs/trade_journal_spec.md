# Trade Journal — Spec

Status: **implemented** (`journal.py`, `pages/Trade_Journal.py`). Agreed in conversation on 2026-10-01.

## Purpose
A local log of every trade the owner places: what it is, why, and what return is
expected by when. Over time it shows how accurate the owner's thinking and
predictions are, and how the chosen contract compared to the other options
available that day.

Guiding principles (from the owner):
- **Numbers, not scores.** No grades or hit/miss. Show expected vs actual and the delta.
- **No regret tracking.** Nothing is tracked after a position is sold.
- **No AI analysis.** Numbers plus the owner's own written reflection.
- **Honest records.** The original prediction is locked once logged.
- Single user (the owner). Friends can view; no authentication.

## Scope
Trade types: **calls** (any expiry, LEAPS mostly), **vertical spreads**, **stock**.
Trades are logged going forward only — no backfill. Fees and commissions are ignored.

---

## 1. Logging a trade

The owner logs a trade **during market hours, right after placing it**, with the app open.

Fields:
| Field | Notes |
|---|---|
| Ticker | Stocks/ETFs or index (^SPX etc.) |
| Trade type | call · spread · stock |
| Legs | call: strike + expiration · spread: long and short legs · stock: none |
| Quantity | contracts (×100 multiplier) or shares |
| Order price | entered manually; per contract (or net debit for spreads) or per share |
| Expected return % | return **on the position itself** (e.g. +150% on the call) |
| Target date | the date by which that return is expected (one target per trade) |
| Thesis | free text — the reasoning |
| Tags | one or more from the fixed list in `journal_config.json` |

Captured automatically at logging time:
- Timestamp (US Eastern) and underlying price.
- **Option chain snapshot**: full call chain, every expiration, for the ticker
  (strike, bid, ask, last, volume, OI, IV, last trade date). Used for the
  "other options" comparison.
- **Context snapshot** when logged from Master Plan / Options Data Explorer: the
  on-screen trend projection, optimum strike, VIX, and similar indicators.

### "Log this trade" buttons
Master Plan (Stocks/Index) and Options Data Explorer (Stocks/Index) get a
**Log this trade** button that pre-fills ticker, contract, and context, then opens
the Trade Journal page's Log tab via `st.switch_page` and session state.

### Locking
- The prediction fields (expected return %, target date, thesis) are locked once
  saved. A **24-hour edit window** allows typo fixes, then they become read-only.
- If the owner's view changes, they add a dated **revision** (new expected %,
  new target date, note). Results are always measured against the **original**
  prediction. Revisions are displayed alongside the original.

### Adds
Buying more of something already held is a **new, separate journal entry** with
its own thesis, target, and chain snapshot.

---

## 2. While open

The Open tab lists open entries with:
- Order price, current price, current return % (live from yfinance).
  - Calls/spreads: show return at **mid** and at **bid** (LEAPS spreads are wide).
  - Spreads: current value = long-leg price − short-leg price.
  - With no two-sided quote (common on illiquid index LEAPS), the last trade price
    is used and flagged "Last trade". Black-Scholes gap-filled estimates are not used here.
- Expected return %, target date, days remaining.
- Quantity still open (after partial closes).

No daily snapshots or price history between logging and closing.

### Partial and full closes
- The owner records a close: date, quantity, price, optional note.
- An entry can be closed in **several parts**. Each part's return is computed
  against that entry's order price; the entry's overall return is the
  quantity-weighted blend.
- When the last part is closed, the owner is asked for a short **reflection**
  (what went right or wrong).
- Expired contracts: the owner enters the final sale or settlement price manually.
- At each close, the full call chain is snapshotted again (used for the comparison).

### Target date reached
When an open entry's target date has passed, the journal page shows a
**"targets due"** banner, and each due entry offers two choices:
1. **"I sold at $X"** → records a close.
2. **"Still holding"** → records the current market value as the
   **target-date result** (plus a chain snapshot). The entry stays open and keeps
   tracking until sold.

---

## 3. Results (closed entries and target-date results)

For each entry:
| Metric | Definition |
|---|---|
| Expected return | original locked % |
| Actual return | blended realized return (plus the target-date mark for any unsold quantity) |
| Delta | actual − expected, in percentage points |
| Direction | **Overshoot** (beat the expectation) or **Undershoot** (expectation too high) |
| Timing | days held vs days planned (e.g. "closed 152 days before target") |

If fully sold before the target date, the realized result is compared to the
expectation as-is, with the timing difference shown. Nothing after the sale is tracked.

### Other options comparison
Using the entry-time chain snapshot and the close-time (or target-date) snapshot:
a grid of every call strike × expiration showing what each would have returned
over the **same period**, with the owner's contract highlighted.
- Entry and exit priced at mid for every contract, including the owner's own, so
  they compare like for like.
- Contracts that expired in between: value = intrinsic value at the underlying's
  close on expiration day.
- Strikes with no valid quote, or with estimated prices, are shown as blank or flagged.
- Spreads: compare the long leg against the other calls.
- Stocks: no options comparison, and no chain snapshot is taken for stock trades.

### Insights tab (aggregates)
Grouped by **tag**, and also by trade type:
- number of trades, average expected %, average actual %, average delta,
  share of trades that overshot vs undershot.
- Overall: average expected vs average actual (shows whether expectations run
  high or low).

---

## 4. Storage

```
<JOURNAL_DIR>/
  journal.db            # SQLite
  chains/<entry_id>/<timestamp>_<event>.csv.gz   # entry, close, target-date snapshots (no new dependency)
```
- `JOURNAL_DIR` comes from the env var `MULTIBAGGER_JOURNAL_DIR`; default `./journal`.
- `journal/` is **gitignored**. Trade data never goes to GitHub.
- All timestamps are stored in US Eastern market time.
- `journal_config.json` (tracked in git) holds the editable tag list.

Starter tags: trend continuation, dip buy, earnings, sector rotation, macro,
breakout, mean reversion.

### Tables (sketch)
- `entries`: id, created_at, ticker, asset_class, trade_type, quantity,
  order_price, underlying_price, expected_return_pct, target_date, thesis,
  locked_at, context_json, source_page
- `legs`: entry_id, side (long/short), strike, expiration, contract_symbol
- `entry_tags`: entry_id, tag
- `revisions`: entry_id, created_at, expected_return_pct, target_date, note
- `closes`: entry_id, closed_at, quantity, price, kind (sale | target_mark), note
- `reflections`: entry_id, created_at, text
- `snapshots`: entry_id, taken_at, event (entry | close | target), path

---

## 5. Screens
New page `pages/Trade_Journal.py` (logic in a shared `journal.py` module), with tabs:
1. **Log Trade**: the form (pre-fillable from other pages).
2. **Open**: live positions, close / partial-close actions, targets-due banner.
3. **Closed**: per-entry results, delta, timing, reflection, other-options grid.
4. **Insights**: aggregates by tag and type.

A link to the journal is added on `MainDashboard.py`.

---

## 6. Deployment (Ubuntu home server `kamrui`)

The app runs as the system service `multibagger.service` as user `mgiller`, from
`/home/mgiller/apps/multibagger` (Streamlit on 127.0.0.1:8510). It is reached over
Tailscale and updated with `git pull`. **Journaling starts directly on the server**,
so no data moves from the Mac. The Mac copy is for development only, and its local
`journal/` holds test data.

The journal data lives **outside the repo directory**, so `git` operations can
never touch it.

```bash
# 1. Pull the code
cd ~/apps/multibagger && git pull
venv/bin/pip install -r requirements.txt   # only if requirements changed

# 2. Create the data directory and the backup directory on the external USB drive
mkdir -p ~/apps/multibagger-data/journal
mkdir -p /mnt/seagate/multibagger-backups

# 3. Point the service at it with a systemd drop-in (leaves the main unit file untouched)
sudo systemctl edit multibagger.service
#   In the editor, add:
#     [Service]
#     Environment=MULTIBAGGER_JOURNAL_DIR=/home/mgiller/apps/multibagger-data/journal

# 4. Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart multibagger.service
systemctl show multibagger.service -p Environment   # verify the variable is set

# 5. Nightly backup to the external USB drive (crontab -e as mgiller), 02:30, keep 30 days
#   needs: sudo apt install sqlite3
#   `mountpoint -q` skips the backup if the drive isn't mounted, rather than silently
#   writing into the empty /mnt/seagate folder on the system disk.
30 2 * * * mountpoint -q /mnt/seagate && cd /home/mgiller/apps/multibagger-data && sqlite3 journal/journal.db ".backup /mnt/seagate/multibagger-backups/journal-$(date +\%F).db" && tar czf /mnt/seagate/multibagger-backups/chains-$(date +\%F).tgz -C journal chains && find /mnt/seagate/multibagger-backups -mtime +30 -delete
```

Backups go to the external USB drive at `/mnt/seagate`, so a failure of the
server's own disk doesn't lose the journal.
