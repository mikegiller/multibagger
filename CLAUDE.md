# Multibagger — Options Analysis Hub

## Purpose
Personal Streamlit app for finding "multibagger" opportunities (Peter Lynch's term:
investments likely to return multiples of the original stake). The owner mostly
**buys long-dated calls (LEAPS)**, often holds close to expiration, and doesn't
roll — they sell and buy more instead.

The app is **informational only**: it presents data the way the owner thinks about
it. All decisions — whether to invest, which recommendation to follow, position
size — are made by the owner. Don't build features that auto-trade or make the
decision for them. Position sizing is undecided (may or may not be added).

## Audience & hosting
- Maintained and used by the owner; two close friends have access but rarely use it.
- Runs locally, private. Never to be made public or deployed publicly.

## Stack
- Streamlit multipage app: entry point `MainDashboard.py`, pages in `pages/`.
- Data: `yfinance` (considered good enough — don't propose paid data sources).
- `plotly` charts, `streamlit-aggrid` grids, `openpyxl` Excel export, `scipy` for
  Black-Scholes / implied vol.
- Gemini AI analysis is implemented (`utils.gemini_*`) but the owner rarely uses it.
  Keep it working; don't invest in expanding it unless asked.
- Run: `source venv/bin/activate && streamlit run MainDashboard.py`

## Layout
- `master_plan.py` — shared Master Plan logic (trend projection → optimum call strike
  per expiration). Thin wrappers: `pages/Master_Plan_Stocks.py`, `pages/Master_Plan_Index.py`.
- `options_explorer.py` — shared Options Data Explorer (LEAPS returns under 5/10/15%
  annual growth). Wrappers: `pages/Options_Data_Explorer_{Stocks,Index}.py`.
- `chart_utils.py` — swings, linear regression, trendlines, MAs, Fibonacci, Bollinger,
  support/resistance, projection table, chart builder.
- `option_pricing.py` — Black-Scholes, implied vol, `fill_illiquid_strikes`
  (interpolates IV between live-quoted strikes; never extrapolates).
- `utils.py` — yfinance cache workaround (import it before using yfinance), favorites,
  ticker input, index detection, VIX badge, Gemini helpers.
- `favorites.json` — `{"stocks": [...], "index": [...]}` quick-select tickers.
- Other pages: Chart Pattern Analyzer, Buy vs Sell Pressure, Capital Reallocation,
  Vertical Call/Put Spread finders (less used — keep them; don't remove pages without asking).

## Conventions
- Stocks/ETFs and Indexes (^SPX, ^NDX, ^OEX) are separate reports sharing one module,
  parameterized by `asset_class`. Index chains are illiquid at the LEAPS end: they get a
  lastPrice fallback and IV gap-filling; stocks require two-sided quotes.
- Estimated (filled) prices are flagged and never chosen as the "optimum strike".
  Stale quotes are excluded by recency (`RECENCY_TRADING_DAYS`).
- Use `width='stretch'` and not the deprecated `use_container_width`.
- Use the `developing-with-streamlit` skill for Streamlit work.
