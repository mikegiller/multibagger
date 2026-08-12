# utils.py — shared helpers for all pages
import json
import os

import streamlit as st
import yfinance.cache as _yfc

# yfinance's tz cache can break if its dir was deleted; replace with dummy to bypass SQLite
os.makedirs(os.path.join(os.path.expanduser("~"), ".cache", "py-yfinance"), exist_ok=True)
_yfc._TzCacheManager._tz_cache = _yfc._TzCacheDummy()
_yfc._CookieCacheManager._Cookie_cache = _yfc._CookieCacheDummy()

# Optional: Google Gemini (only needed for AI analysis)
try:
    from google import genai as _genai
    GEMINI_AVAILABLE = True
except ImportError:
    _genai = None
    GEMINI_AVAILABLE = False

GEMINI_MODEL = "gemini-2.5-flash"

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FAVORITES = {
    "stocks": ["SPY", "QQQ", "VOO", "XLK", "NVDA"],
    "index": ["^SPX", "^NDX", "^OEX"],
}


# === Favorites ===
def load_favorites(category="stocks"):
    """Favorites for a category ("stocks" or "index"), backed by favorites.json's
    {"stocks": [...], "index": [...]} structure."""
    try:
        with open(os.path.join(_ROOT, "favorites.json")) as f:
            data = json.load(f)
        return data[category]
    except Exception:
        return DEFAULT_FAVORITES.get(category, [])


def favorites_html(favs, extra_params=""):
    """Markdown/HTML line of ?fav= links for st.markdown(..., unsafe_allow_html=True)."""
    links = ", ".join(
        f'<a href="?fav={f}{extra_params}" target="_self" style="text-decoration:none">{f}</a>'
        for f in favs
    )
    return f"**Favorites:** {links}"


# === Active Ticker (tracked separately per asset category) ===
DEFAULT_ACTIVE_TICKER = {"stocks": "SPY", "index": "^SPX"}


def set_active_ticker(t, category="stocks"):
    st.session_state[f"active_ticker_{category}"] = t


def ticker_input(widget_key, category="stocks", label="Ticker", **kwargs):
    """Ticker text input backed by the category's active ticker.

    Streamlit drops widget-keyed state when a page isn't rendered, so each page
    keeps its own widget key and seeds it from the shared `active_ticker_<category>` —
    pick a ticker on one stock tool and every other stock tool opens with it,
    independently of whatever ticker is active on the index tools (and vice versa).
    """
    active_key = f"active_ticker_{category}"
    if active_key not in st.session_state:
        st.session_state[active_key] = DEFAULT_ACTIVE_TICKER.get(category, "SPY")
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[active_key]
    t = st.text_input(label, key=widget_key, **kwargs).upper().strip()
    if t:
        st.session_state[active_key] = t
    return t


def is_index_ticker(ticker):
    """Yahoo Finance indexes are prefixed with '^' (e.g. ^SPX, ^NDX, ^VIX)."""
    return ticker.startswith("^")


def warn_if_wrong_report(ticker, asset_class, stocks_page, index_page):
    """On the wrong-asset-class report, warn and link to the right one.

    asset_class is the report's own class ("stocks" or "index"); the check fires
    when the entered ticker looks like the *other* class.
    """
    if not ticker:
        return
    ticker_is_index = is_index_ticker(ticker)
    if asset_class == "index" and not ticker_is_index:
        st.warning(f"⚠️ **{ticker}** looks like a stock or ETF ticker, not an index. "
                   "This report is tuned for indexes (illiquid LEAPS handling, index favorites).")
        st.page_link(stocks_page, label="Use the Stocks & ETFs report instead", icon="📈")
    elif asset_class == "stocks" and ticker_is_index:
        st.warning(f"⚠️ **{ticker}** looks like an index ticker, not a stock or ETF. "
                   "For indexes, use the report below — it correctly handles illiquid index LEAPS.")
        st.page_link(index_page, label="Use the Index report instead", icon="📈")


# === Gemini AI helpers ===
def gemini_sidebar():
    """Render the Gemini API-key block in the sidebar; return a client or None."""
    with st.sidebar:
        st.header("🤖 AI Analysis Settings")
        if not GEMINI_AVAILABLE:
            st.warning("⚠️ Google Gemini not installed")
            st.code("pip install google-genai", language="bash")
            return None

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get free API key at: https://aistudio.google.com/app/apikey",
            key="gemini_api_key",
        )
        client = None
        if api_key:
            try:
                client = _genai.Client(api_key=api_key)
                st.success("✅ Gemini API configured")
            except Exception as e:
                st.error(f"❌ API configuration failed: {e}")
        else:
            st.info("💡 Add API key to enable AI analysis")

        st.markdown("---")
        st.caption("**Free Tier**: 1,500 requests/day")
        st.caption("[Get API Key →](https://aistudio.google.com/app/apikey)")
        return client


def gemini_generate(client, prompt):
    """One-shot generation; returns the response text."""
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text


def gemini_chat_reply(client, initial_context, chat_history, follow_up):
    """Continue a conversation: initial analysis context + prior messages + follow-up."""
    history = [{"role": "user", "parts": [{"text": initial_context}]}]
    for msg in chat_history:
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["content"]}],
        })
    chat = client.chats.create(model=GEMINI_MODEL, history=history)
    return chat.send_message(follow_up).text
