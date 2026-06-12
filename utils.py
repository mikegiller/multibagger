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
DEFAULT_FAVORITES = ["SPY", "QQQ", "VOO", "XLK", "NVDA"]


# === Favorites ===
def load_favorites():
    try:
        with open(os.path.join(_ROOT, "favorites.json")) as f:
            return json.load(f)
    except Exception:
        return DEFAULT_FAVORITES


def favorites_html(favs, extra_params=""):
    """Markdown/HTML line of ?fav= links for st.markdown(..., unsafe_allow_html=True)."""
    links = ", ".join(
        f'<a href="?fav={f}{extra_params}" target="_self" style="text-decoration:none">{f}</a>'
        for f in favs
    )
    return f"**Favorites:** {links}"


# === App-wide Active Ticker ===
def set_active_ticker(t):
    st.session_state.active_ticker = t


def ticker_input(widget_key, label="Ticker", **kwargs):
    """Ticker text input backed by the app-wide active ticker.

    Streamlit drops widget-keyed state when a page isn't rendered, so each page
    keeps its own widget key and seeds it from the shared `active_ticker` —
    pick a ticker on one tool and every other tool opens with it.
    """
    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = "SPY"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.active_ticker
    t = st.text_input(label, key=widget_key, **kwargs).upper().strip()
    if t:
        st.session_state.active_ticker = t
    return t


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
