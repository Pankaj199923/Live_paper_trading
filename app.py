# ============================================================
# app.py  —  QuantDesk Pro | Main Entry Point
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pytz
import os
from datetime import datetime, time as dtime
from streamlit_autorefresh import st_autorefresh
import requests

# ── Module imports ──────────────────────────────────────────
from config import (ACCESS_TOKEN, IST, now_ist, now_ist_dt, MARKET_OPEN,
                    session_label, session_color, df_indices, INDEX_SHORT,
                    LOT_SIZES, BASE_DIR, TRADE_FILE, TODAY_TRADES_FILE,
                    CLOSED_POS_FILE, AI_LOG_FILE, SNAPSHOT_DIR, instrument_df)
from utils import (load_csv_as_list, get_atm_strike, pnl_color, pnl_badge,
                   compute_grand_total, calculate_net_book)
from api import fetch_ltp

# ── Tab modules ──────────────────────────────────────────────
import tab1_option_chain
import tab2_smart_money
import tab3_positions
import tab4_snr_pain
import tab5_ai_advisor
import tab6_trade_log
import tab8_stocks
import tab9_chart
# import tab10_history
import tab11_basket
import tab12_paper_trading
# import tab13_WIRING_PATCH

st.set_page_config(
    page_title="QuantDesk Pro | Options Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS — Bloomberg-inspired pro desk ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Barlow+Condensed:wght@300;400;500;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
  --bg-primary:   #070b0f;
  --bg-secondary: #0d1117;
  --bg-card:      #111920;
  --bg-raised:    #162030;
  --border:       #1e3040;
  --border-bright:#2a4560;
  --accent-orange:#ff8c00;
  --accent-cyan:  #00d4ff;
  --accent-green: #00e676;
  --accent-red:   #ff3d57;
  --accent-yellow:#ffd600;
  --accent-purple:#c084fc;
  --text-primary: #e8f4ff;
  --text-secondary:#7fa8c8;
  --text-dim:     #3a6080;
  --font-mono:    'JetBrains Mono', monospace;
  --font-display: 'Barlow Condensed', sans-serif;
  --font-body:    'Barlow', sans-serif;
}

/* ── Base ── */
html, body, .stApp {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0.5rem 1rem 2rem !important; max-width: 100% !important; }
.stDeployButton { display: none; }

/* ── Tabs — Bloomberg style ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-secondary) !important;
  border-bottom: 1px solid var(--border-bright) !important;
  gap: 0 !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-display) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  padding: 10px 20px !important;
  border: none !important;
  border-right: 1px solid var(--border) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--bg-card) !important;
  color: var(--accent-orange) !important;
  border-bottom: 2px solid var(--accent-orange) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--bg-primary) !important;
  padding: 12px 0 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: 10px 14px !important;
  border-left: 3px solid var(--accent-orange) !important;
}
[data-testid="metric-container"] label {
  font-family: var(--font-display) !important;
  font-size: 10px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: var(--text-secondary) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
}

/* ── Dataframes ── */
.stDataFrame, [data-testid="stDataFrame"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
}
iframe[title="st_aggrid"] { background: var(--bg-card) !important; }

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--accent-orange) !important;
  color: var(--accent-orange) !important;
  font-family: var(--font-display) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  border-radius: 2px !important;
  transition: all 0.15s !important;
  padding: 6px 18px !important;
}
.stButton > button:hover {
  background: var(--accent-orange) !important;
  color: #000 !important;
  box-shadow: 0 0 15px rgba(255,140,0,0.4) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent-orange) !important;
  color: #000 !important;
  font-weight: 700 !important;
}

/* ── Select / Input ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-bright) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 13px !important;
  border-radius: 3px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border-bright) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-display) !important;
  font-size: 12px !important;
  letter-spacing: 1px !important;
  border-radius: 3px !important;
}
.streamlit-expanderContent {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
}

/* ── Alerts ── */
.stSuccess > div { background: #051a0e !important; border-left: 3px solid var(--accent-green) !important; border-radius: 2px !important; }
.stError > div   { background: #1a0508 !important; border-left: 3px solid var(--accent-red) !important; border-radius: 2px !important; }
.stWarning > div { background: #1a1000 !important; border-left: 3px solid var(--accent-yellow) !important; border-radius: 2px !important; }
.stInfo > div    { background: #00101a !important; border-left: 3px solid var(--accent-cyan) !important; border-radius: 2px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-orange); }

/* ── Toggle ── */
.stToggle label { font-family: var(--font-display) !important; letter-spacing: 0.5px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 12px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ======================

# ======================
# 🔄 AUTO REFRESH — only fires outside chart tab
# ======================
# st_autorefresh triggers a full Streamlit rerun every N ms.
# Chart tab uses st.fragment to isolate itself from global reruns.
_active_tab = st.session_state.get("active_tab_key", "OPTION CHAIN")
_skip_refresh = _active_tab in ("📊 CHART",)
if not _skip_refresh:
    st_autorefresh(interval=5000, key="datarefresh")
else:
    # Still refresh header/spot data but skip expensive full rerun
    st_autorefresh(interval=30000, key="datarefresh_slow")


# ======================
# 🗃️ SESSION STATE
# ======================
defaults = {
    "executed_trades":    lambda: load_csv_as_list(TRADE_FILE),
    "today_trades":       lambda: load_csv_as_list(TODAY_TRADES_FILE),
    "closed_positions":   lambda: load_csv_as_list(CLOSED_POS_FILE),
    "ai_trade_log":       lambda: load_csv_as_list(AI_LOG_FILE),
    "selected_index":     lambda: df_indices['index'].tolist()[0],
    "atm_strike":         lambda: None,
    "last_ai_run":        lambda: now_ist_dt,
    "last_ai_trade_time": lambda: None,
    "prev_spot_price":    lambda: 0,
    "prev_pcr":           lambda: 1.0,
    "prev_net_gex":       lambda: 0,
    "pcr_history":        lambda: [],
    "spot_history":       lambda: [],
    "option_data":        lambda: {},
    "daily_loss_limit":   lambda: -10000,
    "max_open_positions": lambda: 6,
    "greeks_history":     lambda: [],
    "vix_history":        lambda: [],
    "pnl_history":        lambda: [],
    "alert_log":          lambda: [],
    "last_snapshot_time": lambda: None,
    "strategy_builder":   lambda: [],
    "header_prev_spots":  lambda: {},
}
for key, factory in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = factory()


# ======================
# ── PRO HEADER BAR ──
# ======================
def render_header():
    # ── Always fetch live prices for ALL indices via API (cached ttl=5s) ──
    spot_vals = {}

    try:
        all_keys = ",".join(df_indices['index'].tolist())
        r = requests.get(
            "https://api.upstox.com/v3/market-quote/ltp",
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            params={"instrument_key": all_keys},
            timeout=5
        )
        data_h = r.json().get("data", {})
        for raw_key, v in data_h.items():
            clean = raw_key.replace("%7C", "|").replace("%7c", "|")
            price = v.get("last_price", 0)
            if price > 0:
                # Direct key match
                if clean in INDEX_SHORT:
                    spot_vals[clean] = price
                # Partial key match (Upstox sometimes returns modified keys)
                for idx_key in INDEX_SHORT:
                    if idx_key.split("|")[-1].lower() in clean.lower():
                        spot_vals[idx_key] = price
                        # Also keep session state in sync for other tabs
                        st.session_state[f"spot_{idx_key}"] = price
    except Exception:
        pass

    # ── Fallback: use session state if API call failed ────────────────────
    if not spot_vals:
        for idx_key in INDEX_SHORT:
            cached_spot = st.session_state.get(f"spot_{idx_key}", 0)
            if cached_spot > 0:
                spot_vals[idx_key] = cached_spot

    # ── Persist prev spots for change % across refreshes ────────────────
    prev_spots = st.session_state.get("header_prev_spots", {})
    if spot_vals:
        updated = {}
        for k in INDEX_SHORT:
            updated[k] = spot_vals.get(k, 0) or prev_spots.get(k, 0)
        st.session_state["header_prev_spots"] = updated

    # ── Build ticker blocks ───────────────────────────────────────────────
    ticker_blocks = []
    for idx_key, idx_short_name in INDEX_SHORT.items():
        price   = spot_vals.get(idx_key, 0)
        prev    = prev_spots.get(idx_key, price)
        chg     = (price - prev) if (price > 0 and prev > 0) else 0
        chg_pct = (chg / prev * 100) if prev > 0 else 0
        col     = "#00e676" if chg >= 0 else "#ff3d57"
        arrow   = "&#9650;" if chg >= 0 else "&#9660;"
        disp    = f"{price:,.0f}" if price > 0 else "&mdash;"
        chg_str = f"{arrow} {abs(chg_pct):.2f}%" if price > 0 else "&mdash;"
        ticker_blocks.append(
            f'<div style="display:flex;flex-direction:column;padding:0 18px;border-right:1px solid #1e3040;min-width:110px;">'
            f'<span style="font-family:Barlow Condensed,sans-serif;font-size:10px;letter-spacing:1.5px;color:#7fa8c8;font-weight:600;">{idx_short_name}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;color:#e8f4ff;">{disp}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{col};">{chg_str}</span>'
            f'</div>'
        )
    tickers_html = "".join(ticker_blocks)

    now_str = datetime.now(IST).strftime("%d %b %Y  %H:%M:%S")
    header_html = (
        '<div style="background:linear-gradient(90deg,#0d1117,#111920);border-bottom:1px solid #2a4560;'
        'border-left:3px solid #ff8c00;padding:8px 0;display:flex;align-items:center;'
        'justify-content:space-between;margin:-0.5rem -1rem 12px -1rem;">'
        '<div style="display:flex;align-items:center;">'
        '<div style="padding:0 20px;border-right:1px solid #1e3040;">'
        '<span style="font-family:Barlow Condensed,sans-serif;font-size:20px;font-weight:800;letter-spacing:2px;color:#ff8c00;">QUANTDESK</span>'
        '<span style="font-family:Barlow Condensed,sans-serif;font-size:12px;letter-spacing:3px;color:#7fa8c8;margin-left:6px;">PRO</span>'
        '</div>'
        + tickers_html +
        '</div>'
        f'<div style="padding:0 20px;text-align:right;">'
        f'<div style="font-size:10px;letter-spacing:1px;color:#7fa8c8;font-family:Barlow Condensed,sans-serif;">SESSION</div>'
        f'<div style="font-size:14px;font-weight:700;color:{session_color};letter-spacing:2px;">{session_label}</div>'
        f'<div style="font-size:11px;color:#3a6080;">{now_str} IST</div>'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)



# ======================
# 📑 MAIN TABS
# ======================
render_header()

tab1, tab2, tab3, tab4, tab5, tab6, tab8, tab9, tab11, tab12 = st.tabs([
    "OPTION CHAIN",
    "SMART MONEY",
    "POSITIONS",
    "S&R / PAIN",
    "AI ADVISOR",
    "TRADE LOG",
    "\U0001f5a5 STOCKS",
    "\U0001f4ca CHART",
    # "\U0001f4f8 HISTORY",
    "\U0001f9fa BASKET",
    "\U0001f3ae PAPER_Trade",
    # "\U0001f3ae FVG LOG",
])

with tab1:  tab1_option_chain.render()
with tab2:  tab2_smart_money.render()
with tab3:  tab3_positions.render()
with tab4:  tab4_snr_pain.render()
with tab5:  tab5_ai_advisor.render()
with tab6:  tab6_trade_log.render()
# with tab7:  tab7_greeks_lab.render()
with tab8:  tab8_stocks.render()
with tab9:  tab9_chart.render()
# with tab10: tab10_history.render()
with tab11: tab11_basket.render()
with tab12: tab12_paper_trading.render()

