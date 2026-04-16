# ============================================================
# app.py  —  QuantDesk Pro | Main Entry Point
# FIXES: Dynamic session label, circular refresh timer, sticky header+tabs
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
import tab7_greeks_lab
import tab8_stocks
import tab9_chart
import tab10_history
import tab11_basket
import tab12_paper_trading

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
.block-container { padding: 0rem 1rem 2rem !important; max-width: 100% !important; }
.stDeployButton { display: none; }

/* ═══════════════════════════════════════════
   STICKY HEADER + TABS
   ═══════════════════════════════════════════ */

/* Make the sticky wrapper stick to top */
.sticky-topbar {
  position: sticky !important;
  top: 0 !important;
  z-index: 9999 !important;
  background: var(--bg-primary) !important;
}

/* Streamlit's tab list — make it sticky */
.stTabs [data-baseweb="tab-list"] {
  position: sticky !important;
  top: 0 !important;
  z-index: 9998 !important;
  background: var(--bg-secondary) !important;
  border-bottom: 1px solid var(--border-bright) !important;
  gap: 0 !important;
  padding: 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-display) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  padding: 10px 18px !important;
  border: none !important;
  border-right: 1px solid var(--border) !important;
  white-space: nowrap !important;
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

/* ═══════════════════════════════════════════
   CIRCULAR REFRESH RING
   ═══════════════════════════════════════════ */
@keyframes ring-countdown {
  0%   { stroke-dasharray: 100 0;  }
  100% { stroke-dasharray: 0   100; }
}
@keyframes ring-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.6; }
}
.refresh-ring-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 16px;
}
.refresh-ring {
  position: relative;
  width: 34px;
  height: 34px;
  flex-shrink: 0;
}
.refresh-ring svg {
  transform: rotate(-90deg);
  width: 34px;
  height: 34px;
}
.refresh-ring .track {
  fill: none;
  stroke: #1e3040;
  stroke-width: 3;
}
.refresh-ring .progress {
  fill: none;
  stroke: #ff8c00;
  stroke-width: 3;
  stroke-linecap: round;
  stroke-dasharray: 100 0;
  animation: ring-countdown 5s linear infinite;
  transition: stroke 0.3s;
}
.refresh-ring .center-dot {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-family: 'JetBrains Mono', monospace;
  font-size: 8px;
  font-weight: 700;
  color: #ff8c00;
  pointer-events: none;
  animation: ring-pulse 5s linear infinite;
}
.refresh-label {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 9px;
  letter-spacing: 1.5px;
  color: #3a6080;
  text-transform: uppercase;
  line-height: 1.2;
}
.refresh-label span {
  display: block;
  font-size: 10px;
  color: #ff8c00;
  font-weight: 700;
}

/* ═══════════════════════════════════════════
   SESSION BADGE
   ═══════════════════════════════════════════ */
.session-live {
  animation: session-glow 2s ease-in-out infinite;
}
@keyframes session-glow {
  0%, 100% { text-shadow: 0 0 8px rgba(0,230,118,0.6); }
  50%       { text-shadow: 0 0 20px rgba(0,230,118,1); }
}
.session-pre {
  animation: session-amber 3s ease-in-out infinite;
}
@keyframes session-amber {
  0%, 100% { text-shadow: 0 0 6px rgba(255,214,0,0.4); }
  50%       { text-shadow: 0 0 14px rgba(255,214,0,0.9); }
}
</style>
""", unsafe_allow_html=True)


# ======================
# 🔄 AUTO REFRESH
# ======================
_active_tab = st.session_state.get("active_tab_key", "OPTION CHAIN")
_skip_refresh = _active_tab in ("📊 CHART",)
if not _skip_refresh:
    st_autorefresh(interval=5000, key="datarefresh")
else:
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
    "_refresh_count":     lambda: 0,
}
for key, factory in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = factory()

# Increment refresh counter for tracking
st.session_state["_refresh_count"] = st.session_state.get("_refresh_count", 0) + 1


# ======================
# 🕒 DYNAMIC SESSION LABEL — Recomputed every render (fixes PRE-MARKET bug)
# ======================
def get_live_session():
    """Always compute current session from real-time IST clock."""
    _now = datetime.now(IST).time()
    _OPEN  = dtime(9, 15)   # NSE actual open
    _CLOSE = dtime(15, 30)
    _PRE   = dtime(8, 59)   # Pre-market OI data available

    if _now < _PRE:
        return "CLOSED",     "#3a6080",  False
    elif _now < _OPEN:
        return "PRE-MARKET", "#ffd600",  False
    elif _now <= _CLOSE:
        return "LIVE",       "#00e676",  True
    else:
        return "POST-MARKET","#7fa8c8",  False


# ======================
# ── PRO HEADER BAR ──
# ======================
def render_header():
    # Live session — always fresh
    live_label, live_color, is_market_open = get_live_session()
    now_str = datetime.now(IST).strftime("%d %b %Y  %H:%M:%S")

    # ── Fetch spot prices ──────────────────────────────────────────
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
                if clean in INDEX_SHORT:
                    spot_vals[clean] = price
                for idx_key in INDEX_SHORT:
                    if idx_key.split("|")[-1].lower() in clean.lower():
                        spot_vals[idx_key] = price
                        st.session_state[f"spot_{idx_key}"] = price
    except Exception:
        pass

    # Fallback to session cache
    if not spot_vals:
        for idx_key in INDEX_SHORT:
            cached = st.session_state.get(f"spot_{idx_key}", 0)
            if cached > 0:
                spot_vals[idx_key] = cached

    # Track prev spots for change %
    prev_spots = st.session_state.get("header_prev_spots", {})
    updated = {}
    for k in INDEX_SHORT:
        updated[k] = spot_vals.get(k, 0) or prev_spots.get(k, 0)
    st.session_state["header_prev_spots"] = updated

    # ── Build ticker blocks ────────────────────────────────────────
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
            f'<div style="display:flex;flex-direction:column;padding:0 16px;'
            f'border-right:1px solid #1e3040;min-width:100px;">'
            f'<span style="font-family:Barlow Condensed,sans-serif;font-size:9px;'
            f'letter-spacing:1.5px;color:#7fa8c8;font-weight:600;">{idx_short_name}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:15px;'
            f'font-weight:700;color:#e8f4ff;">{disp}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:{col};">{chg_str}</span>'
            f'</div>'
        )
    tickers_html = "".join(ticker_blocks)

    # ── Session badge CSS class ────────────────────────────────────
    sess_class = "session-live" if live_label == "LIVE" else "session-pre" if live_label == "PRE-MARKET" else ""
    sess_dot   = "●" if live_label == "LIVE" else "◑" if live_label == "PRE-MARKET" else "○"

    # ── Circular refresh ring (pure CSS animation, resets on rerun) ──
    ring_html = """
    <div class="refresh-ring-wrap">
      <div class="refresh-ring">
        <svg viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg">
          <circle class="track"    cx="18" cy="18" r="15.9"/>
          <circle class="progress" cx="18" cy="18" r="15.9"/>
        </svg>
        <div class="center-dot">5s</div>
      </div>
      <div class="refresh-label">
        AUTO<span>REFRESH</span>
      </div>
    </div>
    """

    # ── Full header HTML ───────────────────────────────────────────
    header_html = f"""
<div class="sticky-topbar">
  <div style="background:linear-gradient(90deg,#0a0f14,#111920,#0a0f14);
              border-bottom:2px solid #1e3040;
              border-left:3px solid #ff8c00;
              padding:6px 0;
              display:flex;align-items:center;justify-content:space-between;">

    <!-- LEFT: Brand -->
    <div style="display:flex;align-items:center;">
      <div style="padding:0 18px;border-right:1px solid #1e3040;min-width:130px;">
        <span style="font-family:Barlow Condensed,sans-serif;font-size:21px;
                     font-weight:800;letter-spacing:2px;color:#ff8c00;">QUANTDESK</span>
        <span style="font-family:Barlow Condensed,sans-serif;font-size:11px;
                     letter-spacing:3px;color:#7fa8c8;margin-left:5px;">PRO</span>
      </div>

      <!-- TICKERS -->
      <div style="display:flex;align-items:center;overflow-x:auto;scrollbar-width:none;">
        {tickers_html}
      </div>
    </div>

    <!-- RIGHT: Refresh ring + Session + Time -->
    <div style="display:flex;align-items:center;gap:0;padding-right:4px;">
      <!-- Circular Refresh Ring -->
      {ring_html}

      <!-- Divider -->
      <div style="width:1px;height:40px;background:#1e3040;margin:0 12px;"></div>

      <!-- Session -->
      <div style="text-align:right;padding-right:16px;">
        <div style="font-family:Barlow Condensed,sans-serif;font-size:9px;
                    letter-spacing:1.5px;color:#7fa8c8;">SESSION</div>
        <div class="{sess_class}" style="font-family:Barlow Condensed,sans-serif;
                    font-size:16px;font-weight:800;letter-spacing:2px;color:{live_color};">
          {sess_dot} {live_label}
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:9px;
                    color:#3a6080;margin-top:1px;">{now_str} IST</div>
      </div>
    </div>
  </div>
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)

    # ── JS: Update ring countdown with real elapsed time ──────────
    # Inject JS to sync the CSS animation phase with actual refresh cycle
    st.markdown("""
<script>
(function() {
  // Find the progress circle and re-trigger animation in sync
  const circles = document.querySelectorAll('.refresh-ring .progress');
  circles.forEach(c => {
    c.style.animation = 'none';
    void c.offsetHeight; // reflow
    c.style.animation = 'ring-countdown 5s linear infinite';
  });
  // Also sync center dot
  const dots = document.querySelectorAll('.refresh-ring .center-dot');
  dots.forEach(d => {
    d.style.animation = 'none';
    void d.offsetHeight;
    d.style.animation = 'ring-pulse 5s linear infinite';
  });
  // Live countdown text in dot
  let secs = 5;
  dots.forEach(d => d.textContent = secs + 's');
  const timer = setInterval(() => {
    secs--;
    if (secs <= 0) secs = 5;
    dots.forEach(d => d.textContent = secs + 's');
  }, 1000);
})();
</script>
""", unsafe_allow_html=True)


# ======================
# 📑 MAIN TABS
# ======================
render_header()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "OPTION CHAIN",
    "SMART MONEY",
    "POSITIONS",
    "S&R / PAIN",
    "AI ADVISOR",
    "TRADE LOG",
    "GREEKS LAB",
    "\U0001f5a5 STOCKS",
    "\U0001f4ca CHART",
    "\U0001f4f8 HISTORY",
    "\U0001f9fa BASKET",
    "\U0001f3ae PAPER_Trade",
])

with tab1:  tab1_option_chain.render()
with tab2:  tab2_smart_money.render()
with tab3:  tab3_positions.render()
with tab4:  tab4_snr_pain.render()
with tab5:  tab5_ai_advisor.render()
with tab6:  tab6_trade_log.render()
with tab7:  tab7_greeks_lab.render()
with tab8:  tab8_stocks.render()
with tab9:  tab9_chart.render()
with tab10: tab10_history.render()
with tab11: tab11_basket.render()
with tab12: tab12_paper_trading.render()