# ============================================================
# tab_paper_trading.py  —  QuantDesk Pro | Paper Trading Engine
# Pro-level live paper trading with AI reasoning, analytics & journal
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime, timedelta
from config import (
    ACCESS_TOKEN, IST, now_ist, now_ist_dt, MARKET_OPEN,
    df_indices, INDEX_SHORT, LOT_SIZES,
    BASE_DIR, TRADE_FILE, TODAY_TRADES_FILE, CLOSED_POS_FILE,
    AI_LOG_FILE, SNAPSHOT_DIR, instrument_df,
)
from utils import (
    get_atm_strike, get_lot_size, idx_short, compute_grand_total,
    pnl_color, calculate_net_book,
    section_header, metric_card, metrics_row, score_bar,
)
from api import fetch_ltp, fetch_option_chain, fetch_intraday_candles
from analytics import (
    bs_price, bs_greeks, implied_vol_newton, calculate_gamma_bs,
    compute_signal_score, check_alerts, call_claude_trade_setup,
)
from chart_utils import (
    compute_technicals, compute_order_flow,
    detect_liquidity_sweeps, detect_order_blocks,
    detect_fvg, detect_bos_choch, get_order_flow_summary,
)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
PT_FILE           = "paper_trades.json"
PT_JOURNAL_FILE   = "paper_journal.json"
PT_ACCOUNT_FILE   = "paper_account.json"
DEFAULT_CAPITAL   = 500_000          # ₹5 lakh virtual account
MAX_RISK_PCT      = 2.0              # max 2 % of capital per trade
BROKERAGE_PER_LOT = 40              # flat ₹40/lot round-trip simulation
STT_RATE          = 0.000625         # STT on sell side for options

# ─────────────────────────────────────────────────────────────
# STATE INIT HELPERS
# ─────────────────────────────────────────────────────────────
def _init_pt_state():
    if "pt_account" not in st.session_state:
        st.session_state.pt_account = {
            "capital":        DEFAULT_CAPITAL,
            "available":      DEFAULT_CAPITAL,
            "peak":           DEFAULT_CAPITAL,
            "created_at":     now_ist_dt.isoformat(),
            "total_trades":   0,
            "wins":           0,
            "losses":         0,
        }
    if "pt_open_trades" not in st.session_state:
        st.session_state.pt_open_trades = []        # list[dict]
    if "pt_closed_trades" not in st.session_state:
        st.session_state.pt_closed_trades = []      # list[dict]
    if "pt_journal" not in st.session_state:
        st.session_state.pt_journal = []            # list[dict]  — AI reasoning per trade
    if "pt_equity_curve" not in st.session_state:
        st.session_state.pt_equity_curve = [
            {"time": now_ist_dt.strftime("%H:%M"), "equity": DEFAULT_CAPITAL}
        ]
    if "pt_pending_ai" not in st.session_state:
        st.session_state.pt_pending_ai = {}         # trade_id → True while generating
    if "pt_reset_confirm" not in st.session_state:
        st.session_state.pt_reset_confirm = False


# ─────────────────────────────────────────────────────────────
# COST SIMULATION
# ─────────────────────────────────────────────────────────────
def _compute_costs(action: str, premium: float, qty: int, lots: int) -> float:
    brokerage = BROKERAGE_PER_LOT * lots
    stt = (premium * qty * STT_RATE) if action == "SELL" else 0.0
    return round(brokerage + stt, 2)


# ─────────────────────────────────────────────────────────────
# MARGIN ESTIMATOR  (SPAN-like flat approximation)
# ─────────────────────────────────────────────────────────────
def _estimate_margin(action: str, premium: float, strike: float, lots: int, lot_size: int) -> float:
    if action == "BUY":
        return premium * lots * lot_size
    else:
        # SELL margin ≈ 10–15 % of notional + premium received
        notional = strike * lots * lot_size
        return round(notional * 0.12 + premium * lots * lot_size, 0)


# ─────────────────────────────────────────────────────────────
# TRADE EXECUTION
# ─────────────────────────────────────────────────────────────
def _execute_paper_trade(
    oc_df, spot, index_key, expiry,
    strike, opt_type, action, lots,
    sl_pts, target_pts,
    tech_summary, signal_scores,
    custom_reason="",
) -> dict | None:
    """Execute a simulated trade and generate AI reasoning."""
    lot_size  = get_lot_size(index_key)
    qty       = lots * lot_size
    col_ltp   = f"{opt_type}_LTP"
    row       = oc_df[oc_df["Strike"] == strike]
    if row.empty:
        st.error(f"Strike {strike} not found in option chain.")
        return None
    premium   = round(float(row[col_ltp].values[0]), 2)
    if premium <= 0:
        st.error("LTP is zero — cannot execute.")
        return None

    margin    = _estimate_margin(action, premium, strike, lots, lot_size)
    costs     = _compute_costs(action, premium, qty, lots)
    acct      = st.session_state.pt_account

    if margin > acct["available"]:
        st.error(f"❌ Insufficient margin. Need ₹{margin:,.0f}, available ₹{acct['available']:,.0f}")
        return None

    # Direction-aware SL & Target
    if action == "BUY":
        sl_price     = round(premium - sl_pts, 2)
        target_price = round(premium + target_pts, 2)
    else:
        sl_price     = round(premium + sl_pts, 2)
        target_price = round(premium - target_pts, 2)

    trade_id = str(uuid.uuid4())[:8].upper()
    trade = {
        "id":            trade_id,
        "timestamp":     now_ist_dt.strftime("%H:%M:%S"),
        "date":          now_ist_dt.strftime("%Y-%m-%d"),
        "index_key":     index_key,
        "index_name":    INDEX_SHORT.get(index_key, index_key.split("|")[-1]),
        "expiry":        expiry,
        "strike":        strike,
        "opt_type":      opt_type,
        "action":        action,
        "lots":          lots,
        "lot_size":      lot_size,
        "qty":           qty,
        "entry_price":   premium,
        "sl_price":      sl_price,
        "target_price":  target_price,
        "sl_pts":        sl_pts,
        "target_pts":    target_pts,
        "margin_used":   margin,
        "costs":         costs,
        "spot_at_entry": round(spot, 2),
        "atm_at_entry":  get_atm_strike(spot, index_key),
        "status":        "OPEN",
        "ltp":           premium,
        "pnl":           0.0,
        "pnl_pts":       0.0,
        "exit_price":    None,
        "exit_time":     None,
        "exit_reason":   None,
        "custom_reason": custom_reason,
        "signal_bull":   signal_scores.get("bull", 0),
        "signal_bear":   signal_scores.get("bear", 0),
        "pcr":           signal_scores.get("pcr", 1.0),
        "market_flow":   signal_scores.get("flow", ""),
        "rsi":           tech_summary.get("rsi14", 50),
        "ema_trend":     tech_summary.get("ema_trend", ""),
        "macd_cross":    tech_summary.get("macd_cross", ""),
        "price_vs_vwap": tech_summary.get("price_vs_vwap", ""),
        "atr14":         tech_summary.get("atr14", 0),
        "ai_reasoning":  None,      # filled async
    }

    # Debit margin
    acct["available"]  = round(acct["available"] - margin, 2)
    acct["total_trades"] += 1

    st.session_state.pt_open_trades.append(trade)
    st.session_state.pt_pending_ai[trade_id] = True

    return trade


# ─────────────────────────────────────────────────────────────
# CLOSE TRADE
# ─────────────────────────────────────────────────────────────
def _close_trade(trade_id: str, oc_df, reason: str = "MANUAL"):
    trades = st.session_state.pt_open_trades
    idx    = next((i for i, t in enumerate(trades) if t["id"] == trade_id), None)
    if idx is None:
        return
    trade    = trades[idx]
    col_ltp  = f"{trade['opt_type']}_LTP"
    row      = oc_df[oc_df["Strike"] == trade["strike"]]
    exit_ltp = round(float(row[col_ltp].values[0]), 2) if not row.empty else trade["ltp"]

    if trade["action"] == "BUY":
        pnl_pts = round(exit_ltp - trade["entry_price"], 2)
    else:
        pnl_pts = round(trade["entry_price"] - exit_ltp, 2)

    pnl_net = round(pnl_pts * trade["qty"] - trade["costs"], 2)

    trade.update({
        "status":      "CLOSED",
        "exit_price":  exit_ltp,
        "exit_time":   now_ist_dt.strftime("%H:%M:%S"),
        "exit_reason": reason,
        "pnl_pts":     pnl_pts,
        "pnl":         pnl_net,
        "ltp":         exit_ltp,
    })

    # Return margin + realised PnL
    acct = st.session_state.pt_account
    acct["available"] = round(acct["available"] + trade["margin_used"] + pnl_net, 2)
    acct["peak"]      = max(acct["peak"], acct["available"])
    if pnl_net >= 0:
        acct["wins"]    += 1
    else:
        acct["losses"]  += 1

    st.session_state.pt_closed_trades.append(trade)
    st.session_state.pt_open_trades.pop(idx)

    # Equity snapshot
    st.session_state.pt_equity_curve.append({
        "time":   now_ist_dt.strftime("%H:%M"),
        "equity": acct["available"],
    })


# ─────────────────────────────────────────────────────────────
# LIVE PNL UPDATE FOR OPEN TRADES
# ─────────────────────────────────────────────────────────────
def _update_live_pnl(oc_df):
    for trade in st.session_state.pt_open_trades:
        col_ltp = f"{trade['opt_type']}_LTP"
        row     = oc_df[oc_df["Strike"] == trade["strike"]]
        if row.empty:
            continue
        ltp = round(float(row[col_ltp].values[0]), 2)
        trade["ltp"] = ltp
        if trade["action"] == "BUY":
            pnl_pts = round(ltp - trade["entry_price"], 2)
            sl_hit  = ltp <= trade["sl_price"]
            tgt_hit = ltp >= trade["target_price"]
        else:
            pnl_pts = round(trade["entry_price"] - ltp, 2)
            sl_hit  = ltp >= trade["sl_price"]
            tgt_hit = ltp <= trade["target_price"]

        trade["pnl_pts"] = pnl_pts
        trade["pnl"]     = round(pnl_pts * trade["qty"] - trade["costs"], 2)

        # Auto-close on SL / Target
        if trade["status"] == "OPEN":
            if sl_hit:
                trade["status"] = "SL_PENDING"
            elif tgt_hit:
                trade["status"] = "TGT_PENDING"

    # Batch close SL/Target hits
    to_close = [(t["id"], "SL HIT") for t in st.session_state.pt_open_trades if t["status"] == "SL_PENDING"]
    to_close += [(t["id"], "TARGET HIT") for t in st.session_state.pt_open_trades if t["status"] == "TGT_PENDING"]
    for tid, reason in to_close:
        _close_trade(tid, oc_df, reason)


# ─────────────────────────────────────────────────────────────
# AI REASONING GENERATOR
# ─────────────────────────────────────────────────────────────
def _generate_ai_reasoning(trade: dict, oc_df, tech_summary: dict, spot: float) -> str:
    """Generate a rich, structured AI reasoning note for a trade."""
    try:
        import anthropic as _ant
        client = _ant.Anthropic()

        atm = get_atm_strike(spot, trade["index_key"])
        strike_distance = trade["strike"] - atm
        money_str = "ATM" if strike_distance == 0 else f"{'OTM' if (trade['opt_type']=='CE' and strike_distance>0) or (trade['opt_type']=='PE' and strike_distance<0) else 'ITM'} ({abs(strike_distance):+.0f})"

        prompt = f"""You are QuantDesk Pro's senior trading analyst. Write a comprehensive, structured trade journal entry explaining WHY this paper trade was taken.

TRADE DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━
ID:          #{trade['id']}
Index:       {trade['index_name']} (Spot at entry: ₹{trade['spot_at_entry']:,.2f})
Expiry:      {trade['expiry']}
Strike:      {trade['strike']} ({money_str})
Type:        {trade['action']} {trade['opt_type']}
Entry:       ₹{trade['entry_price']}
SL:          ₹{trade['sl_price']}  ({trade['sl_pts']} pts)
Target:      ₹{trade['target_price']}  ({trade['target_pts']} pts)
Lots:        {trade['lots']}  (Qty: {trade['qty']})
R:R Ratio:   1:{round(trade['target_pts']/max(trade['sl_pts'],1), 1)}
Margin Used: ₹{trade['margin_used']:,.0f}
Time:        {trade['timestamp']}

SIGNAL CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Market Flow: {trade['market_flow']}
Bull Score:  {trade['signal_bull']}/20
Bear Score:  {trade['signal_bear']}/20
PCR:         {trade['pcr']:.3f}

TECHNICAL CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━
RSI(14):          {trade['rsi']:.1f}
EMA Trend:        {trade['ema_trend']}
MACD Cross:       {trade['macd_cross']}
Price vs VWAP:    {trade['price_vs_vwap']}
ATR(14):          {trade['atr14']:.2f} pts
Day High:         ₹{tech_summary.get('high_of_day','N/A')}
Day Low:          ₹{tech_summary.get('low_of_day','N/A')}
VWAP:             ₹{tech_summary.get('vwap','N/A')}
BB Condition:     {tech_summary.get('bb_condition','N/A')}
SuperTrend:       {tech_summary.get('supertrend_bias','N/A')}
Candle Count:     {tech_summary.get('candles_count',0)}

TRADER'S NOTE: {trade.get('custom_reason','')}

Write a journal entry with these EXACT sections (use rich markdown):

**📊 TRADE SETUP SUMMARY**
One-line punchy description of the exact setup.

**🔍 WHY THIS TRADE (Primary Logic)**
3-4 sentences explaining the core reason — what confluence of factors triggered this exact entry. Reference specific numbers.

**📈 TECHNICAL JUSTIFICATION**
Bullet-point breakdown of each technical indicator and what it signaled. Be specific with values.

**🏦 OPTIONS FLOW LOGIC**
Why this specific strike and option type. What OI data, PCR, and GEX regime told us. Why {trade['action']} not the opposite.

**⚠️ RISK MANAGEMENT**
Why this SL level is logical (structural or ATR-based). What invalidates the thesis. Max risk in rupees.

**🎯 TARGET RATIONALE**  
Why this specific target. What level acts as a magnet. R:R justification.

**🚦 TRADE SCORE: X/10**
Overall quality score with brief verdict.

Be direct, professional, and specific. Mention actual price levels. No fluff."""

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"⚠️ AI reasoning unavailable: {e}"


# ─────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────
def _calc_performance() -> dict:
    closed = st.session_state.pt_closed_trades
    acct   = st.session_state.pt_account
    if not closed:
        return {}
    pnls    = [t["pnl"] for t in closed]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p < 0]
    wr      = len(wins) / len(pnls) * 100
    avg_w   = sum(wins)   / max(len(wins),   1)
    avg_l   = abs(sum(losses) / max(len(losses), 1))
    expect  = (wr/100 * avg_w) - ((1-wr/100) * avg_l)
    pf      = sum(wins) / max(abs(sum(losses)), 0.01)
    equity  = acct["available"]
    ret_pct = (equity - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100
    dd      = (acct["peak"] - equity) / acct["peak"] * 100
    return {
        "total_trades": len(closed),
        "win_rate":     round(wr, 1),
        "avg_win":      round(avg_w, 0),
        "avg_loss":     round(avg_l, 0),
        "expectancy":   round(expect, 0),
        "profit_factor":round(pf, 2),
        "total_pnl":    round(sum(pnls), 0),
        "equity":       round(equity, 0),
        "return_pct":   round(ret_pct, 2),
        "max_drawdown": round(dd, 2),
        "wins":         len(wins),
        "losses":       len(losses),
    }


# ─────────────────────────────────────────────────────────────
# HTML COMPONENTS
# ─────────────────────────────────────────────────────────────
def _header_html(acct: dict, open_pnl: float, closed_pnl: float) -> str:
    total_pnl = open_pnl + closed_pnl
    pnl_c     = "#00e676" if total_pnl >= 0 else "#ff3d57"
    avail_pct = acct["available"] / acct["capital"] * 100
    dd_pct    = max(0, (acct["peak"] - acct["available"]) / acct["peak"] * 100)
    dd_c      = "#ff3d57" if dd_pct > 10 else "#ffd600" if dd_pct > 5 else "#00e676"
    return f"""
<div style="background:linear-gradient(90deg,#0d1117,#111920);
            border:1px solid #2a4560;border-left:4px solid #ff8c00;
            border-radius:4px;padding:14px 20px;margin-bottom:14px;">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:2px;color:#7fa8c8;">
        🎮 PAPER TRADING ACCOUNT</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;color:#e8f4ff;margin-top:2px;">
        ₹{acct['available']:,.0f}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3a6080;margin-top:2px;">
        Started: ₹{acct['capital']:,.0f} &nbsp;|&nbsp; Peak: ₹{acct['peak']:,.0f}</div>
    </div>
    <div style="display:flex;gap:20px;flex-wrap:wrap;">
      <div style="text-align:center;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:1.5px;color:#7fa8c8;">OPEN P&L</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;
                    color:{'#00e676' if open_pnl>=0 else '#ff3d57'};">₹{open_pnl:+,.0f}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:1.5px;color:#7fa8c8;">CLOSED P&L</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;
                    color:{'#00e676' if closed_pnl>=0 else '#ff3d57'};">₹{closed_pnl:+,.0f}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:1.5px;color:#7fa8c8;">NET P&L</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:{pnl_c};">
          ₹{total_pnl:+,.0f}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:1.5px;color:#7fa8c8;">MARGIN USED</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:#ffd600;">
          {100-avail_pct:.1f}%</div>
      </div>
      <div style="text-align:center;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:1.5px;color:#7fa8c8;">DRAWDOWN</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:{dd_c};">
          {dd_pct:.1f}%</div>
      </div>
    </div>
  </div>
</div>"""


def _trade_card_html(trade: dict) -> str:
    pnl   = trade.get("pnl", 0)
    pnl_c = "#00e676" if pnl >= 0 else "#ff3d57"
    ac    = "#00e676" if trade["action"] == "BUY" else "#ff3d57"
    tc    = "#ff3d57" if trade["opt_type"] == "CE" else "#00e676"
    # Progress bar toward target
    entry = trade["entry_price"]
    tgt   = trade["target_price"]
    sl    = trade["sl_price"]
    ltp   = trade.get("ltp", entry)
    if trade["action"] == "BUY":
        prog = min(max((ltp - entry) / (tgt - entry + 0.001) * 100, 0), 100)
        sl_prog = min(max((entry - ltp) / (entry - sl + 0.001) * 100, 0), 100)
    else:
        prog = min(max((entry - ltp) / (entry - tgt + 0.001) * 100, 0), 100)
        sl_prog = min(max((ltp - entry) / (sl - entry + 0.001) * 100, 0), 100)
    bar_c = "#00e676" if prog > 50 else "#ffd600" if prog > 20 else "#7fa8c8"
    sl_bar_c = "#ff3d57" if sl_prog > 50 else "#ff8c00" if sl_prog > 25 else "#3a6080"

    return f"""
<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid {pnl_c};
            border-radius:3px;padding:12px 14px;margin:6px 0;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
                     font-weight:800;letter-spacing:1px;color:#e8f4ff;">
          {trade['index_name']} {int(trade['strike'])} {trade['opt_type']}</span>
        <span style="background:{'#051a0e' if trade['action']=='BUY' else '#1a0508'};
                     border:1px solid {ac};color:{ac};
                     font-family:'Barlow Condensed',sans-serif;font-size:11px;
                     font-weight:700;letter-spacing:1px;padding:2px 8px;border-radius:2px;">
          {trade['action']}</span>
        <span style="background:#070b0f;border:1px solid #1e3040;color:{tc};
                     font-family:'Barlow Condensed',sans-serif;font-size:11px;
                     padding:2px 8px;border-radius:2px;">{trade['opt_type']}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;">
          #{trade['id']}</span>
      </div>
      <div style="display:flex;gap:16px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#7fa8c8;">
        <span>Entry: <b style="color:#e8f4ff;">₹{trade['entry_price']}</b></span>
        <span>LTP: <b style="color:#00d4ff;">₹{ltp}</b></span>
        <span>SL: <b style="color:#ff3d57;">₹{trade['sl_price']}</b></span>
        <span>Tgt: <b style="color:#00e676;">₹{trade['target_price']}</b></span>
        <span>Lots: <b style="color:#ffd600;">{trade['lots']}</b></span>
        <span>Qty: {trade['qty']}</span>
        <span>@{trade['timestamp']}</span>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                  font-weight:700;color:{pnl_c};">₹{pnl:+,.0f}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3a6080;">
        {trade.get('pnl_pts',0):+.2f} pts · Mrgn ₹{trade['margin_used']:,.0f}</div>
    </div>
  </div>
  <!-- Target Progress Bar -->
  <div style="margin-top:8px;">
    <div style="display:flex;justify-content:space-between;
                font-family:'Barlow Condensed',sans-serif;font-size:9px;
                letter-spacing:1px;color:#3a6080;margin-bottom:3px;">
      <span>🎯 TO TARGET {prog:.0f}%</span>
      <span>⛔ TO SL {sl_prog:.0f}%</span>
    </div>
    <div style="display:flex;height:4px;border-radius:2px;overflow:hidden;background:#1e3040;">
      <div style="width:{prog:.0f}%;background:{bar_c};border-radius:2px;transition:width 0.3s;"></div>
    </div>
    <div style="display:flex;height:3px;border-radius:2px;overflow:hidden;background:#1e3040;margin-top:2px;">
      <div style="width:{sl_prog:.0f}%;background:{sl_bar_c};border-radius:2px;transition:width 0.3s;"></div>
    </div>
  </div>
  <!-- Signal context chips -->
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">
    <span style="background:#0d1117;border:1px solid #1e3040;color:#7fa8c8;
                 font-family:'Barlow Condensed',sans-serif;font-size:9px;
                 letter-spacing:1px;padding:2px 8px;border-radius:10px;">
      {trade.get('market_flow','').upper()}</span>
    <span style="background:#0d1117;border:1px solid #1e3040;color:#7fa8c8;
                 font-family:'Barlow Condensed',sans-serif;font-size:9px;
                 padding:2px 8px;border-radius:10px;">
      BULL {trade.get('signal_bull',0)}/20 · BEAR {trade.get('signal_bear',0)}/20</span>
    <span style="background:#0d1117;border:1px solid #1e3040;color:#c084fc;
                 font-family:'Barlow Condensed',sans-serif;font-size:9px;
                 padding:2px 8px;border-radius:10px;">
      RSI {trade.get('rsi',50):.0f} · {trade.get('ema_trend','')}</span>
    <span style="background:#0d1117;border:1px solid #1e3040;color:#ffd600;
                 font-family:'Barlow Condensed',sans-serif;font-size:9px;
                 padding:2px 8px;border-radius:10px;">
      PCR {trade.get('pcr',1):.3f} · {trade.get('price_vs_vwap','')}</span>
  </div>
</div>"""


def _closed_card_html(trade: dict) -> str:
    pnl   = trade.get("pnl", 0)
    pnl_c = "#00e676" if pnl >= 0 else "#ff3d57"
    icon  = "🏆" if trade.get("exit_reason") == "TARGET HIT" else "⛔" if trade.get("exit_reason") == "SL HIT" else "✋"
    return f"""
<div style="background:#070b0f;border:1px solid #1e3040;border-left:3px solid {pnl_c};
            border-radius:2px;padding:10px 14px;margin:4px 0;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#e8f4ff;font-weight:600;">
      {trade['index_name']} {int(trade['strike'])} {trade['opt_type']} 
      {trade['action']}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3a6080;margin-left:12px;">
      #{trade['id']} · {trade.get('exit_time','')}</span>
  </div>
  <div style="display:flex;gap:16px;align-items:center;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#7fa8c8;">
      E:₹{trade['entry_price']} → X:₹{trade.get('exit_price',0)} 
      ({trade.get('pnl_pts',0):+.1f}pts)</span>
    <span style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
                 color:{'#00e676' if 'TARGET' in str(trade.get('exit_reason','')) else '#ff3d57' if 'SL' in str(trade.get('exit_reason','')) else '#7fa8c8'};
                 letter-spacing:1px;">{icon} {trade.get('exit_reason','MANUAL')}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:16px;
                 font-weight:700;color:{pnl_c};">₹{pnl:+,.0f}</span>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────
# EQUITY CURVE CHART
# ─────────────────────────────────────────────────────────────
def _equity_chart():
    ec   = st.session_state.pt_equity_curve
    if len(ec) < 2 or not HAS_PLOTLY:
        return
    times  = [e["time"] for e in ec]
    equity = [e["equity"] for e in ec]
    colors = ["#00e676" if e >= DEFAULT_CAPITAL else "#ff3d57" for e in equity]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=equity,
        mode="lines+markers",
        name="Equity",
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        marker=dict(size=5, color=colors),
    ))
    fig.add_hline(y=DEFAULT_CAPITAL, line_color="#ff8c00", line_dash="dot",
                  line_width=1.2,
                  annotation_text=f"Starting ₹{DEFAULT_CAPITAL:,.0f}",
                  annotation_font_color="#ff8c00", annotation_font_size=9)
    peak = max(equity)
    fig.add_hline(y=peak, line_color="#00e676", line_dash="dash",
                  line_width=0.8,
                  annotation_text=f"Peak ₹{peak:,.0f}",
                  annotation_font_color="#00e676", annotation_font_size=9)
    fig.update_layout(
        height=240, paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
        font=dict(family="JetBrains Mono", color="#7fa8c8", size=10),
        margin=dict(l=10, r=10, t=20, b=30),
        showlegend=False,
        xaxis=dict(gridcolor="#1a2a3a", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1a2a3a", showgrid=True, zeroline=False, tickformat="₹,.0f"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _pnl_distribution_chart():
    closed = st.session_state.pt_closed_trades
    if not closed or not HAS_PLOTLY:
        return
    pnls   = [t["pnl"] for t in closed]
    colors = ["#00e676" if p >= 0 else "#ff3d57" for p in pnls]
    labels = [f"#{t['id']}" for t in closed]
    fig = go.Figure(go.Bar(
        x=labels, y=pnls,
        marker_color=colors,
        text=[f"₹{p:+,.0f}" for p in pnls],
        textposition="outside",
        textfont=dict(size=9, color="#7fa8c8"),
    ))
    fig.add_hline(y=0, line_color="#3a6080", line_dash="dot", line_width=0.8)
    fig.update_layout(
        height=220, paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
        font=dict(family="JetBrains Mono", color="#7fa8c8", size=10),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
        xaxis=dict(gridcolor="#1a2a3a"),
        yaxis=dict(gridcolor="#1a2a3a", title="P&L ₹", tickformat="₹,.0f"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────
def render():
    st.session_state["active_tab_key"] = "🎮 PAPER"
    _init_pt_state()

    # ── Live data deps ────────────────────────────────────────────────
    oc_df   = st.session_state.get("current_option_chain", pd.DataFrame())
    spot    = st.session_state.get("current_spot_price", 0)
    sel_idx = st.session_state.get("current_selected_index", "")
    expiry  = st.session_state.get("oc_expiry_select", "")

    # Page header
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;
                border-bottom:1px solid #1e3040;padding-bottom:10px;margin-bottom:14px;">
      <div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:24px;
                    font-weight:800;letter-spacing:3px;color:#e8f4ff;">
          🎮 PAPER <span style="color:#ff8c00;">TRADING</span> ENGINE</div>
        <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#7fa8c8;margin-top:2px;">
          Live-price simulation · AI trade reasoning · Full analytics · Zero real money</div>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;text-align:right;">
        VIRTUAL ACCOUNT<br>
        <span style="color:#ff8c00;font-size:12px;">REAL PRICES · SIMULATED ORDERS</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Guard: need option chain
    if oc_df is None or (isinstance(oc_df, pd.DataFrame) and oc_df.empty) or not spot:
        st.warning("⏳ Load the Option Chain from **Tab 1** first — paper trading needs live data.")
        return

    # ── Live PnL update every render ─────────────────────────────────
    _update_live_pnl(oc_df)

    open_pnl   = sum(t.get("pnl", 0) for t in st.session_state.pt_open_trades)
    closed_pnl = sum(t.get("pnl", 0) for t in st.session_state.pt_closed_trades)

    # ── Account Header ────────────────────────────────────────────────
    st.markdown(_header_html(st.session_state.pt_account, open_pnl, closed_pnl),
                unsafe_allow_html=True)

    # ── MAIN LAYOUT: Trade Entry | Open Positions ────────────────────
    left_col, right_col = st.columns([1.1, 1.9], gap="medium")

    # ══════════════════════════════════════════════════════
    # LEFT — ORDER ENTRY PANEL
    # ══════════════════════════════════════════════════════
    with left_col:
        section_header("⚡ Order Entry")

        atm = get_atm_strike(spot, sel_idx)
        strikes = sorted(oc_df["Strike"].unique().tolist())

        # Strike selector with ATM highlighted
        try:
            atm_idx_entry = strikes.index(atm)
        except ValueError:
            atm_idx_entry = 0

        col_a, col_b = st.columns(2)
        with col_a:
            action   = st.radio("Action", ["BUY", "SELL"], horizontal=True, key="pt_action")
        with col_b:
            opt_type = st.radio("Type", ["CE", "PE"], horizontal=True, key="pt_otype")

        strike_sel = st.selectbox(
            "Strike", strikes,
            index=atm_idx_entry, key="pt_strike",
            format_func=lambda s: f"{int(s)}  {'★ ATM' if s == atm else ''}",
        )
        lots_sel = st.number_input(
            "Lots", min_value=1, max_value=50, value=1, step=1, key="pt_lots"
        )

        # Live preview
        col_ltp = f"{opt_type}_LTP"
        row_sel = oc_df[oc_df["Strike"] == strike_sel]
        ltp_now = round(float(row_sel[col_ltp].values[0]), 2) if not row_sel.empty else 0.0
        lot_sz  = get_lot_size(sel_idx)
        qty_tot = lots_sel * lot_sz
        margin_est = _estimate_margin(action, ltp_now, strike_sel, lots_sel, lot_sz)
        max_risk   = ltp_now * qty_tot if action == "BUY" else ltp_now * qty_tot * 3

        preview_c = "#00e676" if action == "BUY" else "#ff3d57"
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid {preview_c};
                    border-radius:3px;padding:12px;margin:8px 0;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;
                      font-family:'JetBrains Mono',monospace;font-size:12px;">
            <div><span style="color:#3a6080;">LTP</span>
              <div style="font-size:22px;font-weight:700;color:{preview_c};">₹{ltp_now}</div></div>
            <div><span style="color:#3a6080;">QTY</span>
              <div style="font-size:18px;color:#e8f4ff;">{qty_tot}</div></div>
            <div><span style="color:#3a6080;">EST. MARGIN</span>
              <div style="color:#ffd600;font-weight:700;">₹{margin_est:,.0f}</div></div>
            <div><span style="color:#3a6080;">MAX RISK</span>
              <div style="color:#ff8c00;">₹{max_risk:,.0f}</div></div>
          </div>
          {'<div style="margin-top:6px;font-family:Barlow Condensed,sans-serif;font-size:11px;color:#ffd600;letter-spacing:1px;">★ ATM STRIKE</div>' if strike_sel == atm else ''}
        </div>""", unsafe_allow_html=True)

        # SL / Target
        col_sl, col_tgt = st.columns(2)
        with col_sl:
            sl_pts  = st.number_input("SL (pts)", value=15, min_value=1, max_value=200, key="pt_sl")
        with col_tgt:
            tgt_pts = st.number_input("Target (pts)", value=30, min_value=1, max_value=500, key="pt_tgt")

        rr = tgt_pts / max(sl_pts, 1)
        rr_c = "#00e676" if rr >= 2 else "#ffd600" if rr >= 1.5 else "#ff3d57"
        st.markdown(f"""<div style="text-align:center;font-family:'Barlow Condensed',sans-serif;
            font-size:14px;font-weight:700;color:{rr_c};letter-spacing:1px;margin:4px 0;">
            R:R = 1:{rr:.1f}</div>""", unsafe_allow_html=True)

        # Custom reason
        custom_reason = st.text_area(
            "Trade Reason (optional — will be included in AI analysis)",
            placeholder="e.g. VWAP reclaim after SSL sweep, expecting CE unwinding...",
            height=60, key="pt_reason"
        )

        # Get signal context
        try:
            bull_s, bear_s, pcr_v, pcr_chg, res_s, sup_s, _ = compute_signal_score(
                oc_df, spot, sel_idx)
            tech_df, tech_sum = compute_technicals(
                fetch_intraday_candles(sel_idx, "5minute"))
            market_flow = (
                "Bullish" if bull_s - bear_s >= 5
                else "Bearish" if bear_s - bull_s >= 5
                else "Range" if abs(bull_s - bear_s) <= 3
                else "Choppy"
            )
        except Exception:
            bull_s, bear_s, pcr_v, market_flow = 10, 10, 1.0, "Range"
            tech_sum = {}

        signal_scores = {
            "bull": bull_s, "bear": bear_s, "pcr": pcr_v, "flow": market_flow
        }

        # Signal context display
        flow_c = {"Bullish": "#00e676", "Bearish": "#ff3d57",
                  "Range": "#ffd600", "Choppy": "#7fa8c8"}.get(market_flow, "#7fa8c8")
        st.markdown(f"""
        <div style="background:#070b0f;border:1px solid #1a2a3a;border-radius:2px;
                    padding:6px 10px;margin:6px 0;display:flex;gap:12px;flex-wrap:wrap;
                    font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:1px;">
          <span style="color:{flow_c};">● {market_flow.upper()}</span>
          <span style="color:#00e676;">▲ {bull_s}/20</span>
          <span style="color:#ff3d57;">▼ {bear_s}/20</span>
          <span style="color:#c084fc;">PCR {pcr_v:.3f}</span>
          <span style="color:#00d4ff;">RSI {tech_sum.get('rsi14',50):.0f}</span>
          <span style="color:#7fa8c8;">{tech_sum.get('ema_trend','')}</span>
        </div>""", unsafe_allow_html=True)

        # EXECUTE button
        mkt_label = "EXECUTE PAPER TRADE" if MARKET_OPEN else "SIMULATE (MKT CLOSED)"
        if st.button(f"🚀 {mkt_label}", use_container_width=True, type="primary", key="pt_exec"):
            new_trade = _execute_paper_trade(
                oc_df, spot, sel_idx, expiry,
                strike_sel, opt_type, action, lots_sel,
                sl_pts, tgt_pts,
                tech_sum, signal_scores, custom_reason
            )
            if new_trade:
                st.success(f"✅ #{new_trade['id']} executed — {action} {opt_type} {int(strike_sel)} @ ₹{ltp_now}")
                st.rerun()

        st.markdown("---")

        # ── Quick Template Buttons ────────────────────────────────
        section_header("⚡ Quick Templates")
        step_series = oc_df["Strike"].diff().dropna()
        step = int(step_series.mode()[0]) if not step_series.empty else 50

        templates = {
            "Bull Put\nSpread":   [("SELL","PE",atm,sl_pts,tgt_pts),   ("BUY","PE",atm-step,sl_pts,tgt_pts)],
            "Bear Call\nSpread":  [("SELL","CE",atm,sl_pts,tgt_pts),   ("BUY","CE",atm+step,sl_pts,tgt_pts)],
            "ATM CE\nBuy":        [("BUY","CE",atm,sl_pts,tgt_pts)],
            "ATM PE\nBuy":        [("BUY","PE",atm,sl_pts,tgt_pts)],
        }
        tc1, tc2, tc3, tc4 = st.columns(4)
        for col, (tname, legs) in zip([tc1,tc2,tc3,tc4], templates.items()):
            with col:
                if st.button(tname, key=f"tmpl_{tname}", use_container_width=True):
                    for leg_action, leg_type, leg_strike, leg_sl, leg_tgt in legs:
                        # Validate strike exists
                        if leg_strike not in strikes:
                            continue
                        _execute_paper_trade(
                            oc_df, spot, sel_idx, expiry,
                            leg_strike, leg_type, leg_action, 1,
                            leg_sl, leg_tgt,
                            tech_sum, signal_scores,
                            f"Template: {tname.replace(chr(10),' ')}"
                        )
                    st.rerun()

        # ── Account controls ──────────────────────────────────────
        st.markdown("---")
        with st.expander("⚙️ ACCOUNT SETTINGS"):
            new_cap = st.number_input(
                "Starting Capital (₹)", value=DEFAULT_CAPITAL,
                step=50000, min_value=100000, key="pt_capital"
            )
            if st.button("Apply Capital", key="pt_apply_cap"):
                diff = new_cap - st.session_state.pt_account["capital"]
                st.session_state.pt_account["capital"]   = new_cap
                st.session_state.pt_account["available"] += diff
                st.session_state.pt_account["peak"]       = max(
                    st.session_state.pt_account["peak"],
                    st.session_state.pt_account["available"]
                )
                st.success(f"Capital updated to ₹{new_cap:,.0f}")

            st.markdown("---")
            if st.button("⚠️ RESET ALL PAPER TRADES", key="pt_reset_btn"):
                st.session_state.pt_reset_confirm = True

            if st.session_state.pt_reset_confirm:
                st.warning("This clears ALL paper trades and resets the account. Are you sure?")
                col_yes, col_no = st.columns(2)
                if col_yes.button("✅ YES, RESET", key="pt_yes_reset"):
                    st.session_state.pt_open_trades   = []
                    st.session_state.pt_closed_trades = []
                    st.session_state.pt_journal       = []
                    st.session_state.pt_pending_ai    = {}
                    st.session_state.pt_equity_curve  = [
                        {"time": now_ist_dt.strftime("%H:%M"), "equity": new_cap}]
                    st.session_state.pt_account = {
                        "capital": new_cap, "available": new_cap,
                        "peak": new_cap,
                        "created_at": now_ist_dt.isoformat(),
                        "total_trades": 0, "wins": 0, "losses": 0,
                    }
                    st.session_state.pt_reset_confirm = False
                    st.success("Account reset.")
                    st.rerun()
                if col_no.button("❌ Cancel", key="pt_no_reset"):
                    st.session_state.pt_reset_confirm = False

    # ══════════════════════════════════════════════════════
    # RIGHT — OPEN POSITIONS + ANALYTICS
    # ══════════════════════════════════════════════════════
    with right_col:

        # ── Open Positions ────────────────────────────────────────
        open_trades = st.session_state.pt_open_trades
        section_header(
            f"📂 Open Positions  ({len(open_trades)})",
            f"Live P&L: ₹{open_pnl:+,.0f}"
        )

        if not open_trades:
            st.markdown("""
            <div style="background:#0d1117;border:1px dashed #1e3040;padding:20px;
                        text-align:center;color:#3a6080;border-radius:3px;
                        font-family:'Barlow Condensed',sans-serif;font-size:14px;letter-spacing:1px;">
              NO OPEN POSITIONS — EXECUTE A TRADE ON THE LEFT
            </div>""", unsafe_allow_html=True)
        else:
            for trade in open_trades:
                st.markdown(_trade_card_html(trade), unsafe_allow_html=True)
                # Close + AI buttons per row
                close_col, ai_col, _ = st.columns([1, 1.5, 3])
                with close_col:
                    if st.button(f"✕ Close #{trade['id']}", key=f"close_{trade['id']}"):
                        _close_trade(trade["id"], oc_df, "MANUAL")
                        st.rerun()
                with ai_col:
                    ai_key = f"aishow_{trade['id']}"
                    if st.button(f"🧠 AI Reasoning #{trade['id']}", key=f"ai_btn_{trade['id']}"):
                        with st.spinner("Generating AI analysis…"):
                            reasoning = _generate_ai_reasoning(trade, oc_df, tech_sum, spot)
                            trade["ai_reasoning"] = reasoning
                        st.session_state.pt_pending_ai.pop(trade["id"], None)
                        st.rerun()
                    if trade.get("ai_reasoning"):
                        with st.expander(f"📖 AI Journal #{trade['id']}", expanded=False):
                            st.markdown(trade["ai_reasoning"])

        # ── Closed Trades ──────────────────────────────────────────
        st.markdown("---")
        closed_trades = st.session_state.pt_closed_trades
        section_header(
            f"✅ Closed Trades  ({len(closed_trades)})",
            f"Realised P&L: ₹{closed_pnl:+,.0f}"
        )
        if not closed_trades:
            st.markdown("""<div style="color:#3a6080;font-family:'Barlow Condensed',sans-serif;
                font-size:12px;letter-spacing:1px;padding:8px;">NO CLOSED TRADES YET</div>""",
                unsafe_allow_html=True)
        else:
            for t in closed_trades[::-1][:15]:      # last 15 closed
                st.markdown(_closed_card_html(t), unsafe_allow_html=True)
                if t.get("ai_reasoning"):
                    with st.expander(f"📖 AI Journal #{t['id']} — {t.get('exit_reason','')}", expanded=False):
                        st.markdown(t["ai_reasoning"])
                elif st.button(f"🧠 Generate AI Journal #{t['id']}", key=f"ai_closed_{t['id']}"):
                    with st.spinner("Generating post-trade analysis…"):
                        reasoning = _generate_ai_reasoning(t, oc_df, tech_sum, spot)
                        t["ai_reasoning"] = reasoning
                    st.rerun()

    # ══════════════════════════════════════════════════════
    # BOTTOM: FULL ANALYTICS DASHBOARD
    # ══════════════════════════════════════════════════════
    st.markdown("---")
    section_header("📊 Performance Analytics",
                   "Equity curve · Win stats · P&L distribution · Trade journal")

    perf = _calc_performance()

    if perf:
        # KPI metrics row
        pnl_c_a = "#00e676" if perf.get("total_pnl", 0) >= 0 else "#ff3d57"
        ret_c   = "#00e676" if perf.get("return_pct", 0) >= 0 else "#ff3d57"
        wr_c    = "#00e676" if perf.get("win_rate", 0) >= 55 else "#ffd600" if perf.get("win_rate", 0) >= 45 else "#ff3d57"
        pf_c    = "#00e676" if perf.get("profit_factor", 0) >= 1.5 else "#ffd600" if perf.get("profit_factor", 0) >= 1 else "#ff3d57"
        dd_c_a  = "#ff3d57" if perf.get("max_drawdown", 0) > 10 else "#ffd600" if perf.get("max_drawdown", 0) > 5 else "#00e676"

        metrics_row(
            metric_card("TOTAL P&L",    f"₹{perf['total_pnl']:+,.0f}", f"{perf['total_trades']} trades", pnl_c_a) +
            metric_card("RETURN",       f"{perf['return_pct']:+.2f}%", f"on ₹{DEFAULT_CAPITAL:,.0f}", ret_c) +
            metric_card("WIN RATE",     f"{perf['win_rate']:.1f}%",  f"{perf['wins']}W / {perf['losses']}L", wr_c) +
            metric_card("AVG WIN",      f"₹{perf['avg_win']:,.0f}",  "", "#00e676") +
            metric_card("AVG LOSS",     f"₹{perf['avg_loss']:,.0f}", "", "#ff3d57") +
            metric_card("EXPECTANCY",   f"₹{perf['expectancy']:+,.0f}", "per trade", "#c084fc") +
            metric_card("PROFIT FACTOR",f"{perf['profit_factor']:.2f}", "gross win/loss", pf_c) +
            metric_card("MAX DRAWDOWN", f"{perf['max_drawdown']:.1f}%", "from peak", dd_c_a)
        )

    else:
        st.caption("📊 Analytics appear after your first closed trade.")

    # Charts row
    chart_a, chart_b = st.columns(2)
    with chart_a:
        section_header("Equity Curve")
        _equity_chart()
    with chart_b:
        section_header("P&L per Trade")
        _pnl_distribution_chart()

    # ── Full Trade Journal ────────────────────────────────────────────
    st.markdown("---")
    section_header("📒 Full Trade Journal",
                   "Every paper trade with AI reasoning, entry logic & outcome")

    all_trades = st.session_state.pt_closed_trades + st.session_state.pt_open_trades
    if not all_trades:
        st.info("Your journal is empty — start paper trading to build your log.")
    else:
        for t in all_trades[::-1]:
            status_c = {
                "OPEN":       "#00d4ff",
                "CLOSED":     "#7fa8c8",
                "TARGET HIT": "#00e676",
                "SL HIT":     "#ff3d57",
            }.get(t.get("exit_reason", t.get("status", "OPEN")), "#7fa8c8")

            pnl_j   = t.get("pnl", 0)
            pnl_j_c = "#00e676" if pnl_j >= 0 else "#ff3d57"
            stat_label = t.get("exit_reason", "OPEN") if t["status"] != "OPEN" else "OPEN"

            with st.expander(
                f"{'🟢' if pnl_j >= 0 else '🔴'} #{t['id']}  {t['index_name']} "
                f"{int(t['strike'])} {t['opt_type']} {t['action']} "
                f"| ₹{pnl_j:+,.0f}  |  {stat_label}  |  {t['date']} {t['timestamp']}",
                expanded=False
            ):
                j1, j2 = st.columns([1, 1])

                with j1:
                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                                background:#0d1117;border:1px solid #1e3040;
                                border-radius:3px;padding:12px;">
                      <div style="color:#7fa8c8;font-size:9px;letter-spacing:1.5px;
                                  font-family:'Barlow Condensed',sans-serif;">TRADE DETAILS</div>
                      <table style="width:100%;margin-top:6px;border-collapse:collapse;">
                        <tr><td style="color:#3a6080;padding:2px 0;">Strike</td>
                            <td style="color:#e8f4ff;text-align:right;">{int(t['strike'])} {t['opt_type']}</td></tr>
                        <tr><td style="color:#3a6080;">Action</td>
                            <td style="color:{'#00e676' if t['action']=='BUY' else '#ff3d57'};text-align:right;font-weight:700;">{t['action']}</td></tr>
                        <tr><td style="color:#3a6080;">Entry</td>
                            <td style="color:#e8f4ff;text-align:right;">₹{t['entry_price']}</td></tr>
                        <tr><td style="color:#3a6080;">Exit</td>
                            <td style="color:#e8f4ff;text-align:right;">₹{t.get('exit_price','—')}</td></tr>
                        <tr><td style="color:#3a6080;">SL</td>
                            <td style="color:#ff3d57;text-align:right;">₹{t['sl_price']}</td></tr>
                        <tr><td style="color:#3a6080;">Target</td>
                            <td style="color:#00e676;text-align:right;">₹{t['target_price']}</td></tr>
                        <tr><td style="color:#3a6080;">Lots / Qty</td>
                            <td style="color:#e8f4ff;text-align:right;">{t['lots']} / {t['qty']}</td></tr>
                        <tr><td style="color:#3a6080;">P&L pts</td>
                            <td style="color:{pnl_j_c};text-align:right;font-weight:700;">{t.get('pnl_pts',0):+.2f}</td></tr>
                        <tr><td style="color:#3a6080;">Net P&L</td>
                            <td style="color:{pnl_j_c};text-align:right;font-size:16px;font-weight:700;">₹{pnl_j:+,.0f}</td></tr>
                        <tr><td style="color:#3a6080;">Costs</td>
                            <td style="color:#ff8c00;text-align:right;">₹{t.get('costs',0)}</td></tr>
                        <tr><td style="color:#3a6080;">Margin</td>
                            <td style="color:#ffd600;text-align:right;">₹{t['margin_used']:,.0f}</td></tr>
                        <tr><td style="color:#3a6080;">Spot@Entry</td>
                            <td style="color:#7fa8c8;text-align:right;">₹{t['spot_at_entry']:,.2f}</td></tr>
                        <tr><td style="color:#3a6080;">Exit Reason</td>
                            <td style="color:{status_c};text-align:right;font-weight:700;">{stat_label}</td></tr>
                      </table>
                    </div>""", unsafe_allow_html=True)

                with j2:
                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                background:#0d1117;border:1px solid #1e3040;
                                border-radius:3px;padding:12px;">
                      <div style="color:#7fa8c8;font-size:9px;letter-spacing:1.5px;
                                  font-family:'Barlow Condensed',sans-serif;margin-bottom:8px;">SIGNAL CONTEXT AT ENTRY</div>
                      <div style="display:flex;flex-wrap:wrap;gap:6px;">
                        <span style="background:#051a0e;border:1px solid #00e676;color:#00e676;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          ▲ BULL {t.get('signal_bull',0)}/20</span>
                        <span style="background:#1a0508;border:1px solid #ff3d57;color:#ff3d57;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          ▼ BEAR {t.get('signal_bear',0)}/20</span>
                        <span style="background:#001020;border:1px solid #00d4ff;color:#00d4ff;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          PCR {t.get('pcr',1):.3f}</span>
                        <span style="background:#0d0018;border:1px solid #c084fc;color:#c084fc;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          RSI {t.get('rsi',50):.0f}</span>
                        <span style="background:#070b0f;border:1px solid #ffd600;color:#ffd600;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          {t.get('market_flow','').upper()}</span>
                        <span style="background:#070b0f;border:1px solid #7fa8c8;color:#7fa8c8;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          {t.get('ema_trend','')}</span>
                        <span style="background:#070b0f;border:1px solid #7fa8c8;color:#7fa8c8;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          MACD {t.get('macd_cross','')}</span>
                        <span style="background:#070b0f;border:1px solid #ff8c00;color:#ff8c00;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          {t.get('price_vs_vwap','')} VWAP</span>
                        <span style="background:#070b0f;border:1px solid #1e3040;color:#3a6080;
                                     padding:3px 8px;border-radius:10px;font-size:10px;">
                          ATR {t.get('atr14',0):.1f}</span>
                      </div>
                      {f'<div style="margin-top:10px;padding:8px;background:#111920;border-radius:2px;color:#ff8c00;font-size:11px;font-family:Barlow,sans-serif;"><b>📝 Trader Note:</b> {t["custom_reason"]}</div>' if t.get("custom_reason") else ""}
                    </div>""", unsafe_allow_html=True)

                # AI reasoning section
                st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
                if t.get("ai_reasoning"):
                    st.markdown(t["ai_reasoning"])
                else:
                    if st.button(f"🧠 Generate AI Reasoning for #{t['id']}", key=f"ai_journal_{t['id']}"):
                        with st.spinner("Claude is analysing this trade…"):
                            reasoning = _generate_ai_reasoning(t, oc_df, tech_sum, spot)
                            t["ai_reasoning"] = reasoning
                        st.rerun()

    # ── Export ─────────────────────────────────────────────────────
    st.markdown("---")
    ex1, ex2 = st.columns(2)
    all_for_export = st.session_state.pt_closed_trades + st.session_state.pt_open_trades
    with ex1:
        if all_for_export:
            export_df = pd.DataFrame([{
                k: v for k, v in t.items() if k != "ai_reasoning"
            } for t in all_for_export])
            csv_out = export_df.to_csv(index=False)
            st.download_button(
                "📥 Export All Trades CSV",
                data=csv_out,
                file_name=f"paper_trades_{now_ist_dt.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with ex2:
        if all_for_export:
            journal_data = [{
                "id":          t["id"],
                "trade":       f"{t['action']} {t['opt_type']} {t['strike']}",
                "entry":       t["entry_price"],
                "pnl":         t.get("pnl", 0),
                "status":      t.get("exit_reason", t["status"]),
                "ai_reasoning":t.get("ai_reasoning", ""),
                "timestamp":   t["timestamp"],
                "date":        t["date"],
            } for t in all_for_export if t.get("ai_reasoning")]
            if journal_data:
                import json as _json
                jstr = _json.dumps(journal_data, indent=2, ensure_ascii=False)
                st.download_button(
                    "📒 Export AI Journal JSON",
                    data=jstr,
                    file_name=f"ai_journal_{now_ist_dt.strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

# ============================================================
# END OF FILE
# ============================================================