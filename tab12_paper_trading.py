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
# ANTHROPIC CLIENT  (singleton — resolves key from all sources)
# ─────────────────────────────────────────────────────────────
import os as _os
import anthropic as _ant

def _get_anthropic_client() -> _ant.Anthropic:
    if "pt_anthropic_client" in st.session_state:
        return st.session_state.pt_anthropic_client

    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("anthropic", {}).get("api_key")
    except Exception:
        pass
    if not api_key:
        api_key = _os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import config as _cfg
            api_key = getattr(_cfg, "ANTHROPIC_API_KEY", None)
        except Exception:
            pass
    if not api_key:
        raise RuntimeError(
            "Anthropic API key not found. Add it to:\n"
            "  • .streamlit/secrets.toml  →  ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "  • OR set env var           →  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "  • OR add to config.py      →  ANTHROPIC_API_KEY = 'sk-ant-...'"
        )

    client = _ant.Anthropic(api_key=api_key)
    st.session_state.pt_anthropic_client = client
    return client


# ─────────────────────────────────────────────────────────────
# FIX D: CACHED SIGNAL + TECHNICALS COMPUTATION
# These are the two most expensive operations in the tab — an API
# call (fetch_intraday_candles) plus heavy pandas computation
# (compute_technicals, compute_signal_score).
# Without caching they re-run on EVERY Streamlit rerun (every
# button click, widget change, etc.) making the tab very slow.
# TTL=60s means data refreshes every minute while staying fast.
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def _cached_technicals(index_key: str, tf: str = "5minute"):
    """Fetch candles + compute technicals, cached 60 s."""
    try:
        candles = fetch_intraday_candles(index_key, tf)
        tech_df, tech_sum = compute_technicals(candles)
        return tech_sum
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def _cached_signals(oc_hash: str, spot: float, index_key: str,
                    oc_df_json: str):
    """Compute signal scores, cached 60 s per (oc_hash, spot, index)."""
    try:
        oc_df = pd.read_json(oc_df_json)
        bull_s, bear_s, pcr_v, pcr_chg, res_s, sup_s, _ = compute_signal_score(
            oc_df, spot, index_key)
        return bull_s, bear_s, pcr_v
    except Exception:
        return 10, 10, 1.0
# Eliminates all oc_df[oc_df["Strike"] == x] scans (O(n) each)
# and replaces with O(1) .at[strike, col] lookups.
# ─────────────────────────────────────────────────────────────
def _build_oc_index(oc_df: pd.DataFrame) -> pd.DataFrame:
    """Return oc_df indexed by Strike for O(1) lookups."""
    if oc_df.index.name == "Strike":
        return oc_df
    return oc_df.set_index("Strike")


def _ltp_from_index(oc_idx: pd.DataFrame, strike, opt_type: str, fallback: float = 0.0) -> float:
    """Safe O(1) LTP lookup from the pre-indexed DataFrame.
    Always coerces strike to float to match the float-keyed index.
    """
    col = f"{opt_type}_LTP"
    try:
        val = oc_idx.at[float(strike), col]
        return round(float(val), 2) if pd.notna(val) and float(val) > 0 else fallback
    except (KeyError, TypeError, ValueError):
        return fallback


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
        st.session_state.pt_open_trades = []
    if "pt_closed_trades" not in st.session_state:
        st.session_state.pt_closed_trades = []
    if "pt_journal" not in st.session_state:
        st.session_state.pt_journal = []
    if "pt_equity_curve" not in st.session_state:
        st.session_state.pt_equity_curve = [
            {"time": now_ist_dt.strftime("%H:%M"), "equity": DEFAULT_CAPITAL}
        ]
    if "pt_pending_ai" not in st.session_state:
        st.session_state.pt_pending_ai = {}
    if "pt_reset_confirm" not in st.session_state:
        st.session_state.pt_reset_confirm = False
    if "pt_suggest_result" not in st.session_state:
        st.session_state.pt_suggest_result = None
    if "pt_suggest_loading" not in st.session_state:
        st.session_state.pt_suggest_loading = False


# ─────────────────────────────────────────────────────────────
# COST SIMULATION
# ─────────────────────────────────────────────────────────────
def _compute_costs(action: str, premium: float, qty: int, lots: int) -> float:
    brokerage = BROKERAGE_PER_LOT * lots
    stt = (premium * qty * STT_RATE) if action == "SELL" else 0.0
    return round(brokerage + stt, 2)


# ─────────────────────────────────────────────────────────────
# MARGIN ESTIMATOR
# ─────────────────────────────────────────────────────────────
def _estimate_margin(action: str, premium: float, strike: float, lots: int, lot_size: int) -> float:
    if action == "BUY":
        return premium * lots * lot_size
    else:
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
    oc_idx=None,                 # FIX A: accept pre-built index
) -> dict | None:
    lot_size  = get_lot_size(index_key)
    qty       = lots * lot_size

    # FIX A: use pre-built index if available, else build it here
    # Force strike to float so .at[] lookup always works regardless of int/float source
    strike = float(strike)
    if oc_idx is None:
        oc_idx = _build_oc_index(oc_df.assign(Strike=oc_df["Strike"].astype(float)))
    premium = _ltp_from_index(oc_idx, strike, opt_type)

    if premium <= 0:
        # Fallback 1: raw DataFrame scan with float comparison
        col_ltp = f"{opt_type}_LTP"
        row = oc_df[oc_df["Strike"].astype(float) == strike]
        if not row.empty:
            v = row[col_ltp].values[0]
            premium = round(float(v), 2) if pd.notna(v) and float(v) > 0 else 0.0

    if premium <= 0:
        # Fallback 2: try live fetch from API
        try:
            fetched = fetch_ltp(index_key)
            premium = round(float(fetched), 2) if fetched and float(fetched) > 0 else 0.0
        except Exception:
            pass

    if premium <= 0:
        st.error(
            f"❌ LTP is zero for {opt_type} {int(strike)} — option chain may not have this strike. "
            "Try refreshing the option chain from Tab 1."
        )
        return None

    margin    = _estimate_margin(action, premium, strike, lots, lot_size)
    costs     = _compute_costs(action, premium, qty, lots)
    acct      = st.session_state.pt_account

    if margin > acct["available"]:
        st.error(f"❌ Insufficient margin. Need ₹{margin:,.0f}, available ₹{acct['available']:,.0f}")
        return None

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
        "ai_reasoning":  None,
    }

    acct["available"]   = round(acct["available"] - margin, 2)
    acct["total_trades"] += 1
    st.session_state.pt_open_trades.append(trade)
    st.session_state.pt_pending_ai[trade_id] = True
    return trade


# ─────────────────────────────────────────────────────────────
# CLOSE TRADE
# ─────────────────────────────────────────────────────────────
def _close_trade(trade_id: str, oc_df, reason: str = "MANUAL", oc_idx=None):
    trades = st.session_state.pt_open_trades
    idx    = next((i for i, t in enumerate(trades) if t["id"] == trade_id), None)
    if idx is None:
        return
    trade = trades[idx]

    # FIX A: use pre-built index; ensure float Strike key
    if oc_idx is None:
        oc_idx = _build_oc_index(oc_df.assign(Strike=oc_df["Strike"].astype(float)))
    exit_ltp = _ltp_from_index(oc_idx, trade["strike"], trade["opt_type"], trade["ltp"])

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

    acct = st.session_state.pt_account
    acct["available"] = round(acct["available"] + trade["margin_used"] + pnl_net, 2)
    acct["peak"]      = max(acct["peak"], acct["available"])
    if pnl_net >= 0:
        acct["wins"]   += 1
    else:
        acct["losses"] += 1

    st.session_state.pt_closed_trades.append(trade)
    st.session_state.pt_open_trades.pop(idx)

    st.session_state.pt_equity_curve.append({
        "time":   now_ist_dt.strftime("%H:%M"),
        "equity": acct["available"],
    })


# ─────────────────────────────────────────────────────────────
# FIX B: LIVE PNL — single-pass with pre-indexed DF
# Old: N separate DataFrame scans (one per open trade)
# New: one .set_index() call, then O(1) lookups per trade
# ─────────────────────────────────────────────────────────────
def _update_live_pnl(oc_df, oc_idx=None):
    if oc_idx is None:
        oc_idx = _build_oc_index(oc_df.assign(Strike=oc_df["Strike"].astype(float)))

    for trade in st.session_state.pt_open_trades:
        ltp = _ltp_from_index(oc_idx, trade["strike"], trade["opt_type"], trade["ltp"])
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

        if trade["status"] == "OPEN":
            if sl_hit:
                trade["status"] = "SL_PENDING"
            elif tgt_hit:
                trade["status"] = "TGT_PENDING"

    to_close = [(t["id"], "SL HIT")     for t in st.session_state.pt_open_trades if t["status"] == "SL_PENDING"]
    to_close += [(t["id"], "TARGET HIT") for t in st.session_state.pt_open_trades if t["status"] == "TGT_PENDING"]
    for tid, reason in to_close:
        _close_trade(tid, oc_df, reason, oc_idx=oc_idx)


# ─────────────────────────────────────────────────────────────
# AI REASONING GENERATOR
# ─────────────────────────────────────────────────────────────
def _generate_ai_reasoning(trade: dict, oc_df, tech_summary: dict, spot: float) -> str:
    try:
        client = _get_anthropic_client()

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
# TRADE SUGGEST ENGINE
# ─────────────────────────────────────────────────────────────
def _build_suggest_prompt(
    oc_df, spot: float, index_key: str, expiry: str,
    tech_sum: dict, signal_scores: dict,
    perf: dict, closed_trades: list, open_trades: list,
    oc_idx=None,                 # FIX A: accept pre-built index
) -> str:
    atm = get_atm_strike(spot, index_key)
    index_name = INDEX_SHORT.get(index_key, index_key.split("|")[-1])

    # FIX A: use pre-built index for option chain snapshot
    if oc_idx is None:
        oc_idx = _build_oc_index(oc_df.assign(Strike=oc_df["Strike"].astype(float)))

    strikes_all = sorted(oc_df["Strike"].unique().tolist())
    atm_pos = strikes_all.index(atm) if atm in strikes_all else len(strikes_all) // 2
    nearby = strikes_all[max(0, atm_pos - 5): atm_pos + 6]

    oc_rows = []
    for s in nearby:
        try:
            row_vals = oc_idx.loc[s]
            ce_ltp = row_vals.get("CE_LTP", "—")
            pe_ltp = row_vals.get("PE_LTP", "—")
            ce_oi  = row_vals.get("CE_OI",  "—")
            pe_oi  = row_vals.get("PE_OI",  "—")
            ce_iv  = row_vals.get("CE_IV",  "—")
            pe_iv  = row_vals.get("PE_IV",  "—")
        except KeyError:
            ce_ltp = pe_ltp = ce_oi = pe_oi = ce_iv = pe_iv = "—"
        marker = " ← ATM" if s == atm else ""
        oc_rows.append(
            f"  {int(s):>6}{marker:<8}  CE LTP={ce_ltp}  CE OI={ce_oi}  CE IV={ce_iv}  "
            f"|  PE LTP={pe_ltp}  PE OI={pe_oi}  PE IV={pe_iv}"
        )
    oc_snapshot = "\n".join(oc_rows) if oc_rows else "  (option chain data unavailable)"

    hist_lines = []
    for t in (closed_trades or [])[-20:]:
        pnl_tag = f"+₹{t['pnl']:,.0f}" if t.get("pnl", 0) >= 0 else f"-₹{abs(t.get('pnl',0)):,.0f}"
        hist_lines.append(
            f"  [{t.get('date','')} {t.get('timestamp','')}] "
            f"{t.get('action','')} {t.get('opt_type','')} {int(t.get('strike',0))} "
            f"@ ₹{t.get('entry_price',0)} → {pnl_tag} "
            f"({t.get('exit_reason','OPEN')})  "
            f"RSI={t.get('rsi',0):.0f}  PCR={t.get('pcr',1):.3f}  "
            f"Bull={t.get('signal_bull',0)}/20  Bear={t.get('signal_bear',0)}/20"
        )
    hist_block = "\n".join(hist_lines) if hist_lines else "  No closed trade history yet."

    open_lines = []
    for t in (open_trades or []):
        open_lines.append(
            f"  {t.get('action','')} {t.get('opt_type','')} {int(t.get('strike',0))} "
            f"@ ₹{t.get('entry_price',0)}  LTP=₹{t.get('ltp',0)}  "
            f"PnL=₹{t.get('pnl',0):+,.0f}  SL=₹{t.get('sl_price',0)}  Tgt=₹{t.get('target_price',0)}"
        )
    open_block = "\n".join(open_lines) if open_lines else "  No open positions."

    if perf:
        perf_block = (
            f"  Total trades  : {perf.get('total_trades',0)}\n"
            f"  Win rate      : {perf.get('win_rate',0):.1f}%  ({perf.get('wins',0)}W / {perf.get('losses',0)}L)\n"
            f"  Avg win       : ₹{perf.get('avg_win',0):,.0f}   Avg loss: ₹{perf.get('avg_loss',0):,.0f}\n"
            f"  Expectancy    : ₹{perf.get('expectancy',0):+,.0f} per trade\n"
            f"  Profit factor : {perf.get('profit_factor',0):.2f}\n"
            f"  Total P&L     : ₹{perf.get('total_pnl',0):+,.0f}\n"
            f"  Return        : {perf.get('return_pct',0):+.2f}%\n"
            f"  Max drawdown  : {perf.get('max_drawdown',0):.1f}%"
        )
    else:
        perf_block = "  No closed trade history — no performance stats available yet."

    acct = st.session_state.pt_account

    return f"""You are QuantDesk Pro's elite AI trading strategist. Analyse ALL the data below and generate a precise, actionable trade suggestion for the CURRENT market conditions.

════════════════════════════════════════
ACCOUNT STATUS
════════════════════════════════════════
Capital      : ₹{acct['capital']:,.0f}
Available    : ₹{acct['available']:,.0f}
Peak         : ₹{acct['peak']:,.0f}
Total trades : {acct['total_trades']}
Wins / Losses: {acct['wins']} / {acct['losses']}

════════════════════════════════════════
LIVE MARKET  —  {index_name}
════════════════════════════════════════
Spot price   : ₹{spot:,.2f}
ATM strike   : {int(atm)}
Expiry       : {expiry}

OPTION CHAIN (ATM ±5 strikes):
  Strike          CE side                          PE side
{oc_snapshot}

════════════════════════════════════════
SIGNAL SCORES
════════════════════════════════════════
Market flow  : {signal_scores.get('flow','—')}
Bull score   : {signal_scores.get('bull',0)}/20
Bear score   : {signal_scores.get('bear',0)}/20
PCR          : {signal_scores.get('pcr',1.0):.3f}

════════════════════════════════════════
TECHNICALS  (5-min candles)
════════════════════════════════════════
RSI(14)      : {tech_sum.get('rsi14', 'N/A')}
EMA trend    : {tech_sum.get('ema_trend', 'N/A')}
MACD cross   : {tech_sum.get('macd_cross', 'N/A')}
Price vs VWAP: {tech_sum.get('price_vs_vwap', 'N/A')}
ATR(14)      : {tech_sum.get('atr14', 'N/A')} pts
BB condition : {tech_sum.get('bb_condition', 'N/A')}
SuperTrend   : {tech_sum.get('supertrend_bias', 'N/A')}
Day High     : ₹{tech_sum.get('high_of_day', 'N/A')}
Day Low      : ₹{tech_sum.get('low_of_day', 'N/A')}
VWAP         : ₹{tech_sum.get('vwap', 'N/A')}
Candles seen : {tech_sum.get('candles_count', 0)}

════════════════════════════════════════
OPEN POSITIONS
════════════════════════════════════════
{open_block}

════════════════════════════════════════
PERFORMANCE METRICS  (closed trades)
════════════════════════════════════════
{perf_block}

════════════════════════════════════════
TRADE HISTORY  (last 20 closed trades)
════════════════════════════════════════
{hist_block}

════════════════════════════════════════
YOUR TASK
════════════════════════════════════════
Using ALL the above data — live signals, option chain, technicals, historical win/loss patterns, and current account state — provide ONE best trade suggestion RIGHT NOW.

Respond in this EXACT format (use rich markdown):

## 🎯 TRADE SUGGESTION

**Instrument:** [Index name] [Strike] [CE/PE]
**Action:** [BUY / SELL]
**Entry Zone:** ₹[X] – ₹[Y]
**Stop Loss:** ₹[X]  ([N] pts from entry)
**Target:** ₹[X]  ([N] pts from entry)
**Lots:** [N]  (based on account size and 2% risk rule)
**R:R Ratio:** 1:[X]
**Confidence:** [High / Medium / Low]

---

## 📊 MARKET CONTEXT
3–4 sentences explaining what the market is doing RIGHT NOW and why this is the right moment.

## 🔍 WHY THIS TRADE
5–7 bullet points, each citing a specific data point from the analysis above. Reference exact numbers (RSI value, PCR level, OI at strike, signal scores, etc.).

## 📈 ENTRY STRATEGY
Precise entry guidance — where to enter, what confirmation to wait for, what invalidates the setup before entry.

## ⚠️ RISK NOTES
- Any open position conflicts or portfolio concentration risk
- What would make this trade wrong
- Account drawdown context

## 📋 PATTERN INSIGHTS  (from your trade history)
Based on the trader's closed trade history above, identify 2–3 specific patterns — e.g. "Your BUY CE trades at RSI>65 have an 80% loss rate — current RSI is [X], so this CE buy has elevated risk." Be data-specific.

## 🚦 FINAL VERDICT
One paragraph. Go or No-Go, and why.

Be direct. Be specific. Use actual numbers. No generic advice."""


def _generate_trade_suggestion(
    oc_df, spot: float, index_key: str, expiry: str,
    tech_sum: dict, signal_scores: dict,
    oc_idx=None,
) -> dict:
    try:
        client = _get_anthropic_client()
        perf          = _calc_performance()
        closed_trades = st.session_state.pt_closed_trades
        open_trades   = st.session_state.pt_open_trades

        prompt = _build_suggest_prompt(
            oc_df, spot, index_key, expiry,
            tech_sum, signal_scores,
            perf, closed_trades, open_trades,
            oc_idx=oc_idx,
        )

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()

        action     = "BUY"  if "**Action:** BUY"  in text else "SELL" if "**Action:** SELL" in text else "—"
        confidence = "High" if "High"  in text[:600] else "Medium" if "Medium" in text[:600] else "Low"

        return {
            "text":       text,
            "action":     action,
            "confidence": confidence,
            "generated":  now_ist_dt.strftime("%H:%M:%S"),
            "spot":       spot,
            "index":      INDEX_SHORT.get(index_key, index_key.split("|")[-1]),
        }
    except Exception as e:
        return {
            "text":       f"⚠️ Trade suggestion unavailable: {e}",
            "action":     "—",
            "confidence": "—",
            "generated":  now_ist_dt.strftime("%H:%M:%S"),
            "spot":       spot,
            "index":      INDEX_SHORT.get(index_key, index_key.split("|")[-1]),
        }


def _render_trade_suggest_section(
    oc_df, spot: float, index_key: str, expiry: str,
    tech_sum: dict, signal_scores: dict,
    perf: dict,            # FIX C: accept pre-computed perf — no double call
    oc_idx=None,
):
    st.markdown("---")

    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:20px;
                  font-weight:800;letter-spacing:3px;color:#e8f4ff;">
        🤖 AI <span style="color:#c084fc;">TRADE SUGGEST</span>
      </div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                  letter-spacing:1.5px;color:#3a6080;border:1px solid #2a3a4a;
                  padding:2px 8px;border-radius:2px;">BETA · EDUCATIONAL ONLY</div>
    </div>
    <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#3a6080;margin-bottom:12px;">
      Analyses live option chain · technicals · your full trade history · account state → one actionable suggestion
    </div>""", unsafe_allow_html=True)

    # FIX C: perf already computed by caller — no second _calc_performance() call
    closed_n  = len(st.session_state.pt_closed_trades)
    open_n    = len(st.session_state.pt_open_trades)
    bull_s    = signal_scores.get("bull", 0)
    bear_s    = signal_scores.get("bear", 0)
    pcr_v     = signal_scores.get("pcr", 1.0)
    flow      = signal_scores.get("flow", "Range")
    rsi_val   = tech_sum.get("rsi14", 50)
    flow_c    = {"Bullish":"#00e676","Bearish":"#ff3d57","Range":"#ffd600","Choppy":"#7fa8c8"}.get(flow,"#7fa8c8")

    st.markdown(f"""
    <div style="background:#070b0f;border:1px solid #1a2a3a;border-radius:3px;
                padding:10px 14px;margin-bottom:12px;
                display:flex;flex-wrap:wrap;gap:14px;align-items:center;
                font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:1px;">
      <span style="color:#3a6080;">FEEDING INTO AI:</span>
      <span style="color:#00d4ff;">📊 {closed_n} closed trades</span>
      <span style="color:#ffd600;">📂 {open_n} open positions</span>
      <span style="color:{flow_c};">● {flow.upper()}</span>
      <span style="color:#00e676;">▲ BULL {bull_s}/20</span>
      <span style="color:#ff3d57;">▼ BEAR {bear_s}/20</span>
      <span style="color:#c084fc;">PCR {pcr_v:.3f}</span>
      <span style="color:#7fa8c8;">RSI {rsi_val:.0f}</span>
      <span style="color:#3a6080;">SPOT ₹{spot:,.0f}</span>
      {'<span style="color:#00e676;">✓ PERF STATS INCLUDED</span>' if perf else '<span style="color:#3a6080;">⚠ NO HISTORY YET</span>'}
    </div>""", unsafe_allow_html=True)

    btn_col, info_col = st.columns([1, 2])

    with btn_col:
        if st.button(
            "🤖 Generate Trade Suggestion",
            use_container_width=True,
            type="primary",
            key="pt_suggest_btn",
        ):
            with st.spinner("🧠 Fetching live data · Analysing option chain · Building AI prompt…"):
                result = _generate_trade_suggestion(
                    oc_df, spot, index_key, expiry, tech_sum, signal_scores,
                    oc_idx=oc_idx,
                )
                st.session_state.pt_suggest_result = result
            st.rerun()

        if st.session_state.pt_suggest_result:
            if st.button("🗑️ Clear Suggestion", use_container_width=True, key="pt_suggest_clear"):
                st.session_state.pt_suggest_result = None
                st.rerun()

    with info_col:
        res = st.session_state.pt_suggest_result
        if res:
            action_c = "#00e676" if res["action"] == "BUY" else "#ff3d57" if res["action"] == "SELL" else "#7fa8c8"
            conf_c   = "#00e676" if res["confidence"] == "High" else "#ffd600" if res["confidence"] == "Medium" else "#ff3d57"
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #2a3a4a;border-radius:3px;
                        padding:8px 14px;font-family:'JetBrains Mono',monospace;font-size:11px;">
              <span style="color:#3a6080;">Last generated:</span>
              <span style="color:#e8f4ff;margin-left:8px;">{res['generated']}</span>
              <span style="margin-left:16px;background:#0d1820;border:1px solid {action_c};
                           color:{action_c};padding:2px 10px;border-radius:10px;
                           font-family:'Barlow Condensed',sans-serif;font-size:12px;
                           letter-spacing:1px;font-weight:700;">{res['action']}</span>
              <span style="margin-left:8px;background:#0d0a18;border:1px solid {conf_c};
                           color:{conf_c};padding:2px 10px;border-radius:10px;
                           font-family:'Barlow Condensed',sans-serif;font-size:12px;
                           letter-spacing:1px;">CONF: {res['confidence'].upper()}</span>
              <span style="margin-left:8px;color:#3a6080;">{res['index']} @ ₹{res['spot']:,.0f}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#070b0f;border:1px dashed #1e3040;border-radius:3px;
                        padding:8px 14px;color:#3a6080;
                        font-family:'Barlow Condensed',sans-serif;font-size:12px;
                        letter-spacing:1px;">
              NO SUGGESTION GENERATED YET — CLICK THE BUTTON TO ANALYSE ALL DATA
            </div>""", unsafe_allow_html=True)

    res = st.session_state.pt_suggest_result
    if res and res.get("text"):
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0a0f1a;border:1px solid #2a3a5a;border-left:3px solid #c084fc;
                    border-radius:3px;padding:18px 20px;margin-top:4px;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                      letter-spacing:2px;color:#c084fc;margin-bottom:10px;">
            AI TRADE SUGGESTION — EDUCATIONAL USE ONLY · NOT FINANCIAL ADVICE
          </div>""", unsafe_allow_html=True)
        st.markdown(res["text"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.info(
            "💡 Review the suggestion carefully. Use the **Order Entry** panel on the left "
            "to place the trade manually with your own SL/Target/Lots. "
            "This is a simulation — no real money at risk.",
            icon="ℹ️",
        )


# ─────────────────────────────────────────────────────────────
# FIX C: PERFORMANCE METRICS — cached per render
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
    entry = trade["entry_price"]
    tgt   = trade["target_price"]
    sl    = trade["sl_price"]
    ltp   = trade.get("ltp", entry)
    if trade["action"] == "BUY":
        prog    = min(max((ltp - entry) / (tgt - entry + 0.001) * 100, 0), 100)
        sl_prog = min(max((entry - ltp) / (entry - sl + 0.001) * 100, 0), 100)
    else:
        prog    = min(max((entry - ltp) / (entry - tgt + 0.001) * 100, 0), 100)
        sl_prog = min(max((ltp - entry) / (sl - entry + 0.001) * 100, 0), 100)
    bar_c    = "#00e676" if prog > 50 else "#ffd600" if prog > 20 else "#7fa8c8"
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
    ec = st.session_state.pt_equity_curve
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

    oc_df   = st.session_state.get("current_option_chain", pd.DataFrame())
    spot    = st.session_state.get("current_spot_price", 0)
    sel_idx = st.session_state.get("current_selected_index", "")
    expiry  = st.session_state.get("oc_expiry_select", "")

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

    if oc_df is None or (isinstance(oc_df, pd.DataFrame) and oc_df.empty) or not spot:
        st.warning("⏳ Load the Option Chain from **Tab 1** first — paper trading needs live data.")
        return

    # ── FIX A: Build indexed DataFrame ONCE — force float Strike key ────────
    # Float key ensures .at[strike, col] always matches regardless of whether
    # the strike value comes in as int or float from selectbox widgets.
    oc_idx = _build_oc_index(oc_df.assign(Strike=oc_df["Strike"].astype(float)))

    # ── FIX B: Live PnL with the pre-built index ──────────────────────────
    _update_live_pnl(oc_df, oc_idx=oc_idx)

    open_pnl   = sum(t.get("pnl", 0) for t in st.session_state.pt_open_trades)
    closed_pnl = sum(t.get("pnl", 0) for t in st.session_state.pt_closed_trades)

    st.markdown(_header_html(st.session_state.pt_account, open_pnl, closed_pnl),
                unsafe_allow_html=True)

    # ── FIX D: Use cached signal + technicals — no API call on every rerun ──
    # Build a cheap hash from spot + oc shape so cache invalidates when data changes
    _signals_status = st.empty()
    try:
        _oc_hash = f"{sel_idx}_{len(oc_df)}_{int(spot)}"
        _oc_json = oc_df.to_json()
        _signals_status.caption("⚡ Loading signals…")
        bull_s, bear_s, pcr_v = _cached_signals(_oc_hash, round(spot, 0), sel_idx, _oc_json)
        tech_sum = _cached_technicals(sel_idx, "5minute")
        market_flow = (
            "Bullish" if bull_s - bear_s >= 5
            else "Bearish" if bear_s - bull_s >= 5
            else "Range" if abs(bull_s - bear_s) <= 3
            else "Choppy"
        )
        _signals_status.empty()   # clear the loading caption
    except Exception:
        bull_s, bear_s, pcr_v, market_flow = 10, 10, 1.0, "Range"
        tech_sum = {}
        _signals_status.empty()

    signal_scores = {"bull": bull_s, "bear": bear_s, "pcr": pcr_v, "flow": market_flow}

    # ── FIX C: Compute performance ONCE — passed to all consumers ─────────
    perf = _calc_performance()

    # ── MAIN LAYOUT ───────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.1, 1.9], gap="medium")

    # ══════════════════════════════════════════════════════
    # LEFT — ORDER ENTRY PANEL
    # ══════════════════════════════════════════════════════
    with left_col:
        section_header("⚡ Order Entry")

        atm = get_atm_strike(spot, sel_idx)
        strikes = sorted(oc_df["Strike"].unique().tolist())

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

        # FIX A: O(1) LTP lookup
        ltp_now    = _ltp_from_index(oc_idx, strike_sel, opt_type)
        lot_sz     = get_lot_size(sel_idx)
        qty_tot    = lots_sel * lot_sz
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

        col_sl, col_tgt = st.columns(2)
        with col_sl:
            sl_pts  = st.number_input("SL (pts)", value=15, min_value=1, max_value=200, key="pt_sl")
        with col_tgt:
            tgt_pts = st.number_input("Target (pts)", value=30, min_value=1, max_value=500, key="pt_tgt")

        rr   = tgt_pts / max(sl_pts, 1)
        rr_c = "#00e676" if rr >= 2 else "#ffd600" if rr >= 1.5 else "#ff3d57"
        st.markdown(f"""<div style="text-align:center;font-family:'Barlow Condensed',sans-serif;
            font-size:14px;font-weight:700;color:{rr_c};letter-spacing:1px;margin:4px 0;">
            R:R = 1:{rr:.1f}</div>""", unsafe_allow_html=True)

        custom_reason = st.text_area(
            "Trade Reason (optional — will be included in AI analysis)",
            placeholder="e.g. VWAP reclaim after SSL sweep, expecting CE unwinding...",
            height=60, key="pt_reason"
        )

        # Signal context display (uses pre-computed signals)
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

        mkt_label = "EXECUTE PAPER TRADE" if MARKET_OPEN else "SIMULATE (MKT CLOSED)"
        if st.button(f"🚀 {mkt_label}", use_container_width=True, type="primary", key="pt_exec"):
            new_trade = _execute_paper_trade(
                oc_df, spot, sel_idx, expiry,
                strike_sel, opt_type, action, lots_sel,
                sl_pts, tgt_pts,
                tech_sum, signal_scores, custom_reason,
                oc_idx=oc_idx,      # FIX A: pass pre-built index
            )
            if new_trade:
                st.success(f"✅ #{new_trade['id']} executed — {action} {opt_type} {int(strike_sel)} @ ₹{ltp_now}")
                st.rerun()

        st.markdown("---")
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
                        if leg_strike not in strikes:
                            continue
                        _execute_paper_trade(
                            oc_df, spot, sel_idx, expiry,
                            leg_strike, leg_type, leg_action, 1,
                            leg_sl, leg_tgt,
                            tech_sum, signal_scores,
                            f"Template: {tname.replace(chr(10),' ')}",
                            oc_idx=oc_idx,
                        )
                    st.rerun()

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
                st.session_state.pt_account["peak"] = max(
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
                close_col, ai_col, _ = st.columns([1, 1.5, 3])
                with close_col:
                    if st.button(f"✕ Close #{trade['id']}", key=f"close_{trade['id']}"):
                        _close_trade(trade["id"], oc_df, "MANUAL", oc_idx=oc_idx)
                        st.rerun()
                with ai_col:
                    if st.button(f"🧠 AI Reasoning #{trade['id']}", key=f"ai_btn_{trade['id']}"):
                        with st.spinner("Generating AI analysis…"):
                            reasoning = _generate_ai_reasoning(trade, oc_df, tech_sum, spot)
                            trade["ai_reasoning"] = reasoning
                        st.session_state.pt_pending_ai.pop(trade["id"], None)
                        st.rerun()
                    if trade.get("ai_reasoning"):
                        with st.expander(f"📖 AI Journal #{trade['id']}", expanded=False):
                            st.markdown(trade["ai_reasoning"])

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
            for t in closed_trades[::-1][:15]:
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
    # TRADE SUGGEST ENGINE
    # ══════════════════════════════════════════════════════
    _render_trade_suggest_section(
        oc_df, spot, sel_idx, expiry, tech_sum, signal_scores,
        perf=perf,          # FIX C: pass pre-computed perf
        oc_idx=oc_idx,      # FIX A: pass pre-built index
    )

    # ══════════════════════════════════════════════════════
    # BOTTOM: FULL ANALYTICS DASHBOARD
    # ══════════════════════════════════════════════════════
    st.markdown("---")
    section_header("📊 Performance Analytics",
                   "Equity curve · Win stats · P&L distribution · Trade journal")

    # FIX C: reuse already-computed perf — no third call
    if perf:
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