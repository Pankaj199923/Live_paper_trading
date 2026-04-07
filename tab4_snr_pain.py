import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime
from config import (ACCESS_TOKEN, IST, now_ist, now_ist_dt, MARKET_OPEN,
                    df_indices, INDEX_SHORT, LOT_SIZES,
                    BASE_DIR, TRADE_FILE, TODAY_TRADES_FILE, CLOSED_POS_FILE,
                    AI_LOG_FILE, SNAPSHOT_DIR, instrument_df)
from utils import (load_csv_safe, load_csv_as_list, save_list_to_csv,
                   get_atm_strike, get_lot_size, idx_short, compute_grand_total,
                   pnl_color, pnl_badge, calculate_net_book, close_position,
                   calculate_max_pain, calculate_portfolio_greeks,
                   save_oc_snapshot, list_daily_files, list_minute_times, load_snapshot,
                   section_header, metric_card, metrics_row, flow_badge, score_bar)
from api import fetch_ltp, fetch_available_expiries, fetch_option_chain, fetch_intraday_candles
from analytics import (bs_price, bs_greeks, implied_vol_newton, calculate_gamma_bs,
                       compute_signal_score, generate_ai_trade, check_alerts, call_claude_trade_setup)
from chart_utils import (compute_technicals, compute_order_flow, detect_liquidity_sweeps,
                         detect_order_blocks, detect_fvg, detect_bos_choch, get_order_flow_summary)

import anthropic as _anthropic
import os as _os


# ──────────────────────────────────────────────────────────────
# ANTHROPIC CLIENT
# ──────────────────────────────────────────────────────────────
def _get_client():
    if "t4_ant_client" in st.session_state:
        return st.session_state.t4_ant_client
    key = None
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("anthropic", {}).get("api_key")
    except Exception:
        pass
    if not key:
        key = _os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            import config as _cfg
            key = getattr(_cfg, "ANTHROPIC_API_KEY", None)
        except Exception:
            pass
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in secrets/env.")
    client = _anthropic.Anthropic(api_key=key)
    st.session_state.t4_ant_client = client
    return client


# ──────────────────────────────────────────────────────────────
# COMPUTE STRUCTURED MARKET DATA FOR PROMPT
# ──────────────────────────────────────────────────────────────
def _build_market_context(oc, spot, index_key, expiry, max_pain_strike, pain_dict):
    """Extract rich structured context from the option chain for the AI prompt."""
    atm = get_atm_strike(spot, index_key)
    idx_name = INDEX_SHORT.get(index_key, index_key.split("|")[-1])

    total_ce_oi  = float(oc["CE_OI"].sum())
    total_pe_oi  = float(oc["PE_OI"].sum())
    total_ce_vol = float(oc["CE_Volume"].sum())
    total_pe_vol = float(oc["PE_Volume"].sum())
    pcr          = total_pe_oi / max(total_ce_oi, 1)

    # Top OI walls
    top_ce = oc.nlargest(5, "CE_OI")[["Strike", "CE_OI", "CE_LTP", "CE_IV"]].to_dict("records")
    top_pe = oc.nlargest(5, "PE_OI")[["Strike", "PE_OI", "PE_LTP", "PE_IV"]].to_dict("records")

    # ATM data
    atm_row = oc[oc["Strike"] == atm]
    atm_data = {}
    if not atm_row.empty:
        r = atm_row.iloc[0]
        atm_data = {
            "ce_ltp":    round(float(r.get("CE_LTP", 0)), 2),
            "pe_ltp":    round(float(r.get("PE_LTP", 0)), 2),
            "ce_oi":     int(r.get("CE_OI", 0)),
            "pe_oi":     int(r.get("PE_OI", 0)),
            "ce_oi_chg": int(r.get("CE_OI_Change", 0)),
            "pe_oi_chg": int(r.get("PE_OI_Change", 0)),
            "ce_iv":     round(float(r.get("CE_IV", 0)), 2),
            "pe_iv":     round(float(r.get("PE_IV", 0)), 2),
            "ce_vol":    int(r.get("CE_Volume", 0)),
            "pe_vol":    int(r.get("PE_Volume", 0)),
        }

    # Nearby strikes context (ATM ±4)
    strikes_sorted = sorted(oc["Strike"].unique())
    try:
        ai = strikes_sorted.index(atm)
        nearby_strikes = strikes_sorted[max(0, ai-4): ai+5]
    except ValueError:
        nearby_strikes = strikes_sorted[:9]

    nearby_data = []
    for s in nearby_strikes:
        row = oc[oc["Strike"] == s]
        if row.empty:
            continue
        r = row.iloc[0]
        nearby_data.append({
            "strike":    int(s),
            "atm":       s == atm,
            "ce_ltp":    round(float(r.get("CE_LTP", 0)), 2),
            "ce_oi_L":   round(float(r.get("CE_OI", 0)) / 1e5, 2),
            "ce_oi_chg": int(r.get("CE_OI_Change", 0)),
            "ce_iv":     round(float(r.get("CE_IV", 0)), 2),
            "pe_ltp":    round(float(r.get("PE_LTP", 0)), 2),
            "pe_oi_L":   round(float(r.get("PE_OI", 0)) / 1e5, 2),
            "pe_oi_chg": int(r.get("PE_OI_Change", 0)),
            "pe_iv":     round(float(r.get("PE_IV", 0)), 2),
        })

    # Max pain
    pain_dist = abs(spot - max_pain_strike)
    pain_dir  = "above" if spot > max_pain_strike else "below"

    # OI change momentum
    total_ce_chg = float(oc["CE_OI_Change"].sum())
    total_pe_chg = float(oc["PE_OI_Change"].sum())

    # IV skew
    avg_ce_iv = float(oc["CE_IV"].mean()) if "CE_IV" in oc.columns else 0
    avg_pe_iv = float(oc["PE_IV"].mean()) if "PE_IV" in oc.columns else 0

    # Top resistance (CE OI above spot)
    above_spot = oc[oc["Strike"] > spot].nlargest(3, "CE_OI")
    below_spot = oc[oc["Strike"] < spot].nlargest(3, "PE_OI")
    main_res = float(above_spot["Strike"].iloc[0]) if not above_spot.empty else 0
    main_sup = float(below_spot["Strike"].iloc[0]) if not below_spot.empty else 0

    # Battle zone detection — same strike heavy on both sides
    oc_copy = oc.copy()
    oc_copy["battle_score"] = oc_copy["CE_OI"] + oc_copy["PE_OI"]
    battle_row = oc_copy.nlargest(1, "battle_score")
    battle_strike = int(battle_row["Strike"].iloc[0]) if not battle_row.empty else atm

    return {
        "index_name":     idx_name,
        "expiry":         expiry,
        "spot":           round(spot, 2),
        "atm":            int(atm),
        "pcr":            round(pcr, 3),
        "total_ce_oi_L":  round(total_ce_oi / 1e5, 1),
        "total_pe_oi_L":  round(total_pe_oi / 1e5, 1),
        "total_ce_vol_K": round(total_ce_vol / 1e3, 1),
        "total_pe_vol_K": round(total_pe_vol / 1e3, 1),
        "total_ce_chg_K": round(total_ce_chg / 1e3, 1),
        "total_pe_chg_K": round(total_pe_chg / 1e3, 1),
        "avg_ce_iv":      round(avg_ce_iv, 2),
        "avg_pe_iv":      round(avg_pe_iv, 2),
        "main_res":       int(main_res),
        "main_sup":       int(main_sup),
        "max_pain":       int(max_pain_strike),
        "pain_dist":      round(pain_dist, 0),
        "pain_dir":       pain_dir,
        "battle_strike":  battle_strike,
        "atm_data":       atm_data,
        "nearby_data":    nearby_data,
        "top_ce_walls":   [{"strike": int(x["Strike"]), "oi_L": round(float(x["CE_OI"])/1e5, 1), "ltp": round(float(x["CE_LTP"]),2), "iv": round(float(x["CE_IV"]),2)} for x in top_ce],
        "top_pe_walls":   [{"strike": int(x["Strike"]), "oi_L": round(float(x["PE_OI"])/1e5, 1), "ltp": round(float(x["PE_LTP"]),2), "iv": round(float(x["PE_IV"]),2)} for x in top_pe],
    }


# ──────────────────────────────────────────────────────────────
# AI ANALYSIS PROMPT + CALL
# ──────────────────────────────────────────────────────────────
def _call_ai_analysis(ctx: dict) -> dict:
    """Call Claude to generate a structured pro-level market analysis."""

    nearby_table = "\n".join([
        f"  {'★ATM' if r['atm'] else '    '} {r['strike']:>6}  |  "
        f"CE LTP={r['ce_ltp']:>7}  OI={r['ce_oi_L']:>5}L  OI_D={r['ce_oi_chg']:>+8}  IV={r['ce_iv']:>5}%  "
        f"|  PE LTP={r['pe_ltp']:>7}  OI={r['pe_oi_L']:>5}L  OI_D={r['pe_oi_chg']:>+8}  IV={r['pe_iv']:>5}%"
        for r in ctx["nearby_data"]
    ])

    ce_walls_str = "  ".join([f"{x['strike']}({x['oi_L']}L)" for x in ctx["top_ce_walls"]])
    pe_walls_str = "  ".join([f"{x['strike']}({x['oi_L']}L)" for x in ctx["top_pe_walls"]])

    prompt = f"""You are an elite NSE F&O institutional trader with 20 years of experience reading option chains. 
Analyze the following LIVE option chain data and give a COMPLETE, structured, actionable pro-level market analysis.

━━━ LIVE SNAPSHOT ━━━
Index        : {ctx['index_name']}
Spot Price   : ₹{ctx['spot']:,.2f}
ATM Strike   : {ctx['atm']}
Expiry       : {ctx['expiry']}
Time         : {now_ist_dt.strftime('%H:%M:%S')} IST

━━━ MACRO OI PICTURE ━━━
Total CE OI  : {ctx['total_ce_oi_L']}L   |  CE Volume: {ctx['total_ce_vol_K']}K
Total PE OI  : {ctx['total_pe_oi_L']}L   |  PE Volume: {ctx['total_pe_vol_K']}K
CE OI Change : {ctx['total_ce_chg_K']:+.1f}K    |  PE OI Change: {ctx['total_pe_chg_K']:+.1f}K
PCR (OI)     : {ctx['pcr']:.3f}
Avg CE IV    : {ctx['avg_ce_iv']:.1f}%   |  Avg PE IV: {ctx['avg_pe_iv']:.1f}%

━━━ KEY LEVELS ━━━
Main Resistance  : {ctx['main_res']} (highest CE OI above spot)
Main Support     : {ctx['main_sup']} (highest PE OI below spot)
Max Pain Strike  : {ctx['max_pain']} (spot {ctx['pain_dir']} by {ctx['pain_dist']:.0f} pts)
Battle Zone      : {ctx['battle_strike']} (heaviest combined CE+PE OI)

Top CE Walls (Resistance): {ce_walls_str}
Top PE Walls (Support)   : {pe_walls_str}

━━━ ATM DATA ━━━
CE LTP: ₹{ctx['atm_data'].get('ce_ltp',0)}  |  PE LTP: ₹{ctx['atm_data'].get('pe_ltp',0)}
CE OI : {ctx['atm_data'].get('ce_oi',0):,}   |  PE OI : {ctx['atm_data'].get('pe_oi',0):,}
CE OI Chg: {ctx['atm_data'].get('ce_oi_chg',0):+,}  |  PE OI Chg: {ctx['atm_data'].get('pe_oi_chg',0):+,}
CE IV : {ctx['atm_data'].get('ce_iv',0):.1f}%   |  PE IV : {ctx['atm_data'].get('pe_iv',0):.1f}%
CE Vol: {ctx['atm_data'].get('ce_vol',0):,}   |  PE Vol: {ctx['atm_data'].get('pe_vol',0):,}

━━━ ATM ±4 STRIKES (Full Chain Context) ━━━
  Flag  Strike  |  ← CALL SIDE →                              |  ← PUT SIDE →
{nearby_table}

━━━ YOUR TASK ━━━
Respond ONLY with a raw JSON object (no markdown, no backticks). Structure:
{{
  "market_view": "BULLISH | BEARISH | RANGE | CHOPPY",
  "confidence": "HIGH | MEDIUM | LOW",
  "range_low": <number>,
  "range_high": <number>,
  "key_resistance": [<strike1>, <strike2>, <strike3>],
  "key_support": [<strike1>, <strike2>, <strike3>],
  "battle_zone": "<low>-<high>",
  "trap_zone": "<description of where price churns and traps retail>",
  "call_side_reading": "<3-4 sentences: what CE OI tells us — is resistance being built or unwound? Any walls shifting?>",
  "put_side_reading": "<3-4 sentences: what PE OI tells us — is support being built or unwound? Any capitulation?>",
  "smart_money_signal": "<2-3 sentences: what institutions are doing based on OI build vs unwind patterns>",
  "pcr_interpretation": "<2 sentences: what current PCR and its direction means>",
  "max_pain_gravity": "<2 sentences: how max pain will likely pull price and what it means for expiry day>",
  "iv_skew_reading": "<2 sentences: what the IV differential between CE and PE tells us about fear direction>",
  "trade_setups": [
    {{
      "name": "<strategy name>",
      "type": "RANGE | BULLISH | BEARISH | BREAKOUT",
      "action": "<BUY CE | BUY PE | SELL CE | SELL PE | SPREAD>",
      "strike": <number>,
      "entry_condition": "<precise entry condition with levels>",
      "sl": "<stop loss level/condition>",
      "target": "<target level/condition>",
      "reason": "<why this setup makes sense now given the OI data>"
    }}
  ],
  "avoid_zone": "<exact price zone to avoid with reason>",
  "breakout_trigger": "<what price/OI event would confirm a breakout above resistance>",
  "breakdown_trigger": "<what price/OI event would confirm a breakdown below support>",
  "final_verdict": "<3-4 bold, punchy sentences. Clear actionable take. What to DO and what to AVOID. Use specific price levels.>"
}}"""

    client = _get_client()
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.content[0].text.strip()
    # Strip markdown fences if present
    if "```" in raw:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        raw = m.group(1) if m else raw.split("```")[1].lstrip("json").strip()
    b_start = raw.find("{")
    b_end   = raw.rfind("}") + 1
    if b_start >= 0 and b_end > b_start:
        raw = raw[b_start:b_end]
    return json.loads(raw)


# ──────────────────────────────────────────────────────────────
# RENDER THE AI ANALYSIS PANEL
# ──────────────────────────────────────────────────────────────
def _render_ai_panel(analysis: dict, ctx: dict):
    mv     = analysis.get("market_view", "RANGE")
    conf   = analysis.get("confidence", "MEDIUM")
    mv_cfg = {
        "BULLISH": ("#00e676", "#051a0e", "📈", "Bull Put Spread / CE Buy on dips"),
        "BEARISH": ("#ff3d57", "#1a0508", "📉", "Bear Call Spread / PE Buy on rallies"),
        "RANGE":   ("#ffd600", "#1a1000", "↔",  "Iron Condor / Short Straddle / Sell premium"),
        "CHOPPY":  ("#7fa8c8", "#001020", "〰",  "Iron Fly / Avoid directional trades"),
    }
    fg, bg, icon, strat = mv_cfg.get(mv, ("#7fa8c8", "#0d1117", "—", "—"))
    conf_c = "#00e676" if conf == "HIGH" else "#ffd600" if conf == "MEDIUM" else "#ff3d57"

    rl  = analysis.get("range_low",  ctx["main_sup"])
    rh  = analysis.get("range_high", ctx["main_res"])
    res = analysis.get("key_resistance", [ctx["main_res"]])
    sup = analysis.get("key_support",    [ctx["main_sup"]])

    # ── HERO CARD ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {fg};border-left:5px solid {fg};
                border-radius:6px;padding:18px 24px;margin:10px 0 16px 0;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
        <div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:2.5px;color:{fg};margin-bottom:6px;">
            🤖 AI MARKET SIGNAL  ·  {ctx['index_name']}  ·  {now_ist_dt.strftime('%H:%M:%S')} IST</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:36px;
                      font-weight:800;letter-spacing:4px;color:{fg};">
            {icon} {mv}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                      color:#7fa8c8;margin-top:6px;">
            → Best Strategy: <b style="color:{fg};">{strat}</b></div>
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;text-align:center;">
          <div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                        letter-spacing:2px;color:#7fa8c8;">CONFIDENCE</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:20px;
                        font-weight:800;color:{conf_c};letter-spacing:2px;">{conf}</div>
          </div>
          <div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                        letter-spacing:2px;color:#7fa8c8;">RANGE</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                        font-weight:700;color:#e8f4ff;">{int(rl):,} — {int(rh):,}</div>
          </div>
          <div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                        letter-spacing:2px;color:#7fa8c8;">PCR</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                        font-weight:700;color:#c084fc;">{ctx['pcr']:.3f}</div>
          </div>
          <div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                        letter-spacing:2px;color:#7fa8c8;">MAX PAIN</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                        font-weight:700;color:#ffd600;">{ctx['max_pain']:,}</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── KEY LEVELS ROW ───────────────────────────────────────────────────
    res_str = "  |  ".join([f"<b style='color:#ff3d57;'>R{i+1}: {int(r):,}</b>" for i, r in enumerate(res[:3])])
    sup_str = "  |  ".join([f"<b style='color:#00e676;'>S{i+1}: {int(s):,}</b>" for i, s in enumerate(sup[:3])])
    bz      = analysis.get("battle_zone", f"{ctx['battle_strike']}")
    trap    = analysis.get("trap_zone", "—")
    avoid   = analysis.get("avoid_zone", "—")

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;">
      <div style="background:#1a0508;border:1px solid #ff3d57;border-radius:4px;padding:12px 14px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:2px;color:#ff3d57;margin-bottom:6px;">🔴 RESISTANCE LEVELS</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.8;">
          {res_str}</div>
      </div>
      <div style="background:#051a0e;border:1px solid #00e676;border-radius:4px;padding:12px 14px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:2px;color:#00e676;margin-bottom:6px;">🟢 SUPPORT LEVELS</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.8;">
          {sup_str}</div>
      </div>
      <div style="background:#1a1200;border:1px solid #ffd600;border-radius:4px;padding:12px 14px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:2px;color:#ffd600;margin-bottom:6px;">⚔️ BATTLE ZONE</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700;color:#ffd600;">
          {bz}</div>
        <div style="font-family:'Barlow',sans-serif;font-size:10px;color:#7fa8c8;margin-top:4px;">
          Both CE & PE walls heavy</div>
      </div>
      <div style="background:#1a0a10;border:1px solid #ff8c00;border-radius:4px;padding:12px 14px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:2px;color:#ff8c00;margin-bottom:6px;">❌ AVOID ZONE</div>
        <div style="font-family:'Barlow',sans-serif;font-size:11px;color:#ff8c00;">
          {avoid[:80]}{'…' if len(avoid)>80 else ''}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 3-COL ANALYSIS CARDS ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    def _card(title, color, icon_e, text, extra=""):
        return f"""
        <div style="background:#0d1117;border:1px solid #1e3040;border-top:2px solid {color};
                    border-radius:4px;padding:14px 16px;height:100%;box-sizing:border-box;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:2px;color:{color};margin-bottom:8px;">{icon_e} {title}</div>
          <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#c8dff0;
                      line-height:1.6;">{text}</div>
          {f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#7fa8c8;margin-top:8px;padding-top:8px;border-top:1px solid #1e3040;">{extra}</div>' if extra else ''}
        </div>"""

    with col1:
        st.markdown(_card("CALL SIDE (CE) READING", "#ff3d57", "🔴",
                          analysis.get("call_side_reading", "—")),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(_card("PUT SIDE (PE) READING", "#00e676", "🟢",
                          analysis.get("put_side_reading", "—")),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(_card("SMART MONEY SIGNAL", "#c084fc", "🏦",
                          analysis.get("smart_money_signal", "—")),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(_card("PCR INTERPRETATION", "#c084fc", "📊",
                          analysis.get("pcr_interpretation", "—"),
                          f"PCR: {ctx['pcr']:.3f}  |  CE OI: {ctx['total_ce_oi_L']}L  |  PE OI: {ctx['total_pe_oi_L']}L"),
                    unsafe_allow_html=True)
    with col5:
        st.markdown(_card("MAX PAIN GRAVITY", "#ffd600", "🧲",
                          analysis.get("max_pain_gravity", "—"),
                          f"Max Pain: {ctx['max_pain']:,}  |  Spot {ctx['pain_dir']} by {ctx['pain_dist']:.0f} pts"),
                    unsafe_allow_html=True)
    with col6:
        st.markdown(_card("IV SKEW READING", "#00d4ff", "📐",
                          analysis.get("iv_skew_reading", "—"),
                          f"Avg CE IV: {ctx['avg_ce_iv']:.1f}%  |  Avg PE IV: {ctx['avg_pe_iv']:.1f}%"),
                    unsafe_allow_html=True)

    # ── BREAKOUT / BREAKDOWN TRIGGERS ───────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown(f"""
        <div style="background:#030d06;border:1px solid #00e676;border-left:4px solid #00e676;
                    border-radius:4px;padding:12px 16px;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:2px;color:#00e676;margin-bottom:6px;">
            🔼 BREAKOUT TRIGGER (BULLISH)</div>
          <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#c8dff0;line-height:1.6;">
            {analysis.get('breakout_trigger','—')}</div>
        </div>""", unsafe_allow_html=True)
    with t_col2:
        st.markdown(f"""
        <div style="background:#0d0305;border:1px solid #ff3d57;border-left:4px solid #ff3d57;
                    border-radius:4px;padding:12px 16px;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:2px;color:#ff3d57;margin-bottom:6px;">
            🔽 BREAKDOWN TRIGGER (BEARISH)</div>
          <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#c8dff0;line-height:1.6;">
            {analysis.get('breakdown_trigger','—')}</div>
        </div>""", unsafe_allow_html=True)

    # ── TRADE SETUPS ─────────────────────────────────────────────────────
    setups = analysis.get("trade_setups", [])
    if setups:
        st.markdown("""
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:14px;
                    font-weight:700;letter-spacing:2px;color:#e8f4ff;
                    border-left:3px solid #ff8c00;padding:4px 12px;
                    background:linear-gradient(90deg,#111920,transparent);
                    margin:18px 0 10px 0;">⚡ ACTIONABLE TRADE SETUPS</div>""",
                    unsafe_allow_html=True)
        setup_cols = st.columns(min(len(setups), 3))
        type_cfg = {
            "BULLISH":  ("#00e676", "#051a0e"),
            "BEARISH":  ("#ff3d57", "#1a0508"),
            "RANGE":    ("#ffd600", "#1a1000"),
            "BREAKOUT": ("#00d4ff", "#001018"),
        }
        for i, setup in enumerate(setups[:3]):
            s_type  = setup.get("type", "RANGE")
            sc, sbg = type_cfg.get(s_type, ("#7fa8c8", "#0d1117"))
            with setup_cols[i]:
                st.markdown(f"""
                <div style="background:{sbg};border:1px solid {sc};border-top:3px solid {sc};
                            border-radius:4px;padding:14px 16px;">
                  <div style="font-family:'Barlow Condensed',sans-serif;font-size:14px;
                              font-weight:700;color:{sc};letter-spacing:1.5px;margin-bottom:4px;">
                    {setup.get('name','Setup')}</div>
                  <div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
                    <span style="background:{sc}22;border:1px solid {sc};color:{sc};
                                 font-family:'Barlow Condensed',sans-serif;font-size:11px;
                                 font-weight:700;padding:2px 10px;border-radius:10px;
                                 letter-spacing:1px;">{s_type}</span>
                    <span style="background:#0d1117;border:1px solid #1e3040;color:#e8f4ff;
                                 font-family:'JetBrains Mono',monospace;font-size:11px;
                                 padding:2px 10px;border-radius:10px;">
                      {setup.get('action','—')}</span>
                    <span style="background:#111920;border:1px solid #2a4560;color:#ffd600;
                                 font-family:'JetBrains Mono',monospace;font-size:11px;
                                 font-weight:700;padding:2px 10px;border-radius:10px;">
                      K: {setup.get('strike','—')}</span>
                  </div>
                  <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                              letter-spacing:1.5px;color:#7fa8c8;margin-top:8px;">ENTRY CONDITION</div>
                  <div style="font-family:'Barlow',sans-serif;font-size:11px;color:#c8dff0;
                              line-height:1.5;margin-bottom:6px;">
                    {setup.get('entry_condition','—')[:100]}{'…' if len(setup.get('entry_condition',''))>100 else ''}</div>
                  <div style="display:flex;gap:12px;font-family:'JetBrains Mono',monospace;font-size:11px;margin-top:8px;">
                    <span>🛑 <b style='color:#ff3d57;'>{setup.get('sl','—')}</b></span>
                    <span>🎯 <b style='color:#00e676;'>{setup.get('target','—')}</b></span>
                  </div>
                  <div style="font-family:'Barlow',sans-serif;font-size:10px;color:#3a6080;
                              margin-top:8px;padding-top:6px;border-top:1px solid #1e3040;
                              line-height:1.4;">
                    {setup.get('reason','')[:120]}{'…' if len(setup.get('reason',''))>120 else ''}</div>
                </div>""", unsafe_allow_html=True)

    # ── FINAL VERDICT ─────────────────────────────────────────────────────
    verdict = analysis.get("final_verdict", "—")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1520,#111920);
                border:1px solid #2a4560;border-left:5px solid #ff8c00;
                border-radius:6px;padding:18px 22px;margin-top:16px;
                box-shadow:0 4px 24px rgba(255,140,0,0.08);">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                  letter-spacing:2.5px;color:#ff8c00;margin-bottom:8px;">
        🏁 FINAL VERDICT — PRO AI ANALYSIS</div>
      <div style="font-family:'Barlow',sans-serif;font-size:14px;
                  color:#e8f4ff;line-height:1.8;font-weight:500;">
        {verdict}</div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 4 — S&R + MAX PAIN  (main render)
# ──────────────────────────────────────────────────────────────
def render():
    section_header("Support & Resistance + Max Pain", "Key OI levels, volume walls, max pain strike")

    oc_t4  = st.session_state.get("current_option_chain", pd.DataFrame())
    spot_t4= st.session_state.get("current_spot_price")
    sel_t4 = st.session_state.get("current_selected_index", "")
    exp_t4 = st.session_state.get("oc_expiry_select", "")

    if oc_t4 is None or (isinstance(oc_t4, pd.DataFrame) and oc_t4.empty) or spot_t4 is None:
        st.info("🔄 Load option chain from Tab 1 first."); return

    ce_top3 = oc_t4.sort_values("CE_OI",     ascending=False).head(3)
    pe_top3 = oc_t4.sort_values("PE_OI",     ascending=False).head(3)
    cv_top3 = oc_t4.sort_values("CE_Volume", ascending=False).head(3)
    pv_top3 = oc_t4.sort_values("PE_Volume", ascending=False).head(3)

    res_c4 = [s for s in ce_top3["Strike"].tolist() + cv_top3["Strike"].tolist() if s > spot_t4]
    sup_c4 = [s for s in pe_top3["Strike"].tolist() + pv_top3["Strike"].tolist() if s < spot_t4]
    main_res4 = min(res_c4) if res_c4 else None
    main_sup4 = max(sup_c4) if sup_c4 else None

    # Max Pain
    max_pain_strike4, pain_dict4 = calculate_max_pain(oc_t4)
    pain_dist4 = abs(spot_t4 - max_pain_strike4)
    pain_dir4  = "above" if spot_t4 > max_pain_strike4 else "below"

    # Key levels display
    metrics_row(
        metric_card("SPOT", f"₹{spot_t4:,.0f}", "", "#ff8c00") +
        metric_card("MAX PAIN", f"{max_pain_strike4:,.0f}", f"Spot {pain_dist4:.0f} pts {pain_dir4} pain", "#ffd600") +
        (metric_card("RESISTANCE", f"{main_res4:,.0f}", f"+{main_res4-spot_t4:.0f} pts", "#ff3d57") if main_res4 else "") +
        (metric_card("SUPPORT", f"{main_sup4:,.0f}", f"-{spot_t4-main_sup4:.0f} pts", "#00e676") if main_sup4 else "") +
        (metric_card("RANGE", f"{main_res4-main_sup4:.0f} pts", "Resistance - Support", "#c084fc") if main_res4 and main_sup4 else "")
    )

    col_sr1, col_sr2, col_sr3, col_sr4 = st.columns(4)
    with col_sr1:
        section_header("🔴 CE OI Resistance")
        for i, (_, row) in enumerate(ce_top3.iterrows(), 1):
            marker = " ◀ ATM" if abs(row['Strike'] - spot_t4) < 100 else ""
            clr = "#ff3d57" if row['Strike'] > spot_t4 else "#7fa8c8"
            st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{clr};padding:4px 0;border-bottom:1px solid #1e3040;"> R{i}: <b>{int(row['Strike'])}</b> | {int(row['CE_OI'])/1e5:.1f}L{marker}</div>""", unsafe_allow_html=True)
    with col_sr2:
        section_header("🟢 PE OI Support")
        for i, (_, row) in enumerate(pe_top3.iterrows(), 1):
            clr = "#00e676" if row['Strike'] < spot_t4 else "#7fa8c8"
            st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{clr};padding:4px 0;border-bottom:1px solid #1e3040;"> S{i}: <b>{int(row['Strike'])}</b> | {int(row['PE_OI'])/1e5:.1f}L</div>""", unsafe_allow_html=True)
    with col_sr3:
        section_header("⚡ CE Vol Resistance")
        for i, (_, row) in enumerate(cv_top3.iterrows(), 1):
            st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#ff8c00;padding:4px 0;border-bottom:1px solid #1e3040;"> VR{i}: <b>{int(row['Strike'])}</b> | {int(row['CE_Volume'])/1e3:.0f}K</div>""", unsafe_allow_html=True)
    with col_sr4:
        section_header("🔥 PE Vol Support")
        for i, (_, row) in enumerate(pv_top3.iterrows(), 1):
            st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#00d4ff;padding:4px 0;border-bottom:1px solid #1e3040;"> VS{i}: <b>{int(row['Strike'])}</b> | {int(row['PE_Volume'])/1e3:.0f}K</div>""", unsafe_allow_html=True)

    # Max Pain Chart
    st.markdown("---")
    section_header("Max Pain Analysis", "Total writer pain at each strike")
    try:
        pain_strikes = list(pain_dict4.keys())
        pain_values  = [pain_dict4[s]/1e7 for s in pain_strikes]
        colors_pain  = ["#ffd600" if s == max_pain_strike4 else
                         "#ff8c00" if abs(s - max_pain_strike4) <= 100 else "#1e3040"
                         for s in pain_strikes]
        pain_fig = go.Figure()
        pain_fig.add_bar(x=pain_strikes, y=pain_values, marker_color=colors_pain, name="Writer Pain")
        pain_fig.add_vline(x=float(max_pain_strike4), line_color="#ffd600",
                           line_dash="dash", line_width=2,
                           annotation_text=f"MAX PAIN {int(max_pain_strike4)}",
                           annotation_font_color="#ffd600", annotation_font_size=11)
        pain_fig.add_vline(x=float(get_atm_strike(spot_t4, sel_t4)), line_color="#00d4ff",
                           line_dash="dot", line_width=1.5,
                           annotation_text="SPOT", annotation_font_color="#00d4ff", annotation_font_size=11)
        pain_fig.update_layout(
            height=320, paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
            font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d"),
            yaxis=dict(gridcolor="#1e3040", title="Writer Pain (Cr)"),
            showlegend=False
        )
        st.plotly_chart(pain_fig, use_container_width=True)
    except ImportError:
        st.info("pip install plotly for Max Pain chart")

    # OI Heatmap
    st.markdown("---")
    section_header("OI Buildup Heatmap")
    try:
        oc_hm4 = oc_t4.sort_values("Strike")
        hm_fig4 = go.Figure()
        hm_fig4.add_bar(x=oc_hm4["Strike"], y=oc_hm4["CE_OI"]/1e5,
                        name="CE OI", marker_color="#ff3d57", opacity=0.85)
        hm_fig4.add_bar(x=oc_hm4["Strike"], y=oc_hm4["PE_OI"]/1e5,
                        name="PE OI", marker_color="#00e676", opacity=0.85)
        hm_fig4.add_vline(x=float(max_pain_strike4), line_color="#ffd600",
                          line_dash="dash", line_width=1.5,
                          annotation_text="PAIN", annotation_font_color="#ffd600", annotation_font_size=10)
        hm_fig4.update_layout(
            barmode="group", height=320,
            paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
            font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d"),
            yaxis=dict(gridcolor="#1e3040", title="OI (Lakhs)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7fa8c8")
        )
        st.plotly_chart(hm_fig4, use_container_width=True)
    except ImportError:
        pass

    # ══════════════════════════════════════════════════════════════════
    #  🤖  PRO AI SIGNAL ANALYSIS SECTION
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")

    # ── Header with controls ──────────────────────────────────────────
    hdr_col, btn_col, auto_col = st.columns([3, 1.2, 1.5])
    with hdr_col:
        st.markdown("""
        <div style="border-left:3px solid #c084fc;padding:4px 12px;
                    background:linear-gradient(90deg,#111920,transparent);margin:8px 0 10px 0;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:20px;
                      font-weight:800;letter-spacing:2px;color:#e8f4ff;">
            🤖 AI <span style="color:#c084fc;">PRO SIGNAL ANALYSIS</span></div>
          <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#7fa8c8;margin-top:2px;">
            Live OI • Smart Money • Resistance/Support • Trade Setups • Final Verdict</div>
        </div>""", unsafe_allow_html=True)

    with btn_col:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        run_now = st.button("🧠 ANALYSE NOW", use_container_width=True,
                            type="primary", key="t4_ai_run")
    with auto_col:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        auto_refresh = st.toggle("⚡ Auto (5 min)", value=False, key="t4_ai_auto")

    # ── Auto-refresh logic ────────────────────────────────────────────
    should_run = run_now
    if auto_refresh:
        last_t4 = st.session_state.get("t4_ai_last_run")
        if last_t4 is None or (now_ist_dt - last_t4).total_seconds() >= 300:
            should_run = True

    # ── Show last run time ────────────────────────────────────────────
    last_t4 = st.session_state.get("t4_ai_last_run")
    if last_t4:
        elapsed = int((now_ist_dt - last_t4).total_seconds())
        remaining = max(0, 300 - elapsed) if auto_refresh else None
        elapsed_str = f"Last run: {last_t4.strftime('%H:%M:%S')}  |  {elapsed}s ago"
        if remaining is not None:
            elapsed_str += f"  |  Next in: {remaining}s"
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#3a6080;margin-bottom:8px;">
          {elapsed_str}</div>""", unsafe_allow_html=True)

    # ── Run analysis ──────────────────────────────────────────────────
    if should_run:
        with st.spinner("🤖 Claude is reading your option chain…  usually takes 5–8 seconds"):
            try:
                ctx = _build_market_context(
                    oc_t4, spot_t4, sel_t4, exp_t4,
                    max_pain_strike4, pain_dict4
                )
                analysis = _call_ai_analysis(ctx)
                st.session_state["t4_ai_analysis"] = analysis
                st.session_state["t4_ai_ctx"]      = ctx
                st.session_state["t4_ai_last_run"] = now_ist_dt
                st.rerun()
            except Exception as e:
                st.error(f"❌ AI analysis error: {e}")

    # ── Display cached analysis ───────────────────────────────────────
    analysis = st.session_state.get("t4_ai_analysis")
    ctx_saved = st.session_state.get("t4_ai_ctx")

    if analysis and ctx_saved:
        _render_ai_panel(analysis, ctx_saved)
    else:
        # Empty state
        st.markdown("""
        <div style="background:#0d1117;border:1px dashed #1e3040;border-radius:6px;
                    padding:40px;text-align:center;margin:10px 0;">
          <div style="font-size:36px;margin-bottom:12px;">🤖</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:18px;
                      letter-spacing:2px;color:#7fa8c8;margin-bottom:8px;">
            AI ANALYSIS NOT YET RUN</div>
          <div style="font-family:'Barlow',sans-serif;font-size:13px;color:#3a6080;">
            Click <b style="color:#c084fc;">ANALYSE NOW</b> to get a pro-level reading of the current option chain.<br>
            Includes: OI analysis · Smart money signals · Key levels · Trade setups · Final verdict
          </div>
        </div>""", unsafe_allow_html=True)
