import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
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
                       compute_signal_score, generate_ai_trade, check_alerts)
from chart_utils import (compute_technicals, compute_order_flow, detect_liquidity_sweeps,
                         detect_order_blocks, detect_fvg, detect_bos_choch, get_order_flow_summary)

# ======================================================
# TAB 5 — AI ADVISOR
# ======================================================
def render():
    section_header("AI Trade Advisor", "Multi-factor signal engine with auto-execution and strategy selection")

    oc_t5  = st.session_state.get("option_chain_df")
    spot_t5= st.session_state.get("current_spot_price")
    sel_t5 = st.session_state.get("current_selected_index")

    if oc_t5 is None or spot_t5 is None:
        st.warning("⏳ Waiting for option chain data from Tab 1..."); return

    df5 = oc_t5.copy()
    df5["distance"] = abs(df5["Strike"] - spot_t5)
    atm_row5  = df5.loc[df5["distance"].idxmin()]
    atm_s5    = float(atm_row5["Strike"])

    bull5, bear5, pcr5, pcr_chg5, m_res5, m_sup5, factors5 = compute_signal_score(df5, spot_t5, sel_t5)

    spot_hist5 = st.session_state.spot_history
    velocity5  = 0
    if len(spot_hist5) >= 2:
        velocity5 = spot_hist5[-1]["spot"] - spot_hist5[-2]["spot"]

    # Unified market flow — safe OI change extraction
    if "CE_OI_Change" in atm_row5.index and "PE_OI_Change" in atm_row5.index:
        ce_chg5 = float(atm_row5["CE_OI_Change"])
        pe_chg5 = float(atm_row5["PE_OI_Change"])
    elif "CE_Prev_OI" in atm_row5.index and "PE_Prev_OI" in atm_row5.index:
        ce_chg5 = float(atm_row5["CE_OI"]) - float(atm_row5["CE_Prev_OI"])
        pe_chg5 = float(atm_row5["PE_OI"]) - float(atm_row5["PE_Prev_OI"])
    else:
        ce_chg5 = 0.0
        pe_chg5 = 0.0
    oi_score5 = pe_chg5 - ce_chg5

    # ── Multi-factor regime engine — calibrated for /20 score scale ──
    net_score5    = bull5 - bear5           # -20 to +20
    score_margin5 = abs(net_score5)
    pcr_bullish5  = pcr5 > 1.05
    pcr_bearish5  = pcr5 < 0.95
    oi_bullish5   = oi_score5 > 50000
    oi_bearish5   = oi_score5 < -50000

    # Velocity over LONGER window (last 10 spot readings = ~50 seconds)
    spot_hist5_long = spot_hist5[-10:] if len(spot_hist5) >= 10 else spot_hist5
    velocity5_long  = 0
    if len(spot_hist5_long) >= 2:
        velocity5_long = spot_hist5_long[-1]["spot"] - spot_hist5_long[0]["spot"]

    if score_margin5 >= 8 and net_score5 > 0 and (pcr_bullish5 or oi_bullish5):
        market_flow5 = "Bullish"          # Strong consensus + OI confirmation
    elif score_margin5 >= 8 and net_score5 < 0 and (pcr_bearish5 or oi_bearish5):
        market_flow5 = "Bearish"          # Strong consensus + OI confirmation
    elif score_margin5 >= 5 and net_score5 > 0 and velocity5_long > 5:
        market_flow5 = "Bullish"          # Moderate edge + price moving up
    elif score_margin5 >= 5 and net_score5 < 0 and velocity5_long < -5:
        market_flow5 = "Bearish"          # Moderate edge + price moving down
    elif score_margin5 <= 3 and abs(velocity5_long) < 10 and abs(oi_score5) < 100000:
        market_flow5 = "Choppy"           # No conviction anywhere
    elif score_margin5 <= 4 and abs(velocity5_long) < 15:
        market_flow5 = "Range"            # Slight lean but price going sideways
    elif net_score5 > 0:
        market_flow5 = "Bullish"          # Mild bullish edge
    elif net_score5 < 0:
        market_flow5 = "Bearish"          # Mild bearish edge
    else:
        market_flow5 = "Range"            # Truly neutral

    flow_cfgs = {
        "Bullish": ("#00e676","#051a0e","📈 BULLISH","BULL PUT SPREAD"),
        "Bearish": ("#ff3d57","#1a0508","📉 BEARISH","BEAR CALL SPREAD"),
        "Range":   ("#ffd600","#1a1000","↔ RANGE",   "IRON CONDOR"),
        "Choppy":  ("#7fa8c8","#001020","〰 CHOPPY",  "IRON FLY"),
    }
    fg5, bg5, flabel5, fstrat5 = flow_cfgs.get(market_flow5, ("#7fa8c8","#0d1117",market_flow5,"—"))
    # Debug strip for regime inputs
    _vm_c5 = "#00e676" if velocity5_long > 0 else "#ff3d57" if velocity5_long < 0 else "#7fa8c8"
    _oi_c5 = "#00e676" if oi_score5 > 0 else "#ff3d57"
    st.markdown(f"""<div style="background:#070b0f;border:1px solid #1a2a3a;border-radius:2px;
        padding:6px 14px;margin-bottom:8px;display:flex;gap:20px;flex-wrap:wrap;
        font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;">
      <span>Net Score: <b style="color:{'#00e676' if net_score5>0 else '#ff3d57'};">{net_score5:+d}</b></span>
      <span>Margin: <b style="color:#e8f4ff;">{score_margin5}</b></span>
      <span>Velocity(10s): <b style="color:{_vm_c5};">{velocity5_long:+.1f} pts</b></span>
      <span>OI Score: <b style="color:{_oi_c5};">{oi_score5:+,.0f}</b></span>
      <span>PCR: <b style="color:#c084fc;">{pcr5:.3f}</b></span>
      <span>Regime Input: <b style="color:{fg5};">{market_flow5}</b></span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f""" <div style="background:{bg5};border:1px solid {fg5};border-left:4px solid {fg5};border-radius:3px;padding:14px 20px;margin-bottom:16px;"><div style="display:flex;align-items:center;justify-content:space-between;"><div><div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:2px;color:{fg5};margin-bottom:4px;">MARKET REGIME</div><div style="font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:800;letter-spacing:3px;color:{fg5};">{flabel5}</div><div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#7fa8c8;margin-top:4px;"> → {fstrat5}</div></div><div style="text-align:right;font-family:'JetBrains Mono',monospace;"><div style="font-size:11px;color:#3a6080;">BULL</div><div style="font-size:22px;font-weight:700;color:#00e676;">{bull5}/20</div><div style="font-size:11px;color:#3a6080;margin-top:4px;">BEAR</div><div style="font-size:22px;font-weight:700;color:#ff3d57;">{bear5}/20</div></div></div><div style="display:flex;gap:16px;margin-top:10px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#7fa8c8;"><span>ATM: <b style='color:#e8f4ff;'>{atm_s5:.0f}</b></span><span>Velocity: <b style='color:{"#00e676" if velocity5 > 0 else "#ff3d57"};'>{velocity5:+.1f} pts</b></span><span>OI Score: <b style='color:{"#00e676" if oi_score5 > 0 else "#ff3d57"};'>{oi_score5:+,.0f}</b></span><span>PCR: <b style='color:#c084fc;'>{pcr5:.3f}</b></span></div></div>""", unsafe_allow_html=True)

    # IV status
    ce_iv5  = float(atm_row5.get("CE_IV", 0))
    pe_iv5  = float(atm_row5.get("PE_IV", 0))
    iv_avg5 = df5["CE_IV"].rolling(5, min_periods=1).mean().iloc[-1]
    iv_spike5 = ce_iv5 > iv_avg5 * 1.25 or pe_iv5 > iv_avg5 * 1.25
    if iv_spike5:
        st.markdown(f"""<div style="background:#1a0a00;border:1px solid #ff8c00;border-left:3px solid #ff8c00;border-radius:2px;padding:8px 14px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#ff8c00;"> ⚡ IV SPIKE DETECTED — CE IV: {ce_iv5:.1f}% | PE IV: {pe_iv5:.1f}% | Avg: {iv_avg5:.1f}% — AI will skip new trades</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 📊 Chart moved to dedicated "📊 CHART" tab
    # ═══════════════════════════════════════════════════════════
    st.info("📊 Live chart has moved to the **📊 CHART** tab for a better view. "
            "All signal data below is still live.")

    st.markdown("---")
    section_header("⚙️ Rule-Based Signal Engine", "15-factor OI/flow signals + auto-trade log")

    # Settings
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1: risk_pts5 = st.number_input("SL (pts)", value=15, min_value=1, key="t5_sl")
    with cfg2: rr5       = st.selectbox("R:R", ["1:2","1:3","1:1"], key="t5_rr")
    with cfg3: auto_en5  = st.toggle("🤖 Auto Every 5 Min", value=False, key="t5_auto")
    with cfg4:
        strategy_override = st.selectbox("Strategy Override",
            ["Auto (AI)", "Iron Fly", "Iron Condor", "Bull Put Spread", "Bear Call Spread", "Directional CE Buy", "Directional PE Buy"],
            key="t5_strat")
    reward5 = risk_pts5 * int(rr5.split(":")[1])

    # Countdown timer
    last_run5   = st.session_state.get("last_ai_trade_time")
    if last_run5:
        elapsed5  = (now_ist_dt - last_run5).total_seconds()
        remaining5= max(0, 300 - int(elapsed5))
        m5, s5    = divmod(remaining5, 60)
        tc5       = "#00e676" if remaining5 > 120 else "#ffd600" if remaining5 > 30 else "#ff3d57"
        prog5     = (300 - remaining5) / 300 * 100
        st.markdown(f""" <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;padding:10px 14px;margin-bottom:8px;"><div style="display:flex;justify-content:space-between;align-items:center;"><span style="font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:1.5px;color:#7fa8c8;"> NEXT AUTO TRADE</span><span style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:{tc5};"> {m5:02d}:{s5:02d}</span></div><div style="background:#1e3040;border-radius:2px;height:3px;margin-top:6px;"><div style="background:{tc5};width:{prog5:.0f}%;height:100%;border-radius:2px;transition:width 0.3s;"></div></div><div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;margin-top:4px;"> Last: {last_run5.strftime('%H:%M:%S')} IST</div></div>""", unsafe_allow_html=True)
    else:
        st.info("⏳ No auto trade yet. Enable toggle or click Generate Now.")

    should_fire5 = False
    if auto_en5 and MARKET_OPEN:
        if last_run5 is None or (now_ist_dt - last_run5).total_seconds() >= 300:
            should_fire5 = True
    elif auto_en5 and not MARKET_OPEN:
        st.warning("🔴 Market closed — auto trade paused.")

    if should_fire5:
        new_trades5 = generate_ai_trade(df5, atm_row5, atm_s5, market_flow5, sel_t5,
                                         risk_pts5, reward5, bull5, bear5)
        for t in new_trades5:
            st.session_state.ai_trade_log.append(t)
        st.session_state.last_ai_trade_time = now_ist_dt
        if new_trades5:
            save_list_to_csv(st.session_state.ai_trade_log, AI_LOG_FILE)
            st.toast(f"🤖 {new_trades5[0]['Type']} @ {new_trades5[0]['Strike']} · Saved", icon="✅")

    # Buttons
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("🚀 GENERATE NOW", use_container_width=True, type="primary"):
            new_t5 = generate_ai_trade(df5, atm_row5, atm_s5, market_flow5, sel_t5,
                                        risk_pts5, reward5, bull5, bear5)
            for t in new_t5:
                st.session_state.ai_trade_log.append(t)
            st.session_state.last_ai_trade_time = now_ist_dt
            if new_t5:
                save_list_to_csv(st.session_state.ai_trade_log, AI_LOG_FILE)
                st.success(f"✅ Generated {len(new_t5)} trade(s) · Auto-saved")
    with b2:
        if st.button("🗑️ CLEAR LOG", use_container_width=True):
            st.session_state.ai_trade_log = []
            st.session_state.last_ai_trade_time = None
    with b3:
        if st.button("💾 SAVE LOG", use_container_width=True):
            save_list_to_csv(st.session_state.ai_trade_log, AI_LOG_FILE)
            st.success("Saved.")
    with b4:
        if st.button("📥 EXPORT CSV", use_container_width=True):
            if st.session_state.ai_trade_log:
                csv5 = pd.DataFrame(st.session_state.ai_trade_log).to_csv(index=False)
                st.download_button("⬇️ Download", csv5, "ai_trades.csv", "text/csv")

    # Recent trades
    st.markdown("---")
    section_header("Recent AI Trades (Last 8)")
    recent5 = st.session_state.ai_trade_log[-8:][::-1]
    if not recent5:
        st.markdown("""<div style="background:#0d1117;border:1px solid #1e3040;padding:16px;
            text-align:center;color:#3a6080;font-family:'Barlow Condensed',sans-serif;letter-spacing:1px;">
            NO AI TRADES YET</div>""", unsafe_allow_html=True)
    else:
        for t in recent5:
            t_type = str(t.get("Type", "")) if t.get("Type") == t.get("Type") else ""  # NaN-safe
            t_status = str(t.get("Status", "")) if t.get("Status") == t.get("Status") else ""
            tc5b = "#00e676" if "PE" in t_type else "#ff3d57"
            status_cfg = {"Active": ("#00d4ff","🟢"), "SL Hit": ("#ff3d57","🔴"), "Target Hit": ("#ffd600","🏆")}
            sc5, si5 = status_cfg.get(t_status, ("#7fa8c8","⚪"))
            pnl5 = t.get("Live_PnL", 0) or 0
            pnl_c5 = "#00e676" if pnl5 > 0 else "#ff3d57" if pnl5 < 0 else "#7fa8c8"
            try:
                entry_val = f"₹{float(t.get('Entry', 0)):.1f}"
                sl_val    = f"₹{float(t.get('SL', 0)):.1f}"
                tgt_val   = f"₹{float(t.get('Target', 0)):.1f}"
            except (TypeError, ValueError):
                entry_val, sl_val, tgt_val = "₹—", "₹—", "₹—"
            st.markdown(f""" <div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid {tc5b};border-radius:2px;padding:10px 14px;margin:4px 0;display:grid;grid-template-columns:80px 100px 80px 80px 80px 80px 80px auto;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:11px;"><span style="color:#3a6080;">{t.get('Entry Time','')}</span><span style="color:{tc5b};font-weight:700;font-size:12px;">{t_type}</span><span style="color:#e8f4ff;">K: {t.get('Strike','—')}</span><span style="color:#7fa8c8;">E: {entry_val}</span><span style="color:#ff3d57;">SL: {sl_val}</span><span style="color:#00e676;">T: {tgt_val}</span><span style="color:{pnl_c5};font-weight:700;">₹{pnl5:,.0f}</span><span style="color:{sc5};">{si5} {t_status}</span></div>""", unsafe_allow_html=True)


    # ======================================================
