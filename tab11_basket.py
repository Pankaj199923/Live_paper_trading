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
                       compute_signal_score, generate_ai_trade, check_alerts, call_claude_trade_setup)
from chart_utils import (compute_technicals, compute_order_flow, detect_liquidity_sweeps,
                         detect_order_blocks, detect_fvg, detect_bos_choch, get_order_flow_summary)

# ======================================================
# TAB 11 — BASKET TRADE BUILDER
# ======================================================
def render():
    st.session_state["active_tab_key"] = "🧺 BASKET"

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;
                border-bottom:1px solid #1e3040;padding-bottom:8px;margin-bottom:14px;">
      <div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:800;
                    letter-spacing:3px;color:#e8f4ff;">🧺 BASKET <span style="color:#ff8c00;">TRADE BUILDER</span></div>
        <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#7fa8c8;margin-top:2px;">
          Build multi-leg strategies · Calculate combined payoff · P&L simulation · One-click execute</div>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3a6080;">STRATEGY DESIGNER</div>
    </div>""", unsafe_allow_html=True)

    oc_b  = st.session_state.get("current_option_chain", pd.DataFrame())
    spot_b= st.session_state.get("current_spot_price")
    sel_b = st.session_state.get("current_selected_index")
    exp_b = st.session_state.get("oc_expiry_select", "")

    if oc_b is None or (isinstance(oc_b, pd.DataFrame) and oc_b.empty) or not spot_b:
        st.warning("⏳ Load option chain from Tab 1 first — basket builder needs live data.")
        return

    atm_b     = get_atm_strike(spot_b, sel_b)
    lot_size_b= get_lot_size(sel_b)
    step_b_s  = oc_b["Strike"].diff().dropna()
    step_b    = int(step_b_s.mode()[0]) if not step_b_s.empty else 50

    # ── Session init ─────────────────────────────────────────────────────
    if "basket_legs" not in st.session_state:
        st.session_state.basket_legs = []

    # ── Strategy Templates ───────────────────────────────────────────────
    st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
        letter-spacing:1.5px;color:#7fa8c8;margin-bottom:8px;">⚡ QUICK TEMPLATES</div>""",
        unsafe_allow_html=True)

    tmpl_cols = st.columns(6)
    templates = {
        "Short Straddle": [
            {"action":"SELL","type":"CE","strike":atm_b,"lots":1},
            {"action":"SELL","type":"PE","strike":atm_b,"lots":1},
        ],
        "Iron Fly": [
            {"action":"SELL","type":"CE","strike":atm_b,          "lots":1},
            {"action":"SELL","type":"PE","strike":atm_b,          "lots":1},
            {"action":"BUY", "type":"CE","strike":atm_b+2*step_b, "lots":1},
            {"action":"BUY", "type":"PE","strike":atm_b-2*step_b, "lots":1},
        ],
        "Iron Condor": [
            {"action":"SELL","type":"CE","strike":atm_b+step_b,   "lots":1},
            {"action":"SELL","type":"PE","strike":atm_b-step_b,   "lots":1},
            {"action":"BUY", "type":"CE","strike":atm_b+2*step_b, "lots":1},
            {"action":"BUY", "type":"PE","strike":atm_b-2*step_b, "lots":1},
        ],
        "Bull Put Spread": [
            {"action":"SELL","type":"PE","strike":atm_b,         "lots":1},
            {"action":"BUY", "type":"PE","strike":atm_b-step_b,  "lots":1},
        ],
        "Bear Call Spread": [
            {"action":"SELL","type":"CE","strike":atm_b,         "lots":1},
            {"action":"BUY", "type":"CE","strike":atm_b+step_b,  "lots":1},
        ],
        "Long Strangle": [
            {"action":"BUY","type":"CE","strike":atm_b+step_b,  "lots":1},
            {"action":"BUY","type":"PE","strike":atm_b-step_b,  "lots":1},
        ],
    }

    for ci, (tname, tlegs) in enumerate(templates.items()):
        with tmpl_cols[ci % 6]:
            if st.button(tname, key=f"tmpl_{tname}", use_container_width=True):
                st.session_state.basket_legs = []
                for leg in tlegs:
                    k = leg["strike"]
                    row_k = oc_b[oc_b["Strike"] == k]
                    col_k = f"{leg['type']}_LTP"
                    prem  = round(float(row_k[col_k].values[0]), 2) if not row_k.empty and col_k in row_k.columns else 0.0
                    st.session_state.basket_legs.append({
                        **leg,
                        "premium": prem,
                        "lots":    leg["lots"],
                    })
                st.toast(f"✅ {tname} loaded — {len(tlegs)} legs", icon="🧺")

    st.markdown("---")

    # ── Manual Leg Builder ───────────────────────────────────────────────
    section_header("Add Leg Manually")
    la, lb, lc, ld = st.columns([1, 1, 2, 1])
    with la: leg_action = st.selectbox("Action", ["SELL","BUY"], key="bl_action")
    with lb: leg_type   = st.selectbox("Type",   ["CE","PE"],    key="bl_type")
    with lc:
        strike_opts = sorted(oc_b["Strike"].unique().tolist())
        leg_strike  = st.selectbox("Strike", strike_opts,
                                    index=strike_opts.index(atm_b) if atm_b in strike_opts else 0,
                                    key="bl_strike")
    with ld: leg_lots = st.number_input("Lots", value=1, min_value=1, max_value=50, key="bl_lots")

    # ── Auto-capture live premium for selected strike + type ──────────────
    col_ltp   = f"{leg_type}_LTP"
    row_s     = oc_b[oc_b["Strike"] == leg_strike]
    auto_prem = round(float(row_s[col_ltp].values[0]), 2) if not row_s.empty and col_ltp in row_s.columns else 0.0

    # Dynamic key — resets widget value whenever strike OR type changes
    prem_widget_key = f"bl_prem_{int(leg_strike)}_{leg_type}"

    # Show live LTP preview bar
    prev_prem_key = f"bl_prev_prem_{leg_type}"
    prev_prem     = st.session_state.get(prev_prem_key, auto_prem)
    prem_delta    = auto_prem - prev_prem
    prem_delta_c  = "#00e676" if prem_delta >= 0 else "#ff3d57"
    st.session_state[prev_prem_key] = auto_prem

    ce_iv = round(float(row_s["CE_IV"].values[0]), 2) if not row_s.empty and "CE_IV" in row_s.columns else 0
    pe_iv = round(float(row_s["PE_IV"].values[0]), 2) if not row_s.empty and "PE_IV" in row_s.columns else 0
    ce_oi = int(row_s["CE_OI"].values[0]) if not row_s.empty and "CE_OI" in row_s.columns else 0
    pe_oi = int(row_s["PE_OI"].values[0]) if not row_s.empty and "PE_OI" in row_s.columns else 0
    ce_ltp= round(float(row_s["CE_LTP"].values[0]), 2) if not row_s.empty and "CE_LTP" in row_s.columns else 0
    pe_ltp= round(float(row_s["PE_LTP"].values[0]), 2) if not row_s.empty and "PE_LTP" in row_s.columns else 0
    atm_tag = "🟡 ATM" if leg_strike == atm_b else ("🔵 ITM" if (leg_type=="CE" and leg_strike < atm_b) or (leg_type=="PE" and leg_strike > atm_b) else "⚪ OTM")

    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ff8c00;
                border-radius:3px;padding:10px 16px;margin:8px 0 6px 0;
                display:flex;flex-wrap:wrap;gap:20px;align-items:center;">
      <div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:1.5px;color:#7fa8c8;">STRIKE</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:20px;
                    font-weight:700;color:#ff8c00;">{int(leg_strike):,}</div>
        <div style="font-size:10px;color:#3a6080;">{atm_tag}</div>
      </div>
      <div style="border-left:1px solid #1e3040;padding-left:20px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:1px;color:#7fa8c8;">CE LTP</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                    font-weight:700;color:#ff3d57;">₹{ce_ltp}</div>
        <div style="font-size:10px;color:#3a6080;">IV: {ce_iv}% | OI: {ce_oi/1e5:.1f}L</div>
      </div>
      <div style="border-left:1px solid #1e3040;padding-left:20px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:1px;color:#7fa8c8;">PE LTP</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                    font-weight:700;color:#00e676;">₹{pe_ltp}</div>
        <div style="font-size:10px;color:#3a6080;">IV: {pe_iv}% | OI: {pe_oi/1e5:.1f}L</div>
      </div>
      <div style="border-left:1px solid #1e3040;padding-left:20px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:1px;color:#7fa8c8;">SELECTED ({leg_type}) AUTO LTP</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                    font-weight:700;color:{'#ff3d57' if leg_type=='CE' else '#00e676'};">
          ₹{auto_prem}</div>
        <div style="font-size:10px;color:{prem_delta_c};">
          {'+' if prem_delta >= 0 else ''}{prem_delta:.2f} change</div>
      </div>
      <div style="margin-left:auto;background:#0a1520;border:1px solid #1e3040;
                  border-radius:3px;padding:6px 12px;text-align:center;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                    letter-spacing:1px;color:#3a6080;">STRADDLE VALUE</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                    font-weight:700;color:#ffd600;">₹{round(ce_ltp+pe_ltp,2)}</div>
        <div style="font-size:10px;color:#3a6080;">CE+PE combined</div>
      </div>
    </div>""", unsafe_allow_html=True)

    pe2, pf2 = st.columns([2, 1])
    with pe2:
        # Use dynamic key so value refreshes when strike/type changes
        leg_prem = st.number_input(
            f"Premium ₹  ({leg_type} @ {int(leg_strike):,})  — auto-filled from LTP",
            value=auto_prem, min_value=0.0, step=0.5,
            key=prem_widget_key
        )
    with pf2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ ADD LEG", use_container_width=True, type="primary"):
            st.session_state.basket_legs.append({
                "action":  leg_action,
                "type":    leg_type,
                "strike":  leg_strike,
                "lots":    leg_lots,
                "premium": leg_prem,
            })
            st.toast(f"✅ {leg_action} {leg_type} {leg_strike} @ ₹{leg_prem}", icon="✅")

    # ── Current Legs Table ───────────────────────────────────────────────
    st.markdown("---")
    section_header("Current Strategy Legs", f"ATM: {atm_b:,.0f} | Lot Size: {lot_size_b} | Spot: ₹{spot_b:,.2f}")

    if not st.session_state.basket_legs:
        st.markdown("""<div style="background:#0d1117;border:1px dashed #1e3040;border-radius:3px;
            padding:20px;text-align:center;color:#3a6080;font-family:'Barlow Condensed',sans-serif;
            font-size:14px;letter-spacing:1px;">
            NO LEGS YET — USE A TEMPLATE OR ADD MANUALLY ABOVE</div>""", unsafe_allow_html=True)
    else:
        # Legs display
        leg_header = """
        <div style="display:grid;grid-template-columns:80px 60px 100px 60px 100px 120px 120px 80px;
                    padding:6px 12px;border-bottom:1px solid #1e3040;
                    font-family:'Barlow Condensed',sans-serif;font-size:10px;
                    letter-spacing:1.5px;color:#3a6080;">
          <span>ACTION</span><span>TYPE</span><span>STRIKE</span><span>LOTS</span>
          <span>ENTRY ₹</span><span>LIVE LTP ₹</span><span>LEG P&L ₹</span><span>REMOVE</span>
        </div>"""
        st.markdown(leg_header, unsafe_allow_html=True)

        total_credit  = 0.0
        total_pnl_b   = 0.0
        legs_to_remove = []

        for li, leg in enumerate(st.session_state.basket_legs):
            # Fetch live LTP
            k       = leg["strike"]
            col_ltp = f"{leg['type']}_LTP"
            row_k   = oc_b[oc_b["Strike"] == k]
            live_ltp = round(float(row_k[col_ltp].values[0]), 2) if not row_k.empty and col_ltp in row_k.columns else leg["premium"]
            # P&L per lot
            entry_p = float(leg["premium"])
            lots_p  = int(leg["lots"])
            if leg["action"] == "SELL":
                leg_pnl = (entry_p - live_ltp) * lots_p * lot_size_b
                total_credit += entry_p * lots_p * lot_size_b
            else:
                leg_pnl = (live_ltp - entry_p) * lots_p * lot_size_b
                total_credit -= entry_p * lots_p * lot_size_b
            total_pnl_b += leg_pnl

            ac   = "#00e676" if leg["action"] == "SELL" else "#ff8c00"
            tc   = "#ff3d57" if leg["type"] == "CE" else "#00e676"
            pnlc = "#00e676" if leg_pnl >= 0 else "#ff3d57"

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:80px 60px 100px 60px 100px 120px 120px 80px;
                        padding:8px 12px;border-bottom:1px solid #0d1117;
                        font-family:'JetBrains Mono',monospace;font-size:12px;
                        background:{'rgba(0,230,118,0.03)' if leg['action']=='SELL' else 'rgba(255,140,0,0.03)'};">
              <span style="color:{ac};font-weight:700;">{leg['action']}</span>
              <span style="color:{tc};font-weight:700;">{leg['type']}</span>
              <span style="color:#e8f4ff;">{int(k):,}</span>
              <span style="color:#7fa8c8;">{lots_p}L</span>
              <span style="color:#e8f4ff;">₹{entry_p:.2f}</span>
              <span style="color:#00d4ff;">₹{live_ltp:.2f}</span>
              <span style="color:{pnlc};font-weight:700;">₹{leg_pnl:+,.2f}</span>
              <span style="color:#3a6080;">-</span>
            </div>""", unsafe_allow_html=True)

            # Remove button beside each row
            if st.button(f"✕ Remove #{li+1}", key=f"rm_leg_{li}", use_container_width=False):
                legs_to_remove.append(li)

        for idx in sorted(legs_to_remove, reverse=True):
            st.session_state.basket_legs.pop(idx)
        if legs_to_remove:
            st.rerun()

        # ── Summary bar ─────────────────────────────────────────────────
        pnl_color_b = "#00e676" if total_pnl_b >= 0 else "#ff3d57"
        cred_color_b= "#00e676" if total_credit >= 0 else "#ff3d57"
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:10px;padding:12px 16px;
                    background:#111920;border:1px solid #1e3040;border-top:2px solid {pnl_color_b};
                    border-radius:3px;font-family:'JetBrains Mono',monospace;">
          <div style="flex:1;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                        letter-spacing:1.5px;color:#7fa8c8;">NET CREDIT/DEBIT</div>
            <div style="font-size:20px;font-weight:700;color:{cred_color_b};">
              ₹{total_credit:+,.2f}</div>
          </div>
          <div style="flex:1;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                        letter-spacing:1.5px;color:#7fa8c8;">LIVE P&L</div>
            <div style="font-size:20px;font-weight:700;color:{pnl_color_b};">
              ₹{total_pnl_b:+,.2f}</div>
          </div>
          <div style="flex:1;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                        letter-spacing:1.5px;color:#7fa8c8;">LEGS</div>
            <div style="font-size:20px;font-weight:700;color:#e8f4ff;">
              {len(st.session_state.basket_legs)}</div>
          </div>
          <div style="flex:1;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                        letter-spacing:1.5px;color:#7fa8c8;">LOT SIZE</div>
            <div style="font-size:20px;font-weight:700;color:#ff8c00;">{lot_size_b}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Payoff Chart ─────────────────────────────────────────────────
        st.markdown("---")
        section_header("📈 Payoff Diagram", "Profit/Loss at expiry across spot range")
        try:
            import plotly.graph_objects as go
            spot_range_b = np.linspace(spot_b * 0.93, spot_b * 1.07, 200)
            payoff_vals  = []
            for s in spot_range_b:
                pnl_at_s = 0.0
                for leg in st.session_state.basket_legs:
                    k   = float(leg["strike"])
                    ep  = float(leg["premium"])
                    lts = int(leg["lots"])
                    if leg["type"] == "CE":
                        intrinsic = max(s - k, 0)
                    else:
                        intrinsic = max(k - s, 0)
                    if leg["action"] == "SELL":
                        pnl_at_s += (ep - intrinsic) * lts * lot_size_b
                    else:
                        pnl_at_s += (intrinsic - ep) * lts * lot_size_b
                payoff_vals.append(pnl_at_s)

            payoff_colors = ["#00e676" if v >= 0 else "#ff3d57" for v in payoff_vals]
            payoff_fig = go.Figure()
            # Fill area
            payoff_fig.add_trace(go.Scatter(
                x=spot_range_b, y=payoff_vals,
                mode="lines", name="P&L at Expiry",
                line=dict(color="#ff8c00", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(255,140,0,0.08)",
            ))
            # Zero line
            payoff_fig.add_hline(y=0, line_color="#3a6080", line_dash="dot", line_width=1)
            # Spot marker
            payoff_fig.add_vline(x=float(spot_b), line_color="#00d4ff", line_dash="dash",
                                  line_width=1.5, annotation_text=f"SPOT {spot_b:,.0f}",
                                  annotation_font_color="#00d4ff", annotation_font_size=10)
            # ATM marker
            payoff_fig.add_vline(x=float(atm_b), line_color="#ff8c00", line_dash="dot",
                                  line_width=1, annotation_text=f"ATM {atm_b:,.0f}",
                                  annotation_font_color="#ff8c00", annotation_font_size=10)
            # Max profit / max loss annotations
            max_p = max(payoff_vals)
            max_l = min(payoff_vals)
            payoff_fig.add_annotation(
                x=spot_range_b[payoff_vals.index(max_p)], y=max_p,
                text=f"MAX PROFIT ₹{max_p:,.0f}", showarrow=True, arrowhead=2,
                font=dict(color="#00e676", size=10, family="JetBrains Mono"),
                bgcolor="#020f05", bordercolor="#00e676", arrowcolor="#00e676",
            )
            if max_l < 0:
                payoff_fig.add_annotation(
                    x=spot_range_b[payoff_vals.index(max_l)], y=max_l,
                    text=f"MAX LOSS ₹{max_l:,.0f}", showarrow=True, arrowhead=2,
                    font=dict(color="#ff3d57", size=10, family="JetBrains Mono"),
                    bgcolor="#120203", bordercolor="#ff3d57", arrowcolor="#ff3d57",
                )
            payoff_fig.update_layout(
                height=380, paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
                font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis=dict(gridcolor="#1a2a3a", title="Spot at Expiry", tickformat=",.0f"),
                yaxis=dict(gridcolor="#1a2a3a", title="Net P&L (₹)", tickformat=",.0f"),
                showlegend=False,
            )
            st.plotly_chart(payoff_fig, use_container_width=True)

            # Breakevens
            bes = []
            for i in range(1, len(payoff_vals)):
                if (payoff_vals[i-1] < 0 and payoff_vals[i] >= 0) or \
                   (payoff_vals[i-1] >= 0 and payoff_vals[i] < 0):
                    be_approx = (spot_range_b[i-1] + spot_range_b[i]) / 2
                    bes.append(round(be_approx, 1))
            if bes:
                be_str = "  ·  ".join([f"₹{b:,.1f}" for b in bes])
                st.markdown(f"""
                <div style="background:#0d1117;border:1px solid #ffd600;border-left:3px solid #ffd600;
                            border-radius:3px;padding:10px 16px;font-family:'JetBrains Mono',monospace;">
                  <span style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                               letter-spacing:1.5px;color:#ffd600;">⚡ BREAKEVEN POINT{'S' if len(bes)>1 else ''}:</span>
                  <span style="font-size:14px;font-weight:700;color:#ffd600;margin-left:12px;">{be_str}</span>
                </div>""", unsafe_allow_html=True)

        except ImportError:
            st.info("pip install plotly for payoff chart")
        except Exception as _be:
            st.warning(f"Payoff chart error: {_be}")

        # ── Action buttons ───────────────────────────────────────────────
        st.markdown("---")
        ba1, ba2, ba3 = st.columns(3)
        with ba1:
            if st.button("🚀 EXECUTE BASKET → TRADE LOG", use_container_width=True, type="primary"):
                now_b  = datetime.now(IST).strftime("%H:%M:%S")
                today_b= datetime.now().strftime("%Y-%m-%d")
                for leg in st.session_state.basket_legs:
                    st.session_state.ai_trade_log.append({
                        "Entry Time": now_b, "Date": today_b,
                        "Index_Key":  sel_b or "",
                        "Strike":     leg["strike"],
                        "Type":       f"{leg['action']} {leg['type']}",
                        "Action":     leg["action"],
                        "Entry":      leg["premium"],
                        "SL":         round(leg["premium"] * 1.5, 2) if leg["action"]=="SELL" else round(leg["premium"] * 0.5, 2),
                        "Target":     round(leg["premium"] * 0.3, 2) if leg["action"]=="SELL" else round(leg["premium"] * 1.8, 2),
                        "Lot_Size":   lot_size_b,
                        "Lots":       leg["lots"],
                        "Live LTP":   leg["premium"],
                        "Live_PnL":   0,
                        "Status":     "Active",
                        "Score":      0,
                        "Flow":       "Basket Trade",
                    })
                save_list_to_csv(st.session_state.ai_trade_log, AI_LOG_FILE)
                st.success(f"✅ {len(st.session_state.basket_legs)} legs executed → Trade Log (Tab 6)")

        with ba2:
            if st.button("🗑️ CLEAR ALL LEGS", use_container_width=True):
                st.session_state.basket_legs = []
                st.rerun()

        with ba3:
            if st.session_state.basket_legs:
                basket_csv = pd.DataFrame(st.session_state.basket_legs).to_csv(index=False)
                st.download_button("📥 EXPORT LEGS CSV", basket_csv,
                                    "basket_legs.csv", "text/csv",
                                    use_container_width=True)