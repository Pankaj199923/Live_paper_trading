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
# TAB 7 — GREEKS LAB
# ======================================================
def render():
    section_header("Greeks Laboratory", "B-S pricer, IV calculator, volatility surface, portfolio Greeks")

    oc_t7   = st.session_state.get("current_option_chain", pd.DataFrame())
    spot_t7 = st.session_state.get("current_spot_price", 0)
    sel_t7  = st.session_state.get("current_selected_index", "")

    if oc_t7 is None or (isinstance(oc_t7, pd.DataFrame) and oc_t7.empty) or not spot_t7:
        st.info("🔄 Load option chain from Tab 1 first."); return

    exp_t7   = st.session_state.get("oc_expiry_select", "2026-03-27")
    exp_dt7  = pd.to_datetime(exp_t7 if exp_t7 else "2026-03-27")
    dte7     = max(1, (exp_dt7 - datetime.now()).days)
    T7       = dte7 / 365
    r7       = 0.065

    gl1, gl2 = st.columns([1, 1])

    with gl1:
        section_header("B-S Option Pricer + Greeks")
        g_S    = st.number_input("Spot Price (S)", value=float(spot_t7), step=10.0, key="g_S")
        g_K    = st.number_input("Strike (K)", value=float(get_atm_strike(spot_t7, sel_t7)), step=50.0, key="g_K")
        g_T    = st.number_input("Days to Expiry", value=float(dte7), step=1.0, key="g_T")
        g_r    = st.number_input("Risk-Free Rate %", value=6.5, step=0.1, key="g_r") / 100
        g_sig  = st.number_input("IV % (sigma)", value=15.0, step=0.5, key="g_sig") / 100
        g_type = st.radio("Option Type", ["CE (Call)", "PE (Put)"], horizontal=True, key="g_type")
        g_opt  = 'c' if "CE" in g_type else 'p'
        g_T_yr = g_T / 365

        if st.button("⚡ CALCULATE", use_container_width=True):
            price_t7 = bs_price(g_S, g_K, g_T_yr, g_r, g_sig, g_opt)
            greeks_t7= bs_greeks(g_S, g_K, g_T_yr, g_r, g_sig, g_opt)
            st.markdown(f""" <div style="background:#0d1117;border:1px solid #ff8c00;border-radius:3px;padding:14px;margin-top:10px;"><div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:2px;color:#ff8c00;margin-bottom:8px;">THEORETICAL PRICE</div><div style="font-family:'JetBrains Mono',monospace;font-size:32px;font-weight:700;color:#e8f4ff;">₹{price_t7:.2f}</div><div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px;"> {"".join(f'''<div style="text-align:center;background:#111920;border:1px solid #1e3040;border-radius:2px;padding:8px;"><div style="font-family:Barlow Condensed,sans-serif;font-size:9px;letter-spacing:1px;color:#7fa8c8;">{k.upper()}</div><div style="font-family:JetBrains Mono,monospace;font-size:14px;font-weight:600;color:{("#00e676" if float(v)>0 else "#ff3d57") if k not in ["gamma"] else "#00d4ff"};">{v}</div></div>''' for k, v in greeks_t7.items())} </div></div>""", unsafe_allow_html=True)

    with gl2:
        section_header("IV Calculator")
        iv_S   = st.number_input("Spot (S)", value=float(spot_t7), step=10.0, key="iv_S")
        iv_K   = st.number_input("Strike (K)", value=float(get_atm_strike(spot_t7, sel_t7)), step=50.0, key="iv_K")
        iv_mkt = st.number_input("Market Price (₹)", value=100.0, step=1.0, key="iv_mkt")
        iv_T   = st.number_input("Days to Expiry", value=float(dte7), step=1.0, key="iv_T")
        iv_r   = 0.065
        iv_type= st.radio("Type", ["CE", "PE"], horizontal=True, key="iv_type")
        iv_opt = 'c' if iv_type == "CE" else 'p'

        if st.button("🧮 SOLVE IV", use_container_width=True):
            iv_result = implied_vol_newton(iv_mkt, iv_S, iv_K, iv_T/365, iv_r, iv_opt)
            iv_pct    = iv_result * 100
            iv_color7 = "#00e676" if iv_pct < 20 else "#ffd600" if iv_pct < 30 else "#ff3d57"
            st.markdown(f""" <div style="background:#0d1117;border:1px solid {iv_color7};border-radius:3px;padding:14px;margin-top:10px;"><div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:2px;color:{iv_color7};margin-bottom:4px;">IMPLIED VOLATILITY</div><div style="font-family:'JetBrains Mono',monospace;font-size:36px;font-weight:700;color:{iv_color7};">{iv_pct:.2f}%</div><div style="font-family:'Barlow Condensed',sans-serif;font-size:12px;color:#7fa8c8;margin-top:4px;"> {"🟢 Low IV — options cheap" if iv_pct < 20 else "🟡 Normal IV" if iv_pct < 30 else "🔴 High IV — options expensive"}</div></div>""", unsafe_allow_html=True)

    # IV Surface
    st.markdown("---")
    section_header("IV Smile / Skew", "Implied volatility across strikes")
    try:
        import plotly.graph_objects as go
        oc_iv = oc_t7.copy()
        oc_iv = oc_iv[(oc_iv["CE_IV"] > 0) & (oc_iv["PE_IV"] > 0)]
        iv_fig = go.Figure()
        iv_fig.add_scatter(x=oc_iv["Strike"], y=oc_iv["CE_IV"],
                           mode='lines+markers', name="CE IV", line=dict(color="#ff3d57", width=2),
                           marker=dict(size=5))
        iv_fig.add_scatter(x=oc_iv["Strike"], y=oc_iv["PE_IV"],
                           mode='lines+markers', name="PE IV", line=dict(color="#00e676", width=2),
                           marker=dict(size=5))
        iv_fig.add_vline(x=spot_t7, line_color="#ff8c00", line_dash="dot", line_width=1.5,
                         annotation_text="SPOT", annotation_font_color="#ff8c00", annotation_font_size=10)
        iv_fig.update_layout(
            height=300, paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
            font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor="#1e3040", title="Strike"),
            yaxis=dict(gridcolor="#1e3040", title="IV %"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7fa8c8")
        )
        st.plotly_chart(iv_fig, use_container_width=True)
    except ImportError:
        st.info("pip install plotly for IV chart")

    # Greeks Table
    st.markdown("---")
    section_header("Strike-wise Greeks", "Delta, Gamma, Theta, Vega for all strikes")
    greek_cols = ["Strike", "CE_Delta","CE_Gamma","CE_Theta","CE_Vega","CE_IV",
                  "PE_Delta","PE_Gamma","PE_Theta","PE_Vega","PE_IV","IV_Skew"]
    greek_cols = [c for c in greek_cols if c in oc_t7.columns]
    atm7       = get_atm_strike(spot_t7, sel_t7)

    def style_greeks(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for idx in df.index:
            if 'Strike' in df.columns and df.at[idx, 'Strike'] == atm7:
                styles.loc[idx, :] = 'background-color:#1a1200;font-weight:bold;'
        return styles

    g_float_cols7 = {c: "{:.4f}" for c in greek_cols if c not in ["Strike"]}
    styled_greeks7 = oc_t7[greek_cols].style.apply(style_greeks, axis=None).format(g_float_cols7)
    st.dataframe(styled_greeks7, use_container_width=True, height=400)


    # ======================================================
