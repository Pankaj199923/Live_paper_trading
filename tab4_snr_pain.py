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
# TAB 4 — S&R + MAX PAIN
# ======================================================
def render():
    section_header("Support & Resistance + Max Pain", "Key OI levels, volume walls, max pain strike")

    oc_t4  = st.session_state.get("current_option_chain", pd.DataFrame())
    spot_t4= st.session_state.get("current_spot_price")
    sel_t4 = st.session_state.get("current_selected_index", "")

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
    section_header("Max Pain Analysis", "Total writer pain at each strike — market tends to drift toward minimum pain")
    try:
        import plotly.graph_objects as go
        pain_strikes = list(pain_dict4.keys())
        pain_values  = [pain_dict4[s]/1e7 for s in pain_strikes]
        min_pain_val = min(pain_values)
        colors_pain  = ["#ffd600" if s == max_pain_strike4 else
                         "#ff8c00" if abs(s - max_pain_strike4) <= 100 else "#1e3040"
                         for s in pain_strikes]
        pain_fig = go.Figure()
        pain_fig.add_bar(x=pain_strikes, y=pain_values,
                         marker_color=colors_pain, name="Writer Pain")
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
    except ImportError: pass


    # ======================================================
