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
# TAB 1 — OPTION CHAIN (Pro)
# ======================================================
def render():
    section_header("Live Option Chain", "Real-time strike-wise OI, OI, Greeks, Volume")

    c_idx, c_exp = st.columns([1, 1])
    with c_idx:
        selected_index = st.selectbox("Index", df_indices['index'].unique().tolist(),
                                      format_func=lambda x: INDEX_SHORT.get(x, x),
                                      key="oc_index_select")
    spot_df    = fetch_ltp([selected_index])
    spot_price = None
    if isinstance(spot_df, pd.DataFrame) and not spot_df.empty:
        spot_price = float(spot_df["Spot Price"].iloc[0])
    if not spot_price:
        st.warning("⏳ Spot price unavailable"); return

    with c_exp:
        live_exp     = fetch_available_expiries(selected_index)
        fallback_exp = df_indices[df_indices['index'] == selected_index]['expiry'].unique().tolist()
        expiry_list  = live_exp if live_exp else fallback_exp
        selected_expiry = st.selectbox("Expiry", expiry_list, key="oc_expiry_select")

    oc = fetch_option_chain(selected_index, selected_expiry)
    if oc.empty:
        st.warning("Option chain data not available"); return

    oc["Strike"] = pd.to_numeric(oc["Strike"], errors="coerce")
    oc = oc.dropna(subset=["Strike"])
    index_name  = selected_index.split('|')[1]
    range_value = 1000 if index_name in ["Nifty Bank", "BANKEX", "SENSEX"] else 500
    oc          = oc[(oc['Strike'] >= spot_price - range_value) & (oc['Strike'] <= spot_price + range_value)]

    # Persist
    st.session_state.current_option_chain   = oc
    st.session_state.option_chain_df        = oc
    st.session_state.current_selected_index = selected_index
    st.session_state.current_spot_price     = spot_price
    st.session_state.option_data[selected_index] = oc
    # Save per-index spot for header ticker bar
    st.session_state[f"spot_{selected_index}"] = spot_price

    # ── 1-Minute Snapshot Trigger (fires at exact HH:MM boundary) ──
    _cur_minute  = now_ist_dt.strftime("%H:%M")
    _last_snap   = st.session_state.get("last_snapshot_time")
    _last_minute = _last_snap.strftime("%H:%M") if _last_snap else None
    _snap_due    = (_last_minute is None or _cur_minute != _last_minute)
    if _snap_due and MARKET_OPEN:
        _saved = save_oc_snapshot(oc, spot_price, selected_index, selected_expiry)
        if _saved:
            st.session_state.last_snapshot_time = now_ist_dt
    # ─────────────────────────────────────────────────────────────

    # Spot history
    spot_hist = st.session_state.spot_history
    spot_hist.append({"time": now_ist_dt, "spot": spot_price})
    st.session_state.spot_history = spot_hist[-60:]

    atm_strike = get_atm_strike(spot_price, selected_index)
    st.session_state.atm_strike = atm_strike

    # Key metrics row
    total_ce_oi = oc["CE_OI"].sum()
    total_pe_oi = oc["PE_OI"].sum()
    pcr_val     = total_pe_oi / total_ce_oi if total_ce_oi else 0
    total_ce_vol= oc["CE_Volume"].sum()
    total_pe_vol= oc["PE_Volume"].sum()
    atm_row1    = oc[oc["Strike"] == atm_strike]
    atm_ce_oi   = float(atm_row1["CE_OI"].iloc[0]) if not atm_row1.empty else 0
    atm_pe_oi   = float(atm_row1["PE_OI"].iloc[0]) if not atm_row1.empty else 0

    metrics_row(
        metric_card("SPOT", f"₹{spot_price:,.0f}", color="#ff8c00") +
        metric_card("ATM STRIKE", f"{atm_strike:,.0f}", color="#00d4ff") +
        metric_card("PCR", f"{pcr_val:.3f}", "Put/Call OI Ratio", "#c084fc") +
        metric_card("CE OI (ATM)", f"{atm_ce_oi/1e5:.1f}L", color="#ff3d57") +
        metric_card("PE OI (ATM)", f"{atm_pe_oi/1e5:.1f}L", color="#00e676") +
        metric_card("CE VOLUME", f"{total_ce_vol/1e5:.1f}L", color="#ffd600") +
        metric_card("PE VOLUME", f"{total_pe_vol/1e5:.1f}L", color="#ffd600")
    )

    # PCR History
    pcr_hist = st.session_state.pcr_history
    pcr_hist.append({"time": datetime.now(IST).strftime("%H:%M"), "PCR": round(pcr_val, 3)})
    st.session_state.pcr_history = pcr_hist[-78:]

    # Compute signal scores
    bull1, bear1, pcr1, pcr_chg1, m_res1, m_sup1, factors1 = compute_signal_score(
        oc, spot_price, selected_index)

    # Alert engine
    alerts = check_alerts(oc, spot_price, selected_index)
    for a_type, a_msg, a_color in alerts:
        st.markdown(f"""<div style="background:#1a0a00;border:1px solid #ff8c00;border-left:3px solid #ff8c00;padding:6px 12px;border-radius:2px;margin:4px 0;font-family:'JetBrains Mono',monospace;font-size:12px;color:#ff8c00;">⚡ <b>{a_type}</b> — {a_msg}</div>""", unsafe_allow_html=True)

    # Chain view toggle
    view_mode = st.radio("View", ["Full Chain", "ATM ±5 Strikes", "Signal Focus"],
                         horizontal=True, key="chain_view")

    oc_display = oc.copy()
    if view_mode == "ATM ±5 Strikes":
        strikes     = sorted(oc_display["Strike"].unique())
        if atm_strike in strikes:
            ai = strikes.index(atm_strike)
            sel_s = strikes[max(0, ai-5): ai+6]
            oc_display = oc_display[oc_display["Strike"].isin(sel_s)]
    elif view_mode == "Signal Focus":
        # Show only strikes with notable OI change
        threshold   = oc_display["CE_OI_Change"].abs().quantile(0.7)
        notable     = oc_display[(oc_display["CE_OI_Change"].abs() > threshold) |
                                  (oc_display["PE_OI_Change"].abs() > threshold)]
        oc_display  = notable

    # Columns to show
    display_cols = ["Strike", "CE_IV", "CE_Delta", "CE_OI", "CE_OI_Change", "CE_OI_Change_%",
                    "CE_Volume", "CE_LTP",
                    "PE_LTP", "PE_Volume", "PE_OI_Change_%", "PE_OI_Change",
                    "PE_OI", "PE_Delta", "PE_IV"]
    display_cols = [c for c in display_cols if c in oc_display.columns]
    oc_show      = oc_display[display_cols].copy()

    def style_chain(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for idx in df.index:
            strike = df.at[idx, 'Strike'] if 'Strike' in df.columns else 0
            if strike == atm_strike:
                styles.loc[idx, :] = 'background-color:#1a1200;border-top:1px solid #ffd600;border-bottom:1px solid #ffd600;font-weight:bold;'
            else:
                if 'CE_OI_Change' in df.columns and df.at[idx, 'CE_OI_Change'] > 100000:
                    styles.at[idx, 'CE_OI_Change'] = 'color:#ff3d57;font-weight:600;'
                if 'PE_OI_Change' in df.columns and df.at[idx, 'PE_OI_Change'] > 100000:
                    styles.at[idx, 'PE_OI_Change'] = 'color:#00e676;font-weight:600;'
        return styles

    float_cols = oc_show.select_dtypes(include=['float64','float32']).columns
    styled_chain = oc_show.style.apply(style_chain, axis=None).format(
        {c: "{:.2f}" for c in float_cols})
    st.dataframe(styled_chain, use_container_width=True, height=520)

    # Signal Engine
    st.markdown("---")
    section_header("10-Factor Signal Engine", "Institutional signal across OI, PCR, Volume, GEX, IV Skew, Theta")

    sig_cols = st.columns(len(factors1))
    for col_s, (fname, (fval, fb, fbr)) in zip(sig_cols, factors1.items()):
        with col_s:
            color = "#00e676" if fb > 0 else "#ff3d57" if fbr > 0 else "#7fa8c8"
            st.markdown(f'<div style="background:#0d1117;border:1px solid #1e3040;border-top:2px solid {color};border-radius:3px;padding:8px;text-align:center;"><div style="font-family:Barlow Condensed,sans-serif;font-size:9px;letter-spacing:1px;color:#7fa8c8;text-transform:uppercase;">{fname}</div><div style="font-family:JetBrains Mono,monospace;font-size:12px;color:{color};font-weight:600;margin-top:4px;">{fval}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        bar_b = score_bar(bull1, 20, "#00e676")
        st.markdown(f'<div style="background:#0d1117;border:1px solid #1e3040;padding:12px;border-radius:3px;"><div style="font-family:Barlow Condensed,sans-serif;font-size:11px;letter-spacing:1px;color:#7fa8c8;">BULLISH SCORE</div><div style="font-family:JetBrains Mono,monospace;font-size:28px;font-weight:700;color:#00e676;">{bull1}<span style="font-size:14px;color:#3a6080;">/20</span></div>{bar_b}</div>', unsafe_allow_html=True)
    with sc2:
        bar_r = score_bar(bear1, 20, "#ff3d57")
        st.markdown(f'<div style="background:#0d1117;border:1px solid #1e3040;padding:12px;border-radius:3px;"><div style="font-family:Barlow Condensed,sans-serif;font-size:11px;letter-spacing:1px;color:#7fa8c8;">BEARISH SCORE</div><div style="font-family:JetBrains Mono,monospace;font-size:28px;font-weight:700;color:#ff3d57;">{bear1}<span style="font-size:14px;color:#3a6080;">/20</span></div>{bar_r}</div>', unsafe_allow_html=True)
    with sc3:
        st.metric("PCR", f"{pcr1:.3f}", f"{pcr_chg1:+.3f}")
    with sc4:
        st.metric("Resistance", f"{m_res1:,.0f}")
        st.metric("Support",    f"{m_sup1:,.0f}")

    # PCR Chart
    if len(st.session_state.pcr_history) >= 3:
        st.markdown("---")
        section_header("PCR Trend", "Intraday Put/Call Ratio history")
        pcr_df_chart = pd.DataFrame(st.session_state.pcr_history).set_index("time")
        st.line_chart(pcr_df_chart, color="#c084fc")


    # ======================================================
