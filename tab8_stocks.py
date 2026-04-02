import os
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
# TAB 8 — STOCKS TERMINAL
# ======================================================
def render():
    import streamlit as st
    import pandas as pd

    st.session_state["active_tab_key"] = "🖥 STOCKS"

    st.title("📊 NSE STOCKS — AUTO SCANNER")

    if instrument_df.empty:
        st.warning("❌ instrument_df not loaded")
        return

    # ==============================
    # 🔹 FILTER UI
    # ==============================
    col1, col2, col3 = st.columns(3)

    with col1:
        min_pct = st.number_input("Min % Move", value=1.0)

    with col2:
        direction = st.selectbox("Direction", ["UP", "DOWN", "BOTH"])

    with col3:
        refresh = st.button("🔄 Refresh")

    # ==============================
    # 🔹 ALL STOCKS
    # ==============================
    sel_df = instrument_df.copy()

    if sel_df.empty:
        st.warning("No instruments found")
        return

    # ==============================
    # 🔥 FAST LTP FETCH (ONE CALL)
    # ==============================
    try:
        keys = sel_df["instrument_key"].tolist()
        ltp_df = fetch_ltp(keys)

        if ltp_df.empty:
            st.error("❌ LTP API returned empty")
            return

        # Map LTP
        ltp_map = dict(zip(ltp_df["instrument_key"], ltp_df["Spot Price"]))

    except Exception as e:
        st.error(f"LTP Fetch Error: {e}")
        return

    # ==============================
    # 🔹 BUILD DATA
    # ==============================
    rows = []

    for _, row in sel_df.iterrows():
        sym = row["Symbol"]
        key = row["instrument_key"]

        ltp = ltp_map.get(key, None)

        # 🔥 FETCH PREV CLOSE (CACHED)
        prev_close = st.session_state.get(f"prev_close_{sym}")

        if prev_close is None:
            prev_close = _fetch_prev_close(key)
            if prev_close:
                st.session_state[f"prev_close_{sym}"] = prev_close

        # Calculate change
        if ltp and prev_close and prev_close > 0:
            chg_pct = ((ltp - prev_close) / prev_close) * 100
            chg_rs = ltp - prev_close
        else:
            chg_pct = None
            chg_rs = None

        rows.append({
            "Symbol": sym,
            "LTP": ltp,
            "Prev Close": prev_close,
            "Chg%": chg_pct,
            "Chg ₹": chg_rs
        })

    df = pd.DataFrame(rows)

    # ==============================
    # 🔥 CLEAN DATA (IMPORTANT)
    # ==============================
    df = df.dropna(subset=["LTP", "Prev Close", "Chg%"])

    if df.empty:
        st.warning("⚠️ No valid data (LTP/Prev Close missing)")
        st.write(df.head())
        return

    # ==============================
    # 🔹 APPLY FILTER
    # ==============================
    if direction == "UP":
        df = df[df["Chg%"] >= min_pct]

    elif direction == "DOWN":
        df = df[df["Chg%"] <= -min_pct]

    else:
        df = df[abs(df["Chg%"]) >= min_pct]

    if df.empty:
        st.warning("⚠️ No stocks match filter — try lowering %")
        return

    # ==============================
    # 🔥 TOP 50 MOVERS
    # ==============================
    df["AbsChg"] = df["Chg%"].abs()
    df = df.sort_values("AbsChg", ascending=False).head(50)

    # ==============================
    # 🔹 DISPLAY TABLE
    # ==============================
    st.subheader(f"🔥 Top 50 Movers ({direction} | > {min_pct}%)")

    df_display = df[["Symbol", "LTP", "Prev Close", "Chg ₹", "Chg%"]].copy()

    df_display["LTP"] = df_display["LTP"].round(2)
    df_display["Prev Close"] = df_display["Prev Close"].round(2)
    df_display["Chg%"] = df_display["Chg%"].round(2)
    df_display["Chg ₹"] = df_display["Chg ₹"].round(2)

    st.dataframe(df_display, use_container_width=True)

    # ==============================
    # 🔹 STATS
    # ==============================
    adv = (df["Chg%"] > 0).sum()
    dec = (df["Chg%"] < 0).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Stocks", len(df))
    col2.metric("Gainers", adv)
    col3.metric("Losers", dec)

    # ==============================
    # 🔍 DEBUG (REMOVE LATER)
    # ==============================
    with st.expander("🔍 Debug Data"):
        st.write(df.head())