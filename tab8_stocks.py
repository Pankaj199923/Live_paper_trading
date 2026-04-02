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
    st.session_state["active_tab_key"] = "🖥 STOCKS"

    st.title("📊 NSE STOCKS — AUTO SCANNER")

    if instrument_df.empty:
        st.warning("NSECMI.csv not found")
        return

    # ==============================
    # 🔹 FILTER SETTINGS
    # ==============================
    col1, col2, col3 = st.columns(3)

    with col1:
        min_pct = st.number_input("Min % Move", value=1.0)

    with col2:
        direction = st.selectbox("Direction", ["UP", "DOWN", "BOTH"])

    with col3:
        refresh_btn = st.button("🔄 Refresh")

    # ==============================
    # 🔹 USE ALL STOCKS
    # ==============================
    sel_df8 = instrument_df.copy()

    rows8 = []

    for _, row in sel_df8.iterrows():
        sym = row["Symbol"]

        try:
            ltp_df = fetch_ltp([row["instrument_key"]])
            ltp = float(ltp_df["Spot Price"].iloc[0]) if not ltp_df.empty else None
        except:
            ltp = None

        prev_close = st.session_state.get(f"prev_close_{sym}")

        if prev_close is None:
            try:
                prev_close = row.get("Close", None)  # fallback
            except:
                prev_close = None

        chg = ((ltp - prev_close) / prev_close * 100) if ltp and prev_close else 0
        chg_rs = (ltp - prev_close) if ltp and prev_close else 0

        rows8.append({
            "Symbol": sym,
            "LTP": ltp,
            "Prev Close": prev_close,
            "Chg%": chg,
            "Chg ₹": chg_rs
        })

    df = pd.DataFrame(rows8)

    # ==============================
    # 🔹 APPLY % FILTER
    # ==============================
    if direction == "UP":
        df = df[df["Chg%"] >= min_pct]

    elif direction == "DOWN":
        df = df[df["Chg%"] <= -min_pct]

    else:
        df = df[abs(df["Chg%"]) >= min_pct]

    # ==============================
    # 🔥 TOP 50 MOVERS ONLY
    # ==============================
    df["AbsChg"] = df["Chg%"].abs()
    df = df.sort_values("AbsChg", ascending=False).head(50)

    # ==============================
    # 🔹 DISPLAY
    # ==============================
    st.subheader(f"🔥 Top 50 Movers ({direction} | > {min_pct}%)")

    if df.empty:
        st.warning("No stocks match criteria")
        return

    # Table display
    df_display = df[["Symbol", "LTP", "Prev Close", "Chg ₹", "Chg%"]].copy()

    df_display["Chg%"] = df_display["Chg%"].round(2)
    df_display["LTP"] = df_display["LTP"].round(2)

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