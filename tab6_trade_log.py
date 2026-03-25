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
# TAB 6 — TRADE LOG
# ======================================================
def render():
    section_header("AI Trade Log", "Live P&L tracking with lot-adjusted returns, SL/target monitoring")

    option_data6 = st.session_state.get("option_data", {})
    today_str6   = datetime.now().strftime("%Y-%m-%d")
    total_pnl6   = 0

    for trade in st.session_state.ai_trade_log:

        # ── Skip non-active; accumulate their locked P&L ──────────────────
        if trade.get("Status") not in ("Active",):
            total_pnl6 += trade.get("Live_PnL", 0)
            continue

        idx_key = trade.get("Index_Key")
        if not idx_key or idx_key not in option_data6:
            continue
        df6  = option_data6[idx_key]
        row6 = df6[df6["Strike"] == trade["Strike"]]
        if row6.empty:
            continue

        # ── Basic values ───────────────────────────────────────────────────
        lot6 = int(trade.get("Lot_Size") or get_lot_size(idx_key))
        ltp6 = float(
            row6["CE_LTP"].iloc[0] if "CE" in trade["Type"]
            else row6["PE_LTP"].iloc[0]
        )
        trade["Live LTP"] = round(ltp6, 2)
        trade["Lot_Size"] = lot6

        entry6  = float(trade["Entry"])
        target6 = float(trade["Target"])

        # ── Direction flag  (BUY CE / BUY PE → is_buy = True) ────────────
        #    SELL CE / SELL PE → is_buy = False
        is_buy = str(trade["Type"]).upper().startswith("BUY")

        # ── FIX 1 ▸ Direction-aware P&L ───────────────────────────────────
        #    BUY  → profit when LTP rises  → pnl_pts = LTP − Entry
        #    SELL → profit when LTP falls  → pnl_pts = Entry − LTP
        if is_buy:
            pnl_pts6 = round(ltp6 - entry6, 2)
        else:
            pnl_pts6 = round(entry6 - ltp6, 2)

        pnl_rupee6        = round(pnl_pts6 * lot6, 2)
        trade["Live_PnL"] = pnl_rupee6

        # ── FIX 2 ▸ Direction-aware Trailing SL ───────────────────────────
        #    progress = how far LTP has moved toward target as a fraction
        if is_buy:
            denom6    = (target6 - entry6) if (target6 - entry6) != 0 else 1
            progress6 = (ltp6 - entry6) / denom6
        else:
            denom6    = (entry6 - target6) if (entry6 - target6) != 0 else 1
            progress6 = (entry6 - ltp6) / denom6

        if progress6 >= 0.8:
            if is_buy:
                # Trail SL up to halfway between entry and target
                new_sl6 = round(entry6 + (target6 - entry6) * 0.5, 2)
                if new_sl6 > trade["SL"]:       # only move SL higher
                    trade["SL"] = new_sl6
            else:
                # Trail SL down to halfway between entry and target
                new_sl6 = round(entry6 - (entry6 - target6) * 0.5, 2)
                if new_sl6 < trade["SL"]:       # only move SL lower
                    trade["SL"] = new_sl6
        elif progress6 >= 0.5:
            # Move SL to breakeven for both directions
            trade["SL"] = entry6

        # ── FIX 3 ▸ Direction-aware SL / Target hit detection ─────────────
        #    BUY  → SL triggered when price DROPS to/below SL
        #           Target triggered when price RISES to/above Target
        #    SELL → SL triggered when price RISES to/above SL
        #           Target triggered when price DROPS to/below Target
        if is_buy:
            sl_hit6     = ltp6 <= trade["SL"]
            target_hit6 = ltp6 >= trade["Target"]
        else:
            sl_hit6     = ltp6 >= trade["SL"]
            target_hit6 = ltp6 <= trade["Target"]

        # ── Close position helpers ─────────────────────────────────────────
        def _close(status_label):
            trade["Status"] = status_label
            st.session_state.closed_positions.append({
                "symbol"    : f"{trade['Type']}_{trade['Strike']}",
                "Date"      : today_str6,
                "final_pnl" : pnl_rupee6,
                "pnl_pts"   : pnl_pts6,
                "lot_size"  : lot6,
                "close_price": ltp6,
                "exit_time" : datetime.now(IST).strftime("%H:%M:%S"),
                "entry_time": trade.get("Entry Time", ""),
                "direction" : "BUY" if is_buy else "SELL",
            })
            save_list_to_csv(st.session_state.closed_positions, CLOSED_POS_FILE)

        if trade["Status"] == "Active":
            if sl_hit6:
                _close("SL Hit")
            elif target_hit6:
                _close("Target Hit")

        total_pnl6 += pnl_rupee6

    # ── Summary metrics ────────────────────────────────────────────────────
    active_count6 = sum(1 for t in st.session_state.ai_trade_log if t.get("Status") == "Active")
    sl_count6     = sum(1 for t in st.session_state.ai_trade_log if t.get("Status") == "SL Hit")
    tgt_count6    = sum(1 for t in st.session_state.ai_trade_log if t.get("Status") == "Target Hit")

    pnl_c6 = "#00e676" if total_pnl6 >= 0 else "#ff3d57"
    metrics_row(
        metric_card("TOTAL P&L",  f"₹{total_pnl6:,.0f}", "Lot-adjusted real ₹", pnl_c6)  +
        metric_card("ACTIVE",     f"{active_count6}",    "",                     "#00d4ff") +
        metric_card("SL HIT",     f"{sl_count6}",        "",                     "#ff3d57") +
        metric_card("TARGET HIT", f"{tgt_count6}",       "",                     "#ffd600") +
        metric_card("TOTAL",      f"{len(st.session_state.ai_trade_log)}", "",   "#7fa8c8")
    )

    st.caption(
        "Live_PnL = (LTP − Entry) × Lot Size  [BUY]  |  (Entry − LTP) × Lot Size  [SELL]  |  "
        + " | ".join(f"{k.split('|')[1]}: {v}" for k, v in LOT_SIZES.items())
    )

    # ── Trade table ────────────────────────────────────────────────────────
    if st.session_state.ai_trade_log:
        df_log6 = pd.DataFrame(st.session_state.ai_trade_log)
        show6   = [c for c in [
                        "Entry Time", "Index_Key", "Strike", "Type", "Flow",
                        "Entry", "Live LTP", "SL", "Target",
                        "Lot_Size", "Live_PnL", "Score", "Status"
                   ] if c in df_log6.columns]

        def color_log_rows(row):
            status = row.get("Status", "")
            pnl    = row.get("Live_PnL", 0)
            if status == "Target Hit":
                return ["background-color:#051a0e; color:#00e676;"] * len(row)
            elif status == "SL Hit":
                return ["background-color:#1a0508; color:#ff3d57;"] * len(row)
            elif pnl > 0:
                return ["background-color:#020e06;"] * len(row)
            elif pnl < 0:
                return ["background-color:#0e0202;"] * len(row)
            return [""] * len(row)

        float6_cols = {
            c: "₹{:.2f}" for c in ["Entry", "Live LTP", "SL", "Target"] if c in show6
        }
        float6_cols["Live_PnL"] = "₹{:,.2f}"

        styled_log6 = (
            df_log6[show6]
            .style
            .apply(color_log_rows, axis=1)
            .format(float6_cols, na_rep="-")
        )
        st.dataframe(styled_log6, use_container_width=True, height=400)

        # ── P&L breakdown by direction ─────────────────────────────────────
        if "Type" in df_log6.columns and "Live_PnL" in df_log6.columns:
            df_log6["Direction"] = df_log6["Type"].str.upper().apply(
                lambda x: "BUY" if x.startswith("BUY") else "SELL"
            )
            buy_pnl  = df_log6[df_log6["Direction"] == "BUY"]["Live_PnL"].sum()
            sell_pnl = df_log6[df_log6["Direction"] == "SELL"]["Live_PnL"].sum()

            c_buy6, c_sell6, c_net6 = st.columns(3)
            c_buy6.metric(
                "BUY Legs P&L",
                f"₹{buy_pnl:,.2f}",
                delta=f"{'▲' if buy_pnl >= 0 else '▼'} {abs(buy_pnl):,.2f}",
                delta_color="normal"
            )
            c_sell6.metric(
                "SELL Legs P&L",
                f"₹{sell_pnl:,.2f}",
                delta=f"{'▲' if sell_pnl >= 0 else '▼'} {abs(sell_pnl):,.2f}",
                delta_color="normal"
            )
            c_net6.metric(
                "Net Combined",
                f"₹{(buy_pnl + sell_pnl):,.2f}",
                delta_color="normal"
            )

        # ── Modify SL / Target ─────────────────────────────────────────────
        with st.expander("✏️ MODIFY SL / TARGET"):
            t_idx6 = st.number_input(
                "Trade Index", 0, len(df_log6) - 1, key="t6_tidx"
            )
            trade_row6 = df_log6.iloc[int(t_idx6)]

            # Show direction context so user knows which way SL/Target should be
            direction_label6 = (
                "BUY  → SL below entry, Target above entry"
                if str(trade_row6.get("Type", "")).upper().startswith("BUY")
                else "SELL → SL above entry, Target below entry"
            )
            st.caption(f"Direction: **{direction_label6}**")

            c_sl6, c_tg6 = st.columns(2)
            with c_sl6:
                new_sl6 = st.number_input(
                    "New SL",
                    value=float(trade_row6["SL"]),
                    key="t6_sl"
                )
            with c_tg6:
                new_tg6 = st.number_input(
                    "New Target",
                    value=float(trade_row6["Target"]),
                    key="t6_tg"
                )
            if st.button("💾 UPDATE", key="t6_update"):
                st.session_state.ai_trade_log[int(t_idx6)]["SL"]     = new_sl6
                st.session_state.ai_trade_log[int(t_idx6)]["Target"] = new_tg6
                st.success("✅ SL / Target updated!")

        # ── Clear closed trades ────────────────────────────────────────────
        with st.expander("🗑️ MANAGE TRADES"):
            col_clr1, col_clr2 = st.columns(2)
            with col_clr1:
                if st.button("🧹 Clear SL Hit Trades", key="t6_clr_sl"):
                    st.session_state.ai_trade_log = [
                        t for t in st.session_state.ai_trade_log
                        if t.get("Status") != "SL Hit"
                    ]
                    st.success("Cleared SL Hit trades.")
            with col_clr2:
                if st.button("🧹 Clear Target Hit Trades", key="t6_clr_tgt"):
                    st.session_state.ai_trade_log = [
                        t for t in st.session_state.ai_trade_log
                        if t.get("Status") != "Target Hit"
                    ]
                    st.success("Cleared Target Hit trades.")
            if st.button("🗑️ Clear ALL Trades", key="t6_clr_all", type="primary"):
                st.session_state.ai_trade_log = []
                st.success("All trades cleared.")

    else:
        st.markdown(
            """<div style="background:#0d1117;border:1px solid #1e3040;padding:20px;
            text-align:center;color:#3a6080;font-family:'Barlow Condensed',sans-serif;
            letter-spacing:1px;">
            NO AI TRADES YET — USE TAB 5 TO GENERATE</div>""",
            unsafe_allow_html=True
        )