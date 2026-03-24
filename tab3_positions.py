import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
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
from api import fetch_ltp, fetch_available_expiries, fetch_option_chain
from analytics import (bs_price, bs_greeks, implied_vol_newton, calculate_gamma_bs,
                       compute_signal_score, generate_ai_trade, check_alerts, call_claude_trade_setup)

# ──────────────────────────────────────────────
# HTML helper functions — keeps render() clean
# ──────────────────────────────────────────────

def _loss_bar_html(pct_used: float, limit3: float, bar_c3: str) -> str:
    return (
        f'<div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;'
        f'padding:10px 14px;margin-bottom:12px;">'
        f'<div style="display:flex;justify-content:space-between;font-family:\'Barlow Condensed\',sans-serif;'
        f'font-size:11px;letter-spacing:1px;color:#7fa8c8;margin-bottom:6px;">'
        f'<span>DAILY LOSS UTILIZATION (DRAWDOWN FROM PEAK)</span>'
        f'<span style="color:{bar_c3};">{pct_used:.0f}% of ₹{abs(limit3):,}</span></div>'
        f'<div style="background:#1e3040;border-radius:2px;height:6px;">'
        f'<div style="background:{bar_c3};width:{pct_used:.0f}%;height:100%;border-radius:2px;'
        f'box-shadow:0 0 8px {bar_c3}66;transition:width 0.3s;"></div></div></div>'
    )


def _closed_trade_row_html(symbol: str, close_price: float, exit_time: str,
                            pnl: float, pnl_c: str) -> str:
    return (
        f'<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid {pnl_c};'
        f'border-radius:2px;padding:8px 12px;margin:4px 0;display:flex;'
        f'justify-content:space-between;align-items:center;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#e8f4ff;">{symbol}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#7fa8c8;">₹{close_price:.2f}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#3a6080;">{exit_time}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:14px;font-weight:700;'
        f'color:{pnl_c};">₹{pnl:,.0f}</span></div>'
    )


def _pos_bar_html(pnl_c: str, pnl_pct: float) -> str:
    return (
        f'<div style="background:#1e3040;height:2px;border-radius:1px;margin:-10px 0 8px 0;">'
        f'<div style="background:{pnl_c};width:{min(abs(pnl_pct), 100):.0f}%;'
        f'height:100%;border-radius:1px;"></div></div>'
    )


def _nearest_weekly_expiry() -> str:
    """Return the next Thursday as a fallback expiry (NSE weekly default)."""
    today = date.today()
    days_ahead = (3 - today.weekday()) % 7  # 3 = Thursday
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).isoformat()


# ======================================================
# TAB 3 — POSITIONS & NET BOOK
# ======================================================
def render():
    section_header("Positions & Net Book", "Open positions, P&L, risk metrics, trade history")

    oc_t3  = st.session_state.get("current_option_chain", pd.DataFrame())
    sel_t3 = st.session_state.get("current_selected_index", "")
    positions, total_open_pnl, total_net_qty = calculate_net_book(
        st.session_state.today_trades, oc_t3)

    # FIX: use IST-aware time consistently instead of datetime.now()
    today_str_t3  = now_ist_dt.strftime("%Y-%m-%d")
    today_closed3 = [t for t in st.session_state.closed_positions if t.get("Date") == today_str_t3]
    closed_pnl3   = sum(t.get("final_pnl", 0) for t in today_closed3)
    grand_total3  = total_open_pnl + closed_pnl3

    # ── Portfolio Greeks ──────────────────────────────────
    exp_t3  = st.session_state.get("oc_expiry_select", "")
    # FIX: replace hardcoded past date with dynamic nearest weekly expiry
    exp_t3  = exp_t3 if exp_t3 else _nearest_weekly_expiry()
    exp_dt3 = pd.to_datetime(exp_t3)
    dte3    = max(1, (exp_dt3 - datetime.now()).days)
    spot3   = st.session_state.get("current_spot_price", 0) or 0
    port_greeks = calculate_portfolio_greeks(
        st.session_state.today_trades, spot3, sel_t3, dte3) if spot3 else {}

    # ── Top metrics ───────────────────────────────────────
    pnl_c3 = "#00e676" if grand_total3 >= 0 else "#ff3d57"
    metrics_row(
        metric_card("GRAND TOTAL", f"₹{grand_total3:,.0f}", "", pnl_c3) +
        metric_card("OPEN P&L",   f"₹{total_open_pnl:,.0f}", "", "#00d4ff") +
        metric_card("CLOSED P&L", f"₹{closed_pnl3:,.0f}", "", "#c084fc") +
        metric_card("NET QTY",    f"{int(total_net_qty)}", "", "#ffd600") +
        metric_card("OPEN POS",   f"{len([p for p in positions.values() if p['net_qty'] != 0])}", "", "#7fa8c8")
    )

    # FIX: show explicit state for Greeks instead of silently hiding
    if not spot3:
        st.caption("⏳ Portfolio Greeks unavailable — spot price not loaded yet")
    elif port_greeks:
        metrics_row("".join(metric_card(k, str(v), "", "#00d4ff") for k, v in port_greeks.items()))
    else:
        st.caption("ℹ️ No open positions to calculate Greeks")

    # ── Daily loss utilization (drawdown from peak) ───────
    # FIX: track drawdown from intraday peak, not just current sign
    peak3 = st.session_state.get("daily_peak_pnl", grand_total3)
    if grand_total3 > peak3:
        peak3 = grand_total3
    st.session_state["daily_peak_pnl"] = peak3
    drawdown3 = max(0.0, peak3 - grand_total3)

    limit3   = st.session_state.daily_loss_limit
    pct_used = min(drawdown3 / max(abs(limit3), 1) * 100, 100)
    bar_c3   = "#ff3d57" if pct_used > 70 else "#ffd600" if pct_used > 40 else "#00e676"
    st.markdown(_loss_bar_html(pct_used, limit3, bar_c3), unsafe_allow_html=True)

    # ── Trade Statistics (Today) ──────────────────────────
    # FIX: use today_closed3 so stats reflect the current session,
    #      not all-time history. Label is explicit.
    MIN_TRADES = 5
    all_closed3 = st.session_state.closed_positions  # kept for export

    if len(today_closed3) >= MIN_TRADES:
        wins3   = [t for t in today_closed3 if t.get("final_pnl", 0) > 0]
        losses3 = [t for t in today_closed3 if t.get("final_pnl", 0) < 0]
        wr3     = len(wins3) / len(today_closed3) * 100
        avg_w3  = sum(t["final_pnl"] for t in wins3)   / len(wins3)   if wins3   else 0
        avg_l3  = abs(sum(t["final_pnl"] for t in losses3) / len(losses3)) if losses3 else 0
        exp3    = (wr3 / 100 * avg_w3) - ((1 - wr3 / 100) * avg_l3)
        pf3     = (sum(t["final_pnl"] for t in wins3) /
                   max(abs(sum(t["final_pnl"] for t in losses3)), 0.01))

        section_header("Today's Trade Statistics")
        metrics_row(
            metric_card("WIN RATE",      f"{wr3:.1f}%", f"{len(wins3)}W / {len(losses3)}L", "#00e676") +
            metric_card("AVG WIN",       f"₹{avg_w3:,.0f}", "", "#00e676") +
            metric_card("AVG LOSS",      f"₹{avg_l3:,.0f}", "", "#ff3d57") +
            metric_card("EXPECTANCY",    f"₹{exp3:,.0f}", "", "#ffd600") +
            metric_card("PROFIT FACTOR", f"{pf3:.2f}", "", "#c084fc")
        )
    elif len(today_closed3) >= 2:
        section_header("Today's Trade Statistics")
        st.caption(f"⚠️ Stats based on only {len(today_closed3)} trades — interpret with caution (min {MIN_TRADES} needed for reliability)")
    # else: no closed trades yet — don't show an empty stats block

    st.markdown("---")

    # ── Open Positions ────────────────────────────────────
    section_header("Open Positions")
    open_pos3 = [(k, v) for k, v in positions.items() if v["net_qty"] != 0]

    if not open_pos3:
        st.markdown(
            '<div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;'
            'padding:20px;text-align:center;color:#3a6080;font-family:\'Barlow Condensed\',sans-serif;'
            'letter-spacing:1px;">NO OPEN POSITIONS</div>',
            unsafe_allow_html=True
        )
    else:
        for sym_key, pos in open_pos3:
            side   = "BUY" if pos["net_qty"] > 0 else "SELL"
            entry  = pos["buy_avg"] if side == "BUY" else pos["sell_avg"]
            pnl    = pos["current_pnl"]
            pnl_c  = "#00e676" if pnl >= 0 else "#ff3d57"
            side_c = "#00e676" if side == "BUY" else "#ff3d57"

            # FIX: correct P&L% direction for SELL; guard against zero/float-dust qty
            qty_abs = abs(pos["net_qty"])
            if entry > 0 and qty_abs > 1e-9:
                if side == "BUY":
                    pnl_pct = (pos["ltp"] - entry) / entry * 100
                else:
                    pnl_pct = (entry - pos["ltp"]) / entry * 100
            else:
                pnl_pct = 0.0

            # FIX: log a warning if key format is unexpected
            parts = sym_key.split("|")
            if len(parts) >= 3:
                disp_name = f"{parts[0]} {parts[1]} {parts[2]}"
            else:
                disp_name = sym_key
                st.warning(f"Unexpected position key format: {sym_key}", icon="⚠️")

            c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 1, 1, 1, 1, 1, 1])
            c1.markdown(
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
                f'font-weight:600;color:#e8f4ff;">{disp_name}</span>',
                unsafe_allow_html=True
            )
            c2.markdown(
                f'<span style="font-family:\'Barlow Condensed\',sans-serif;font-size:13px;'
                f'font-weight:700;color:{side_c};letter-spacing:1px;">{side}</span>',
                unsafe_allow_html=True
            )
            c3.markdown(
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
                f'color:#7fa8c8;">{qty_abs:.0f}</span>',
                unsafe_allow_html=True
            )
            c4.markdown(
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
                f'color:#7fa8c8;">₹{entry:.2f}</span>',
                unsafe_allow_html=True
            )
            c5.markdown(
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
                f'color:#e8f4ff;">₹{pos["ltp"]:.2f}</span>',
                unsafe_allow_html=True
            )
            c6.markdown(
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
                f'font-weight:700;color:{pnl_c};">₹{pnl:,.0f}'
                f'<br><small style="font-size:10px;">{pnl_pct:+.1f}%</small></span>',
                unsafe_allow_html=True
            )

            # FIX: two-step confirmation before closing a live position
            confirm_key = f"confirm_close_{sym_key}"
            if c7.button("CLOSE", key=f"close_{sym_key}"):
                st.session_state[confirm_key] = True

            if st.session_state.get(confirm_key):
                st.warning(f"⚠️ Confirm close **{disp_name}**? This cannot be undone.")
                col_y, col_n = st.columns(2)
                if col_y.button("✅ Yes, close", key=f"yes_{sym_key}"):
                    close_position(sym_key, positions)
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
                if col_n.button("❌ Cancel", key=f"no_{sym_key}"):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()

            st.markdown(_pos_bar_html(pnl_c, pnl_pct), unsafe_allow_html=True)

    # ── Closed Today ──────────────────────────────────────
    st.markdown("---")
    section_header("Closed Today")
    if not today_closed3:
        st.markdown(
            '<div style="background:#0d1117;border:1px solid #1e3040;padding:12px;'
            'text-align:center;color:#3a6080;font-family:\'Barlow Condensed\',sans-serif;">'
            'NO CLOSED TRADES TODAY</div>',
            unsafe_allow_html=True
        )
    else:
        for t3 in today_closed3:
            t_pnl3 = t3.get("final_pnl", 0)
            t_c3   = "#00e676" if t_pnl3 >= 0 else "#ff3d57"
            st.markdown(
                _closed_trade_row_html(
                    symbol      = t3.get("symbol", "—"),
                    close_price = t3.get("close_price", 0.0),
                    exit_time   = t3.get("exit_time", ""),
                    pnl         = t_pnl3,
                    pnl_c       = t_c3,
                ),
                unsafe_allow_html=True
            )

    # ── Export ────────────────────────────────────────────
    # FIX: use st.download_button directly — no wrapper button needed.
    #      The two-button pattern causes the download to disappear on rerender.
    st.markdown("---")
    ex1, ex2 = st.columns(2)

    with ex1:
        csv3 = pd.DataFrame(all_closed3).to_csv(index=False) if all_closed3 else ""
        st.download_button(
            label    = "📥 Export Closed Trades",
            data     = csv3,
            file_name= "closed_trades.csv",
            mime     = "text/csv",
            disabled = not all_closed3,
        )

    with ex2:
        csv3b = pd.DataFrame(st.session_state.today_trades).to_csv(index=False) \
                if st.session_state.today_trades else ""
        st.download_button(
            label    = "📋 Export Tradebook",
            data     = csv3b,
            file_name= "tradebook.csv",
            mime     = "text/csv",
            disabled = not st.session_state.today_trades,
        )

    # ── Full Tradebook ────────────────────────────────────
    with st.expander("📋 FULL TRADEBOOK"):
        if st.session_state.today_trades:
            st.dataframe(pd.DataFrame(st.session_state.today_trades), use_container_width=True)
        else:
            st.info("No trades today.")

# ======================================================