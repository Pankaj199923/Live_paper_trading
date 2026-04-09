"""
tab_fvg_tradelog.py  —  FVG Auto-Trade Log & Live P&L Tracker
═══════════════════════════════════════════════════════════════════════════════
• Saves every FVG trade to  fvg_trade_log.csv  (persists across restarts)
• Shows live LTP via fetch_ltp()
• Calculates unrealised P&L per position
• Close / Square-off button per trade
• Daily summary: total trades, win/loss, net P&L
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from datetime import datetime, date

from config import BASE_DIR, LOT_SIZES
from utils import (section_header, metric_card, metrics_row,
                   get_lot_size, idx_short)
from api import fetch_ltp

# ── File path ────────────────────────────────────────────────────────────────
FVG_LOG_FILE = os.path.join(BASE_DIR, "fvg_trade_log.csv")

LOG_COLS = [
    "date", "time", "index", "expiry", "action",
    "option_type", "strike", "entry_price", "qty",
    "fvg_type", "fvg_bot", "fvg_top", "fvg_size",
    "trigger_price", "reason",
    "exit_price", "exit_time", "status",   # OPEN / CLOSED / SQUARED
    "pnl", "auto",
]


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _load_log() -> pd.DataFrame:
    if os.path.exists(FVG_LOG_FILE):
        try:
            df = pd.read_csv(FVG_LOG_FILE, dtype=str)
            for c in LOG_COLS:
                if c not in df.columns:
                    df[c] = ""
            return df[LOG_COLS]
        except Exception:
            pass
    return pd.DataFrame(columns=LOG_COLS)


def _save_log(df: pd.DataFrame):
    try:
        df.to_csv(FVG_LOG_FILE, index=False)
    except Exception as e:
        st.error(f"Log save error: {e}")


def append_fvg_trade(
    index_key   : str,
    expiry      : str,
    action      : str,          # "SELL_PE" | "SELL_CE"
    strike      : int,
    entry_price : float,
    fvg_type    : str,          # "BEAR" | "BULL" | "MANUAL"
    fvg_bot     : float,
    fvg_top     : float,
    fvg_size    : float,
    trigger_price: float,
    reason      : str,
    auto        : bool,
    qty         : int | None = None,
):
    """
    Public function — call this from tab9_chart.py place_order_fn callback
    to persist every FVG trade to the CSV log.

    Example in tab9_chart.py:
        from tab_fvg_tradelog import append_fvg_trade

        def _place_order(action, strike, index_key):
            ltp = fetch_ltp(...)   # get current option LTP
            append_fvg_trade(
                index_key=index_key, expiry=_cur_expiry,
                action=action, strike=strike, entry_price=ltp,
                fvg_type=sig["fvg_type"], fvg_bot=sig["bot"],
                fvg_top=sig["top"], fvg_size=sig["size"],
                trigger_price=sig["trigger"], reason=sig["reason"],
                auto=True,
            )
            # then call your broker API...
    """
    df = _load_log()
    opt_type = "PE" if action == "SELL_PE" else "CE"
    lot = qty if qty is not None else get_lot_size(index_key)
    now = datetime.now()
    new_row = {
        "date"          : now.strftime("%Y-%m-%d"),
        "time"          : now.strftime("%H:%M:%S"),
        "index"         : index_key,
        "expiry"        : expiry,
        "action"        : action,
        "option_type"   : opt_type,
        "strike"        : str(strike),
        "entry_price"   : str(round(entry_price, 2)),
        "qty"           : str(lot),
        "fvg_type"      : fvg_type,
        "fvg_bot"       : str(round(fvg_bot, 1)),
        "fvg_top"       : str(round(fvg_top, 1)),
        "fvg_size"      : str(round(fvg_size, 1)),
        "trigger_price" : str(round(trigger_price, 2)),
        "reason"        : reason,
        "exit_price"    : "",
        "exit_time"     : "",
        "status"        : "OPEN",
        "pnl"           : "",
        "auto"          : "YES" if auto else "NO",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_log(df)


def close_fvg_trade(row_idx: int, exit_price: float):
    """Mark a trade as CLOSED and calculate final P&L."""
    df = _load_log()
    if row_idx >= len(df):
        return
    row = df.iloc[row_idx]
    try:
        entry = float(row["entry_price"])
        qty   = float(row["qty"])
        # SELL → profit when price falls
        pnl   = round((entry - exit_price) * qty, 2)
        df.at[row_idx, "exit_price"] = str(round(exit_price, 2))
        df.at[row_idx, "exit_time"]  = datetime.now().strftime("%H:%M:%S")
        df.at[row_idx, "status"]     = "CLOSED"
        df.at[row_idx, "pnl"]        = str(pnl)
        _save_log(df)
    except Exception as e:
        st.error(f"Close trade error: {e}")


def _get_live_ltp(index_key: str, expiry: str,
                  strike: str, opt_type: str) -> float | None:
    """Fetch live LTP for an open option position."""
    try:
        symbol = f"{index_key}{expiry}{strike}{opt_type}"
        ltp = fetch_ltp(symbol)
        return float(ltp) if ltp else None
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═════════════════════════════════════════════════════════════════════════════

def render():
    st.session_state["active_tab_key"] = "📋 FVG LOG"
    section_header(
        "FVG Trade Log  ·  Live P&L Tracker",
        "All trades triggered by Fair Value Gap auto-trade engine",
    )

    df = _load_log()

    # ── Top controls ──────────────────────────────────────────────────────
    _hc1, _hc2, _hc3, _hc4, _hc5 = st.columns([2, 2, 2, 2, 2])
    with _hc1:
        today_str = date.today().strftime("%Y-%m-%d")
        date_filter = st.selectbox(
            "Filter", ["Today", "This Week", "All Time"],
            key="fvg_log_date_filter",
        )
    with _hc2:
        status_filter = st.selectbox(
            "Status", ["All", "OPEN", "CLOSED"],
            key="fvg_log_status_filter",
        )
    with _hc3:
        type_filter = st.selectbox(
            "Option", ["All", "PE", "CE"],
            key="fvg_log_type_filter",
        )
    with _hc4:
        refresh = st.button("🔄 Refresh LTP", key="fvg_log_refresh",
                            use_container_width=True)
    with _hc5:
        if st.button("🗑️ Clear All Closed", key="fvg_log_clear_closed",
                     use_container_width=True):
            df = df[df["status"] != "CLOSED"].reset_index(drop=True)
            _save_log(df)
            st.success("Closed trades cleared")
            st.rerun()

    # ── Apply filters ─────────────────────────────────────────────────────
    dff = df.copy()
    if date_filter == "Today":
        dff = dff[dff["date"] == today_str]
    elif date_filter == "This Week":
        week_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=pd.Timestamp.now().dayofweek)
        dff = dff[pd.to_datetime(dff["date"], errors="coerce") >= week_start]
    if status_filter != "All":
        dff = dff[dff["status"] == status_filter]
    if type_filter != "All":
        dff = dff[dff["option_type"] == type_filter]

    dff = dff.reset_index(drop=True)

    # ── Summary cards ─────────────────────────────────────────────────────
    _total   = len(dff)
    _open    = len(dff[dff["status"] == "OPEN"])
    _closed  = len(dff[dff["status"] == "CLOSED"])
    _pnl_vals = pd.to_numeric(dff["pnl"], errors="coerce").dropna()
    _net_pnl  = _pnl_vals.sum()
    _winners  = (_pnl_vals > 0).sum()
    _losers   = (_pnl_vals < 0).sum()
    _win_rate = round(_winners / len(_pnl_vals) * 100, 1) if len(_pnl_vals) > 0 else 0
    _pnl_clr  = "#00e676" if _net_pnl >= 0 else "#ff3d57"

    metrics_row(
        metric_card("TOTAL TRADES", str(_total),   f"{date_filter}", "#7fa8c8") +
        metric_card("OPEN",         str(_open),    "Live positions",  "#ff8c00") +
        metric_card("CLOSED",       str(_closed),  "Exited",          "#3a6080") +
        metric_card("NET P&L",      f"₹{_net_pnl:,.0f}",
                    f"W:{_winners}  L:{_losers}", _pnl_clr) +
        metric_card("WIN RATE",     f"{_win_rate}%",
                    f"Closed trades only", "#c084fc") +
        metric_card("AVG WIN",
                    f"₹{_pnl_vals[_pnl_vals>0].mean():,.0f}" if _winners else "—",
                    "", "#00e676") +
        metric_card("AVG LOSS",
                    f"₹{_pnl_vals[_pnl_vals<0].mean():,.0f}" if _losers else "—",
                    "", "#ff3d57")
    )

    # P&L bar (net)
    if _net_pnl != 0:
        _bar_w  = min(abs(_net_pnl) / (abs(_net_pnl) + 1000) * 100, 95)
        _bar_c  = "#00e676" if _net_pnl >= 0 else "#ff3d57"
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;
                    padding:8px 14px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;
                      font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:1px;color:#3a6080;">
            <span>FVG NET P&L  ({date_filter})</span>
            <span style="color:{_bar_c};font-size:13px;font-weight:700;">
              ₹{_net_pnl:,.0f}</span>
          </div>
          <div style="background:#1e3040;border-radius:2px;height:6px;overflow:hidden;">
            <div style="background:{_bar_c};width:{_bar_w}%;height:100%;
                        border-radius:2px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Trade cards ───────────────────────────────────────────────────────
    if dff.empty:
        st.markdown("""
        <div style="background:#0d1117;border:1px dashed #1e3040;border-radius:4px;
                    padding:30px;text-align:center;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
                      letter-spacing:2px;color:#3a6080;">NO FVG TRADES LOGGED</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                      color:#1e3040;margin-top:6px;">
            Trades appear here when the FVG Auto-Trade Engine fires in Tab 9</div>
        </div>""", unsafe_allow_html=True)
        return

    # Reverse: newest first
    dff = dff.iloc[::-1].reset_index()  # keep original index in "index" col for close_trade

    for _, row in dff.iterrows():
        orig_idx = row["index"]   # original CSV row index for close operation
        _ot      = row["option_type"]
        _act     = row["action"]
        _status  = row["status"]
        _strike  = row["strike"]
        _entry   = row["entry_price"]
        _qty     = row["qty"]
        _idx     = row["index_x"] if "index_x" in row else row.get("index", "")

        # Try to get _idx properly
        try:
            _idx = str(df.iloc[int(orig_idx)]["index"])
        except Exception:
            _idx = str(row.get("index", ""))

        # Color scheme
        _opt_clr = "#ffd600" if _ot == "PE" else "#c084fc"
        _stat_clr = {"OPEN": "#ff8c00", "CLOSED": "#3a6080", "SQUARED": "#3a6080"}.get(_status, "#3a6080")

        # Live LTP for open trades
        _ltp   = None
        _upnl  = None
        _upnl_clr = "#7fa8c8"
        if _status == "OPEN" and refresh:
            _ltp = _get_live_ltp(
                index_key=str(df.iloc[int(orig_idx)]["index"]),
                expiry=row["expiry"],
                strike=_strike,
                opt_type=_ot,
            )
        if _ltp is not None:
            try:
                _upnl = round((float(_entry) - _ltp) * float(_qty), 2)
                _upnl_clr = "#00e676" if _upnl >= 0 else "#ff3d57"
            except Exception:
                pass

        # Closed P&L
        _closed_pnl = ""
        _cpnl_clr   = "#3a6080"
        if _status == "CLOSED" and row["pnl"]:
            try:
                _cv = float(row["pnl"])
                _closed_pnl = f"₹{_cv:,.0f}"
                _cpnl_clr   = "#00e676" if _cv >= 0 else "#ff3d57"
            except Exception:
                pass

        # FVG info
        _fvg_label = ""
        if row["fvg_type"] == "BEAR":
            _fvg_label = f"Bearish FVG {row['fvg_bot']}–{row['fvg_top']} ({row['fvg_size']} pts)"
        elif row["fvg_type"] == "BULL":
            _fvg_label = f"Bullish FVG {row['fvg_bot']}–{row['fvg_top']} ({row['fvg_size']} pts)"
        else:
            _fvg_label = "Manual trade"

        _auto_tag = "🤖 AUTO" if row["auto"] == "YES" else "👆 MANUAL"
        _border_clr = _opt_clr if _status == "OPEN" else "#1e3040"

        # ── Card HTML ────────────────────────────────────────────────
        _ltp_str  = f"LTP: ₹{_ltp:,.2f}" if _ltp is not None else ("LTP: —" if _status == "OPEN" else "")
        _upnl_str = f"Unrealised: ₹{_upnl:,.0f}" if _upnl is not None else ""

        st.markdown(f"""
        <div style="background:#0a0e14;border:1px solid {_border_clr};
                    border-left:4px solid {_opt_clr};border-radius:4px;
                    padding:12px 16px;margin:6px 0;">

          <!-- Row 1: action + status + time -->
          <div style="display:flex;justify-content:space-between;align-items:center;
                      flex-wrap:wrap;gap:6px;margin-bottom:6px;">
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="font-family:'Barlow Condensed',sans-serif;font-size:18px;
                           font-weight:700;color:{_opt_clr};letter-spacing:2px;">
                {_act.replace('_',' ')}  {_strike}</span>
              <span style="background:{_opt_clr}22;border:1px solid {_opt_clr};
                           border-radius:3px;padding:1px 8px;
                           font-family:'Barlow Condensed',sans-serif;font-size:11px;
                           font-weight:700;color:{_opt_clr};">{_ot}</span>
              <span style="background:{_stat_clr}22;border:1px solid {_stat_clr};
                           border-radius:3px;padding:1px 8px;
                           font-family:'Barlow Condensed',sans-serif;font-size:11px;
                           font-weight:700;color:{_stat_clr};">{_status}</span>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;">
              {row['date']}  {row['time']}  ·  {_auto_tag}
            </div>
          </div>

          <!-- Row 2: price / P&L -->
          <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:6px;">
            <div>
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                          letter-spacing:1.5px;color:#3a6080;">ENTRY</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:15px;
                          font-weight:700;color:#e8f4ff;">₹{float(_entry):,.2f}</div>
            </div>
            <div>
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                          letter-spacing:1.5px;color:#3a6080;">QTY (LOTS)</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:15px;
                          font-weight:700;color:#e8f4ff;">{_qty}</div>
            </div>
            {"<div><div style='font-family:Barlow Condensed,sans-serif;font-size:9px;letter-spacing:1.5px;color:#3a6080;'>LIVE LTP</div><div style='font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:#e8f4ff;'>" + _ltp_str + "</div></div>" if _ltp_str else ""}
            {"<div><div style='font-family:Barlow Condensed,sans-serif;font-size:9px;letter-spacing:1.5px;color:#3a6080;'>UNREALISED P&L</div><div style='font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:" + _upnl_clr + ";'>" + _upnl_str + "</div></div>" if _upnl_str else ""}
            {"<div><div style='font-family:Barlow Condensed,sans-serif;font-size:9px;letter-spacing:1.5px;color:#3a6080;'>EXIT / P&L</div><div style='font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:" + _cpnl_clr + ";'>" + _closed_pnl + " @ ₹" + str(row['exit_price']) + "</div></div>" if _closed_pnl else ""}
          </div>

          <!-- Row 3: FVG + reason -->
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                      color:#7fa8c8;margin-bottom:4px;">{row.get('reason','')}</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      color:#3a6080;">{_fvg_label}  ·  {row.get('index','')}  {row.get('expiry','')}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Close trade UI (only for OPEN trades) ────────────────────
        if _status == "OPEN":
            _cl1, _cl2, _cl3 = st.columns([2, 2, 6])
            with _cl1:
                _exit_px = st.number_input(
                    "Exit price",
                    min_value=0.05, max_value=99999.0,
                    value=float(_entry), step=0.5,
                    key=f"fvg_exit_px_{orig_idx}",
                    label_visibility="collapsed",
                )
            with _cl2:
                if st.button(
                    f"✅ Close Trade",
                    key=f"fvg_close_btn_{orig_idx}",
                    use_container_width=True,
                ):
                    close_fvg_trade(int(orig_idx), _exit_px)
                    _pnl_val = round((float(_entry) - _exit_px) * float(_qty), 2)
                    _icon = "🟢" if _pnl_val >= 0 else "🔴"
                    st.success(f"{_icon} Trade closed  |  P&L: ₹{_pnl_val:,.0f}")
                    st.rerun()

    # ── Daily P&L chart ───────────────────────────────────────────────────
    st.markdown("---")
    section_header("📊 Daily P&L  (Closed Trades)", "FVG strategy cumulative performance")

    _closed_df = df[df["status"] == "CLOSED"].copy()
    if not _closed_df.empty:
        _closed_df["pnl_num"] = pd.to_numeric(_closed_df["pnl"], errors="coerce").fillna(0)
        _closed_df["date"]    = pd.to_datetime(_closed_df["date"], errors="coerce")
        _daily = _closed_df.groupby("date")["pnl_num"].sum().reset_index()
        _daily["cum_pnl"] = _daily["pnl_num"].cumsum()

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.6, 0.4], vertical_spacing=0.05,
                             subplot_titles=["Cumulative P&L", "Daily P&L"])

        _cum_clr = "#00e676" if _daily["cum_pnl"].iloc[-1] >= 0 else "#ff3d57"
        _fig.add_trace(go.Scatter(
            x=_daily["date"], y=_daily["cum_pnl"],
            mode="lines+markers", name="Cum P&L",
            line=dict(color=_cum_clr, width=2.5),
            fill="tozeroy",
            fillcolor=f"{'rgba(0,230,118,0.10)' if _cum_clr=='#00e676' else 'rgba(255,61,87,0.10)'}",
            marker=dict(size=6, color=_cum_clr),
        ), row=1, col=1)

        _day_colors = ["#00e676" if v >= 0 else "#ff3d57" for v in _daily["pnl_num"]]
        _fig.add_trace(go.Bar(
            x=_daily["date"], y=_daily["pnl_num"],
            marker_color=_day_colors, name="Daily P&L", opacity=0.8,
        ), row=2, col=1)

        _fig.add_hline(y=0, line_color="#3a6080", line_dash="dot",
                       line_width=0.8, row=1, col=1)
        _fig.add_hline(y=0, line_color="#3a6080", line_dash="dot",
                       line_width=0.8, row=2, col=1)

        _fig.update_layout(
            height=380, paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
            font=dict(family="JetBrains Mono", color="#7fa8c8", size=10),
            margin=dict(l=10, r=20, t=40, b=10),
            showlegend=False,
        )
        _fig.update_yaxes(gridcolor="#1a2a3a", tickprefix="₹")
        _fig.update_xaxes(gridcolor="#1a2a3a")
        st.plotly_chart(_fig, use_container_width=True)

    else:
        st.markdown(
            "<div style='color:#3a6080;font-size:12px;font-family:JetBrains Mono,monospace;"
            "padding:20px;text-align:center;'>No closed trades yet — P&L chart will appear here</div>",
            unsafe_allow_html=True,
        )

    # ── Raw table (collapsible) ───────────────────────────────────────────
    with st.expander("📄 Raw Trade Table (CSV view)"):
        st.dataframe(
            dff.drop(columns=["index"], errors="ignore"),
            use_container_width=True,
            height=300,
        )
        csv_data = dff.drop(columns=["index"], errors="ignore").to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name=f"fvg_trades_{date.today()}.csv",
            mime="text/csv",
            key="fvg_dl_csv",
        )
