import os
import requests as _req
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
from api import fetch_ltp, fetch_available_expiries, fetch_option_chain, fetch_intraday_candles
from analytics import (bs_price, bs_greeks, implied_vol_newton, calculate_gamma_bs,
                       compute_signal_score, generate_ai_trade, check_alerts, call_claude_trade_setup)
from chart_utils import (compute_technicals, compute_order_flow, detect_liquidity_sweeps,
                         detect_order_blocks, detect_fvg, detect_bos_choch, get_order_flow_summary)

# ── Nifty 50 + Nifty Next 50 — Market Scanner Universe ──────────────────────
SCANNER_UNIVERSE = [
    # Nifty 50
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN","HINDUNILVR",
    "ITC","LT","BAJFINANCE","HCLTECH","MARUTI","SUNPHARMA","AXISBANK","KOTAKBANK",
    "TITAN","WIPRO","ONGC","NTPC","POWERGRID","ULTRACEMCO","ASIANPAINT","BAJAJFINSV",
    "TATAMOTORS","TATASTEEL","JSWSTEEL","NESTLEIND","TECHM","HINDALCO","ADANIENT",
    "ADANITOTAL","COALINDIA","BPCL","DIVISLAB","DRREDDY","CIPLA","EICHERMOT",
    "APOLLOHOSP","INDUSINDBK","BRITANNIA","HEROMOTOCO","SHREECEM","HDFCLIFE",
    "SBILIFE","BAJAJ-AUTO","TATACONSUM","M&M","GRASIM","LTIM",
    # Nifty Next 50
    "ADANIPORTS","ADANIGREEN","AMBUJACEM","AUROPHARMA","BANKBARODA","BEL","BERGEPAINT",
    "BOSCHLTD","CANBK","CHOLAFIN","COLPAL","DALBHARAT","DABUR","DLF","GAIL",
    "GODREJCP","HAL","HAVELLS","ICICIPRULI","ICICIGI","INDUSTOWER","IRCTC","JINDALSTEL",
    "LICI","LUPIN","MANKIND","MARICO","MUTHOOTFIN","NAVINFLUOR","OBEROIRLTY",
    "OFSS","PAYTM","PIDILITIND","PGHH","PIIND","RECLTD","SBICARD","SIEMENS",
    "TATAPOWER","TORNTPHARM","TRENT","TVSMOTOR","UBL","UNIONBANK","UPL","VEDL",
    "VOLTAS","WHIRLPOOL","ZOMATO","ZYDUSLIFE",
    # Extra liquid mid-caps
    "ABB","ABCAPITAL","ABFRL","ACC","ALKEM","ATUL","BALKRISIND","BANDHANBNK",
    "BIOCON","CEATLTD","CONCOR","COROMANDEL","CROMPTON","CUB","DELHIVERY",
    "ESCORTS","FEDERALBNK","GLENMARK","GNFC","GODREJPROP","GRANULES","HDFCAMC",
    "HONAUT","IDFCFIRSTB","IIFL","INDHOTEL","IOC","JKCEMENT","JUBLFOOD",
    "KANSAINER","LICHSGFIN","LALPATHLAB","MFSL","MOTHERSON","MPHASIS","MRF",
    "NAUKRI","NBCC","NCC","NHPC","NMDC","PAGEIND","PEL","PERSISTENT",
    "PFC","PHOENIXLTD","POLYCAB","RAYMOND","SAIL","STARHEALTH","SUNTV",
    "SUPREMEIND","TATACHEM","TATACOMM","TORNTPOWER","TRIDENT","TVSL","VBL",
]

@st.cache_data(ttl=60, show_spinner=False)
def _batch_fetch_ltps(instrument_keys_str: str) -> dict:
    """
    Batch fetch LTPs for all scanner stocks in ONE API call.
    Returns dict: {instrument_key → last_price}
    """
    try:
        r = _req.get(
            "https://api.upstox.com/v3/market-quote/ltp",
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            params={"instrument_key": instrument_keys_str},
            timeout=10,
        )
        data = r.json().get("data", {})
        result = {}
        for raw_key, v in data.items():
            clean = raw_key.replace("%7C", "|").replace("%7c", "|")
            price = v.get("last_price", 0)
            if price > 0:
                result[clean] = price
        return result
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_prev_close_batch(instrument_keys_list: tuple) -> dict:
    """
    Fetch previous day closing price for all scanner stocks.
    Returns dict: {instrument_key → prev_close}
    """
    result = {}
    today = date.today()
    for days_back in range(1, 6):
        d = today - timedelta(days=days_back)
        if d.weekday() < 5:
            from_date = to_date = d.strftime("%Y-%m-%d")
            break

    for ikey in instrument_keys_list:
        try:
            key_enc = ikey.replace("|", "%7C")
            url = f"https://api.upstox.com/v2/historical-candle/{key_enc}/day/{to_date}/{from_date}"
            resp = _req.get(url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}, timeout=5)
            candles = resp.json().get("data", {}).get("candles", [])
            if candles:
                result[ikey] = float(candles[0][4])
        except Exception:
            pass
    return result


def _run_market_scanner():
    """
    Scan ALL stocks in SCANNER_UNIVERSE, fetch live LTP + prev close,
    return sorted DataFrame of movers.
    """
    # Build instrument key map from instrument_df
    if instrument_df.empty:
        return pd.DataFrame()

    idf = instrument_df[instrument_df["Symbol"].isin(SCANNER_UNIVERSE)].copy()
    if idf.empty:
        return pd.DataFrame()

    # Batch fetch LTPs (all in one call)
    all_keys    = idf["instrument_key"].tolist()
    keys_str    = ",".join(all_keys)
    ltp_map     = _batch_fetch_ltps(keys_str)

    # Batch fetch prev close (cached 5 min)
    # Use session state to avoid re-fetching every refresh
    cache_key = "scanner_prev_close"
    if cache_key not in st.session_state or not st.session_state[cache_key]:
        pc_map = _fetch_prev_close_batch(tuple(all_keys))
        st.session_state[cache_key] = pc_map
    else:
        pc_map = st.session_state[cache_key]

    rows = []
    for _, row in idf.iterrows():
        sym   = row["Symbol"]
        ikey  = row["instrument_key"]
        ltp   = ltp_map.get(ikey, 0)
        prev  = pc_map.get(ikey, 0)
        if ltp <= 0:
            continue
        chg_rs  = round(ltp - prev, 2)  if prev > 0 else 0.0
        chg_pct = round((chg_rs / prev) * 100, 2) if prev > 0 else 0.0
        rows.append({
            "Symbol":    sym,
            "LTP":       round(ltp, 2),
            "PrevClose": round(prev, 2) if prev > 0 else None,
            "Chg₹":     chg_rs,
            "Chg%":      chg_pct,
            "Abs%":      abs(chg_pct),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Abs%", ascending=False).reset_index(drop=True)
    return df


# ======================================================
# TAB 8 — STOCKS TERMINAL
# ======================================================
def render():
    st.session_state["active_tab_key"] = "🖥 STOCKS"

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;
                border-bottom:1px solid #1e3040;padding-bottom:8px;margin-bottom:14px;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:800;
                  letter-spacing:3px;color:#e8f4ff;">NSE <span style="color:#ff8c00;">EQUITIES</span> TERMINAL</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3a6080;">
        LIVE · NSE CM · AUTO 5s</div>
    </div>""", unsafe_allow_html=True)

    if instrument_df.empty:
        st.warning(f"⚠️ **NSECMI.csv not found.** Place it in the `optionapp/` folder (same folder as `app.py`) or its parent folder. Looking in: `{os.path.dirname(os.path.abspath(__file__))}`")
        return

    # ── Watchlist management ─────────────────────────────────────────────
    wl_col1, wl_col2, wl_col3 = st.columns([2, 1, 1])
    with wl_col1:
        selected_symbols8 = st.multiselect(
            "ADD SYMBOLS", options=instrument_df["Symbol"].tolist(),
            default=st.session_state.get("watchlist8",
                ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","BHARTIARTL","ITC"])[:8],
            label_visibility="collapsed",
            placeholder="🔍 Search & add symbols...",
            key="sym8_ms"
        )
        st.session_state["watchlist8"] = selected_symbols8
    with wl_col2:
        sort_mode8 = st.selectbox("SORT", ["Name A-Z","LTP ↓","LTP ↑","Chg% ↓","Chg% ↑"],
                                   label_visibility="collapsed", key="sort8")
    with wl_col3:
        view_mode8 = st.selectbox("VIEW", ["Grid Cards","Table","Mini Ticker"],
                                   label_visibility="collapsed", key="view8")

    if not selected_symbols8:
        st.markdown("""<div style="background:#0d1117;border:1px dashed #1e3040;border-radius:3px;
            padding:30px;text-align:center;color:#3a6080;font-family:'Barlow Condensed',sans-serif;
            font-size:16px;letter-spacing:2px;">ADD SYMBOLS ABOVE TO BEGIN</div>""", unsafe_allow_html=True)
        return

    # ── Fetch yesterday's closing price (cached 5 min) ─────────────────
    @st.cache_data(ttl=300)
    def _fetch_prev_close(instrument_key: str) -> float | None:
        """Fetch previous trading day's closing price from Upstox daily candles."""
        try:
            import requests as _req
            from datetime import date, timedelta
            key_enc  = instrument_key.replace("|", "%7C")
            today    = date.today()
            # Go back up to 5 days to skip weekends/holidays
            for days_back in range(1, 6):
                d = today - timedelta(days=days_back)
                if d.weekday() < 5:   # Mon–Fri
                    from_date = to_date = d.strftime("%Y-%m-%d")
                    break
            url  = f"https://api.upstox.com/v2/historical-candle/{key_enc}/day/{to_date}/{from_date}"
            resp = _req.get(url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}, timeout=8)
            candles = resp.json().get("data", {}).get("candles", [])
            if candles:
                # candle format: [timestamp, open, high, low, close, volume, oi]
                return float(candles[0][4])   # close price
        except Exception:
            pass
        return None

    # ── Fetch all LTPs + prev close ──────────────────────────────────────
    sel_df8 = instrument_df[instrument_df["Symbol"].isin(selected_symbols8)]
    rows8   = []
    for _, row8 in sel_df8.iterrows():
        sym8 = row8["Symbol"]
        try:
            ltp_df8 = fetch_ltp([row8["instrument_key"]])
            ltp8    = float(ltp_df8["Spot Price"].iloc[0]) if not ltp_df8.empty else None
        except:
            ltp8 = None

        # Yesterday's close
        prev_close8 = st.session_state.get(f"prev_close_{sym8}")
        if prev_close8 is None:
            prev_close8 = _fetch_prev_close(row8["instrument_key"])
            if prev_close8:
                st.session_state[f"prev_close_{sym8}"] = prev_close8

        # Change from yesterday's close
        chg8    = ((ltp8 - prev_close8) / prev_close8 * 100) if ltp8 and prev_close8 and prev_close8 > 0 else 0.0
        chg_rs8 = (ltp8 - prev_close8) if ltp8 and prev_close8 else 0.0

        # Intraday change (LTP vs last refresh)
        last_ltp8 = st.session_state.get(f"prev_ltp_{sym8}", ltp8 or 0)
        intra8    = ((ltp8 - last_ltp8) / last_ltp8 * 100) if ltp8 and last_ltp8 and last_ltp8 > 0 else 0.0
        if ltp8: st.session_state[f"prev_ltp_{sym8}"] = ltp8

        rows8.append({
            "Symbol":      sym8,
            "LTP":         ltp8,
            "Prev Close":  prev_close8,
            "Chg ₹":       chg_rs8,
            "Chg%":        chg8,
            "Intra%":      intra8,
            "Status":      "▲" if chg8 > 0 else "▼" if chg8 < 0 else "—",
        })

    df8 = pd.DataFrame(rows8)
    # ── Sort ──
    if sort_mode8 == "Name A-Z":   df8 = df8.sort_values("Symbol")
    elif sort_mode8 == "LTP ↓":    df8 = df8.sort_values("LTP", ascending=False)
    elif sort_mode8 == "LTP ↑":    df8 = df8.sort_values("LTP")
    elif sort_mode8 == "Chg% ↓":   df8 = df8.sort_values("Chg%", ascending=False)
    elif sort_mode8 == "Chg% ↑":   df8 = df8.sort_values("Chg%")

    now8 = datetime.now(IST).strftime("%H:%M:%S")

    # ── Grid Cards view ───────────────────────────────────────────────────
    if view_mode8 == "Grid Cards":
        n_cols8 = 4
        rows_groups = [df8.iloc[i:i+n_cols8] for i in range(0, len(df8), n_cols8)]
        for grp in rows_groups:
            cols_row = st.columns(n_cols8)
            for ci, (_, r) in enumerate(grp.iterrows()):
                with cols_row[ci]:
                    ltp_val  = r["LTP"]
                    chg_val  = r["Chg%"]
                    has_data = ltp_val is not None
                    c_main   = "#00e676" if chg_val > 0 else "#ff3d57" if chg_val < 0 else "#7fa8c8"
                    c_bg     = "rgba(0,230,118,0.06)" if chg_val > 0 else "rgba(255,61,87,0.06)" if chg_val < 0 else "transparent"
                    arrow    = "▲" if chg_val > 0 else "▼" if chg_val < 0 else "—"
                    ltp_str   = f"₹{ltp_val:,.2f}" if has_data else "N/A"
                    chg_str   = f"{arrow} {abs(chg_val):.2f}%" if has_data else "—"
                    chg_rs    = r.get("Chg ₹", 0)
                    chg_rs_str= f"{'+' if chg_rs >= 0 else ''}₹{chg_rs:,.2f}" if has_data else "—"
                    prev_cl   = r.get("Prev Close")
                    prev_str  = f"₹{prev_cl:,.2f}" if prev_cl else "—"
                    intra_val = r.get("Intra%", 0)
                    intra_c   = "#00e676" if intra_val > 0 else "#ff3d57" if intra_val < 0 else "#3a6080"
                    st.markdown(f"""
                    <div style="background:#0d1117;border:1px solid #1e3040;
                                border-top:2px solid {c_main};border-radius:4px;
                                padding:12px 14px;margin-bottom:6px;
                                background:{c_bg};
                                font-family:'JetBrains Mono',monospace;">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:15px;
                                    font-weight:800;letter-spacing:1px;color:#e8f4ff;">{r['Symbol']}</div>
                        <div style="font-size:9px;color:#3a6080;">{now8}</div>
                      </div>
                      <div style="font-size:22px;font-weight:700;color:{c_main};
                                  margin-top:4px;letter-spacing:-0.5px;">{ltp_str}</div>
                      <div style="display:flex;justify-content:space-between;margin-top:3px;">
                        <span style="font-size:12px;color:{c_main};font-weight:600;">{chg_str} &nbsp; {chg_rs_str}</span>
                      </div>
                      <div style="border-top:1px solid #1e3040;margin-top:8px;padding-top:6px;
                                  display:flex;justify-content:space-between;align-items:center;">
                        <div>
                          <div style="font-size:9px;letter-spacing:1px;color:#3a6080;font-family:'Barlow Condensed',sans-serif;">PREV CLOSE</div>
                          <div style="font-size:12px;color:#7fa8c8;font-weight:600;">{prev_str}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-size:9px;letter-spacing:1px;color:#3a6080;font-family:'Barlow Condensed',sans-serif;">INTRADAY</div>
                          <div style="font-size:11px;color:{intra_c};font-weight:600;">
                            {'▲' if intra_val>0 else '▼' if intra_val<0 else '—'} {abs(intra_val):.2f}%</div>
                        </div>
                      </div>
                      <div style="background:#1e3040;height:2px;margin-top:6px;border-radius:1px;">
                        <div style="background:{c_main};width:{'100%' if chg_val > 1 else '60%' if chg_val > 0 else '40%' if chg_val < 0 else '50%'};height:100%;border-radius:1px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

    # ── Table view ────────────────────────────────────────────────────────
    elif view_mode8 == "Table":
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;overflow:hidden;">
          <div style="display:grid;grid-template-columns:130px 110px 110px 90px 100px 80px;
                      padding:8px 14px;border-bottom:1px solid #1e3040;
                      font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:1.5px;color:#3a6080;">
            <span>SYMBOL</span><span>LTP (₹)</span><span>PREV CLOSE</span>
            <span>CHG ₹</span><span>CHG %</span><span>DIR</span>
          </div>""", unsafe_allow_html=True)
        for _, r in df8.iterrows():
            ltp_v  = r["LTP"]; chg_v = r["Chg%"]
            prev_v = r.get("Prev Close"); chg_rs = r.get("Chg ₹", 0)
            has_d  = ltp_v is not None
            col    = "#00e676" if chg_v > 0 else "#ff3d57" if chg_v < 0 else "#7fa8c8"
            bg     = "background:#071008;" if chg_v > 0.5 else "background:#120307;" if chg_v < -0.5 else ""
            prev_str = f"₹{prev_v:,.2f}" if prev_v else "—"
            chg_rs_str = f"{'+' if chg_rs >= 0 else ''}₹{chg_rs:,.2f}" if has_d and prev_v else "—"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:130px 110px 110px 90px 100px 80px;
                        padding:9px 14px;border-bottom:1px solid #0d1117;{bg}
                        font-family:'JetBrains Mono',monospace;font-size:13px;">
              <span style="font-family:'Barlow Condensed',sans-serif;font-size:14px;
                           font-weight:700;color:#e8f4ff;letter-spacing:0.5px;">{r['Symbol']}</span>
              <span style="color:#e8f4ff;font-weight:600;">{'₹{:,.2f}'.format(ltp_v) if has_d else 'N/A'}</span>
              <span style="color:#7fa8c8;">{prev_str}</span>
              <span style="color:{col};font-weight:600;">{chg_rs_str}</span>
              <span style="color:{col};font-weight:600;">{'▲' if chg_v>0 else '▼' if chg_v<0 else '—'} {abs(chg_v):.2f}%</span>
              <span style="font-size:16px;color:{col};">{'▲' if chg_v>0 else '▼' if chg_v<0 else '—'}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Mini Ticker view ──────────────────────────────────────────────────
    else:
        ticker_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;">'
        for _, r in df8.iterrows():
            ltp_v = r["LTP"]; chg_v = r["Chg%"]
            c = "#00e676" if chg_v > 0 else "#ff3d57" if chg_v < 0 else "#7fa8c8"
            ar = "▲" if chg_v > 0 else "▼" if chg_v < 0 else "—"
            ticker_html += f"""
            <div style="background:#0d1117;border:1px solid {c};border-radius:3px;
                        padding:6px 12px;font-family:'JetBrains Mono',monospace;">
              <span style="font-family:'Barlow Condensed',sans-serif;font-size:13px;
                           font-weight:700;color:#e8f4ff;">{r['Symbol']}</span>
              <span style="font-size:13px;color:{c};font-weight:700;margin-left:10px;">
                {'₹{:,.2f}'.format(ltp_v) if ltp_v else 'N/A'}</span>
              <span style="font-size:11px;color:{c};margin-left:6px;">{ar} {abs(chg_v):.2f}%</span>
              <span style="font-size:10px;color:#3a6080;margin-left:6px;">
                {'Prev: ₹{:,.0f}'.format(r['Prev Close']) if r.get('Prev Close') else ''}</span>
            </div>"""
        ticker_html += "</div>"
        st.markdown(ticker_html, unsafe_allow_html=True)

    # ── Terminal footer ───────────────────────────────────────────────────
    advancers8 = sum(1 for _, r in df8.iterrows() if (r["Chg%"] or 0) > 0)
    decliners8 = sum(1 for _, r in df8.iterrows() if (r["Chg%"] or 0) < 0)
    st.markdown(f"""
    <div style="display:flex;gap:16px;margin-top:12px;padding:8px 14px;
                background:#070b0f;border:1px solid #1e3040;border-radius:3px;
                font-family:'JetBrains Mono',monospace;font-size:11px;">
      <span style="color:#3a6080;">WATCHLIST: <b style="color:#e8f4ff;">{len(df8)}</b></span>
      <span style="color:#3a6080;">ADVANCERS: <b style="color:#00e676;">{advancers8}</b></span>
      <span style="color:#3a6080;">DECLINERS: <b style="color:#ff3d57;">{decliners8}</b></span>
      <span style="color:#3a6080;">UNCHANGED: <b style="color:#7fa8c8;">{len(df8)-advancers8-decliners8}</b></span>
      <span style="color:#3a6080;margin-left:auto;">UPDATED: <b style="color:#ff8c00;">{now8}</b></span>
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 📡 MARKET-WIDE SCANNER  — Nifty 50 + Next 50 + Top Midcaps (~150 stocks)
    #     Completely INDEPENDENT of watchlist — no selection needed
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;
                border-bottom:1px solid #1e3040;padding-bottom:8px;margin-bottom:14px;">
      <div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:800;
                    letter-spacing:3px;color:#e8f4ff;">
          📡 MARKET <span style="color:#ff8c00;">SCANNER</span></div>
        <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#7fa8c8;margin-top:2px;">
          Auto-scans Nifty50 + NiftyNext50 + Top Midcaps (~150 stocks) · No selection needed ·
          Ranked by % move from prev close</div>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;text-align:right;">
        REFRESHES EVERY 60s<br>
        <span style="color:#ff8c00;">{now8}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Manual refresh button + prev close reset ──────────────────────────
    sc1, sc2, sc3 = st.columns([1, 1, 4])
    with sc1:
        if st.button("🔄 Refresh Scanner", key="scanner_refresh", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("scanner_prev_close", None)
            st.rerun()
    with sc2:
        if st.button("🗑️ Reset Prev Close", key="scanner_pc_reset", use_container_width=True):
            st.session_state.pop("scanner_prev_close", None)
            st.rerun()
    with sc3:
        st.markdown(f"""<div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
            color:#3a6080;padding:6px 0;letter-spacing:1px;">
            UNIVERSE: {len(SCANNER_UNIVERSE)} stocks &nbsp;|&nbsp;
            Prev close: {'✅ Loaded' if st.session_state.get('scanner_prev_close') else '⏳ Loading on first run'}
            &nbsp;|&nbsp; LTP cache: 60s</div>""", unsafe_allow_html=True)

    # ── Run scanner ───────────────────────────────────────────────────────
    with st.spinner("📡 Scanning market... fetching live prices"):
        scan_df = _run_market_scanner()

    if scan_df.empty:
        st.markdown("""
        <div style="background:#0d1117;border:1px dashed #1e3040;padding:24px;
                    text-align:center;border-radius:4px;color:#3a6080;
                    font-family:'Barlow Condensed',sans-serif;font-size:14px;letter-spacing:1px;">
          ⏳ SCANNER DATA LOADING...<br>
          <span style="font-size:12px;">First load takes ~10s as prev close is fetched.
          If stuck, click Refresh Scanner.</span>
        </div>""", unsafe_allow_html=True)
    else:
        # Separate valid (has prev close) and LTP-only rows
        scan_with_prev = scan_df[scan_df["PrevClose"].notna() & (scan_df["PrevClose"] > 0)].copy()
        scan_ltp_only  = scan_df[scan_df["PrevClose"].isna()  | (scan_df["PrevClose"] == 0)].copy()

        total_scan    = len(scan_with_prev)
        adv_scan      = len(scan_with_prev[scan_with_prev["Chg%"] > 0])
        dec_scan      = len(scan_with_prev[scan_with_prev["Chg%"] < 0])
        unch_scan     = total_scan - adv_scan - dec_scan
        adv_pct_scan  = adv_scan / max(total_scan, 1) * 100
        dec_pct_scan  = dec_scan / max(total_scan, 1) * 100
        unch_pct_scan = 100 - adv_pct_scan - dec_pct_scan
        breadth_sent  = ("🟢 BULLISH" if adv_scan > dec_scan * 1.4
                         else "🔴 BEARISH" if dec_scan > adv_scan * 1.4
                         else "🟡 MIXED")
        breadth_c     = ("#00e676" if "BULL" in breadth_sent
                         else "#ff3d57" if "BEAR" in breadth_sent
                         else "#ffd600")

        # ── Summary metrics ───────────────────────────────────────────────
        best_gainer = scan_with_prev[scan_with_prev["Chg%"] > 0].iloc[0] if adv_scan > 0 else None
        best_loser  = scan_with_prev[scan_with_prev["Chg%"] < 0].sort_values("Chg%").iloc[0] if dec_scan > 0 else None

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;
                border-top:2px solid #ff8c00;border-radius:3px;padding:10px 14px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                            letter-spacing:1.5px;color:#7fa8c8;">STOCKS SCANNED</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                            font-weight:700;color:#ff8c00;">{total_scan}</div>
                <div style="font-size:10px;color:#3a6080;">of {len(SCANNER_UNIVERSE)} universe</div>
                </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;
                border-top:2px solid #00e676;border-radius:3px;padding:10px 14px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                            letter-spacing:1.5px;color:#7fa8c8;">ADVANCERS</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                            font-weight:700;color:#00e676;">{adv_scan}</div>
                <div style="font-size:10px;color:#3a6080;">{adv_pct_scan:.0f}% of scanned</div>
                </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;
                border-top:2px solid #ff3d57;border-radius:3px;padding:10px 14px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                            letter-spacing:1.5px;color:#7fa8c8;">DECLINERS</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                            font-weight:700;color:#ff3d57;">{dec_scan}</div>
                <div style="font-size:10px;color:#3a6080;">{dec_pct_scan:.0f}% of scanned</div>
                </div>""", unsafe_allow_html=True)
        with m4:
            if best_gainer is not None:
                st.markdown(f"""<div style="background:#010e06;border:1px solid #00e676;
                    border-top:2px solid #00e676;border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                                letter-spacing:1.5px;color:#7fa8c8;">TOP GAINER</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:18px;
                                font-weight:800;color:#e8f4ff;">{best_gainer['Symbol']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                                font-weight:700;color:#00e676;">▲ {best_gainer['Chg%']:.2f}%</div>
                    <div style="font-size:10px;color:#3a6080;">LTP ₹{best_gainer['LTP']:,.2f}</div>
                    </div>""", unsafe_allow_html=True)
        with m5:
            if best_loser is not None:
                st.markdown(f"""<div style="background:#120103;border:1px solid #ff3d57;
                    border-top:2px solid #ff3d57;border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;
                                letter-spacing:1.5px;color:#7fa8c8;">TOP LOSER</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:18px;
                                font-weight:800;color:#e8f4ff;">{best_loser['Symbol']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                                font-weight:700;color:#ff3d57;">▼ {abs(best_loser['Chg%']):.2f}%</div>
                    <div style="font-size:10px;color:#3a6080;">LTP ₹{best_loser['LTP']:,.2f}</div>
                    </div>""", unsafe_allow_html=True)

        # ── Market Breadth Bar ────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;
                    padding:10px 16px;margin:10px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      font-family:'Barlow Condensed',sans-serif;font-size:11px;
                      letter-spacing:1px;margin-bottom:5px;">
            <span style="color:#00e676;">▲ ADVANCE {adv_scan} ({adv_pct_scan:.0f}%)</span>
            <span style="color:{breadth_c};font-size:15px;font-weight:700;letter-spacing:2px;">
              BREADTH: {breadth_sent}</span>
            <span style="color:#ff3d57;">▼ DECLINE {dec_scan} ({dec_pct_scan:.0f}%)</span>
          </div>
          <div style="display:flex;height:12px;border-radius:6px;overflow:hidden;gap:2px;">
            <div style="width:{adv_pct_scan:.1f}%;background:#00e676;border-radius:4px 0 0 4px;"></div>
            <div style="width:{unch_pct_scan:.1f}%;background:#1e3040;"></div>
            <div style="width:{dec_pct_scan:.1f}%;background:#ff3d57;border-radius:0 4px 4px 0;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:4px;
                      font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;">
            <span>Unchanged: {unch_scan}</span>
            <span>Total scanned with prev close: {total_scan}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Top 10 Gainers + Top 10 Losers — side by side ─────────────────
        gainers_scan = scan_with_prev[scan_with_prev["Chg%"] > 0].head(10)
        losers_scan  = scan_with_prev[scan_with_prev["Chg%"] < 0].sort_values("Chg%").head(10)
        max_abs_move = scan_with_prev["Abs%"].max() if not scan_with_prev.empty else 1.0

        g_col, l_col = st.columns(2)

        # ── TOP GAINERS ────────────────────────────────────────────────
        with g_col:
            st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:14px;
                font-weight:700;letter-spacing:2px;color:#00e676;margin-bottom:8px;
                border-left:3px solid #00e676;padding-left:10px;">
                🟢 TOP 10 GAINERS — ALL STOCKS</div>""", unsafe_allow_html=True)

            if gainers_scan.empty:
                st.markdown("<div style='color:#3a6080;font-size:12px;padding:8px;'>No gainers found</div>",
                            unsafe_allow_html=True)
            else:
                for rank_g, (_, row_g) in enumerate(gainers_scan.iterrows(), 1):
                    bar_w_g = min(row_g["Abs%"] / max(max_abs_move, 0.01) * 100, 100)
                    st.markdown(f"""
                    <div style="background:#010e06;border:1px solid #1a4025;
                                border-left:3px solid #00e676;border-radius:3px;
                                padding:9px 14px;margin:4px 0;position:relative;overflow:hidden;">
                      <div style="position:absolute;top:0;left:0;width:{bar_w_g:.0f}%;height:100%;
                                  background:rgba(0,230,118,0.06);pointer-events:none;"></div>
                      <div style="position:relative;display:flex;justify-content:space-between;
                                  align-items:center;">
                        <div>
                          <span style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
                                       color:#3a6080;margin-right:8px;">#{rank_g}</span>
                          <span style="font-family:'Barlow Condensed',sans-serif;font-size:17px;
                                       font-weight:800;color:#e8f4ff;">{row_g['Symbol']}</span>
                          <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                      color:#7fa8c8;margin-top:2px;">
                            LTP ₹{row_g['LTP']:,.2f}
                            &nbsp;|&nbsp;Prev ₹{row_g['PrevClose']:,.2f}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-family:'JetBrains Mono',monospace;font-size:20px;
                                      font-weight:700;color:#00e676;">
                            ▲ {row_g['Chg%']:.2f}%</div>
                          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                                      color:#00e676;">+₹{row_g['Chg₹']:,.2f}</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        # ── TOP LOSERS ─────────────────────────────────────────────────
        with l_col:
            st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:14px;
                font-weight:700;letter-spacing:2px;color:#ff3d57;margin-bottom:8px;
                border-left:3px solid #ff3d57;padding-left:10px;">
                🔴 TOP 10 LOSERS — ALL STOCKS</div>""", unsafe_allow_html=True)

            if losers_scan.empty:
                st.markdown("<div style='color:#3a6080;font-size:12px;padding:8px;'>No losers found</div>",
                            unsafe_allow_html=True)
            else:
                for rank_l, (_, row_l) in enumerate(losers_scan.iterrows(), 1):
                    bar_w_l = min(row_l["Abs%"] / max(max_abs_move, 0.01) * 100, 100)
                    st.markdown(f"""
                    <div style="background:#120103;border:1px solid #401520;
                                border-left:3px solid #ff3d57;border-radius:3px;
                                padding:9px 14px;margin:4px 0;position:relative;overflow:hidden;">
                      <div style="position:absolute;top:0;left:0;width:{bar_w_l:.0f}%;height:100%;
                                  background:rgba(255,61,87,0.06);pointer-events:none;"></div>
                      <div style="position:relative;display:flex;justify-content:space-between;
                                  align-items:center;">
                        <div>
                          <span style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
                                       color:#3a6080;margin-right:8px;">#{rank_l}</span>
                          <span style="font-family:'Barlow Condensed',sans-serif;font-size:17px;
                                       font-weight:800;color:#e8f4ff;">{row_l['Symbol']}</span>
                          <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                      color:#7fa8c8;margin-top:2px;">
                            LTP ₹{row_l['LTP']:,.2f}
                            &nbsp;|&nbsp;Prev ₹{row_l['PrevClose']:,.2f}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-family:'JetBrains Mono',monospace;font-size:20px;
                                      font-weight:700;color:#ff3d57;">
                            ▼ {abs(row_l['Chg%']):.2f}%</div>
                          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                                      color:#ff3d57;">-₹{abs(row_l['Chg₹']):,.2f}</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        # ── Full ranked table (ALL scanned stocks) ────────────────────────
        st.markdown("---")
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;
            font-weight:700;letter-spacing:2px;color:#7fa8c8;margin-bottom:6px;
            border-left:3px solid #ff8c00;padding-left:10px;">
            📋 ALL SCANNED STOCKS — FULL RANKED TABLE</div>""", unsafe_allow_html=True)

        if not scan_with_prev.empty:
            # Build compact HTML table
            tbl_html = """
            <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;
                        overflow:hidden;max-height:520px;overflow-y:auto;">
              <div style="display:grid;
                          grid-template-columns:36px 110px 90px 90px 80px 80px 1fr;
                          padding:7px 14px;border-bottom:1px solid #2a3a4a;position:sticky;top:0;
                          background:#111920;font-family:'Barlow Condensed',sans-serif;
                          font-size:10px;letter-spacing:1.5px;color:#3a6080;">
                <span>#</span><span>SYMBOL</span><span>LTP ₹</span>
                <span>PREV ₹</span><span>CHG ₹</span><span>CHG %</span>
                <span>MOVE BAR</span>
              </div>"""

            for ri, (_, r) in enumerate(scan_with_prev.iterrows()):
                cv   = r["Chg%"]
                cr   = r["Chg₹"]
                col_ = "#00e676" if cv > 0 else "#ff3d57" if cv < 0 else "#7fa8c8"
                arr  = "▲" if cv > 0 else "▼" if cv < 0 else "—"
                bg_  = "background:#020d04;" if cv > 0.5 else "background:#0d0101;" if cv < -0.5 else ""
                bw_  = min(r["Abs%"] / max(max_abs_move, 0.01) * 100, 100)
                tbl_html += f"""
              <div style="display:grid;
                          grid-template-columns:36px 110px 90px 90px 80px 80px 1fr;
                          padding:7px 14px;border-bottom:1px solid #0d1117;{bg_}
                          font-family:'JetBrains Mono',monospace;font-size:12px;
                          align-items:center;">
                <span style="color:#3a6080;font-size:10px;">{ri+1}</span>
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:14px;
                             font-weight:800;color:#e8f4ff;">{r['Symbol']}</span>
                <span style="color:#e8f4ff;font-weight:600;">₹{r['LTP']:,.2f}</span>
                <span style="color:#7fa8c8;">₹{r['PrevClose']:,.2f}</span>
                <span style="color:{col_};font-weight:600;">{'+' if cr>=0 else ''}₹{cr:,.2f}</span>
                <span style="color:{col_};font-weight:700;">{arr} {abs(cv):.2f}%</span>
                <div style="background:#1e3040;border-radius:2px;height:5px;overflow:hidden;">
                  <div style="width:{bw_:.0f}%;background:{col_};height:100%;
                               box-shadow:0 0 4px {col_}55;"></div>
                </div>
              </div>"""

            tbl_html += "</div>"
            st.markdown(tbl_html, unsafe_allow_html=True)

        if not scan_ltp_only.empty:
            with st.expander(f"⏳ {len(scan_ltp_only)} stocks — LTP available but prev close missing"):
                st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;
                    font-size:11px;color:#3a6080;letter-spacing:1px;padding:6px;">
                    These stocks have live LTP but prev close failed to load.
                    Click 'Reset Prev Close' and refresh to retry.</div>""",
                    unsafe_allow_html=True)
                ltp_only_html = ""
                for _, rr in scan_ltp_only.iterrows():
                    ltp_only_html += (
                        f'<span style="background:#0d1117;border:1px solid #1e3040;color:#7fa8c8;'
                        f'font-family:Barlow Condensed,sans-serif;font-size:13px;font-weight:700;'
                        f'padding:4px 10px;border-radius:3px;margin:3px;display:inline-block;">'
                        f'{rr["Symbol"]} ₹{rr["LTP"]:,.2f}</span>'
                    )
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;padding:6px;">{ltp_only_html}</div>',
                            unsafe_allow_html=True)

    # ======================================================
    # ======================================================