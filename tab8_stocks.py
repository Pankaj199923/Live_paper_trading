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

    # ── TOP MOVERS (ALL watchlist stocks, ranked by absolute move) ────────
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:18px;font-weight:800;
                letter-spacing:3px;color:#e8f4ff;border-left:3px solid #ff8c00;
                padding:4px 12px;background:linear-gradient(90deg,#111920,transparent);
                margin-bottom:12px;">
      📊 TOP <span style="color:#ff8c00;">MOVERS</span>
      <span style="font-size:11px;color:#3a6080;font-weight:400;letter-spacing:1px;margin-left:8px;">
        ALL WATCHLIST STOCKS — RANKED BY % CHANGE FROM PREV CLOSE</span>
    </div>""", unsafe_allow_html=True)

    # Filter only rows with valid data (LTP and Prev Close available)
    df8_valid = df8[df8["LTP"].notna() & df8["Prev Close"].notna()].copy()
    df8_valid["AbsChg%"] = df8_valid["Chg%"].abs()

    if df8_valid.empty:
        st.markdown("""<div style="color:#3a6080;font-family:'Barlow Condensed',sans-serif;
            font-size:13px;letter-spacing:1px;padding:12px;">
            No price data available — prices load after first refresh.</div>""",
            unsafe_allow_html=True)
    else:
        # Sort ALL stocks by absolute % change (biggest movers first)
        movers_all = df8_valid.sort_values("AbsChg%", ascending=False).reset_index(drop=True)

        # Split into top gainers and top losers
        gainers8 = movers_all[movers_all["Chg%"] > 0].head(5)
        losers8  = movers_all[movers_all["Chg%"] < 0].sort_values("Chg%").head(5)

        # ── Market sentiment bar ─────────────────────────────────────────
        total_valid  = len(df8_valid)
        adv_pct8     = advancers8 / max(total_valid, 1) * 100
        dec_pct8     = decliners8 / max(total_valid, 1) * 100
        unch_pct8    = 100 - adv_pct8 - dec_pct8
        overall_sent = "🟢 BULLISH" if advancers8 > decliners8 * 1.5 else \
                       "🔴 BEARISH" if decliners8 > advancers8 * 1.5 else "🟡 MIXED"
        sent_c       = "#00e676" if "BULL" in overall_sent else "#ff3d57" if "BEAR" in overall_sent else "#ffd600"

        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;
                    padding:10px 16px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;
                      font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:1px;margin-bottom:5px;">
            <span style="color:#00e676;">▲ ADVANCE {advancers8} ({adv_pct8:.0f}%)</span>
            <span style="color:{sent_c};font-size:13px;font-weight:700;">
              BREADTH: {overall_sent}</span>
            <span style="color:#ff3d57;">▼ DECLINE {decliners8} ({dec_pct8:.0f}%)</span>
          </div>
          <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;">
            <div style="width:{adv_pct8:.0f}%;background:#00e676;"></div>
            <div style="width:{unch_pct8:.0f}%;background:#1e3040;"></div>
            <div style="width:{dec_pct8:.0f}%;background:#ff3d57;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Two-column: Gainers | Losers ─────────────────────────────────
        gcol, lcol = st.columns(2)

        with gcol:
            st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;
                font-weight:700;letter-spacing:2px;color:#00e676;margin-bottom:6px;">
                🟢 TOP GAINERS</div>""", unsafe_allow_html=True)
            if gainers8.empty:
                st.markdown("<div style='color:#3a6080;font-size:12px;'>No gainers in watchlist</div>",
                            unsafe_allow_html=True)
            for ri, (_, r) in enumerate(gainers8.iterrows()):
                bar_w = min(abs(r["Chg%"]) / max(movers_all["AbsChg%"].max(), 0.01) * 100, 100)
                chg_rs = r.get("Chg ₹", 0)
                st.markdown(f"""
                <div style="background:#010e06;border:1px solid #1e4025;border-left:3px solid #00e676;
                            border-radius:3px;padding:10px 14px;margin:5px 0;position:relative;
                            overflow:hidden;">
                  <div style="position:absolute;top:0;left:0;width:{bar_w:.0f}%;height:100%;
                              background:rgba(0,230,118,0.05);z-index:0;"></div>
                  <div style="position:relative;z-index:1;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                      <div>
                        <span style="font-family:'Barlow Condensed',sans-serif;font-size:17px;
                                     font-weight:800;color:#e8f4ff;letter-spacing:0.5px;">
                          #{ri+1} &nbsp;{r['Symbol']}</span>
                      </div>
                      <div style="text-align:right;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                                    font-weight:700;color:#00e676;">▲ {abs(r['Chg%']):.2f}%</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                    color:#00e676;">+₹{abs(chg_rs):.2f}</div>
                      </div>
                    </div>
                    <div style="display:flex;gap:16px;margin-top:4px;
                                font-family:'JetBrains Mono',monospace;font-size:11px;color:#7fa8c8;">
                      <span>LTP: <b style="color:#e8f4ff;">₹{r['LTP']:,.2f}</b></span>
                      <span>Prev: ₹{r['Prev Close']:,.2f}</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        with lcol:
            st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;
                font-weight:700;letter-spacing:2px;color:#ff3d57;margin-bottom:6px;">
                🔴 TOP LOSERS</div>""", unsafe_allow_html=True)
            if losers8.empty:
                st.markdown("<div style='color:#3a6080;font-size:12px;'>No losers in watchlist</div>",
                            unsafe_allow_html=True)
            for ri, (_, r) in enumerate(losers8.iterrows()):
                bar_w = min(abs(r["Chg%"]) / max(movers_all["AbsChg%"].max(), 0.01) * 100, 100)
                chg_rs = r.get("Chg ₹", 0)
                st.markdown(f"""
                <div style="background:#120103;border:1px solid #401520;border-left:3px solid #ff3d57;
                            border-radius:3px;padding:10px 14px;margin:5px 0;position:relative;
                            overflow:hidden;">
                  <div style="position:absolute;top:0;left:0;width:{bar_w:.0f}%;height:100%;
                              background:rgba(255,61,87,0.05);z-index:0;"></div>
                  <div style="position:relative;z-index:1;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                      <div>
                        <span style="font-family:'Barlow Condensed',sans-serif;font-size:17px;
                                     font-weight:800;color:#e8f4ff;letter-spacing:0.5px;">
                          #{ri+1} &nbsp;{r['Symbol']}</span>
                      </div>
                      <div style="text-align:right;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                                    font-weight:700;color:#ff3d57;">▼ {abs(r['Chg%']):.2f}%</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                    color:#ff3d57;">-₹{abs(chg_rs):.2f}</div>
                      </div>
                    </div>
                    <div style="display:flex;gap:16px;margin-top:4px;
                                font-family:'JetBrains Mono',monospace;font-size:11px;color:#7fa8c8;">
                      <span>LTP: <b style="color:#e8f4ff;">₹{r['LTP']:,.2f}</b></span>
                      <span>Prev: ₹{r['Prev Close']:,.2f}</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── Full ranked leaderboard (all stocks) ──────────────────────────
        st.markdown("---")
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;
            font-weight:700;letter-spacing:2px;color:#7fa8c8;margin-bottom:6px;">
            📋 ALL STOCKS — RANKED BY MOVE SIZE</div>""", unsafe_allow_html=True)

        rank_html = """
        <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;overflow:hidden;">
          <div style="display:grid;grid-template-columns:40px 130px 100px 80px 100px 90px 100%;
                      padding:7px 14px;border-bottom:1px solid #1e3040;
                      font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:1.5px;color:#3a6080;">
            <span>#</span><span>SYMBOL</span><span>LTP ₹</span><span>PREV ₹</span>
            <span>CHG ₹</span><span>CHG %</span><span>MOVE BAR</span>
          </div>"""

        for ri, (_, r) in enumerate(movers_all.iterrows()):
            chg_v   = r["Chg%"] or 0
            chg_rs  = r.get("Chg ₹", 0) or 0
            col_r   = "#00e676" if chg_v > 0 else "#ff3d57" if chg_v < 0 else "#7fa8c8"
            arrow_r = "▲" if chg_v > 0 else "▼" if chg_v < 0 else "—"
            bg_r    = "background:#071008;" if chg_v > 0 else "background:#120307;" if chg_v < 0 else ""
            bar_fill= min(abs(chg_v) / max(movers_all["AbsChg%"].max(), 0.01) * 100, 100)
            prev_str = f"₹{r['Prev Close']:,.2f}" if r.get("Prev Close") else "—"
            rank_html += f"""
          <div style="display:grid;grid-template-columns:40px 130px 100px 80px 100px 90px 100%;
                      padding:8px 14px;border-bottom:1px solid #0d1117;{bg_r}
                      font-family:'JetBrains Mono',monospace;font-size:12px;align-items:center;">
            <span style="color:#3a6080;font-size:11px;">{ri+1}</span>
            <span style="font-family:'Barlow Condensed',sans-serif;font-size:14px;
                         font-weight:800;color:#e8f4ff;">{r['Symbol']}</span>
            <span style="color:#e8f4ff;font-weight:600;">₹{r['LTP']:,.2f}</span>
            <span style="color:#7fa8c8;">{prev_str}</span>
            <span style="color:{col_r};font-weight:600;">
              {'+' if chg_rs >= 0 else ''}₹{chg_rs:,.2f}</span>
            <span style="color:{col_r};font-weight:700;">
              {arrow_r} {abs(chg_v):.2f}%</span>
            <div style="display:flex;align-items:center;gap:6px;">
              <div style="background:#1e3040;border-radius:2px;height:6px;flex:1;overflow:hidden;">
                <div style="width:{bar_fill:.0f}%;background:{col_r};height:100%;
                             border-radius:2px;box-shadow:0 0 4px {col_r}66;"></div>
              </div>
            </div>
          </div>"""

        rank_html += "</div>"
        st.markdown(rank_html, unsafe_allow_html=True)

    # ======================================================