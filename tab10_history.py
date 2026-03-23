import streamlit as st
import pandas as pd
import numpy as np
import os
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
# TAB 10 — OPTION CHAIN HISTORY
# ======================================================
def render():
    section_header("📸 Option Chain History", "One CSV per day — browse any minute's full chain, signals & S&R")

    # ── Index selector ─────────────────────────────────────────────────
    h9_idx_options = list(INDEX_SHORT.keys())
    h9_idx_sel = st.selectbox(
        "Index",
        h9_idx_options,
        format_func=lambda x: INDEX_SHORT.get(x, x),
        key="h9_idx_sel"
    )

    daily_files = list_daily_files(h9_idx_sel)

    if not daily_files:
        st.markdown(
            '''<div style="background:#0d1117;border:1px dashed #1e3040;padding:36px;
            text-align:center;border-radius:4px;color:#3a6080;
            font-family:Barlow Condensed,sans-serif;font-size:14px;letter-spacing:1px;">
            📸 NO HISTORY YET<br>
            <span style="font-size:12px;color:#1e3040;">
            One CSV per day is created automatically.<br>
            Open Tab 1 during market hours — data saves every minute.</span></div>''',
            unsafe_allow_html=True
        )
        return

    # ── Date selector ──────────────────────────────────────────────────
    date_labels = [d[0] for d in daily_files]
    date_paths  = [d[1] for d in daily_files]

    h9a, h9b, h9c = st.columns([1, 2, 1])
    with h9a:
        sel_date_label = st.selectbox(
            f"📅 Date  ({len(daily_files)} day{'s' if len(daily_files)>1 else ''})",
            date_labels, key="h9_date"
        )
    sel_date_path = date_paths[date_labels.index(sel_date_label)]

    # ── Minute selector ────────────────────────────────────────────────
    avail_times = list_minute_times(sel_date_path, h9_idx_sel)
    if not avail_times:
        st.warning("⚠️ No minute data found in this file."); return

    with h9b:
        sel_time = st.select_slider(
            f"⏱ Time  ({len(avail_times)} snapshots)",
            options=avail_times,
            value=avail_times[-1],
            key="h9_time"
        )

    # ── Load that minute's data ────────────────────────────────────────
    h_oc, h_spot, h_index, h_expiry, h_ts = load_snapshot(sel_date_path, sel_time)

    if h_oc.empty:
        st.error("❌ No data for this minute."); return

    h_index_name = INDEX_SHORT.get(h_index, h_index.split("|")[-1] if h_index else "?")
    atm_h        = get_atm_strike(h_spot, h_index)
    today_date   = sel_date_label

    with h9c:
        st.markdown(f"""
        <div style="background:#111920;border:1px solid #1e3040;border-top:2px solid #ff8c00;
                    border-radius:3px;padding:10px 14px;margin-top:22px;">
          <div style="font-family:Barlow Condensed,sans-serif;font-size:9px;
                      letter-spacing:1.5px;color:#7fa8c8;">VIEWING</div>
          <div style="font-family:JetBrains Mono,monospace;font-size:18px;
                      font-weight:700;color:#ff8c00;">{sel_time}</div>
          <div style="font-family:JetBrains Mono,monospace;font-size:11px;
                      color:#7fa8c8;">{today_date} · {h_index_name}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Snapshot Metrics ───────────────────────────────────────────────
    h_total_ce_oi  = h_oc["CE_OI"].sum()     if "CE_OI"     in h_oc.columns else 0
    h_total_pe_oi  = h_oc["PE_OI"].sum()     if "PE_OI"     in h_oc.columns else 0
    h_total_ce_vol = h_oc["CE_Volume"].sum()  if "CE_Volume" in h_oc.columns else 0
    h_total_pe_vol = h_oc["PE_Volume"].sum()  if "PE_Volume" in h_oc.columns else 0
    h_pcr          = h_total_pe_oi / h_total_ce_oi if h_total_ce_oi else 0
    h_atm_row      = h_oc[h_oc["Strike"] == atm_h]
    h_ce_iv = float(h_atm_row["CE_IV"].iloc[0]) if (not h_atm_row.empty and "CE_IV" in h_atm_row.columns) else 0
    h_pe_iv = float(h_atm_row["PE_IV"].iloc[0]) if (not h_atm_row.empty and "PE_IV" in h_atm_row.columns) else 0

    section_header("Snapshot Metrics")
    metrics_row(
        metric_card("SPOT",         f"₹{h_spot:,.0f}",           "", "#ff8c00") +
        metric_card("ATM",          f"{atm_h:,.0f}",              "", "#00d4ff") +
        metric_card("PCR",          f"{h_pcr:.3f}",   "PE/CE OI",    "#c084fc") +
        metric_card("CE IV (ATM)",  f"{h_ce_iv:.1f}%",            "", "#ff3d57") +
        metric_card("PE IV (ATM)",  f"{h_pe_iv:.1f}%",            "", "#00e676") +
        metric_card("CE VOLUME",    f"{h_total_ce_vol/1e5:.1f}L", "", "#ffd600") +
        metric_card("PE VOLUME",    f"{h_total_pe_vol/1e5:.1f}L", "", "#7fa8c8")
    )

    # ── 10-Factor Signal ───────────────────────────────────────────────
    st.markdown("---")
    section_header(f"10-Factor Signal  —  {h_index_name} @ {sel_time}")
    try:
        h_bull, h_bear, _, _, h_res, h_sup, h_factors = compute_signal_score(h_oc, h_spot, h_index)

        hs1, hs2, hs3, hs4 = st.columns(4)
        with hs1:
            bar_hb = score_bar(h_bull, 20, "#00e676")
            st.markdown(f'''<div style="background:#0d1117;border:1px solid #1e3040;padding:12px;border-radius:3px;">
              <div style="font-family:Barlow Condensed,sans-serif;font-size:11px;letter-spacing:1px;color:#7fa8c8;">BULLISH SCORE</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:28px;font-weight:700;color:#00e676;">{h_bull}<span style="font-size:14px;color:#3a6080;">/20</span></div>
              {bar_hb}</div>''', unsafe_allow_html=True)
        with hs2:
            bar_hr = score_bar(h_bear, 20, "#ff3d57")
            st.markdown(f'''<div style="background:#0d1117;border:1px solid #1e3040;padding:12px;border-radius:3px;">
              <div style="font-family:Barlow Condensed,sans-serif;font-size:11px;letter-spacing:1px;color:#7fa8c8;">BEARISH SCORE</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:28px;font-weight:700;color:#ff3d57;">{h_bear}<span style="font-size:14px;color:#3a6080;">/20</span></div>
              {bar_hr}</div>''', unsafe_allow_html=True)
        with hs3:
            h_res_str = f"{h_res:,.0f}" if h_res else "N/A"
            h_sup_str = f"{h_sup:,.0f}" if h_sup else "N/A"
            st.markdown(f'''<div style="background:#0d1117;border:1px solid #1e3040;padding:12px;border-radius:3px;">
              <div style="font-family:Barlow Condensed,sans-serif;font-size:11px;letter-spacing:1px;color:#7fa8c8;">RESISTANCE</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:20px;font-weight:700;color:#ff3d57;">{h_res_str}</div>
              <div style="font-family:Barlow Condensed,sans-serif;font-size:10px;letter-spacing:1px;color:#7fa8c8;margin-top:6px;">SUPPORT</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:20px;font-weight:700;color:#00e676;">{h_sup_str}</div>
              </div>''', unsafe_allow_html=True)
        with hs4:
            h_net = h_bull - h_bear
            h_bias_color = "#00e676" if h_net > 0 else "#ff3d57" if h_net < 0 else "#ffd600"
            h_bias_label = "BULLISH" if h_net > 3 else "BEARISH" if h_net < -3 else "NEUTRAL"
            st.markdown(f'''<div style="background:#0d1117;border:1px solid {h_bias_color};border-top:2px solid {h_bias_color};padding:12px;border-radius:3px;">
              <div style="font-family:Barlow Condensed,sans-serif;font-size:11px;letter-spacing:1px;color:#7fa8c8;">MARKET BIAS</div>
              <div style="font-family:Barlow Condensed,sans-serif;font-size:22px;font-weight:800;letter-spacing:3px;color:{h_bias_color};">{h_bias_label}</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:16px;color:{h_bias_color};">Net: {h_net:+d}/20</div>
              </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fac_cols9 = st.columns(len(h_factors))
        for col_f9, (fname9, (fval9, fb9, fbr9)) in zip(fac_cols9, h_factors.items()):
            with col_f9:
                color9 = "#00e676" if fb9 > 0 else "#ff3d57" if fbr9 > 0 else "#7fa8c8"
                st.markdown(f'''<div style="background:#0d1117;border:1px solid #1e3040;border-top:2px solid {color9};
                  border-radius:3px;padding:8px;text-align:center;">
                  <div style="font-family:Barlow Condensed,sans-serif;font-size:9px;letter-spacing:1px;
                              color:#7fa8c8;text-transform:uppercase;">{fname9}</div>
                  <div style="font-family:JetBrains Mono,monospace;font-size:12px;color:{color9};font-weight:600;margin-top:4px;">{fval9}</div>
                  </div>''', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠️ Signal error: {e}")

    # ── S&R Levels ─────────────────────────────────────────────────────
    st.markdown("---")
    section_header("Support & Resistance")
    try:
        h_ce_top3 = h_oc.sort_values("CE_OI",    ascending=False).head(3)
        h_pe_top3 = h_oc.sort_values("PE_OI",    ascending=False).head(3)
        h_cv_top3 = h_oc.sort_values("CE_Volume", ascending=False).head(3)
        h_pv_top3 = h_oc.sort_values("PE_Volume", ascending=False).head(3)

        h_res_c    = [s for s in h_ce_top3["Strike"].tolist() + h_cv_top3["Strike"].tolist() if s > h_spot]
        h_sup_c    = [s for s in h_pe_top3["Strike"].tolist() + h_pv_top3["Strike"].tolist() if s < h_spot]
        h_main_res = min(h_res_c) if h_res_c else None
        h_main_sup = max(h_sup_c) if h_sup_c else None

        metrics_row(
            metric_card("SPOT",        f"₹{h_spot:,.0f}", "", "#ff8c00") +
            (metric_card("RESISTANCE", f"{h_main_res:,.0f}", f"+{h_main_res-h_spot:.0f} pts", "#ff3d57") if h_main_res else "") +
            (metric_card("SUPPORT",    f"{h_main_sup:,.0f}", f"-{h_spot-h_main_sup:.0f} pts", "#00e676") if h_main_sup else "") +
            (metric_card("RANGE",      f"{h_main_res-h_main_sup:.0f} pts", "Res − Sup", "#c084fc") if h_main_res and h_main_sup else "")
        )

        sr9c1, sr9c2, sr9c3, sr9c4 = st.columns(4)
        with sr9c1:
            section_header("🔴 CE OI Resistance")
            for i, (_, row) in enumerate(h_ce_top3.iterrows(), 1):
                clr = "#ff3d57" if row["Strike"] > h_spot else "#7fa8c8"
                st.markdown(f'''<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:{clr};
                  padding:4px 0;border-bottom:1px solid #1e3040;">
                  R{i}: <b>{int(row["Strike"])}</b> | {int(row["CE_OI"])/1e5:.1f}L</div>''', unsafe_allow_html=True)
        with sr9c2:
            section_header("🟢 PE OI Support")
            for i, (_, row) in enumerate(h_pe_top3.iterrows(), 1):
                clr = "#00e676" if row["Strike"] < h_spot else "#7fa8c8"
                st.markdown(f'''<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:{clr};
                  padding:4px 0;border-bottom:1px solid #1e3040;">
                  S{i}: <b>{int(row["Strike"])}</b> | {int(row["PE_OI"])/1e5:.1f}L</div>''', unsafe_allow_html=True)
        with sr9c3:
            section_header("⚡ CE Vol Resistance")
            for i, (_, row) in enumerate(h_cv_top3.iterrows(), 1):
                st.markdown(f'''<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#ff8c00;
                  padding:4px 0;border-bottom:1px solid #1e3040;">
                  VR{i}: <b>{int(row["Strike"])}</b> | {int(row["CE_Volume"])/1e3:.0f}K</div>''', unsafe_allow_html=True)
        with sr9c4:
            section_header("🔥 PE Vol Support")
            for i, (_, row) in enumerate(h_pv_top3.iterrows(), 1):
                st.markdown(f'''<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#00d4ff;
                  padding:4px 0;border-bottom:1px solid #1e3040;">
                  VS{i}: <b>{int(row["Strike"])}</b> | {int(row["PE_Volume"])/1e3:.0f}K</div>''', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠️ S&R error: {e}")

    # ── Full Option Chain Table ─────────────────────────────────────────
    st.markdown("---")
    section_header("Full Option Chain")

    h_disp_cols = ["Strike","CE_IV","CE_Delta","CE_OI","CE_OI_Change","CE_OI_Change_%",
                   "CE_Volume","CE_LTP",
                   "PE_LTP","PE_Volume","PE_OI_Change_%","PE_OI_Change",
                   "PE_OI","PE_Delta","PE_IV"]
    h_disp_cols = [c for c in h_disp_cols if c in h_oc.columns]
    h_oc_show   = h_oc[h_disp_cols].copy()

    def style_h_chain(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for idx in df.index:
            strike = df.at[idx, 'Strike'] if 'Strike' in df.columns else 0
            if strike == atm_h:
                styles.loc[idx, :] = 'background-color:#1a1200;border-top:1px solid #ffd600;border-bottom:1px solid #ffd600;font-weight:bold;'
            else:
                if 'CE_OI_Change' in df.columns and df.at[idx, 'CE_OI_Change'] > 100000:
                    styles.at[idx, 'CE_OI_Change'] = 'color:#ff3d57;font-weight:600;'
                if 'PE_OI_Change' in df.columns and df.at[idx, 'PE_OI_Change'] > 100000:
                    styles.at[idx, 'PE_OI_Change'] = 'color:#00e676;font-weight:600;'
        return styles

    h_float_cols = h_oc_show.select_dtypes(include=['float64','float32']).columns
    h_styled = h_oc_show.style.apply(style_h_chain, axis=None).format({c: "{:.2f}" for c in h_float_cols})
    st.dataframe(h_styled, use_container_width=True, height=520)

    # ── Max Pain ────────────────────────────────────────────────────────
    st.markdown("---")
    section_header("Max Pain")
    try:
        import plotly.graph_objects as go
        h_mp_strike, h_pain_dict = calculate_max_pain(h_oc)
        h_pain_strikes = list(h_pain_dict.keys())
        h_pain_values  = [h_pain_dict[s]/1e7 for s in h_pain_strikes]
        h_pain_colors  = ["#ffd600" if s == h_mp_strike else
                          "#ff8c00" if abs(s - h_mp_strike) <= 100 else "#1e3040"
                          for s in h_pain_strikes]
        h_pain_fig = go.Figure()
        h_pain_fig.add_bar(x=h_pain_strikes, y=h_pain_values, marker_color=h_pain_colors)
        h_pain_fig.add_vline(x=float(h_mp_strike), line_color="#ffd600", line_dash="dash", line_width=2,
                             annotation_text=f"MAX PAIN {int(h_mp_strike)}",
                             annotation_font_color="#ffd600", annotation_font_size=11)
        h_pain_fig.add_vline(x=float(atm_h), line_color="#00d4ff", line_dash="dot", line_width=1.5,
                             annotation_text=f"SPOT {h_spot:,.0f}",
                             annotation_font_color="#00d4ff", annotation_font_size=11)
        h_pain_fig.update_layout(
            height=300, paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
            font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d"),
            yaxis=dict(gridcolor="#1e3040", title="Writer Pain (Cr)"),
            showlegend=False
        )
        mp9col1, mp9col2 = st.columns([3, 1])
        with mp9col1:
            st.plotly_chart(h_pain_fig, use_container_width=True)
        with mp9col2:
            pain_dir9 = "above" if h_spot > h_mp_strike else "below"
            st.markdown(f"""
            <div style="background:#111920;border:1px solid #ffd600;border-radius:4px;padding:14px;margin-top:10px;">
              <div style="font-family:Barlow Condensed,sans-serif;font-size:10px;letter-spacing:1.5px;color:#7fa8c8;">MAX PAIN</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:26px;font-weight:700;color:#ffd600;">{int(h_mp_strike):,}</div>
              <div style="font-family:Barlow Condensed,sans-serif;font-size:10px;color:#3a6080;margin-top:4px;">
                Spot {pain_dir9} pain by {abs(h_spot-h_mp_strike):.0f} pts
              </div>
            </div>""", unsafe_allow_html=True)
    except ImportError:
        st.info("pip install plotly for chart")
    except Exception as e:
        st.warning(f"Max Pain error: {e}")

    # ── Export & File Info ──────────────────────────────────────────────
    st.markdown("---")
    exp9c1, exp9c2 = st.columns(2)
    with exp9c1:
        csv_export9 = h_oc_show.to_csv(index=False)
        st.download_button(
            "📥 EXPORT THIS MINUTE AS CSV",
            csv_export9,
            file_name=f"oc_{h_index_name}_{sel_date_label.replace('/','')}_{sel_time.replace(':','')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with exp9c2:
        try:
            fsize = os.path.getsize(sel_date_path) / 1024
        except:
            fsize = 0
        st.markdown(f"""
        <div style="background:#111920;border:1px solid #1e3040;border-radius:3px;padding:10px 14px;
                    font-family:JetBrains Mono,monospace;">
          <div style="font-family:Barlow Condensed,sans-serif;font-size:9px;letter-spacing:1.5px;color:#7fa8c8;">
            TODAY'S DAILY FILE</div>
          <div style="font-size:16px;font-weight:700;color:#e8f4ff;">{len(avail_times)} minute snapshots</div>
          <div style="font-size:11px;color:#3a6080;">{os.path.basename(sel_date_path)} · {fsize:.1f} KB · parquet</div>
        </div>""", unsafe_allow_html=True)

    # ======================================================
