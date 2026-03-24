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
# TAB 2 — SMART MONEY + GEX
# ======================================================
def render():
    section_header("Smart Money Flow + GEX Analysis", "Institutional positioning, Gamma Exposure, Call/Put Walls")

    oc2     = st.session_state.get("current_option_chain", pd.DataFrame())
    spot2   = st.session_state.get("current_spot_price")
    sel_idx2= st.session_state.get("current_selected_index", "")
    sel_exp2= st.session_state.get("oc_expiry_select", "2026-03-27")

    if oc2 is None or (isinstance(oc2, pd.DataFrame) and oc2.empty) or spot2 is None:
        st.warning("⏳ Load option chain from Tab 1 first"); return

    expiry_dt      = pd.to_datetime(sel_exp2 if sel_exp2 else "2026-03-27")
    days_to_expiry = max(1, (expiry_dt - datetime.now()).days)
    T              = days_to_expiry / 365
    lot_size2      = get_lot_size(sel_idx2)
    risk_free_rate = 0.065

    gex_list = []
    for _, row in oc2.iterrows():
        ce_iv  = float(row["CE_IV"]) / 100 if float(row["CE_IV"]) > 0 else 0.001
        pe_iv  = float(row["PE_IV"]) / 100 if float(row["PE_IV"]) > 0 else 0.001
        K      = float(row["Strike"])
        ce_gex = calculate_gamma_bs(spot2, K, T, risk_free_rate, ce_iv) * float(row["CE_OI"]) * lot_size2 * spot2**2
        pe_gex = calculate_gamma_bs(spot2, K, T, risk_free_rate, pe_iv) * float(row["PE_OI"]) * lot_size2 * spot2**2
        gex_list.append(ce_gex - pe_gex)

    oc2 = oc2.copy()
    oc2["GEX"] = gex_list
    total_net_gex  = oc2["GEX"].sum()
    max_gex_strike = oc2.loc[oc2["GEX"].abs().idxmax(), "Strike"]
    gex_sorted     = oc2.sort_values("Strike")
    zero_idx       = (gex_sorted["GEX"].abs()).argsort().iloc[0]
    zero_gex_strike= gex_sorted.iloc[zero_idx]["Strike"]

    # GEX flip
    prev_gex = st.session_state.get("prev_net_gex", 0)
    if prev_gex > 0 and total_net_gex < 0:
        st.toast("⚡ GEX FLIP → TRENDING MODE", icon="🔴")
        st.session_state.alert_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"),
                                           "msg": "GEX Flip → Trending", "type": "DANGER"})
    elif prev_gex < 0 and total_net_gex > 0:
        st.toast("⚡ GEX FLIP → MEAN REVERSION MODE", icon="🟢")
        st.session_state.alert_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"),
                                           "msg": "GEX Flip → Mean Rev", "type": "INFO"})
    st.session_state.prev_net_gex = total_net_gex

    gex_color = "#00e676" if total_net_gex > 0 else "#ff3d57"
    metrics_row(
        metric_card("NET GEX", f"{total_net_gex/1e6:.1f}M", "Total Gamma Exposure", gex_color) +
        metric_card("MAX GEX WALL", f"{max_gex_strike:,.0f}", "Highest gamma pin", "#ffd600") +
        metric_card("ZERO GAMMA", f"{zero_gex_strike:,.0f}", "Flip level", "#00d4ff") +
        metric_card("DAYS TO EXPIRY", f"{days_to_expiry}d", color="#c084fc")
    )

    gex_mode = "📌 MEAN REVERSION" if total_net_gex > 0 else "🌊 TRENDING/VOLATILE"
    gex_m_color = "#00e676" if total_net_gex > 0 else "#ff3d57"
    st.markdown(f'<div style="background:#0d1117;border:1px solid {gex_m_color};border-left:4px solid {gex_m_color};padding:10px 16px;border-radius:3px;font-family:Barlow Condensed,sans-serif;font-size:16px;font-weight:700;letter-spacing:2px;color:{gex_m_color};">GEX REGIME: {gex_mode}</div>', unsafe_allow_html=True)

    # GEX Bar Chart
    st.markdown("---")
    section_header("GEX by Strike", "Positive = Market Makers long gamma (stabilizing) | Negative = short gamma (amplifying)")
    try:
        import plotly.graph_objects as go
        gex_fig = go.Figure()
        gex_sorted_chart = oc2.sort_values("Strike")
        colors_gex = ["#00e676" if v > 0 else "#ff3d57" for v in gex_sorted_chart["GEX"]]
        gex_fig.add_bar(x=gex_sorted_chart["Strike"], y=gex_sorted_chart["GEX"]/1e6,
                        marker_color=colors_gex, name="GEX")
        gex_fig.add_vline(x=float(max_gex_strike), line_color="#ffd600", line_dash="dash", line_width=1.5,
                          annotation_text="MAX WALL", annotation_font_color="#ffd600", annotation_font_size=10)
        gex_fig.add_vline(x=float(get_atm_strike(spot2, sel_idx2)), line_color="#00d4ff", line_dash="dot", line_width=1.5,
                          annotation_text="ATM", annotation_font_color="#00d4ff", annotation_font_size=10)
        gex_fig.update_layout(
            height=300, paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
            font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d"),
            yaxis=dict(gridcolor="#1e3040", title="GEX (Millions)"),
            showlegend=False
        )
        st.plotly_chart(gex_fig, use_container_width=True)
    except ImportError:
        st.info("pip install plotly for GEX chart")

    # Smart Money Flow
    st.markdown("---")
    section_header("Smart Money Flow Detector", "ATM OI change momentum")

    atm_s2   = get_atm_strike(spot2, sel_idx2)
    df2      = oc2.copy()
    df2["distance"] = abs(df2["Strike"] - spot2)
    atm_row2 = df2.loc[df2["distance"].idxmin()]

    # Use pre-computed OI_Change columns; fall back to raw subtraction only if column missing
    if "CE_OI_Change" in oc2.columns and "PE_OI_Change" in oc2.columns:
        ce_oi_chg2 = float(atm_row2["CE_OI_Change"])
        pe_oi_chg2 = float(atm_row2["PE_OI_Change"])
    elif "CE_Prev_OI" in oc2.columns and "PE_Prev_OI" in oc2.columns:
        ce_oi_chg2 = float(atm_row2["CE_OI"]) - float(atm_row2["CE_Prev_OI"])
        pe_oi_chg2 = float(atm_row2["PE_OI"]) - float(atm_row2["PE_Prev_OI"])
    else:
        ce_oi_chg2 = 0.0
        pe_oi_chg2 = 0.0
    ce_pct2 = (ce_oi_chg2 / float(atm_row2["CE_OI"])) * 100 if float(atm_row2.get("CE_OI", 0)) != 0 else 0
    pe_pct2 = (pe_oi_chg2 / float(atm_row2["PE_OI"])) * 100 if float(atm_row2.get("PE_OI", 0)) != 0 else 0

    if pe_pct2 > ce_pct2 * 1.25:   flow2, flow_color2 = "BULLISH", "#00e676"
    elif ce_pct2 > pe_pct2 * 1.25: flow2, flow_color2 = "BEARISH", "#ff3d57"
    else:                           flow2, flow_color2 = "RANGE",   "#ffd600"

    call_wall2 = df2.loc[df2["CE_OI"].idxmax()]["Strike"]
    put_wall2  = df2.loc[df2["PE_OI"].idxmax()]["Strike"]

    metrics_row(
        metric_card("FLOW", flow2, "", flow_color2) +
        metric_card("CALL WALL", f"{call_wall2:,.0f}", "Max CE OI = Resistance", "#ff3d57") +
        metric_card("PUT WALL", f"{put_wall2:,.0f}", "Max PE OI = Support", "#00e676") +
        metric_card("CE OI CHG%", f"{ce_pct2:+.1f}%", color="#ff3d57") +
        metric_card("PE OI CHG%", f"{pe_pct2:+.1f}%", color="#00e676")
    )

    # 5-Strike Scalper
    st.markdown("---")
    section_header("5-Strike Institutional Scalper", "CE vs PE power at ATM ±2 strikes")
    strikes2 = sorted(df2["Strike"].unique())
    if atm_s2 in strikes2:
        ai2    = strikes2.index(atm_s2)
        sel_s2 = strikes2[max(0, ai2-2): ai2+3]
        mini2  = df2[df2["Strike"].isin(sel_s2)].copy().sort_values("Strike")
        ce_s_list, pe_s_list = [], []
        for _, row2 in mini2.iterrows():
            ce_c = sum([row2["CE_OI_Change"]>row2["PE_OI_Change"],
                        row2["CE_Volume"]>row2["PE_Volume"],
                        row2["CE_OI"]>row2["PE_OI"]])
            pe_c = 3 - ce_c
            ce_s_list.append(ce_c); pe_s_list.append(pe_c)
        mini2["CE_Strength"] = ce_s_list
        mini2["PE_Strength"] = pe_s_list

        show2 = ["Strike","CE_LTP","CE_OI","CE_OI_Change","CE_Strength",
                 "PE_Strength","PE_OI_Change","PE_OI","PE_LTP","GEX"]
        show2 = [c for c in show2 if c in mini2.columns]
        st.dataframe(mini2[show2], use_container_width=True, height=200)

    # OI Heatmap
    st.markdown("---")
    section_header("OI Buildup Heatmap", "CE vs PE Open Interest by strike")
    try:
        oc_hm = oc2.sort_values("Strike")
        hm_fig = go.Figure()
        hm_fig.add_bar(x=oc_hm["Strike"], y=oc_hm["CE_OI"]/1e5,
                       name="CE OI", marker_color="#ff3d57", opacity=0.85)
        hm_fig.add_bar(x=oc_hm["Strike"], y=oc_hm["PE_OI"]/1e5,
                       name="PE OI", marker_color="#00e676", opacity=0.85)
        hm_fig.update_layout(
            barmode="group", height=320,
            paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
            font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d"),
            yaxis=dict(gridcolor="#1e3040", title="OI (Lakhs)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7fa8c8")
        )
        st.plotly_chart(hm_fig, use_container_width=True)
    except ImportError:
        st.info("pip install plotly for OI heatmap")

    # ======================================================
    # CHANGE IN OI CHART — CE vs PE per strike
    # ======================================================
    st.markdown("---")
    section_header("Change in OI by Strike", "CE (red) vs PE (green) — net OI added/removed this session")
    try:
        oc_chg = oc2.sort_values("Strike").copy()
        if "CE_OI_Change" in oc_chg.columns and "PE_OI_Change" in oc_chg.columns:
            atm_strike_chg = get_atm_strike(spot2, sel_idx2)
            # Focus on ±10 strikes around ATM for readability
            strikes_all = sorted(oc_chg["Strike"].unique())
            if atm_strike_chg in strikes_all:
                ai_chg = strikes_all.index(atm_strike_chg)
                focus_strikes = strikes_all[max(0, ai_chg - 10): ai_chg + 11]
                oc_chg = oc_chg[oc_chg["Strike"].isin(focus_strikes)]

            chg_fig = go.Figure()
            chg_fig.add_bar(
                x=oc_chg["Strike"], y=oc_chg["CE_OI_Change"] / 1e3,
                name="CE OI Chg", marker_color="#ff3d57", opacity=0.85
            )
            chg_fig.add_bar(
                x=oc_chg["Strike"], y=oc_chg["PE_OI_Change"] / 1e3,
                name="PE OI Chg", marker_color="#00e676", opacity=0.85
            )
            # Highlight ATM
            chg_fig.add_vline(
                x=float(atm_strike_chg), line_color="#00d4ff",
                line_dash="dot", line_width=1.5,
                annotation_text="ATM", annotation_font_color="#00d4ff", annotation_font_size=10
            )
            # Zero line
            chg_fig.add_hline(y=0, line_color="#3a6080", line_width=0.8)
            chg_fig.update_layout(
                barmode="group", height=300,
                paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
                font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d", title="Strike"),
                yaxis=dict(gridcolor="#1e3040", title="OI Change (000s)"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7fa8c8"),
            )
            st.plotly_chart(chg_fig, use_container_width=True)

            # Net OI Change summary
            total_ce_chg = oc2["CE_OI_Change"].sum()
            total_pe_chg = oc2["PE_OI_Change"].sum()
            net_chg_bias = "🟢 PE ADDING (BULLISH)" if total_pe_chg > total_ce_chg else "🔴 CE ADDING (BEARISH)"
            c1c, c2c, c3c = st.columns(3)
            with c1c:
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ff3d57;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">TOTAL CE OI CHG</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;
                        color:{'#00e676' if total_ce_chg > 0 else '#ff3d57'};">{total_ce_chg/1e3:+.0f}K</div>
                    </div>""", unsafe_allow_html=True)
            with c2c:
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #00e676;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">TOTAL PE OI CHG</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;
                        color:{'#00e676' if total_pe_chg > 0 else '#ff3d57'};">{total_pe_chg/1e3:+.0f}K</div>
                    </div>""", unsafe_allow_html=True)
            with c3c:
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ffd600;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">NET BIAS</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;color:#ffd600;">{net_chg_bias}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("OI Change data not available in current option chain.")
    except Exception as e:
        st.warning(f"OI Change chart error: {e}")

    # ======================================================
    # OI BUILDUP TABLE — Long/Short Build/Unwind per strike
    # ======================================================
    st.markdown("---")
    section_header("OI Buildup Analysis", "Classify institutional activity per strike: Long Build · Short Cover · Long Unwind · Short Build")
    try:
        oc_bu = oc2.copy()
        if all(c in oc_bu.columns for c in ["CE_OI_Change", "PE_OI_Change", "CE_LTP", "PE_LTP"]):

            def _classify(oi_chg, ltp_chg_proxy, side):
                """
                OI up + price up   → Long Build   (bullish)
                OI up + price down → Short Build  (bearish)
                OI down + price up → Short Cover  (bullish)
                OI down + price dn → Long Unwind  (bearish)
                Using OI change sign as proxy for price direction (CE rises when market bullish).
                """
                if side == "CE":
                    if oi_chg > 0:
                        return ("Long Build",  "#00e676", "↑ BULL")
                    else:
                        return ("Long Unwind", "#ff3d57", "↓ BEAR")
                else:  # PE
                    if oi_chg > 0:
                        return ("Short Build", "#ff3d57", "↓ BEAR")
                    else:
                        return ("Short Cover", "#00e676", "↑ BULL")

            rows_bu = []
            atm_bu = get_atm_strike(spot2, sel_idx2)
            strikes_bu = sorted(oc_bu["Strike"].unique())
            if atm_bu in strikes_bu:
                ai_bu = strikes_bu.index(atm_bu)
                focus_bu = strikes_bu[max(0, ai_bu - 7): ai_bu + 8]
            else:
                focus_bu = strikes_bu[:15]

            for _, r in oc_bu[oc_bu["Strike"].isin(focus_bu)].iterrows():
                ce_cls, ce_col, ce_signal = _classify(r["CE_OI_Change"], r["CE_LTP"], "CE")
                pe_cls, pe_col, pe_signal = _classify(r["PE_OI_Change"], r["PE_LTP"], "PE")
                atm_tag = " ★" if r["Strike"] == atm_bu else ""

                # Net Bias rule:
                # CE OI Chg > PE OI Chg by >20%  → BEARISH (more call writing = resistance)
                # PE OI Chg > CE OI Chg by >20%  → BULLISH (more put writing = support)
                # Difference <20%                 → NEUTRAL
                ce_chg    = abs(float(r["CE_OI_Change"]))
                pe_chg    = abs(float(r["PE_OI_Change"]))
                total_chg = ce_chg + pe_chg
                if total_chg == 0:
                    net_bias = "⚪ NEUT"
                else:
                    diff_pct = abs(ce_chg - pe_chg) / max(total_chg, 1) * 100
                    if diff_pct < 20:
                        net_bias = "⚪ NEUT"
                    elif ce_chg > pe_chg:
                        net_bias = "🔴 BEAR"   # More CE OI added → bearish
                    else:
                        net_bias = "🟢 BULL"   # More PE OI added → bullish

                rows_bu.append({
                    "Strike":      f"{int(r['Strike']):,}{atm_tag}",
                    "CE OI Chg":   f"{r['CE_OI_Change']/1e3:+.1f}K",
                    "CE Activity": ce_cls,
                    "CE Signal":   ce_signal,
                    "PE OI Chg":   f"{r['PE_OI_Change']/1e3:+.1f}K",
                    "PE Activity": pe_cls,
                    "PE Signal":   pe_signal,
                    "Net Bias":    net_bias,
                })

            df_bu = pd.DataFrame(rows_bu)
            st.dataframe(
                df_bu,
                use_container_width=True,
                height=min(35 * len(df_bu) + 38, 480),
                column_config={
                    "Strike":      st.column_config.TextColumn("Strike", width=90),
                    "CE OI Chg":   st.column_config.TextColumn("CE OI Δ", width=85),
                    "CE Activity": st.column_config.TextColumn("CE Activity", width=110),
                    "CE Signal":   st.column_config.TextColumn("CE Signal", width=75),
                    "PE OI Chg":   st.column_config.TextColumn("PE OI Δ", width=85),
                    "PE Activity": st.column_config.TextColumn("PE Activity", width=110),
                    "PE Signal":   st.column_config.TextColumn("PE Signal", width=75),
                    "Net Bias":    st.column_config.TextColumn("Net Bias", width=90),
                }
            )
        else:
            st.info("OI Change columns not available.")
    except Exception as e:
        st.warning(f"OI Buildup error: {e}")

    # ======================================================
    # VOLUME SPIKE ALERTS
    # ======================================================
    st.markdown("---")
    section_header("🚨 Volume Spike Alerts", "Strikes where CE or PE volume is 2× or more above average — unusual institutional activity")
    try:
        oc_vs = oc2.copy()
        if "CE_Volume" in oc_vs.columns and "PE_Volume" in oc_vs.columns:
            avg_ce_vol = oc_vs["CE_Volume"].mean()
            avg_pe_vol = oc_vs["PE_Volume"].mean()
            spike_threshold = 2.0  # 2× average

            ce_spikes = oc_vs[oc_vs["CE_Volume"] >= avg_ce_vol * spike_threshold].copy()
            pe_spikes = oc_vs[oc_vs["PE_Volume"] >= avg_pe_vol * spike_threshold].copy()

            atm_vs = get_atm_strike(spot2, sel_idx2)

            def _spike_bias(ce_chg, pe_chg):
                """CE OI Chg vs PE OI Chg with 20% threshold — same rule as OI Buildup."""
                ce_abs = abs(float(ce_chg)); pe_abs = abs(float(pe_chg))
                total  = ce_abs + pe_abs
                if total == 0: return "⚪ NEUTRAL"
                diff_pct = abs(ce_abs - pe_abs) / max(total, 1) * 100
                if diff_pct < 10:   return "⚪ NEUTRAL"
                elif ce_abs > pe_abs: return "🔴 BEARISH"
                else:                 return "🟢 BULLISH"

            spike_alerts = []
            for _, r in ce_spikes.iterrows():
                mult   = r["CE_Volume"] / max(avg_ce_vol, 1)
                rel    = "ITM" if r["Strike"] < spot2 else "OTM" if r["Strike"] > spot2 else "ATM"
                ce_chg = r.get("CE_OI_Change", 0); pe_chg = r.get("PE_OI_Change", 0)
                spike_alerts.append({
                    "Type": "CE 🔴", "Strike": int(r["Strike"]),
                    "Volume": f"{int(r['CE_Volume']):,}",
                    "vs Avg": f"{mult:.1f}×",
                    "CE OI Chg": f"{ce_chg/1e3:+.1f}K",
                    "PE OI Chg": f"{pe_chg/1e3:+.1f}K",
                    "Moneyness": rel,
                    "Net Bias": _spike_bias(ce_chg, pe_chg),
                })
            for _, r in pe_spikes.iterrows():
                mult   = r["PE_Volume"] / max(avg_pe_vol, 1)
                rel    = "ITM" if r["Strike"] > spot2 else "OTM" if r["Strike"] < spot2 else "ATM"
                ce_chg = r.get("CE_OI_Change", 0); pe_chg = r.get("PE_OI_Change", 0)
                spike_alerts.append({
                    "Type": "PE 🟢", "Strike": int(r["Strike"]),
                    "Volume": f"{int(r['PE_Volume']):,}",
                    "vs Avg": f"{mult:.1f}×",
                    "CE OI Chg": f"{ce_chg/1e3:+.1f}K",
                    "PE OI Chg": f"{pe_chg/1e3:+.1f}K",
                    "Moneyness": rel,
                    "Net Bias": _spike_bias(ce_chg, pe_chg),
                })

            if spike_alerts:
                df_spikes = pd.DataFrame(spike_alerts).sort_values("Strike")
                # Fire toast for very high spikes (>3×)
                high_spikes = [a for a in spike_alerts if float(a["vs Avg"].replace("×","")) >= 3.0]
                for hs in high_spikes[:3]:
                    st.toast(f"🚨 Vol Spike {hs['Type']} {hs['Strike']} — {hs['vs Avg']} avg", icon="🔔")
                    if "alert_log" in st.session_state:
                        st.session_state.alert_log.append({
                            "time": datetime.now(IST).strftime("%H:%M:%S"),
                            "msg":  f"Vol Spike {hs['Type']} {hs['Strike']} ({hs['vs Avg']})",
                            "type": "DANGER"
                        })

                # ── Total OI comparison with 20% rule ────────────────────
                total_ce_oi_vs = float(oc_vs["CE_OI"].sum())
                total_pe_oi_vs = float(oc_vs["PE_OI"].sum())
                total_oi_sum   = total_ce_oi_vs + total_pe_oi_vs
                oi_diff_pct    = abs(total_ce_oi_vs - total_pe_oi_vs) / max(total_oi_sum, 1) * 100
                if oi_diff_pct < 10:
                    oi_bias, oi_bias_color = "⚪ NEUTRAL",  "#7fa8c8"
                elif total_ce_oi_vs > total_pe_oi_vs:
                    oi_bias, oi_bias_color = "🔴 BEARISH",  "#ff3d57"
                else:
                    oi_bias, oi_bias_color = "🟢 BULLISH",  "#00e676"

                # ── 5-card summary row ────────────────────────────────────
                sv1, sv2, sv3, sv4, sv5 = st.columns(5)
                with sv1:
                    st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ff3d57;
                        border-radius:3px;padding:10px 14px;">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">TOTAL CE OI</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:#ff3d57;">{total_ce_oi_vs/1e5:.1f}L</div>
                        <div style="font-size:10px;color:#3a6080;">CE Spikes: {len(ce_spikes)} strikes</div>
                        </div>""", unsafe_allow_html=True)
                with sv2:
                    st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #00e676;
                        border-radius:3px;padding:10px 14px;">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">TOTAL PE OI</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:#00e676;">{total_pe_oi_vs/1e5:.1f}L</div>
                        <div style="font-size:10px;color:#3a6080;">PE Spikes: {len(pe_spikes)} strikes</div>
                        </div>""", unsafe_allow_html=True)
                with sv3:
                    st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid {oi_bias_color};
                        border-radius:3px;padding:10px 14px;">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">OI NET BIAS</div>
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:700;color:{oi_bias_color};margin-top:2px;">{oi_bias}</div>
                        <div style="font-size:10px;color:#3a6080;">Diff: {oi_diff_pct:.1f}% {'> 10% threshold' if oi_diff_pct >= 10 else '< 10% → Neutral'}</div>
                        </div>""", unsafe_allow_html=True)
                with sv4:
                    pcr_vs = round(total_pe_oi_vs / max(total_ce_oi_vs, 1), 3)
                    pcr_c  = "#00e676" if pcr_vs > 1.2 else "#ff3d57" if pcr_vs < 0.8 else "#ffd600"
                    st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid {pcr_c};
                        border-radius:3px;padding:10px 14px;">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">PCR (OI)</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:{pcr_c};">{pcr_vs:.3f}</div>
                        <div style="font-size:10px;color:#3a6080;">PE÷CE OI ratio</div>
                        </div>""", unsafe_allow_html=True)
                with sv5:
                    top_spike = max(spike_alerts, key=lambda x: float(x["vs Avg"].replace("×","")))
                    st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ffd600;
                        border-radius:3px;padding:10px 14px;">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">BIGGEST SPIKE</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:#ffd600;">
                            {top_spike['Type']} {top_spike['Strike']}</div>
                        <div style="font-size:11px;color:#ff8c00;">{top_spike['vs Avg']} of avg volume</div>
                        </div>""", unsafe_allow_html=True)

                # ── OI comparison bar ────────────────────────────────────
                ce_pct_bar = total_ce_oi_vs / max(total_oi_sum, 1) * 100
                pe_pct_bar = 100 - ce_pct_bar
                st.markdown(f"""
                <div style="margin:10px 0 4px 0;font-family:'Barlow Condensed',sans-serif;
                            font-size:9px;letter-spacing:1.5px;color:#3a6080;">
                  TOTAL OI SPLIT — CE {ce_pct_bar:.1f}% vs PE {pe_pct_bar:.1f}%
                </div>
                <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:12px;">
                  <div style="width:{ce_pct_bar:.1f}%;background:#ff3d57;"></div>
                  <div style="width:{pe_pct_bar:.1f}%;background:#00e676;"></div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.dataframe(
                    df_spikes,
                    use_container_width=True,
                    height=min(35 * len(df_spikes) + 38, 420),
                    column_config={
                        "Type":       st.column_config.TextColumn("Type",      width=70),
                        "Strike":     st.column_config.NumberColumn("Strike",   format="%d", width=80),
                        "Volume":     st.column_config.TextColumn("Volume",     width=100),
                        "vs Avg":     st.column_config.TextColumn("vs Avg",     width=65),
                        "CE OI Chg":  st.column_config.TextColumn("CE OI Δ",   width=85),
                        "PE OI Chg":  st.column_config.TextColumn("PE OI Δ",   width=85),
                        "Moneyness":  st.column_config.TextColumn("Money",      width=60),
                        "Net Bias":   st.column_config.TextColumn("Net Bias",   width=100),
                    }
                )

                # Volume spike chart
                vs_fig = go.Figure()
                oc_sorted_vs = oc_vs.sort_values("Strike")
                vs_fig.add_bar(
                    x=oc_sorted_vs["Strike"],
                    y=oc_sorted_vs["CE_Volume"] / 1e3,
                    name="CE Volume", marker_color="#ff3d57", opacity=0.75
                )
                vs_fig.add_bar(
                    x=oc_sorted_vs["Strike"],
                    y=oc_sorted_vs["PE_Volume"] / 1e3,
                    name="PE Volume", marker_color="#00e676", opacity=0.75
                )
                vs_fig.add_hline(y=avg_ce_vol * spike_threshold / 1e3,
                                  line_color="#ff8c00", line_dash="dash", line_width=1.2,
                                  annotation_text=f"CE {spike_threshold:.0f}× threshold",
                                  annotation_font_color="#ff8c00", annotation_font_size=9)
                vs_fig.add_hline(y=avg_pe_vol * spike_threshold / 1e3,
                                  line_color="#00d4ff", line_dash="dash", line_width=1.2,
                                  annotation_text=f"PE {spike_threshold:.0f}× threshold",
                                  annotation_font_color="#00d4ff", annotation_font_size=9)
                vs_fig.update_layout(
                    barmode="group", height=280,
                    paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
                    font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
                    margin=dict(l=40, r=20, t=10, b=40),
                    xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d"),
                    yaxis=dict(gridcolor="#1e3040", title="Volume (000s)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7fa8c8"),
                )
                st.plotly_chart(vs_fig, use_container_width=True)
            else:
                st.success("✅ No unusual volume spikes detected — market activity is normal.")
        else:
            st.info("Volume data not available in current option chain.")
    except Exception as e:
        st.warning(f"Volume spike error: {e}")


    # ======================================================
    # SIDEBAR — MANUAL TRADE EXECUTION
    # ======================================================
    st.sidebar.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:700;
            letter-spacing:2px;color:#ff8c00;border-bottom:1px solid #1e3040;
            padding-bottom:8px;margin-bottom:12px;">
      ⚡ EXECUTE TRADE
    </div>""", unsafe_allow_html=True)

    oc_sb   = st.session_state.get("current_option_chain", pd.DataFrame())
    sel_sb  = st.session_state.get("current_selected_index")
    spot_sb = st.session_state.get("current_spot_price")

    if not oc_sb.empty and sel_sb and MARKET_OPEN:
        lot_sb      = get_lot_size(sel_sb)
    strikes_sb  = sorted(oc_sb['Strike'].unique())
    atm_sb      = get_atm_strike(spot_sb, sel_sb)
    try:    atm_idx_sb = strikes_sb.index(atm_sb)
    except: atm_idx_sb = 0

    strike_sb   = st.sidebar.selectbox("Strike", strikes_sb, index=atm_idx_sb, key="sb_strike")
    opt_type_sb = st.sidebar.radio("Option Type", ["CE", "PE"], horizontal=True, key="sb_opt")
    action_sb   = st.sidebar.radio("Action", ["BUY", "SELL"], horizontal=True, key="sb_action")
    qty_sb      = st.sidebar.number_input("Qty", min_value=lot_sb, value=lot_sb, step=lot_sb, key="sb_qty")
    lots_cnt_sb = qty_sb // lot_sb

    row_sb     = oc_sb[oc_sb['Strike'] == strike_sb].iloc[0]
    cur_price_sb = float(row_sb['CE_LTP' if opt_type_sb == "CE" else 'PE_LTP'])

    sb_color = "#00e676" if action_sb == "BUY" else "#ff3d57"
    st.sidebar.markdown(f"""
    <div style="background:#0d1117;border:1px solid {sb_color};border-radius:3px;
                padding:10px;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:12px;">
      <div style="color:#7fa8c8;">LTP</div>
      <div style="color:{sb_color};font-size:20px;font-weight:700;">₹{cur_price_sb:.2f}</div>
      <div style="color:#7fa8c8;font-size:11px;">{lots_cnt_sb} Lot{'s' if lots_cnt_sb > 1 else ''} = {qty_sb} units</div>
      {"<div style='color:#ffd600;font-size:11px;font-weight:700;'>★ ATM</div>" if strike_sb == atm_sb else ""}
    </div>""", unsafe_allow_html=True)

    # Risk preview
    if action_sb == "SELL":
        max_loss_preview = cur_price_sb * 2 * lot_sb  # rough estimate
        st.sidebar.markdown(f"""<div style="font-family:'Barlow Condensed',sans-serif;
            font-size:11px;color:#ff3d57;letter-spacing:1px;">EST. MAX RISK ≈ ₹{max_loss_preview:,.0f}</div>""",
            unsafe_allow_html=True)

    if st.sidebar.button("✅ EXECUTE", use_container_width=True, type="primary"):
        _, _, grand_sb = compute_grand_total()
        if grand_sb <= st.session_state.daily_loss_limit:
            st.sidebar.error("🔴 Daily loss limit hit.")
        else:
            trade_sb = {
                "Entry_Time": datetime.now(IST).strftime("%H:%M:%S"),
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Index": sel_sb, "Strike": strike_sb,
                "Type": opt_type_sb, "Action": action_sb,
                "Qty": qty_sb, "Entry": cur_price_sb, "Status": "OPEN"
            }
            st.session_state.today_trades.append(trade_sb)
            save_list_to_csv(st.session_state.today_trades, TODAY_TRADES_FILE)
            st.sidebar.success(f"✅ {action_sb} {opt_type_sb} @ {strike_sb}")
            st.rerun()

    elif not MARKET_OPEN:
        st.sidebar.markdown(f"""<div style="background:#1a0a00;border:1px solid #ff8c00;
        border-radius:3px;padding:8px 12px;font-family:'Barlow Condensed',sans-serif;
        font-size:13px;color:#ff8c00;letter-spacing:1px;">🔴 MARKET {session_label}</div>""",
        unsafe_allow_html=True)
    else:
        st.sidebar.info("⏳ Load option chain from Tab 1")

    # Risk Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;
    font-weight:700;letter-spacing:2px;color:#7fa8c8;">⚙️ RISK CONTROLS</div>""",
    unsafe_allow_html=True)
    new_limit_sb = st.sidebar.number_input("Daily Loss Limit (₹)",
    value=int(st.session_state.daily_loss_limit), step=1000, key="dl_limit_sb")
    st.session_state.daily_loss_limit = new_limit_sb
    new_max_sb = st.sidebar.number_input("Max AI Positions",
    value=st.session_state.get("max_open_positions", 6), min_value=1, max_value=20, step=1, key="max_pos_sb")
    st.session_state["max_open_positions"] = new_max_sb

    # Daily P&L in sidebar
    open_pnl_sb, closed_pnl_sb, grand_sb = compute_grand_total()
    pnl_c_sb = "#00e676" if grand_sb >= 0 else "#ff3d57"
    st.sidebar.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;
            padding:10px;margin-top:8px;font-family:'JetBrains Mono',monospace;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;letter-spacing:1.5px;
              color:#7fa8c8;text-transform:uppercase;">Today's P&L</div>
      <div style="font-size:22px;font-weight:700;color:{pnl_c_sb};">₹{grand_sb:,.0f}</div>
      <div style="font-size:11px;color:#3a6080;">Open: ₹{open_pnl_sb:,.0f} | Closed: ₹{closed_pnl_sb:,.0f}</div>
      <div style="margin-top:6px;">
    <div style="font-size:10px;color:#3a6080;margin-bottom:2px;">vs Limit ₹{st.session_state.daily_loss_limit:,}</div>
    <div style="background:#1e3040;border-radius:2px;height:4px;">
      <div style="background:{pnl_c_sb};width:{min(abs(grand_sb)/max(abs(st.session_state.daily_loss_limit),1)*100,100):.0f}%;height:100%;border-radius:2px;"></div>
    </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Alert log in sidebar
    if st.session_state.alert_log:
        st.sidebar.markdown("---")
        st.sidebar.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
        letter-spacing:1.5px;color:#ff8c00;">🚨 ALERTS</div>""", unsafe_allow_html=True)
        for al in st.session_state.alert_log[-5:][::-1]:
            al_c = "#ff3d57" if al["type"] == "DANGER" else "#00d4ff"
            st.sidebar.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
            color:{al_c};padding:2px 0;">{al['time']} {al['msg']}</div>""", unsafe_allow_html=True)


    # ======================================================
