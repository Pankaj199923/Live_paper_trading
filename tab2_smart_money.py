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
# CUSTOM POPUP SYSTEM — slides LEFT → RIGHT at top of screen
# ======================================================

def _inject_popup_css():
    """Inject the popup container + keyframe CSS once per render."""
    st.markdown("""
    <style>
    #smart-popup-rail {
        position: fixed;
        top: 12px;
        left: 0;
        right: 0;
        z-index: 999999;
        pointer-events: none;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
        padding-left: 12px;
    }
    .smart-popup {
        pointer-events: auto;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 18px 9px 13px;
        border-radius: 4px;
        border-left: 4px solid;
        font-family: 'Barlow Condensed', 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.2px;
        white-space: nowrap;
        box-shadow: 0 4px 24px rgba(0,0,0,0.55);
        backdrop-filter: blur(6px);
        animation: popupSlide 0.38s cubic-bezier(.22,.68,0,1.2) forwards,
                   popupFade  4.5s ease-in 0.4s forwards;
    }
    @keyframes popupSlide {
        0%   { opacity: 0; transform: translateX(-120%); }
        100% { opacity: 1; transform: translateX(0);     }
    }
    @keyframes popupFade {
        0%   { opacity: 1; }
        75%  { opacity: 1; }
        100% { opacity: 0; pointer-events: none; }
    }
    .smart-popup.danger  { background: rgba(25,5,5,0.92);  border-color: #ff1744; color: #ff6b6b; }
    .smart-popup.success { background: rgba(3,18,10,0.92); border-color: #00e676; color: #69f0ae; }
    .smart-popup.warning { background: rgba(22,14,2,0.92); border-color: #ff8c00; color: #ffb74d; }
    .smart-popup.info    { background: rgba(2,12,22,0.92); border-color: #00d4ff; color: #80deea; }
    .smart-popup .pop-icon { font-size: 16px; line-height: 1; }
    .smart-popup .pop-msg  { font-size: 13px; }
    </style>
    <div id="smart-popup-rail"></div>
    """, unsafe_allow_html=True)


def _show_popup(message: str, kind: str = "danger", icon: str = "🚨", duration_ms: int = 5000):
    """
    Fire a popup that slides LEFT → RIGHT at the TOP of the screen.
    kind: 'danger' | 'success' | 'warning' | 'info'
    """
    import html as _html
    safe_msg  = _html.escape(str(message))
    safe_icon = icon
    js = f"""
    <script>
    (function() {{
        var rail = document.getElementById('smart-popup-rail');
        if (!rail) {{
            rail = document.createElement('div');
            rail.id = 'smart-popup-rail';
            Object.assign(rail.style, {{
                position:'fixed', top:'12px', left:'0', right:'0',
                zIndex:'999999', pointerEvents:'none',
                display:'flex', flexDirection:'column',
                alignItems:'flex-start', gap:'6px', paddingLeft:'12px'
            }});
            document.body.appendChild(rail);
        }}
        var p = document.createElement('div');
        p.className = 'smart-popup {kind}';
        p.innerHTML = '<span class="pop-icon">{safe_icon}</span>'
                    + '<span class="pop-msg">{safe_msg}</span>';
        rail.insertBefore(p, rail.firstChild);
        setTimeout(function() {{
            if (p.parentNode) p.parentNode.removeChild(p);
        }}, {duration_ms});
    }})();
    </script>
    """
    st.components.v1.html(js, height=0, scrolling=False)


# ======================================================
# TAB 2 — SMART MONEY + GEX
# ======================================================
def render():
    # Inject sliding popup CSS + rail once per render
    _inject_popup_css()

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
        _show_popup("⚡ GEX FLIP → TRENDING MODE", kind="danger", icon="🔴")
        st.session_state.alert_log.append({"time": datetime.now(IST).strftime("%H:%M:%S"),
                                           "msg": "GEX Flip → Trending", "type": "DANGER"})
    elif prev_gex < 0 and total_net_gex > 0:
        _show_popup("⚡ GEX FLIP → MEAN REVERSION MODE", kind="success", icon="🟢")
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
    # CE vs PE OI COMPARISON — Single chart + strike-wise Net Bias table
    # ======================================================
    st.markdown("---")
    section_header("CE vs PE OI per Strike", "OI Δ chart + strike-wise CE OI vs PE OI Net Bias — institutional conviction at each strike")
    try:
        oc_cmp = oc2.copy()
        if all(c in oc_cmp.columns for c in ["CE_OI", "PE_OI", "CE_OI_Change", "PE_OI_Change"]):
            atm_cmp     = get_atm_strike(spot2, sel_idx2)
            strikes_cmp = sorted(oc_cmp["Strike"].unique())
            if atm_cmp in strikes_cmp:
                ai_cmp    = strikes_cmp.index(atm_cmp)
                focus_cmp = strikes_cmp[max(0, ai_cmp - 7): ai_cmp + 8]
            else:
                focus_cmp = strikes_cmp[:15]

            oc_cmp = oc_cmp[oc_cmp["Strike"].isin(focus_cmp)].sort_values("Strike")

            # ── Tab view: OI Δ chart | Total OI chart ──────────────────────
            tab_oi_delta, tab_oi_total = st.tabs(["📊 OI Change (Δ) by Strike", "📈 Total CE OI vs PE OI"])

            with tab_oi_delta:
                cmp_fig = go.Figure()
                ce_colors = ["#ff3d57" if v >= 0 else "#ff8a80" for v in oc_cmp["CE_OI_Change"]]
                pe_colors = ["#00e676" if v >= 0 else "#69f0ae" for v in oc_cmp["PE_OI_Change"]]
                cmp_fig.add_bar(
                    x=oc_cmp["Strike"], y=oc_cmp["CE_OI_Change"] / 1e3,
                    name="CE OI Δ", marker_color=ce_colors, opacity=0.9,
                    text=[f"{v/1e3:+.1f}K" for v in oc_cmp["CE_OI_Change"]],
                    textposition="outside", textfont=dict(size=9, color="#ff3d57"),
                )
                cmp_fig.add_bar(
                    x=oc_cmp["Strike"], y=oc_cmp["PE_OI_Change"] / 1e3,
                    name="PE OI Δ", marker_color=pe_colors, opacity=0.9,
                    text=[f"{v/1e3:+.1f}K" for v in oc_cmp["PE_OI_Change"]],
                    textposition="outside", textfont=dict(size=9, color="#00e676"),
                )
                cmp_fig.add_vline(
                    x=float(atm_cmp), line_color="#00d4ff", line_dash="dot", line_width=1.5,
                    annotation_text="ATM", annotation_font_color="#00d4ff", annotation_font_size=10
                )
                cmp_fig.add_hline(y=0, line_color="#3a6080", line_width=0.8)
                cmp_fig.update_layout(
                    barmode="group", height=320,
                    paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
                    font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
                    margin=dict(l=40, r=20, t=20, b=40),
                    xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d", title="Strike"),
                    yaxis=dict(gridcolor="#1e3040", title="OI Change (000s)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7fa8c8"),
                )
                st.plotly_chart(cmp_fig, use_container_width=True, key="oi_delta_chart")

            with tab_oi_total:
                tot_fig = go.Figure()
                tot_fig.add_bar(
                    x=oc_cmp["Strike"], y=oc_cmp["CE_OI"] / 1e5,
                    name="CE OI", marker_color="#ff3d57", opacity=0.85,
                )
                tot_fig.add_bar(
                    x=oc_cmp["Strike"], y=oc_cmp["PE_OI"] / 1e5,
                    name="PE OI", marker_color="#00e676", opacity=0.85,
                )
                tot_fig.add_vline(
                    x=float(atm_cmp), line_color="#00d4ff", line_dash="dot", line_width=1.5,
                    annotation_text="ATM", annotation_font_color="#00d4ff", annotation_font_size=10
                )
                tot_fig.update_layout(
                    barmode="group", height=320,
                    paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
                    font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
                    margin=dict(l=40, r=20, t=20, b=40),
                    xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d", title="Strike"),
                    yaxis=dict(gridcolor="#1e3040", title="OI (Lakhs)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7fa8c8"),
                )
                st.plotly_chart(tot_fig, use_container_width=True, key="oi_total_chart")

            # ── Net Bias: based on CE_OI vs PE_OI (total OI dominance) ──────
            # Logic: whichever side has more OI dominates conviction at that strike
            # CE OI > PE OI  → more call writing / bearish hedging → BEARISH
            # PE OI > CE OI  → more put writing / bullish support   → BULLISH
            # Difference < 15% of total → NEUTRAL
            def _net_bias_oi(ce_oi, pe_oi, ce_chg, pe_chg):
                ce_oi = abs(float(ce_oi)); pe_oi = abs(float(pe_oi))
                total_oi = ce_oi + pe_oi
                if total_oi == 0:
                    return "⚪ NEUTRAL"
                oi_diff_pct = abs(ce_oi - pe_oi) / max(total_oi, 1) * 100
                # OI dominance
                if oi_diff_pct < 15:
                    oi_verdict = "⚪ NEUTRAL"
                elif ce_oi > pe_oi:
                    oi_verdict = "🔴 BEARISH"  # CE OI dominates → resistance / call selling
                else:
                    oi_verdict = "🟢 BULLISH"  # PE OI dominates → support / put selling
                return oi_verdict

            rows_cmp = []
            for _, r in oc_cmp.iterrows():
                atm_tag   = " ★" if r["Strike"] == atm_cmp else ""
                money     = "ATM" if r["Strike"] == atm_cmp else ("ITM" if r["Strike"] < spot2 else "OTM")
                ce_oi     = float(r["CE_OI"])
                pe_oi     = float(r["PE_OI"])
                ce_oi_chg = float(r["CE_OI_Change"])
                pe_oi_chg = float(r["PE_OI_Change"])
                total_oi  = ce_oi + pe_oi
                ce_oi_pct = ce_oi / max(total_oi, 1) * 100
                pe_oi_pct = 100 - ce_oi_pct
                rows_cmp.append({
                    "Strike":    f"{int(r['Strike']):,}{atm_tag}",
                    "CE OI":     f"{ce_oi/1e5:.2f}L",
                    "CE OI Δ":   f"{ce_oi_chg/1e3:+.1f}K",
                    "CE %":      f"{ce_oi_pct:.0f}%",
                    "PE %":      f"{pe_oi_pct:.0f}%",
                    "PE OI Δ":   f"{pe_oi_chg/1e3:+.1f}K",
                    "PE OI":     f"{pe_oi/1e5:.2f}L",
                    "Money":     money,
                    "Net Bias":  _net_bias_oi(ce_oi, pe_oi, ce_oi_chg, pe_oi_chg),
                })

            df_cmp = pd.DataFrame(rows_cmp)

            # ── Summary metrics ──────────────────────────────────────────────
            total_ce_cmp  = oc_cmp["CE_OI_Change"].sum()
            total_pe_cmp  = oc_cmp["PE_OI_Change"].sum()
            total_ce_oi   = oc_cmp["CE_OI"].sum()
            total_pe_oi   = oc_cmp["PE_OI"].sum()
            pcr_cmp       = total_pe_oi / max(total_ce_oi, 1)
            bull_count    = sum(1 for r in rows_cmp if "BULLISH" in r["Net Bias"])
            bear_count    = sum(1 for r in rows_cmp if "BEARISH" in r["Net Bias"])
            neut_count    = len(rows_cmp) - bull_count - bear_count
            if bull_count > bear_count:
                overall_bias = "🟢 BULLISH"
            elif bear_count > bull_count:
                overall_bias = "🔴 BEARISH"
            else:
                overall_bias = "⚪ NEUTRAL"
            ob_color      = "#00e676" if "BULL" in overall_bias else ("#ff3d57" if "BEAR" in overall_bias else "#7fa8c8")

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ff3d57;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">TOTAL CE OI Δ</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;
                        color:{'#ff3d57' if total_ce_cmp > 0 else '#00e676'};">{total_ce_cmp/1e3:+.0f}K</div>
                    </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #00e676;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">TOTAL PE OI Δ</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;
                        color:{'#00e676' if total_pe_cmp > 0 else '#ff3d57'};">{total_pe_cmp/1e3:+.0f}K</div>
                    </div>""", unsafe_allow_html=True)
            with mc3:
                pcr_c = "#00e676" if pcr_cmp > 1.2 else "#ff3d57" if pcr_cmp < 0.8 else "#ffd600"
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid {pcr_c};
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">PCR (OI)</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:{pcr_c};">{pcr_cmp:.2f}</div>
                    <div style="font-size:10px;color:#3a6080;">{'Bullish' if pcr_cmp > 1.2 else 'Bearish' if pcr_cmp < 0.8 else 'Neutral'}</div>
                    </div>""", unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ffd600;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">STRIKE BIAS COUNT</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700;color:#ffd600;margin-top:3px;">
                        🟢 {bull_count} &nbsp; 🔴 {bear_count} &nbsp; ⚪ {neut_count}</div>
                    </div>""", unsafe_allow_html=True)
            with mc5:
                st.markdown(f"""<div style="background:#0d1117;border:1px solid {ob_color};border-left:3px solid {ob_color};
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">OVERALL BIAS</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:700;color:{ob_color};margin-top:3px;">{overall_bias}</div>
                    </div>""", unsafe_allow_html=True)

            # ── OI split bar ──────────────────────────────────────────────────
            ce_pct_split = total_ce_oi / max(total_ce_oi + total_pe_oi, 1) * 100
            pe_pct_split = 100 - ce_pct_split
            st.markdown(f"""
            <div style="margin:10px 0 4px 0;font-family:'Barlow Condensed',sans-serif;
                        font-size:9px;letter-spacing:1.5px;color:#3a6080;">
              TOTAL OI SPLIT — CE {ce_pct_split:.1f}% &nbsp;|&nbsp; PE {pe_pct_split:.1f}%
            </div>
            <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:12px;">
              <div style="width:{ce_pct_split:.1f}%;background:#ff3d57;"></div>
              <div style="width:{pe_pct_split:.1f}%;background:#00e676;"></div>
            </div>""", unsafe_allow_html=True)

            # ── Strike-wise table ────────────────────────────────────────────
            st.dataframe(
                df_cmp,
                use_container_width=True,
                height=min(35 * len(df_cmp) + 38, 550),
                column_config={
                    "Strike":   st.column_config.TextColumn("Strike",   width=95),
                    "CE OI":    st.column_config.TextColumn("CE OI",    width=80),
                    "CE OI Δ":  st.column_config.TextColumn("CE OI Δ", width=85),
                    "CE %":     st.column_config.TextColumn("CE %",     width=55),
                    "PE %":     st.column_config.TextColumn("PE %",     width=55),
                    "PE OI Δ":  st.column_config.TextColumn("PE OI Δ", width=85),
                    "PE OI":    st.column_config.TextColumn("PE OI",    width=80),
                    "Money":    st.column_config.TextColumn("Money",    width=55),
                    "Net Bias": st.column_config.TextColumn("Net Bias", width=100),
                }
            )
        else:
            st.info("CE_OI, PE_OI, CE_OI_Change, PE_OI_Change columns required.")
    except Exception as e:
        st.warning(f"CE vs PE OI comparison error: {e}")

    # ======================================================
    # OI SPIKE CHANGE DETECTOR + STRIKE CONTINUE
    # ======================================================
    st.markdown("---")
    section_header("🔥 OI Spike Change Detector", "Strikes with sudden large OI build-up or unwinding — smart money footprint")
    try:
        oc_ois = oc2.copy()
        if all(c in oc_ois.columns for c in ["CE_OI_Change", "PE_OI_Change", "CE_OI", "PE_OI"]):

            # ── Thresholds ─────────────────────────────────────────────────────
            ce_oi_mean = oc_ois["CE_OI_Change"].abs().mean()
            pe_oi_mean = oc_ois["PE_OI_Change"].abs().mean()
            oi_spike_mult = 2.5  # 2.5× avg OI change = spike

            ce_oi_spikes = oc_ois[oc_ois["CE_OI_Change"].abs() >= ce_oi_mean * oi_spike_mult].copy()
            pe_oi_spikes = oc_ois[oc_ois["PE_OI_Change"].abs() >= pe_oi_mean * oi_spike_mult].copy()

            atm_ois = get_atm_strike(spot2, sel_idx2)

            # ── Fire POPUP toasts for large OI spikes ──────────────────────────
            for _, r in ce_oi_spikes.iterrows():
                mult_oi = abs(float(r["CE_OI_Change"])) / max(ce_oi_mean, 1)
                direction = "📈 BUILD-UP" if float(r["CE_OI_Change"]) > 0 else "📉 UNWIND"
                if mult_oi >= 3.0:
                    _show_popup(f"🚨 CE OI SPIKE {int(r['Strike'])} — {direction} ({mult_oi:.1f}× avg)", kind="danger", icon="🔴")
                    if "alert_log" in st.session_state:
                        st.session_state.alert_log.append({
                            "time": datetime.now(IST).strftime("%H:%M:%S"),
                            "msg": f"CE OI Spike {int(r['Strike'])} {direction} ({mult_oi:.1f}×)",
                            "type": "DANGER"
                        })
            for _, r in pe_oi_spikes.iterrows():
                mult_oi = abs(float(r["PE_OI_Change"])) / max(pe_oi_mean, 1)
                direction = "📈 BUILD-UP" if float(r["PE_OI_Change"]) > 0 else "📉 UNWIND"
                if mult_oi >= 3.0:
                    _show_popup(f"🚨 PE OI SPIKE {int(r['Strike'])} — {direction} ({mult_oi:.1f}× avg)", kind="success", icon="🟢")
                    if "alert_log" in st.session_state:
                        st.session_state.alert_log.append({
                            "time": datetime.now(IST).strftime("%H:%M:%S"),
                            "msg": f"PE OI Spike {int(r['Strike'])} {direction} ({mult_oi:.1f}×)",
                            "type": "INFO"
                        })

            # ── STRIKE CONTINUE detection ──────────────────────────────────────
            # "Strike Continue" = consecutive strikes where OI is building in same direction
            def _detect_strike_continue(df_sorted, col_chg, threshold_mult, mean_chg):
                """Find runs of consecutive strikes with OI building in same direction."""
                runs = []
                current_run = []
                prev_dir = None
                for _, row_sc in df_sorted.iterrows():
                    chg = float(row_sc[col_chg])
                    if abs(chg) < mean_chg * 0.5:  # too small, treat as neutral
                        if len(current_run) >= 2:
                            runs.append(current_run[:])
                        current_run = []
                        prev_dir = None
                        continue
                    direction = "UP" if chg > 0 else "DOWN"
                    if direction == prev_dir:
                        current_run.append({"strike": int(row_sc["Strike"]), "chg": chg, "dir": direction})
                    else:
                        if len(current_run) >= 2:
                            runs.append(current_run[:])
                        current_run = [{"strike": int(row_sc["Strike"]), "chg": chg, "dir": direction}]
                        prev_dir = direction
                if len(current_run) >= 2:
                    runs.append(current_run)
                return runs

            oc_ois_sorted = oc_ois.sort_values("Strike")
            ce_runs = _detect_strike_continue(oc_ois_sorted, "CE_OI_Change", oi_spike_mult, ce_oi_mean)
            pe_runs = _detect_strike_continue(oc_ois_sorted, "PE_OI_Change", oi_spike_mult, pe_oi_mean)

            # Fire toast for strong continues (3+ strikes)
            for run in ce_runs:
                if len(run) >= 3:
                    strikes_str = "→".join([str(r["strike"]) for r in run])
                    direction_lbl = "BUILD-UP 📈" if run[0]["dir"] == "UP" else "UNWIND 📉"
                    _show_popup(f"⚡ CE CONTINUE {direction_lbl}: {strikes_str}", kind="danger", icon="🔴")
                    if "alert_log" in st.session_state:
                        st.session_state.alert_log.append({
                            "time": datetime.now(IST).strftime("%H:%M:%S"),
                            "msg": f"CE Continue {direction_lbl} {run[0]['strike']}→{run[-1]['strike']}",
                            "type": "DANGER"
                        })
            for run in pe_runs:
                if len(run) >= 3:
                    strikes_str = "→".join([str(r["strike"]) for r in run])
                    direction_lbl = "BUILD-UP 📈" if run[0]["dir"] == "UP" else "UNWIND 📉"
                    _show_popup(f"⚡ PE CONTINUE {direction_lbl}: {strikes_str}", kind="success", icon="🟢")
                    if "alert_log" in st.session_state:
                        st.session_state.alert_log.append({
                            "time": datetime.now(IST).strftime("%H:%M:%S"),
                            "msg": f"PE Continue {direction_lbl} {run[0]['strike']}→{run[-1]['strike']}",
                            "type": "INFO"
                        })

            # ── Summary cards ─────────────────────────────────────────────────
            oi_sc1, oi_sc2, oi_sc3, oi_sc4 = st.columns(4)
            with oi_sc1:
                total_ce_spike = len(ce_oi_spikes)
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ff3d57;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">CE OI SPIKES</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#ff3d57;">{total_ce_spike}</div>
                    <div style="font-size:10px;color:#3a6080;">≥{oi_spike_mult:.0f}× avg OI change</div>
                    </div>""", unsafe_allow_html=True)
            with oi_sc2:
                total_pe_spike = len(pe_oi_spikes)
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #00e676;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">PE OI SPIKES</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#00e676;">{total_pe_spike}</div>
                    <div style="font-size:10px;color:#3a6080;">≥{oi_spike_mult:.0f}× avg OI change</div>
                    </div>""", unsafe_allow_html=True)
            with oi_sc3:
                ce_continues = len([r for r in ce_runs if len(r) >= 2])
                best_ce_run  = max((len(r) for r in ce_runs), default=0)
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #ff8c00;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">CE CONTINUE RUNS</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#ff8c00;">{ce_continues}</div>
                    <div style="font-size:10px;color:#3a6080;">Longest: {best_ce_run} strikes</div>
                    </div>""", unsafe_allow_html=True)
            with oi_sc4:
                pe_continues = len([r for r in pe_runs if len(r) >= 2])
                best_pe_run  = max((len(r) for r in pe_runs), default=0)
                st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e3040;border-left:3px solid #00d4ff;
                    border-radius:3px;padding:10px 14px;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:2px;color:#7fa8c8;">PE CONTINUE RUNS</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:#00d4ff;">{pe_continues}</div>
                    <div style="font-size:10px;color:#3a6080;">Longest: {best_pe_run} strikes</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # ── Strike Continue table ──────────────────────────────────────────
            if ce_runs or pe_runs:
                sc_tab1, sc_tab2 = st.tabs(["📊 CE Strike Continue", "📊 PE Strike Continue"])

                with sc_tab1:
                    if ce_runs:
                        ce_run_rows = []
                        for run in sorted(ce_runs, key=lambda x: -len(x)):
                            direction_icon = "📈 BUILD-UP" if run[0]["dir"] == "UP" else "📉 UNWIND"
                            strike_range   = f"{run[0]['strike']} → {run[-1]['strike']}"
                            total_chg      = sum(r["chg"] for r in run)
                            strength       = "🔥 STRONG" if len(run) >= 3 else "⚡ MODERATE"
                            ce_run_rows.append({
                                "CE Strikes": strike_range,
                                "Count": len(run),
                                "Direction": direction_icon,
                                "Total OI Δ": f"{total_chg/1e3:+.1f}K",
                                "Strength": strength,
                                "Meaning": "Resistance building" if run[0]["dir"] == "UP" else "CE Unwinding (Bullish)",
                            })
                        st.dataframe(pd.DataFrame(ce_run_rows), use_container_width=True,
                                     height=min(35 * len(ce_run_rows) + 38, 320))
                    else:
                        st.info("No CE strike continue patterns detected.")

                with sc_tab2:
                    if pe_runs:
                        pe_run_rows = []
                        for run in sorted(pe_runs, key=lambda x: -len(x)):
                            direction_icon = "📈 BUILD-UP" if run[0]["dir"] == "UP" else "📉 UNWIND"
                            strike_range   = f"{run[0]['strike']} → {run[-1]['strike']}"
                            total_chg      = sum(r["chg"] for r in run)
                            strength       = "🔥 STRONG" if len(run) >= 3 else "⚡ MODERATE"
                            pe_run_rows.append({
                                "PE Strikes": strike_range,
                                "Count": len(run),
                                "Direction": direction_icon,
                                "Total OI Δ": f"{total_chg/1e3:+.1f}K",
                                "Strength": strength,
                                "Meaning": "Support building" if run[0]["dir"] == "UP" else "PE Unwinding (Bearish)",
                            })
                        st.dataframe(pd.DataFrame(pe_run_rows), use_container_width=True,
                                     height=min(35 * len(pe_run_rows) + 38, 320))
                    else:
                        st.info("No PE strike continue patterns detected.")

            # ── OI Spike Change Table with ▲▼ arrows ──────────────────────────
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
            st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:12px;
            letter-spacing:2px;color:#ff8c00;margin-bottom:8px;">⚡ OI SPIKE STRIKES — INCREASE ▲ &amp; DECREASE ▼</div>""",
            unsafe_allow_html=True)

            oi_spike_rows = []
            all_spike_strikes = set(ce_oi_spikes["Strike"].tolist()) | set(pe_oi_spikes["Strike"].tolist())
            for strike_val in sorted(all_spike_strikes):
                row_s = oc_ois[oc_ois["Strike"] == strike_val]
                if row_s.empty:
                    continue
                row_s = row_s.iloc[0]
                ce_chg_s = float(row_s["CE_OI_Change"])
                pe_chg_s = float(row_s["PE_OI_Change"])
                ce_mult_s = abs(ce_chg_s) / max(ce_oi_mean, 1)
                pe_mult_s = abs(pe_chg_s) / max(pe_oi_mean, 1)
                atm_tag = " ★ATM" if strike_val == atm_ois else ""
                money = "ATM" if strike_val == atm_ois else ("ITM" if strike_val < spot2 else "OTM")

                # OI direction arrow icons
                ce_arrow = "▲ INC" if ce_chg_s > 0 else ("▼ DEC" if ce_chg_s < 0 else "— FLAT")
                pe_arrow = "▲ INC" if pe_chg_s > 0 else ("▼ DEC" if pe_chg_s < 0 else "— FLAT")
                is_ce_spike = strike_val in ce_oi_spikes["Strike"].values
                is_pe_spike = strike_val in pe_oi_spikes["Strike"].values

                oi_spike_rows.append({
                    "Strike":     f"{int(strike_val):,}{atm_tag}",
                    "Money":      money,
                    "CE OI Δ":    f"{ce_chg_s/1e3:+.1f}K",
                    "CE Dir":     f"{'🔴' if ce_chg_s > 0 else '🟢'} {ce_arrow}",
                    "CE Spike":   f"🚨 {ce_mult_s:.1f}×" if is_ce_spike else "—",
                    "PE OI Δ":    f"{pe_chg_s/1e3:+.1f}K",
                    "PE Dir":     f"{'🟢' if pe_chg_s > 0 else '🔴'} {pe_arrow}",
                    "PE Spike":   f"🚨 {pe_mult_s:.1f}×" if is_pe_spike else "—",
                    "Signal":     "🔴 BEARISH" if (is_ce_spike and ce_chg_s > 0 and ce_chg_s > pe_chg_s)
                                  else "🟢 BULLISH" if (is_pe_spike and pe_chg_s > 0 and pe_chg_s > ce_chg_s)
                                  else "⚪ WATCH",
                })

            if oi_spike_rows:
                df_oi_spk = pd.DataFrame(oi_spike_rows)
                st.dataframe(
                    df_oi_spk,
                    use_container_width=True,
                    height=min(35 * len(df_oi_spk) + 38, 420),
                    column_config={
                        "Strike":   st.column_config.TextColumn("Strike",   width=110),
                        "Money":    st.column_config.TextColumn("Money",    width=55),
                        "CE OI Δ":  st.column_config.TextColumn("CE OI Δ", width=85),
                        "CE Dir":   st.column_config.TextColumn("CE Dir",   width=95),
                        "CE Spike": st.column_config.TextColumn("CE Spike", width=80),
                        "PE OI Δ":  st.column_config.TextColumn("PE OI Δ", width=85),
                        "PE Dir":   st.column_config.TextColumn("PE Dir",   width=95),
                        "PE Spike": st.column_config.TextColumn("PE Spike", width=80),
                        "Signal":   st.column_config.TextColumn("Signal",   width=100),
                    }
                )

            # ── OI Change chart with spike markers ────────────────────────────
            oi_spk_fig = go.Figure()
            oc_ois_ch = oc_ois.sort_values("Strike")
            ce_bar_colors = [
                "#ff1744" if abs(v) >= ce_oi_mean * oi_spike_mult else
                "#ff3d57" if v > 0 else "#ff8a80"
                for v in oc_ois_ch["CE_OI_Change"]
            ]
            pe_bar_colors = [
                "#00c853" if abs(v) >= pe_oi_mean * oi_spike_mult else
                "#00e676" if v > 0 else "#69f0ae"
                for v in oc_ois_ch["PE_OI_Change"]
            ]
            oi_spk_fig.add_bar(
                x=oc_ois_ch["Strike"], y=oc_ois_ch["CE_OI_Change"] / 1e3,
                name="CE OI Δ", marker_color=ce_bar_colors, opacity=0.9,
                text=[f"{v/1e3:+.1f}K" for v in oc_ois_ch["CE_OI_Change"]],
                textposition="outside", textfont=dict(size=8, color="#ff3d57"),
            )
            oi_spk_fig.add_bar(
                x=oc_ois_ch["Strike"], y=oc_ois_ch["PE_OI_Change"] / 1e3,
                name="PE OI Δ", marker_color=pe_bar_colors, opacity=0.9,
                text=[f"{v/1e3:+.1f}K" for v in oc_ois_ch["PE_OI_Change"]],
                textposition="outside", textfont=dict(size=8, color="#00e676"),
            )
            # Spike threshold lines
            oi_spk_fig.add_hline(y=ce_oi_mean * oi_spike_mult / 1e3,
                                  line_color="#ff8c00", line_dash="dash", line_width=1.2,
                                  annotation_text=f"CE Spike {oi_spike_mult:.0f}× threshold",
                                  annotation_font_color="#ff8c00", annotation_font_size=9)
            oi_spk_fig.add_hline(y=-ce_oi_mean * oi_spike_mult / 1e3,
                                  line_color="#ff8c00", line_dash="dash", line_width=1.2)
            oi_spk_fig.add_hline(y=pe_oi_mean * oi_spike_mult / 1e3,
                                  line_color="#00d4ff", line_dash="dash", line_width=1.2,
                                  annotation_text=f"PE Spike {oi_spike_mult:.0f}× threshold",
                                  annotation_font_color="#00d4ff", annotation_font_size=9)
            oi_spk_fig.add_hline(y=-pe_oi_mean * oi_spike_mult / 1e3,
                                  line_color="#00d4ff", line_dash="dash", line_width=1.2)
            oi_spk_fig.add_hline(y=0, line_color="#3a6080", line_width=0.8)
            oi_spk_fig.add_vline(x=float(atm_ois), line_color="#00d4ff", line_dash="dot", line_width=1.5,
                                  annotation_text="ATM", annotation_font_color="#00d4ff", annotation_font_size=10)
            # Mark spike strikes with scatter annotations
            for _, r in ce_oi_spikes.iterrows():
                oi_spk_fig.add_annotation(
                    x=float(r["Strike"]), y=float(r["CE_OI_Change"]) / 1e3,
                    text="🚨", showarrow=True, arrowhead=2,
                    arrowcolor="#ff1744", font=dict(size=13),
                    ay=-20 if float(r["CE_OI_Change"]) > 0 else 20
                )
            for _, r in pe_oi_spikes.iterrows():
                oi_spk_fig.add_annotation(
                    x=float(r["Strike"]), y=float(r["PE_OI_Change"]) / 1e3,
                    text="🚨", showarrow=True, arrowhead=2,
                    arrowcolor="#00c853", font=dict(size=13),
                    ay=-20 if float(r["PE_OI_Change"]) > 0 else 20
                )
            oi_spk_fig.update_layout(
                barmode="group", height=340,
                paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
                font=dict(family="JetBrains Mono", color="#7fa8c8", size=11),
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(gridcolor="#1e3040", tickangle=-45, tickformat="d", title="Strike"),
                yaxis=dict(gridcolor="#1e3040", title="OI Change (000s)"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7fa8c8"),
            )
            st.plotly_chart(oi_spk_fig, use_container_width=True, key="oi_spike_chart")

        else:
            st.info("CE_OI_Change and PE_OI_Change columns required for OI spike detection.")
    except Exception as e:
        st.warning(f"OI spike detector error: {e}")

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
                    _show_popup(f"🚨 Vol Spike {hs['Type']} {hs['Strike']} — {hs['vs Avg']} avg", kind="warning", icon="🔔")
                    if "alert_log" in st.session_state:
                        st.session_state.alert_log.append({
                            "time": datetime.now(IST).strftime("%H:%M:%S"),
                            "msg":  f"Vol Spike {hs['Type']} {hs['Strike']} ({hs['vs Avg']})",
                            "type": "DANGER"
                        })

                # ── Total OI comparison with 10% rule ────────────────────
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

    if not oc_sb.empty and sel_sb and spot_sb:
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

        row_sb       = oc_sb[oc_sb['Strike'] == strike_sb].iloc[0]
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

        if MARKET_OPEN:
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
        else:
            session_label = st.session_state.get("session_label", "CLOSED")
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