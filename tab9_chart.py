import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from config import (TF_OPTIONS, TF_RESAMPLE, ACCESS_TOKEN, IST, now_ist, now_ist_dt, MARKET_OPEN,
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
# TAB 9 — LIVE CHART · ORDER FLOW
# ======================================================
def render():
    st.session_state["active_tab_key"] = "📊 CHART"
    section_header("Live Chart  ·  Order Flow  ·  Liquidity Sweep",
                   "1m–1H  ·  Delta  ·  Cum Delta  ·  Sweeps  ·  Order Blocks  ·  FVG  ·  BOS/CHoCH")

    oc_t9   = st.session_state.get("current_option_chain", pd.DataFrame())
    spot_t9 = st.session_state.get("current_spot_price")
    sel_t9  = st.session_state.get("current_selected_index")

    if spot_t9 is None or sel_t9 is None:
        st.warning("⏳ Load option chain from Tab 1 first."); return

    # S&R levels
    m_res9, m_sup9 = None, None
    if not oc_t9.empty:
        try:
            _ce9 = oc_t9.sort_values("CE_OI", ascending=False)
            _pe9 = oc_t9.sort_values("PE_OI", ascending=False)
            _r9  = [s for s in _ce9["Strike"].tolist() if s > spot_t9]
            _s9  = [s for s in _pe9["Strike"].tolist() if s < spot_t9]
            m_res9 = min(_r9) if _r9 else None
            m_sup9 = max(_s9) if _s9 else None
        except Exception: pass

    # ── Timeframe Buttons — NO st.rerun(), use session_state directly ────
    tf_labels9 = list(TF_OPTIONS.keys())
    if "chart_tf9" not in st.session_state:
        st.session_state.chart_tf9 = "5m"

    tf_row_html = '<div style="display:flex;gap:4px;align-items:center;margin-bottom:10px;">'
    tf_row_html += '<span style="font-family:Barlow Condensed,sans-serif;font-size:10px;letter-spacing:1.5px;color:#3a6080;margin-right:6px;">TIMEFRAME</span>'
    for _tl in tf_labels9:
        _is_sel = st.session_state.chart_tf9 == _tl
        _bg  = "#ff8c00" if _is_sel else "transparent"
        _clr = "#000" if _is_sel else "#7fa8c8"
        _bdr = "#ff8c00"
        tf_row_html += f'<span style="background:{_bg};border:1px solid {_bdr};color:{_clr};font-family:Barlow Condensed,sans-serif;font-size:13px;font-weight:700;padding:4px 12px;border-radius:2px;cursor:pointer;">{_tl}</span>'
    tf_row_html += "</div>"
    st.markdown(tf_row_html, unsafe_allow_html=True)

    # Actual TF selector via pills (Streamlit pills avoid st.rerun)
    _tl_idx = tf_labels9.index(st.session_state.chart_tf9) if st.session_state.chart_tf9 in tf_labels9 else 0
    _new_tf = st.selectbox("", tf_labels9, index=_tl_idx, key="chart_tf9_sel",
                            label_visibility="collapsed")
    if _new_tf != st.session_state.chart_tf9:
        st.session_state.chart_tf9 = _new_tf
        # NO st.rerun() — the selectbox change itself triggers a targeted rerun

    sel_tf9     = st.session_state.chart_tf9
    sel_tf9_api = TF_OPTIONS[sel_tf9]

    # ── Indicator Toggles — granular on/off ───────────────────────────────
    st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
        letter-spacing:1.5px;color:#3a6080;margin-bottom:4px;">INDICATORS</div>""",
        unsafe_allow_html=True)
    ind_c1, ind_c2, ind_c3, ind_c4, ind_c5, ind_c6, ind_c7, ind_c8, ind_c9, ind_c10 = st.columns(10)
    with ind_c1:  show_ema9    = st.toggle("EMA9",    value=True,  key="ind_ema9")
    with ind_c2:  show_ema21   = st.toggle("EMA21",   value=True,  key="ind_ema21")
    with ind_c3:  show_ema50   = st.toggle("EMA50",   value=True,  key="ind_ema50")
    with ind_c4:  show_vwap    = st.toggle("VWAP",    value=True,  key="ind_vwap")
    with ind_c5:  show_bb      = st.toggle("BB",      value=True,  key="ind_bb")
    with ind_c6:  show_sweeps  = st.toggle("Sweeps",  value=True,  key="ov_sw9")
    with ind_c7:  show_ob      = st.toggle("OB",      value=True,  key="ov_ob9")
    with ind_c8:  show_fvg     = st.toggle("FVG",     value=True,  key="ov_fvg9")
    with ind_c9:  show_bos     = st.toggle("BOS",     value=True,  key="ov_bos9")
    with ind_c10: show_delta   = st.toggle("Δ Row",   value=True,  key="ov_dl9")
    # Second row of toggles
    ind_c11, ind_c12, ind_c13, ind_c14 = st.columns([1,1,1,7])
    with ind_c11: show_eqlev   = st.toggle("EqH/L",  value=True,  key="ov_eq9")
    with ind_c12: show_rsi     = st.toggle("RSI",    value=True,  key="ind_rsi9")
    with ind_c13: show_macd    = st.toggle("MACD",   value=True,  key="ind_macd9")

    # ── Fetch & compute ───────────────────────────────────────────────────
    raw_df9        = fetch_intraday_candles(sel_t9, sel_tf9_api)
    tech_df9, ts9  = compute_technicals(raw_df9)

    if tech_df9.empty:
        st.warning("⏳ No candle data available."); return

    of_df9 = compute_order_flow(tech_df9)
    of_sum9 = get_order_flow_summary(of_df9)

    # Run detectors
    sweeps_bsl9, sweeps_ssl9, eq_highs9, eq_lows9 = detect_liquidity_sweeps(of_df9)
    bull_obs9, bear_obs9 = detect_order_blocks(of_df9)
    bull_fvg9, bear_fvg9 = detect_fvg(of_df9)
    bos9, choch9         = detect_bos_choch(of_df9)

    # ── Multi-TF Signal Matrix ────────────────────────────────────────────
    st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
        letter-spacing:2px;color:#7fa8c8;margin:10px 0 6px 0;">⚡ MULTI-TIMEFRAME SIGNAL MATRIX</div>""",
        unsafe_allow_html=True)
    _mhtml = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;">'
    for _tfl2, _tfa2 in TF_OPTIONS.items():
        try:
            _, _ms2 = compute_technicals(fetch_intraday_candles(sel_t9, _tfa2))
            if not _ms2: raise ValueError
            _r2  = _ms2.get("rsi14",50); _e2 = _ms2.get("ema_trend","MIXED")
            _m2  = _ms2.get("macd_cross","BEARISH"); _v2 = _ms2.get("price_vs_vwap","BELOW")
            _b2  = sum([_e2=="BULLISH",_m2=="BULLISH",_v2=="ABOVE",30<_r2<65])
            _br2 = sum([_e2=="BEARISH",_m2=="BEARISH",_v2=="BELOW",_r2>70 or _r2<30])
            if _b2>=3:   _la2,_c2,_i2 = "BULL","#00e676","▲"
            elif _br2>=3: _la2,_c2,_i2 = "BEAR","#ff3d57","▼"
            elif _b2==2:  _la2,_c2,_i2 = "MILD↑","#7fc97f","↗"
            elif _br2==2: _la2,_c2,_i2 = "MILD↓","#e06060","↘"
            else:         _la2,_c2,_i2 = "NEUT","#ffd600","–"
            _sb2 = "border-width:2px;" if _tfl2==sel_tf9 else ""
            _mhtml += f"""<div style="background:#0d1117;border:1px solid {_c2};{_sb2}border-radius:3px;
                padding:8px 12px;min-width:78px;text-align:center;">
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;
                          color:#e8f4ff;">{_tfl2}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;
                          color:{_c2};margin:2px 0;">{_i2} {_la2}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7fa8c8;">
                RSI {_r2:.0f}</div>
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:9px;color:#3a6080;
                          margin-top:2px;">{_e2[:4]} · {_m2[:4]}</div>
            </div>"""
        except Exception:
            _mhtml += f"""<div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;
                padding:8px 12px;min-width:78px;text-align:center;">
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:14px;color:#3a6080;">{_tfl2}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3a6080;">N/A</div>
            </div>"""
    _mhtml += "</div>"
    st.markdown(_mhtml, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # 📊 ORDER FLOW SUMMARY CARDS
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    section_header("Order Flow Dashboard",
                   "Delta · Cum Delta · Buy/Sell pressure · Absorption · Divergence")

    _prs_c = "#00e676" if of_sum9.get("pressure","") == "BUY DOMINANT" else "#ff3d57"
    _dlt_c = "#00e676" if of_sum9.get("delta_trend") == "RISING" else "#ff3d57"
    _buy_p = of_sum9.get("buy_pct", 50)
    _sel_p = 100 - _buy_p

    metrics_row(
        metric_card("PRESSURE",   of_sum9.get("pressure","—"), "", _prs_c) +
        metric_card("NET DELTA",  f"{of_sum9.get('net_delta',0):+,.0f}",
                    f"Cum: {of_sum9.get('cum_delta',0):+,.0f}", _dlt_c) +
        metric_card("BUY VOL %",  f"{_buy_p:.1f}%",
                    f"Sell: {_sel_p:.1f}%", "#00e676") +
        metric_card("DELTA TREND",of_sum9.get("delta_trend","—"), "", _dlt_c) +
        metric_card("ABSORPTION", f"{of_sum9.get('absorption_candles',0)} candles",
                    "High vol + small body", "#ffd600") +
        metric_card("BULL DIV",   f"{of_sum9.get('bull_divergence',0)}",
                    "Up price, neg delta", "#c084fc") +
        metric_card("BEAR DIV",   f"{of_sum9.get('bear_divergence',0)}",
                    "Down price, pos delta","#ff8c00") +
        metric_card("HVN LEVELS", ", ".join([f"{p:,.0f}" for p in of_sum9.get("hvn_prices",[])]) or "—",
                    "High Vol Nodes", "#00d4ff")
    )

    # Buy/Sell pressure bar
    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;padding:10px 14px;margin-bottom:8px;">
      <div style="display:flex;justify-content:space-between;font-family:'Barlow Condensed',sans-serif;
                  font-size:10px;letter-spacing:1px;margin-bottom:4px;">
        <span style="color:#00e676;">BUY {_buy_p:.1f}%</span>
        <span style="color:#7fa8c8;">VOLUME DELTA SPLIT</span>
        <span style="color:#ff3d57;">SELL {_sel_p:.1f}%</span>
      </div>
      <div style="background:#1e3040;border-radius:2px;height:8px;overflow:hidden;">
        <div style="background:linear-gradient(90deg,#00e676,#ff3d57);
                    width:100%;height:100%;border-radius:2px;">
        </div>
      </div>
      <div style="background:#1e3040;border-radius:2px;height:8px;overflow:hidden;position:relative;margin-top:2px;">
        <div style="background:#00e676;width:{_buy_p}%;height:100%;border-radius:2px 0 0 2px;"></div>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;margin-top:4px;">
        Last 5 candle deltas: {" | ".join([f"{d:+,.0f}" for d in of_sum9.get("last5_delta",[])])}
      </div>
    </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # 🕯️ MAIN CHART — 4-panel
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    section_header(f"{idx_short(sel_t9)}  [{sel_tf9}]  Candle Chart",
                   "Candlesticks · EMA · VWAP · BB · Sweeps · OB · FVG · BOS/CHoCH")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        _rows    = 4 if show_delta else 3
        # RSI/MACD row only shown if at least one is on
        _show_indicator_row = show_rsi or show_macd
        if not _show_indicator_row:
            _rows = max(2, _rows - 1)
        _heights = [0.52, 0.14, 0.17, 0.17] if show_delta else [0.58, 0.18, 0.24]
        if not _show_indicator_row:
            _heights = [0.65, 0.35] if not show_delta else [0.55, 0.14, 0.31]
        _titles  = (
            f"{idx_short(sel_t9)}  [{sel_tf9}]  —  {len(of_df9)} candles",
            "Volume  /  Delta Bars",
            "Cumulative Delta",
            "RSI(14)  /  MACD Hist",
        ) if show_delta else (
            f"{idx_short(sel_t9)}  [{sel_tf9}]  —  {len(of_df9)} candles",
            "Volume  /  Delta Bars",
            "RSI(14)  /  MACD Hist",
        )
        if not _show_indicator_row:
            _titles = tuple(t for t in _titles if "RSI" not in t)

        fig9 = make_subplots(
            rows=_rows, cols=1,
            shared_xaxes=True,
            row_heights=_heights[:_rows],
            vertical_spacing=0.02,
            subplot_titles=_titles[:_rows],
        )

        # ── ROW 1: Candlesticks ─────────────────────────────────────────
        fig9.add_trace(go.Candlestick(
            x=of_df9["timestamp"],
            open=of_df9["open"], high=of_df9["high"],
            low=of_df9["low"],   close=of_df9["close"],
            increasing=dict(line=dict(color="#00e676",width=1), fillcolor="#00e676"),
            decreasing=dict(line=dict(color="#ff3d57",width=1), fillcolor="#ff3d57"),
            name="OHLC", whiskerwidth=0.5,
        ), row=1, col=1)

        # EMAs — each individually controlled
        ema_cfg = [(9,"#ff8c00","EMA9",show_ema9),(21,"#00d4ff","EMA21",show_ema21),(50,"#c084fc","EMA50",show_ema50)]
        for _sp9, _ce9, _nm9, _show in ema_cfg:
            if _show and f"ema{_sp9}" in of_df9.columns:
                fig9.add_trace(go.Scatter(
                    x=of_df9["timestamp"], y=of_df9[f"ema{_sp9}"],
                    mode="lines", name=_nm9, line=dict(color=_ce9,width=1.2),
                ), row=1, col=1)

        # VWAP
        if show_vwap and "vwap" in of_df9.columns:
            fig9.add_trace(go.Scatter(
                x=of_df9["timestamp"], y=of_df9["vwap"],
                mode="lines", name="VWAP", line=dict(color="#ffd600",width=1.8,dash="dot"),
            ), row=1, col=1)
            # VWAP ±2σ bands
            if "vwap_upper" in of_df9.columns:
                fig9.add_trace(go.Scatter(
                    x=of_df9["timestamp"], y=of_df9["vwap_upper"],
                    mode="lines", name="VWAP+2σ", line=dict(color="rgba(255,214,0,0.267)",width=0.8,dash="dot"),
                    showlegend=False,
                ), row=1, col=1)
                fig9.add_trace(go.Scatter(
                    x=of_df9["timestamp"], y=of_df9["vwap_lower"],
                    mode="lines", name="VWAP-2σ", line=dict(color="rgba(255,214,0,0.267)",width=0.8,dash="dot"),
                    fill="tonexty", fillcolor="rgba(255,214,0,0.03)", showlegend=False,
                ), row=1, col=1)

        # Bollinger Bands
        if show_bb and "bb_upper" in of_df9.columns:
            fig9.add_trace(go.Scatter(
                x=of_df9["timestamp"], y=of_df9["bb_upper"],
                mode="lines", name="BB Upper", line=dict(color="#2a5070",width=0.9,dash="dash"),
                showlegend=True,
            ), row=1, col=1)
            fig9.add_trace(go.Scatter(
                x=of_df9["timestamp"], y=of_df9["bb_lower"],
                mode="lines", name="BB Lower", line=dict(color="#2a5070",width=0.9,dash="dash"),
                fill="tonexty", fillcolor="rgba(42,80,112,0.07)", showlegend=False,
            ), row=1, col=1)

        # Absorption markers (gold diamond)
        _abs9 = of_df9[of_df9["absorption"]]
        if not _abs9.empty:
            fig9.add_trace(go.Scatter(
                x=_abs9["timestamp"], y=_abs9["low"] * 0.9995,
                mode="markers", name="Absorption",
                marker=dict(symbol="diamond", size=8, color="#ffd600",
                            line=dict(color="#000",width=0.5)),
            ), row=1, col=1)

        # Bull divergence (hidden selling) markers
        _bd9 = of_df9[of_df9["bull_div"]]
        if not _bd9.empty:
            fig9.add_trace(go.Scatter(
                x=_bd9["timestamp"], y=_bd9["high"] * 1.0003,
                mode="markers", name="Bear Div (hidden sell)",
                marker=dict(symbol="triangle-down", size=9, color="#ff8c00",
                            line=dict(color="#000",width=0.5)),
            ), row=1, col=1)

        # Bear divergence (hidden buying) markers
        _brd9 = of_df9[of_df9["bear_div"]]
        if not _brd9.empty:
            fig9.add_trace(go.Scatter(
                x=_brd9["timestamp"], y=_brd9["low"] * 0.9997,
                mode="markers", name="Bull Div (hidden buy)",
                marker=dict(symbol="triangle-up", size=9, color="#00d4ff",
                            line=dict(color="#000",width=0.5)),
            ), row=1, col=1)

        # ── Overlays: Liquidity Sweeps ──────────────────────────────────
        if show_sweeps:
            for _sw in sweeps_bsl9:
                fig9.add_trace(go.Scatter(
                    x=[_sw["time"]], y=[_sw["wick"]],
                    mode="markers+text",
                    name="BSL Sweep",
                    marker=dict(symbol="triangle-down-open", size=14,
                                color="#ff3d57", line=dict(color="#ff3d57",width=2)),
                    text=["BSL"], textposition="top center",
                    textfont=dict(color="#ff3d57", size=8, family="JetBrains Mono"),
                    showlegend=False,
                ), row=1, col=1)
            for _sw in sweeps_ssl9:
                fig9.add_trace(go.Scatter(
                    x=[_sw["time"]], y=[_sw["wick"]],
                    mode="markers+text",
                    name="SSL Sweep",
                    marker=dict(symbol="triangle-up-open", size=14,
                                color="#00e676", line=dict(color="#00e676",width=2)),
                    text=["SSL"], textposition="bottom center",
                    textfont=dict(color="#00e676", size=8, family="JetBrains Mono"),
                    showlegend=False,
                ), row=1, col=1)
            # Equal highs/lows as dashed lines (use shapes later)

        # ── Overlays: Fair Value Gaps (rectangles) ──────────────────────
        fvg_shapes9 = []
        if show_fvg:
            _t_min = of_df9["timestamp"].iloc[0]
            _t_max = of_df9["timestamp"].iloc[-1]
            for _fg in bull_fvg9:
                fvg_shapes9.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=_fg["time"], x1=_t_max,
                    y0=_fg["bot"],  y1=_fg["top"],
                    fillcolor="rgba(0,230,118,0.08)",
                    line=dict(color="#00e676",width=0.5,dash="dot"),
                ))
            for _fg in bear_fvg9:
                fvg_shapes9.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=_fg["time"], x1=_t_max,
                    y0=_fg["bot"],  y1=_fg["top"],
                    fillcolor="rgba(255,61,87,0.08)",
                    line=dict(color="#ff3d57",width=0.5,dash="dot"),
                ))

        # ── Overlays: Order Blocks (rectangles) ─────────────────────────
        ob_shapes9 = []
        if show_ob:
            _t_max = of_df9["timestamp"].iloc[-1]
            for _ob in bull_obs9:
                ob_shapes9.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=_ob["time"], x1=_t_max,
                    y0=_ob["bot"],  y1=_ob["top"],
                    fillcolor="rgba(0,230,118,0.12)",
                    line=dict(color="#00e676",width=1.2),
                ))
            for _ob in bear_obs9:
                ob_shapes9.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=_ob["time"], x1=_t_max,
                    y0=_ob["bot"],  y1=_ob["top"],
                    fillcolor="rgba(255,61,87,0.12)",
                    line=dict(color="#ff3d57",width=1.2),
                ))

        # ── Key H-lines ─────────────────────────────────────────────────
        fig9.add_hline(y=spot_t9, line_color="#ff8c00", line_dash="dot",
                       line_width=1.4, row=1, col=1,
                       annotation_text=f"SPOT {spot_t9:,.0f}",
                       annotation_font_color="#ff8c00",
                       annotation_position="right")
        if m_res9:
            fig9.add_hline(y=m_res9, line_color="#ff3d57", line_dash="dash",
                           line_width=1.0, row=1, col=1,
                           annotation_text=f"RES {m_res9:,.0f}",
                           annotation_font_color="#ff3d57",
                           annotation_position="right")
        if m_sup9:
            fig9.add_hline(y=m_sup9, line_color="#00e676", line_dash="dash",
                           line_width=1.0, row=1, col=1,
                           annotation_text=f"SUP {m_sup9:,.0f}",
                           annotation_font_color="#00e676",
                           annotation_position="right")

        # Equal H/L lines
        eq_shapes9 = []
        eq_annots9 = []
        if show_eqlev:
            for _eh in eq_highs9:
                eq_shapes9.append(dict(type="line",xref="paper",yref="y",
                    x0=0,x1=1,y0=_eh["level"],y1=_eh["level"],
                    line=dict(color="#ff8c00",width=0.8,dash="dashdot")))
                eq_annots9.append(dict(xref="paper",yref="y",x=1.01,y=_eh["level"],
                    text=f"EQH {_eh['level']:,.0f}",showarrow=False,
                    font=dict(color="#ff8c00",size=8,family="JetBrains Mono")))
            for _el in eq_lows9:
                eq_shapes9.append(dict(type="line",xref="paper",yref="y",
                    x0=0,x1=1,y0=_el["level"],y1=_el["level"],
                    line=dict(color="#00d4ff",width=0.8,dash="dashdot")))
                eq_annots9.append(dict(xref="paper",yref="y",x=1.01,y=_el["level"],
                    text=f"EQL {_el['level']:,.0f}",showarrow=False,
                    font=dict(color="#00d4ff",size=8,family="JetBrains Mono")))

        # BOS/CHoCH annotations
        bos_annots9 = []
        if show_bos:
            for _bev in bos9:
                bos_annots9.append(dict(
                    x=_bev["time"], y=_bev["price"], xref="x", yref="y",
                    text=_bev["label"], showarrow=True, arrowhead=2,
                    arrowcolor="#00d4ff", font=dict(color="#00d4ff",size=9,family="JetBrains Mono"),
                    bgcolor="#001020", bordercolor="#00d4ff", borderwidth=1,
                    ax=0, ay=-20 if _bev["dir"]=="UP" else 20,
                ))
            for _cev in choch9:
                bos_annots9.append(dict(
                    x=_cev["time"], y=_cev["price"], xref="x", yref="y",
                    text=_cev["label"], showarrow=True, arrowhead=2,
                    arrowcolor="#c084fc", font=dict(color="#c084fc",size=9,family="JetBrains Mono"),
                    bgcolor="#0d0018", bordercolor="#c084fc", borderwidth=1,
                    ax=0, ay=-20 if _cev["dir"]=="UP" else 20,
                ))

        all_shapes9 = fvg_shapes9 + ob_shapes9 + eq_shapes9
        all_annots9 = eq_annots9 + bos_annots9
        if all_shapes9: fig9.update_layout(shapes=all_shapes9)
        if all_annots9: fig9.update_layout(annotations=all_annots9)

        # ── ROW 2: Volume + Delta bars ──────────────────────────────────
        _vcols9 = ["#00e676" if c>=o else "#ff3d57"
                   for c,o in zip(of_df9["close"],of_df9["open"])]
        fig9.add_trace(go.Bar(
            x=of_df9["timestamp"], y=of_df9["volume"],
            marker_color=_vcols9, name="Volume", showlegend=False,
            opacity=0.7,
        ), row=2, col=1)
        # Delta bars overlaid (secondary y — we normalize to volume scale)
        _dmax9 = float(of_df9["volume"].max())
        _dnorm9 = (of_df9["delta"] / max(float(of_df9["delta"].abs().max()), 1.0) * _dmax9 * 0.4)
        _dcols9 = ["#00e676" if v>=0 else "#ff3d57" for v in of_df9["delta"]]
        fig9.add_trace(go.Bar(
            x=of_df9["timestamp"], y=_dnorm9,
            marker_color=_dcols9, name="Delta", opacity=0.55, showlegend=True,
        ), row=2, col=1)

        # ── ROW 3 (if show_delta): Cumulative Delta ─────────────────────
        _cum_row = 3
        if show_delta:
            _cd_pos = [v for v in of_df9["cum_delta"]]
            _cd_col = "#00e676" if _cd_pos[-1] >= 0 else "#ff3d57"
            fig9.add_trace(go.Scatter(
                x=of_df9["timestamp"], y=of_df9["cum_delta"],
                mode="lines", name="Cum Delta",
                line=dict(color=_cd_col, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({'0,230,118' if _cd_col=='#00e676' else '255,61,87'},0.10)",
            ), row=3, col=1)
            fig9.add_hline(y=0, line_color="#3a6080", line_dash="dot",
                           line_width=0.8, row=3, col=1)
            _cum_row = 4

        # ── RSI + MACD row ───────────────────────────────────────────────
        if show_rsi and "rsi14" in of_df9.columns:
            fig9.add_trace(go.Scatter(
                x=of_df9["timestamp"], y=of_df9["rsi14"],
                mode="lines", name="RSI(14)", line=dict(color="#c084fc",width=1.8),
            ), row=_cum_row, col=1)
            for _yl9,_yc9,_yd9 in [(70,"#ff3d57","dash"),(50,"#3a6080","dot"),(30,"#00e676","dash")]:
                fig9.add_hline(y=_yl9,line_color=_yc9,line_dash=_yd9,
                               line_width=0.7,row=_cum_row,col=1)
        if show_macd and "macd_hist" in of_df9.columns:
            _mhcols9 = ["#00e676" if v>=0 else "#ff3d57" for v in of_df9["macd_hist"]]
            fig9.add_trace(go.Bar(
                x=of_df9["timestamp"], y=of_df9["macd_hist"],
                marker_color=_mhcols9, name="MACD Hist", opacity=0.55, showlegend=True,
            ), row=_cum_row, col=1)

        # ── Global layout ────────────────────────────────────────────────
        fig9.update_layout(
            height=720 if show_delta else 620,
            paper_bgcolor="#070b0f", plot_bgcolor="#070b0f",
            font=dict(family="JetBrains Mono", color="#7fa8c8", size=10),
            margin=dict(l=10, r=110, t=40, b=10),
            legend=dict(bgcolor="rgba(7,11,15,0.88)", bordercolor="#1e3040",
                        borderwidth=1, font=dict(color="#7fa8c8",size=9),
                        orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            xaxis_rangeslider_visible=False,
            barmode="overlay",
        )
        for _xi in [f"xaxis{i}" for i in ["","2","3","4"]]:
            try:
                fig9.update_layout(**{_xi: dict(rangeslider_visible=False,
                    gridcolor="#1a2a3a",showgrid=True,zeroline=False)})
            except Exception: pass
        _ax = dict(gridcolor="#1a2a3a",showgrid=True,zeroline=False)
        fig9.update_xaxes(**_ax); fig9.update_yaxes(**_ax)
        fig9.update_yaxes(title_text="Price (₹)",tickformat=",.0f",side="right",row=1,col=1)
        fig9.update_yaxes(title_text="Vol/Delta",row=2,col=1)
        if show_delta:
            fig9.update_yaxes(title_text="Cum Δ",row=3,col=1)
        fig9.update_yaxes(title_text="RSI",range=[0,100],row=_cum_row,col=1)

        st.plotly_chart(fig9, use_container_width=True,
                        config={"scrollZoom":True,"displayModeBar":True,
                                "modeBarButtonsToRemove":["autoScale2d"]})

    except ImportError:
        st.info("pip install plotly")
    except Exception as _ce9:
        st.error(f"Chart error: {_ce9}")
        import traceback; st.code(traceback.format_exc())

    # ─────────────────────────────────────────────────────────────────────
    # 💧 LIQUIDITY SWEEP TABLE
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    ls1, ls2, ls3, ls4 = st.columns(4)

    with ls1:
        section_header("🔴 BSL Sweeps", "Buy-side liq swept (stop hunt above high)")
        if sweeps_bsl9:
            for _s in sweeps_bsl9[::-1]:
                _lb_badge = f"LB{_s.get('lookback',15)}"
                st.markdown(f"""<div style="background:#1a0305;border:1px solid #ff3d57;
                    border-left:3px solid #ff3d57;border-radius:3px;padding:8px 12px;margin:4px 0;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:12px;
                                 font-weight:700;color:#ff3d57;letter-spacing:1px;">BSL SWEEP</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                 color:#3a6080;">{str(_s["time"])[-8:]}  {_lb_badge}</span>
                  </div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:15px;
                              font-weight:700;color:#e8f4ff;margin:3px 0;">
                    Wick → {_s["wick"]:,.1f}</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#7fa8c8;">
                    Swept HH: {_s["swept"]:,.1f} | Close: {_s["close"]:,.1f}</div>
                  <div style="display:flex;gap:10px;margin-top:3px;">
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
                                 color:#ff3d57;">↓ {_s.get("reversal",0):.1f} pts</span>
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
                                 color:#ff8c00;">Strength: {_s.get("strength",0):.3f}%</span>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:#0d1117;border:1px dashed #1e3040;border-radius:3px;
                padding:10px 14px;color:#3a6080;font-family:'Barlow Condensed',sans-serif;
                font-size:12px;letter-spacing:1px;">
                NO BSL SWEEPS IN THIS SESSION<br>
                <span style="font-size:10px;color:#1e3040;">
                BSL = wick above prior swing high, closes below (bearish stop hunt)</span>
            </div>""", unsafe_allow_html=True)

    with ls2:
        section_header("🟢 SSL Sweeps", "Sell-side liq swept (stop hunt below low)")
        if sweeps_ssl9:
            for _s in sweeps_ssl9[::-1]:
                _lb_badge = f"LB{_s.get('lookback',15)}"
                st.markdown(f"""<div style="background:#010e06;border:1px solid #00e676;
                    border-left:3px solid #00e676;border-radius:3px;padding:8px 12px;margin:4px 0;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:12px;
                                 font-weight:700;color:#00e676;letter-spacing:1px;">SSL SWEEP</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                 color:#3a6080;">{str(_s["time"])[-8:]}  {_lb_badge}</span>
                  </div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:15px;
                              font-weight:700;color:#e8f4ff;margin:3px 0;">
                    Wick → {_s["wick"]:,.1f}</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#7fa8c8;">
                    Swept LL: {_s["swept"]:,.1f} | Close: {_s["close"]:,.1f}</div>
                  <div style="display:flex;gap:10px;margin-top:3px;">
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
                                 color:#00e676;">↑ {_s.get("reversal",0):.1f} pts</span>
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
                                 color:#ff8c00;">Strength: {_s.get("strength",0):.3f}%</span>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:#0d1117;border:1px dashed #1e3040;border-radius:3px;
                padding:10px 14px;color:#3a6080;font-family:'Barlow Condensed',sans-serif;
                font-size:12px;letter-spacing:1px;">
                NO SSL SWEEPS IN THIS SESSION<br>
                <span style="font-size:10px;color:#1e3040;">
                SSL = wick below prior swing low, closes above (bullish stop hunt)</span>
            </div>""", unsafe_allow_html=True)

    with ls3:
        section_header("🧱 Order Blocks", "ICT supply/demand zones")
        for _ob in (bull_obs9 + bear_obs9)[::-1]:
            _oc = "#00e676" if _ob["type"]=="BULL_OB" else "#ff3d57"
            _ol = "BULL OB" if _ob["type"]=="BULL_OB" else "BEAR OB"
            st.markdown(f"""<div style="background:#0d1117;border:1px solid {_oc};
                border-left:3px solid {_oc};border-radius:2px;padding:6px 10px;margin:3px 0;">
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                          color:{_oc};letter-spacing:1px;">{_ol}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#e8f4ff;">
                {_ob["bot"]:,.0f}  –  {_ob["top"]:,.0f}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;">
                {str(_ob["time"])[-8:]}</div>
            </div>""", unsafe_allow_html=True)
        if not bull_obs9 and not bear_obs9:
            st.markdown("<div style='color:#3a6080;font-size:12px;'>No order blocks found</div>",
                        unsafe_allow_html=True)

    with ls4:
        section_header("🔵 BOS / CHoCH", "Structure breaks & character changes")
        for _ev in (bos9 + choch9)[-6:][::-1]:
            _ec = "#00d4ff" if "BOS" in _ev["label"] else "#c084fc"
            st.markdown(f"""<div style="background:#0d1117;border:1px solid {_ec};
                border-left:3px solid {_ec};border-radius:2px;padding:6px 10px;margin:3px 0;">
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:13px;
                          font-weight:700;color:{_ec};">{_ev["label"]}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#e8f4ff;">
                @ {_ev["price"]:,.0f}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a6080;">
                {str(_ev["time"])[-8:]}</div>
            </div>""", unsafe_allow_html=True)
        if not bos9 and not choch9:
            st.markdown("<div style='color:#3a6080;font-size:12px;'>No structure breaks found</div>",
                        unsafe_allow_html=True)

    # ── FVG table ──
    st.markdown("---")
    section_header("🟡 Fair Value Gaps  (Imbalances)", "Price tends to return to fill these gaps")
    fg1, fg2 = st.columns(2)
    with fg1:
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
            letter-spacing:1px;color:#00e676;margin-bottom:6px;">BULLISH FVG (Support zones)</div>""",
            unsafe_allow_html=True)
        if bull_fvg9:
            for _fg in bull_fvg9[::-1]:
                _sz = _fg["top"] - _fg["bot"]
                st.markdown(f"""<div style="background:#010e06;border:1px solid #00e676;
                    border-radius:2px;padding:5px 10px;margin:3px 0;
                    font-family:'JetBrains Mono',monospace;font-size:11px;">
                  <span style="color:#00e676;">{_fg["bot"]:,.0f} — {_fg["top"]:,.0f}</span>
                  <span style="color:#3a6080;font-size:9px;margin-left:8px;">
                    size: {_sz:.1f} pts | {str(_fg["time"])[-8:]}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#3a6080;font-size:12px;'>None detected</div>",
                        unsafe_allow_html=True)
    with fg2:
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;
            letter-spacing:1px;color:#ff3d57;margin-bottom:6px;">BEARISH FVG (Resistance zones)</div>""",
            unsafe_allow_html=True)
        if bear_fvg9:
            for _fg in bear_fvg9[::-1]:
                _sz = _fg["top"] - _fg["bot"]
                st.markdown(f"""<div style="background:#1a0105;border:1px solid #ff3d57;
                    border-radius:2px;padding:5px 10px;margin:3px 0;
                    font-family:'JetBrains Mono',monospace;font-size:11px;">
                  <span style="color:#ff3d57;">{_fg["bot"]:,.0f} — {_fg["top"]:,.0f}</span>
                  <span style="color:#3a6080;font-size:9px;margin-left:8px;">
                    size: {_sz:.1f} pts | {str(_fg["time"])[-8:]}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#3a6080;font-size:12px;'>None detected</div>",
                        unsafe_allow_html=True)

    # ── Indicator cards ──
    st.markdown("---")
    _rc9  = "#ff3d57" if ts9.get("rsi14",50)>70 else "#00e676" if ts9.get("rsi14",50)<30 else "#c084fc"
    _mc9  = "#00e676" if ts9.get("macd_cross")=="BULLISH" else "#ff3d57"
    _ec9  = "#00e676" if ts9.get("ema_trend")=="BULLISH" else "#ff3d57" if ts9.get("ema_trend")=="BEARISH" else "#ffd600"
    _vc9  = "#00e676" if ts9.get("price_vs_vwap")=="ABOVE" else "#ff3d57"
    metrics_row(
        metric_card("RSI(14)",   f"{ts9.get('rsi14',50):.1f}",   ts9.get('rsi_zone',''), _rc9) +
        metric_card("MACD HIST", f"{ts9.get('macd_hist',0):+.4f}",ts9.get('macd_cross',''), _mc9) +
        metric_card("EMA TREND", ts9.get('ema_trend','MIXED'),    f"9:{ts9.get('ema9','?')} 21:{ts9.get('ema21','?')}", _ec9) +
        metric_card("VWAP",      f"₹{ts9.get('vwap','?')}",      f"Price {ts9.get('price_vs_vwap','?')} VWAP", _vc9) +
        metric_card("DAY HIGH",  f"₹{ts9.get('high_of_day','?')}", "", "#ff3d57") +
        metric_card("DAY LOW",   f"₹{ts9.get('low_of_day','?')}",  "", "#00e676") +
        metric_card("CANDLES",   f"{ts9.get('candles_count',0)}", f"Last: {str(ts9.get('last_candle_time','?'))[-8:]}", "#ff8c00")
    )

