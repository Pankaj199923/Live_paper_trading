import streamlit as st
import pandas as pd
import numpy as np
import json
from scipy.stats import norm
import anthropic as _anthropic
from datetime import datetime
from config import (TF_OPTIONS, TF_RESAMPLE, ACCESS_TOKEN, IST, now_ist_dt, INDEX_SHORT, LOT_SIZES,
                    AI_LOG_FILE, TRADE_FILE, TODAY_TRADES_FILE)
from utils import load_csv_safe, save_list_to_csv, idx_short, get_lot_size, get_atm_strike, compute_grand_total

# ======================
# 📐 BLACK-SCHOLES
# ======================
# ======================
# 📐 BLACK-SCHOLES
# ======================
def bs_price(S, K, T, r, sigma, opt='c'):
    if sigma <= 0 or T <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == 'c': return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:          return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, opt='c'):
    if sigma <= 0 or T <= 0:
        return {"delta":0,"gamma":0,"theta":0,"vega":0,"rho":0}
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    nd1= norm.pdf(d1)
    if opt == 'c':
        delta = norm.cdf(d1)
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    gamma = nd1 / (S*sigma*np.sqrt(T))
    vega  = S*nd1*np.sqrt(T)/100
    theta = (-(S*nd1*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*(norm.cdf(d2) if opt=='c' else norm.cdf(-d2)))/365
    return {"delta": round(delta,4), "gamma": round(gamma,6),
            "theta": round(theta,4), "vega": round(vega,4), "rho": round(rho,4)}

def calculate_gamma_bs(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return float(norm.pdf(d1) / (S*sigma*np.sqrt(T)))

def implied_vol_newton(price, S, K, T, r, opt='c', tol=1e-5, max_iter=100):
    """Newton-Raphson IV solver."""
    if T <= 0 or price <= 0: return 0.0
    sigma = 0.3
    for _ in range(max_iter):
        p = bs_price(S, K, T, r, sigma, opt)
        g = bs_greeks(S, K, T, r, sigma, opt)
        vega = g['vega'] * 100
        if abs(vega) < 1e-10: break
        diff = p - price
        if abs(diff) < tol: break
        sigma = sigma - diff/vega
        sigma = max(0.001, min(sigma, 5.0))
    return round(sigma, 4)

# ======================
# 🧠 10-FACTOR SIGNAL
# ======================

# ======================
# 🧠 10-FACTOR SIGNAL SCORE
# ======================
def compute_signal_score(oc, spot_price, index_key):
    bullish, bearish, factors = 0, 0, {}
    total_ce_oi  = oc["CE_OI"].sum()
    total_pe_oi  = oc["PE_OI"].sum()
    total_ce_chg = oc["CE_OI_Change"].sum()
    total_pe_chg = oc["PE_OI_Change"].sum()
    total_ce_vol = oc["CE_Volume"].sum()
    total_pe_vol = oc["PE_Volume"].sum()

    pcr      = total_pe_oi / total_ce_oi if total_ce_oi != 0 else 1.0
    prev_pcr = st.session_state.get("prev_pcr", pcr)
    pcr_change = pcr - prev_pcr
    st.session_state.prev_pcr = pcr

    atm_strike = get_atm_strike(spot_price, index_key)
    atm_rows   = oc[oc["Strike"] == atm_strike]
    atm_ce_chg = atm_rows["CE_OI_Change"].values[0] if not atm_rows.empty else 0
    atm_pe_chg = atm_rows["PE_OI_Change"].values[0] if not atm_rows.empty else 0
    atm_ce_iv  = atm_rows["CE_IV"].values[0]         if not atm_rows.empty else 0
    atm_pe_iv  = atm_rows["PE_IV"].values[0]         if not atm_rows.empty else 0
    atm_ce_th  = atm_rows["CE_Theta"].values[0]      if not atm_rows.empty else 0
    atm_pe_th  = atm_rows["PE_Theta"].values[0]      if not atm_rows.empty else 0

    # Resistance = highest CE OI wall ABOVE spot; Support = highest PE OI wall BELOW spot
    above_spot = oc[oc["Strike"] > spot_price]
    below_spot = oc[oc["Strike"] < spot_price]
    main_res = above_spot.sort_values("CE_OI", ascending=False).iloc[0]["Strike"] if not above_spot.empty else oc.sort_values("CE_OI", ascending=False).iloc[0]["Strike"]
    main_sup = below_spot.sort_values("PE_OI", ascending=False).iloc[0]["Strike"] if not below_spot.empty else oc.sort_values("PE_OI", ascending=False).iloc[0]["Strike"]
    total_gex= oc["NET_GEX"].sum() if "NET_GEX" in oc.columns else 0

# ======================
# 🧠 15-FACTOR PRO SIGNAL ENGINE
# ======================
def compute_signal_score(oc, spot_price, index_key):
    """
    15-factor institutional signal engine (max 20 weighted pts).
    Covers: OI bias/momentum, PCR level/momentum, volume, ATM build,
    S/R breakout, GEX regime, IV skew, theta, OI concentration,
    wall proximity, OI acceleration, vol-PCR divergence, IV term structure.
    """
    bullish, bearish, factors = 0, 0, {}
    total_ce_oi  = oc["CE_OI"].sum()
    total_pe_oi  = oc["PE_OI"].sum()
    total_ce_chg = oc["CE_OI_Change"].sum()
    total_pe_chg = oc["PE_OI_Change"].sum()
    total_ce_vol = oc["CE_Volume"].sum()
    total_pe_vol = oc["PE_Volume"].sum()

    pcr        = total_pe_oi / total_ce_oi if total_ce_oi != 0 else 1.0
    prev_pcr   = st.session_state.get("prev_pcr", pcr)
    pcr_change = pcr - prev_pcr
    st.session_state.prev_pcr = pcr

    atm_strike = get_atm_strike(spot_price, index_key)
    atm_rows   = oc[oc["Strike"] == atm_strike]
    atm_ce_chg = float(atm_rows["CE_OI_Change"].values[0]) if not atm_rows.empty else 0
    atm_pe_chg = float(atm_rows["PE_OI_Change"].values[0]) if not atm_rows.empty else 0
    atm_ce_iv  = float(atm_rows["CE_IV"].values[0])         if not atm_rows.empty else 15.0
    atm_pe_iv  = float(atm_rows["PE_IV"].values[0])         if not atm_rows.empty else 15.0
    atm_ce_th  = float(atm_rows["CE_Theta"].values[0]) if not atm_rows.empty and "CE_Theta" in atm_rows.columns else 0
    atm_pe_th  = float(atm_rows["PE_Theta"].values[0]) if not atm_rows.empty and "PE_Theta" in atm_rows.columns else 0
    atm_ce_oi  = float(atm_rows["CE_OI"].values[0])   if not atm_rows.empty else 1
    atm_pe_oi  = float(atm_rows["PE_OI"].values[0])   if not atm_rows.empty else 1

    # Resistance = nearest high CE OI wall ABOVE spot; Support = nearest high PE OI wall BELOW spot
    above_spot = oc[oc["Strike"] > spot_price]
    below_spot = oc[oc["Strike"] < spot_price]
    main_res = above_spot.sort_values("CE_OI", ascending=False).iloc[0]["Strike"] if not above_spot.empty else oc.sort_values("CE_OI", ascending=False).iloc[0]["Strike"]
    main_sup = below_spot.sort_values("PE_OI", ascending=False).iloc[0]["Strike"] if not below_spot.empty else oc.sort_values("PE_OI", ascending=False).iloc[0]["Strike"]
    total_gex = oc["NET_GEX"].sum() if "NET_GEX" in oc.columns else 0

    # 1 — OI Bias (w=2)
    if total_pe_oi > total_ce_oi:   bullish += 2; factors["OI Bias"]   = ("🟢 Bullish", 2, 0)
    else:                           bearish += 2; factors["OI Bias"]   = ("🔴 Bearish", 0, 2)

    # 2 — OI Momentum (w=2)
    if total_pe_chg > total_ce_chg: bullish += 2; factors["OI Mom"]    = ("🟢 PE Acc", 2, 0)
    else:                           bearish += 2; factors["OI Mom"]    = ("🔴 CE Acc", 0, 2)

    # 3 — PCR Level (w=1) — >1.2 = extreme put buyers = bullish contrarian signal
    if pcr > 1.2:    bullish += 1; factors["PCR Lvl"] = (f"🟢 {pcr:.2f}", 1, 0)
    elif pcr < 0.8:  bearish += 1; factors["PCR Lvl"] = (f"🔴 {pcr:.2f}", 0, 1)
    else:                          factors["PCR Lvl"] = (f"🟡 {pcr:.2f}", 0, 0)

    # 4 — PCR Momentum (w=1) — use meaningful threshold to avoid noise
    if pcr_change > 0.05:    bullish += 1; factors["PCR Mom"] = ("🟢 Rising", 1, 0)
    elif pcr_change < -0.05: bearish += 1; factors["PCR Mom"] = ("🔴 Falling", 0, 1)
    else:                                  factors["PCR Mom"] = ("🟡 Flat", 0, 0)

    # 5 — Volume Bias (w=1)
    vol_ratio = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 1.0
    if vol_ratio > 1.1:   bullish += 1; factors["Vol Bias"]  = (f"🟢 PE {vol_ratio:.2f}x", 1, 0)
    elif vol_ratio < 0.9: bearish += 1; factors["Vol Bias"]  = (f"🔴 CE {1/vol_ratio:.2f}x", 0, 1)
    else:                               factors["Vol Bias"]  = ("🟡 Equal", 0, 0)

    # 6 — ATM OI Build (w=1)
    if atm_pe_chg > atm_ce_chg * 1.2:   bullish += 1; factors["ATM Build"] = ("🟢 PE Acc", 1, 0)
    elif atm_ce_chg > atm_pe_chg * 1.2: bearish += 1; factors["ATM Build"] = ("🔴 CE Acc", 0, 1)
    else:                                              factors["ATM Build"] = ("🟡 Neutral", 0, 0)

    # 7 — S/R Breakout (w=2)
    if spot_price > main_res:   bullish += 2; factors["Breakout"]  = ("🟢 Above Res", 2, 0)
    elif spot_price < main_sup: bearish += 2; factors["Breakout"]  = ("🔴 Below Sup", 0, 2)
    else:                                     factors["Breakout"]  = ("🟡 In Range", 0, 0)

    # 8 — GEX Regime (w=1)
    if total_gex > 0:   bullish += 1; factors["GEX"]      = (f"🟢 +{total_gex/1e6:.1f}M", 1, 0)
    elif total_gex < 0: bearish += 1; factors["GEX"]      = (f"🔴 {total_gex/1e6:.1f}M", 0, 1)
    else:                             factors["GEX"]      = ("🟡 Neutral", 0, 0)

    # 9 — IV Skew (w=1)
    iv_ratio = atm_pe_iv / atm_ce_iv if atm_ce_iv > 0 else 1.0
    if iv_ratio > 1.05:   bullish += 1; factors["IV Skew"]  = (f"🟢 PE {iv_ratio:.2f}x", 1, 0)
    elif iv_ratio < 0.95: bearish += 1; factors["IV Skew"]  = (f"🔴 CE {1/iv_ratio:.2f}x", 0, 1)
    else:                               factors["IV Skew"]  = ("🟡 Flat", 0, 0)

    # 10 — Theta Pressure (w=1)
    if abs(atm_ce_th) > abs(atm_pe_th) * 1.1:   bearish += 1; factors["Theta"]    = ("🔴 CE Decay", 0, 1)
    elif abs(atm_pe_th) > abs(atm_ce_th) * 1.1: bullish += 1; factors["Theta"]    = ("🟢 PE Decay", 1, 0)
    else:                                                       factors["Theta"]    = ("🟡 Equal", 0, 0)

    # 11 — ATM OI Concentration ratio (w=1)
    atm_conc = atm_pe_oi / atm_ce_oi if atm_ce_oi > 0 else 1.0
    if atm_conc > 1.3:    bullish += 1; factors["ATM Conc"] = (f"🟢 {atm_conc:.2f}x", 1, 0)
    elif atm_conc < 0.77: bearish += 1; factors["ATM Conc"] = (f"🔴 {atm_conc:.2f}x", 0, 1)
    else:                               factors["ATM Conc"] = (f"🟡 {atm_conc:.2f}x", 0, 0)

    # 12 — Wall Proximity (w=1) — nearer to support = bullish, nearer to resistance = bearish
    dist_res = abs(spot_price - main_res)
    dist_sup = abs(spot_price - main_sup)
    prox_ratio = dist_sup / (dist_res + dist_sup + 1)
    if prox_ratio < 0.35:   bullish += 1; factors["Wall Prox"] = ("🟢 Near Sup", 1, 0)
    elif prox_ratio > 0.65: bearish += 1; factors["Wall Prox"] = ("🔴 Near Res", 0, 1)
    else:                                 factors["Wall Prox"] = ("🟡 Mid Range", 0, 0)

    # 13 — OI Acceleration (top-3 strikes net change) (w=1)
    top3_ce_chg = oc.nlargest(3, "CE_OI_Change")["CE_OI_Change"].sum()
    top3_pe_chg = oc.nlargest(3, "PE_OI_Change")["PE_OI_Change"].sum()
    if top3_pe_chg > top3_ce_chg: bullish += 1; factors["OI Accel"] = ("🟢 PE Surge", 1, 0)
    elif top3_ce_chg > top3_pe_chg: bearish += 1; factors["OI Accel"] = ("🔴 CE Surge", 0, 1)
    else:                                          factors["OI Accel"] = ("🟡 Neutral", 0, 0)

    # 14 — Vol-PCR Confluence (w=1) — same direction signals = aligned conviction
    if vol_ratio > 1.1 and pcr_change >= 0:    bullish += 1; factors["Vol-PCR"]  = ("🟢 Aligned", 1, 0)
    elif vol_ratio < 0.9 and pcr_change <= 0:  bearish += 1; factors["Vol-PCR"]  = ("🔴 Aligned", 0, 1)
    else:                                                     factors["Vol-PCR"]  = ("🟡 Diverge", 0, 0)

    # 15 — IV Term Structure (chain-wide PE vs CE IV spread) (w=1)
    avg_ce_iv = float(oc["CE_IV"].mean()) if "CE_IV" in oc.columns else 15.0
    avg_pe_iv = float(oc["PE_IV"].mean()) if "PE_IV" in oc.columns else 15.0
    iv_diff   = avg_pe_iv - avg_ce_iv
    if iv_diff > 1.5:    bearish += 1; factors["IV Term"]  = (f"🔴 PE Skew+{iv_diff:.1f}", 0, 1)
    elif iv_diff < -1.5: bullish += 1; factors["IV Term"]  = (f"🟢 CE Skew{iv_diff:.1f}", 1, 0)
    else:                              factors["IV Term"]  = (f"🟡 Flat {iv_diff:.1f}", 0, 0)

    return bullish, bearish, pcr, pcr_change, main_res, main_sup, factors


# ======================
# 📸 SNAPSHOT ENGINE — One Parquet per day, all minutes appended

# ======================
# 🤖 AI TRADE GENERATOR
# ======================
def generate_ai_trade(df, atm_row, atm_strike, market_flow, selected_index,
                      risk_points, reward, bullish_score, bearish_score):
    """
    Pro-grade rule-based trade generator.
    - Choppy / Range → Iron Fly (defined risk) instead of naked straddle
    - Bullish → Short Put Spread (Bull Put Spread) for defined risk
    - Bearish → Short Call Spread (Bear Call Spread) for defined risk
    - Strong directional (score ≥8) → Naked directional buy
    - IV spike guard: skip if ATM IV > 1.25x chain avg
    - Daily loss limit + max position checks
    """
    now_str   = datetime.now(IST).strftime("%H:%M:%S")
    today_str = datetime.now().strftime("%Y-%m-%d")
    trades    = []

    # ── Guard: daily loss limit ──
    open_pnl, closed_pnl, grand_total = compute_grand_total()
    daily_limit = st.session_state.get("daily_loss_limit", -10000)
    if grand_total <= daily_limit:
        st.error(f"🔴 Daily Loss Limit Hit (₹{grand_total:.0f}). Trading halted.")
        return trades

    # ── Guard: max open positions ──
    max_open   = st.session_state.get("max_open_positions", 6)
    open_count = sum(1 for t in st.session_state.ai_trade_log if t.get("Status") == "Active")
    if open_count >= max_open:
        st.warning(f"⚠️ Max open positions ({max_open}) reached.")
        return trades

    # ── Guard: IV spike — avoid selling into vol crush ──
    atm_ce_iv = float(atm_row.get("CE_IV", 0))
    atm_pe_iv = float(atm_row.get("PE_IV", 0))
    iv_avg    = df["CE_IV"].rolling(5, min_periods=1).mean().iloc[-1]
    if atm_ce_iv > iv_avg * 1.25 or atm_pe_iv > iv_avg * 1.25:
        st.warning("⚡ IV Spike detected — skipping to avoid vol crush. Wait for IV to settle.")
        return trades

    # ── Setup helpers ──
    step_series = df["Strike"].diff().dropna()
    step        = int(step_series.mode()[0]) if not step_series.empty else 50
    lot_size    = get_lot_size(selected_index)
    score_margin = abs(bullish_score - bearish_score)

    def get_row_ltp(strike, col):
        r = df[df["Strike"] == strike]
        return round(float(r[col].values[0]), 2) if not r.empty and col in r.columns else 0.0

    def make_trade(strike, t_type, action, entry, score, flow, sl_val=None, tgt_val=None):
        """Build a trade dict with proper directional SL/Target."""
        if sl_val is None:
            sl_val = round(entry + risk_points, 2) if action == "SELL" else round(entry - risk_points, 2)
        if tgt_val is None:
            tgt_val = round(entry - reward, 2) if action == "SELL" else round(entry + reward, 2)
        return {
            "Entry Time": now_str, "Date": today_str,
            "Index_Key":  selected_index, "Strike": strike,
            "Type":       t_type, "Action": action,
            "Entry":      entry,
            "SL":         sl_val,
            "Target":     tgt_val,
            "Lot_Size":   lot_size,
            "Live LTP":   0, "Live_PnL": 0,
            "Status":     "Active",
            "Score":      score, "Flow": flow,
        }

    # ── Regime: Choppy or weak signal → Iron Fly (defined risk) ──
    is_choppy = (market_flow == "Choppy") or (score_margin <= 2)
    if is_choppy:
        # Iron Fly: SELL ATM CE + SELL ATM PE, BUY OTM CE + BUY OTM PE (hedge wings)
        ce_e  = get_row_ltp(atm_strike, "CE_LTP")
        pe_e  = get_row_ltp(atm_strike, "PE_LTP")
        # Wing buys 2 steps OTM
        wing_ce_k = atm_strike + 2 * step
        wing_pe_k = atm_strike - 2 * step
        wing_ce_e = get_row_ltp(wing_ce_k, "CE_LTP")
        wing_pe_e = get_row_ltp(wing_pe_k, "PE_LTP")
        net_credit = round((ce_e + pe_e) - (wing_ce_e + wing_pe_e), 2)
        max_loss   = round((2 * step) - net_credit, 2)
        flow_label = "Choppy → Iron Fly"
        trades.append(make_trade(atm_strike, "SELL CE", "SELL", ce_e,  bullish_score, flow_label,
                                 sl_val=round(ce_e * 1.5, 2), tgt_val=round(ce_e * 0.3, 2)))
        trades.append(make_trade(atm_strike, "SELL PE", "SELL", pe_e,  bearish_score, flow_label,
                                 sl_val=round(pe_e * 1.5, 2), tgt_val=round(pe_e * 0.3, 2)))
        if wing_ce_e > 0:
            trades.append(make_trade(wing_ce_k, "BUY CE", "BUY", wing_ce_e, 0, flow_label + " [Hedge]"))
        if wing_pe_e > 0:
            trades.append(make_trade(wing_pe_k, "BUY PE", "BUY", wing_pe_e, 0, flow_label + " [Hedge]"))
        return trades

    # ── Regime: Range → Iron Condor ──
    if market_flow == "Range":
        # Sell 1 step OTM on both sides, buy 2 steps OTM as wings
        sell_ce_k = atm_strike + step;  sell_pe_k = atm_strike - step
        buy_ce_k  = atm_strike + 2*step; buy_pe_k  = atm_strike - 2*step
        sell_ce_e = get_row_ltp(sell_ce_k, "CE_LTP")
        sell_pe_e = get_row_ltp(sell_pe_k, "PE_LTP")
        buy_ce_e  = get_row_ltp(buy_ce_k,  "CE_LTP")
        buy_pe_e  = get_row_ltp(buy_pe_k,  "PE_LTP")
        if all(x > 0 for x in [sell_ce_e, sell_pe_e]):
            trades.append(make_trade(sell_ce_k, "SELL CE", "SELL", sell_ce_e,
                                     max(bullish_score, bearish_score), "Range → Condor",
                                     sl_val=round(sell_ce_e * 1.8, 2), tgt_val=round(sell_ce_e * 0.25, 2)))
            trades.append(make_trade(sell_pe_k, "SELL PE", "SELL", sell_pe_e,
                                     max(bullish_score, bearish_score), "Range → Condor",
                                     sl_val=round(sell_pe_e * 1.8, 2), tgt_val=round(sell_pe_e * 0.25, 2)))
            if buy_ce_e > 0:
                trades.append(make_trade(buy_ce_k, "BUY CE", "BUY", buy_ce_e, 0, "Range → Condor [Hedge]"))
            if buy_pe_e > 0:
                trades.append(make_trade(buy_pe_k, "BUY PE", "BUY", buy_pe_e, 0, "Range → Condor [Hedge]"))
        return trades

    # ── Regime: Bullish → Bull Put Spread (SELL ATM PE, BUY 1-step OTM PE) ──
    if market_flow == "Bullish":
        sell_pe_k = atm_strike
        buy_pe_k  = atm_strike - step
        sell_pe_e = get_row_ltp(sell_pe_k, "PE_LTP")
        buy_pe_e  = get_row_ltp(buy_pe_k,  "PE_LTP")
        net_credit = round(sell_pe_e - buy_pe_e, 2)
        # If strong directional (score ≥ 8), also add naked CE buy for leverage
        if sell_pe_e > 0:
            trades.append(make_trade(sell_pe_k, "SELL PE", "SELL", sell_pe_e,
                                     bullish_score, "Bullish → Bull Put Spread",
                                     sl_val=round(sell_pe_e * 1.5, 2), tgt_val=round(sell_pe_e * 0.2, 2)))
        if buy_pe_e > 0:
            trades.append(make_trade(buy_pe_k, "BUY PE", "BUY", buy_pe_e,
                                     bullish_score, "Bullish → Bull Put Spread [Hedge]"))
        if bullish_score >= 8:
            # Add OTM CE buy for directional leverage (BUY CE one step OTM)
            buy_ce_k = atm_strike + step
            buy_ce_e = get_row_ltp(buy_ce_k, "CE_LTP")
            if buy_ce_e > 0:
                trades.append(make_trade(buy_ce_k, "BUY CE", "BUY", buy_ce_e,
                                         bullish_score, "Bullish → Directional CE"))
        return trades

    # ── Regime: Bearish → Bear Call Spread (SELL ATM CE, BUY 1-step OTM CE) ──
    if market_flow == "Bearish":
        sell_ce_k = atm_strike
        buy_ce_k  = atm_strike + step
        sell_ce_e = get_row_ltp(sell_ce_k, "CE_LTP")
        buy_ce_e  = get_row_ltp(buy_ce_k,  "CE_LTP")
        if sell_ce_e > 0:
            trades.append(make_trade(sell_ce_k, "SELL CE", "SELL", sell_ce_e,
                                     bearish_score, "Bearish → Bear Call Spread",
                                     sl_val=round(sell_ce_e * 1.5, 2), tgt_val=round(sell_ce_e * 0.2, 2)))
        if buy_ce_e > 0:
            trades.append(make_trade(buy_ce_k, "BUY CE", "BUY", buy_ce_e,
                                     bearish_score, "Bearish → Bear Call Spread [Hedge]"))
        if bearish_score >= 8:
            # Add OTM PE buy for directional leverage
            buy_pe_k = atm_strike - step
            buy_pe_e = get_row_ltp(buy_pe_k, "PE_LTP")
            if buy_pe_e > 0:
                trades.append(make_trade(buy_pe_k, "BUY PE", "BUY", buy_pe_e,
                                         bearish_score, "Bearish → Directional PE"))
        return trades

    return trades

# ======================
# 🚨 ALERT ENGINE  [NEW PRO FEATURE]

# ======================
# 🔔 ALERT ENGINE
# ======================
def check_alerts(oc, spot_price, index_key):
    """Check and fire alerts for key threshold events."""
    alerts = []
    atm = get_atm_strike(spot_price, index_key)
    atm_row = oc[oc["Strike"] == atm]
    if not atm_row.empty:
        ce_iv = float(atm_row["CE_IV"].iloc[0])
        pe_iv = float(atm_row["PE_IV"].iloc[0])
        avg_iv = oc["CE_IV"].mean()
        if ce_iv > avg_iv * 1.3:
            alerts.append(("⚡ IV SPIKE", f"ATM CE IV {ce_iv:.1f}% >> avg {avg_iv:.1f}%", "red"))
        if pe_iv > avg_iv * 1.3:
            alerts.append(("⚡ IV SPIKE", f"ATM PE IV {pe_iv:.1f}% >> avg {avg_iv:.1f}%", "red"))
        ce_oi_chg = float(atm_row["CE_OI_Change"].iloc[0])
        pe_oi_chg = float(atm_row["PE_OI_Change"].iloc[0])
        if abs(ce_oi_chg) > 500000:
            alerts.append(("📊 OI SURGE", f"ATM CE OI Change: {ce_oi_chg:+,.0f}", "orange"))
        if abs(pe_oi_chg) > 500000:
            alerts.append(("📊 OI SURGE", f"ATM PE OI Change: {pe_oi_chg:+,.0f}", "orange"))
    return alerts

# ======================
# 📊 INTRADAY CANDLES (Upstox 1-min OHLCV)
# ======================
TF_OPTIONS = {
    "1m":  "1minute",
    "3m":  "3minute",
    "5m":  "5minute",
    "10m": "10minute",
    "15m": "15minute",
    "30m": "30minute",
    "1H":  "1hour",
}
TF_RESAMPLE = {
    "3minute": "3min", "5minute": "5min",
    "10minute": "10min", "15minute": "15min",
}


# ======================
# 🧠 CLAUDE TRADE SETUP
# ======================
def call_claude_trade_setup(spot, index_name, expiry, oc_df, atm_strike,
                             bull_score, bear_score, market_flow, pcr, pcr_chg,
                             main_res, main_sup, max_pain, net_gex,
                             tech_summary, lot_size, dte):
    """Send ALL live market data to Claude and get a structured trade setup."""
    try:
        client = _anthropic.Anthropic()

        # ATM row
        atm_rows = oc_df[oc_df["Strike"] == atm_strike]
        atm_data = {}
        if not atm_rows.empty:
            r = atm_rows.iloc[0]
            atm_data = {
                "ce_ltp": round(float(r.get("CE_LTP", 0)), 2),
                "pe_ltp": round(float(r.get("PE_LTP", 0)), 2),
                "ce_iv":  round(float(r.get("CE_IV", 0)), 2),
                "pe_iv":  round(float(r.get("PE_IV", 0)), 2),
                "ce_oi":  int(r.get("CE_OI", 0)),
                "pe_oi":  int(r.get("PE_OI", 0)),
                "ce_oi_chg": int(r.get("CE_OI_Change", 0)),
                "pe_oi_chg": int(r.get("PE_OI_Change", 0)),
                "ce_delta":  round(float(r.get("CE_Delta", 0)), 4),
                "pe_delta":  round(float(r.get("PE_Delta", 0)), 4),
                "ce_theta":  round(float(r.get("CE_Theta", 0)), 4),
                "pe_theta":  round(float(r.get("PE_Theta", 0)), 4),
                "ce_vega":   round(float(r.get("CE_Vega", 0)), 4),
                "iv_skew":   round(float(r.get("IV_Skew", 1)), 3),
            }

        top_ce = oc_df.nlargest(3, "CE_OI")[["Strike","CE_OI"]].to_dict("records")
        top_pe = oc_df.nlargest(3, "PE_OI")[["Strike","PE_OI"]].to_dict("records")

        # OTM strikes for strangle reference
        step_s = oc_df["Strike"].diff().dropna().mode()
        step_s = int(step_s.iloc[0]) if not step_s.empty else 50
        otm_ce_s = atm_strike + step_s
        otm_pe_s = atm_strike - step_s
        otm_ce_row = oc_df[oc_df["Strike"] == otm_ce_s]
        otm_pe_row = oc_df[oc_df["Strike"] == otm_pe_s]
        otm_ce_ltp = round(float(otm_ce_row["CE_LTP"].iloc[0]), 2) if not otm_ce_row.empty else "N/A"
        otm_pe_ltp = round(float(otm_pe_row["PE_LTP"].iloc[0]), 2) if not otm_pe_row.empty else "N/A"

        # ── Kill Zone classification ──
        now_str_ai = datetime.now(IST).strftime("%H:%M:%S")
        hour_ist = int(now_str_ai.split(":")[0])
        if 9 <= hour_ist < 10:
            kill_zone = "INDIA OPEN KILL ZONE (9:15–10:00) — High volatility, wait for direction"
        elif 11 <= hour_ist < 13:
            kill_zone = "MID-SESSION (11:00–13:00) — Low volatility, premium decay window"
        elif 14 <= hour_ist < 15:
            kill_zone = "LONDON OVERLAP / PRE-EXPIRY (14:00–15:00) — Directional moves possible"
        elif 15 <= hour_ist < 16:
            kill_zone = "EXPIRY HOUR (15:00–15:30) — Gamma risk extreme, reduce size"
        else:
            kill_zone = "REGULAR SESSION"

        # ── OTE (Optimal Trade Entry) level — 62–79% fib retracement ──
        day_high = tech_summary.get("high_of_day", spot)
        day_low  = tech_summary.get("low_of_day", spot)
        fib_range = day_high - day_low
        ote_bull = round(day_high - 0.618 * fib_range, 1)  # 61.8% retrace for longs
        ote_bear = round(day_low  + 0.618 * fib_range, 1)  # 61.8% retrace for shorts

        # ── IV Rank (IVR) estimate — ATM IV vs chain avg ──
        avg_chain_iv = float(oc_df["CE_IV"].mean()) if "CE_IV" in oc_df.columns else 15.0
        ivr_est = round((atm_data.get("ce_iv", avg_chain_iv) / avg_chain_iv - 1) * 100, 1)
        ivr_regime = "HIGH IV — Sell Premium" if ivr_est > 20 else "LOW IV — Buy Premium" if ivr_est < -10 else "NEUTRAL IV"

        # ── ATR info ──
        atr_val      = tech_summary.get("atr14", 0)
        atr_sl_tight = tech_summary.get("atr_sl_tight", 15)
        atr_sl_wide  = tech_summary.get("atr_sl_wide", 25)

        prompt = f"""You are QuantDesk Pro AI — an elite NSE derivatives trader with 15+ years of F&O experience using ICT/SMC and institutional frameworks.

Analyze the following LIVE market data and generate a COMPLETE, actionable intraday trade setup for {index_name}.

━━━ MARKET SNAPSHOT ({now_str_ai}) ━━━
Index:        {index_name}
Spot Price:   ₹{spot:,.2f}
ATM Strike:   {atm_strike}
Expiry:       {expiry}  ({dte} days to expiry)
Session:      {"LIVE MARKET" if MARKET_OPEN else "POST-MARKET"}
Kill Zone:    {kill_zone}

━━━ 15-FACTOR SIGNAL SCORES ━━━
Bullish: {bull_score}/20  |  Bearish: {bear_score}/20
Market Regime: {market_flow}
PCR: {pcr:.3f}  ({'+' if pcr_chg > 0 else ''}{pcr_chg:.3f} — {"Rising ▲" if pcr_chg > 0 else "Falling ▼" if pcr_chg < 0 else "Flat"})
Net GEX: {net_gex/1e6:.2f}M  ({"Positive → MM long gamma → Mean Reversion" if net_gex > 0 else "Negative → MM short gamma → Trending/Volatile"})

━━━ KEY OI LEVELS ━━━
Resistance (CE OI Wall above spot): {main_res:,.0f}  (+{main_res-spot:.0f} pts)
Support    (PE OI Wall below spot): {main_sup:,.0f}  (-{spot-main_sup:.0f} pts)
Max Pain Strike:                    {max_pain:,.0f}   ({'+' if spot>max_pain else '-'}{abs(spot-max_pain):.0f} pts from spot)
Range Width:                        {main_res - main_sup:.0f} pts
Top CE Walls: {', '.join([f"{int(x['Strike'])}({int(x['CE_OI'])/1e5:.1f}L)" for x in top_ce])}
Top PE Walls: {', '.join([f"{int(x['Strike'])}({int(x['PE_OI'])/1e5:.1f}L)" for x in top_pe])}

━━━ ATM OPTIONS (Strike {atm_strike}) ━━━
CE LTP: ₹{atm_data.get('ce_ltp', 0)}   |  PE LTP: ₹{atm_data.get('pe_ltp', 0)}
CE IV:  {atm_data.get('ce_iv', 0)}%    |  PE IV:  {atm_data.get('pe_iv', 0)}%
CE OI:  {atm_data.get('ce_oi', 0):,}  |  PE OI:  {atm_data.get('pe_oi', 0):,}
CE OI Δ:{atm_data.get('ce_oi_chg', 0):+,}  |  PE OI Δ:{atm_data.get('pe_oi_chg', 0):+,}
CE Delta: {atm_data.get('ce_delta', 0)}  |  PE Delta: {atm_data.get('pe_delta', 0)}
CE Theta: {atm_data.get('ce_theta', 0)}  |  PE Theta: {atm_data.get('pe_theta', 0)}
CE Vega:  {atm_data.get('ce_vega', 0)}   |  IV Skew (PE/CE): {atm_data.get('iv_skew', 1)}

━━━ IV ENVIRONMENT ━━━
IVR vs Chain Avg: {ivr_est:+.1f}%  →  {ivr_regime}
OTM CE ({otm_ce_s}): ₹{otm_ce_ltp}
OTM PE ({otm_pe_s}): ₹{otm_pe_ltp}
Lot Size: {lot_size}

━━━ INTRADAY TECHNICALS (1-Min Chart) ━━━
Day Open: ₹{tech_summary.get('open_price', 'N/A')}  |  Day High: ₹{tech_summary.get('high_of_day', 'N/A')}  |  Day Low: ₹{tech_summary.get('low_of_day', 'N/A')}
Volume:   {tech_summary.get('volume', 0):,} ({tech_summary.get('candles_count', 0)} candles)

ATR(14): {atr_val} pts  →  Tight SL: {atr_sl_tight} pts  |  Wide SL: {atr_sl_wide} pts
RSI(14):  {tech_summary.get('rsi14', 50)} → {tech_summary.get('rsi_zone', 'NEUTRAL')}
EMA(9/21/50/200): {tech_summary.get('ema9','N/A')} / {tech_summary.get('ema21','N/A')} / {tech_summary.get('ema50','N/A')} / {tech_summary.get('ema200','N/A')}
EMA Trend: {tech_summary.get('ema_trend', 'MIXED')}
SuperTrend: {tech_summary.get('supertrend_bias', 'N/A')}
VWAP: ₹{tech_summary.get('vwap','N/A')} (±2σ: {tech_summary.get('vwap_lower','N/A')} / {tech_summary.get('vwap_upper','N/A')})  |  Price: {tech_summary.get('price_vs_vwap','N/A')} VWAP
MACD: {tech_summary.get('macd',0):.4f}  Signal: {tech_summary.get('macd_signal',0):.4f}  Hist: {tech_summary.get('macd_hist',0):.4f} → {tech_summary.get('macd_cross','NEUTRAL')}
BB({tech_summary.get('bb_condition','NORMAL')}): {tech_summary.get('bb_lower','N/A')} / {tech_summary.get('bb_mid','N/A')} / {tech_summary.get('bb_upper','N/A')}

━━━ ICT / SMC CONTEXT ━━━
OTE Bull Level (61.8% retrace): ₹{ote_bull} (potential long entry if spot pulls here)
OTE Bear Level (61.8% retrace): ₹{ote_bear} (potential short entry if spot rallies here)
Max Pain Gravity: Price tends to drift toward ₹{max_pain:,.0f} by expiry
GEX Zero Level: Flip zone (from GEX tab) — price behavior changes character here

━━━ OUTPUT FORMAT ━━━
Respond ONLY with a raw JSON object (no markdown, no explanation outside JSON):
{{
  "regime": "BULLISH|BEARISH|RANGE|CHOPPY",
  "confidence": "HIGH|MEDIUM|LOW",
  "strategy": "Short exact strategy name e.g. Iron Fly / Bull Put Spread / Bear Call Spread",
  "quick_take": "One bold punchy sentence summarizing the entire trade idea with key numbers",
  "rationale": "4-5 sentence analysis fusing technicals + options data + OI levels + IVR regime + kill zone. Mention specific prices and levels.",
  "legs": [
    {{"action": "SELL|BUY", "type": "CE|PE", "strike": 0, "premium": 0.0, "lots": 1}}
  ],
  "entry_trigger": "Specific price/indicator condition to enter (e.g. spot holds above VWAP at {tech_summary.get('vwap','N/A')} and RSI < 65)",
  "stop_loss_points": 0,
  "target_points": 0,
  "sl_description": "ATR-based or structural SL logic (e.g. close above {main_res:.0f} on 5-min = invalid)",
  "target_description": "Specific target with level reason (e.g. max pain {max_pain:.0f} = natural magnet)",
  "max_risk_per_lot": 0,
  "max_profit_per_lot": 0,
  "key_levels": ["{main_res:.0f}", "{atm_strike}", "{main_sup:.0f}", "{max_pain:.0f}"],
  "avoid_if": "Clear invalidation conditions (e.g. GEX flips negative, RSI breaks 70, IV spikes 30%+)",
  "time_in_trade": "Expected trade duration based on regime and kill zone",
  "trailing_sl": "ATR-based or structure-based trailing stop logic",
  "position_sizing": "Suggested lots based on 1% account risk at current ATR ({atr_val} pts/lot)"
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        # Robust JSON extraction — strip any markdown fences
        if "```" in raw:
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            raw = match.group(1) if match else raw.split("```")[1].lstrip("json").strip()
        # Find first { ... } block if extra text around it
        brace_start = raw.find("{")
        brace_end   = raw.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            raw = raw[brace_start:brace_end]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": raw[:600] if 'raw' in dir() else ""}
    except _anthropic.APIStatusError as e:
        status = e.status_code
        msg    = str(e.message) if hasattr(e, "message") else str(e)
        if status == 400 and "credit" in msg.lower():
            return {"error": "CREDIT_ERROR",
                    "billing": True,
                    "detail": msg}
        if status == 401:
            return {"error": "Invalid Anthropic API key. Check your st.secrets."}
        if status == 429:
            return {"error": "Rate limit hit. Wait a moment and retry."}
        return {"error": f"API error {status}: {msg}"}
    except _anthropic.APIConnectionError:
        return {"error": "Network error reaching Anthropic API. Check internet."}
    except Exception as e:
        return {"error": str(e)}


# ======================
# 🎨 REUSABLE UI COMPONENTS
# ======================
