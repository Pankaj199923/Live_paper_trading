import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import (ACCESS_TOKEN, IST, now_ist, now_ist_dt, MARKET_OPEN,
                    session_label, session_color, df_indices, INDEX_SHORT,
                    LOT_SIZES, BASE_DIR, TRADE_FILE, TODAY_TRADES_FILE,
                    CLOSED_POS_FILE, INSTRUMENT_FILE, AI_LOG_FILE, SNAPSHOT_DIR,
                    instrument_df)

# ======================
# 📂 CSV HELPERS
# ======================
def load_csv_safe(fp):
    if os.path.exists(fp):
        try: return pd.read_csv(fp)
        except: return pd.DataFrame()
    return pd.DataFrame()

def load_csv_as_list(fp):
    df_t = load_csv_safe(fp)
    if isinstance(df_t, pd.DataFrame) and not df_t.empty:
        return df_t.to_dict("records")
    return []

def save_list_to_csv(data_list, fp):
    try:
        if data_list: pd.DataFrame(data_list).to_csv(fp, index=False)
    except: pass

# ======================

# ======================
# 🔧 HELPERS
# ======================
# ======================
# 🔧 HELPERS
# ======================
def get_atm_strike(spot_price, index_key):
    name = index_key.upper()
    step = 100 if ("SENSEX" in name or "BANK" in name) else 50
    return round(spot_price / step) * step

def get_lot_size(index_key):
    return LOT_SIZES.get(index_key, 50)

def idx_short(index_key):
    return INDEX_SHORT.get(index_key, index_key.split("|")[-1])

def compute_grand_total():
    oc_trade = st.session_state.get("current_option_chain", pd.DataFrame())
    positions, open_pnl, _ = calculate_net_book(st.session_state.today_trades, oc_trade)
    today_str = datetime.now().strftime("%Y-%m-%d")
    closed_pnl = sum(
        t.get("final_pnl", 0) for t in st.session_state.closed_positions
        if t.get("Date") == today_str
    )
    return open_pnl, closed_pnl, open_pnl + closed_pnl

def pnl_color(val):
    if val > 0: return "#00e676"
    if val < 0: return "#ff3d57"
    return "#7fa8c8"

def pnl_badge(val):
    c = pnl_color(val)
    sign = "▲" if val > 0 else "▼" if val < 0 else "–"
    return f"<span style='color:{c};font-family:JetBrains Mono,monospace;font-weight:700;'>{sign} ₹{abs(val):,.0f}</span>"

# ======================
# 🌐 API
# ======================

# ======================
# 📊 NET BOOK / POSITIONS
# ======================
def calculate_net_book(today_trades, oc_trade):
    positions, total_net_pnl, total_net_qty = {}, 0, 0
    if not isinstance(today_trades, list): return {}, 0, 0

    for trade in today_trades:
        if trade.get("Status") != "OPEN": continue
        index_clean = trade['Index'].split("|")[-1]
        symbol_key  = f"{index_clean}|{trade['Type']}|{trade['Strike']}"
        qty   = float(trade.get("Qty", 1))
        entry = float(trade.get("Entry", 0))
        action= trade.get("Action", "BUY")
        if symbol_key not in positions:
            positions[symbol_key] = {"symbol": symbol_key, "buy_qty": 0, "sell_qty": 0,
                                     "buy_value": 0, "sell_value": 0, "ltp": entry}
        if action == "BUY":
            positions[symbol_key]["buy_qty"]   += qty
            positions[symbol_key]["buy_value"] += entry * qty
        else:
            positions[symbol_key]["sell_qty"]   += qty
            positions[symbol_key]["sell_value"] += entry * qty

    for symbol_key, pos in positions.items():
        parts      = symbol_key.split("|")
        opt_type   = parts[1] if len(parts) >= 2 else "CE"
        strike     = float(parts[2]) if len(parts) >= 3 else 0
        buy_qty    = pos["buy_qty"]; sell_qty = pos["sell_qty"]
        net_qty    = buy_qty - sell_qty
        buy_avg    = pos["buy_value"]  / buy_qty  if buy_qty  > 0 else 0
        sell_avg   = pos["sell_value"] / sell_qty if sell_qty > 0 else 0
        ltp_val    = buy_avg if net_qty >= 0 else sell_avg
        if not oc_trade.empty:
            row = oc_trade[oc_trade["Strike"] == strike]
            if not row.empty:
                col     = "CE_LTP" if opt_type == "CE" else "PE_LTP"
                raw_ltp = row[col].values[0]
                ltp_val = float(raw_ltp) if pd.notna(raw_ltp) else ltp_val
        pos["ltp"] = ltp_val
        if net_qty > 0:    current_pnl = (ltp_val - buy_avg)  * net_qty
        elif net_qty < 0:  current_pnl = (sell_avg - ltp_val) * abs(net_qty)
        else:              current_pnl = 0
        pos.update({"net_qty": net_qty, "buy_avg": buy_avg,
                    "sell_avg": sell_avg, "current_pnl": current_pnl})
        total_net_pnl += current_pnl
        total_net_qty += abs(net_qty)

    return positions, total_net_pnl, total_net_qty

def close_position(symbol_key, positions):
    pos = positions.get(symbol_key)
    if not pos: return
    ltp       = pos["ltp"]
    today_str = datetime.now().strftime("%Y-%m-%d")
    exit_time = datetime.now(IST).strftime("%H:%M:%S")
    entry_time= ""
    for trade in st.session_state.today_trades:
        index_clean = trade['Index'].split("|")[-1]
        key = f"{index_clean}|{trade['Type']}|{trade['Strike']}"
        if key == symbol_key and trade["Status"] == "OPEN":
            trade["Exit_Price"] = ltp
            trade["Exit_Time"]  = exit_time
            trade["Status"]     = "CLOSED"
            entry_time = trade.get("Entry_Time", "")
    st.session_state.closed_positions.append({
        "symbol": symbol_key, "Date": today_str,
        "net_qty": abs(pos["net_qty"]), "close_price": ltp,
        "final_pnl": round(pos["current_pnl"], 2),
        "entry_time": entry_time, "exit_time": exit_time,
    })
    save_list_to_csv(st.session_state.closed_positions, CLOSED_POS_FILE)
    st.success(f"✅ {symbol_key} CLOSED | P&L: ₹{pos['current_pnl']:.2f}")
    st.rerun()

# ======================
# 📐 BLACK-SCHOLES
# ======================

# ======================
# 📸 SNAPSHOT HELPERS
# ======================
def _daily_snapshot_path(index_key):
    """Return path to today's single snapshot Parquet file for this index."""
    today   = datetime.now(IST).strftime("%Y-%m-%d")
    idx_tag = index_key.replace("|", "_").replace(" ", "-")
    fname   = f"{idx_tag}_{today}.parquet"
    return os.path.join(SNAPSHOT_DIR, fname)

def save_oc_snapshot(oc_df, spot_price, index_key, expiry):
    """Append this minute's option chain rows to today's single Parquet file."""
    try:
        snap             = oc_df.copy()
        snap["_time"]    = datetime.now(IST).strftime("%H:%M") + ":00"
        snap["_spot"]    = spot_price
        snap["_index"]   = index_key
        snap["_expiry"]  = expiry
        fpath = _daily_snapshot_path(index_key)
        # Parquet doesn't support append — read existing, concat, rewrite
        if os.path.exists(fpath):
            existing = pd.read_parquet(fpath)
            snap = pd.concat([existing, snap], ignore_index=True)
        snap.to_parquet(fpath, index=False, compression="snappy")
        return True
    except Exception:
        return False

def list_daily_files(index_key=None):
    """Return sorted list of (date_label, filepath) for all daily Parquet files."""
    try:
        files = sorted(
            [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".parquet")],
            reverse=True
        )
        result = []
        for f in files:
            if index_key:
                idx_tag = index_key.replace("|", "_").replace(" ", "-")
                if not f.startswith(idx_tag):
                    continue
            # filename: IDXTAG_YYYY-MM-DD.parquet
            date_part = f.replace(".parquet", "").split("_")[-1]   # last segment
            try:
                d = datetime.strptime(date_part, "%Y-%m-%d")
                label = d.strftime("%d/%m/%Y")
            except:
                label = date_part
            result.append((label, os.path.join(SNAPSHOT_DIR, f)))
        return result
    except:
        return []

def list_minute_times(fpath, index_key=None):
    """Return sorted unique HH:MM times available in a daily Parquet file."""
    try:
        df = pd.read_parquet(fpath, columns=["_time"])
        times = sorted(df["_time"].dropna().unique().tolist())
        return times
    except:
        return []

def load_snapshot(fpath, time_str):
    """Load one minute's rows from the daily Parquet file. Returns (oc_df, spot, index_key, expiry, ts)."""
    try:
        df = pd.read_parquet(fpath)
        df = df[df["_time"] == time_str].copy()
        if df.empty:
            return pd.DataFrame(), 0, "", "", ""
        spot      = float(df["_spot"].iloc[0])  if "_spot"   in df.columns else 0
        index_key = str(df["_index"].iloc[0])   if "_index"  in df.columns else ""
        expiry    = str(df["_expiry"].iloc[0])  if "_expiry" in df.columns else ""
        ts        = f"{df['_time'].iloc[0]}"    if "_time"   in df.columns else ""
        meta_cols = ["_time","_spot","_index","_expiry"]
        df = df.drop(columns=[c for c in meta_cols if c in df.columns])
        df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
        df = df.dropna(subset=["Strike"])
        return df, spot, index_key, expiry, ts
    except Exception:
        return pd.DataFrame(), 0, "", "", ""

# ======================
# 🌡️ MAX PAIN CALCULATION  [NEW PRO FEATURE]

# ======================
# 📐 MAX PAIN & PORTFOLIO GREEKS
# ======================
def calculate_max_pain(oc):
    """Calculate Options Max Pain — the strike where options expire worthless for most buyers."""
    strikes = oc["Strike"].tolist()
    pain    = {}
    for test_strike in strikes:
        total_pain = 0
        for _, row in oc.iterrows():
            k = row["Strike"]
            # CE writers pain: if spot above strike, CE buyers profit → writers lose
            if test_strike > k:
                total_pain += (test_strike - k) * row["CE_OI"]
            # PE writers pain: if spot below strike, PE buyers profit → writers lose
            if test_strike < k:
                total_pain += (k - test_strike) * row["PE_OI"]
        pain[test_strike] = total_pain
    max_pain_strike = min(pain, key=pain.get)
    return max_pain_strike, pain

# ======================
# 📐 PORTFOLIO GREEKS  [NEW PRO FEATURE]
# ======================
def calculate_portfolio_greeks(trades, spot_price, index_key, days_to_expiry):
    """Aggregate greeks across all open positions."""
    from scipy.stats import norm as _norm
    import numpy as _np

    def _bs_greeks(S, K, T, r, sigma, opt='c'):
        if sigma <= 0 or T <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
        d1 = (_np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * _np.sqrt(T))
        d2 = d1 - sigma * _np.sqrt(T)
        nd1 = _norm.pdf(d1)
        delta = _norm.cdf(d1) if opt == 'c' else _norm.cdf(d1) - 1
        gamma = nd1 / (S * sigma * _np.sqrt(T))
        vega  = S * nd1 * _np.sqrt(T) / 100
        theta = (-(S * nd1 * sigma) / (2 * _np.sqrt(T)) -
                 r * K * _np.exp(-r * T) * (_norm.cdf(d2) if opt == 'c' else _norm.cdf(-d2))) / 365
        return {"delta": round(delta, 4), "gamma": round(gamma, 6),
                "theta": round(theta, 4), "vega":  round(vega, 4)}

    T = max(days_to_expiry, 1) / 365
    r = 0.065
    net_delta = net_gamma = net_theta = net_vega = 0
    for trade in trades:
        if trade.get("Status") != "OPEN": continue
        K      = float(trade.get("Strike", spot_price))
        qty    = float(trade.get("Qty", 1))
        action = trade.get("Action", "SELL")
        opt_t  = trade.get("Type", "CE")
        iv_est = 0.2  # fallback IV
        sign   = 1 if action == "BUY" else -1
        opt_bs = 'c' if opt_t == "CE" else 'p'
        g      = _bs_greeks(spot_price, K, T, r, iv_est, opt_bs)
        net_delta += sign * g['delta'] * qty
        net_gamma += sign * g['gamma'] * qty
        net_theta += sign * g['theta'] * qty
        net_vega  += sign * g['vega']  * qty
    return {"Δ Delta": round(net_delta, 2), "Γ Gamma": round(net_gamma, 4),
            "Θ Theta": round(net_theta, 2), "V Vega":  round(net_vega, 2)}

# ======================
# 🤖 PRO AI TRADE GENERATOR — Defined-Risk Strategies

# ======================
# 🎨 UI COMPONENTS
# ======================
def section_header(title, subtitle=""):
    sub_html = f'<div style="font-family:Barlow,sans-serif;font-size:12px;color:#7fa8c8;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div style="border-left:3px solid #ff8c00;padding:4px 12px;margin:16px 0 10px 0;background:linear-gradient(90deg,#111920,transparent);">'
        f'<div style="font-family:Barlow Condensed,sans-serif;font-size:18px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#e8f4ff;">{title}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True
    )

def metric_card(label, value, sub="", color="#ff8c00", width="100%"):
    sub_html = f'<div style="font-family:Barlow,sans-serif;font-size:11px;color:#3a6080;margin-top:2px;">{sub}</div>' if sub else ""
    return (
        f'<div style="background:#111920;border:1px solid #1e3040;border-top:2px solid {color};'
        f'border-radius:3px;padding:10px 14px;min-width:100px;flex:1;box-sizing:border-box;">'
        f'<div style="font-family:Barlow Condensed,sans-serif;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#7fa8c8;">{label}</div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:{color};margin-top:4px;">{value}</div>'
        f'{sub_html}</div>'
    )

def metrics_row(cards_html):
    """Wrap metric cards in a flex row and render."""
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">{cards_html}</div>',
        unsafe_allow_html=True
    )

def flow_badge(flow):
    cfg = {
        "Bullish": ("#00e676", "#051a0e", "&#9650; BULLISH"),
        "Bearish": ("#ff3d57", "#1a0508", "&#9660; BEARISH"),
        "Range":   ("#ffd600", "#1a1000", "&#8596; RANGE"),
        "Choppy":  ("#7fa8c8", "#00101a", "&#8776; CHOPPY"),
    }
    fg, bg, label = cfg.get(flow, ("#7fa8c8","#00101a", flow.upper()))
    return (f'<span style="background:{bg};border:1px solid {fg};color:{fg};'
            f'font-family:Barlow Condensed,sans-serif;font-size:13px;font-weight:700;'
            f'letter-spacing:2px;padding:4px 12px;border-radius:2px;">{label}</span>')

def score_bar(score, max_score=20, color="#00e676"):
    pct = min(score / max_score * 100, 100)
    return (f'<div style="background:#0d1117;border:1px solid #1e3040;border-radius:2px;height:6px;width:100%;">'
            f'<div style="background:{color};width:{pct}%;height:100%;border-radius:2px;'
            f'box-shadow:0 0 6px {color}44;transition:width 0.3s;"></div></div>')
