import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from config import ACCESS_TOKEN, IST, TF_OPTIONS, TF_RESAMPLE

# ======================
# 🌐 API — FETCH FUNCTIONS
# ======================
@st.cache_data(ttl=5)
def fetch_ltp(keys):
    url     = "https://api.upstox.com/v3/market-quote/ltp"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    if isinstance(keys, list): keys = ",".join(keys)
    params  = {"instrument_key": keys}
    try:
        r    = requests.get(url, headers=headers, params=params, timeout=5)
        data = r.json().get("data", {})
        rows = [{"Index": k.replace("%7C", "|"), "Spot Price": v.get("last_price", 0)} for k, v in data.items()]
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame(columns=['Index', 'Spot Price'])

@st.cache_data(ttl=300)
def fetch_available_expiries(index_key):
    url     = "https://api.upstox.com/v2/option/contract"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    params  = {"instrument_key": index_key}
    try:
        r    = requests.get(url, headers=headers, params=params, timeout=8)
        data = r.json().get("data", [])
        return sorted(set(item.get("expiry", "") for item in data if item.get("expiry")))
    except:
        return []

@st.cache_data(ttl=5)
def fetch_option_chain(index_key, expiry):
    url     = "https://api.upstox.com/v2/option/chain"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    params  = {"mode": "option_chain", "instrument_key": index_key, "expiry_date": expiry}
    try:
        r    = requests.get(url, headers=headers, params=params, timeout=8)
        raw  = r.json()
        if raw.get("status") == "error" or raw.get("errors"):
            err = raw.get("errors", [{}])[0].get("message", str(raw))
            st.error(f"🔴 API Error: {err}")
            return pd.DataFrame()
        data = raw.get("data", [])
        if not data:
            st.warning(f"⚠️ Empty data for {index_key} expiry {expiry}")
            return pd.DataFrame()
        rows = []
        ts   = datetime.now(IST).strftime("%H:%M:%S")
        for item in data:
            call = item.get("call_options", {})
            put  = item.get("put_options",  {})
            co   = call.get("market_data",   {})
            cg   = call.get("option_greeks", {})
            po   = put.get("market_data",    {})
            pg   = put.get("option_greeks",  {})
            rows.append({
                "Time":       ts,
                "Strike":     item.get("strike_price", 0),
                "CE_LTP":     co.get("ltp",     0),
                "CE_OI":      co.get("oi",      0),
                "CE_Prev_OI": co.get("prev_oi", 0),
                "CE_Volume":  co.get("volume",  0),
                "CE_IV":      cg.get("iv",      0),
                "CE_Delta":   cg.get("delta",   0),
                "CE_Gamma":   cg.get("gamma",   0),
                "CE_Theta":   cg.get("theta",   0),
                "CE_Vega":    cg.get("vega",    0),
                "PE_LTP":     po.get("ltp",     0),
                "PE_OI":      po.get("oi",      0),
                "PE_Prev_OI": po.get("prev_oi", 0),
                "PE_Volume":  po.get("volume",  0),
                "PE_IV":      pg.get("iv",      0),
                "PE_Delta":   pg.get("delta",   0),
                "PE_Gamma":   pg.get("gamma",   0),
                "PE_Theta":   pg.get("theta",   0),
                "PE_Vega":    pg.get("vega",    0),
            })
        df_oc = pd.DataFrame(rows)
        if df_oc.empty: return df_oc

        df_oc["CE_OI_Change"]   = df_oc["CE_OI"] - df_oc["CE_Prev_OI"]
        df_oc["PE_OI_Change"]   = df_oc["PE_OI"] - df_oc["PE_Prev_OI"]
        df_oc["CE_OI_Change_%"] = (df_oc["CE_OI_Change"] / df_oc["CE_Prev_OI"].replace(0, np.nan) * 100).fillna(0)
        df_oc["PE_OI_Change_%"] = (df_oc["PE_OI_Change"] / df_oc["PE_Prev_OI"].replace(0, np.nan) * 100).fillna(0)
        df_oc["CE_Volume_Avg"]  = df_oc["CE_Volume"].rolling(5, min_periods=1).mean()
        df_oc["PE_Volume_Avg"]  = df_oc["PE_Volume"].rolling(5, min_periods=1).mean()
        df_oc["CE_Vol_Spike_%"] = (df_oc["CE_Volume"] / df_oc["CE_Volume_Avg"].replace(0, 1)) * 100
        df_oc["PE_Vol_Spike_%"] = (df_oc["PE_Volume"] / df_oc["PE_Volume_Avg"].replace(0, 1)) * 100
        df_oc["CE_IV_Change_%"] = df_oc["CE_IV"].pct_change() * 100
        df_oc["PE_IV_Change_%"] = df_oc["PE_IV"].pct_change() * 100
        df_oc["CE_GEX"]         = df_oc["CE_Gamma"] * df_oc["CE_OI"]
        df_oc["PE_GEX"]         = df_oc["PE_Gamma"] * df_oc["PE_OI"]
        df_oc["NET_GEX"]        = df_oc["CE_GEX"]   - df_oc["PE_GEX"]
        df_oc["IV_Skew"]        = (df_oc["PE_IV"] / df_oc["CE_IV"].replace(0, np.nan)).fillna(1).round(3)
        df_oc["Strike_PCR"]     = (df_oc["PE_OI"] / df_oc["CE_OI"].replace(0, np.nan)).fillna(0).round(3)
        # Net premium (synthetic position value)
        df_oc["Net_Premium"]    = df_oc["PE_LTP"] - df_oc["CE_LTP"]
        # Pain index contribution
        df_oc["CE_OI_x_Strike"] = df_oc["CE_OI"] * df_oc["Strike"]
        df_oc["PE_OI_x_Strike"] = df_oc["PE_OI"] * df_oc["Strike"]

        float_cols = df_oc.select_dtypes(include=['float64','float32']).columns
        df_oc[float_cols] = df_oc[float_cols].round(4)
        return df_oc
    except Exception as e:
        st.error(f"Option chain error: {e}")
        return pd.DataFrame()

# ======================
# 📊 NET BOOK

# ======================
# 📈 INTRADAY CANDLES
# ======================
@st.cache_data(ttl=60)
def fetch_intraday_candles(instrument_key, timeframe="1minute"):
    """Fetch OHLCV from Upstox for requested timeframe.
    - 1min, 30min, 1hour  → native Upstox intraday endpoint
    - 3/5/10/15min        → fetched as 1-min then resampled
    """
    try:
        key_encoded = instrument_key.replace("|", "%7C")
        api_tf = "1minute" if timeframe in TF_RESAMPLE else timeframe
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{key_encoded}/{api_tf}"
        headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        r = requests.get(url, headers=headers, timeout=8)
        data = r.json()
        candles = data.get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume","oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Resample if needed
        if timeframe in TF_RESAMPLE:
            rule = TF_RESAMPLE[timeframe]
            df = df.set_index("timestamp")
            df = df.resample(rule, label="left", closed="left").agg(
                open=("open","first"), high=("high","max"),
                low=("low","min"),   close=("close","last"),
                volume=("volume","sum"), oi=("oi","last")
            ).dropna(subset=["open"]).reset_index()
        return df
    except Exception:
        return pd.DataFrame()

# ======================
# 📐 TECHNICAL INDICATORS
# ======================
