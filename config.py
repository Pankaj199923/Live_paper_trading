import streamlit as st
import pandas as pd
import os
import pytz
from datetime import datetime, time as dtime

# ======================
# 🔐 UPSTOX TOKEN
# ======================
try:
    ACCESS_TOKEN = st.secrets["UPSTOX_TOKEN"]
except KeyError:
    st.error("""
## ⚠️ Missing Secret: UPSTOX_TOKEN

Your Upstox API token has not been configured.

**To fix this on Streamlit Cloud:**
1. Go to your app → click **⋮ (three dots)** in the bottom-right → **Settings**
2. Click the **Secrets** tab
3. Paste the following and replace with your real token:

```toml
UPSTOX_TOKEN = "your_upstox_access_token_here"
```

4. Click **Save** — the app will restart automatically.
""")
    st.stop()

# ======================
# 🕒 MARKET TIME
# ======================
IST = pytz.timezone("Asia/Kolkata")
now_ist    = datetime.now(IST).time()
now_ist_dt = datetime.now(IST)

MARKET_OPEN_TIME  = dtime(9, 19)
MARKET_CLOSE_TIME = dtime(15, 30)
MARKET_OPEN = MARKET_OPEN_TIME <= now_ist <= MARKET_CLOSE_TIME

if now_ist < MARKET_OPEN_TIME:
    session_label = "PRE-MARKET"
    session_color = "#ffd600"
elif now_ist > MARKET_CLOSE_TIME:
    session_label = "POST-MARKET"
    session_color = "#7fa8c8"
else:
    session_label = "LIVE"
    session_color = "#00e676"

# ======================
# 📌 INDEX CONFIG
# ======================
df_list = [
    ['NSE_INDEX|Nifty 50',          '2026-03-17'],
    ['NSE_INDEX|Nifty Fin Service',  '2026-03-30'],
    ['NSE_INDEX|Nifty Bank',         '2026-03-30'],
    ['BSE_INDEX|SENSEX',             '2026-03-12'],
    ['BSE_INDEX|BANKEX',             '2026-03-25'],
]
df_indices = pd.DataFrame(df_list, columns=['index', 'expiry'])

# Short display names
INDEX_SHORT = {
    'NSE_INDEX|Nifty 50':          'NIFTY',
    'NSE_INDEX|Nifty Fin Service':  'FINNIFTY',
    'NSE_INDEX|Nifty Bank':         'BANKNIFTY',
    'BSE_INDEX|SENSEX':             'SENSEX',
    'BSE_INDEX|BANKEX':             'BANKEX',
}

LOT_SIZES = {
    'NSE_INDEX|Nifty 50':          65,
    'BSE_INDEX|SENSEX':            20,
    'NSE_INDEX|Nifty Bank':        30,
    'BSE_INDEX|BANKEX':            30,
    'NSE_INDEX|Nifty Fin Service': 60,
}

# ======================
# 📁 FILE PATHS
# ======================
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
TRADE_FILE        = os.path.join(BASE_DIR, "executed_trades.csv")
TODAY_TRADES_FILE = os.path.join(BASE_DIR, "today_trades.csv")
CLOSED_POS_FILE   = os.path.join(BASE_DIR, "closed_positions.csv")
INSTRUMENT_FILE   = os.path.join(BASE_DIR, "NSECMI.csv")
AI_LOG_FILE       = os.path.join(BASE_DIR, "ai_trade_log.csv")
SNAPSHOT_DIR      = os.path.join(BASE_DIR, "oc_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ======================
# 📌 INSTRUMENT FILE
# ======================
# Try current dir first, then parent dir (original single-file layout)
_instrument_candidates = [
    os.path.join(BASE_DIR, "NSECMI.csv"),
    os.path.join(os.path.dirname(BASE_DIR), "NSECMI.csv"),
]
INSTRUMENT_FILE = next((p for p in _instrument_candidates if os.path.exists(p)), _instrument_candidates[0])

try:
    instrument_df = pd.read_csv(INSTRUMENT_FILE, dtype=str)
    instrument_df = instrument_df[instrument_df["Exchange"] == "NSECM"]
    instrument_df = instrument_df[instrument_df["Trading Symbol"].str.endswith("-EQ", na=False)]
    instrument_df = instrument_df[["Trading Symbol", "ISIN"]]
    instrument_df["Trading Symbol"] = instrument_df["Trading Symbol"].str.replace("-EQ", "", regex=False)
    instrument_df.columns = ["Symbol", "ISIN"]
    instrument_df["instrument_key"] = "NSE_EQ|" + instrument_df["ISIN"]
    instrument_df = instrument_df.dropna().drop_duplicates().sort_values("Symbol").reset_index(drop=True)
except Exception:
    instrument_df = pd.DataFrame(columns=["Symbol", "ISIN", "instrument_key"])


# ======================
# 📊 TIMEFRAME OPTIONS
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
