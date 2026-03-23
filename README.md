# QuantDesk Pro — Options Terminal

A Bloomberg-style live options trading terminal built with Streamlit + Upstox API.

## 🚀 Deploy on Streamlit Community Cloud

### 1. Push to GitHub
Make sure your repo contains **all files including `NSECMI.csv`**.

### 2. Connect on share.streamlit.io
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click **New app**
- Select your GitHub repo
- Set **Main file path** → `app.py`
- Click **Deploy**

### 3. Add your secret token
After deploy, go to **⋮ → Settings → Secrets** and paste:

```toml
UPSTOX_TOKEN = "your_upstox_access_token_here"
```

Then click **Save** — the app will restart automatically.

---

## 🗂 File Structure

| File | Purpose |
|------|---------|
| `app.py` | Entry point — run this |
| `config.py` | Constants, token, index config |
| `utils.py` | CSV helpers, UI components |
| `api.py` | Upstox API fetch functions |
| `analytics.py` | Black-Scholes, signals, Greeks |
| `chart_utils.py` | Technicals, order flow, ICT |
| `tab1_option_chain.py` | Live Option Chain |
| `tab2_smart_money.py` | Smart Money + GEX |
| `tab3_positions.py` | Positions & Net Book |
| `tab4_snr_pain.py` | S&R + Max Pain |
| `tab5_ai_advisor.py` | AI Signal Engine |
| `tab6_trade_log.py` | Trade Log |
| `tab7_greeks_lab.py` | Greeks Lab |
| `tab8_stocks.py` | Stocks Terminal |
| `tab9_chart.py` | Live Chart + Order Flow |
| `tab10_history.py` | Option Chain History |
| `tab11_basket.py` | Basket Trade Builder |
| `NSECMI.csv` | NSE instrument master (required for Stocks tab) |

## ⚠️ Important Notes

- **`NSECMI.csv`** must be in the repo root (same folder as `app.py`)
- **Never commit** `.streamlit/secrets.toml` — it's in `.gitignore`
- The `oc_snapshots/` folder is created automatically at runtime
- Runtime CSVs (`executed_trades.csv`, etc.) are **ephemeral** on Community Cloud — they reset on each redeploy. For persistent trade history, use an external DB.

## Local Development

```bash
pip install -r requirements.txt
# Add your token to .streamlit/secrets.toml
streamlit run app.py
```
