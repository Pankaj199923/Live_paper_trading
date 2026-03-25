# 🎮 Paper Trading Engine — Integration Guide

## Step 1: Copy the file
Place `tab_paper_trading.py` in your project folder (same level as `app.py`).

## Step 2: Edit app.py

### Add import (after the other tab imports):
```python
import tab_paper_trading
```

### Add tab to the tab list (add 12th tab):
```python
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "OPTION CHAIN",
    "SMART MONEY",
    "POSITIONS",
    "S&R / PAIN",
    "AI ADVISOR",
    "TRADE LOG",
    "GREEKS LAB",
    "\U0001f5a5 STOCKS",
    "\U0001f4ca CHART",
    "\U0001f4f8 HISTORY",
    "\U0001f9fa BASKET",
    "\U0001f3ae PAPER",        # ← ADD THIS
])
```

### Add render call (at the bottom):
```python
with tab12: tab_paper_trading.render()
```

## Step 3: requirements.txt
All dependencies are already in your requirements.txt. No additions needed.

---

## Features Summary

| Feature | Details |
|---|---|
| **Virtual Account** | ₹5L default capital, configurable |
| **Live Prices** | Uses your existing Upstox option chain data |
| **Margin Simulation** | ~12% notional for SELL, full premium for BUY |
| **Cost Simulation** | ₹40/lot brokerage + STT on SELL |
| **Auto SL/Target** | Triggered on every rerender (5s cycle) |
| **AI Reasoning** | Claude explains WHY each trade was taken |
| **Equity Curve** | Visual account growth over time |
| **Journal** | Every trade with signals, technicals, custom note |
| **Templates** | Bull Put, Bear Call, ATM CE Buy, ATM PE Buy |
| **Export** | CSV trades + JSON AI journal |
| **Reset** | Two-step confirmation before clearing |

## How AI Reasoning Works

When you click **🧠 AI Reasoning** on any trade, Claude generates:
- **Trade Setup Summary** — one-line punchy description
- **Why This Trade** — primary logic with specific prices
- **Technical Justification** — RSI, EMA, VWAP, MACD breakdown
- **Options Flow Logic** — OI, PCR, GEX, strike selection reasoning
- **Risk Management** — SL rationale + invalidation conditions
- **Target Rationale** — magnet levels, R:R justification
- **Trade Score** — 1–10 quality rating with verdict

## Paper Trading Workflow

```
Tab 1 (Load OC) → Tab PAPER → Execute Trade
                              ↓
                    AI generates reasoning
                              ↓
                    Monitor live P&L + SL/Target
                              ↓
                    Auto-close or manual close
                              ↓
                    Review in Full Journal
                              ↓
                    Export CSV / JSON
```