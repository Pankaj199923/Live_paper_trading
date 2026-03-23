# QuantDesk Pro — Modular Structure

## File Overview

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~391 | Main entry point — page config, CSS, session state, header, tab routing |
| `config.py` | ~96 | Constants — token, timezone, index config, file paths, lot sizes |
| `utils.py` | ~332 | Utility functions — CSV helpers, ATM/lot helpers, position calc, snapshot, UI components |
| `api.py` | ~155 | API fetch functions with `@st.cache_data` — LTP, expiries, option chain, intraday candles |
| `analytics.py` | ~638 | Computation — Black-Scholes, Greeks, IV, signal scores, AI trade gen, alerts, Claude setup |
| `chart_utils.py` | ~420 | Charting — technicals, order flow, liquidity sweeps, order blocks, FVG, BOS/CHOCH |
| `tab1_option_chain.py` | ~195 | Tab 1 — Live Option Chain |
| `tab2_smart_money.py` | ~307 | Tab 2 — Smart Money + GEX |
| `tab3_positions.py` | ~161 | Tab 3 — Positions & Net Book |
| `tab4_snr_pain.py` | ~140 | Tab 4 — S&R + Max Pain |
| `tab5_ai_advisor.py` | ~441 | Tab 5 — AI Advisor |
| `tab6_trade_log.py` | ~133 | Tab 6 — Trade Log |
| `tab7_greeks_lab.py` | ~124 | Tab 7 — Greeks Lab |
| `tab8_stocks.py` | ~195 | Tab 8 — Stocks Terminal |
| `tab9_chart.py` | ~704 | Tab 9 — Live Chart + Order Flow |
| `tab10_history.py` | ~325 | Tab 10 — Option Chain History |
| `tab11_basket.py` | ~373 | Tab 11 — Basket Trade Builder |

## Dependency Graph

```
app.py
  ├── config.py          (constants, secrets)
  ├── utils.py           (helpers)  → config
  ├── api.py             (fetchers) → config
  ├── analytics.py       (math/AI)  → config, utils
  ├── chart_utils.py     (charts)   → (pure pandas/numpy)
  └── tab*.py            (UI tabs)  → config, utils, api, analytics, chart_utils
```

## Run

```bash
streamlit run app.py
```

## Notes
- Each tab is a module with a single `render()` function called by `app.py`
- All shared constants live in `config.py` — update expiry dates, lot sizes, etc. there
- All `@st.cache_data` fetch functions are in `api.py`
- To add a new tab: create `tab12_name.py` with a `render()` function, then import and wire it in `app.py`
