# ============================================================
# ADDON: Paste this into tab8_stocks.py
# 1. Add _render_market_heatmap() function before render()
# 2. At the bottom of render(), add:
#        st.markdown("---")
#        _render_market_heatmap()
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

# ── NIFTY 50 with sectors (ISINs map to instrument_df) ────────────────────
NIFTY50_META = [
    # (Symbol, Sector, approx_mcap_rank)
    ("RELIANCE",     "Energy",         1),
    ("TCS",          "IT",             2),
    ("HDFCBANK",     "Banking",        3),
    ("BHARTIARTL",   "Telecom",        4),
    ("ICICIBANK",    "Banking",        5),
    ("INFOSYS",      "IT",             6),
    ("SBIN",         "Banking",        7),
    ("HINDUNILVR",   "FMCG",          8),
    ("ITC",          "FMCG",          9),
    ("LT",           "Infra",         10),
    ("KOTAKBANK",    "Banking",       11),
    ("HCLTECH",      "IT",            12),
    ("BAJFINANCE",   "NBFC",          13),
    ("AXISBANK",     "Banking",       14),
    ("WIPRO",        "IT",            15),
    ("ASIANPAINT",   "Paints",        16),
    ("MARUTI",       "Auto",          17),
    ("M&M",          "Auto",          18),
    ("SUNPHARMA",    "Pharma",        19),
    ("ULTRACEMCO",   "Cement",        20),
    ("NTPC",         "Power",         21),
    ("POWERGRID",    "Power",         22),
    ("TITAN",        "Jewellery",     23),
    ("TECHM",        "IT",            24),
    ("JSWSTEEL",     "Metals",        25),
    ("TATASTEEL",    "Metals",        26),
    ("TATAMOTORS",   "Auto",          27),
    ("DRREDDY",      "Pharma",        28),
    ("ADANIENT",     "Conglomerate",  29),
    ("ADANIPORTS",   "Infra",         30),
    ("BAJAJFINSV",   "NBFC",          31),
    ("BAJAJ-AUTO",   "Auto",          32),
    ("CIPLA",        "Pharma",        33),
    ("DIVISLAB",     "Pharma",        34),
    ("EICHERMOT",    "Auto",          35),
    ("GRASIM",       "Diversified",   36),
    ("HDFCLIFE",     "Insurance",     37),
    ("HEROMOTOCO",   "Auto",          38),
    ("HINDALCO",     "Metals",        39),
    ("INDUSINDBK",   "Banking",       40),
    ("COALINDIA",    "Mining",        41),
    ("BEL",          "Defence",       42),
    ("ONGC",         "Energy",        43),
    ("NESTLEIND",    "FMCG",         44),
    ("SBILIFE",      "Insurance",     45),
    ("TATACONSUM",   "FMCG",         46),
    ("BPCL",         "Energy",        47),
    ("BRITANNIA",    "FMCG",         48),
    ("APOLLOHOSP",   "Healthcare",    49),
    ("SHRIRAMFIN",   "NBFC",         50),
]

SECTOR_COLORS = {
    "Banking":     "#1a4a6b",
    "IT":          "#1a3d5c",
    "FMCG":        "#1a5c2e",
    "Energy":      "#5c3d1a",
    "Auto":        "#4a1a5c",
    "Pharma":      "#5c1a2e",
    "NBFC":        "#1a4a3d",
    "Metals":      "#3d3d1a",
    "Infra":       "#1a2e5c",
    "Power":       "#2e1a5c",
    "Telecom":     "#5c1a4a",
    "Cement":      "#3d1a1a",
    "Jewellery":   "#4a3d1a",
    "Conglomerate":"#1a3d4a",
    "Insurance":   "#2e3d1a",
    "Mining":      "#3d2e1a",
    "Defence":     "#1a4a2e",
    "Healthcare":  "#4a1a1a",
    "Paints":      "#1a4a4a",
    "Diversified": "#3d1a3d",
}


@st.cache_data(ttl=30, show_spinner=False)
def _bulk_ltp_fetch(instrument_keys_str: str) -> dict:
    """Fetch LTP for up to 500 symbols in one API call."""
    try:
        from config import ACCESS_TOKEN
        r = requests.get(
            "https://api.upstox.com/v3/market-quote/ltp",
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            params={"instrument_key": instrument_keys_str},
            timeout=10,
        )
        data = r.json().get("data", {})
        result = {}
        for raw_key, v in data.items():
            clean = raw_key.replace("%7C", "|").replace("%7c", "|")
            price = v.get("last_price", 0)
            if price > 0:
                result[clean] = price
        return result
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _bulk_ohlc_fetch(instrument_keys_str: str) -> dict:
    """
    Fetch OHLC + prev_close for many symbols using the faster OHLC endpoint.
    Returns {instrument_key: {"open":x,"high":x,"low":x,"close":x,"prev_close":x}}
    """
    try:
        from config import ACCESS_TOKEN
        r = requests.get(
            "https://api.upstox.com/v2/market-quote/ohlc",
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            params={"instrument_key": instrument_keys_str, "interval": "1d"},
            timeout=12,
        )
        data = r.json().get("data", {})
        result = {}
        for raw_key, v in data.items():
            clean = raw_key.replace("%7C", "|").replace("%7c", "|")
            ohlc  = v.get("ohlc", {})
            result[clean] = {
                "open":       ohlc.get("open", 0),
                "high":       ohlc.get("high", 0),
                "low":        ohlc.get("low",  0),
                "close":      ohlc.get("close", 0),  # current LTP
                "prev_close": v.get("prev_close", 0) or ohlc.get("open", 0),
                "volume":     v.get("volume", 0),
            }
        return result
    except Exception:
        return {}


def _render_market_heatmap():
    """
    Full market heatmap section — Nifty 50 + user's watchlist.
    Add at bottom of tab8_stocks render().
    """
    from config import IST
    from utils import section_header, metrics_row, metric_card

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;
                border-bottom:1px solid #1e3040;padding-bottom:8px;margin:20px 0 14px 0;">
      <div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:22px;
                    font-weight:800;letter-spacing:3px;color:#e8f4ff;">
          MARKET <span style="color:#ff8c00;">HEATMAP</span></div>
        <div style="font-family:'Barlow',sans-serif;font-size:12px;color:#7fa8c8;margin-top:2px;">
          Nifty 50 · % change from previous close · Size = market cap weight</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────────────
    hm_c1, hm_c2, hm_c3, hm_c4 = st.columns([1, 1, 1, 3])
    with hm_c1:
        hm_universe = st.selectbox(
            "Universe",
            ["Nifty 50", "My Watchlist", "Nifty 50 + Watchlist"],
            key="hm_universe",
        )
    with hm_c2:
        hm_size_by = st.selectbox(
            "Size by",
            ["Market Cap Rank", "Volume", "Equal"],
            key="hm_size",
        )
    with hm_c3:
        hm_group = st.selectbox(
            "Group by",
            ["Sector", "None"],
            key="hm_group",
        )
    with hm_c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_hm = st.button(
            "🔄 Refresh Heatmap",
            use_container_width=False,
            type="primary",
            key="hm_refresh",
        )

    # ── Build universe ────────────────────────────────────────────────────
    # Pull watchlist from session state (set in render())
    watchlist_syms = st.session_state.get("watchlist8", [])

    target_syms = []
    if hm_universe == "Nifty 50":
        target_syms = [(s, sec, rank) for s, sec, rank in NIFTY50_META]
    elif hm_universe == "My Watchlist":
        target_syms = [(s, "Watchlist", i+1) for i, s in enumerate(watchlist_syms)]
    else:
        n50_set  = {s for s, _, _ in NIFTY50_META}
        extra    = [(s, "Watchlist", 99) for s in watchlist_syms if s not in n50_set]
        target_syms = [(s, sec, rank) for s, sec, rank in NIFTY50_META] + extra

    if not target_syms:
        st.warning("No symbols to display — add stocks to your watchlist or select Nifty 50.")
        return

    # ── Map symbols → instrument_keys ────────────────────────────────────
    from config import instrument_df as idf
    sym_to_key = dict(zip(idf["Symbol"], idf["instrument_key"]))

    rows_meta = []
    for sym, sector, rank in target_syms:
        ikey = sym_to_key.get(sym)
        if ikey:
            rows_meta.append({"symbol": sym, "sector": sector, "rank": rank, "ikey": ikey})

    if not rows_meta:
        st.error("Could not map any symbols to instrument keys. Check NSECMI.csv.")
        return

    # ── Fetch prices ──────────────────────────────────────────────────────
    all_keys_str = ",".join(r["ikey"] for r in rows_meta)

    # Use cache invalidation key so button forces a fresh fetch
    _cache_key = st.session_state.get("hm_cache_key", 0)
    if run_hm:
        _cache_key += 1
        st.session_state["hm_cache_key"] = _cache_key
        _bulk_ohlc_fetch.clear()
        _bulk_ltp_fetch.clear()

    with st.spinner("📡 Fetching live prices for heatmap…"):
        ohlc_data = _bulk_ohlc_fetch(all_keys_str)

    # Fallback to LTP-only if OHLC fails
    if not ohlc_data:
        with st.spinner("Falling back to LTP fetch…"):
            ltp_data = _bulk_ltp_fetch(all_keys_str)
    else:
        ltp_data = {}

    # ── Build DataFrame ───────────────────────────────────────────────────
    hm_rows = []
    for r in rows_meta:
        sym   = r["symbol"]
        ikey  = r["ikey"]
        sec   = r["sector"]
        rank  = r["rank"]

        if ikey in ohlc_data:
            d          = ohlc_data[ikey]
            ltp        = d["close"] or d.get("prev_close", 0)
            prev_close = d["prev_close"] or ltp
            day_high   = d["high"]
            day_low    = d["low"]
            day_open   = d["open"]
            volume     = d.get("volume", 0)
        elif ikey in ltp_data:
            ltp        = ltp_data[ikey]
            prev_close = st.session_state.get(f"prev_close_{sym}", ltp)
            day_high   = ltp
            day_low    = ltp
            day_open   = ltp
            volume     = 0
        else:
            # Try session state cache
            ltp        = st.session_state.get(f"prev_ltp_{sym}", 0)
            prev_close = st.session_state.get(f"prev_close_{sym}", ltp)
            day_high = day_low = day_open = ltp
            volume = 0

        if ltp <= 0:
            continue

        chg_pct = ((ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0
        chg_rs  = ltp - prev_close

        # Size value
        if hm_size_by == "Market Cap Rank":
            size_val = max(51 - rank, 1) * 10   # rank 1 → size 500, rank 50 → size 10
        elif hm_size_by == "Volume":
            size_val = max(volume, 1)
        else:
            size_val = 100

        hm_rows.append({
            "symbol":     sym,
            "sector":     sec,
            "ltp":        round(ltp, 2),
            "prev_close": round(prev_close, 2),
            "chg_pct":    round(chg_pct, 2),
            "chg_rs":     round(chg_rs, 2),
            "day_high":   round(day_high, 2),
            "day_low":    round(day_low,  2),
            "day_open":   round(day_open, 2),
            "volume":     volume,
            "size_val":   size_val,
            "rank":       rank,
        })

    if not hm_rows:
        st.warning("No price data retrieved. Check API token or try refreshing.")
        return

    df_hm = pd.DataFrame(hm_rows).sort_values("chg_pct", ascending=False)

    # ── Summary metrics ───────────────────────────────────────────────────
    total_n    = len(df_hm)
    gainers_n  = (df_hm["chg_pct"] > 0).sum()
    losers_n   = (df_hm["chg_pct"] < 0).sum()
    unch_n     = total_n - gainers_n - losers_n
    avg_chg    = df_hm["chg_pct"].mean()
    best_sym   = df_hm.iloc[0]
    worst_sym  = df_hm.iloc[-1]
    breadth_c  = "#00e676" if gainers_n > losers_n * 1.5 else "#ff3d57" if losers_n > gainers_n * 1.5 else "#ffd600"

    metrics_row(
        metric_card("UNIVERSE",    f"{total_n} stocks",        hm_universe, "#ff8c00") +
        metric_card("ADVANCERS",   f"{gainers_n}",             f"{gainers_n/total_n*100:.0f}% of universe", "#00e676") +
        metric_card("DECLINERS",   f"{losers_n}",              f"{losers_n/total_n*100:.0f}% of universe", "#ff3d57") +
        metric_card("AVG CHANGE",  f"{avg_chg:+.2f}%",        "Breadth signal", breadth_c) +
        metric_card("BEST",        f"{best_sym['symbol']}",    f"+{best_sym['chg_pct']:.2f}%", "#00e676") +
        metric_card("WORST",       f"{worst_sym['symbol']}",   f"{worst_sym['chg_pct']:.2f}%", "#ff3d57")
    )

    # Breadth bar
    adv_pct = gainers_n / total_n * 100
    dec_pct = losers_n  / total_n * 100
    unc_pct = 100 - adv_pct - dec_pct
    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;
                padding:10px 16px;margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;
                  font-family:'Barlow Condensed',sans-serif;font-size:10px;
                  letter-spacing:1px;margin-bottom:5px;">
        <span style="color:#00e676;">▲ {gainers_n} GAINERS ({adv_pct:.0f}%)</span>
        <span style="color:{breadth_c};font-weight:700;">MARKET BREADTH</span>
        <span style="color:#ff3d57;">{losers_n} LOSERS ({dec_pct:.0f}%) ▼</span>
      </div>
      <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;gap:1px;">
        <div style="width:{adv_pct:.0f}%;background:linear-gradient(90deg,#00e676,#00c853);border-radius:4px 0 0 4px;"></div>
        <div style="width:{unc_pct:.0f}%;background:#1e3040;"></div>
        <div style="width:{dec_pct:.0f}%;background:linear-gradient(90deg,#c62828,#ff3d57);border-radius:0 4px 4px 0;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── TREEMAP ───────────────────────────────────────────────────────────
    # Color scale: deep red → neutral → deep green
    df_hm["label_text"] = df_hm.apply(
        lambda r: f"<b>{r['symbol']}</b><br>{r['chg_pct']:+.2f}%<br>₹{r['ltp']:,.0f}",
        axis=1
    )

    # Custom hover
    df_hm["hover"] = df_hm.apply(lambda r: (
        f"<b>{r['symbol']}</b><br>"
        f"LTP: ₹{r['ltp']:,.2f}<br>"
        f"Chg: {r['chg_pct']:+.2f}%  (₹{r['chg_rs']:+.2f})<br>"
        f"Prev Close: ₹{r['prev_close']:,.2f}<br>"
        f"High: ₹{r['day_high']:,.2f}  Low: ₹{r['day_low']:,.2f}<br>"
        f"Volume: {r['volume']:,}"
    ), axis=1)

    # Clamp color range to ±3% for better color differentiation
    color_max = min(df_hm["chg_pct"].abs().quantile(0.9), 3.0)
    color_max = max(color_max, 0.5)

    if hm_group == "Sector":
        # Sector-grouped treemap
        fig_treemap = go.Figure(go.Treemap(
            ids=[f"root"] + list(df_hm["sector"].unique()) + list(df_hm["symbol"]),
            labels=(
                ["ALL STOCKS"]
                + list(df_hm["sector"].unique())
                + list(df_hm["symbol"])
            ),
            parents=(
                [""]
                + ["ALL STOCKS"] * df_hm["sector"].nunique()
                + list(df_hm["sector"])
            ),
            values=(
                [0]
                + [df_hm[df_hm["sector"] == s]["size_val"].sum() for s in df_hm["sector"].unique()]
                + list(df_hm["size_val"])
            ),
            customdata=df_hm[["chg_pct","ltp","chg_rs","prev_close","day_high","day_low"]].values,
            text=list(df_hm["symbol"]),
            texttemplate=(
                "<b>%{label}</b><br>"
                "%{customdata[0]:+.2f}%<br>"
                "₹%{customdata[1]:,.0f}"
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "LTP: ₹%{customdata[1]:,.2f}<br>"
                "Chg: %{customdata[0]:+.2f}%%<br>"
                "Prev: ₹%{customdata[3]:,.2f}<br>"
                "H: ₹%{customdata[4]:,.2f}  L: ₹%{customdata[5]:,.2f}"
                "<extra></extra>"
            ),
            marker=dict(
                colors=([0] + [0] * df_hm["sector"].nunique() + list(df_hm["chg_pct"])),
                colorscale=[
                    [0.0,  "#b71c1c"],   # deep red  (-3%+)
                    [0.2,  "#c62828"],
                    [0.38, "#4a1a1a"],   # dark red  (-1%)
                    [0.47, "#1a1a1a"],   # near zero
                    [0.5,  "#0d1117"],   # zero
                    [0.53, "#1a1a1a"],   # near zero
                    [0.62, "#1a4a1a"],   # dark green (+1%)
                    [0.8,  "#1b5e20"],
                    [1.0,  "#00c853"],   # deep green (+3%+)
                ],
                cmin=-color_max,
                cmax=color_max,
                colorbar=dict(
                    title="% Chg",
                    thickness=12,
                    len=0.8,
                    tickfont=dict(color="#7fa8c8", size=10, family="JetBrains Mono"),
                    titlefont=dict(color="#7fa8c8", size=10),
                ),
                pad=dict(t=1, l=1, r=1, b=1),
                line=dict(width=1.5, color="#0d1117"),
            ),
            textfont=dict(
                family="JetBrains Mono",
                size=[12] * (1 + df_hm["sector"].nunique() + len(df_hm)),
                color="#ffffff",
            ),
            tiling=dict(packing="squarify", squarifyratio=1),
        ))
    else:
        # Flat treemap (no grouping)
        fig_treemap = go.Figure(go.Treemap(
            labels=list(df_hm["symbol"]),
            parents=[""] * len(df_hm),
            values=list(df_hm["size_val"]),
            customdata=df_hm[["chg_pct","ltp","chg_rs","prev_close","day_high","day_low"]].values,
            texttemplate=(
                "<b>%{label}</b><br>"
                "%{customdata[0]:+.2f}%<br>"
                "₹%{customdata[1]:,.0f}"
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "LTP: ₹%{customdata[1]:,.2f}<br>"
                "Chg: %{customdata[0]:+.2f}%%<br>"
                "Prev: ₹%{customdata[3]:,.2f}<br>"
                "H: ₹%{customdata[4]:,.2f}  L: ₹%{customdata[5]:,.2f}"
                "<extra></extra>"
            ),
            marker=dict(
                colors=list(df_hm["chg_pct"]),
                colorscale=[
                    [0.0,  "#b71c1c"],
                    [0.2,  "#c62828"],
                    [0.38, "#4a1a1a"],
                    [0.47, "#1a1a1a"],
                    [0.5,  "#0d1117"],
                    [0.53, "#1a1a1a"],
                    [0.62, "#1a4a1a"],
                    [0.8,  "#1b5e20"],
                    [1.0,  "#00c853"],
                ],
                cmin=-color_max,
                cmax=color_max,
                colorbar=dict(
                    title="% Chg",
                    thickness=12,
                    tickfont=dict(color="#7fa8c8", size=10, family="JetBrains Mono"),
                    titlefont=dict(color="#7fa8c8", size=10),
                ),
                line=dict(width=1.5, color="#0d1117"),
            ),
            textfont=dict(family="JetBrains Mono", size=12, color="#ffffff"),
            tiling=dict(packing="squarify", squarifyratio=1),
        ))

    fig_treemap.update_layout(
        height=600,
        paper_bgcolor="#070b0f",
        plot_bgcolor="#070b0f",
        font=dict(family="JetBrains Mono", color="#7fa8c8"),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    st.plotly_chart(fig_treemap, use_container_width=True, config={"displayModeBar": False})

    # ── TOP GAINERS / LOSERS TABLE ────────────────────────────────────────
    st.markdown("---")
    tbl_c1, tbl_c2 = st.columns(2)

    def _rank_table(df_sorted, is_gainer):
        color = "#00e676" if is_gainer else "#ff3d57"
        arrow = "▲" if is_gainer else "▼"
        bg    = "rgba(0,230,118,0.04)" if is_gainer else "rgba(255,61,87,0.04)"
        border = "#1a4a1a" if is_gainer else "#4a1a1a"
        html  = ""
        for i, (_, r) in enumerate(df_sorted.head(10).iterrows()):
            bar_w = min(abs(r["chg_pct"]) / max(df_sorted["chg_pct"].abs().max(), 0.1) * 100, 100)
            html += f"""
            <div style="background:{bg};border:1px solid {border};border-left:3px solid {color};
                        border-radius:3px;padding:8px 12px;margin:4px 0;position:relative;overflow:hidden;">
              <div style="position:absolute;left:0;top:0;width:{bar_w:.0f}%;height:100%;
                          background:{'rgba(0,230,118,0.06)' if is_gainer else 'rgba(255,61,87,0.06)'};z-index:0;"></div>
              <div style="position:relative;z-index:1;display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <span style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
                               font-weight:800;color:#e8f4ff;letter-spacing:0.5px;">
                    #{i+1} {r['symbol']}</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                               color:#3a6080;margin-left:8px;">{r.get('sector','')}</span>
                </div>
                <div style="text-align:right;">
                  <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                              font-weight:700;color:{color};">{arrow} {abs(r['chg_pct']):.2f}%</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#7fa8c8;">
                    ₹{r['ltp']:,.2f} &nbsp;|&nbsp;
                    <span style="color:{color};">{'+' if r['chg_rs']>=0 else ''}₹{r['chg_rs']:.2f}</span>
                  </div>
                </div>
              </div>
            </div>"""
        return html

    with tbl_c1:
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
            font-weight:800;letter-spacing:2px;color:#00e676;margin-bottom:8px;">
            🟢 TOP GAINERS</div>""", unsafe_allow_html=True)
        top_g = df_hm[df_hm["chg_pct"] > 0].sort_values("chg_pct", ascending=False)
        if top_g.empty:
            st.markdown("<div style='color:#3a6080;'>No gainers today</div>", unsafe_allow_html=True)
        else:
            st.markdown(_rank_table(top_g, True), unsafe_allow_html=True)

    with tbl_c2:
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
            font-weight:800;letter-spacing:2px;color:#ff3d57;margin-bottom:8px;">
            🔴 TOP LOSERS</div>""", unsafe_allow_html=True)
        top_l = df_hm[df_hm["chg_pct"] < 0].sort_values("chg_pct")
        if top_l.empty:
            st.markdown("<div style='color:#3a6080;'>No losers today</div>", unsafe_allow_html=True)
        else:
            st.markdown(_rank_table(top_l, False), unsafe_allow_html=True)

    # ── SECTOR PERFORMANCE TABLE ──────────────────────────────────────────
    if hm_group == "Sector":
        st.markdown("---")
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-size:16px;
            font-weight:800;letter-spacing:2px;color:#e8f4ff;margin-bottom:8px;">
            📊 SECTOR PERFORMANCE</div>""", unsafe_allow_html=True)

        sector_summary = (
            df_hm.groupby("sector")
            .agg(
                avg_chg=("chg_pct", "mean"),
                count=("symbol", "count"),
                gainers=("chg_pct", lambda x: (x > 0).sum()),
                losers=("chg_pct",  lambda x: (x < 0).sum()),
                best_sym=("symbol",  lambda x: x.iloc[df_hm.loc[x.index, "chg_pct"].argmax()]),
                worst_sym=("symbol", lambda x: x.iloc[df_hm.loc[x.index, "chg_pct"].argmin()]),
            )
            .reset_index()
            .sort_values("avg_chg", ascending=False)
        )

        sec_html = """
        <div style="background:#0d1117;border:1px solid #1e3040;border-radius:3px;overflow:hidden;">
          <div style="display:grid;grid-template-columns:130px 80px 80px 70px 70px 110px 100%;
                      padding:7px 14px;border-bottom:1px solid #1e3040;
                      font-family:'Barlow Condensed',sans-serif;font-size:10px;
                      letter-spacing:1.5px;color:#3a6080;">
            <span>SECTOR</span><span>AVG CHG</span><span>STOCKS</span>
            <span>UP</span><span>DOWN</span><span>BEST STOCK</span><span>WORST STOCK</span>
          </div>"""

        for _, sr in sector_summary.iterrows():
            sc   = "#00e676" if sr["avg_chg"] > 0 else "#ff3d57"
            bg_s = "background:#071008;" if sr["avg_chg"] > 0.3 else "background:#120307;" if sr["avg_chg"] < -0.3 else ""
            sec_html += f"""
          <div style="display:grid;grid-template-columns:130px 80px 80px 70px 70px 110px 100%;
                      padding:8px 14px;border-bottom:1px solid #0d1117;{bg_s}
                      font-family:'JetBrains Mono',monospace;font-size:12px;align-items:center;">
            <span style="font-family:'Barlow Condensed',sans-serif;font-size:13px;
                         font-weight:700;color:#e8f4ff;">{sr['sector']}</span>
            <span style="color:{sc};font-weight:700;">
              {'▲' if sr['avg_chg']>0 else '▼'} {abs(sr['avg_chg']):.2f}%</span>
            <span style="color:#7fa8c8;">{int(sr['count'])}</span>
            <span style="color:#00e676;">{int(sr['gainers'])}</span>
            <span style="color:#ff3d57;">{int(sr['losers'])}</span>
            <span style="color:#00e676;font-size:11px;">{sr['best_sym']}</span>
            <span style="color:#ff3d57;font-size:11px;">{sr['worst_sym']}</span>
          </div>"""

        sec_html += "</div>"
        st.markdown(sec_html, unsafe_allow_html=True)

    # ── RAW DATA EXPANDER ─────────────────────────────────────────────────
    with st.expander("📋 Full Heatmap Data Table"):
        display_df = df_hm[["symbol","sector","ltp","prev_close","chg_pct",
                             "chg_rs","day_high","day_low","volume"]].copy()
        display_df.columns = ["Symbol","Sector","LTP","Prev Close",
                               "Chg%","Chg ₹","Day High","Day Low","Volume"]

        def _color_rows(row):
            chg = row["Chg%"]
            if chg > 1:    return ["background-color:#010e06;color:#00e676;"] * len(row)
            elif chg > 0:  return ["background-color:#020a04;"] * len(row)
            elif chg < -1: return ["background-color:#1a0305;color:#ff3d57;"] * len(row)
            elif chg < 0:  return ["background-color:#0d0203;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style
                .apply(_color_rows, axis=1)
                .format({
                    "LTP":        "₹{:,.2f}",
                    "Prev Close": "₹{:,.2f}",
                    "Chg%":       "{:+.2f}%",
                    "Chg ₹":      "₹{:+.2f}",
                    "Day High":   "₹{:,.2f}",
                    "Day Low":    "₹{:,.2f}",
                    "Volume":     "{:,.0f}",
                }),
            use_container_width=True,
            height=450,
        )

        csv_hm = display_df.to_csv(index=False)
        st.download_button(
            "📥 Export Heatmap CSV",
            data=csv_hm,
            file_name="market_heatmap.csv",
            mime="text/csv",
        )