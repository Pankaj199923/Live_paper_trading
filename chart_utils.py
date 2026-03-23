import pandas as pd
import numpy as np

# ======================
# 📊 TECHNICAL INDICATORS
# ======================
def compute_technicals(df):
    """Compute RSI, EMA, MACD, Bollinger Bands, VWAP, ATR, SuperTrend on 1-min OHLCV data."""
    if df.empty or len(df) < 5:
        return df, {}
    df = df.copy()

    # ── EMAs ──
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"]= df["close"].ewm(span=200, adjust=False).mean()

    # ── RSI(14) ──
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).fillna(50)

    # ── MACD(12,26,9) ──
    ema12          = df["close"].ewm(span=12, adjust=False).mean()
    ema26          = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]     = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]= df["macd"] - df["macd_sig"]

    # ── Bollinger Bands(20,2) ──
    df["bb_mid"]   = df["close"].rolling(20, min_periods=1).mean()
    bb_std         = df["close"].rolling(20, min_periods=1).std().fillna(0)
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan) * 100  # BB%

    # ── VWAP ──
    df["tp"]       = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_tpvol"]= (df["tp"] * df["volume"]).cumsum()
    df["cum_vol"]  = df["volume"].cumsum()
    df["vwap"]     = (df["cum_tpvol"] / df["cum_vol"].replace(0, np.nan)).ffill()
    # VWAP standard deviation bands
    df["vwap_var"] = ((df["tp"] - df["vwap"]) ** 2 * df["volume"]).cumsum() / df["cum_vol"].replace(0, np.nan)
    df["vwap_std"] = np.sqrt(df["vwap_var"].clip(lower=0))
    df["vwap_upper"] = df["vwap"] + 2 * df["vwap_std"]
    df["vwap_lower"] = df["vwap"] - 2 * df["vwap_std"]

    # ── ATR(14) — True Range ──
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["prev_close"]).abs(),
            (df["low"]  - df["prev_close"]).abs()
        )
    )
    df["atr14"] = df["tr"].ewm(span=14, adjust=False).mean()

    # ── SuperTrend(10, 3) ──
    atr_mult = 3.0
    hl_avg   = (df["high"] + df["low"]) / 2
    df["st_upper"] = hl_avg + atr_mult * df["atr14"]
    df["st_lower"] = hl_avg - atr_mult * df["atr14"]
    supertrend = pd.Series(np.nan, index=df.index)
    direction  = pd.Series(1,      index=df.index)
    for i in range(1, len(df)):
        prev_close = df["close"].iloc[i - 1]
        curr_close = df["close"].iloc[i]
        # Upper band
        if df["st_upper"].iloc[i] < df["st_upper"].iloc[i-1] or prev_close > df["st_upper"].iloc[i-1]:
            pass  # keep as-is
        else:
            df.at[df.index[i], "st_upper"] = df["st_upper"].iloc[i-1]
        # Lower band
        if df["st_lower"].iloc[i] > df["st_lower"].iloc[i-1] or prev_close < df["st_lower"].iloc[i-1]:
            pass
        else:
            df.at[df.index[i], "st_lower"] = df["st_lower"].iloc[i-1]
        # Direction
        if pd.isna(supertrend.iloc[i-1]):
            direction.iloc[i] = 1
        elif supertrend.iloc[i-1] == df["st_upper"].iloc[i-1]:
            direction.iloc[i] = -1 if curr_close > df["st_upper"].iloc[i] else 1
        else:
            direction.iloc[i] = 1 if curr_close < df["st_lower"].iloc[i] else -1
        supertrend.iloc[i] = df["st_lower"].iloc[i] if direction.iloc[i] == -1 else df["st_upper"].iloc[i]
    df["supertrend"] = supertrend
    df["st_dir"]     = direction   # -1 = bullish (price above), 1 = bearish

    last = df.iloc[-1]
    atr_val = round(float(last.get("atr14", 0)), 2)
    # ATR-based SL distances (1.5x ATR for tight, 2.5x for wide)
    atr_sl_tight = round(atr_val * 1.5, 1)
    atr_sl_wide  = round(atr_val * 2.5, 1)

    # BB squeeze: width < 1% of price = compressed volatility
    bb_w = float(last.get("bb_width", 5))
    bb_condition = "SQUEEZE" if bb_w < 1.0 else "EXPANDING" if bb_w > 3.0 else "NORMAL"

    # SuperTrend bias
    st_bias = "BULLISH" if float(last.get("st_dir", 1)) == -1 else "BEARISH"

    summary = {
        "current_price": round(float(last["close"]), 2),
        "open_price":    round(float(df.iloc[0]["open"]), 2),
        "high_of_day":   round(float(df["high"].max()), 2),
        "low_of_day":    round(float(df["low"].min()), 2),
        "volume":        int(df["volume"].sum()),
        "rsi14":         round(float(last["rsi14"]), 1),
        "ema9":          round(float(last["ema9"]),  2),
        "ema21":         round(float(last["ema21"]), 2),
        "ema50":         round(float(last["ema50"]), 2),
        "ema200":        round(float(last["ema200"]),2),
        "macd":          round(float(last["macd"]),  4),
        "macd_signal":   round(float(last["macd_sig"]), 4),
        "macd_hist":     round(float(last["macd_hist"]), 4),
        "bb_upper":      round(float(last["bb_upper"]), 2),
        "bb_lower":      round(float(last["bb_lower"]), 2),
        "bb_mid":        round(float(last["bb_mid"]),   2),
        "bb_condition":  bb_condition,
        "vwap":          round(float(last["vwap"]),     2),
        "vwap_upper":    round(float(last.get("vwap_upper", last["vwap"])), 2),
        "vwap_lower":    round(float(last.get("vwap_lower", last["vwap"])), 2),
        "atr14":         atr_val,
        "atr_sl_tight":  atr_sl_tight,
        "atr_sl_wide":   atr_sl_wide,
        "supertrend_bias": st_bias,
        "price_vs_vwap": "ABOVE" if last["close"] > last["vwap"] else "BELOW",
        "ema_trend":     ("BULLISH" if last["ema9"] > last["ema21"] > last["ema50"] else
                          "BEARISH" if last["ema9"] < last["ema21"] < last["ema50"] else "MIXED"),
        "rsi_zone":      ("OVERBOUGHT" if last["rsi14"] > 70 else
                          "OVERSOLD"   if last["rsi14"] < 30 else "NEUTRAL"),
        "macd_cross":    "BULLISH" if last["macd_hist"] > 0 else "BEARISH",
        "candles_count": len(df),
        "last_candle_time": str(last["timestamp"]),
    }
    return df, summary

# ======================

# ======================
# 📦 ORDER FLOW
# ======================
# ======================
def compute_order_flow(df):
    """Approximate order flow from OHLCV: delta, cumulative delta, absorption, imbalance."""
    if df.empty or len(df) < 3:
        return df
    df = df.copy()
    df["range"]      = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"]       = abs(df["close"] - df["open"])
    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]
    # Close position in range (0=low,1=high) → proxy for buy/sell split
    df["close_pos"]  = ((df["close"] - df["low"]) / df["range"]).fillna(0.5).clip(0,1)
    df["buy_vol"]    = (df["volume"] * df["close_pos"]).round()
    df["sell_vol"]   = (df["volume"] * (1 - df["close_pos"])).round()
    df["delta"]      = df["buy_vol"] - df["sell_vol"]
    df["cum_delta"]  = df["delta"].cumsum()
    df["body_pct"]   = (df["body"] / df["range"].replace(0, np.nan) * 100).fillna(0)
    vol75            = df["volume"].quantile(0.75)
    vol90            = df["volume"].quantile(0.90)
    df["absorption"] = (df["volume"] >= vol75) & (df["body_pct"] < 25)
    df["is_hvn"]     = df["volume"] >= vol90
    # Bearish/Bullish delta divergence: price up but delta negative (hidden selling)
    df["bull_div"]   = (df["close"] > df["open"]) & (df["delta"] < 0)
    df["bear_div"]   = (df["close"] < df["open"]) & (df["delta"] > 0)
    return df


# ======================
# 💧 LIQUIDITY SWEEP DETECTOR
# ======================

# ======================
# 💧 LIQUIDITY SWEEPS
# ======================
def detect_liquidity_sweeps(df):
    """
    Detect BSL/SSL liquidity sweeps across multiple lookback windows (5, 10, 20 candles).
    BSL = Buy-Side Liquidity sweep: wick above a prior swing high, closes back below.
    SSL = Sell-Side Liquidity sweep: wick below a prior swing low, closes back above.
    Also detects Equal Highs/Lows (unmitigated liquidity pools) with relaxed tolerance.
    """
    sweeps_bsl, sweeps_ssl, eq_highs, eq_lows = [], [], [], []
    if df.empty or len(df) < 8:
        return sweeps_bsl, sweeps_ssl, eq_highs, eq_lows

    highs  = df["high"].astype(float).values
    lows   = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    opens  = df["open"].astype(float).values
    times  = df["timestamp"].values
    n      = len(df)

    seen_bsl_idx, seen_ssl_idx = set(), set()

    # ── Multi-window sweep scan ──────────────────────────────────────────
    for lb in [5, 10, 20]:
        for i in range(lb, n):
            win_h = highs[i - lb : i]
            win_l = lows[i - lb : i]
            sh    = float(win_h.max())   # swing high of window
            sl    = float(win_l.min())   # swing low of window
            h_i   = float(highs[i])
            l_i   = float(lows[i])
            c_i   = float(closes[i])
            o_i   = float(opens[i])

            # ── BSL sweep conditions (relaxed) ──────────────────────────
            # 1. Wick pierces above swing high (even by 1 tick = 0.5 pts on Nifty)
            # 2. Candle CLOSES below the swing high (rejection = stop hunt)
            # 3. Upper wick > 30% of candle range (showing real rejection)
            upper_wick = h_i - max(c_i, o_i)
            candle_range = h_i - l_i if h_i > l_i else 1
            if (h_i > sh and               # pierced above swing high
                c_i < sh and               # closed back below
                upper_wick / candle_range > 0.30 and  # meaningful wick
                i not in seen_bsl_idx):
                seen_bsl_idx.add(i)
                strength = round((sh - c_i) / sh * 100, 3)  # % reversal
                sweeps_bsl.append({
                    "time":     times[i],
                    "wick":     round(h_i, 2),
                    "swept":    round(sh, 2),
                    "close":    round(c_i, 2),
                    "reversal": round(sh - c_i, 1),
                    "strength": strength,
                    "lookback": lb,
                    "type":     "BSL",
                })

            # ── SSL sweep conditions (relaxed) ──────────────────────────
            lower_wick = min(c_i, o_i) - l_i
            if (l_i < sl and               # pierced below swing low
                c_i > sl and               # closed back above
                lower_wick / candle_range > 0.30 and  # meaningful wick
                i not in seen_ssl_idx):
                seen_ssl_idx.add(i)
                strength = round((c_i - sl) / sl * 100, 3)
                sweeps_ssl.append({
                    "time":     times[i],
                    "wick":     round(l_i, 2),
                    "swept":    round(sl, 2),
                    "close":    round(c_i, 2),
                    "reversal": round(c_i - sl, 1),
                    "strength": strength,
                    "lookback": lb,
                    "type":     "SSL",
                })

    # ── Equal Highs / Lows (liquidity pools not yet swept) ──────────────
    # Tolerance: 0.15% of price (wider than before — catches near-equal levels)
    for i in range(3, n):
        win_h = highs[max(0, i-20):i]
        win_l = lows[max(0, i-20):i]
        sh    = float(win_h.max())
        sl    = float(win_l.min())
        tol_h = sh * 0.0015    # 0.15%
        tol_l = sl * 0.0015

        # Count how many highs are within tolerance of the swing high
        eq_h_count = int(sum(1 for h in win_h if abs(float(h) - sh) <= tol_h))
        if eq_h_count >= 2:
            eq_highs.append({"level": round(sh, 1), "time": times[i],
                             "touches": eq_h_count})
        eq_l_count = int(sum(1 for l in win_l if abs(float(l) - sl) <= tol_l))
        if eq_l_count >= 2:
            eq_lows.append({"level": round(sl, 1),  "time": times[i],
                            "touches": eq_l_count})

    # Deduplicate by rounding to nearest 10 pts (Nifty step)
    seen_h, seen_l = set(), set()
    eq_highs_u, eq_lows_u = [], []
    for x in sorted(eq_highs, key=lambda d: d["touches"], reverse=True):
        k = round(x["level"] / 10) * 10
        if k not in seen_h:
            seen_h.add(k); eq_highs_u.append(x)
    for x in sorted(eq_lows, key=lambda d: d["touches"], reverse=True):
        k = round(x["level"] / 10) * 10
        if k not in seen_l:
            seen_l.add(k); eq_lows_u.append(x)

    # Sort by time desc, deduplicate by proximity
    sweeps_bsl.sort(key=lambda x: str(x["time"]))
    sweeps_ssl.sort(key=lambda x: str(x["time"]))

    return (sweeps_bsl[-10:], sweeps_ssl[-10:],
            eq_highs_u[:5], eq_lows_u[:5])


# ======================
# 🧱 ORDER BLOCKS & FVG (ICT / SMC)

# ======================
# 🧱 ORDER BLOCKS
# ======================
def detect_order_blocks(df):
    """Bullish OB = last bearish candle before bullish impulse. Bearish OB = last bullish before drop."""
    bull_obs, bear_obs = [], []
    if df.empty or len(df) < 5:
        return bull_obs, bear_obs
    for i in range(1, len(df) - 2):
        c = df.iloc[i]; n1 = df.iloc[i+1]; n2 = df.iloc[i+2]
        # Bullish OB: bearish candle followed by 2 bullish candles breaking above
        if (float(c["close"]) < float(c["open"]) and
            float(n1["close"]) > float(n1["open"]) and
            float(n2["high"]) > float(c["high"])):
            bull_obs.append({
                "time": c["timestamp"],
                "top":  float(c["open"]),
                "bot":  float(c["close"]),
                "type": "BULL_OB"
            })
        # Bearish OB: bullish candle followed by 2 bearish candles breaking below
        if (float(c["close"]) > float(c["open"]) and
            float(n1["close"]) < float(n1["open"]) and
            float(n2["low"])  < float(c["low"])):
            bear_obs.append({
                "time": c["timestamp"],
                "top":  float(c["close"]),
                "bot":  float(c["open"]),
                "type": "BEAR_OB"
            })
    return bull_obs[-5:], bear_obs[-5:]



# ======================
# ⚡ FAIR VALUE GAPS (FVG)
# ======================
def detect_fvg(df):
    """Fair Value Gaps: candle[i].low > candle[i-2].high (bull) or vice versa (bear)."""
    bull_fvg, bear_fvg = [], []
    if df.empty or len(df) < 3:
        return bull_fvg, bear_fvg
    for i in range(2, len(df)):
        c0 = df.iloc[i-2]; c2 = df.iloc[i]; mid_t = df.iloc[i-1]["timestamp"]
        if float(c2["low"]) > float(c0["high"]):
            bull_fvg.append({"time": mid_t, "top": float(c2["low"]),
                             "bot": float(c0["high"]), "filled": False})
        if float(c2["high"]) < float(c0["low"]):
            bear_fvg.append({"time": mid_t, "top": float(c0["low"]),
                              "bot": float(c2["high"]), "filled": False})
    return bull_fvg[-6:], bear_fvg[-6:]



# ======================
# 🔀 BOS / CHOCH
# ======================
def detect_bos_choch(df, lookback=12):
    """Break of Structure and Change of Character detection."""
    bos, choch = [], []
    if df.empty or len(df) < lookback + 2:
        return bos, choch
    trend = None
    for i in range(lookback, len(df)):
        win  = df.iloc[i - lookback : i]
        curr = df.iloc[i]; prev = df.iloc[i - 1]
        sh   = float(win["high"].max())
        sl   = float(win["low"].min())
        # Breaks above swing high
        if float(curr["close"]) > sh and float(prev["close"]) <= sh:
            ev = {"time": curr["timestamp"], "price": sh,
                  "dir": "UP", "label": "BOS ▲" if trend == "UP" else "CHoCH ▲"}
            (bos if trend == "UP" else choch).append(ev)
            trend = "UP"
        # Breaks below swing low
        if float(curr["close"]) < sl and float(prev["close"]) >= sl:
            ev = {"time": curr["timestamp"], "price": sl,
                  "dir": "DOWN", "label": "BOS ▼" if trend == "DOWN" else "CHoCH ▼"}
            (bos if trend == "DOWN" else choch).append(ev)
            trend = "DOWN"
    return bos[-6:], choch[-6:]



# ======================
# 📋 ORDER FLOW SUMMARY
# ======================
def get_order_flow_summary(of_df):
    """Generate a plain-language order flow summary from the enriched dataframe."""
    if of_df.empty or len(of_df) < 5:
        return {}
    last5      = of_df.tail(5)
    last       = of_df.iloc[-1]
    total_buy  = float(of_df["buy_vol"].sum())
    total_sell = float(of_df["sell_vol"].sum())
    net_delta  = float(of_df["delta"].sum())
    cum_d_last = float(last["cum_delta"])
    cum_d_prev = float(of_df.iloc[-6]["cum_delta"]) if len(of_df) >= 6 else 0
    abs_count  = int(of_df["absorption"].sum())
    bull_div_c = int(of_df["bull_div"].sum())
    bear_div_c = int(of_df["bear_div"].sum())
    hvn_prices = of_df[of_df["is_hvn"]]["close"].tolist()[-3:]
    delta_trend = "RISING" if cum_d_last > cum_d_prev else "FALLING"
    pressure    = "BUY DOMINANT" if net_delta > 0 else "SELL DOMINANT"
    return {
        "total_buy_vol":  round(total_buy),
        "total_sell_vol": round(total_sell),
        "net_delta":      round(net_delta),
        "cum_delta":      round(cum_d_last),
        "delta_trend":    delta_trend,
        "pressure":       pressure,
        "buy_pct":        round(total_buy / (total_buy + total_sell) * 100, 1) if (total_buy+total_sell) > 0 else 50,
        "absorption_candles": abs_count,
        "bull_divergence": bull_div_c,
        "bear_divergence": bear_div_c,
        "hvn_prices":     [round(p,0) for p in hvn_prices],
        "last5_delta":    [round(float(d)) for d in last5["delta"].tolist()],
    }


# ======================
# 🧠 CLAUDE AI TRADE SETUP (Full Data Fusion)
# ======================
