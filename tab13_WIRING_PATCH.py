"""
STEP 1 — Add this import at top of tab9_chart.py
═══════════════════════════════════════════════════
"""
from tab_fvg_tradelog import append_fvg_trade


"""
STEP 2 — Replace the _render_fvg_auto_trade(...) call at the bottom of
          tab9_chart.py render() with this version that wires real logging.

Find the block:
    _render_fvg_auto_trade(
        bull_fvgs      = bull_fvg9,
        ...
        place_order_fn = None,
    )

Replace entirely with the block below:
═══════════════════════════════════════════════════
"""

    # Build the place_order callback with logging + broker call
    _cur_expiry = st.session_state.get("current_selected_expiry", "")

    def _fvg_place_order(action: str, strike: int, index_key: str) -> bool:
        """
        1. Fetch live LTP of the option being sold (for entry price record)
        2. Log the trade to fvg_trade_log.csv
        3. Call your broker API  ← fill in your real API call here
        """
        import traceback
        opt_type = "PE" if action == "SELL_PE" else "CE"

        # ── Get entry LTP ─────────────────────────────────────────────
        entry_px = 0.0
        try:
            symbol   = f"{index_key}{_cur_expiry}{strike}{opt_type}"
            entry_px = float(fetch_ltp(symbol) or 0)
        except Exception:
            pass

        # ── Log to CSV ────────────────────────────────────────────────
        # We grab the last triggered signal from session state for FVG meta
        _log_kwargs = dict(
            index_key    = index_key,
            expiry       = _cur_expiry,
            action       = action,
            strike       = strike,
            entry_price  = entry_px,
            fvg_type     = "BEAR" if action == "SELL_PE" else "BULL",
            fvg_bot      = 0.0,    # overridden below if signal context available
            fvg_top      = 0.0,
            fvg_size     = 0.0,
            trigger_price= spot_t9,
            reason       = f"FVG auto: {action} {strike} @ {entry_px}",
            auto         = st.session_state.get("fvg_auto_enabled", False),
        )

        # Try to enrich with last signal context from session state
        _last_sig = st.session_state.get("fvg_last_signal", {})
        if _last_sig:
            _log_kwargs.update({
                "fvg_type"    : _last_sig.get("fvg_type",  _log_kwargs["fvg_type"]),
                "fvg_bot"     : _last_sig.get("bot",       0.0),
                "fvg_top"     : _last_sig.get("top",       0.0),
                "fvg_size"    : _last_sig.get("size",      0.0),
                "trigger_price": _last_sig.get("trigger",  spot_t9),
                "reason"      : _last_sig.get("reason",    _log_kwargs["reason"]),
            })

        append_fvg_trade(**_log_kwargs)

        # ── Broker API call ───────────────────────────────────────────
        # Uncomment + adapt when ready:
        #
        # try:
        #     resp = your_broker.place_order(
        #         tradingsymbol = symbol,
        #         quantity      = get_lot_size(index_key),
        #         transaction_type = "SELL",
        #         order_type    = "MARKET",
        #     )
        #     return resp.get("status") == "complete"
        # except Exception as e:
        #     st.error(f"Broker error: {e}")
        #     return False

        return True   # ← remove this line once broker API is wired

    _render_fvg_auto_trade(
        bull_fvgs      = bull_fvg9,
        bear_fvgs      = bear_fvg9,
        of_df9         = of_df9,
        spot           = spot_t9,
        index_key      = sel_t9,
        expiry         = _cur_expiry,
        place_order_fn = _fvg_place_order,   # ← NOW WIRED
    )


"""
STEP 3 — Store last signal in session_state inside _scan_fvg_triggers()
          so _fvg_place_order can pick it up.

Inside _render_fvg_auto_trade(), after:
    for sig in new_signals:
        trig_ids.add(sig["zone_id"])

Add this line:
"""
        st.session_state["fvg_last_signal"] = sig   # ← add this


"""
STEP 4 — Register the new tab in your main app.py / app_main.py

Find your tab definition block, e.g.:
    tab1, tab2, ... tab9 = st.tabs([...])

Add the FVG log tab:
"""
# Example — adapt to match your actual tab list:
tab_names = [
    "📈 OPTION CHAIN",
    "🎯 STRATEGY",
    "💼 POSITIONS",
    "📊 GREEKS",
    "🧮 ANALYTICS",
    "🤖 AI TRADE",
    "📷 SNAPSHOT",
    "⚙️ SETTINGS",
    "📊 CHART",
    "📋 FVG LOG",     # ← NEW TAB
]

# In the tab rendering section:
with tab10:   # or whatever index
    from tab_fvg_tradelog import render as render_fvg_log
    render_fvg_log()