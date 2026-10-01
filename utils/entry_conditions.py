"""
utils/entry_conditions.py
─────────────────────────
Scout Trading Agent — V2 Finite State Machine Entry Engine

Called by the agent every session before any trade decision.
Evaluates:
  1. Market calendar, cutoffs, capital safety (2 Lots = 950 Qty)
  2. Model conviction (weekday >= 65%, Friday >= 70%)
  3. Pre-Trade Gates: Spread (<= 1% or <= Rs 0.40) & Anti-Chase (< +Rs 6.50)
  4. Opening Phase: VWAP +/- 1.5-sigma bands & 1-minute Rejection Candle
  5. 2-Tranche Entry: T1 (Spike Rejection) + T2 (VWAP Candle Close)
  6. Hybrid SL Regime: 09:15-09:25 Emergency SL (-Rs 8.00) vs Post-09:25 Precision SL (-Rs 5.00)
  7. Decoupled Runner: Lot 1 Target (+Rs 10.00) & Lot 2 Runner (5-Min 20 EMA)
  8. FSM State: STATE 0 through STATE 4
"""

import datetime as dt
import logging
import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

from utils.market_calendar import is_trading_holiday
from utils.paper_trading import load_paper_trades, calculate_bsm_option_price

logging.basicConfig(level=logging.WARNING)

# ─── Constants ────────────────────────────────────────────────────────────────
AUDIT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prediction_audit.json")

GAP_THRESHOLD_RS        = 12.0   # Rs 12 gap -> delayed entry to 09:35
CONVICTION_WEEKDAY      = 65.0   # % minimum to trade Monday-Thursday
CONVICTION_FRIDAY       = 70.0   # % minimum to trade on Friday
CAPITAL_MINIMUM         = 20000  # below this -> stop trading
CAPITAL_ONE_LOT_MAX     = 30000  # below this -> 1 lot only
CONSECUTIVE_LOSS_WARN   = 2      # warn at N consecutive SL hits
CONSECUTIVE_LOSS_ALERT  = 3      # alert at N consecutive SL hits
CONSECUTIVE_LOSS_STOP   = 4      # hard-stop at N consecutive SL hits
FRIDAY_PUT_CLOSE_HOUR   = 13     # 1 PM - close PUT if CDSL up > Rs 8 on Friday
FRIDAY_PUT_UP_THRESHOLD = 8.0    # Rs 8 move UP -> theta danger on Friday PUTs
NO_ENTRY_AFTER_HOUR     = 14     # 2:30 PM -> no new entries
NO_ENTRY_AFTER_MIN      = 30

from config import CDSL_LOT_SIZE
LOT_SIZE                = CDSL_LOT_SIZE  # CDSL NSE lot size (475 shares per lot)
TOTAL_LOTS              = 2              # 2 Lots (950 Qty)
TOTAL_QTY               = LOT_SIZE * TOTAL_LOTS

# ─── V2 Specific Parameters ───────────────────────────────────────────────────
VWAP_SIGMA_MULT         = 1.5    # VWAP +/- 1.5-sigma band trigger
EMERGENCY_SL_PTS        = 8.0    # 09:15-09:25 Emergency stop (-Rs 8.00 buffer)
PRECISION_SL_PTS        = 5.0    # Post-09:25 Precision stop (-Rs 5.00 hard stop)
LOT1_TARGET_PTS         = 10.0   # Lot 1 Profit Target (+Rs 10.00 lock)
ANTI_CHASE_MAX_PTS      = 6.50   # Max allowed option premium gain before entry abort
SPREAD_MAX_PCT          = 1.0    # Max bid-ask spread % of midpoint
SPREAD_MAX_RS           = 0.40   # Max absolute bid-ask spread in Rs


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_audit() -> list:
    if os.path.exists(AUDIT_FILE):
        try:
            with open(AUDIT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            pass
    return []


def _get_today_prediction(symbol: str) -> dict | None:
    """Return today's audit record for the given symbol."""
    today_str = dt.date.today().strftime("%a, %d %b %Y")
    audit = _load_audit()
    for rec in reversed(audit):
        if rec.get("symbol") == symbol and rec.get("target_date") == today_str:
            return rec
    return None


def _get_consecutive_losses(symbol: str) -> int:
    """Count trailing consecutive SL hits for the symbol from paper_trading_ledger."""
    history = load_paper_trades()
    trades = [
        t for t in history
        if t.get("symbol") == symbol
        and t.get("strategy") == "AI Intraday +₹4 Scalp"
        and t.get("status") in ("✅ WIN", "❌ LOSS")
    ]
    streak = 0
    for trade in reversed(trades):
        if trade["status"] == "❌ LOSS":
            streak += 1
        else:
            break
    return streak


def _get_live_capital() -> tuple[float, str]:
    """Retrieve actual live unencumbered cash directly from Groww API; fallback to ledger."""
    try:
        from utils.options_feed import get_groww_client
        client = get_groww_client()
        if client:
            margin = client.get_available_margin_details()
            cash = float(margin.get("clear_cash") or 0.0)
            if cash > 0:
                return round(cash, 2), "Groww Live Broker Balance"
    except Exception:
        pass

    history = load_paper_trades()
    baseline = 35000.0
    total_pnl = sum(
        t.get("net_pnl") or 0.0
        for t in history
        if t.get("symbol") == "CDSL.NS"
        and t.get("strategy") == "AI Intraday +₹4 Scalp"
        and t.get("net_pnl") is not None
    )
    return round(baseline + total_pnl, 2), "Ledger Estimate"


def calculate_vwap_and_bands(df: pd.DataFrame, sigma_mult: float = 1.5):
    """Compute session VWAP and +/- 1.5 sigma bands."""
    if df.empty or df['Volume'].sum() == 0:
        last_close = float(df['Close'].iloc[-1]) if not df.empty else 0.0
        return last_close, last_close, last_close, 0.0
    
    typical_p = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume'].astype(float)
    cum_vp = (typical_p * vol).cumsum()
    cum_vol = vol.cumsum()
    vwap = cum_vp / cum_vol.replace(0, np.nan)
    
    sq_diff = ((typical_p - vwap) ** 2 * vol).cumsum()
    variance = sq_diff / cum_vol.replace(0, np.nan)
    std = np.sqrt(variance.fillna(0.0))
    
    cur_vwap = round(float(vwap.iloc[-1]), 2)
    cur_std = round(float(std.iloc[-1]), 2)
    upper_band = round(cur_vwap + sigma_mult * cur_std, 2)
    lower_band = round(cur_vwap - sigma_mult * cur_std, 2)
    
    return cur_vwap, upper_band, lower_band, cur_std


def is_rejection_candle(bar, is_call: bool = False) -> bool:
    """Check if 1-min or 5-min bar is a valid mechanical rejection candle."""
    o, h, l, c = float(bar['Open']), float(bar['High']), float(bar['Low']), float(bar['Close'])
    body = abs(c - o)
    candle_range = h - l
    if candle_range < 0.10 or body < 0.05:
        return False
    
    if not is_call:  # PE Rejection (Bearish Rejection at Top)
        if c >= o:
            return False
        upper_wick = h - o
        close_pos = (c - l) / candle_range
        return upper_wick >= 1.5 * body and close_pos <= 0.40
    else:  # CE Rejection (Bullish Rejection at Bottom)
        if c <= o:
            return False
        lower_wick = o - l
        close_pos = (c - l) / candle_range
        return lower_wick >= 1.5 * body and close_pos >= 0.60


def check_pre_trade_gates(quote_data: dict, open_prem: float) -> tuple[bool, dict]:
    """Validate Spread, Anti-Chase, and Liquidity before order execution."""
    ltp = quote_data.get("ltp", 0.0)
    bid = quote_data.get("bid", ltp - 0.10)
    ask = quote_data.get("ask", ltp + 0.10)
    spread = round(ask - bid, 2) if ask >= bid else 0.10
    midpoint = (ask + bid) / 2.0 if (ask + bid) > 0 else ltp
    spread_pct = round((spread / midpoint) * 100.0, 2) if midpoint > 0 else 0.0
    
    # 1. Spread Check: Spread <= 1.0% OR Spread <= Rs 0.40
    spread_pass = (spread_pct <= SPREAD_MAX_PCT) or (spread <= SPREAD_MAX_RS)
    
    # 2. Anti-Chase Check: LTP - Open <= Rs 6.50
    chase_gain = round(ltp - open_prem, 2) if open_prem > 0 else 0.0
    chase_pass = chase_gain <= ANTI_CHASE_MAX_PTS
    
    # 3. Liquidity Depth
    liquidity_pass = ltp > 0 and (ask - bid) < 2.0
    
    all_passed = spread_pass and chase_pass and liquidity_pass
    details = {
        "passed": all_passed,
        "spread_rs": spread,
        "spread_pct": spread_pct,
        "spread_pass": spread_pass,
        "chase_gain": chase_gain,
        "chase_pass": chase_pass,
        "liquidity_pass": liquidity_pass
    }
    return all_passed, details


# ─── Main Function ────────────────────────────────────────────────────────────

def check_entry_conditions(symbol: str = "CDSL.NS", capital_override: float | None = None,
                           simulate_morning: bool = False) -> dict:
    """
    Master pre-trade checklist & FSM state analyzer. Runs all conditions in order:
    """
    today        = dt.date.today()
    now          = dt.datetime.now()
    is_friday    = (today.weekday() == 4)
    reasons      = []
    warnings     = []
    friday_rules = []

    result = {
        "verdict":            "UNKNOWN",
        "active_state":       "STATE 0: SCANNING",
        "entry_time":         None,
        "lots":               TOTAL_LOTS,
        "action":             None,
        "direction":          None,
        "conviction":         0.0,
        "gap_rs":             0.0,
        "gap_type":           "UNKNOWN",
        "current_price":      0.0,
        "open_price":         0.0,
        "prev_close":         0.0,
        "is_friday":          is_friday,
        "consecutive_losses": 0,
        "capital":            0.0,
        "reasons":            reasons,
        "warnings":           warnings,
        "friday_rules":       friday_rules,
        "summary":            "",
    }

    # ── CHECK 0: Market holiday? ───────────────────────────────────────────────
    is_holiday, holiday_name = is_trading_holiday(today)
    if is_holiday:
        reasons.append(f"🚫 Market CLOSED today — {holiday_name}")
        result.update({"verdict": "STOP", "active_state": "STATE 4: EXITED", "summary": _build_summary(result, reasons, warnings, friday_rules)})
        return result

    if today.weekday() >= 5:
        reasons.append("🚫 Weekend — market closed.")
        result.update({"verdict": "STOP", "active_state": "STATE 4: EXITED", "summary": _build_summary(result, reasons, warnings, friday_rules)})
        return result

    # ── CHECK 1: Too late to enter? ────────────────────────────────────────────
    past_cutoff = (now.hour > NO_ENTRY_AFTER_HOUR or
                   (now.hour == NO_ENTRY_AFTER_HOUR and now.minute >= NO_ENTRY_AFTER_MIN))
    if past_cutoff and not simulate_morning:
        reasons.append(f"⏰ Time is {now.strftime('%I:%M %p')} — past 02:30 PM cutoff. No new entries.")
        result.update({"verdict": "STOP", "active_state": "STATE 4: EXITED", "summary": _build_summary(result, reasons, warnings, friday_rules)})
        return result

    # ── CHECK 2: Capital safety ────────────────────────────────────────────────
    if capital_override is not None:
        capital, cap_source = capital_override, "Manual Override"
    else:
        capital, cap_source = _get_live_capital()
    result["capital"] = capital
    result["capital_source"] = cap_source

    if capital < CAPITAL_MINIMUM:
        reasons.append(f"🚫 CAPITAL DANGER: ₹{capital:,.0f} < ₹{CAPITAL_MINIMUM:,}. STOP TRADING immediately.")
        result.update({"verdict": "STOP", "active_state": "STATE 4: EXITED", "summary": _build_summary(result, reasons, warnings, friday_rules)})
        return result

    # Enforce 2 Lots as primary trading unit
    lots = TOTAL_LOTS
    result["lots"] = lots

    # ── CHECK 3: Consecutive losses ────────────────────────────────────────────
    consec = _get_consecutive_losses(symbol)
    result["consecutive_losses"] = consec

    if consec >= CONSECUTIVE_LOSS_STOP:
        reasons.append(f"🚫 {consec} consecutive SL hits. Hard stop — DO NOT trade today.")
        result.update({"verdict": "STOP", "active_state": "STATE 4: EXITED", "summary": _build_summary(result, reasons, warnings, friday_rules)})
        return result
    elif consec >= CONSECUTIVE_LOSS_ALERT:
        warnings.append(f"🔴 {consec} consecutive SL hits — risk alert. Consider skipping today.")
    elif consec >= CONSECUTIVE_LOSS_WARN:
        warnings.append(f"⚠️ {consec} consecutive SL hits — review gap-day rule before entering.")

    # ── CHECK 4: Pull live market data ────────────────────────────────────────
    try:
        ticker  = yf.Ticker(symbol)
        df_5m   = ticker.history(period="1d", interval="5m")
        df_day  = ticker.history(period="5d", interval="1d")

        if df_5m.empty or df_day.empty:
            reasons.append("❌ Could not pull live data from Yahoo Finance.")
            result.update({"verdict": "NO-TRADE", "active_state": "STATE 0: SCANNING", "summary": _build_summary(result, reasons, warnings, friday_rules)})
            return result

        if df_5m.index.tz is not None:
            df_5m.index = df_5m.index.tz_convert("Asia/Kolkata")
        if df_day.index.tz is not None:
            df_day.index = df_day.index.tz_convert("Asia/Kolkata")

        open_price     = round(float(df_5m["Open"].iloc[0]), 2)
        current_price  = round(float(df_5m["Close"].iloc[-1]), 2)
        prev_close     = round(float(df_day["Close"].iloc[-2]), 2) if len(df_day) >= 2 else open_price
        gap_rs         = round(open_price - prev_close, 2)
        gap_pct        = round((gap_rs / prev_close) * 100, 2)
        day_high       = round(float(df_5m["High"].max()), 2)
        day_low        = round(float(df_5m["Low"].min()), 2)
        open_to_now    = round(current_price - open_price, 2)

        # ── V2 VWAP & +/- 1.5 Sigma Bands ─────────────────────────────────────
        vwap, upper_band, lower_band, std_dev = calculate_vwap_and_bands(df_5m, sigma_mult=VWAP_SIGMA_MULT)

        # 20 EMA on 5-min chart for runner trail
        ema_20 = round(float(df_5m['Close'].ewm(span=20, adjust=False).mean().iloc[-1]), 2)

        result.update({
            "open_price":    open_price,
            "current_price": current_price,
            "prev_close":    prev_close,
            "gap_rs":        gap_rs,
            "day_high":      day_high,
            "day_low":       day_low,
            "vwap":          vwap,
            "upper_band":    upper_band,
            "lower_band":    lower_band,
            "std_dev":       std_dev,
            "ema_20":        ema_20,
        })

    except Exception as e:
        reasons.append(f"❌ Live data error: {e}")
        result.update({"verdict": "NO-TRADE", "active_state": "STATE 0: SCANNING", "summary": _build_summary(result, reasons, warnings, friday_rules)})
        return result

    # ── CHECK 5: Gap-Day rule ─────────────────────────────────────────────────
    gap_type = "GAP_DAY" if abs(gap_rs) > GAP_THRESHOLD_RS else "NORMAL"
    result["gap_type"] = gap_type

    if gap_type == "GAP_DAY":
        entry_time = "09:35 AM"
        warnings.append(
            f"⚠️ Gap day detected — gap = ₹{gap_rs:+.2f} ({gap_pct:+.2f}%). "
            f"Morning momentum settlement active. Tranche 2 entry delayed to {entry_time}."
        )
    else:
        entry_time = "09:20 AM"
    result["entry_time"] = entry_time

    # ── CHECK 6: Model conviction ──────────────────────────────────────────────
    pred = _get_today_prediction(symbol)
    conviction_threshold = CONVICTION_FRIDAY if is_friday else CONVICTION_WEEKDAY

    if pred is None:
        reasons.append("❌ No model prediction found for today. Run the morning prediction script first.")
        result.update({"verdict": "NO-TRADE", "active_state": "STATE 0: SCANNING", "summary": _build_summary(result, reasons, warnings, friday_rules)})
        return result

    pred_result   = pred.get("pred_result", {}) or {}
    prob_up       = float(pred_result.get("probability_up_pct",   pred.get("prob_up",   50.0)))
    prob_down     = float(pred_result.get("probability_down_pct", pred.get("prob_down", 50.0)))
    conviction    = max(prob_up, prob_down)
    direction     = "UP" if prob_up >= prob_down else "DOWN"
    is_call       = (direction == "UP")
    action        = "BUY CALL (CE)" if is_call else "BUY PUT (PE)"
    strike_int    = 20
    atm_strike    = round(open_price / strike_int) * strike_int
    option_type   = "CE" if is_call else "PE"
    option_contract = f"{symbol.replace('.NS', '')} ₹{atm_strike} {option_type}"

    from utils.options_feed import get_live_option_quote
    ref_spot = current_price if current_price > 0 else open_price
    quote_data = get_live_option_quote(symbol, atm_strike, is_call=is_call, spot=ref_spot)
    est_entry_premium = quote_data["ltp"]
    quote_source      = quote_data["source"]

    # ── V2 Pre-Trade Gates ─────────────────────────────────────────────────────
    gates_pass, gate_details = check_pre_trade_gates(quote_data, open_prem=est_entry_premium)
    result["gates"] = gate_details

    # ── V2 2-Tranche Pricing Engine ────────────────────────────────────────────
    # Tranche 1: Premium at the spike/dip extreme
    spike_ref_spot = upper_band if not is_call else lower_band
    t1_quote = get_live_option_quote(symbol, atm_strike, is_call=is_call, spot=spike_ref_spot)
    t1_prem = t1_quote["ltp"]
    
    # Tranche 2: Premium at the VWAP breakdown/breakout
    t2_quote = get_live_option_quote(symbol, atm_strike, is_call=is_call, spot=vwap)
    t2_prem = t2_quote["ltp"]
    
    blended_entry = round((t1_prem + t2_prem) / 2.0, 2)

    # ── V2 Hybrid Risk Regime ──────────────────────────────────────────────────
    is_pre_925 = (now.time() < dt.time(9, 25))
    if is_pre_925 and not simulate_morning:
        active_sl_regime = "09:15–09:25 Emergency Buffer (-₹8.00)"
        sl_premium = round(max(blended_entry - EMERGENCY_SL_PTS, 0.5), 2)
        active_sl_pts = EMERGENCY_SL_PTS
    else:
        active_sl_regime = "Post-09:25 Precision Hard Stop (-₹5.00)"
        sl_premium = round(max(blended_entry - PRECISION_SL_PTS, 0.5), 2)
        active_sl_pts = PRECISION_SL_PTS

    lot1_target_prem = round(blended_entry + LOT1_TARGET_PTS, 2)
    capital_needed   = round(blended_entry * LOT_SIZE * lots, 2)
    max_loss_rs      = round(active_sl_pts * LOT_SIZE * lots + 100.0, 2)
    max_gain_lot1    = round(LOT1_TARGET_PTS * LOT_SIZE * 1 - 50.0, 2)

    result.update({
        "conviction":         round(conviction, 1),
        "direction":          direction,
        "action":             action,
        "option_contract":    option_contract,
        "atm_strike":         atm_strike,
        "option_type":        option_type,
        "entry_premium":      round(est_entry_premium, 2),
        "t1_prem":            round(t1_prem, 2),
        "t2_prem":            round(t2_prem, 2),
        "blended_entry":      blended_entry,
        "quote_source":       quote_source,
        "lot1_target_prem":   lot1_target_prem,
        "sl_premium":         sl_premium,
        "active_sl_regime":   active_sl_regime,
        "capital_needed":     capital_needed,
        "max_loss_rs":        max_loss_rs,
        "max_gain_lot1":      max_gain_lot1,
    })

    if conviction < conviction_threshold:
        reasons.append(
            f"⚪ Model conviction {conviction:.1f}% — below {conviction_threshold:.0f}% threshold"
            f"{'(Friday rule)' if is_friday else ''}. Sidelined today."
        )
        result.update({"verdict": "NO-TRADE", "active_state": "STATE 0: SCANNING", "summary": _build_summary(result, reasons, warnings, friday_rules)})
        return result

    # ── CHECK 7: V2 Finite State Machine Detection ─────────────────────────────
    # Check if morning spike formed rejection
    spike_occurred = False
    rejection_formed = False
    first_candle = df_5m.iloc[0] if len(df_5m) > 0 else None
    if first_candle is not None:
        if not is_call and float(first_candle['High']) >= upper_band:
            spike_occurred = True
            rejection_formed = is_rejection_candle(first_candle, is_call=False)
        elif is_call and float(first_candle['Low']) <= lower_band:
            spike_occurred = True
            rejection_formed = is_rejection_candle(first_candle, is_call=True)

    # VWAP Confirmation
    vwap_confirmed = (current_price <= vwap) if not is_call else (current_price >= vwap)
    result["vwap_confirmed"] = vwap_confirmed

    # Determine State
    if not gates_pass:
        active_state = "STATE 0: GATES BLOCKED"
        warnings.append(f"🛡️ Pre-Trade Gates Failed: Spread {gate_details['spread_rs']} (Pass: {gate_details['spread_pass']}) | Chase +{gate_details['chase_gain']} (Pass: {gate_details['chase_pass']})")
    elif spike_occurred and rejection_formed and not vwap_confirmed:
        active_state = "STATE 1: T1 ACTIVE (1 Lot / 475 Qty)"
    elif vwap_confirmed:
        active_state = "STATE 2: FULL POSITION ACTIVE (T1+T2 = 2 Lots / 950 Qty)"
    else:
        active_state = "STATE 0: SCANNING / WAITING FOR VWAP"

    result["active_state"] = active_state

    # ── Friday-specific rules ─────────────────────────────────────────────────
    if is_friday:
        friday_rules.append(f"📅 Friday → Conviction threshold raised to {CONVICTION_FRIDAY}%.")
        friday_rules.append("📅 Friday → Never hold options past 02:45 PM.")

        if direction == "DOWN" and open_to_now > FRIDAY_PUT_UP_THRESHOLD:
            friday_rules.append(
                f"⚠️ Friday PUT WARNING: CDSL spot has moved UP ₹{open_to_now:+.2f} from open. "
                f"If holding PUT past 01:00 PM, EXIT — theta decay accelerates after noon on expiry week."
            )

        if now.hour >= FRIDAY_PUT_CLOSE_HOUR and direction == "DOWN":
            friday_rules.append(
                f"🔴 It is past 01:00 PM on Friday. Close any open PUT positions NOW."
            )

    # ── Final Verdict ─────────────────────────────────────────────────────────
    if not gates_pass:
        reasons.append("🟡 Pre-Trade Gates not passed. Wait for tight spread and settled price.")
        result["verdict"] = "WAIT"
    elif "STATE 1" in active_state:
        reasons.append(f"🎯 Spike Rejection confirmed! Buy Tranche 1 (1 Lot) @ ~₹{t1_prem:.2f}.")
        result["verdict"] = "GO"
    elif "STATE 2" in active_state:
        reasons.append(f"✅ VWAP confirmation active. Trade full 2 Lots @ Blended Avg ~₹{blended_entry:.2f}.")
        result["verdict"] = "GO"
    else:
        reasons.append(f"🟡 Waiting for VWAP confirmation or spike rejection setup.")
        result["verdict"] = "WAIT"

    result["summary"] = _build_summary(result, reasons, warnings, friday_rules,
                                       atm_strike=atm_strike, open_to_now=open_to_now)
    return result


# ─── Summary Builder ──────────────────────────────────────────────────────────

def _build_summary(result: dict, reasons: list, warnings: list,
                    friday_rules: list, atm_strike: float = 0, open_to_now: float = 0) -> str:
    today          = dt.date.today()
    now            = dt.datetime.now()
    is_friday      = result.get("is_friday", False)
    verdict        = result.get("verdict", "UNKNOWN")
    active_state   = result.get("active_state", "STATE 0: SCANNING")
    capital        = result.get("capital", 0.0)
    direction      = result.get("direction")
    conviction     = result.get("conviction", 0.0)
    gap_rs         = result.get("gap_rs", 0.0)
    gap_type       = result.get("gap_type", "UNKNOWN")
    entry_time     = result.get("entry_time", "—")
    lots           = result.get("lots", TOTAL_LOTS)
    action         = result.get("action", "—")
    cur_price      = result.get("current_price", 0.0)
    prev_close     = result.get("prev_close", 0.0)
    consec         = result.get("consecutive_losses", 0)

    # V2 Option & State fields
    contract       = result.get("option_contract", f"CDSL ₹{atm_strike:.0f} ATM")
    t1_prem        = result.get("t1_prem", 0.0)
    t2_prem        = result.get("t2_prem", 0.0)
    blended_entry  = result.get("blended_entry", result.get("entry_premium", 0.0))
    target_prem    = result.get("lot1_target_prem", 0.0)
    sl_prem        = result.get("sl_premium", 0.0)
    sl_regime      = result.get("active_sl_regime", "Precision -₹5.00")
    cap_needed     = result.get("capital_needed", 0.0)
    max_loss       = result.get("max_loss_rs", 0.0)
    max_gain_lot1  = result.get("max_gain_lot1", 0.0)
    vwap_val       = result.get("vwap", cur_price)
    upper_band     = result.get("upper_band", cur_price)
    lower_band     = result.get("lower_band", cur_price)
    ema_20         = result.get("ema_20", cur_price)
    gates          = result.get("gates", {})

    day_label = f"📅 {today.strftime('%a, %d %b %Y')}" + (" (FRIDAY ⚠️)" if is_friday else "")

    verdict_icons = {
        "GO":       "🟢 GO — EXECUTE SETUP",
        "NO-TRADE": "⚪ NO TRADE TODAY (Sidelined)",
        "WAIT":     "🟡 WAIT — GATEKEEPER / SETUP ACTIVE",
        "STOP":     "🔴 STOP — Do NOT trade",
        "UNKNOWN":  "❓ UNKNOWN",
    }

    lines = [
        "=" * 60,
        f"  SCOUT AGENT — V2 FINITE STATE MACHINE REPORT",
        f"  {day_label}   {now.strftime('%I:%M %p')}",
        "=" * 60,
        f"  🏷️ ACTIVE STATE : {active_state}",
        f"  VERDICT        : {verdict_icons.get(verdict, verdict)}",
    ]

    if verdict in ("GO", "WAIT"):
        spread_str = f"₹{gates.get('spread_rs', 0.20):.2f} ({'PASS' if gates.get('spread_pass', True) else 'FAIL'})"
        chase_str  = f"+₹{gates.get('chase_gain', 0.0):.2f} ({'PASS' if gates.get('chase_pass', True) else 'FAIL'})"

        lines += [
            f"  🎟️ CONTRACT    : {contract} (ATM Strike)",
            f"  💵 ENTRY STATUS :",
            f"    • Tranche 1 (1 Lot @ Spike Rejection) : ~₹{t1_prem:.2f}",
            f"    • Tranche 2 (1 Lot @ VWAP Breakdown)  : ~₹{t2_prem:.2f}",
            f"    • Blended Avg Entry (2 Lots / 950 Qty): ~₹{blended_entry:.2f} (Required: ₹{cap_needed:,.0f})",
            f"  🛑 ACTIVE SL   : ₹{sl_prem:.2f} [{sl_regime}]",
            f"  🎯 TARGETS     :",
            f"    • Lot 1 Target : ₹{target_prem:.2f} (+₹{LOT1_TARGET_PTS:.2f} | Lock +₹{max_gain_lot1:,.0f} Cash)",
            f"    • Lot 2 Runner : Trail with 5-Min 20 EMA @ ₹{ema_20:,.2f}",
            f"  🛡️ PRE-GATES   : Spread: {spread_str} | Chase: {chase_str}",
            f"  ⏰ CUTOFF      : 03:05 PM (Hard Market Exit)",
            "-" * 60,
            f"  SPOT & BANDS REFERENCE :",
            f"    • CDSL Live Spot   : ₹{cur_price:,.2f}",
            f"    • Upper Band (+1.5σ): ₹{upper_band:,.2f}  |  Lower Band (-1.5σ): ₹{lower_band:,.2f}",
            f"    • Session VWAP     : ₹{vwap_val:,.2f}",
            f"    • AI Model Convict : {'📉 DOWN' if direction == 'DOWN' else '📈 UP'} @ {conviction:.1f}%",
            "-" * 60,
        ]
    else:
        lines += [
            f"  CAPITAL        : ₹{capital:,.0f}  →  Lots: {lots}",
            f"  PREV CLOSE     : ₹{prev_close:,.2f}",
            f"  TODAY OPEN     : ₹{result.get('open_price', 0):,.2f}  (Gap: ₹{gap_rs:+.2f}  {gap_type})",
            f"  LIVE PRICE     : ₹{cur_price:,.2f}",
            "-" * 60,
        ]

    if consec > 0:
        lines.append(f"  ⚠️ Consecutive SL hits: {consec}")

    if warnings:
        lines.append("  WARNINGS / GATES:")
        for w in warnings:
            lines.append(f"    {w}")

    if friday_rules:
        lines.append("  FRIDAY RULES ACTIVE:")
        for f in friday_rules:
            lines.append(f"    {f}")

    lines.append("  DECISION REASONS:")
    for r in reasons:
        lines.append(f"    {r}")

    lines.append("=" * 60)
    return "\n".join(line for line in lines if line.strip() != "")


# ─── CLI Quick-Run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    sym      = "CDSL.NS"
    cap      = None
    simulate = False

    for arg in sys.argv[1:]:
        if arg == "--simulate":
            simulate = True
        elif arg.replace(".", "").replace("NS", "").replace("BO", "").isalpha():
            sym = arg
        else:
            try:
                cap = float(arg)
            except ValueError:
                pass

    if simulate:
        _orig_NO_ENTRY_AFTER_HOUR = NO_ENTRY_AFTER_HOUR
        _orig_NO_ENTRY_AFTER_MIN  = NO_ENTRY_AFTER_MIN
        import utils.entry_conditions as _ec
        _ec.NO_ENTRY_AFTER_HOUR = 23
        _ec.NO_ENTRY_AFTER_MIN  = 59
        print("[SIMULATION MODE — time cutoff bypassed]\n")

    res = check_entry_conditions(sym, capital_override=cap, simulate_morning=simulate)
    print(res["summary"])

    if simulate:
        _ec.NO_ENTRY_AFTER_HOUR = _orig_NO_ENTRY_AFTER_HOUR
        _ec.NO_ENTRY_AFTER_MIN  = _orig_NO_ENTRY_AFTER_MIN
