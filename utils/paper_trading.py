import os
import json
import math
import logging
import datetime as dt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)

LEDGER_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_trading_ledger.json")

# Standard parameters for Indian stock options
BROKERAGE_AND_TAX_PER_TRADE = 100.0  # ₹100 flat round-trip brokerage + STT + exchange turnover
DEFAULT_STARTING_BANKROLL = 60000.0   # ₹60,000 virtual capital
ANNUAL_RISK_FREE_RATE = 0.065        # 6.5% RBI Repo Rate proxy
DEFAULT_STOCK_IV = 0.32              # 32% Implied Volatility for CDSL

def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function approximation."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def calculate_bsm_option_price(spot: float, strike: float, days_to_expiry: float, iv: float = DEFAULT_STOCK_IV, is_call: bool = True) -> float:
    """
    Compute European Option price using the Black-Scholes-Merton formula.
    Accurately captures Delta, Gamma, and calendar Theta time decay.
    """
    if spot <= 0 or strike <= 0:
        return 0.05
    
    T = max(days_to_expiry, 0.5) / 365.0
    r = ANNUAL_RISK_FREE_RATE
    sigma = max(iv, 0.10)

    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if is_call:
            price = spot * normal_cdf(d1) - strike * math.exp(-r * T) * normal_cdf(d2)
        else:
            price = strike * math.exp(-r * T) * normal_cdf(-d2) - spot * normal_cdf(-d1)

        return round(max(float(price), 0.05), 2)
    except Exception as e:
        logging.warning(f"BSM calculation fallback: {e}")
        # Intrinsic value fallback
        intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
        return round(max(intrinsic, 0.05), 2)

def load_paper_trades() -> list:
    """Load simulated trade records from JSON ledger."""
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    return saved
        except Exception as e:
            logging.error(f"Error loading paper trades ledger: {e}")
    return []

def save_paper_trades(trades: list):
    """Persist simulated trade records to JSON ledger."""
    try:
        with open(LEDGER_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving paper trades ledger: {e}")

def get_portfolio_performance_summary(trades: list = None) -> dict:
    """
    Compute comprehensive paper trading metrics:
    Starting Bankroll, Current Bankroll, Net P&L, Win Rate %, Profit Factor, and Equity Curve.
    """
    if trades is None:
        trades = load_paper_trades()

    completed = [t for t in trades if t.get("status") in ("✅ WIN", "❌ LOSS")]

    starting_capital = DEFAULT_STARTING_BANKROLL
    current_capital = starting_capital
    total_net_pnl = 0.0
    total_gross_pnl = 0.0
    total_brokerage = 0.0

    wins = 0
    losses = 0
    gross_wins = 0.0
    gross_losses = 0.0

    equity_curve = [{"Date": "Initial", "Bankroll": starting_capital, "Net P&L": 0.0}]

    for t in completed:
        net_pl = float(t.get("net_pnl", 0.0))
        gross_pl = float(t.get("gross_pnl", 0.0))
        brok = float(t.get("brokerage_tax", BROKERAGE_AND_TAX_PER_TRADE))

        total_net_pnl += net_pl
        total_gross_pnl += gross_pl
        total_brokerage += brok
        current_capital += net_pl

        if net_pl > 0:
            wins += 1
            gross_wins += net_pl
        else:
            losses += 1
            gross_losses += abs(net_pl)

        date_label = t.get("exit_date") or t.get("entry_date") or "Unknown"
        equity_curve.append({
            "Date": date_label,
            "Bankroll": round(current_capital, 2),
            "Net P&L": round(net_pl, 2),
            "Trade": f"{t.get('strategy')} ({t.get('symbol')})"
        })

    total_completed = len(completed)
    win_rate_pct = round((wins / total_completed * 100.0), 1) if total_completed > 0 else 0.0
    profit_factor = round((gross_wins / gross_losses), 2) if gross_losses > 0 else (round(gross_wins, 2) if gross_wins > 0 else 1.0)
    net_return_pct = round((total_net_pnl / starting_capital) * 100.0, 1)

    return {
        "starting_capital": starting_capital,
        "current_capital": round(current_capital, 2),
        "total_net_pnl": round(total_net_pnl, 2),
        "total_gross_pnl": round(total_gross_pnl, 2),
        "total_brokerage": round(total_brokerage, 2),
        "net_return_pct": net_return_pct,
        "total_trades": total_completed,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "equity_curve": equity_curve
    }


def record_simulated_intraday_entry(symbol: str, target_date_str: str, pred_result: dict):
    """
    Log a simulated paper trade for the CDSL Intraday Options Strategy.
    1. Pre-09:20 AM: Logs state as '⏳ In Progress' (Awaiting 09:20 Entry), entry_premium=None.
    2. Post-09:20 AM: Records entry premium, computes Target (+₹10) & SL (-₹5).
    3. If conviction is below threshold (Weekday < 65%, Friday < 70%), logs Capital Preserved.
    """
    import yfinance as yf
    from utils.options_feed import get_live_option_quote

    prob_up = float(pred_result.get("probability_up_pct", 50.0))
    prob_down = float(pred_result.get("probability_down_pct", 50.0))
    conviction = float(pred_result.get("calibrated_conviction_pct") or pred_result.get("predicted_conviction") or max(prob_up, prob_down))

    # Clean target date formatting (e.g. "05 Oct 2026")
    try:
        parsed_d = dt.datetime.strptime(target_date_str, "%d %b %Y").date()
    except Exception:
        try:
            parsed_d = dt.datetime.strptime(target_date_str, "%a, %d %b %Y").date()
        except Exception:
            parsed_d = dt.date.today()

    norm_date_str = parsed_d.strftime("%d %b %Y")
    trade_id = f"TRADE-{symbol}-{parsed_d.strftime('%Y%m%d')}"
    history = load_paper_trades()
    existing = next((t for t in history if (t.get("trade_id") == trade_id or norm_date_str in str(t.get("entry_date", "")))), None)

    # If already completed with a final outcome, do not overwrite
    if existing and existing.get("status") in ("✅ WIN", "❌ LOSS", "🛡️ Capital Preserved (No Trade)"):
        return existing

    quant_bp = pred_result.get("quant_blueprint", {}) or pred_result.get("blueprint", {})
    action = quant_bp.get("action", "")

    # Rulebook Conviction Gate: Weekday >= 65%, Friday >= 70%
    is_friday = (parsed_d.weekday() == 4)
    min_conviction = 70.0 if is_friday else 65.0
    is_neutral = ("NEUTRAL" in action) or (conviction < min_conviction)

    spot = float(pred_result.get("latest_close") or pred_result.get("current_price") or pred_result.get("baseline_close") or 1260.0)
    lot_size = 950 if "CDSL" in symbol else 1100

    now_utc = dt.datetime.now(dt.timezone.utc)
    now_ist = now_utc + dt.timedelta(hours=5, minutes=30)
    is_today = (parsed_d == now_ist.date())
    is_before_entry = is_today and (now_ist.time() < dt.time(9, 20))

    if is_neutral:
        record = {
            "trade_id": trade_id,
            "strategy": "CDSL Intraday Options",
            "symbol": symbol,
            "entry_date": norm_date_str,
            "entry_time": "09:20 AM",
            "exit_date": norm_date_str,
            "exit_time": "03:05 PM",
            "action": "⚪ NO TRADE (CASH)",
            "strike": "100% Cash Buffer",
            "contract": f"{'CDSL' if 'CDSL' in symbol else symbol} Cash Buffer",
            "is_call": False,
            "atm_strike": spot,
            "lot_size": 0,
            "entry_spot": spot,
            "entry_premium": 0.0,
            "exit_spot": spot,
            "exit_premium": 0.0,
            "capital_invested": 0.0,
            "gross_pnl": 0.0,
            "brokerage_tax": 0.0,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "status": "🛡️ Capital Preserved (No Trade)",
            "ai_probability": f"{conviction:.1f}% NEUTRAL",
            "what_was_expected": f"AI model predicted {prob_up:.1f}% Up / {prob_down:.1f}% Down. Conviction {conviction:.1f}% is below {min_conviction:.0f}% threshold: recommended NO TRADE to preserve capital.",
            "what_had_happened": "Position kept in 100% Cash. Protected capital with ₹0 market exposure.",
            "cutoff": "—"
        }
    else:
        is_call = prob_up >= 50.0
        strike_round = 10 if spot < 500 else (20 if spot < 2000 else 50)
        atm_strike = round(spot / strike_round) * strike_round
        action_label = "BUY CALL (CE)" if is_call else "BUY PUT (PE)"
        strike_label = f"{atm_strike} {'CE' if is_call else 'PE'}"
        contract_label = f"{'CDSL' if 'CDSL' in symbol else symbol} {strike_label}"
        ai_prob_str = f"{conviction:.1f}% {'UP' if is_call else 'DOWN'}"

        if is_before_entry:
            record = {
                "trade_id": trade_id,
                "strategy": "CDSL Intraday Options",
                "symbol": symbol,
                "entry_date": norm_date_str,
                "entry_time": "09:20 AM",
                "exit_date": norm_date_str,
                "exit_time": None,
                "action": action_label,
                "strike": strike_label,
                "contract": contract_label,
                "is_call": is_call,
                "atm_strike": atm_strike,
                "lot_size": lot_size,
                "entry_spot": spot,
                "entry_premium": None,
                "exit_spot": None,
                "exit_premium": None,
                "capital_invested": 0.0,
                "gross_pnl": None,
                "brokerage_tax": BROKERAGE_AND_TAX_PER_TRADE,
                "net_pnl": None,
                "return_pct": None,
                "status": "⏳ In Progress",
                "ai_probability": ai_prob_str,
                "what_was_expected": f"AI model predicted {action_label} with {conviction:.1f}% confidence. Target: +₹10.00 | SL: -₹5.00.",
                "what_had_happened": f"⏳ Pre-Market Setup: AI predicted {action_label} ({conviction:.1f}%). Awaiting 09:18–09:20 VWAP confirmation & 09:20 AM entry fill.",
                "cutoff": "03:05 PM Hard Cutoff"
            }
        else:
            entry_premium = None
            try:
                t_obj = yf.Ticker(symbol)
                df_5m = t_obj.history(period="1d", interval="5m")
                if not df_5m.empty:
                    for idx_t, row in df_5m.iterrows():
                        if hasattr(idx_t, "time") and idx_t.time() >= dt.time(9, 20):
                            spot = float(row['Open'])
                            break
                q = get_live_option_quote(symbol, atm_strike, is_call=is_call, spot=spot)
                entry_premium = q.get("ltp")
            except Exception:
                pass

            if not entry_premium or float(entry_premium) <= 0:
                entry_premium = calculate_bsm_option_price(spot, atm_strike, days_to_expiry=10.0, is_call=is_call)

            entry_premium = round(float(entry_premium), 2)
            capital_invested = round(entry_premium * lot_size, 2)
            tgt_prem = round(entry_premium + 10.0, 2)
            sl_prem = round(max(entry_premium - 5.0, 0.5), 2)

            record = {
                "trade_id": trade_id,
                "strategy": "CDSL Intraday Options",
                "symbol": symbol,
                "entry_date": norm_date_str,
                "entry_time": "09:20 AM",
                "exit_date": norm_date_str,
                "exit_time": None,
                "action": action_label,
                "strike": strike_label,
                "contract": contract_label,
                "is_call": is_call,
                "atm_strike": atm_strike,
                "lot_size": lot_size,
                "entry_spot": spot,
                "entry_premium": entry_premium,
                "exit_spot": None,
                "exit_premium": None,
                "capital_invested": capital_invested,
                "gross_pnl": None,
                "brokerage_tax": BROKERAGE_AND_TAX_PER_TRADE,
                "net_pnl": None,
                "return_pct": None,
                "status": "⏳ In Progress",
                "ai_probability": ai_prob_str,
                "what_was_expected": f"AI model predicted {action_label} with {conviction:.1f}% confidence. Target: +₹10.00 (₹{tgt_prem:.2f}) | SL: -₹5.00 (₹{sl_prem:.2f}).",
                "what_had_happened": f"⏳ Position entered at 09:20 AM @ ₹{entry_premium:.2f} (Spot ₹{spot:,.1f}). Active monitoring for +₹10 target or -₹5 SL.",
                "cutoff": "03:05 PM Hard Cutoff"
            }

    if existing:
        existing.update(record)
    else:
        history.append(record)

    save_paper_trades(history)
    return record


def evaluate_simulated_intraday_scalp():
    """
    Evaluates active intraday paper trades against 5-minute candlesticks.
    Monitors Target (+₹10.00), Stop-Loss (-₹5.00), Trailing SL (+7 / +8.5), and 03:05 PM cutoff.
    """
    import yfinance as yf
    from utils.options_feed import get_live_option_quote

    now_utc = dt.datetime.now(dt.timezone.utc)
    now_ist = now_utc + dt.timedelta(hours=5, minutes=30)
    history = load_paper_trades()
    updated = False

    for trade in history:
        status_val = str(trade.get("status", ""))
        if not ("In Progress" in status_val or "Active" in status_val):
            continue

        symbol = trade.get("symbol", "CDSL.NS")
        is_call = trade.get("is_call", False)
        atm_strike = float(trade.get("atm_strike", 1260.0))
        lot_size = int(trade.get("lot_size", 950))
        entry_date_str = str(trade.get("entry_date", ""))

        try:
            trade_d = dt.datetime.strptime(entry_date_str, "%d %b %Y").date()
        except Exception:
            trade_d = now_ist.date()

        # If today and before 09:20 AM, wait for entry candle
        if trade_d == now_ist.date() and now_ist.time() < dt.time(9, 20):
            continue

        # If entry premium is pending, fill it at 09:20 AM
        entry_prem = trade.get("entry_premium")
        if entry_prem is None or float(entry_prem) <= 0:
            spot = float(trade.get("entry_spot") or 1260.0)
            try:
                q = get_live_option_quote(symbol, atm_strike, is_call=is_call, spot=spot)
                entry_prem = q.get("ltp") or calculate_bsm_option_price(spot, atm_strike, days_to_expiry=10.0, is_call=is_call)
            except Exception:
                entry_prem = calculate_bsm_option_price(spot, atm_strike, days_to_expiry=10.0, is_call=is_call)
            entry_prem = round(float(entry_prem), 2)
            trade["entry_premium"] = entry_prem
            trade["capital_invested"] = round(entry_prem * lot_size, 2)
            trade["what_had_happened"] = f"⏳ Position entered at 09:20 AM @ ₹{entry_prem:.2f}. Active monitoring for +₹10 target or -₹5 SL."
            updated = True

        entry_prem = float(trade["entry_premium"])
        target_prem = round(entry_prem + 10.0, 2)
        current_sl = round(max(entry_prem - 5.0, 0.5), 2)
        max_favorable_prem = entry_prem
        trailing_status = "Initial SL (-₹5.00)"

        try:
            t = yf.Ticker(symbol)
            df_5m = t.history(period="1d", interval="5m")
            if not df_5m.empty and len(df_5m) >= 2:
                hit_target = False
                hit_stop = False
                hit_cutoff = False
                hit_time_str = None
                exit_prem = entry_prem
                exit_spot = float(df_5m['Close'].iloc[-1])

                if df_5m.index.tz is not None:
                    df_5m.index = df_5m.index.tz_localize(None)

                for idx_time, row in df_5m.iterrows():
                    if hasattr(idx_time, "time") and idx_time.time() < dt.time(9, 20):
                        continue

                    time_str = idx_time.strftime("%I:%M %p")
                    prem_high = calculate_bsm_option_price(float(row['High']), atm_strike, days_to_expiry=10.0, is_call=is_call)
                    prem_low  = calculate_bsm_option_price(float(row['Low']), atm_strike, days_to_expiry=10.0, is_call=is_call)

                    favorable_prem = prem_high if is_call else prem_low
                    adverse_prem   = prem_low if is_call else prem_high

                    # Update trailing stop rules
                    if favorable_prem > max_favorable_prem:
                        max_favorable_prem = favorable_prem
                        gain = max_favorable_prem - entry_prem
                        if gain >= 8.5:
                            if current_sl < entry_prem + 5.0:
                                current_sl = round(entry_prem + 5.0, 2)
                                trailing_status = "Trailed to +₹5.00 (Profit Locked)"
                        elif gain >= 7.0:
                            if current_sl < entry_prem:
                                current_sl = round(entry_prem, 2)
                                trailing_status = "Trailed to Cost (Zero Risk)"

                    # Check Target Hit (+₹10.00)
                    if favorable_prem >= target_prem:
                        hit_target = True
                        hit_time_str = time_str
                        exit_prem = target_prem
                        exit_spot = float(row['High'] if is_call else row['Low'])
                        break

                    # Check Stop Loss Hit (-₹5.00 or trailed)
                    if adverse_prem <= current_sl:
                        hit_stop = True
                        hit_time_str = time_str
                        exit_prem = current_sl
                        exit_spot = float(row['Low'] if is_call else row['High'])
                        break

                    # Check 03:05 PM Cutoff Hit
                    if hasattr(idx_time, "time") and idx_time.time() >= dt.time(15, 5):
                        hit_cutoff = True
                        hit_time_str = time_str
                        exit_spot = float(row['Close'])
                        exit_prem = calculate_bsm_option_price(exit_spot, atm_strike, days_to_expiry=10.0, is_call=is_call)
                        break

                if hit_target:
                    gross_pnl = round((exit_prem - entry_prem) * lot_size, 2)
                    net_pnl = round(gross_pnl - BROKERAGE_AND_TAX_PER_TRADE, 2)
                    ret_pct = round((net_pnl / trade.get("capital_invested", 10000.0)) * 100.0, 1)
                    trade.update({
                        "exit_spot": exit_spot,
                        "exit_premium": exit_prem,
                        "exit_time": hit_time_str,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "return_pct": ret_pct,
                        "status": "✅ WIN",
                        "cutoff": "Not Reached (Hit Target)",
                        "what_had_happened": f"🎯 Target +₹10.00 Hit at {hit_time_str}! Option premium reached ₹{exit_prem:.2f}."
                    })
                    updated = True
                elif hit_stop:
                    gross_pnl = round((exit_prem - entry_prem) * lot_size, 2)
                    net_pnl = round(gross_pnl - BROKERAGE_AND_TAX_PER_TRADE, 2)
                    ret_pct = round((net_pnl / trade.get("capital_invested", 10000.0)) * 100.0, 1)
                    outcome_status = "✅ WIN" if net_pnl > 0 else "❌ LOSS"
                    trade.update({
                        "exit_spot": exit_spot,
                        "exit_premium": exit_prem,
                        "exit_time": hit_time_str,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "return_pct": ret_pct,
                        "status": outcome_status,
                        "cutoff": "Not Reached (Hit SL)",
                        "what_had_happened": f"🛑 Stop Triggered at {hit_time_str} ({trailing_status}). Premium exited at ₹{exit_prem:.2f}."
                    })
                    updated = True
                elif hit_cutoff:
                    gross_pnl = round((exit_prem - entry_prem) * lot_size, 2)
                    net_pnl = round(gross_pnl - BROKERAGE_AND_TAX_PER_TRADE, 2)
                    ret_pct = round((net_pnl / trade.get("capital_invested", 10000.0)) * 100.0, 1)
                    outcome_status = "✅ WIN" if net_pnl > 0 else "❌ LOSS"
                    trade.update({
                        "exit_spot": exit_spot,
                        "exit_premium": exit_prem,
                        "exit_time": hit_time_str,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "return_pct": ret_pct,
                        "status": outcome_status,
                        "cutoff": "03:05 PM Market Exit",
                        "what_had_happened": f"⏱️ 03:05 PM Cutoff Exit. Spot: ₹{exit_spot:,.2f}, Option Premium: ₹{exit_prem:.2f}."
                    })
                    updated = True
                else:
                    current_spot = float(df_5m['Close'].iloc[-1])
                    current_prem = calculate_bsm_option_price(current_spot, atm_strike, days_to_expiry=10.0, is_call=is_call)
                    now_str = dt.datetime.now().strftime("%I:%M %p")
                    trade.update({
                        "what_had_happened": f"⏳ Live In Progress ({now_str}) — Spot: ₹{current_spot:,.2f}. Option Prem: ₹{current_prem:.2f} (Entry: ₹{entry_prem:.2f} | Current SL: ₹{current_sl:.2f} [{trailing_status}] | Target: ₹{target_prem:.2f})."
                    })
                    updated = True
        except Exception as e:
            logging.warning(f"Error evaluating intraday trade: {e}")

    if updated:
        save_paper_trades(history)
    return history


def sync_today_paper_trades():
    """
    Ensure today's morning forecast is recorded in the paper trading ledger,
    and runs exit evaluation for intraday trades.
    1. By 09:00 AM: Logs trade row with AI probability and '⏳ In Progress'.
    2. At/After 09:20 AM: Fills entry premium, targets, and tracks exits.
    """
    from utils.prediction_audit import load_saved_audit_history

    now_utc = dt.datetime.now(dt.timezone.utc)
    now_ist = now_utc + dt.timedelta(hours=5, minutes=30)
    today_ist = now_ist.date()

    today_str_short = today_ist.strftime("%d %b %Y")       # e.g., "05 Oct 2026"
    today_str_full  = today_ist.strftime("%a, %d %b %Y")  # e.g., "Mon, 05 Oct 2026"

    history = load_paper_trades()
    existing_today = next(
        (t for t in history if (today_str_short in str(t.get("entry_date", "")) or today_str_short in str(t.get("trade_id", "")))),
        None
    )

    if not existing_today and now_ist.weekday() < 5:
        audit_history = load_saved_audit_history()
        pred_rec = next(
            (r for r in audit_history if (r.get("symbol") == "CDSL.NS" and (today_str_short in str(r.get("target_date", "")) or today_str_full in str(r.get("target_date", ""))))),
            None
        )

        if not pred_rec and now_ist.time() >= dt.time(8, 30):
            try:
                from utils.ml_model import train_and_predict
                from utils.prediction_audit import record_prediction
                res = train_and_predict("CDSL.NS", period="2y")
                if res.get("status") == "success":
                    record_prediction("CDSL.NS", today_str_full, res)
                    pred_rec = {"symbol": "CDSL.NS", "target_date": today_str_full, "pred_result": res}
            except Exception as e:
                logging.warning(f"Auto-predict in sync failed: {e}")

        if pred_rec:
            pred_data = pred_rec.get("pred_result") or pred_rec
            record_simulated_intraday_entry("CDSL.NS", today_str_short, pred_data)

    evaluate_simulated_intraday_scalp()


