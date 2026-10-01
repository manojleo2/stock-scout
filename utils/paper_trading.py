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
    Log a simulated paper trade at 9:20 AM for the AI Intraday +₹4 Scalp Strategy.
    If conviction is below threshold (< 60%), logs a Capital Preserved entry with 0 lots.
    """
    prob_up = float(pred_result.get("probability_up_pct", 50.0))
    prob_down = float(pred_result.get("probability_down_pct", 50.0))
    conviction = max(prob_up, prob_down)

    trade_id = f"SCALP-{symbol}-{target_date_str.replace(' ', '-').replace(',', '')}"
    history = load_paper_trades()
    existing = next((t for t in history if t.get("trade_id") == trade_id), None)

    # If already completed with a final outcome, do not overwrite
    if existing and existing.get("status") in ("✅ WIN", "❌ LOSS", "🛡️ Capital Preserved (No Trade)"):
        return existing

    quant_bp = pred_result.get("quant_blueprint", {}) or pred_result.get("blueprint", {})
    action = quant_bp.get("action", "")
    is_neutral = ("NEUTRAL" in action) or (conviction < 60.0)

    spot = float(pred_result.get("latest_close") or pred_result.get("current_price", 1360.0))
    lot_size = 950 if "CDSL" in symbol else 1100

    if is_neutral:
        record = {
            "trade_id": trade_id,
            "strategy": "AI Intraday +₹4 Scalp",
            "symbol": symbol,
            "entry_date": target_date_str,
            "entry_time": "09:20",
            "exit_date": target_date_str,
            "exit_time": "15:30",
            "action": "⚪ NO TRADE (CASH)",
            "strike": "100% Cash Buffer",
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
            "what_was_expected": f"AI Intraday Model predicted {prob_up:.1f}% Up / {prob_down:.1f}% Down (Neutral Chop). Conviction {conviction:.1f}% is below 60% threshold: recommended NO TRADE to preserve capital.",
            "what_had_happened": f"Position kept in 100% Cash throughout session. Protected capital with ₹0 market exposure and zero brokerage fees."
        }
    else:
        is_call = prob_up >= 50.0
        strike_round = 10 if spot < 500 else (20 if spot < 2000 else 50)
        atm_strike = round(spot / strike_round) * strike_round
        entry_premium = calculate_bsm_option_price(spot, atm_strike, days_to_expiry=10.0, is_call=is_call)
        capital_invested = round(entry_premium * lot_size, 2)
        action_label = "BUY CALL (CE)" if is_call else "BUY PUT (PE)"
        strike_label = f"₹{atm_strike} {'CE' if is_call else 'PE'}"

        record = {
            "trade_id": trade_id,
            "strategy": "AI Intraday +₹4 Scalp",
            "symbol": symbol,
            "entry_date": target_date_str,
            "entry_time": "09:20",
            "exit_date": target_date_str,
            "exit_time": "Pending...",
            "action": action_label,
            "strike": strike_label,
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
            "status": "⏳ Live Scalp Active",
            "what_was_expected": f"AI Up/Down model predicted {action_label} with {conviction:.1f}% confidence. Scalp target: +₹4.00 profit on {strike_label} option premium. Stop-loss: -₹3.00.",
            "what_had_happened": "⏳ Position entered at 9:20 AM. Live 5-minute candle tracking active for +₹4 target or stop-loss."
        }

    if existing:
        existing.update(record)
    else:
        history.append(record)

    save_paper_trades(history)
    return record


def evaluate_simulated_intraday_scalp():
    """
    Evaluates active intraday scalp paper trades against 5-minute intraday candlesticks.
    Detects the exact minute (e.g., 10:05 AM) when target (+₹4) or stop-loss (-₹3) was hit.
    """
    import yfinance as yf

    history = load_paper_trades()
    updated = False

    for trade in history:
        if trade.get("strategy") == "AI Intraday +₹4 Scalp" and "Active" in trade.get("status", ""):
            symbol = trade.get("symbol")
            is_call = trade.get("is_call", True)
            atm_strike = float(trade.get("atm_strike", 1360.0))
            lot_size = int(trade.get("lot_size", 950))
            entry_prem = float(trade.get("entry_premium", 28.0))
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

                    # Clean timezone
                    if df_5m.index.tz is not None:
                        df_5m.index = df_5m.index.tz_localize(None)

                    for idx_time, row in df_5m.iterrows():
                        # Evaluate starting from 09:20 AM
                        if hasattr(idx_time, "time") and idx_time.time() < dt.time(9, 20):
                            continue

                        time_str = idx_time.strftime("%I:%M %p")

                        # BSM option price for bar High and Low
                        prem_high = calculate_bsm_option_price(float(row['High']), atm_strike, days_to_expiry=10.0, is_call=is_call)
                        prem_low  = calculate_bsm_option_price(float(row['Low']), atm_strike, days_to_expiry=10.0, is_call=is_call)

                        favorable_prem = prem_high if is_call else prem_low
                        adverse_prem   = prem_low if is_call else prem_high

                        # 1. Update max favorable premium achieved & Trailing SL
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

                        # 2. Check Target Hit (+₹10.00)
                        if favorable_prem >= target_prem:
                            hit_target = True
                            hit_time_str = time_str
                            exit_prem = target_prem
                            exit_spot = float(row['High'] if is_call else row['Low'])
                            break

                        # 3. Check Stop Loss Hit (Initial or Trailed)
                        if adverse_prem <= current_sl:
                            hit_stop = True
                            hit_time_str = time_str
                            exit_prem = current_sl
                            exit_spot = float(row['Low'] if is_call else row['High'])
                            break

                        # 4. Check Cutoff (03:05 PM Hard Cutoff)
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
                            "what_had_happened": f"🎯 Target +₹10.00 Hit at {hit_time_str}! Option premium expanded from ₹{entry_prem:.2f} to ₹{exit_prem:.2f} (+₹10.00 gain). Net profit: +₹{net_pnl:,.2f}."
                        })
                        updated = True
                    elif hit_stop:
                        gross_pnl = round((exit_prem - entry_prem) * lot_size, 2)
                        net_pnl = round(gross_pnl - BROKERAGE_AND_TAX_PER_TRADE, 2)
                        ret_pct = round((net_pnl / trade.get("capital_invested", 10000.0)) * 100.0, 1)
                        outcome_status = "✅ WIN" if net_pnl > 0 else ("🛡️ BREAKEVEN" if net_pnl >= -BROKERAGE_AND_TAX_PER_TRADE else "❌ LOSS")
                        trade.update({
                            "exit_spot": exit_spot,
                            "exit_premium": exit_prem,
                            "exit_time": hit_time_str,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "return_pct": ret_pct,
                            "status": outcome_status,
                            "what_had_happened": f"🛑 Stop Triggered at {hit_time_str} ({trailing_status}). Premium exited at ₹{exit_prem:.2f}. Net P&L: ₹{net_pnl:+,.2f}."
                        })
                        updated = True
                    elif hit_cutoff:
                        gross_pnl = round((exit_prem - entry_prem) * lot_size, 2)
                        net_pnl = round(gross_pnl - BROKERAGE_AND_TAX_PER_TRADE, 2)
                        ret_pct = round((net_pnl / trade.get("capital_invested", 10000.0)) * 100.0, 1)
                        outcome_status = "✅ WIN" if net_pnl > 0 else ("🛡️ BREAKEVEN" if net_pnl >= -BROKERAGE_AND_TAX_PER_TRADE else "❌ LOSS")
                        trade.update({
                            "exit_spot": exit_spot,
                            "exit_premium": exit_prem,
                            "exit_time": hit_time_str,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "return_pct": ret_pct,
                            "status": outcome_status,
                            "what_had_happened": f"⏱️ 03:05 PM Cutoff Exit. Spot: ₹{exit_spot:,.2f}, Option Premium: ₹{exit_prem:.2f}. Net P&L: ₹{net_pnl:+,.2f}."
                        })
                        updated = True
                    else:
                        current_spot = float(df_5m['Close'].iloc[-1])
                        current_prem = calculate_bsm_option_price(current_spot, atm_strike, days_to_expiry=10.0, is_call=is_call)
                        day_high = float(df_5m['High'].max())
                        day_low = float(df_5m['Low'].min())
                        now_str = dt.datetime.now().strftime("%I:%M %p")
                        trade.update({
                            "what_had_happened": f"⏳ Live In Progress ({now_str}) — Spot: ₹{current_spot:,.2f} (Range: ₹{day_low:,.2f}-₹{day_high:,.2f}). Option Prem: ₹{current_prem:.2f} (Entry: ₹{entry_prem:.2f} | Current SL: ₹{current_sl:.2f} [{trailing_status}] | Target: ₹{target_prem:.2f})."
                        })
                        updated = True
            except Exception as e:
                logging.warning(f"Error evaluating intraday scalp: {e}")

    if updated:
        save_paper_trades(history)
    return history


def sync_today_paper_trades():
    """
    Ensure today's morning forecast is recorded in the paper trading ledger,
    and runs exit evaluation for intraday scalp trades.
    Only checks today's session to ensure past history remains cleared.
    """
    from utils.prediction_audit import load_saved_audit_history

    audit_history = load_saved_audit_history()
    history = load_paper_trades()

    # Collect all existing SCALP trade_ids so we know what's already logged
    existing_scalp_ids = {t.get("trade_id") for t in history if t.get("strategy") == "AI Intraday +₹4 Scalp"}

    today = dt.date.today()
    today_str = today.strftime("%a, %d %b %Y")

    # Only sync today's session if available in audit
    for rec in audit_history:
        target_date = rec.get("target_date", "")
        if target_date != today_str:
            continue
        if "pred_result" not in rec:
            continue
        sym = rec.get("symbol", "CDSL.NS")
        trade_id = f"SCALP-{sym}-{target_date.replace(' ', '-').replace(',', '')}"
        if trade_id not in existing_scalp_ids:
            record_simulated_intraday_entry(sym, target_date, rec["pred_result"])

    # Evaluate active intraday scalps
    evaluate_simulated_intraday_scalp()


