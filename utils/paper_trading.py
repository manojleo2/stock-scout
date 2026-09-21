import os
import json
import math
import logging
import datetime as dt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)

LEDGER_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_trading_ledger.json")

from config import MIN_GAP_CONVICTION_THRESHOLD

# Standard parameters for Indian stock options
BROKERAGE_AND_TAX_PER_TRADE = 100.0  # ₹100 flat round-trip brokerage + STT + exchange turnover
DEFAULT_STARTING_BANKROLL = 50000.0   # ₹50,000 virtual capital
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

def record_simulated_gap_entry(symbol: str, target_date_str: str, pred_result: dict):
    """
    Log a simulated paper trade at 3:10 PM for the 3:05 PM Gap Overnight Strategy.
    """
    prob_up = float(pred_result.get("probability_up_pct", 50.0))
    conviction = max(prob_up, 100.0 - prob_up)
    
    # Enforce minimum conviction filter (skip trades below 65% conviction)
    if conviction < MIN_GAP_CONVICTION_THRESHOLD:
        logging.info(f"Paper trade skipped for {symbol}: Conviction ({conviction:.1f}%) below {MIN_GAP_CONVICTION_THRESHOLD:.0f}% threshold.")
        return None

    options_call = pred_result.get("options_call", {})
    if "NEUTRAL" in options_call.get("action", "") or not options_call.get("is_tradeable", True):
        logging.info(f"Paper trade skipped for {symbol}: Action is NEUTRAL / NO TRADE.")
        return None

    is_call = prob_up >= 50.0
    spot = float(pred_result.get("current_price") or pred_result.get("latest_close", 1360.0))
    
    strike_round = 10 if spot < 500 else (20 if spot < 2000 else 50)
    atm_strike = round(spot / strike_round) * strike_round
    lot_size = 700 if "CDSL" in symbol else 1000  # 2 lots for ₹50,000 standard bankroll

    # Calculate realistic BSM entry premium at 3:10 PM (approx 10-15 days to expiry)
    entry_premium = calculate_bsm_option_price(spot, atm_strike, days_to_expiry=12.0, is_call=is_call)
    capital_invested = round(entry_premium * lot_size, 2)

    action_label = "BUY CALL (CE)" if is_call else "BUY PUT (PE)"
    strike_label = f"₹{atm_strike} {'CE' if is_call else 'PE'}"

    what_expected = (
        f"AI 3:05 PM Model predicted {'GAP UP 📈' if is_call else 'GAP DOWN 📉'} with {prob_up if is_call else round(100-prob_up,1)}% confidence. "
        f"Expected overnight gap {'surge above' if is_call else 'drop below'} ₹{atm_strike} at tomorrow's 9:15 AM open."
    )

    trade_id = f"GAP-{symbol}-{target_date_str.replace(' ', '-').replace(',', '')}"
    history = load_paper_trades()

    existing = next((t for t in history if t.get("trade_id") == trade_id), None)
    record = {
        "trade_id": trade_id,
        "strategy": "3:05 PM Gap Overnight",
        "symbol": symbol,
        "entry_date": dt.date.today().strftime("%a, %d %b %Y"),
        "entry_time": "15:10",
        "exit_date": target_date_str,
        "exit_time": "09:18",
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
        "status": "⏳ Overnight Carry Active",
        "what_was_expected": what_expected,
        "what_had_happened": "⏳ Position carried overnight. Awaiting tomorrow morning 9:15 AM market open evaluation."
    }

    if existing:
        if "Active" in existing.get("status", ""):
            existing.update(record)
    else:
        history.append(record)

    save_paper_trades(history)
    return record

def evaluate_simulated_gap_exit():
    """
    Evaluate all pending overnight paper trades after 9:15 AM market open.
    Computes exact BSM exit premium, gross/net P&L, and generates 'What Had Happened' narrative.
    """
    from utils.data_loader import get_stock_data

    history = load_paper_trades()
    updated = False
    today = dt.date.today()

    for trade in history:
        if trade.get("strategy") == "3:05 PM Gap Overnight" and "Active" in trade.get("status", ""):
            target_str = trade.get("exit_date", "")
            try:
                target_date = dt.datetime.strptime(target_str, "%a, %d %b %Y").date()
            except Exception:
                try:
                    target_date = dt.datetime.strptime(target_str, "%Y-%m-%d").date()
                except Exception:
                    target_date = today

            # Evaluate only if target session has opened
            if target_date <= today:
                symbol = trade.get("symbol")
                df_stock = get_stock_data(symbol, period="1mo")

                if not df_stock.empty and len(df_stock) >= 2:
                    df_dates = pd.to_datetime(df_stock.index).date
                    matching = [idx for idx, d in enumerate(df_dates) if d == target_date]

                    if matching:
                        t_bar = df_stock.iloc[matching[-1]]
                        open_price = float(t_bar["Open"])
                        # Approximate 9:18 AM price (midpoint of open and high/low depending on trend)
                        exit_spot = round(open_price, 2)

                        is_call = trade.get("is_call", True)
                        atm_strike = float(trade.get("atm_strike", open_price))
                        lot_size = int(trade.get("lot_size", 350))
                        entry_prem = float(trade.get("entry_premium", 30.0))

                        # Exit option price via BSM (11.0 days to expiry, 1 night theta elapsed)
                        exit_prem = calculate_bsm_option_price(exit_spot, atm_strike, days_to_expiry=11.0, is_call=is_call)
                        
                        gross_pnl = round((exit_prem - entry_prem) * lot_size, 2)
                        brok = BROKERAGE_AND_TAX_PER_TRADE
                        net_pnl = round(gross_pnl - brok, 2)
                        ret_pct = round((net_pnl / trade.get("capital_invested", 10000.0)) * 100.0, 1)

                        is_win = net_pnl > 0
                        status = "✅ WIN" if is_win else "❌ LOSS"

                        entry_spot = trade.get("entry_spot", open_price)
                        gap_diff = exit_spot - entry_spot
                        gap_pct = round((gap_diff / entry_spot) * 100.0, 2)

                        if is_win:
                            what_happened = (
                                f"Stock opened at ₹{open_price:,.2f} ({gap_diff:+.2f} ₹ / {gap_pct:+.2f}% gap) in line with prediction. "
                                f"Option premium expanded from ₹{entry_prem:.2f} to ₹{exit_prem:.2f}. "
                                f"Exited at 9:18 AM capturing +₹{gross_pnl:,.2f} profit (+₹{net_pnl:,.2f} net after taxes)."
                            )
                        else:
                            what_happened = (
                                f"Stock opened at ₹{open_price:,.2f} ({gap_diff:+.2f} ₹ move). "
                                f"Overnight theta time decay and opening counter-move reduced option premium from ₹{entry_prem:.2f} to ₹{exit_prem:.2f}. "
                                f"Exited at 9:18 AM at ₹{exit_prem:.2f} to contain loss to -₹{abs(net_pnl):,.2f}."
                            )

                        trade.update({
                            "exit_spot": exit_spot,
                            "exit_premium": exit_prem,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "return_pct": ret_pct,
                            "status": status,
                            "what_had_happened": what_happened
                        })
                        updated = True

    if updated:
        save_paper_trades(history)
    return history


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
    lot_size = 700 if "CDSL" in symbol else 1100

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
            lot_size = int(trade.get("lot_size", 700))
            entry_prem = float(trade.get("entry_premium", 28.0))
            target_prem = round(entry_prem + 4.0, 2)
            stop_prem = round(max(entry_prem - 3.0, 0.5), 2)

            try:
                t = yf.Ticker(symbol)
                df_5m = t.history(period="1d", interval="5m")
                if not df_5m.empty and len(df_5m) >= 2:
                    hit_target = False
                    hit_stop = False
                    hit_time_str = None
                    exit_prem = entry_prem
                    exit_spot = float(df_5m['Close'].iloc[-1])

                    # Clean timezone
                    if df_5m.index.tz is not None:
                        df_5m.index = df_5m.index.tz_localize(None)

                    for idx_time, row in df_5m.iterrows():
                        time_str = idx_time.strftime("%I:%M %p")

                        # BSM option price for bar High and Low
                        prem_high = calculate_bsm_option_price(float(row['High']), atm_strike, days_to_expiry=10.0, is_call=is_call)
                        prem_low = calculate_bsm_option_price(float(row['Low']), atm_strike, days_to_expiry=10.0, is_call=is_call)

                        if is_call:
                            if prem_high >= target_prem:
                                hit_target = True
                                hit_time_str = time_str
                                exit_prem = target_prem
                                exit_spot = float(row['High'])
                                break
                            elif prem_low <= stop_prem:
                                hit_stop = True
                                hit_time_str = time_str
                                exit_prem = stop_prem
                                exit_spot = float(row['Low'])
                                break
                        else:
                            if prem_high >= target_prem:
                                hit_target = True
                                hit_time_str = time_str
                                exit_prem = target_prem
                                exit_spot = float(row['Low'])
                                break
                            elif prem_low <= stop_prem:
                                hit_stop = True
                                hit_time_str = time_str
                                exit_prem = stop_prem
                                exit_spot = float(row['High'])
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
                            "what_had_happened": f"🎯 Target +₹4.00 Hit at {hit_time_str}! Option premium expanded from ₹{entry_prem:.2f} to ₹{exit_prem:.2f} (+₹4.00 gain). Scalp target reached, locking in +₹{net_pnl:,.2f} net profit."
                        })
                        updated = True
                    elif hit_stop:
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
                            "status": "❌ LOSS",
                            "what_had_happened": f"🛑 Stop-Loss Triggered at {hit_time_str}. Option premium fell to ₹{exit_prem:.2f}. Exited to contain loss to -₹{abs(net_pnl):,.2f}."
                        })
                        updated = True
                    else:
                        current_spot = float(df_5m['Close'].iloc[-1])
                        current_prem = calculate_bsm_option_price(current_spot, atm_strike, days_to_expiry=10.0, is_call=is_call)
                        day_high = float(df_5m['High'].max())
                        day_low = float(df_5m['Low'].min())
                        now_str = dt.datetime.now().strftime("%I:%M %p")
                        trade.update({
                            "what_had_happened": f"⏳ Live In Progress ({now_str}) — Current Spot: ₹{current_spot:,.2f} (Day Range: ₹{day_low:,.2f} - ₹{day_high:,.2f}). Current Option: ₹{current_prem:.2f} vs Entry ₹{entry_prem:.2f}. Target +₹4.00 (₹{target_prem:.2f}) / SL (₹{stop_prem:.2f}) pending."
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
    and runs exit evaluation for both overnight gap and intraday scalp trades.
    """
    from utils.prediction_audit import load_saved_audit_history

    # Check if today's intraday forecast exists in audit
    audit_history = load_saved_audit_history()
    today_str = dt.date.today().strftime("%a, %d %b %Y")

    for rec in reversed(audit_history):
        # Match today's target date
        if rec.get("target_date") == today_str and "pred_result" in rec:
            sym = rec.get("symbol", "CDSL.NS")
            record_simulated_intraday_entry(sym, today_str, rec["pred_result"])
            break

    # Evaluate both gap exits and intraday scalps
    evaluate_simulated_gap_exit()
    evaluate_simulated_intraday_scalp()


