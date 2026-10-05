import os
import json
import logging

logging.basicConfig(level=logging.INFO)

MANUAL_TRADES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "manual_sub60_trades.json")

def load_manual_sub60_trades() -> list:
    """Load manually tracked sub-60% probability trades from JSON file."""
    if os.path.exists(MANUAL_TRADES_FILE):
        try:
            with open(MANUAL_TRADES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            logging.error(f"Error loading manual sub-60 trades: {e}")
    return []

def save_manual_sub60_trades(trades: list):
    """Save manual sub-60 trades to JSON file."""
    try:
        with open(MANUAL_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving manual sub-60 trades: {e}")

def add_or_update_manual_trade(trade_data: dict, orig_index: int = None) -> list:
    """Add a new manual trade or update an existing one."""
    trades = load_manual_sub60_trades()
    
    if orig_index is not None and 0 <= orig_index < len(trades):
        trades[orig_index] = trade_data
    else:
        d_target = str(trade_data.get("Date", "")).strip()
        stock_target = str(trade_data.get("Stock", "")).strip()

        existing_idx = next(
            (i for i, t in enumerate(trades) if str(t.get("Date", "")).strip() == d_target and str(t.get("Stock", "")).strip() == stock_target),
            None
        )

        if existing_idx is not None:
            trades[existing_idx] = trade_data
        else:
            trades.append(trade_data)

    save_manual_sub60_trades(trades)
    return trades

def delete_manual_trade_by_index(index: int) -> list:
    """Delete a manual trade record by its index."""
    trades = load_manual_sub60_trades()
    if 0 <= index < len(trades):
        trades.pop(index)
        save_manual_sub60_trades(trades)
    return trades

def delete_manual_trade(date_str: str, stock_str: str) -> list:
    """Delete a manual trade record by Date and Stock."""
    trades = load_manual_sub60_trades()
    trades = [t for t in trades if not (str(t.get("Date", "")).strip() == date_str.strip() and str(t.get("Stock", "")).strip() == stock_str.strip())]
    save_manual_sub60_trades(trades)
    return trades

