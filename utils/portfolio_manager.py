import os
import json
import logging
import pandas as pd
from utils.data_loader import get_stock_fundamentals
from config import STOCK_NAME_MAP

logging.basicConfig(level=logging.INFO)

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolio.json")

DEFAULT_PORTFOLIO = [
    {
        "symbol": "CDSL.NS",
        "buy_price": 1250.00,
        "quantity": 50,
        "target_price": 1600.00,
        "notes": "Primary depository holding"
    },
    {
        "symbol": "NSDL.BO",
        "buy_price": 800.00,
        "quantity": 75,
        "target_price": 1000.00,
        "notes": "IPO allocation holding"
    }
]

def load_saved_portfolio() -> list:
    """Load persistent portfolio holdings from JSON file, falling back to default."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    return saved
        except Exception as e:
            logging.error(f"Error loading portfolio from file: {e}")
    return DEFAULT_PORTFOLIO.copy()

def save_persistent_portfolio(portfolio: list):
    """Save portfolio holdings to JSON file so it survives reloads and deployments."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving portfolio to file: {e}")

def calculate_portfolio_performance(portfolio: list) -> dict:
    """
    Computes real-time P&L, invested amount, current value, and target progress.
    """
    if not portfolio:
        return {
            "status": "empty",
            "total_invested": 0.0,
            "total_current_value": 0.0,
            "total_pnl_rs": 0.0,
            "total_pnl_pct": 0.0,
            "today_pnl_rs": 0.0,
            "holdings": []
        }

    holdings_summary = []
    total_invested = 0.0
    total_current_value = 0.0
    today_pnl_rs = 0.0

    for item in portfolio:
        symbol = item.get("symbol")
        buy_price = float(item.get("buy_price", 0.0))
        qty = int(item.get("quantity", 0))
        target_price = float(item.get("target_price", buy_price * 1.20))
        notes = item.get("notes", "")

        fund = get_stock_fundamentals(symbol)
        name = fund.get("Name", STOCK_NAME_MAP.get(symbol, symbol))
        cur_price_val = fund.get("Current Price")
        
        current_price = float(cur_price_val) if isinstance(cur_price_val, (int, float)) else buy_price
        day_change = float(fund.get("Day Change", 0.0)) if isinstance(fund.get("Day Change"), (int, float)) else 0.0
        day_change_pct = float(fund.get("Day Change %", 0.0)) if isinstance(fund.get("Day Change %"), (int, float)) else 0.0

        invested = buy_price * qty
        current_val = current_price * qty
        pnl_rs = current_val - invested
        pnl_pct = (pnl_rs / invested * 100.0) if invested > 0 else 0.0

        today_item_pnl = day_change * qty

        total_invested += invested
        total_current_value += current_val
        today_pnl_rs += today_item_pnl

        # Progress towards Target Sell Price
        if target_price > buy_price:
            target_progress_pct = min(max(((current_price - buy_price) / (target_price - buy_price)) * 100.0, 0.0), 100.0)
        else:
            target_progress_pct = 100.0

        holdings_summary.append({
            "symbol": symbol,
            "name": name,
            "buy_price": buy_price,
            "quantity": qty,
            "target_price": target_price,
            "current_price": current_price,
            "day_change": day_change,
            "day_change_pct": day_change_pct,
            "invested": round(invested, 2),
            "current_value": round(current_val, 2),
            "pnl_rs": round(pnl_rs, 2),
            "pnl_pct": round(pnl_pct, 2),
            "today_pnl_rs": round(today_item_pnl, 2),
            "target_progress_pct": round(target_progress_pct, 1),
            "notes": notes
        })

    total_pnl_rs = total_current_value - total_invested
    total_pnl_pct = (total_pnl_rs / total_invested * 100.0) if total_invested > 0 else 0.0

    return {
        "status": "success",
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current_value, 2),
        "total_pnl_rs": round(total_pnl_rs, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "today_pnl_rs": round(today_pnl_rs, 2),
        "holdings": holdings_summary
    }
