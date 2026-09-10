import os
import sys
import datetime as dt
import logging

logging.basicConfig(level=logging.INFO)

# Add repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.portfolio_manager import load_saved_portfolio
from utils.ml_model import train_and_predict
from utils.opening_predictor import predict_opening_gap
from utils.market_calendar import get_market_dates
from utils.data_loader import get_stock_data
from utils.notifications import (
    send_prediction_alert_notification,
    send_gap_alert_notification,
    send_opening_gap_alert_notification,
    send_telegram_alert
)
from utils.prediction_audit import evaluate_and_update_audit_outcomes, record_prediction
from utils.opening_audit import record_opening_gap_prediction, evaluate_opening_gap_outcomes
from config import STOCK_NAME_MAP

def run_pre_market_cron():
    logging.info('Running 8:45 AM IST Pre-Market Cron Workflow...')
    portfolio = load_saved_portfolio()
    symbols = list(set([item.get('symbol') for item in portfolio] + ['CDSL.NS', 'NSDL.BO']))

    for sym in symbols:
        try:
            df_raw = get_stock_data(sym, period='1y')
            if df_raw.empty:
                continue
            dates_info = get_market_dates(df_raw)
            result = train_and_predict(sym, period='2y')
            if result.get('status') == 'success':
                name = STOCK_NAME_MAP.get(sym, sym)
                record_prediction(sym, dates_info['next_date_str'], result)
                
                news_score = result.get('news_info', {}).get('score', 0.0)
                trading_call = result.get('trading_call')
                success, err = send_prediction_alert_notification(
                    sym, name, result['direction'],
                    result['probability_up_pct'], result['confidence'],
                    news_score, dates_info['next_date_str'],
                    trading_call=trading_call
                )
                logging.info(f'Pre-market alert for {sym}: success={success}, err={err}')
        except Exception as e:
            logging.error(f'Pre-market cron error for {sym}: {e}')

def run_intraday_cron():
    logging.info('Running 9:30 AM IST Intraday Breakout Cron Workflow...')
    portfolio = load_saved_portfolio()
    symbols = list(set([item.get('symbol') for item in portfolio] + ['CDSL.NS']))

    for sym in symbols:
        try:
            df_stock = get_stock_data(sym, period='5d')
            if not df_stock.empty and len(df_stock) >= 2:
                latest = df_stock.iloc[-1]
                prev = df_stock.iloc[-2]
                name = STOCK_NAME_MAP.get(sym, sym)
                price = float(latest['Close'])
                vwap = float(latest['Close'] * 0.995)
                chg = float(((price - prev['Close']) / prev['Close']) * 100.0)
                if chg >= 0.5:
                    send_gap_alert_notification(sym, name, '🟢 GAP CONTINUATION (BULLISH)', price, vwap, round(price * 1.025, 2))
                    logging.info(f'Intraday alert sent for {sym}')
        except Exception as e:
            logging.error(f'Intraday cron error for {sym}: {e}')

def run_opening_gap_cron():
    logging.info('Running 3:05 PM IST Opening Gap Cron Workflow...')
    portfolio = load_saved_portfolio()
    symbols = list(set([item.get('symbol') for item in portfolio] + ['CDSL.NS', 'NSDL.BO']))

    for sym in symbols:
        try:
            df_raw = get_stock_data(sym, period='1y')
            if df_raw.empty:
                continue
            dates_info = get_market_dates(df_raw)
            result = predict_opening_gap(sym, period='2y')
            if result.get('status') == 'success':
                name = STOCK_NAME_MAP.get(sym, sym)
                record_opening_gap_prediction(sym, dates_info['next_date_str'], result)

                success, err = send_opening_gap_alert_notification(
                    symbol=sym,
                    name=name,
                    gap_direction=result['direction'],
                    prob_up=result['probability_up_pct'],
                    confidence=result['confidence'],
                    options_call=result.get('options_call', {}),
                    target_date=dates_info['next_date_str'],
                    current_price=result.get('current_price', 0.0)
                )
                logging.info(f'Opening gap alert for {sym}: success={success}, err={err}')
        except Exception as e:
            logging.error(f'Opening gap cron error for {sym}: {e}')

def run_post_market_cron():
    logging.info('Running 3:45 PM IST Post-Market Audit Cron Workflow...')
    eval_list = evaluate_and_update_audit_outcomes()
    # Also evaluate opening gap outcomes for completed sessions
    try:
        evaluate_opening_gap_outcomes()
    except Exception as e_gap:
        logging.warning(f'Opening gap audit eval error: {e_gap}')

    today_str = dt.datetime.now().strftime('%a, %d %b %Y')
    
    msg = f"🌆 *Stock Scout Post-Market Daily Audit Summary*\n📅 *Date:* {today_str}\n\n"
    if eval_list:
        for item in eval_list:
            sym = item.get('symbol')
            pred = item.get('predicted_direction')
            act = item.get('actual_direction')
            status = '✅ ACCURATE' if item.get('is_correct') else '❌ DIVERGED'
            msg += f"• *{sym}:* Pred {pred} | Actual {act} ({status})\n"
    else:
        msg += "No completed session forecasts to evaluate today."
    
    msg += "\n👉 Inspect full journal: https://stock-scout-mn.streamlit.app"
    send_telegram_alert(msg)

if __name__ == '__main__':
    raw_mode = sys.argv[1] if len(sys.argv) > 1 else 'auto'
    if raw_mode in ('auto', '', 'default'):
        now_utc = dt.datetime.now(dt.timezone.utc)
        hour = now_utc.hour
        minute = now_utc.minute
        if 2 <= hour <= 3:
            mode = 'pre_market'
        elif 4 <= hour <= 6:
            mode = 'intraday'
        elif hour == 9 and minute >= 25:
            # 9:25-9:59 UTC = 2:55 PM - 3:29 PM IST (Opening Gap window)
            mode = 'opening_gap'
        elif 10 <= hour <= 12:
            mode = 'post_market'
        else:
            mode = 'pre_market'
    else:
        mode = raw_mode

    logging.info(f"Executing cron runner in '{mode}' mode.")
    if mode == 'pre_market':
        run_pre_market_cron()
    elif mode == 'intraday':
        run_intraday_cron()
    elif mode == 'opening_gap':
        run_opening_gap_cron()
    elif mode == 'post_market':
        run_post_market_cron()

