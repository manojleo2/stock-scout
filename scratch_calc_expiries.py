import calendar, datetime as dt
import sys
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_last_thursday(year, month):
    cal = calendar.monthcalendar(year, month)
    thursdays = [week[3] for week in cal if week[3] != 0]
    return dt.date(year, month, thursdays[-1])

def get_active_expiry(trade_date):
    # If trade_date <= last thursday of that month, it's that month's expiry
    # If trade_date > last thursday of that month, it rolls to next month's expiry
    m_exp = get_last_thursday(trade_date.year, trade_date.month)
    if trade_date <= m_exp:
        return m_exp
    else:
        next_month = trade_date.month + 1 if trade_date.month < 12 else 1
        next_year = trade_date.year if trade_date.month < 12 else trade_date.year + 1
        return get_last_thursday(next_year, next_month)

df = pd.read_csv('backtest_results/cdsl_intraday_2lots_weekly.csv')
active = df[df['model_action'] != 'NO_TRADE'].copy().reset_index(drop=True)

active['trade_date_obj'] = pd.to_datetime(active['date']).dt.date
active['expiry_date'] = active['trade_date_obj'].apply(get_active_expiry)
active['days_to_expiry'] = (pd.to_datetime(active['expiry_date']) - pd.to_datetime(active['trade_date_obj'])).dt.days

for idx, r in active.iterrows():
    exp_str = r['expiry_date'].strftime('%d-%b-%Y')
    strike_type = r['contract'].split()[-1]
    strike_val = r['contract'].split()[-2]
    trading_symbol = f"CDSL {r['expiry_date'].strftime('%d%b%y').upper()} {strike_val} {strike_type}"
    print(f"{idx+1:02d} | Trade: {r['date']} | Contract: {trading_symbol:<24} | Expiry: {exp_str} ({r['days_to_expiry']:>2} DTE) | Outcome: {r['hit_status']} | PnL: Rs {r['net_pnl']:+,.0f}")
