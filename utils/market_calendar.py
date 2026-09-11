import pandas as pd
import datetime as dt

# Official NSE / BSE Trading Holidays for Calendar Year 2026
NSE_HOLIDAYS_2026 = {
    dt.date(2026, 1, 15): "Municipal Corporation Election - Maharashtra",
    dt.date(2026, 1, 26): "Republic Day",
    dt.date(2026, 2, 15): "Mahashivratri",
    dt.date(2026, 3, 3): "Holi",
    dt.date(2026, 3, 21): "Id-Ul-Fitr",
    dt.date(2026, 3, 26): "Shri Ram Navami",
    dt.date(2026, 3, 31): "Shri Mahavir Jayanti",
    dt.date(2026, 4, 3): "Good Friday",
    dt.date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    dt.date(2026, 5, 1): "Maharashtra Day",
    dt.date(2026, 5, 28): "Bakri Id",
    dt.date(2026, 6, 26): "Muharram",
    dt.date(2026, 8, 15): "Independence Day",
    dt.date(2026, 9, 14): "Ganesh Chaturthi",
    dt.date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    dt.date(2026, 10, 20): "Dussehra",
    dt.date(2026, 11, 8): "Diwali Laxmi Pujan (Muhurat Trading Only)",
    dt.date(2026, 11, 10): "Diwali-Balipratipada",
    dt.date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev Jayanti",
    dt.date(2026, 12, 25): "Christmas",
}

def is_trading_holiday(target_date: dt.date) -> tuple:
    """
    Check if a given date is an official NSE/BSE market holiday.
    Returns (is_holiday: bool, holiday_name: str).
    """
    if isinstance(target_date, (pd.Timestamp, dt.datetime)):
        target_date = target_date.date()
    if target_date in NSE_HOLIDAYS_2026:
        return True, NSE_HOLIDAYS_2026[target_date]
    return False, ""

def get_next_trading_date(start_date: dt.date) -> tuple:
    """
    Compute next valid trading date, skipping Saturdays, Sundays, and all official 2026 NSE market holidays.
    Returns (next_date: dt.date, skipped_holidays: list[dict]).
    """
    if isinstance(start_date, (pd.Timestamp, dt.datetime)):
        start_date = start_date.date()

    next_date = start_date + dt.timedelta(days=1)
    skipped_holidays = []

    while True:
        # 1. Skip weekends (Saturday=5, Sunday=6)
        if next_date.weekday() >= 5:
            next_date += dt.timedelta(days=1)
            continue

        # 2. Skip NSE 2026 market holidays
        is_holiday, holiday_name = is_trading_holiday(next_date)
        if is_holiday:
            skipped_holidays.append({
                "date": next_date,
                "date_str": next_date.strftime("%a, %d %b %Y"),
                "name": holiday_name
            })
            next_date += dt.timedelta(days=1)
            continue

        # Valid trading day found
        break

    return next_date, skipped_holidays

def get_market_dates(df: pd.DataFrame) -> dict:
    """
    Extract the last market trading date from the dataset and compute the next valid trading session date,
    strictly taking into account weekends and the full 2026 NSE/BSE holiday calendar.
    """
    if df.empty:
        today = dt.date.today()
        next_date, skipped_holidays = get_next_trading_date(today)
        last_date = today
    else:
        last_timestamp = df.index[-1]
        last_date = last_timestamp.date() if isinstance(last_timestamp, (pd.Timestamp, dt.datetime)) else last_timestamp
        next_date, skipped_holidays = get_next_trading_date(last_date)

    is_today_holiday, today_holiday_name = is_trading_holiday(dt.date.today())

    # Format holiday alert if any upcoming holidays were skipped
    holiday_alert = None
    if skipped_holidays:
        holiday_names_str = ", ".join([f"{h['date_str']} ({h['name']})" for h in skipped_holidays])
        holiday_alert = f"Market is CLOSED on {holiday_names_str}. Next active trading session opens on {next_date.strftime('%a, %d %b %Y')}."

    return {
        "last_date_str": last_date.strftime("%a, %d %b %Y"),
        "next_date_str": next_date.strftime("%a, %d %b %Y"),
        "last_date": last_date,
        "next_date": next_date,
        "is_today_holiday": is_today_holiday,
        "today_holiday_name": today_holiday_name,
        "skipped_holidays": skipped_holidays,
        "holiday_alert": holiday_alert
    }

def get_daily_ups_downs_history(df: pd.DataFrame, max_days: int = 30) -> pd.DataFrame:
    """
    Construct a clean date-wise history table showing daily price movements (UP/DOWN) and percentage change.
    """
    if df.empty or len(df) == 0:
        return pd.DataFrame()

    data = df.tail(max_days).copy()
    
    data['Prev_Close'] = data['Close'].shift(1)
    data['Change_Rs'] = data['Close'] - data['Prev_Close']
    data['Change_Pct'] = (data['Change_Rs'] / data['Prev_Close']) * 100

    history_rows = []
    for i in range(len(data) - 1, -1, -1):
        row = data.iloc[i]
        date_idx = data.index[i]
        
        date_str = date_idx.strftime("%d %b %Y (%a)") if isinstance(date_idx, (pd.Timestamp, dt.datetime)) else str(date_idx)
        close_price = round(row['Close'], 2)
        chg_rs = row.get('Change_Rs')
        chg_pct = row.get('Change_Pct')

        if pd.isna(chg_rs) or pd.isna(chg_pct):
            movement = "⚪ UNCHANGED"
            chg_str = "₹0.00 (0.00%)"
        elif chg_rs > 0:
            movement = "🟢 UP"
            chg_str = f"+₹{chg_rs:.2f} (+{chg_pct:.2f}%)"
        elif chg_rs < 0:
            movement = "🔴 DOWN"
            chg_str = f"-₹{abs(chg_rs):.2f} ({chg_pct:.2f}%)"
        else:
            movement = "⚪ UNCHANGED"
            chg_str = "₹0.00 (0.00%)"

        vol = int(row['Volume']) if pd.notna(row['Volume']) else 0

        history_rows.append({
            "Date": date_str,
            "Movement": movement,
            "Close Price (₹)": f"₹{close_price:,.2f}",
            "Daily Change": chg_str,
            "High (₹)": f"₹{row['High']:,.2f}",
            "Low (₹)": f"₹{row['Low']:,.2f}",
            "Volume": f"{vol:,}"
        })

    return pd.DataFrame(history_rows)
