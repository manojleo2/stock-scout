import pandas as pd
import datetime as dt

def get_market_dates(df: pd.DataFrame) -> dict:
    """
    Extract the last market trading date from the dataset and compute the next valid trading session date.
    Automatically accounts for weekends (Saturday/Sunday).
    """
    if df.empty:
        today = dt.date.today()
        return {
            "last_date_str": today.strftime("%A, %d %b %Y"),
            "next_date_str": (today + dt.timedelta(days=1)).strftime("%A, %d %b %Y"),
            "last_date": today,
            "next_date": today + dt.timedelta(days=1)
        }

    # Extract last timestamp from DataFrame index
    last_timestamp = df.index[-1]
    last_date = last_timestamp.date() if isinstance(last_timestamp, (pd.Timestamp, dt.datetime)) else last_timestamp

    # Compute next valid trading date (skipping Saturday=5 and Sunday=6)
    next_date = last_date + dt.timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += dt.timedelta(days=1)

    return {
        "last_date_str": last_date.strftime("%A, %d %b %Y"),
        "next_date_str": next_date.strftime("%A, %d %b %Y"),
        "last_date": last_date,
        "next_date": next_date
    }

def get_daily_ups_downs_history(df: pd.DataFrame, max_days: int = 30) -> pd.DataFrame:
    """
    Construct a clean date-wise history table showing daily price movements (UP/DOWN) and percentage change.
    """
    if df.empty or len(df) == 0:
        return pd.DataFrame()

    data = df.tail(max_days).copy()
    
    # Calculate daily price differences
    data['Prev_Close'] = data['Close'].shift(1)
    data['Change_Rs'] = data['Close'] - data['Prev_Close']
    data['Change_Pct'] = (data['Change_Rs'] / data['Prev_Close']) * 100

    history_rows = []
    # Loop from latest date backwards
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
            "Trading Volume": f"{vol:,}"
        })

    return pd.DataFrame(history_rows)
