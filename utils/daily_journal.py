import os
import json
import logging
import pandas as pd
import datetime as dt
from utils.data_loader import get_stock_data

logging.basicConfig(level=logging.INFO)

JOURNAL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'daily_market_journal.json')

def load_daily_journal() -> list:
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                if isinstance(saved, list):
                    return saved
        except Exception as e:
            logging.error(f'Error loading daily journal: {e}')
    return []

def save_daily_journal(journal_list: list):
    try:
        with open(JOURNAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(journal_list, f, indent=2)
    except Exception as e:
        logging.error(f'Error saving daily journal: {e}')

def create_daily_session_snapshot(symbol: str, target_date_str: str, pred_direction: str, prob_up: float) -> dict:
    snapshot = {
        'symbol': symbol,
        'date': target_date_str,
        'prediction_time': dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'predicted_direction': pred_direction,
        'predicted_prob_up': float(prob_up),
        'open': None, 'high': None, 'low': None, 'close': None, 'prev_close': None,
        'day_change_pct': None, 'volume': None, 'vol_vs_10d_sma': None,
        'upper_wick_pct': None, 'lower_wick_pct': None, 'body_pct': None,
        'actual_direction': '⏳ Pending Session Close',
        'is_correct': None, 'divergence_tags': [], 'post_mortem_narrative': '⏳ Session pending completion.'
    }

    try:
        df_stock = get_stock_data(symbol, period='1mo')
        if not df_stock.empty and len(df_stock) >= 2:
            try:
                target_date = dt.datetime.strptime(target_date_str, "%a, %d %b %Y").date()
            except Exception:
                try:
                    target_date = dt.datetime.strptime(target_date_str, "%Y-%m-%d").date()
                except Exception:
                    target_date = dt.date.today()

            df_stock_dates = pd.to_datetime(df_stock.index).date
            matching_indices = [idx for idx, d in enumerate(df_stock_dates) if d == target_date]
            target_idx = matching_indices[-1] if matching_indices else -1

            latest = df_stock.iloc[target_idx]
            prev = df_stock.iloc[target_idx - 1] if (target_idx > 0 or (target_idx == -1 and len(df_stock) >= 2)) else latest

            o = float(latest['Open'])
            h = float(latest['High'])
            l = float(latest['Low'])
            c = float(latest['Close'])
            prev_c = float(prev['Close'])
            vol = int(latest['Volume'])

            vol_10d_avg = float(df_stock['Volume'].iloc[-10:].mean()) if len(df_stock) >= 10 else float(vol)
            vol_multiplier = round(vol / vol_10d_avg, 2) if vol_10d_avg > 0 else 1.0

            total_range = h - l
            if total_range > 0:
                body = abs(c - o)
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l

                body_pct = round((body / total_range) * 100.0, 1)
                upper_wick_pct = round((upper_wick / total_range) * 100.0, 1)
                lower_wick_pct = round((lower_wick / total_range) * 100.0, 1)
            else:
                body_pct, upper_wick_pct, lower_wick_pct = 100.0, 0.0, 0.0

            day_chg_pct = round(((c - prev_c) / prev_c) * 100.0, 2)
            act_dir = 'UP 📈' if day_chg_pct > 0 else ('DOWN 📉' if day_chg_pct < 0 else 'FLAT ⚪')
            is_corr = bool(('UP' in pred_direction and day_chg_pct > 0) or ('DOWN' in pred_direction and day_chg_pct < 0))

            snapshot.update({
                'open': round(o, 2),
                'high': round(h, 2),
                'low': round(l, 2),
                'close': round(c, 2),
                'prev_close': round(prev_c, 2),
                'day_change_pct': day_chg_pct,
                'volume': vol,
                'vol_vs_10d_sma': vol_multiplier,
                'upper_wick_pct': upper_wick_pct,
                'lower_wick_pct': lower_wick_pct,
                'body_pct': body_pct,
                'actual_direction': act_dir,
                'is_correct': is_corr
            })

            tags = []
            if vol_multiplier >= 1.5:
                tags.append('VOLUME_ACCUMULATION' if day_chg_pct >= 0 else 'HEAVY_SELLING_SURGE')
            if lower_wick_pct >= 35.0:
                tags.append('SUPPORT_DEFENSE')
            if upper_wick_pct >= 35.0:
                tags.append('PROFIT_BOOKING_REJECTION')
            if not is_corr:
                tags.append('AI_MODEL_DIVERGENCE')
            else:
                tags.append('PREDICTION_VERIFIED')

            snapshot['divergence_tags'] = tags

            if is_corr:
                narrative = f'✅ **Prediction Verified**: Stock moved {day_chg_pct:+.2f}% as predicted by AI ensemble model.'
            else:
                if 'UP' in pred_direction and day_chg_pct < 0:
                    narrative = f'🔴 **Bearish Divergence**: Predicted UP, but stock fell {day_chg_pct:.2f}% (Closed at ₹{c:,.2f} vs ₹{prev_c:,.2f}). '
                    if upper_wick_pct >= 35.0:
                        narrative += f'Intraday rejection created an upper wick ({upper_wick_pct}% of candle range).'
                    elif vol_multiplier >= 1.5:
                        narrative += f'Volume was {vol_multiplier}x above average, indicating distribution.'
                    else:
                        narrative += 'Intraday profit booking outweighed pre-market bullish signals.'
                else:
                    narrative = f'🟢 **Bullish Divergence**: Predicted DOWN, but stock gained +{day_chg_pct:.2f}% (Closed at ₹{c:,.2f} vs ₹{prev_c:,.2f}). '
                    if lower_wick_pct >= 35.0:
                        narrative += f'Buyers aggressively defended support, creating a lower wick ({lower_wick_pct}% of candle range).'
                    if vol_multiplier >= 1.5:
                        narrative += f' Volume spiked {vol_multiplier}x higher than 10-day average.'

            snapshot['post_mortem_narrative'] = narrative

    except Exception as e:
        logging.error(f'Error creating daily session snapshot: {e}')

    return snapshot

def update_or_append_journal_entry(snapshot: dict):
    journal = load_daily_journal()
    symbol = snapshot.get('symbol')
    date_str = snapshot.get('date')

    existing = next((item for item in journal if item.get('symbol') == symbol and item.get('date') == date_str), None)
    if existing:
        existing.update(snapshot)
    else:
        journal.append(snapshot)

    save_daily_journal(journal)
    return journal
