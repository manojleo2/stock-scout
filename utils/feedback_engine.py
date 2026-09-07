import logging
import numpy as np
from utils.daily_journal import load_daily_journal

logging.basicConfig(level=logging.INFO)

def calculate_feedback_recalibration_offset(symbol: str, raw_prob_up: float) -> tuple:
    journal = load_daily_journal()
    if not journal:
        return 0.0, 'Neutral baseline (No past journal memory)'

    symbol_entries = [e for e in journal if e.get('symbol') == symbol and e.get('is_correct') is not None]
    if not symbol_entries:
        return 0.0, 'Neutral baseline (First forecast for this stock)'

    recent_entries = symbol_entries[-10:]
    diverged = [e for e in recent_entries if not e.get('is_correct')]

    if not diverged:
        return 0.0, '✅ 100% Recent Accuracy: High confidence baseline.'

    bullish_misses = sum(1 for e in diverged if 'DOWN' in e.get('predicted_direction', '') and e.get('day_change_pct', 0) > 0)
    bearish_misses = sum(1 for e in diverged if 'UP' in e.get('predicted_direction', '') and e.get('day_change_pct', 0) < 0)

    net_bias = bullish_misses - bearish_misses
    feedback_offset = float(np.clip(net_bias * 4.5, -12.0, 12.0))

    if feedback_offset > 0:
        reason = f'🟢 **Historical Bullish Feedback (+{feedback_offset:.1f}%)**: AI previously understated bullish support defense in {bullish_misses} session(s).'
    elif feedback_offset < 0:
        reason = f'🔴 **Historical Bearish Feedback ({feedback_offset:.1f}%)**: AI previously understated distribution selling in {bearish_misses} session(s).'
    else:
        reason = '⚪ **Balanced Feedback**: Equal bullish/bearish error balance.'

    return feedback_offset, reason

def generate_actionable_trading_call(symbol: str, latest_close: float, recalibrated_prob_up: float, pred_result: dict) -> dict:
    close = float(latest_close)
    prob_up = float(recalibrated_prob_up)

    if prob_up >= 65.0:
        call_signal = '🚀 STRONG BUY'
        badge = '🟢 BULLISH CONTINUATION'
        entry_min = round(close * 0.995, 2)
        entry_max = round(close * 1.005, 2)
        target_1 = round(close * 1.030, 2)
        target_2 = round(close * 1.055, 2)
        stop_loss = round(close * 0.980, 2)
        rr_ratio = '1 : 2.5'
        strategy_note = 'High bullish probability supported by technical momentum & positive feedback bias.'
    elif prob_up >= 55.0:
        call_signal = '🟢 MILD BUY'
        badge = '🟢 MODERATE BULLISH'
        entry_min = round(close * 0.992, 2)
        entry_max = round(close * 1.002, 2)
        target_1 = round(close * 1.020, 2)
        target_2 = round(close * 1.040, 2)
        stop_loss = round(close * 0.985, 2)
        rr_ratio = '1 : 2.0'
        strategy_note = 'Moderate upward tilt. Accumulate on minor intraday dips near VWAP.'
    elif prob_up >= 45.0:
        call_signal = '⚪ HOLD / NEUTRAL'
        badge = '⚪ SIDEWAYS / CONSOLIDATION'
        entry_min = round(close * 0.990, 2)
        entry_max = round(close * 1.000, 2)
        target_1 = round(close * 1.015, 2)
        target_2 = round(close * 1.030, 2)
        stop_loss = round(close * 0.988, 2)
        rr_ratio = '1 : 1.5'
        strategy_note = 'Neutral equilibrium. Wait for 9:30 AM Opening Range Breakout confirmation.'
    else:
        call_signal = '🔴 BEARISH / CAUTION'
        badge = '🔴 DOWNWARD REGIME'
        entry_min = 'N/A (Avoid Long Entry)'
        entry_max = 'N/A'
        target_1 = round(close * 0.970, 2)
        target_2 = round(close * 0.950, 2)
        stop_loss = round(close * 1.020, 2)
        rr_ratio = '1 : 2.2'
        strategy_note = 'Bearish bias dominant. Protect capital or wait for lower support retest.'

    return {
        'symbol': symbol,
        'latest_close': close,
        'recalibrated_prob_up': round(prob_up, 1),
        'call_signal': call_signal,
        'badge': badge,
        'entry_zone': f'₹{entry_min:,.2f} - ₹{entry_max:,.2f}' if isinstance(entry_min, (int, float)) else str(entry_min),
        'target_1': target_1,
        'target_2': target_2,
        'stop_loss': stop_loss,
        'risk_reward_ratio': rr_ratio,
        'strategy_note': strategy_note
    }
