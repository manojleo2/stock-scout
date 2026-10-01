"""
utils/feedback_engine.py
────────────────────────
Offline Model Calibration, Reliability, and Drift Analytics Engine.

Architectural Safeguard:
  This module operates strictly as a READ-ONLY diagnostic monitor.
  It computes Brier scores, reliability tables, and prediction drift metrics.
  It has ZERO write access to live trading probabilities or model execution state.
"""

import logging
import numpy as np
from utils.daily_journal import load_daily_journal

logging.basicConfig(level=logging.INFO)


def compute_calibration_and_drift_diagnostics(symbol: str = "CDSL.NS") -> dict:
    """
    Offline monitoring: Compute empirical Brier score, reliability bins,
    and prediction distribution drift across historical forecasts.
    Does NOT alter live model weights or probabilities.
    """
    journal = load_daily_journal()
    if not journal:
        return {"status": "insufficient_data", "message": "No journal history available."}

    symbol_entries = [e for e in journal if e.get('symbol') == symbol and e.get('is_correct') is not None]
    if len(symbol_entries) < 5:
        return {"status": "insufficient_data", "count": len(symbol_entries), "message": "Need >= 5 completed sessions."}

    # Extract predicted probabilities and actual binary outcomes (1 for UP, 0 for DOWN)
    probs = []
    outcomes = []
    for e in symbol_entries:
        prob = e.get('predicted_prob_up', 50.0) / 100.0 if e.get('predicted_prob_up') else 0.50
        actual = 1 if (e.get('day_change_pct') or 0.0) > 0 else 0
        probs.append(prob)
        outcomes.append(actual)

    probs = np.array(probs)
    outcomes = np.array(outcomes)

    # 1. Rolling Brier Score: Mean squared error of probabilistic predictions
    brier_score = float(np.mean((probs - outcomes) ** 2))

    # 2. Reliability Bins: Compare predicted bucket with observed event frequency
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.00)]
    reliability_report = []

    for low, high in bins:
        mask = (probs >= low) & (probs < high)
        if np.sum(mask) > 0:
            avg_pred = float(np.mean(probs[mask])) * 100.0
            actual_freq = float(np.mean(outcomes[mask])) * 100.0
            reliability_report.append({
                "bucket": f"{int(low*100)}–{int(high*100)}%",
                "sample_count": int(np.sum(mask)),
                "avg_predicted_pct": round(avg_pred, 1),
                "actual_observed_pct": round(actual_freq, 1),
                "calibration_error": round(abs(avg_pred - actual_freq), 1)
            })

    # 3. Overall Empirical Hit Rate
    hit_rate = float(np.mean([1 if e.get('is_correct') else 0 for e in symbol_entries])) * 100.0

    return {
        "status": "success",
        "symbol": symbol,
        "sample_count": len(symbol_entries),
        "brier_score": round(brier_score, 4),
        "overall_hit_rate_pct": round(hit_rate, 1),
        "reliability_bins": reliability_report
    }


def calculate_feedback_recalibration_offset(symbol: str, raw_prob_up: float) -> tuple:
    """
    DIAGNOSTIC ONLY: Retained for historical audit analysis.
    Decoupled from production: NEVER injected into live trading probability vectors.
    """
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
