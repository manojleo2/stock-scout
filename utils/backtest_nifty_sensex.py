"""
utils/backtest_nifty_sensex.py
───────────────────────────────
Walk-Forward Backtester for Nifty & Sensex Intraday Predictor.

Fix #1: Expiry-aware P&L — uses correct lot size & expiry regime per date.
Fix #2: Features strictly T-1 (from nifty_features.py — pre-shifted).
Fix #3: Group B (OI/PCR) NOT used in backtest — no synthetic fill.
Fix #4: Dynamic target/SL as % of estimated entry premium (BSM).

Walk-Forward Design:
  - Warmup: 300 days
  - Each step: train on all history, predict next day's intraday direction
  - Target: close > open + 0.30% (UP=1), close < open - 0.30% (DOWN=0), chop=skip
  - P&L: estimated from ATR-derived BSM premium at market open
  - Signal filter: conviction >= regime-adjusted threshold
"""

import os
import sys
import logging
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nifty_features import (
    build_feature_matrix, ALL_ML_FEATURES,
    get_days_to_nifty_expiry, get_days_to_sensex_expiry,
    is_nifty_expiry_day,
)
from utils.nifty_regime import classify_regime, get_conviction_threshold
from utils.nifty_sensex_model import estimate_atm_premium
from config import (
    NIFTY_TICKER, SENSEX_TICKER,
    NIFTY_LOT_SIZE, SENSEX_LOT_SIZE,
    NIFTY_STRIKE_STEP, SENSEX_STRIKE_STEP,
    ML_N_ESTIMATORS, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF, ML_WARMUP_DAYS,
    NIFTY_MIN_CONVICTION_WEEKDAY, NIFTY_MIN_CONVICTION_FRIDAY,
    NIFTY_MIN_CONVICTION_EXPIRY_DAY, NIFTY_MIN_CONVICTION_CHOP,
    NIFTY_MIN_CONVICTION_VOLATILE,
    NIFTY_TARGET_PCT, NIFTY_SL_PCT,
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logger = logging.getLogger("backtest_nifty")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quick_ensemble_prob(X_train, y_train, X_test_row, sw) -> float:
    """Train RF+HGB mini-ensemble and return probability for one test row."""
    rf = RandomForestClassifier(
        n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH,
        min_samples_leaf=ML_MIN_SAMPLES_LEAF, random_state=42, n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=100, max_depth=3,
        min_samples_leaf=ML_MIN_SAMPLES_LEAF, random_state=42,
    )
    rf.fit(X_train,  y_train, sample_weight=sw)
    hgb.fit(X_train, y_train, sample_weight=sw)
    p_rf  = float(rf.predict_proba(X_test_row)[0][1])
    p_hgb = float(hgb.predict_proba(X_test_row)[0][1])

    if HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            min_child_samples=ML_MIN_SAMPLES_LEAF, random_state=42,
            n_jobs=-1, verbosity=-1,
        )
        lgbm.fit(X_train, y_train, sample_weight=sw)
        p_lgbm = float(lgbm.predict_proba(X_test_row)[0][1])
        return (p_rf + p_hgb + p_lgbm) / 3.0
    return (p_rf + p_hgb) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward_backtest(
    ticker: str = NIFTY_TICKER,
    period: str = "5y",
    min_train_days: int = ML_WARMUP_DAYS,
) -> pd.DataFrame:
    """
    Run walk-forward backtest. Returns session-level results DataFrame.

    For each OOS session i (from min_train_days to end):
      - Features: rows 0..i-1 (no lookahead — all T-1 shifted in nifty_features.py)
      - Target:   row i intraday direction (close vs open)
      - Model:    RF + HGB (+LGBM if available)
      - Filter:   conviction >= regime-aware threshold
      - P&L:      based on ATR-derived BSM entry premium, dynamic target/SL
    """
    print(f"\n{'='*70}")
    print(f"  Nifty/Sensex Walk-Forward Backtest — {ticker}")
    print(f"  Period: {period} | Warmup: {min_train_days} days")
    print(f"  Fix #3: Group B (OI/PCR) excluded from training")
    print(f"  Fix #4: Dynamic Target +{NIFTY_TARGET_PCT}% / SL -{NIFTY_SL_PCT}% of premium")
    print(f"{'='*70}\n")

    # Build feature matrix (all T-1 shifted for Groups A+D — Fix #2)
    print("  Building feature matrix (Groups A+C+D+E)...")
    df = build_feature_matrix(ticker=ticker, period=period)
    clean = df.dropna(subset=ALL_ML_FEATURES + ['Target_Intraday']).copy()

    if len(clean) < min_train_days + 20:
        raise ValueError(f"Only {len(clean)} clean rows — need {min_train_days + 20}+")

    lot_size    = NIFTY_LOT_SIZE if ticker == NIFTY_TICKER else SENSEX_LOT_SIZE
    strike_step = NIFTY_STRIKE_STEP if ticker == NIFTY_TICKER else SENSEX_STRIKE_STEP
    total_steps = len(clean) - min_train_days

    print(f"  Clean rows: {len(clean)} | OOS sessions: {total_steps}")
    print(f"  Running walk-forward (this may take a few minutes)...\n")

    results = []

    for i in range(min_train_days, len(clean)):
        date       = clean.index[i]
        ts         = pd.Timestamp(date)
        row        = clean.iloc[i]
        train_data = clean.iloc[:i]

        # Target (today's intraday direction)
        target = int(row['Target_Intraday'])

        # Market data for this session
        close   = float(row.get('Close', 0.0))
        open_p  = float(df.loc[date, 'Open']) if date in df.index else close
        high_p  = float(df.loc[date, 'High']) if date in df.index else close
        low_p   = float(df.loc[date, 'Low'])  if date in df.index else close
        atr     = float(row.get('ATR_14', close * 0.015))
        weekday = ts.weekday()
        is_fri  = (weekday == 4)
        if ticker == SENSEX_TICKER:
            is_exp  = (weekday == 3)  # Sensex expiry is Thursday
            days_ex = int(row.get('E_Days_Sensex_Expiry', 7))
        else:
            is_exp  = is_nifty_expiry_day(ts)
            days_ex = int(row.get('E_Days_Nifty_Expiry', 7))

        X_train = train_data[ALL_ML_FEATURES]
        y_train = train_data['Target_Intraday'].astype(int)
        X_test  = pd.DataFrame([row[ALL_ML_FEATURES]])

        # Sample weights (exponential recency)
        sw = np.exp(np.linspace(-0.5, 0.0, len(X_train)))

        try:
            prob_up = _quick_ensemble_prob(X_train, y_train, X_test, sw)
        except Exception as e:
            logger.warning(f"Step {i} model failed: {e}")
            continue

        prob_up    = float(np.clip(prob_up, 0.05, 0.95))
        conviction = max(prob_up, 1 - prob_up) * 100.0
        direction  = "UP" if prob_up >= 0.50 else "DOWN"

        # Regime classification (using last 20 rows of training data)
        regime_window = df.loc[:date].tail(21)
        regime, _, _  = classify_regime(regime_window, lookback=20)

        threshold = get_conviction_threshold(
            regime=regime,
            is_friday=is_fri,
            is_expiry_day=is_exp,
            weekday_threshold=NIFTY_MIN_CONVICTION_WEEKDAY,
            friday_threshold=NIFTY_MIN_CONVICTION_FRIDAY,
            expiry_threshold=NIFTY_MIN_CONVICTION_EXPIRY_DAY,
            chop_threshold=NIFTY_MIN_CONVICTION_CHOP,
            volatile_threshold=NIFTY_MIN_CONVICTION_VOLATILE,
        )
        take_trade = conviction >= threshold

        # Conservative Intraday Wick-Aware P&L Simulation
        is_call = (direction == "UP")
        entry_prem = estimate_atm_premium(
            spot=open_p, days_to_expiry=days_ex,
            atr=atr, strike_step=strike_step, is_call=is_call,
        ) if take_trade else 0.0

        delta = 0.50
        pnl = 0.0
        hit = None
        exit_reason = "NO_TRADE" if not take_trade else "PENDING"

        if take_trade and entry_prem > 0:
            target_pts_needed = (entry_prem * (NIFTY_TARGET_PCT / 100.0)) / delta
            sl_pts_needed     = (entry_prem * (NIFTY_SL_PCT / 100.0))     / delta

            if direction == "UP":
                # CALL: High gives favorable move, Low gives adverse wick
                fav_move = max(0.0, high_p - open_p)
                adv_move = max(0.0, open_p - low_p)
            else:
                # PUT: Low gives favorable move, High gives adverse wick
                fav_move = max(0.0, open_p - low_p)
                adv_move = max(0.0, high_p - open_p)

            breached_target = fav_move >= target_pts_needed
            breached_sl     = adv_move >= sl_pts_needed

            is_favorable_close = (close > open_p) if direction == "UP" else (open_p > close)

            if breached_sl and not breached_target:
                hit = False
                exit_reason = "STOP_LOSS_WICK"
                loss_prem = entry_prem * (NIFTY_SL_PCT / 100.0)
                pnl = -loss_prem * lot_size
            elif breached_target and not breached_sl:
                hit = True
                exit_reason = "TARGET_HIT"
                gain_prem = entry_prem * (NIFTY_TARGET_PCT / 100.0)
                pnl = gain_prem * lot_size
            elif breached_target and breached_sl:
                # Both breached: Conservative pessimistic weighting
                if not is_favorable_close:
                    # Closed against trade direction -> 100% SL hit
                    hit = False
                    exit_reason = "DOUBLE_TOUCH_SL"
                    loss_prem = entry_prem * (NIFTY_SL_PCT / 100.0)
                    pnl = -loss_prem * lot_size
                else:
                    # Closed in trade direction: must show strong dominance (fav >= 2.0 * adv)
                    if fav_move >= 2.0 * adv_move:
                        hit = True
                        exit_reason = "TARGET_HIT_DOMINANT"
                        gain_prem = entry_prem * (NIFTY_TARGET_PCT / 100.0)
                        pnl = gain_prem * lot_size
                    else:
                        hit = False
                        exit_reason = "DOUBLE_TOUCH_SL"
                        loss_prem = entry_prem * (NIFTY_SL_PCT / 100.0)
                        pnl = -loss_prem * lot_size
            else:
                # Neither touched: Marked to market at EOD close
                close_move = (close - open_p) if direction == "UP" else (open_p - close)
                opt_chg    = delta * close_move
                net_prem   = max(0.05 * entry_prem, entry_prem + opt_chg)
                pnl        = (net_prem - entry_prem) * lot_size
                hit        = pnl > 0
                exit_reason = "EOD_CLOSE_MTM"

        results.append({
            "date":         date,
            "day_name":     ts.strftime("%A"),
            "weekday":      weekday,
            "is_friday":    is_fri,
            "is_expiry":    is_exp,
            "days_to_exp":  days_ex,
            "regime":       regime,
            "close":        round(close, 2),
            "open":         round(open_p, 2),
            "target":       target,
            "direction":    direction,
            "prob_up":      round(prob_up, 4),
            "conviction":   round(conviction, 2),
            "threshold":    round(threshold, 1),
            "take_trade":   take_trade,
            "hit":          hit,
            "exit_reason":  exit_reason,
            "entry_prem":   round(entry_prem, 2),
            "pnl":          round(pnl, 2),
        })

        # Progress
        if (i - min_train_days) % 50 == 0:
            done = i - min_train_days
            print(f"  [{done}/{total_steps}] {pd.Timestamp(date).date()} "
                  f"| regime={regime} | conv={conviction:.1f}% "
                  f"| trade={'YES' if take_trade else 'NO'}"
                  f"{'  HIT' if hit is True else ('  MISS' if hit is False else '')}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Scorecard Printer
# ─────────────────────────────────────────────────────────────────────────────

def print_scorecard(df: pd.DataFrame, ticker: str = NIFTY_TICKER):
    """Print comprehensive backtest scorecard."""
    lot = NIFTY_LOT_SIZE if ticker == NIFTY_TICKER else SENSEX_LOT_SIZE
    total = len(df)
    trades = df[df['take_trade'] == True]
    n_trades = len(trades)
    n_skip   = total - n_trades

    if n_trades == 0:
        print("No trades taken.")
        return

    hit_df     = trades.dropna(subset=['hit'])
    accuracy   = hit_df['hit'].mean() * 100 if len(hit_df) > 0 else 0.0
    total_pnl  = trades['pnl'].sum()
    wins       = trades[trades['pnl'] > 0]
    losses     = trades[trades['pnl'] < 0]
    win_rate   = len(wins) / n_trades * 100
    pf         = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else 99.0

    print(f"\n{'='*72}")
    print(f"  NIFTY/SENSEX INTRADAY PREDICTOR BACKTEST SCORECARD — {ticker}")
    print(f"{'='*72}")
    print(f"  Total OOS Sessions    : {total}")
    print(f"  Trades Taken          : {n_trades} ({n_trades/total*100:.1f}%)")
    print(f"  Sessions Skipped      : {n_skip} (conviction filter / chop)")
    print(f"  Directional Accuracy  : {accuracy:.1f}%")
    print(f"  Win Rate (by P&L)     : {win_rate:.1f}%")
    print(f"  Net P&L               : ₹{total_pnl:,.0f}")
    print(f"  Profit Factor         : {pf:.2f}")
    print(f"  Avg P&L / Trade       : ₹{trades['pnl'].mean():,.0f}")
    print(f"  Max Win (₹)           : ₹{trades['pnl'].max():,.0f}")
    print(f"  Max Loss (₹)          : ₹{trades['pnl'].min():,.0f}")
    print(f"  Lot Size              : {lot}")
    print(f"{'─'*72}")

    print(f"\n  INTRADAY EXIT BREAKDOWN (Conservative Wick Check):")
    for reason in ['TARGET_HIT', 'TARGET_HIT_DOMINANT', 'STOP_LOSS_WICK', 'DOUBLE_TOUCH_SL', 'EOD_CLOSE_MTM']:
        r_df = trades[trades['exit_reason'] == reason]
        if len(r_df) == 0: continue
        r_pnl = r_df['pnl'].sum()
        print(f"  {reason:<22s}: {len(r_df):>4d} trades ({len(r_df)/n_trades*100:.1f}%) | Total P&L=₹{r_pnl:,.0f}")
    print(f"{'─'*72}")

    print(f"\n  BY REGIME:")
    for rg in ['TRENDING', 'CHOP', 'VOLATILE']:
        rg_df = trades[trades['regime'] == rg]
        if len(rg_df) == 0: continue
        rg_hit = rg_df.dropna(subset=['hit'])
        rg_acc = rg_hit['hit'].mean() * 100 if len(rg_hit) > 0 else 0
        print(f"  {rg:<12s}: {len(rg_df):>4d} trades | Acc={rg_acc:.1f}% | P&L=₹{rg_df['pnl'].sum():,.0f}")

    print(f"\n  BY DAY:")
    for day in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
        d_df = trades[trades['day_name'] == day]
        if len(d_df) == 0: continue
        d_hit = d_df.dropna(subset=['hit'])
        d_acc = d_hit['hit'].mean() * 100 if len(d_hit) > 0 else 0
        print(f"  {day:<12s}: {len(d_df):>4d} trades | Acc={d_acc:.1f}% | P&L=₹{d_df['pnl'].sum():,.0f}")
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="^NSEI", choices=["^NSEI", "^BSESN"])
    parser.add_argument("--period", default="5y")
    args = parser.parse_args()

    res_df = run_walk_forward_backtest(ticker=args.ticker, period=args.period)
    print_scorecard(res_df, args.ticker)

    # Save results
    out_dir = os.path.join(PROJECT_ROOT, "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym = "nifty" if args.ticker == "^NSEI" else "sensex"
    out = os.path.join(out_dir, f"{sym}_wf_backtest_{ts}.csv")
    res_df.to_csv(out, index=False)
    print(f"  Results saved → {out}")


if __name__ == "__main__":
    main()
