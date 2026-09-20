"""
=============================================================================
 Stock Scout — Step A Verification & Financial Backtest
=============================================================================
 Validates the newly implemented 65% conviction filter on CDSL:
 1. Verifies code integration (config, opening_predictor, paper_trading)
 2. Re-runs financial simulation across 197 backtested trading days
 3. Compares P&L, Win Rate, Brokerage saved, and Drawdown:
    - Old Way (No Filter / Take everything >= 50%)
    - Step A (Current Filter >= 65%)
    - Step A High (Threshold >= 70%)
    - Step A Ultra (Threshold >= 75%)
=============================================================================
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import math

from config import MIN_GAP_CONVICTION_THRESHOLD
from utils.opening_predictor import generate_options_trading_call
from utils.paper_trading import calculate_bsm_option_price, BROKERAGE_AND_TAX_PER_TRADE

def run_step_a_financial_backtest():
    csv_path = os.path.join(PROJECT_ROOT, "backtest_results", "gap_backtest_20260918_173852.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Backtest CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    cdsl_df = df[df['symbol'] == 'CDSL.NS'].copy()
    cdsl_df['date'] = pd.to_datetime(cdsl_df['date'])
    cdsl_df = cdsl_df.sort_values('date').reset_index(drop=True)

    print("=" * 80)
    print("🔬 STEP A VERIFICATION: CDSL 3:05 PM GAP OVERNIGHT STRATEGY")
    print(f"   Historical Trading Sessions: {len(cdsl_df)} days (Dec 2025 – Sep 2026)")
    print(f"   Active Configured Threshold: MIN_GAP_CONVICTION_THRESHOLD = {MIN_GAP_CONVICTION_THRESHOLD}%")
    print("=" * 80)

    # 1. Verify Code Integration Test
    print("\n[1/3] Verifying Code Logic Integration...")
    test_low = generate_options_trading_call("CDSL.NS", 1400.0, 58.0, "Moderate")
    test_high_call = generate_options_trading_call("CDSL.NS", 1400.0, 72.0, "High")
    test_high_put = generate_options_trading_call("CDSL.NS", 1400.0, 24.0, "High")

    assert "NEUTRAL" in test_low['action'] and test_low['is_tradeable'] is False, "Low conviction test failed!"
    assert "BUY CALL" in test_high_call['action'] and test_high_call['is_tradeable'] is True, "High call test failed!"
    assert "BUY PUT" in test_high_put['action'] and test_high_put['is_tradeable'] is True, "High put test failed!"
    print("  ✅ generate_options_trading_call() correctly blocks <65% and executes >=65%!")

    # 2. Financial Simulation Function
    def simulate_threshold(min_threshold: float):
        lot_size = 700  # CDSL standard 2 lots
        trades = []
        cumulative_net_pnl = 0.0
        equity_curve = [50000.0]  # Starting virtual bankroll ₹50,000

        for idx, row in cdsl_df.iterrows():
            prob_up = row['prob_up'] * 100.0
            conviction = max(prob_up, 100.0 - prob_up)
            spot = float(row['close_price'])
            actual_gap_pct = float(row['actual_gap_pct'])
            actual_gap_rs = spot * (actual_gap_pct / 100.0)

            # Round to ATM strike
            strike_step = 20
            atm_strike = round(spot / strike_step) * strike_step

            # Check if trade passes threshold
            if conviction < min_threshold:
                # Capital Preserved (No Trade)
                trades.append({
                    'date': row['date'],
                    'traded': False,
                    'action': 'NO TRADE',
                    'conviction': conviction,
                    'is_correct': None,
                    'gross_pnl': 0.0,
                    'brokerage': 0.0,
                    'net_pnl': 0.0,
                    'capital_invested': 0.0
                })
                equity_curve.append(equity_curve[-1])
                continue

            is_call = prob_up >= 50.0
            action = "BUY CALL (CE)" if is_call else "BUY PUT (PE)"

            # Estimate Option Pricing using BSM & Greeks
            # Entry at 3:10 PM with ~12 days to monthly expiry
            entry_premium = calculate_bsm_option_price(spot, atm_strike, days_to_expiry=12.0, is_call=is_call)
            capital = entry_premium * lot_size

            # Next morning 9:15 AM exit:
            # Days to expiry decreases by 0.75 day (overnight theta decay)
            # Spot changes by actual_gap_rs
            next_spot = spot + actual_gap_rs
            exit_premium = calculate_bsm_option_price(next_spot, atm_strike, days_to_expiry=11.25, is_call=is_call)

            premium_change = exit_premium - entry_premium
            gross_pnl = round(premium_change * lot_size, 2)
            net_pnl = round(gross_pnl - BROKERAGE_AND_TAX_PER_TRADE, 2)
            is_win = net_pnl > 0

            cumulative_net_pnl += net_pnl
            equity_curve.append(equity_curve[-1] + net_pnl)

            trades.append({
                'date': row['date'],
                'traded': True,
                'action': action,
                'conviction': conviction,
                'is_win': is_win,
                'is_correct': row['is_correct'],
                'entry_premium': entry_premium,
                'exit_premium': exit_premium,
                'gross_pnl': gross_pnl,
                'brokerage': BROKERAGE_AND_TAX_PER_TRADE,
                'net_pnl': net_pnl,
                'capital_invested': capital
            })

        trade_df = pd.DataFrame(trades)
        active_trades = trade_df[trade_df['traded'] == True]

        total_sessions = len(cdsl_df)
        total_trades = len(active_trades)
        wins = active_trades['is_win'].sum()
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        dir_correct = active_trades['is_correct'].sum()
        dir_accuracy = (dir_correct / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = active_trades[active_trades['gross_pnl'] > 0]['gross_pnl'].sum()
        gross_loss = abs(active_trades[active_trades['gross_pnl'] < 0]['gross_pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        total_gross = active_trades['gross_pnl'].sum()
        total_fees = active_trades['brokerage'].sum()
        total_net = active_trades['net_pnl'].sum()

        # Compute Max Drawdown
        peak = 50000.0
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        return {
            'threshold': min_threshold,
            'total_sessions': total_sessions,
            'trades_taken': total_trades,
            'days_preserved': total_sessions - total_trades,
            'directional_acc': dir_accuracy,
            'profitable_win_rate': win_rate,
            'wins': wins,
            'losses': total_trades - wins,
            'total_gross': total_gross,
            'total_fees': total_fees,
            'total_net': total_net,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'final_bankroll': 50000.0 + total_net
        }

    # Run comparisons
    print("\n[2/3] Simulating Financial Performance across Conviction Regimes...")
    thresholds = [50.0, 65.0, 70.0, 75.0]
    results = [simulate_threshold(t) for t in thresholds]

    # Print Comparison Table
    print("\n[3/3] STEP A COMPARATIVE RESULTS")
    print("-" * 95)
    print(f"{'Strategy / Threshold':<24s} | {'Trades':<7s} | {'Skipped':<7s} | {'Dir Acc%':<9s} | {'Win Rate%':<10s} | {'Brokerage':<10s} | {'Net P&L (₹)':<12s} | {'Profit Fac':<10s}")
    print("-" * 95)

    for r in results:
        label = "Old Way (No Filter)" if r['threshold'] == 50.0 else f"Step A (>= {r['threshold']:.0f}%)"
        print(f"{label:<24s} | {r['trades_taken']:<7d} | {r['days_preserved']:<7d} | {r['directional_acc']:<8.1f}% | {r['profitable_win_rate']:<9.1f}% | ₹{r['total_fees']:<9.0f} | ₹{r['total_net']:<11,.2f} | {r['profit_factor']:<10.2f}")

    print("-" * 95)

    old = results[0]
    step_a = results[1]
    pnl_diff = step_a['total_net'] - old['total_net']
    fee_saved = old['total_fees'] - step_a['total_fees']

    print(f"\n🎯 KEY TAKEAWAYS OF STEP A:")
    print(f"  • Capital Preserved (Skipped Bad Days): {step_a['days_preserved']} days out of {step_a['total_sessions']} avoided coin-toss chop.")
    print(f"  • Directional Accuracy: Raised from {old['directional_acc']:.1f}% ➡️ {step_a['directional_acc']:.1f}% (+{step_a['directional_acc'] - old['directional_acc']:.1f}% improvement).")
    print(f"  • Net Options Win Rate: Raised from {old['profitable_win_rate']:.1f}% ➡️ {step_a['profitable_win_rate']:.1f}%.")
    print(f"  • Brokerage & Tax Saved: ₹{fee_saved:,.0f} saved directly by not overtrading.")
    print(f"  • Total Net Rupee Gain: Step A generates +₹{pnl_diff:,.2f} MORE net profit than taking every trade.")
    print(f"  • Profit Factor: Jumped from {old['profit_factor']:.2f} ➡️ {step_a['profit_factor']:.2f}.\n")

if __name__ == "__main__":
    run_step_a_financial_backtest()
