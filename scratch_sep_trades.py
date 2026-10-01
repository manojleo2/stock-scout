import sys, os
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

df = pd.read_csv('backtest_results/cdsl_intraday_2lots_weekly.csv')
active = df[(df['date'] >= '2026-09-01') & (df['date'] <= '2026-09-25') & (df['model_action'] != 'NO_TRADE')].copy().reset_index(drop=True)

print(f"Total Executed Trades in September: {len(active)}")
for idx, r in active.iterrows():
    pnl = r['net_pnl']
    print(f"{idx+1:02d} | Date: {r['date']} ({r['day_name'][:3]}) | Contract: {r['contract']}")
    print(f"   Entry Spot: Rs {r['entry_spot']:.2f} at {r['entry_time']} | Entry Prem: Rs {r['entry_prem']:.2f}")
    print(f"   Exit Prem : Rs {r['exit_prem']:.2f} | Premium Delta: {r['opt_pnl_pt']:+.2f} pts")
    print(f"   Outcome   : {r['hit_status']} at {r['hit_time']}")
    print(f"   Net P&L   : Rs {pnl:+,.2f} (on 2 Lots = 950 Qty after Rs 100 brokerage)")
    print("-" * 75)

print(f"\nSEPTEMBER SUMMARY FOR THE 11 EXECUTED TRADES:")
wins = active[active['hit_status'].str.contains('TARGET')]
losses = active[active['hit_status'].str.contains('STOP LOSS')]
cutoffs = active[active['hit_status'].str.contains('CUTOFF')]

print(f"Targets Hit (+10 pt) : {len(wins)} trades -> Rs {wins['net_pnl'].sum():+,.2f}")
print(f"SL Hit (-5 pt)       : {len(losses)} trades -> Rs {losses['net_pnl'].sum():+,.2f}")
print(f"03:05 PM Cutoffs     : {len(cutoffs)} trades -> Rs {cutoffs['net_pnl'].sum():+,.2f}")
print(f"TOTAL NET PROFIT     : Rs {active['net_pnl'].sum():+,.2f}")
