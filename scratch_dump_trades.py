import sys
import os
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

df = pd.read_csv('backtest_results/cdsl_intraday_2lots_weekly.csv')
active = df[df['model_action'] != 'NO_TRADE'].copy().reset_index(drop=True)

# Separate into Wins, Losses, Cutoffs
wins = active[active['hit_status'].str.contains('TARGET')].copy()
losses = active[active['hit_status'].str.contains('STOP LOSS')].copy()
cutoffs = active[active['hit_status'].str.contains('CUTOFF')].copy()

def format_table(sub_df, title):
    print(f"\n=========================================================================================")
    print(f" {title.upper()} ({len(sub_df)} Trades) | Total PnL: Rs {sub_df['net_pnl'].sum():+,.2f}")
    print(f"=========================================================================================")
    headers = ["#", "Date", "Contract", "Entry Spot", "Entry Prem", "Exit Prem", "Prem Move", "Exit Time", "Net PnL (Rs)"]
    print(f"{headers[0]:<3} {headers[1]:<12} {headers[2]:<14} {headers[3]:<12} {headers[4]:<12} {headers[5]:<11} {headers[6]:<11} {headers[7]:<10} {headers[8]:<14}")
    print("-" * 105)
    for idx, r in sub_df.reset_index(drop=True).iterrows():
        print(f"{idx+1:<3} {r['date']:<12} {r['contract']:<14} Rs {r['entry_spot']:<9.2f} Rs {r['entry_prem']:<9.2f} Rs {r['exit_prem']:<8.2f} {r['opt_pnl_pt']:+6.2f} pt  {r['hit_time']:<10} Rs {r['net_pnl']:+10.2f}")

format_table(wins, "1. ALL WINNING TRADES (TARGET +Rs 10.00 HIT)")
format_table(losses, "2. ALL LOSING TRADES (STOP LOSS -Rs 5.00 HIT)")
format_table(cutoffs, "3. ALL CUTOFF TRADES (03:05 PM CUTOFF EXITS)")

print(f"\n=========================================================================================")
print(f" GRAND TOTAL VERIFICATION (2 LOTS / 950 QTY)")
print(f"=========================================================================================")
print(f" 🎯 Wins Total (9 trades)   : Rs {wins['net_pnl'].sum():+,.2f}  (Avg: Rs {wins['net_pnl'].mean():+,.2f}/trade)")
print(f" 🛑 Losses Total (7 trades) : Rs {losses['net_pnl'].sum():+,.2f}  (Avg: Rs {losses['net_pnl'].mean():+,.2f}/trade)")
print(f" ⏱️ Cutoffs Total (14 trades): Rs {cutoffs['net_pnl'].sum():+,.2f}  (Avg: Rs {cutoffs['net_pnl'].mean():+,.2f}/trade)")
print(f" -----------------------------------------------------------------------------------------")
print(f" 💰 NET PROFIT TOTAL        : Rs {active['net_pnl'].sum():+,.2f}")
print(f"=========================================================================================")
