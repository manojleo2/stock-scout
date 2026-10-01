import time
import sys
import yfinance as yf
from utils.paper_trading import calculate_bsm_option_price

ENTRY_PREM = 35.52
TARGET_PREM = ENTRY_PREM + 10.0   # 45.52
TRAIL_1 = ENTRY_PREM + 7.0       # 42.52
TRAIL_2 = ENTRY_PREM + 8.5       # 44.02
SL_PREM = ENTRY_PREM - 5.0        # 30.52
STRIKE = 1260
IS_CALL = False
MAX_RUN_SECONDS = 7200  # 2 hours

start_time = time.time()
print(f"Monitoring started for CDSL 1260 PE. Entry: {ENTRY_PREM}, Target: {TARGET_PREM}, Trail1: {TRAIL_1}, SL: {SL_PREM}")
sys.stdout.flush()

while time.time() - start_time < MAX_RUN_SECONDS:
    try:
        ticker = yf.Ticker("CDSL.NS")
        spot = ticker.fast_info.last_price
        
        est_prem = calculate_bsm_option_price(spot=spot, strike=STRIKE, is_call=IS_CALL, days_to_expiry=28, iv=0.30)
        
        if est_prem >= TARGET_PREM or spot <= 1245.0:
            print(f"TRIGGER_TARGET: CDSL Spot={spot:.2f} | 1260 PE Premium est={est_prem:.2f} >= {TARGET_PREM}!")
            sys.stdout.flush()
            sys.exit(0)
            
        if est_prem >= TRAIL_1 or spot <= 1251.0:
            print(f"TRIGGER_TRAIL_COST: CDSL Spot={spot:.2f} | 1260 PE Premium est={est_prem:.2f} >= {TRAIL_1}! Move SL to Cost {ENTRY_PREM}!")
            sys.stdout.flush()
            sys.exit(0)
            
        if est_prem <= SL_PREM or spot >= 1272.0:
            print(f"TRIGGER_SL: CDSL Spot={spot:.2f} | 1260 PE Premium est={est_prem:.2f} <= {SL_PREM}!")
            sys.stdout.flush()
            sys.exit(0)
            
    except Exception as e:
        pass
        
    time.sleep(15)

print("Monitoring finished.")
