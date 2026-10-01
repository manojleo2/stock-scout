import time
import sys
import yfinance as yf
from utils.paper_trading import calculate_bsm_option_price

TARGET_PREM = 47.58
SL_PREM = 35.58
STRIKE = 1280
IS_CALL = False
MAX_RUN_SECONDS = 3600  # Run for up to 1 hour

start_time = time.time()
print(f"Monitoring started for CDSL 1280 PE. Target alert: >={TARGET_PREM}, SL alert: <={SL_PREM}")
sys.stdout.flush()

while time.time() - start_time < MAX_RUN_SECONDS:
    try:
        ticker = yf.Ticker("CDSL.NS")
        # Use fast_info to get instant latest spot
        spot = ticker.fast_info.last_price
        
        # Approximate option premium: intrinsic + extrinsic
        # Intrinsic for 1280 PE is max(0, 1280 - spot)
        # Or BSM price with IV ~0.30
        est_prem = calculate_bsm_option_price(spot=spot, strike=STRIKE, is_call=IS_CALL, days_to_expiry=29, iv=0.30)
        
        # If spot drops below 1264, 1280 PE is at or above ~47.50
        if est_prem >= TARGET_PREM or spot <= 1264.0:
            print(f"TRIGGER_TARGET: CDSL Spot={spot:.2f} | 1280 PE Premium est={est_prem:.2f} >= {TARGET_PREM}!")
            sys.stdout.flush()
            sys.exit(0)
            
        if est_prem <= SL_PREM or spot >= 1278.0:
            print(f"TRIGGER_SL: CDSL Spot={spot:.2f} | 1280 PE Premium est={est_prem:.2f} <= {SL_PREM}!")
            sys.stdout.flush()
            sys.exit(0)
            
    except Exception as e:
        pass
        
    time.sleep(15)

print("Monitoring timed out after 1 hour.")
