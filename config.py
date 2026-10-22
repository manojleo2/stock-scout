"""
config.py — Nifty & Sensex Index Options Predictor
All constants for the 10-module ML options prediction system.
Fix #1: 2026 exchange rules — Nifty Tuesday expiry, Sensex Thursday expiry.
Fix #4: Dynamic Target/SL as % of entry premium (not fixed ₹).
Fix #5: VWAP fallback thresholds.
"""

# ── Index & Proxy Tickers ─────────────────────────────────────────────────────
NIFTY_TICKER           = "^NSEI"
SENSEX_TICKER          = "^BSESN"
NIFTYBEES_TICKER       = "NIFTYBEES.NS"      # ETF proxy — real volume for VWAP fallback
BANKNIFTY_TICKER       = "^NSEBANK"
INDIA_VIX_TICKER       = "^INDIAVIX"

# Global macro tickers
SP500_TICKER           = "^GSPC"
NASDAQ_TICKER          = "^IXIC"
DOW_TICKER             = "^DJI"
NIKKEI_TICKER          = "^N225"
HANGSENG_TICKER        = "^HSI"

# ── 2026 Lot Sizes (Fix #1) ───────────────────────────────────────────────────
NIFTY_LOT_SIZE         = 65       # NSE Nifty 50 weekly options (2026)
SENSEX_LOT_SIZE        = 20       # BSE Sensex weekly options (2026)

# ── Strike Intervals ──────────────────────────────────────────────────────────
NIFTY_STRIKE_STEP      = 50       # ₹50 per strike for Nifty weekly
SENSEX_STRIKE_STEP     = 100      # ₹100 per strike for Sensex weekly

# ── Weekly Expiry Regime (Fix #1: 2026 Exchange Rules) ───────────────────────
# Nifty weekly expiry changed Thursday → Tuesday on Sep 4, 2025
NIFTY_EXPIRY_WEEKDAY       = 1    # Tuesday (0=Mon,1=Tue,2=Wed,3=Thu,4=Fri)
NIFTY_OLD_EXPIRY_WEEKDAY   = 3    # Thursday (pre-Sep 2025 legacy)
NIFTY_EXPIRY_REGIME_SHIFT  = "2025-09-04"  # Backtest regime boundary

# Sensex weekly expiry = Thursday
SENSEX_EXPIRY_WEEKDAY      = 3    # Thursday

# ── ML Model Settings ─────────────────────────────────────────────────────────
ML_N_ESTIMATORS        = 150
ML_MAX_DEPTH           = 5
ML_MIN_SAMPLES_LEAF    = 20
ML_TRAIN_SPLIT         = 0.70
ML_CAL_SPLIT           = 0.85    # Calibration window (70–85%)
ML_WARMUP_DAYS         = 300     # Minimum sessions before walk-forward starts

# ── Conviction Thresholds ─────────────────────────────────────────────────────
NIFTY_MIN_CONVICTION_WEEKDAY    = 62.0   # Mon–Thu baseline
NIFTY_MIN_CONVICTION_FRIDAY     = 68.0   # Friday weekend theta shield
NIFTY_MIN_CONVICTION_EXPIRY_DAY = 75.0   # Expiry day — strict
NIFTY_MIN_CONVICTION_CHOP       = 80.0   # Chop regime hard block (<80% filtered)
NIFTY_MIN_CONVICTION_VOLATILE   = 72.0   # Volatile regime (VIX > 18)

# ── Dynamic Option Risk Management (Optimized Wider 2:1 Bracket) ─────────────
NIFTY_TARGET_PCT          = 30.0    # Target: entry_premium × 1.30  (+30%)
NIFTY_SL_PCT              = 15.0    # Stop loss: entry_premium × 0.85 (−15%)
NIFTY_TRAIL_TRIGGER_PCT   = 18.0    # At +18% premium gain → trail SL to cost
NIFTY_TRAIL_LOCK_PCT      = 25.0    # At +25% premium gain → lock profit
NIFTY_TRAIL_LOCK_PROFIT_PCT = 15.0  # Locked SL level = cost + 15%
NIFTY_HARD_CUTOFF_TIME    = "15:05"

# Minimum open→close % move to classify day as directional (filter chop)
NIFTY_INTRADAY_MIN_MOVE_PCT = 0.30   # 0.30% minimum

# ── OI/PCR Conviction Modifier (Fix #3: live-only post-model) ─────────────────
PCR_BEARISH_THRESHOLD   = 1.30       # PCR_OI > 1.30 → bearish pressure
PCR_BULLISH_THRESHOLD   = 0.75       # PCR_OI < 0.75 → bullish pressure
PCR_CONVICTION_BOOST    = 3.0        # ±3% per confirming PCR signal
MAX_PCR_ADJUSTMENT      = 6.0        # Hard cap on total PCR modifier

# ── VWAP Gate Settings (Fix #5) ──────────────────────────────────────────────
VWAP_FETCH_PERIOD         = "1d"     # 1-day intraday window
VWAP_INTERVAL             = "1m"     # 1-minute candles
VWAP_MIN_VOLUME_THRESHOLD = 0.05     # If >5% of candles have non-zero volume → use spot VWAP
VWAP_NIFTYBEES_SCALE      = 100.0    # NIFTYBEES ≈ Nifty / 100

# ── Backtest BSM Option Pricing ───────────────────────────────────────────────
BSM_RISK_FREE_RATE     = 0.065       # 6.5% Indian risk-free rate
BSM_DEFAULT_IV         = 0.18        # 18% IV fallback if ATR-derived IV fails
