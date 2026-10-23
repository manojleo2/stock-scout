"""
tests/test_baseline.py
Baseline unit tests for System 2 (Nifty & Sensex Options Predictor).
Validates dynamic levels, strike rounding, lot sizes, and regime classification without live market data.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from utils.nifty_trade_signal import compute_dynamic_levels, get_atm_strike, get_lot_size
from utils.nifty_features import get_days_to_nifty_expiry, get_days_to_sensex_expiry
from utils.nifty_regime import classify_regime
from config import (
    NIFTY_LOT_SIZE, SENSEX_LOT_SIZE,
    NIFTY_TARGET_PCT, NIFTY_SL_PCT,
    NIFTY_TRAIL_TRIGGER_PCT, NIFTY_TRAIL_LOCK_PCT, NIFTY_TRAIL_LOCK_PROFIT_PCT,
)


class TestBaselineSystem2(unittest.TestCase):
    def test_dynamic_levels_exact_calculation(self):
        """Verifies 2:1 dynamic brackets (+30% / -15% / +18% trail / +25% lock)."""
        entry_premium = 120.0
        lot_size = 65
        levels = compute_dynamic_levels(entry_premium, lot_size=lot_size)

        expected_target = round(entry_premium * (1 + NIFTY_TARGET_PCT / 100.0), 2)
        expected_sl = round(entry_premium * (1 - NIFTY_SL_PCT / 100.0), 2)
        expected_trail = round(entry_premium * (1 + NIFTY_TRAIL_TRIGGER_PCT / 100.0), 2)
        expected_lock = round(entry_premium * (1 + NIFTY_TRAIL_LOCK_PCT / 100.0), 2)
        expected_locked_sl = round(entry_premium * (1 + NIFTY_TRAIL_LOCK_PROFIT_PCT / 100.0), 2)

        self.assertEqual(levels["entry_premium"], entry_premium)
        self.assertEqual(levels["target_premium"], expected_target)  # 156.0
        self.assertEqual(levels["sl_premium"], expected_sl)          # 102.0
        self.assertEqual(levels["trail_trigger_premium"], expected_trail)  # 141.6
        self.assertEqual(levels["lock_trigger_premium"], expected_lock)    # 150.0
        self.assertEqual(levels["locked_sl_premium"], expected_locked_sl)  # 138.0
        self.assertEqual(levels["rr_ratio"], 2.0)

    def test_strike_and_lot_specifications(self):
        """Verifies 2026 exchange lot rules and strike steps."""
        self.assertEqual(get_lot_size("^NSEI"), NIFTY_LOT_SIZE)
        self.assertEqual(get_lot_size("^BSESN"), SENSEX_LOT_SIZE)
        self.assertEqual(get_atm_strike(24523, "^NSEI"), 24500)
        self.assertEqual(get_atm_strike(24535, "^NSEI"), 24550)
        self.assertEqual(get_atm_strike(81360, "^BSESN"), 81400)
        self.assertEqual(get_atm_strike(81340, "^BSESN"), 81300)

    def test_regime_classification_synthetic(self):
        """Verifies regime classifier output structure."""
        dates = pd.date_range("2026-01-01", periods=30)
        prices = np.linspace(24000, 25000, 30)
        df = pd.DataFrame({
            "Open": prices - 10,
            "High": prices + 25,
            "Low": prices - 25,
            "Close": prices,
            "Volume": [5000] * 30
        }, index=dates)

        regime, score, detail = classify_regime(df)
        self.assertIn(regime, ["TRENDING", "CHOP", "VOLATILE"])
        self.assertIsInstance(score, float)
        self.assertIsInstance(detail, str)


if __name__ == "__main__":
    unittest.main()
