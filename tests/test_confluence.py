"""
tests/test_confluence.py
────────────────────────
Unit tests for the Multi-Factor Confluence & Early Defense Engine.
"""

import unittest
from datetime import time as dtime

from utils.nifty_confluence import (
    compute_confluence_score,
    get_market_phase,
    evaluate_early_defense_exit,
    CONFLUENCE_THRESHOLD_TRENDING,
    CONFLUENCE_THRESHOLD_CHOP,
    CONFLUENCE_THRESHOLD_EXPIRY,
)


class TestNiftyConfluence(unittest.TestCase):

    def setUp(self):
        self.mock_model_up = {
            "status": "success",
            "direction": "UP",
            "conviction_pct": 68.0,
            "confidence": "High",
            "regime": "TRENDING",
            "is_friday": False,
            "is_expiry_day": False,
            "latest_close": 24500.0,
        }

        self.mock_vwap_above = {
            "vwap": 24480.0,
            "spot": 24520.0,
            "method": "SPOT_VWAP",
        }

        self.mock_vwap_below = {
            "vwap": 24530.0,
            "spot": 24490.0,
            "method": "SPOT_VWAP",
        }

        self.mock_oi_bullish = {
            "pcr_oi": 0.72,
            "spot": 24520.0,
            "atm_ce_ltp": 125.0,
            "atm_pe_ltp": 95.0,
        }

        self.mock_overnight_bullish = {
            "global_markets": [
                {"label": "S&P 500", "change": 0.55},
                {"label": "Nasdaq", "change": 0.70},
            ],
            "sgx_gap_pct": 0.25,
        }

    def test_market_phase_detection(self):
        # 09:05 AM -> Pre-Market Active
        p1 = get_market_phase(dtime(9, 5))
        self.assertEqual(p1["phase"], "PRE_MARKET_ACTIVE")
        self.assertFalse(p1["can_scan"])

        # 09:20 AM -> Active Scanning
        p2 = get_market_phase(dtime(9, 20))
        self.assertEqual(p2["phase"], "ACTIVE_SCANNING")
        self.assertTrue(p2["can_scan"])

        # 12:00 PM -> Lunch window
        p3 = get_market_phase(dtime(12, 0))
        self.assertEqual(p3["phase"], "ACTIVE_SCANNING")
        self.assertTrue(p3.get("is_lunch_window", False))

        # 15:06 PM -> Hard Cutoff
        p4 = get_market_phase(dtime(15, 6))
        self.assertEqual(p4["phase"], "HARD_CUTOFF")
        self.assertFalse(p4["can_scan"])

    def test_high_confluence_passes(self):
        res = compute_confluence_score(
            model_result=self.mock_model_up,
            vwap_data=self.mock_vwap_above,
            oi_summary=self.mock_oi_bullish,
            overnight_cues=self.mock_overnight_bullish,
            other_index_result={"status": "success", "direction": "UP"},
            current_time=dtime(9, 25),
        )

        self.assertTrue(res["vwap_confirmed"])
        self.assertGreaterEqual(res["total_confluence"], CONFLUENCE_THRESHOLD_TRENDING)
        self.assertTrue(res["is_passed"])
        self.assertEqual(len(res["factors"]), 5)

    def test_vwap_mismatch_fails_gate(self):
        # Model is UP, but spot is below VWAP
        res = compute_confluence_score(
            model_result=self.mock_model_up,
            vwap_data=self.mock_vwap_below,
            oi_summary=self.mock_oi_bullish,
            current_time=dtime(9, 25),
        )

        self.assertFalse(res["vwap_confirmed"])
        self.assertFalse(res["is_passed"])

    def test_chop_regime_requires_balanced_confluence(self):
        model_chop = dict(self.mock_model_up)
        model_chop["regime"] = "CHOP"
        model_chop["conviction_pct"] = 56.0  # Would have failed 80% old wall

        res = compute_confluence_score(
            model_result=model_chop,
            vwap_data=self.mock_vwap_above,
            oi_summary=self.mock_oi_bullish,
            overnight_cues=self.mock_overnight_bullish,
            other_index_result={"status": "success", "direction": "UP"},
            current_time=dtime(9, 30),
        )

        self.assertEqual(res["threshold"], CONFLUENCE_THRESHOLD_CHOP)
        # With multi-factor confluence, 56% model + strong VWAP + PCR + Macro passes 65 pts!
        self.assertTrue(res["is_passed"])

    def test_early_defense_triggers_on_vwap_loss(self):
        active_pos = {
            "contract": "NIFTY 24500 CE",
            "direction": "UP",
            "entry_premium": 120.0,
            "lots": 1,
            "total_qty": 65,
        }

        # Case 1: Premium dropped to 110 (-8.3%) AND spot fell below VWAP
        triggered, reason = evaluate_early_defense_exit(
            active_position=active_pos,
            current_spot=24470.0,
            current_option_ltp=110.0,
            current_vwap=24500.0,
        )
        self.assertTrue(triggered)
        self.assertIn("Early Defense Triggered", reason)

        # Case 2: Premium is at 125 (+4.1% profit) -> Early defense must NOT trigger
        triggered2, _ = evaluate_early_defense_exit(
            active_position=active_pos,
            current_spot=24520.0,
            current_option_ltp=125.0,
            current_vwap=24500.0,
        )
        self.assertFalse(triggered2)

        # Case 3: Premium dropped only to 117 (-2.5%, normal noise) -> Must NOT trigger
        triggered3, _ = evaluate_early_defense_exit(
            active_position=active_pos,
            current_spot=24490.0,
            current_option_ltp=117.0,
            current_vwap=24500.0,
        )
        self.assertFalse(triggered3)


if __name__ == "__main__":
    unittest.main()
