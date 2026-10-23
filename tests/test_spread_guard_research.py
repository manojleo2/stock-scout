"""
tests/test_spread_guard_research.py
───────────────────────────────────
Unit tests for Spread Guard research module, Indian F&O cost calculator,
and paired comparison evaluation harness.
"""

import unittest
import time
from utils.nifty_spread_guard import (
    SpreadGuardConfig,
    evaluate_spread_guard,
    compute_executable_exit_price,
    calculate_fo_transaction_costs,
    SPREAD_CONFIG_NIFTY,
    SPREAD_CONFIG_SENSEX,
)
from utils.spread_guard_evaluator import SpreadGuardEvaluator


class TestSpreadGuardResearch(unittest.TestCase):

    def test_normal_tight_spread_passes(self):
        """Verifies tight liquid ATM option quote (₹0.20 spread) passes the guard."""
        res = evaluate_spread_guard(
            bid=120.00,
            ask=120.20,
            ltp=120.10,
            config=SPREAD_CONFIG_NIFTY
        )
        self.assertTrue(res["is_tradable"])
        self.assertIsNone(res["rejection_reason"])
        self.assertEqual(res["spread_abs"], 0.20)
        self.assertAlmostEqual(res["spread_pct"], 0.167, places=2)
        # Executable entry = Ask + (1 tick * 0.05) = 120.20 + 0.05 = 120.25
        self.assertEqual(res["effective_entry_price"], 120.25)

    def test_wide_absolute_spread_rejected(self):
        """Verifies illiquid option quote with spread > max_spread_abs (₹2.50 > ₹1.50) is rejected."""
        res = evaluate_spread_guard(
            bid=118.00,
            ask=120.50,
            ltp=119.00,
            config=SPREAD_CONFIG_NIFTY
        )
        self.assertFalse(res["is_tradable"])
        self.assertIn("WIDE_ABSOLUTE_SPREAD", res["rejection_reason"])
        self.assertEqual(res["spread_abs"], 2.50)

    def test_wide_relative_spread_rejected(self):
        """Verifies low-premium OTM/expiry option where spread is small in ₹ but large in % is rejected."""
        # e.g. Bid ₹10.00, Ask ₹10.50 -> spread is ₹0.50 (< 1.50 cap), but 4.88% (> 2.0% cap)
        res = evaluate_spread_guard(
            bid=10.00,
            ask=10.50,
            ltp=10.20,
            config=SPREAD_CONFIG_NIFTY
        )
        self.assertFalse(res["is_tradable"])
        self.assertIn("WIDE_RELATIVE_SPREAD", res["rejection_reason"])
        self.assertGreater(res["spread_pct"], 2.0)

    def test_zero_or_crossed_book_depth_rejected(self):
        """Verifies invalid order book quotes (zero bid/ask or crossed book) fail-closed."""
        # Zero bid
        res_zero = evaluate_spread_guard(bid=0.0, ask=120.0, ltp=120.0)
        self.assertFalse(res_zero["is_tradable"])
        self.assertEqual(res_zero["spread_research_result"], "UNOBSERVABLE")
        self.assertIn("UNOBSERVABLE", res_zero["rejection_reason"])

        # Crossed book (Ask < Bid)
        res_crossed = evaluate_spread_guard(bid=125.0, ask=120.0, ltp=122.0)
        self.assertFalse(res_crossed["is_tradable"])
        self.assertEqual(res_crossed["spread_research_result"], "UNOBSERVABLE")
        self.assertIn("CROSSED_BOOK", res_crossed["rejection_reason"])

    def test_stale_quote_rejected(self):
        """Verifies quote older than max_quote_age_sec is rejected."""
        old_time = time.time() - 25.0  # 25 seconds ago (cap is 10s)
        res = evaluate_spread_guard(
            bid=120.0,
            ask=120.30,
            ltp=120.15,
            quote_timestamp=old_time,
            config=SPREAD_CONFIG_NIFTY
        )
        self.assertFalse(res["is_tradable"])
        self.assertIn("STALE_QUOTE", res["rejection_reason"])

    def test_indian_fo_transaction_costs_2026_schedule(self):
        """
        Verifies exact 2026 statutory Indian F&O tax schedule:
        1 lot Nifty (65 qty): Buy @ ₹120.00, Sell @ ₹156.00 (+30% target)
        Turnovers:
          Buy Turnover  = 120 * 65 = ₹7,800
          Sell Turnover = 156 * 65 = ₹10,140
          Total Turnover = ₹17,940
        """
        costs = calculate_fo_transaction_costs(
            buy_premium=120.00,
            sell_premium=156.00,
            quantity=65,
            brokerage_per_order=20.0
        )
        # Brokerage = 20 * 2 = 40.0
        self.assertEqual(costs["brokerage"], 40.0)
        # STT = 0.10% on Sell Turnover (10,140 * 0.001) = 10.14
        self.assertEqual(costs["stt"], 10.14)
        # Exchange charges = 17,940 * 0.000505 = 9.06
        self.assertEqual(costs["exchange_charges"], 9.06)
        # GST = 18% on (40 + 9.06) = 18% of 49.06 = 8.83
        self.assertEqual(costs["gst"], 8.83)
        # Total cost should be around ₹68 - ₹70
        self.assertGreater(costs["total_cost"], 68.0)
        self.assertLess(costs["total_cost"], 70.0)

    def test_paired_evaluator_zero_data_verdict(self):
        """Verifies empty dataset returns INSUFFICIENT DATA without fabricating backtests."""
        evaluator = SpreadGuardEvaluator()
        result = evaluator.evaluate_dataset([])
        self.assertEqual(result["verdict"], "INSUFFICIENT DATA")
        self.assertEqual(result["sample_size"], 0)

    def test_paired_evaluator_controlled_comparison(self):
        """Verifies paired evaluation on a controlled test dataset with known spread anomalies."""
        evaluator = SpreadGuardEvaluator()
        test_opportunities = [
            # Opp 1: Tight spread (0.20), Winner
            {
                "ticker": "^NSEI", "contract": "NIFTY 24500 CE", "lot_size": 65, "lots": 1,
                "ltp": 120.0, "bid": 119.90, "ask": 120.10, "quote_timestamp": time.time(),
                "exit_ltp": 156.0, "exit_bid": 155.90, "exit_ask": 156.10,
                "regime": "TRENDING", "days_to_expiry": 4,
            },
            # Opp 2: Illiquid spread (₹3.50), Loser (Avoided by guard)
            {
                "ticker": "^NSEI", "contract": "NIFTY 24600 CE", "lot_size": 65, "lots": 1,
                "ltp": 80.0, "bid": 78.00, "ask": 81.50, "quote_timestamp": time.time(),
                "exit_ltp": 68.0, "exit_bid": 67.00, "exit_ask": 70.00,
                "regime": "CHOP", "days_to_expiry": 2,
            },
        ]

        res = evaluator.evaluate_dataset(test_opportunities)
        # Rejections must record Opp 2
        self.assertEqual(res["rejections"]["total_rejected"], 1)
        self.assertEqual(res["rejections"]["avoided_losers_count"], 1)
        self.assertEqual(res["guarded_stats"]["trade_count"], 1)
        self.assertEqual(res["baseline_stats"]["trade_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
