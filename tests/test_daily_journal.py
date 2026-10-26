"""
tests/test_daily_journal.py
───────────────────────────
Tests for daily decision journal loading, validation, and dynamic entry formatting.
"""

import os
import json
import unittest
from utils.daily_journal import load_daily_journal, get_today_entry, save_daily_journal


class TestDailyDecisionJournal(unittest.TestCase):
    def test_load_daily_journal(self):
        records = load_daily_journal()
        self.assertIsInstance(records, list)
        self.assertGreaterEqual(len(records), 5, "Must have at least 5 historical daily session records")
        
        # Verify required keys in each record
        required_keys = [
            "date", "day_of_week", "ticker", "contract", "decision", 
            "decision_badge", "direction", "conviction_pct", "required_threshold_pct",
            "regime", "summary", "why_no_trade"
        ]
        for rec in records:
            for k in required_keys:
                self.assertIn(k, rec, f"Record for {rec.get('date')} missing key: {k}")

    def test_oct_23_session_record(self):
        records = load_daily_journal()
        rec_oct23 = next((r for r in records if r.get("date") == "2026-10-23"), None)
        self.assertIsNotNone(rec_oct23, "Oct 23 record must exist")
        self.assertEqual(rec_oct23["decision"], "NO_TRADE")
        self.assertEqual(rec_oct23["conviction_pct"], 54.2)
        self.assertEqual(rec_oct23["required_threshold_pct"], 72.0)
        self.assertIn("strictly blocked trade execution", rec_oct23["summary"])

    def test_get_today_entry_with_subthreshold_model(self):
        mock_model = {
            "status": "success",
            "direction": "DOWN",
            "conviction_pct": 54.49,
            "required_threshold": 68.0,
            "regime": "TRENDING",
            "is_tradeable": False,
            "latest_close": 22888.95
        }
        today_entry = get_today_entry(mock_model)
        self.assertEqual(today_entry["decision"], "NO_TRADE")
        self.assertEqual(today_entry["conviction_pct"], 54.49)
        self.assertEqual(today_entry["required_threshold_pct"], 68.0)
        self.assertIn("Capital Preserved", today_entry["decision_badge"])
        self.assertIn("sub-threshold", today_entry["summary"])


if __name__ == "__main__":
    unittest.main()
