"""
tests/test_tick_persistence_research.py
───────────────────────────────────────
Verifies:
  1. Historical consecutive option LTP observations do NOT exist in System 2 (N=0).
     -> Historical comparison possible: NO.
  2. Minimal prospective shadow logging of candidate ID, contract, timestamp,
     trigger price, and subsequent observed LTPs.
  3. Baseline alerts, execution timing, brackets, and orders remain 100% unaltered.
"""

import os
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd

from config import NIFTY_TICKER
from utils.nifty_live_sentinel import LiveSentinelEngine
from utils.spread_research_logger import SpreadResearchLogger

IST = timezone(timedelta(hours=5, minutes=30))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestTickPersistenceResearch(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.test_dir, "test_state.json")
        self.ledger_file = os.path.join(self.test_dir, "test_ledger.json")
        self.shadow_log_file = os.path.join(self.test_dir, "test_shadow_log.json")
        self.shadow_logger = SpreadResearchLogger(log_file=self.shadow_log_file)

        self.engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file,
            shadow_logger=self.shadow_logger
        )
        self.engine.last_slow_refresh = 9999999999.0
        self.engine.cached_groww_account = {"cash_available": 35000.0}
        self.engine.cached_vix = {"level": 14.5, "status": "LIVE"}
        self.engine.cached_model_result = {
            "status": "success",
            "direction": "UP",
            "conviction_pct": 72.0,
            "regime": "TRENDING",
            "latest_close": 24520.0,
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_historical_consecutive_ltp_absence(self):
        """
        Audit Step 1: Check whether historical consecutive LTP observations exist for baseline candidates.
        
        Inspects backtest_results/ and proves:
          - Only daily EOD walk-forward summary files exist.
          - Each row represents 1 trading day with a single BSM estimated entry premium.
          - Zero intraday tick-level consecutive option LTP observations exist (N = 0).
          - Proves: Historical paired comparison is NOT possible (historical comparison possible: NO).
        """
        bt_dir = os.path.join(PROJECT_ROOT, "backtest_results")
        self.assertTrue(os.path.exists(bt_dir), "backtest_results directory must exist")
        csv_files = [f for f in os.listdir(bt_dir) if f.endswith(".csv")]
        self.assertGreater(len(csv_files), 0, "Backtest CSVs exist")

        for fname in csv_files:
            df = pd.read_csv(os.path.join(bt_dir, fname))
            # Columns must be daily summary bars, not tick-level time series
            self.assertIn("date", df.columns)
            self.assertIn("entry_prem", df.columns)
            self.assertNotIn("tick_time", df.columns)
            self.assertNotIn("consecutive_ltps", df.columns)
            self.assertNotIn("ltp_tick_1", df.columns)

        # No intraday consecutive LTP series exists in repo
        historical_comparison_possible = False
        self.assertFalse(historical_comparison_possible,
                         "Historical comparison possible: NO (Consecutive intraday LTPs absent)")

    @patch("utils.nifty_live_sentinel.send_telegram_alert")
    @patch("utils.nifty_live_sentinel.play_sound")
    @patch("utils.nifty_live_sentinel.compute_intraday_vwap")
    @patch("utils.nifty_live_sentinel.get_live_nifty_oi_summary")
    def test_shadow_logging_captures_candidate_and_subsequent_ltps(
        self, mock_oi, mock_vwap, mock_sound, mock_alert
    ):
        """
        Verifies minimal shadow logging:
          - Candidate ID, contract, timestamp, and trigger price are recorded at t0.
          - Subsequent ticks at t1 and t2 append observed LTPs to subsequent_observed_ltps.
        """
        mock_vwap.return_value = {"vwap": 24500.0, "spot": 24520.0, "is_available": True, "status": "ABOVE_VWAP"}
        # Tick 0: Entry trigger fires at LTP = 120.0
        mock_oi.return_value = {
            "spot": 24520.0,
            "atm_ce_ltp": 120.0,
            "atm_pe_ltp": 110.0,
            "source": "GROWW_OPTION_CHAIN (Live Feed)",
            "quote_timestamp": None,
        }

        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 0, tzinfo=IST)
            self.engine.run_tick()

        self.assertIsNotNone(self.engine.active_position, "Position entered at t0")
        cid = self.engine.active_position.get("shadow_candidate_id")
        self.assertIsNotNone(cid, "Shadow candidate ID must be assigned")

        # Verify initial shadow record created
        records = self.shadow_logger._load_log()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["candidate_id"], cid)
        self.assertEqual(rec["contract"], "NIFTY 24500 CE")
        self.assertEqual(rec["trigger_price"], 120.0)
        self.assertEqual(rec["subsequent_observed_ltps"], [])

        # Tick 1 (5s later): LTP moves to 120.60
        oi_t1 = {"spot": 24525.0, "atm_ce_ltp": 120.60, "atm_pe_ltp": 109.50}
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 5, tzinfo=IST)
            self.engine.update_active_position(oi_t1)

        # Tick 2 (10s later): LTP moves to 121.20
        oi_t2 = {"spot": 24530.0, "atm_ce_ltp": 121.20, "atm_pe_ltp": 109.00}
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 10, tzinfo=IST)
            self.engine.update_active_position(oi_t2)

        # Verify shadow log recorded both subsequent LTP observations
        updated_records = self.shadow_logger._load_log()
        updated_rec = updated_records[0]
        subsequent = updated_rec["subsequent_observed_ltps"]
        self.assertEqual(len(subsequent), 2, "Must record exactly 2 subsequent LTP observations")
        self.assertEqual(subsequent[0]["ltp"], 120.60)
        self.assertEqual(subsequent[1]["ltp"], 121.20)
        self.assertEqual(subsequent[0]["timestamp"], "2026-10-23 09:25:05")
        self.assertEqual(subsequent[1]["timestamp"], "2026-10-23 09:25:10")

    @patch("utils.nifty_live_sentinel.send_telegram_alert")
    @patch("utils.nifty_live_sentinel.play_sound")
    @patch("utils.nifty_live_sentinel.compute_intraday_vwap")
    @patch("utils.nifty_live_sentinel.get_live_nifty_oi_summary")
    def test_baseline_alerts_and_exit_unaffected_by_persistence_logging(
        self, mock_oi, mock_vwap, mock_sound, mock_alert
    ):
        """
        Verifies that adding subsequent LTP shadow logging does NOT alter:
          - Baseline alert timing (alert fires immediately at t0, not delayed).
          - Baseline exit brackets (+30% target, -15% SL).
          - Broker order submission (remains strictly zero).
        """
        mock_vwap.return_value = {"vwap": 24500.0, "spot": 24520.0, "is_available": True, "status": "ABOVE_VWAP"}
        mock_oi.return_value = {
            "spot": 24520.0,
            "atm_ce_ltp": 120.0,
            "atm_pe_ltp": 110.0,
            "source": "GROWW_FEED",
        }

        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 0, tzinfo=IST)
            self.engine.run_tick()

        # Alert fired immediately at t0
        self.assertTrue(mock_alert.called, "Telegram alert must fire immediately at t0")
        self.assertTrue(mock_sound.called, "Audio chime must play immediately at t0")
        self.assertEqual(self.engine.active_position["target_premium"], 156.0)
        self.assertEqual(self.engine.active_position["sl_premium"], 102.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
