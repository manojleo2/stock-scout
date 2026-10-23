"""
tests/test_shadow_logger_integration.py
───────────────────────────────────────
Verifies end-to-end integration of SpreadResearchLogger with LiveSentinelEngine:
  1. Shadow logger is invoked for every eligible System 2 candidate opportunity.
  2. Timestamped bid, ask, and quote provenance are recorded at candidate time.
  3. Guard-rejected opportunities are recorded with rejection reason and counterfactual flag.
  4. Missing quote depth (bid/ask absent) is recorded safely without disrupting execution.
  5. Subsequent baseline exit outcomes are tracked and recorded upon trade close.
  6. Unmeasurable outcomes are handled and flagged when contemporaneous exit quotes are unavailable.
  7. Live alert flow remains completely unmodified (no gating or disruption).
"""

import os
import time
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from utils.nifty_live_sentinel import LiveSentinelEngine, get_ist_now
from utils.spread_research_logger import SpreadResearchLogger
from utils.nifty_spread_guard import SpreadGuardConfig
from config import NIFTY_TICKER


class TestShadowLoggerIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.test_dir, "test_state.json")
        self.ledger_file = os.path.join(self.test_dir, "test_ledger.json")
        self.shadow_log_file = os.path.join(self.test_dir, "test_shadow_log.json")
        self.shadow_logger = SpreadResearchLogger(log_file=self.shadow_log_file)

        # Base engine instance
        self.engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file,
            shadow_logger=self.shadow_logger
        )

        # Ensure slow loop refresh does not run or overwrite test state
        self.engine.last_slow_refresh = 9999999999.0
        self.engine.cached_groww_account = {"cash_available": 35000.0}
        self.engine.cached_vix = {"level": 14.5, "status": "LIVE"}

        # Mock ML model cache with valid bullish signal
        self.engine.cached_model_result = {
            "status": "success",
            "direction": "UP",
            "conviction_pct": 72.0,
            "regime": "TRENDING",
            "latest_close": 24520.0,
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("utils.nifty_live_sentinel.send_telegram_alert")
    @patch("utils.nifty_live_sentinel.play_sound")
    @patch("utils.nifty_live_sentinel.compute_intraday_vwap")
    @patch("utils.nifty_live_sentinel.get_live_nifty_oi_summary")
    def test_candidate_logged_with_observed_depth_and_unaltered_alert(
        self, mock_oi, mock_vwap, mock_sound, mock_alert
    ):
        """Verifies eligible candidate with tight spread is logged and baseline alert fires."""
        mock_vwap.return_value = {"vwap": 24500.0, "spot": 24520.0, "is_available": True, "status": "ABOVE_VWAP"}
        mock_oi.return_value = {
            "spot": 24520.0,
            "atm_ce_ltp": 120.0,
            "atm_pe_ltp": 110.0,
            "atm_ce_bid": 119.90,
            "atm_ce_ask": 120.10,
            "atm_pe_bid": 109.90,
            "atm_pe_ask": 110.10,
            "pcr_oi": 0.85,
            "source": "GROWW_OPTION_CHAIN (Live Feed)",
            "quote_timestamp": time.time(),
        }

        # Mock market hours to ensure candidate is eligible
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            from datetime import datetime, timezone, timedelta
            IST = timezone(timedelta(hours=5, minutes=30))
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 0, tzinfo=IST)

            state = self.engine.run_tick()

        # 1. Verify baseline execution was NOT blocked
        self.assertIsNotNone(self.engine.active_position, "Baseline trade must execute")
        self.assertTrue(mock_alert.called, "Telegram alert must fire identically")
        self.assertTrue(mock_sound.called, "Audio alert must fire identically")

        # 2. Verify shadow log recorded the candidate opportunity
        records = self.shadow_logger._load_log()
        self.assertEqual(len(records), 1, "Exactly one candidate opportunity must be logged")
        rec = records[0]

        self.assertEqual(rec["contract"], "NIFTY 24500 CE")
        self.assertEqual(rec["bid"], 119.90)
        self.assertEqual(rec["ask"], 120.10)
        self.assertEqual(rec["spread_abs"], 0.20)
        self.assertTrue(rec["has_observed_depth"])
        self.assertEqual(rec["quote_provenance"], "GROWW_OPTION_CHAIN (Live Feed)")
        self.assertEqual(rec["baseline_decision"], "EXECUTE")
        self.assertEqual(rec["guard_decision"], "PASS")
        self.assertIsNone(rec["guard_rejection_reason"])
        self.assertEqual(rec["outcome_status"], "PENDING_EXIT")

    @patch("utils.nifty_live_sentinel.send_telegram_alert")
    @patch("utils.nifty_live_sentinel.play_sound")
    @patch("utils.nifty_live_sentinel.compute_intraday_vwap")
    @patch("utils.nifty_live_sentinel.get_live_nifty_oi_summary")
    def test_guard_rejected_candidate_logged_and_executed_in_baseline(
        self, mock_oi, mock_vwap, mock_sound, mock_alert
    ):
        """Verifies candidate with wide spread is rejected by guard in shadow log BUT executes in baseline."""
        mock_vwap.return_value = {"vwap": 24500.0, "spot": 24520.0, "is_available": True, "status": "ABOVE_VWAP"}
        # Wide spread: Bid 118.00, Ask 121.00 (Spread ₹3.00 > ₹1.50 cap)
        mock_oi.return_value = {
            "spot": 24520.0,
            "atm_ce_ltp": 120.0,
            "atm_pe_ltp": 110.0,
            "atm_ce_bid": 118.00,
            "atm_ce_ask": 121.00,
            "pcr_oi": 0.85,
            "source": "GROWW_OPTION_CHAIN",
            "quote_timestamp": time.time(),
        }

        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            from datetime import datetime, timezone, timedelta
            IST = timezone(timedelta(hours=5, minutes=30))
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 0, tzinfo=IST)

            state = self.engine.run_tick()

        # Baseline execution MUST proceed (do not turn on spread guard!)
        self.assertIsNotNone(self.engine.active_position, "Baseline must take trade even if wide spread")
        self.assertTrue(mock_alert.called, "Telegram alert must fire")

        # Shadow log must record guard_decision: REJECT
        records = self.shadow_logger._load_log()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["baseline_decision"], "EXECUTE")
        self.assertEqual(rec["guard_decision"], "REJECT")
        self.assertIn("WIDE_ABSOLUTE_SPREAD", rec["guard_rejection_reason"])
        self.assertEqual(rec["spread_abs"], 3.00)

    @patch("utils.nifty_live_sentinel.send_telegram_alert")
    @patch("utils.nifty_live_sentinel.play_sound")
    @patch("utils.nifty_live_sentinel.compute_intraday_vwap")
    @patch("utils.nifty_live_sentinel.get_live_nifty_oi_summary")
    def test_missing_quote_depth_recorded_safely(
        self, mock_oi, mock_vwap, mock_sound, mock_alert
    ):
        """Verifies when bid/ask are absent from live feed, candidate is logged as missing depth and executes."""
        mock_vwap.return_value = {"vwap": 24500.0, "spot": 24520.0, "is_available": True, "status": "ABOVE_VWAP"}
        # Feed returns LTP only, bid/ask are 0 or None
        mock_oi.return_value = {
            "spot": 24520.0,
            "atm_ce_ltp": 120.0,
            "atm_pe_ltp": 110.0,
            "atm_ce_bid": 0.0,
            "atm_ce_ask": 0.0,
            "source": "GROWW_BULK_CHAIN (LTP ONLY)",
        }

        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            from datetime import datetime, timezone, timedelta
            IST = timezone(timedelta(hours=5, minutes=30))
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 0, tzinfo=IST)

            state = self.engine.run_tick()

        self.assertIsNotNone(self.engine.active_position)
        records = self.shadow_logger._load_log()
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertFalse(rec["has_observed_depth"])
        self.assertIsNone(rec["bid"])
        self.assertIsNone(rec["ask"])
        self.assertEqual(rec["spread_research_result"], "UNOBSERVABLE")
        self.assertEqual(rec["guard_decision"], "UNOBSERVABLE")
        self.assertIn("UNOBSERVABLE", rec["guard_rejection_reason"])

    @patch("utils.nifty_live_sentinel.send_telegram_alert")
    @patch("utils.nifty_live_sentinel.play_sound")
    @patch("utils.nifty_live_sentinel.compute_intraday_vwap")
    @patch("utils.nifty_live_sentinel.get_live_nifty_oi_summary")
    def test_ltp_only_groww_candidate_alert_unchanged_and_spread_unobservable(
        self, mock_oi, mock_vwap, mock_sound, mock_alert
    ):
        """
        Proves an LTP-only Groww candidate:
          1. Produces an unchanged baseline Telegram alert and enters trade at LTP.
          2. Produces a spread-research result of 'UNOBSERVABLE' (no synthetic inference).
          3. Leaves bid, ask, and spread as None.
          4. Records quote freshness as 'UNKNOWN' when timestamp is absent.
        """
        mock_vwap.return_value = {"vwap": 24500.0, "spot": 24520.0, "is_available": True, "status": "ABOVE_VWAP"}
        # Real-world Groww bulk feed format: LTP present, bestBid/bestAsk absent (0.0/None), no exchange quote timestamp
        mock_oi.return_value = {
            "spot": 24520.0,
            "atm_ce_ltp": 120.0,
            "atm_pe_ltp": 110.0,
            "atm_ce_bid": 0.0,
            "atm_ce_ask": 0.0,
            "atm_pe_bid": 0.0,
            "atm_pe_ask": 0.0,
            "source": "GROWW_OPTION_CHAIN (Live Feed)",
            "quote_timestamp": None,
        }

        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            from datetime import datetime, timezone, timedelta
            IST = timezone(timedelta(hours=5, minutes=30))
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 0, tzinfo=IST)

            self.engine.run_tick()

        # 1. Baseline trade enters and alert fires UNCHANGED
        self.assertIsNotNone(self.engine.active_position, "Baseline trade must enter at LTP")
        self.assertEqual(self.engine.active_position["entry_premium"], 120.0)
        self.assertTrue(mock_alert.called, "Baseline Telegram alert must fire")
        alert_text = mock_alert.call_args[0][0]
        self.assertIn("NIFTY 24500 CE", alert_text)
        self.assertIn("₹120.00", alert_text)

        # 2. Spread outcome is explicitly UNOBSERVABLE
        records = self.shadow_logger._load_log()
        self.assertEqual(len(records), 1)
        rec = records[0]

        self.assertEqual(rec["baseline_decision"], "EXECUTE")
        self.assertEqual(rec["spread_research_result"], "UNOBSERVABLE")
        self.assertEqual(rec["guard_decision"], "UNOBSERVABLE")
        self.assertIn("UNOBSERVABLE", rec["guard_rejection_reason"])
        self.assertFalse(rec["has_observed_depth"])
        self.assertIsNone(rec["bid"])
        self.assertIsNone(rec["ask"])
        self.assertIsNone(rec["spread_abs"])
        self.assertIsNone(rec["spread_pct"])
        self.assertEqual(rec["quote_freshness"], "UNKNOWN")

    def test_outcome_tracking_updates_matching_candidate(self):
        """Verifies trade close updates the identical baseline exit outcome on the shadow record."""
        # Manually create a shadow candidate entry
        cand_opp = {
            "ticker": NIFTY_TICKER, "ticker_name": "NIFTY", "contract": "NIFTY 24500 CE",
            "atm_strike": 24500, "option_type": "CE", "direction": "UP",
            "entry_premium": 120.0, "bid": 119.80, "ask": 120.20,
            "quote_provenance": "GROWW_LIVE",
        }
        spread_eval = {"is_tradable": True, "spread_abs": 0.40, "spread_pct": 0.33, "quote_age_sec": 1.0}
        logged = self.shadow_logger.log_opportunity(cand_opp, spread_eval)
        cid = logged["candidate_id"]

        # Simulate position close in LiveSentinelEngine
        pos = {
            "contract": "NIFTY 24500 CE",
            "entry_premium": 120.0,
            "exit_price": 156.0,
            "exit_reason": "TARGET_HIT (+30%)",
            "exit_time": "2026-10-23 10:15:00",
            "pnl_rs": 2340.0,
            "pnl_pct": 30.0,
            "lots": 1,
            "shadow_candidate_id": cid,
        }
        self.engine.active_position = pos

        with patch("utils.nifty_live_sentinel.send_telegram_alert"), patch("utils.nifty_live_sentinel.play_sound"):
            self.engine._close_position(pos, is_win=True)

        # Check that shadow record was updated with the measured outcome
        records = self.shadow_logger._load_log()
        rec = records[0]
        self.assertEqual(rec["outcome_status"], "MEASURED")
        self.assertEqual(rec["subsequent_exit_price"], 156.0)
        self.assertEqual(rec["subsequent_exit_reason"], "TARGET_HIT (+30%)")
        self.assertEqual(rec["actual_pnl_rs"], 2340.0)
        self.assertEqual(rec["actual_pnl_pct"], 30.0)

    def test_unmeasurable_outcome_flagged(self):
        """Verifies open opportunity with no exit quote can be flagged unmeasurable."""
        cand_opp = {"contract": "NIFTY 24500 CE", "entry_premium": 120.0}
        spread_eval = {"is_tradable": False, "rejection_reason": "WIDE_SPREAD", "spread_abs": 3.0}
        logged = self.shadow_logger.log_opportunity(cand_opp, spread_eval)
        cid = logged["candidate_id"]

        # Mark unmeasurable
        self.shadow_logger.mark_unmeasurable(cid, reason="FEED_ABRUPTLY_DISCONNECTED")

        records = self.shadow_logger._load_log()
        rec = records[0]
        self.assertEqual(rec["outcome_status"], "UNMEASURABLE")
        self.assertEqual(rec["unmeasurable_reason"], "FEED_ABRUPTLY_DISCONNECTED")
        self.assertIsNone(rec["subsequent_exit_price"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
