"""
tests/test_end_to_end_session_audit.py
──────────────────────────────────────
End-to-End Non-Live Session Audit for System 2:
  1. Executes a complete baseline session from signal generation through paper exit.
  2. Verifies contract identity consistency across state, ledger, and shadow log.
  3. Verifies timestamp alignment and monotonicity.
  4. Audits for missing/stale data, duplicate signals, and incomplete exits.
  5. Verifies alert and risk rule preservation (+30% Target, -15% SL, Trailing).
"""

import os
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from config import NIFTY_TICKER
from utils.nifty_live_sentinel import LiveSentinelEngine, get_ist_now
from utils.spread_research_logger import SpreadResearchLogger

IST = timezone(timedelta(hours=5, minutes=30))


class TestEndToEndSessionAudit(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.test_dir, "live_sentinel_state.json")
        self.ledger_file = os.path.join(self.test_dir, "sentinel_trades_ledger.json")
        self.shadow_log_file = os.path.join(self.test_dir, "spread_guard_shadow_log.json")
        self.shadow_logger = SpreadResearchLogger(log_file=self.shadow_log_file)

        self.engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file,
            shadow_logger=self.shadow_logger
        )
        self.engine.max_trades_per_day = 1
        self.engine.last_slow_refresh = 9999999999.0
        self.engine.cached_groww_account = {"cash_available": 25000.0}  # 1 lot allowed
        self.engine.cached_vix = {"level": 14.5, "status": "LIVE"}
        self.engine.cached_model_result = {
            "status": "success",
            "direction": "UP",
            "conviction_pct": 74.0,
            "regime": "TRENDING",
            "latest_close": 24520.0,
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("utils.nifty_live_sentinel.send_telegram_alert")
    @patch("utils.nifty_live_sentinel.play_sound")
    @patch("utils.nifty_live_sentinel.compute_intraday_vwap")
    @patch("utils.nifty_live_sentinel.get_live_nifty_oi_summary")
    def test_full_session_from_signal_through_paper_exit(
        self, mock_oi, mock_vwap, mock_sound, mock_alert
    ):
        """
        Executes an end-to-end unchanged baseline session:
          t0: 09:25:00 - Entry signal fires @ LTP ₹120.00
          t1: 09:25:05 - Active position tick @ LTP ₹128.00 (+6.7%)
          t2: 09:25:10 - Trail trigger hit (+18%) @ LTP ₹142.00 -> SL moves to cost (₹120.00)
          t3: 09:25:15 - Target hit (+30%) @ LTP ₹156.50 -> Position closed & ledger saved
          t4: 09:25:20 - Next tick: duplicate order prevention & daily discipline enforced
        """
        mock_vwap.return_value = {"vwap": 24500.0, "spot": 24520.0, "is_available": True, "status": "ABOVE_VWAP"}

        # ── Tick 0: 09:25:00 IST — Candidate Entry Trigger ──────────────────
        mock_oi.return_value = {
            "spot": 24520.0,
            "atm_ce_ltp": 120.0,
            "atm_pe_ltp": 110.0,
            "atm_ce_bid": 0.0,
            "atm_ce_ask": 0.0,
            "pcr_oi": 0.85,
            "source": "GROWW_OPTION_CHAIN (Live Feed)",
            "quote_timestamp": None,
        }
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 0, tzinfo=IST)
            state_t0 = self.engine.run_tick()

        # Audit t0 entry state
        self.assertIsNotNone(self.engine.active_position, "Position must be active at t0")
        pos = self.engine.active_position
        self.assertEqual(pos["contract"], "NIFTY 24500 CE")
        self.assertEqual(pos["entry_premium"], 120.0)
        self.assertEqual(pos["target_premium"], 156.0)     # +30%
        self.assertEqual(pos["sl_premium"], 102.0)         # -15%
        self.assertEqual(pos["trail_trigger"], 141.6)      # +18%
        self.assertEqual(pos["trail_state"], "INITIAL")
        self.assertEqual(pos["lots"], 1)
        self.assertEqual(pos["total_qty"], 65)

        # Audit t0 alert
        self.assertTrue(mock_alert.called, "Telegram alert must fire on entry")
        entry_alert = mock_alert.call_args_list[0][0][0]
        self.assertIn("NIFTY 24500 CE", entry_alert)
        self.assertIn("₹120.00", entry_alert)
        self.assertIn("₹156.00", entry_alert)

        # ── Tick 1: 09:25:05 IST — Subsequent Monitoring ────────────────────
        mock_oi.return_value["atm_ce_ltp"] = 128.00
        mock_oi.return_value["spot"] = 24535.0
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 5, tzinfo=IST)
            state_t1 = self.engine.run_tick()

        self.assertEqual(pos["current_ltp"], 128.00)
        self.assertEqual(pos["pnl_rs"], (128.00 - 120.00) * 65)
        self.assertEqual(pos["trail_state"], "INITIAL")

        # ── Tick 2: 09:25:10 IST — Trailing Trigger (+18%) ──────────────────
        mock_oi.return_value["atm_ce_ltp"] = 142.00
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 10, tzinfo=IST)
            state_t2 = self.engine.run_tick()

        self.assertEqual(pos["trail_state"], "TRAILED_TO_COST")
        self.assertEqual(pos["current_sl"], 120.00, "SL must trail to cost price at +18%")

        # ── Tick 3: 09:25:15 IST — Target Hit (+30%) & Position Close ────────
        mock_oi.return_value["atm_ce_ltp"] = 156.50
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 15, tzinfo=IST)
            state_t3 = self.engine.run_tick()

        # Active position must now be closed
        self.assertIsNone(self.engine.active_position, "Position must be None after close")
        self.assertEqual(self.engine.today_trades_count, 1, "Today trade count must be 1")

        # ── Tick 4: 09:25:20 IST — Duplicate Prevention / Daily Cap ─────────
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 20, tzinfo=IST)
            state_t4 = self.engine.run_tick()

        self.assertIsNone(self.engine.active_position, "Must not re-enter on same day")
        self.assertEqual(self.engine.today_trades_count, 1)

        # ─────────────────────────────────────────────────────────────────────
        # Multi-File Audit of Resulting Data Records
        # ─────────────────────────────────────────────────────────────────────

        # 1. Audit Closed Trades Ledger (sentinel_trades_ledger.json)
        self.assertTrue(os.path.exists(self.ledger_file), "Ledger file must exist")
        with open(self.ledger_file, "r", encoding="utf-8") as f:
            ledger_data = json.load(f)

        self.assertEqual(len(ledger_data), 1, "Exactly 1 closed trade must be logged")
        trade = ledger_data[0]
        self.assertEqual(trade["contract"], "NIFTY 24500 CE")
        self.assertEqual(trade["ticker"], NIFTY_TICKER)
        self.assertEqual(trade["atm_strike"], 24500)
        self.assertEqual(trade["option_type"], "CE")
        self.assertEqual(trade["direction"], "UP")
        self.assertEqual(trade["entry_premium"], 120.00)
        self.assertEqual(trade["exit_price"], 156.50)
        self.assertEqual(trade["exit_reason"], "TARGET_HIT (+30%)")
        self.assertEqual(trade["pnl_rs"], round((156.50 - 120.00) * 65, 2))
        self.assertEqual(trade["lots"], 1)
        self.assertEqual(trade["total_qty"], 65)

        # 2. Audit Prospective Shadow Log (spread_guard_shadow_log.json)
        self.assertTrue(os.path.exists(self.shadow_log_file), "Shadow log file must exist")
        with open(self.shadow_log_file, "r", encoding="utf-8") as f:
            shadow_data = json.load(f)

        self.assertEqual(len(shadow_data), 1, "Exactly 1 shadow opportunity must be logged")
        shadow_rec = shadow_data[0]

        # Audit Contract Identity & Consistency
        self.assertEqual(shadow_rec["contract"], trade["contract"])
        self.assertEqual(shadow_rec["atm_strike"], trade["atm_strike"])
        self.assertEqual(shadow_rec["option_type"], trade["option_type"])
        self.assertEqual(shadow_rec["direction"], trade["direction"])
        self.assertEqual(shadow_rec["ltp"], trade["entry_premium"])
        self.assertEqual(shadow_rec["trigger_price"], trade["entry_premium"])

        # Audit Timestamps Alignment
        self.assertEqual(shadow_rec["timestamp"], "2026-10-23 09:25:00")
        self.assertEqual(shadow_rec["exit_timestamp"], "2026-10-23 09:25:15")
        self.assertGreater(shadow_rec["exit_timestamp"], shadow_rec["timestamp"])

        # Audit Exit Outcome Tracking
        self.assertEqual(shadow_rec["outcome_status"], "MEASURED")
        self.assertEqual(shadow_rec["subsequent_exit_price"], trade["exit_price"])
        self.assertEqual(shadow_rec["subsequent_exit_reason"], trade["exit_reason"])
        self.assertEqual(shadow_rec["actual_pnl_rs"], trade["pnl_rs"])

        # Audit Spread Research Classification (LTP-only feed)
        self.assertEqual(shadow_rec["spread_research_result"], "UNOBSERVABLE")
        self.assertEqual(shadow_rec["guard_decision"], "UNOBSERVABLE")
        self.assertFalse(shadow_rec["has_observed_depth"])
        self.assertIsNone(shadow_rec["bid"])
        self.assertIsNone(shadow_rec["ask"])
        self.assertIsNone(shadow_rec["spread_abs"])
        self.assertEqual(shadow_rec["quote_freshness"], "UNKNOWN")

        # Audit Tick-Persistence Sequence (Captured subsequent ticks)
        subsequent_ltps = shadow_rec["subsequent_observed_ltps"]
        self.assertGreaterEqual(len(subsequent_ltps), 2, "Must capture subsequent monitoring ticks")
        self.assertEqual(subsequent_ltps[0]["ltp"], 128.00)
        self.assertEqual(subsequent_ltps[0]["timestamp"], "2026-10-23 09:25:05")
        self.assertEqual(subsequent_ltps[1]["ltp"], 142.00)
        self.assertEqual(subsequent_ltps[1]["timestamp"], "2026-10-23 09:25:10")

        # 3. Audit Runtime Telemetry State (live_sentinel_state.json)
        self.assertTrue(os.path.exists(self.state_file), "State file must exist")
        with open(self.state_file, "r", encoding="utf-8") as f:
            state_data = json.load(f)

        self.assertFalse(state_data["has_active_position"])
        self.assertIsNone(state_data["active_position"])
        self.assertEqual(state_data["today_trades_count"], 1)

    @patch("utils.nifty_live_sentinel.send_telegram_alert")
    @patch("utils.nifty_live_sentinel.play_sound")
    @patch("utils.nifty_live_sentinel.compute_intraday_vwap")
    @patch("utils.nifty_live_sentinel.get_live_nifty_oi_summary")
    def test_default_max_trades_allows_immediate_reentry_without_cooldown(
        self, mock_oi, mock_vwap, mock_sound, mock_alert
    ):
        """
        Audits default LiveSentinelEngine configuration (max_trades_per_day = 2):
        Proves that after Trade 1 exits, Trade 2 enters on the very next 5-second tick
        if conditions still qualify, because there is no post-trade cooldown timer.
        """
        engine2 = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file + ".2",
            ledger_file=self.ledger_file + ".2",
            shadow_logger=self.shadow_logger
        )
        self.assertEqual(engine2.max_trades_per_day, 2, "Default max_trades_per_day is 2")
        engine2.last_slow_refresh = 9999999999.0
        engine2.cached_groww_account = {"cash_available": 25000.0}
        engine2.cached_model_result = self.engine.cached_model_result

        mock_vwap.return_value = {"vwap": 24500.0, "spot": 24520.0, "is_available": True, "status": "ABOVE_VWAP"}
        mock_oi.return_value = {"spot": 24520.0, "atm_ce_ltp": 120.0, "atm_pe_ltp": 110.0}

        # Trade 1 entry
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 0, tzinfo=IST)
            engine2.run_tick()
        self.assertIsNotNone(engine2.active_position)

        # Trade 1 exit
        mock_oi.return_value["atm_ce_ltp"] = 156.50
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 5, tzinfo=IST)
            engine2.run_tick()
        self.assertIsNone(engine2.active_position)
        self.assertEqual(engine2.today_trades_count, 1)

        # Immediate Next Tick: 5 seconds later at 09:25:10
        mock_oi.return_value["atm_ce_ltp"] = 150.00
        with patch("utils.nifty_live_sentinel.get_ist_now") as mock_now:
            mock_now.return_value = datetime(2026, 10, 23, 9, 25, 10, tzinfo=IST)
            engine2.run_tick()

        # Audit finding: Engine enters Trade 2 immediately without cooldown
        self.assertIsNotNone(engine2.active_position, "Engine enters Trade 2 immediately (no cooldown timer)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
