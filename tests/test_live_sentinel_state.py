"""
tests/test_live_sentinel_state.py
Unit tests for Sentinel crash recovery, duplicate order prevention, and daily trade limit discipline.
Verifies normal behavior, crash recovery, duplicate-order rejection, and failure/corruption handling.
"""

import os
import json
import shutil
import tempfile
import unittest
from datetime import datetime, time as dtime

from utils.nifty_live_sentinel import LiveSentinelEngine, get_ist_now
from config import NIFTY_TICKER


class TestLiveSentinelStateRecovery(unittest.TestCase):
    def setUp(self):
        # Create isolated temporary directory for test state and ledger files
        self.test_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.test_dir, "test_state.json")
        self.ledger_file = os.path.join(self.test_dir, "test_ledger.json")

    def tearDown(self):
        # Clean up temporary test files
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_clean_startup_without_existing_files(self):
        """Verifies clean startup when no state or ledger files exist."""
        engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file
        )
        self.assertIsNone(engine.active_position)
        self.assertEqual(engine.today_trades_count, 0)
        self.assertEqual(len(engine.trades_history), 0)

    def test_crash_recovery_restores_active_position(self):
        """Verifies active position is restored across process restart, preventing trade abandonment."""
        now = get_ist_now()
        today_str = now.strftime("%Y-%m-%d %H:%M:%S")

        mock_active_position = {
            "trigger_time": today_str,
            "ticker": NIFTY_TICKER,
            "ticker_name": "NIFTY",
            "contract": "NIFTY 24500 CE",
            "atm_strike": 24500,
            "option_type": "CE",
            "direction": "UP",
            "conviction": 65.0,
            "spot": 24520.0,
            "entry_premium": 120.0,
            "target_premium": 156.0,
            "sl_premium": 102.0,
            "trail_trigger": 141.6,
            "lock_trigger": 150.0,
            "locked_sl": 138.0,
            "lots": 1,
            "total_qty": 65,
            "trail_state": "INITIAL",
            "current_sl": 102.0,
            "pnl_rs": 0.0,
            "pnl_pct": 0.0,
            "high_premium_seen": 120.0,
        }

        # Write state file simulating active trade before a crash/restart
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": now.strftime("%H:%M:%S"),
                "ticker": NIFTY_TICKER,
                "has_active_position": True,
                "active_position": mock_active_position
            }, f)

        # Boot new engine instance (simulating restart)
        engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file
        )

        self.assertIsNotNone(engine.active_position)
        self.assertEqual(engine.active_position["contract"], "NIFTY 24500 CE")
        self.assertEqual(engine.active_position["entry_premium"], 120.0)
        self.assertEqual(engine.active_position["current_sl"], 102.0)

    def test_duplicate_order_prevention_when_position_is_active(self):
        """Verifies evaluate_entry_trigger rejects new entry if a position is already active."""
        engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file
        )

        # Manually assign an active position
        engine.active_position = {"contract": "NIFTY 24500 CE", "entry_premium": 120.0}

        # Simulate ideal market conditions that would otherwise trigger entry
        oi_mock = {"spot": 24520.0, "atm_ce_ltp": 120.0, "atm_pe_ltp": 110.0, "pcr_oi": 0.8}
        vwap_mock = {"vwap": 24500.0, "status": "ABOVE_VWAP"}
        engine.cached_model_result = {
            "status": "success",
            "direction": "UP",
            "conviction_pct": 75.0,
            "regime": "TRENDING",
            "latest_close": 24520.0
        }

        # Trigger check must strictly return None to prevent duplicate orders
        trigger = engine.evaluate_entry_trigger(oi_mock, vwap_mock)
        self.assertIsNone(trigger)

    def test_ledger_synchronization_recovers_daily_trades_count(self):
        """Verifies daily trades count is restored from disk ledger across restarts."""
        today_date = get_ist_now().strftime("%Y-%m-%d")

        mock_ledger = [
            # Trade 1 from today
            {
                "ticker": NIFTY_TICKER,
                "contract": "NIFTY 24500 CE",
                "trigger_time": f"{today_date} 09:25:00",
                "exit_time": f"{today_date} 10:15:00",
                "entry_premium": 120.0,
                "exit_price": 156.0,
                "exit_reason": "TARGET_HIT (+30%)"
            },
            # Trade 2 from today
            {
                "ticker": NIFTY_TICKER,
                "contract": "NIFTY 24550 CE",
                "trigger_time": f"{today_date} 11:30:00",
                "exit_time": f"{today_date} 12:05:00",
                "entry_premium": 115.0,
                "exit_price": 97.75,
                "exit_reason": "SL_HIT (INITIAL)"
            },
            # Trade from yesterday (should NOT be counted towards today)
            {
                "ticker": NIFTY_TICKER,
                "contract": "NIFTY 24400 PE",
                "trigger_time": "2026-01-01 09:30:00",
                "exit_time": "2026-01-01 10:00:00",
                "entry_premium": 100.0,
                "exit_price": 130.0,
                "exit_reason": "TARGET_HIT (+30%)"
            }
        ]

        with open(self.ledger_file, "w", encoding="utf-8") as f:
            json.dump(mock_ledger, f)

        # Boot engine on restart
        engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file
        )

        # Engine must know 2 trades were executed today
        self.assertEqual(engine.today_trades_count, 2)
        # Entry trigger must reject any new entry because max_trades_per_day (2) is reached
        trigger = engine.evaluate_entry_trigger({"spot": 24500.0}, {"vwap": 24500.0})
        self.assertIsNone(trigger)

    def test_stale_previous_day_state_is_safely_discarded(self):
        """Verifies active position from a previous date is discarded on startup."""
        stale_position = {
            "trigger_time": "2026-01-01 09:25:00",  # past date
            "ticker": NIFTY_TICKER,
            "contract": "NIFTY 24500 CE",
            "entry_premium": 120.0
        }

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": "09:25:00",
                "has_active_position": True,
                "active_position": stale_position
            }, f)

        engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file
        )

        # Stale position must be ignored
        self.assertIsNone(engine.active_position)

    def test_malformed_json_resilience(self):
        """Verifies resilience against corrupted state or ledger JSON files without crashing."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            f.write("{ corrupt json ...")

        with open(self.ledger_file, "w", encoding="utf-8") as f:
            f.write("invalid content")

        engine = LiveSentinelEngine(
            ticker=NIFTY_TICKER,
            interval=5,
            state_file=self.state_file,
            ledger_file=self.ledger_file
        )

        # Must initialize safely with clean defaults
        self.assertIsNone(engine.active_position)
        self.assertEqual(engine.today_trades_count, 0)


if __name__ == "__main__":
    unittest.main()
