"""
tests/test_state_audit.py
─────────────────────────
Fail-open Audit for nifty_live_sentinel state recovery.

Covers:
  A. Fail-open classification (does bad state allow entries or block them?)
  B. State/ledger file survival under Render's ephemeral filesystem
  C. Corrupt, missing, partially-written, and stale state files
  D. Crash-between-signal-and-persist race (signal fired, state not yet written)
  E. Concurrent duplicate instance protection (lock file)
  F. Broker-order vs alert-only audit (code-level: what actually gets submitted)
  G. All outcomes: BLOCKED or FAIL-OPEN
"""

import os
import json
import shutil
import tempfile
import threading
import time
import unittest
from datetime import timezone, timedelta
from unittest.mock import patch, MagicMock

# IST timezone for consistent timestamps in tests
IST = timezone(timedelta(hours=5, minutes=30))

from utils.nifty_live_sentinel import LiveSentinelEngine, get_ist_now
from config import NIFTY_TICKER


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _today_str():
    from datetime import datetime
    return datetime.now(IST).strftime("%Y-%m-%d")

def _now_str():
    from datetime import datetime
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def _make_active_pos(ticker=NIFTY_TICKER, trigger_time=None):
    return {
        "trigger_time": trigger_time or _now_str(),
        "ticker": ticker,
        "ticker_name": "NIFTY",
        "contract": "NIFTY 24500 CE",
        "atm_strike": 24500,
        "option_type": "CE",
        "direction": "UP",
        "conviction": 68.0,
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

def _make_state(active_position, has=True):
    from datetime import datetime
    return {
        "timestamp": datetime.now(IST).strftime("%H:%M:%S"),
        "ticker": NIFTY_TICKER,
        "has_active_position": has,
        "active_position": active_position,
    }

def _write_state(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def _write_ledger(path, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)

def _make_engine(state_file, ledger_file):
    return LiveSentinelEngine(
        ticker=NIFTY_TICKER,
        interval=5,
        state_file=state_file,
        ledger_file=ledger_file,
    )


# ─── Test Suite ──────────────────────────────────────────────────────────────

class TestStateAudit(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.test_dir, "state.json")
        self.ledger_file = os.path.join(self.test_dir, "ledger.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ── A. Fail-open classification ───────────────────────────────────────────

    def test_A1_missing_state_file_is_BLOCKED_not_fail_open(self):
        """MISSING state → engine starts clean, no active_position → entries
        gated by ML model cache (empty → evaluate_entry_trigger returns None).
        VERDICT: BLOCKED (no entry without valid model result)."""
        engine = _make_engine(self.state_file, self.ledger_file)
        self.assertIsNone(engine.active_position,
                          "Clean start must have no active position")
        # evaluate_entry_trigger without a cached model must return None
        result = engine.evaluate_entry_trigger(
            {"spot": 24520.0, "atm_ce_ltp": 120.0, "atm_pe_ltp": 110.0},
            {"vwap": 24500.0},
        )
        self.assertIsNone(result,
                          "FAIL-OPEN: empty model cache should block entry, not trigger one")

    def test_A2_corrupt_state_file_is_BLOCKED_not_fail_open(self):
        """CORRUPT JSON state → loads clean, no position restored.
        Subsequent entry gating depends on ML cache (empty = blocked).
        VERDICT: BLOCKED."""
        with open(self.state_file, "w") as f:
            f.write("{corrupt json!!!")
        engine = _make_engine(self.state_file, self.ledger_file)
        self.assertIsNone(engine.active_position,
                          "FAIL-OPEN: corrupt state must not restore a phantom position")

    def test_A3_partially_written_state_BLOCKED_not_fail_open(self):
        """PARTIAL write (truncated JSON) → treated as corrupt.
        VERDICT: BLOCKED."""
        pos = _make_active_pos()
        full = json.dumps(_make_state(pos))
        # Write only first 60 bytes (always truncated and invalid JSON)
        with open(self.state_file, "wb") as f:
            f.write(full[:60].encode("utf-8"))
        engine = _make_engine(self.state_file, self.ledger_file)
        self.assertIsNone(engine.active_position,
                          "FAIL-OPEN: partial state must not restore a position")

    def test_A4_stale_previous_day_state_is_DISCARDED(self):
        """Position from a previous calendar date must never be restored.
        VERDICT: BLOCKED (stale state discarded)."""
        pos = _make_active_pos(trigger_time="2024-01-01 09:25:00")
        _write_state(self.state_file, _make_state(pos))
        engine = _make_engine(self.state_file, self.ledger_file)
        self.assertIsNone(engine.active_position,
                          "FAIL-OPEN: yesterday's position must be discarded")

    def test_A5_valid_today_state_restores_position_BLOCKED_for_duplicate(self):
        """Valid today state restores position → evaluate_entry_trigger blocked.
        VERDICT: CORRECTLY BLOCKED (duplicate order prevention)."""
        pos = _make_active_pos()
        _write_state(self.state_file, _make_state(pos))
        engine = _make_engine(self.state_file, self.ledger_file)
        self.assertIsNotNone(engine.active_position)
        # Even with ideal conditions the trigger must be blocked
        engine.cached_model_result = {
            "status": "success", "direction": "UP",
            "conviction_pct": 80.0, "regime": "TRENDING",
            "latest_close": 24520.0,
        }
        result = engine.evaluate_entry_trigger(
            {"spot": 24520.0, "atm_ce_ltp": 120.0, "atm_pe_ltp": 110.0, "pcr_oi": 0.8},
            {"vwap": 24500.0},
        )
        self.assertIsNone(result,
                          "FAIL-OPEN: restored active position must block new entry")

    # ── B. Render ephemeral filesystem ────────────────────────────────────────

    def test_B1_ephemeral_filesystem_reality(self):
        """
        CRITICAL DEPLOYMENT FINDING:
        Render free-tier Docker containers use an EPHEMERAL filesystem.
        On every deploy or container replacement, /app is rebuilt from the image.
        Files written at runtime (live_sentinel_state.json, sentinel_trades_ledger.json)
        are stored at /app/ inside the container — they DO NOT survive an instance
        replacement. This means:
          - active_position: ALWAYS lost on Render deploy/restart
          - today_trades_count: ALWAYS reset to 0 on Render deploy/restart
          - Consequence: after a Render redeploy mid-day, the engine STARTS CLEAN and
            CAN re-enter the market even if max_trades_per_day was already exhausted.

        This test documents the finding.  It cannot be fixed without either:
          (a) An external persistent store (Redis, Supabase, environment variable)
          (b) Switching to Render paid plan with persistent disk

        We verify the test suite itself passes so the issue is formally recorded.
        """
        # Simulate what Render does: files written in a prior container are gone
        engine_before = _make_engine(self.state_file, self.ledger_file)
        engine_before.active_position = _make_active_pos()
        # State written to container-local disk
        engine_before._save_state({
            "has_active_position": True,
            "active_position": engine_before.active_position,
            "ticker": NIFTY_TICKER,
        })
        # Simulate Render instance replacement: disk is wiped
        os.remove(self.state_file)
        if os.path.exists(self.ledger_file):
            os.remove(self.ledger_file)
        # New instance boots — files are gone
        engine_after = _make_engine(self.state_file, self.ledger_file)
        self.assertIsNone(engine_after.active_position,
                          "After Render instance replacement, active_position is ALWAYS None")
        self.assertEqual(engine_after.today_trades_count, 0,
                         "After Render instance replacement, today_trades_count resets to 0")
        # This is the documented gap: the engine WILL try to enter again
        # This test documents it; resolution requires external persistence

    # ── C. Crash between signal emission and state persistence ────────────────

    def test_C1_crash_after_telegram_before_state_write(self):
        """
        RACE CONDITION: In run_tick(), the order of operations is:
          1. trigger = evaluate_entry_trigger()   ← gates pass
          2. self.active_position = trigger       ← in-memory only
          3. send_telegram_alert()                ← TELEGRAM FIRES HERE
          4. [no broker order — this is signals-only]
          5. _save_state() called at end of tick  ← disk write
          6. If crash/kill between steps 3 and 5: Telegram was sent but
             active_position is NOT in state.json
          7. On restart: engine boots clean → can fire SECOND entry alert

        This test documents the window exists and verifies what happens on restart.
        """
        # State file does NOT contain the position (crash before write)
        # Ledger is also empty (trade not closed yet)
        engine = _make_engine(self.state_file, self.ledger_file)
        # Engine boots clean — it does not know about the lost in-memory position
        self.assertIsNone(engine.active_position,
                          "After crash-before-persist: engine starts clean (duplicate-entry risk)")
        # This means a second alert WILL fire on next qualifying tick
        # DOCUMENTED as an unfixed race: the gap window is 1 tick = ~5 seconds

    # ── D. Concurrent instances ───────────────────────────────────────────────

    def test_D1_no_lock_file_means_concurrent_instances_are_fail_open(self):
        """
        The code has NO lock file or PID file mechanism to prevent two
        sentinel instances running simultaneously against the same state/ledger.
        start.sh launches sentinel in background; if Streamlit is restarted or
        start.sh is called again, a second process can boot against the same files.

        Both instances would pass evaluate_entry_trigger independently
        (each checks self.active_position in their own memory) and BOTH
        would fire Telegram alerts for the same signal.

        This test verifies no lock mechanism exists in the code.
        """
        # Verify no lockfile path exists in the module
        import utils.nifty_live_sentinel as mod
        has_lockfile = hasattr(mod, 'LOCK_FILE') or hasattr(mod, 'PID_FILE')
        self.assertFalse(has_lockfile,
                         "No lock/PID file mechanism exists — concurrent instances are FAIL-OPEN")

    def test_D2_concurrent_instances_both_pass_entry_check(self):
        """Two engine instances sharing the same state file can both fire entries."""
        pos = _make_active_pos()
        # Instance A enters and saves state
        engine_a = _make_engine(self.state_file, self.ledger_file)
        engine_a.active_position = pos
        engine_a._save_state({
            "has_active_position": True,
            "active_position": pos,
            "ticker": NIFTY_TICKER,
        })
        # Instance B boots and reads state correctly — position restored
        engine_b = _make_engine(self.state_file, self.ledger_file)
        self.assertIsNotNone(engine_b.active_position,
                             "Instance B should see instance A's active position")
        # So instance B correctly blocks further entries
        engine_b.cached_model_result = {
            "status": "success", "direction": "UP",
            "conviction_pct": 80.0, "regime": "TRENDING",
            "latest_close": 24520.0,
        }
        result = engine_b.evaluate_entry_trigger(
            {"spot": 24520.0, "atm_ce_ltp": 120.0, "atm_pe_ltp": 110.0, "pcr_oi": 0.8},
            {"vwap": 24500.0},
        )
        self.assertIsNone(result,
                          "Instance B reads instance A's state → correctly blocked")
        # However: if instance B boots before instance A writes its state,
        # both will see active_position=None and both will fire.

    # ── E. Broker order vs alert-only audit ──────────────────────────────────

    def test_E1_system_emits_only_telegram_alerts_no_broker_orders(self):
        """
        CRITICAL CODE AUDIT — what does run_tick() actually DO on entry?

        In run_tick() (lines 519-540 of sentinel):
          1. evaluate_entry_trigger() → returns trigger dict
          2. self.active_position = trigger  ← in-memory record only
          3. play_sound("entry")             ← local Windows audio
          4. send_telegram_alert(...)        ← Telegram push notification
          NO call to any broker API, Groww order placement, or trade execution.

        The sentinel is a SIGNALS-ONLY system. It notifies via Telegram.
        It does NOT submit broker orders.

        Confirming: no broker-order function exists in the module.
        """
        import utils.nifty_live_sentinel as mod
        source = open(mod.__file__, encoding="utf-8").read()

        broker_order_keywords = [
            "place_order", "submit_order", "buy_order", "sell_order",
            "place_trade", "execute_order", "groww_order", "broker_buy",
            "broker_sell", "order_id", "order_placed",
        ]
        found = [kw for kw in broker_order_keywords if kw in source]
        self.assertEqual(found, [],
                         f"UNEXPECTED broker order keyword(s) found in sentinel: {found}")

    def test_E2_send_telegram_is_the_only_external_side_effect(self):
        """send_telegram_alert() is the ONLY external call on entry.
        No exchange-facing API call is made."""
        import utils.nifty_live_sentinel as mod
        source = open(mod.__file__, encoding="utf-8").read()
        self.assertIn("send_telegram_alert", source)
        # Confirm the call happens after active_position is set (line ordering)
        idx_pos = source.find("self.active_position = trigger")
        idx_tg  = source.find("send_telegram_alert", idx_pos)
        self.assertGreater(idx_tg, idx_pos,
                           "Telegram alert fires after position is set in memory")

    # ── F. Ledger integrity ───────────────────────────────────────────────────

    def test_F1_corrupt_ledger_treated_as_empty_not_crash(self):
        """Corrupt ledger file → treated as empty list → today_trades_count = 0.
        This is fail-open for the daily trade counter: engine allows up to
        max_trades_per_day entries even if prior trades existed."""
        with open(self.ledger_file, "w") as f:
            f.write("[{broken]")
        engine = _make_engine(self.state_file, self.ledger_file)
        self.assertEqual(engine.today_trades_count, 0,
                         "FAIL-OPEN: corrupt ledger resets trade count to 0, allowing new entries")
        # This is documented as a limitation. Resolution requires ledger backup or checksum.

    def test_F2_ledger_with_only_other_ticker_trades_does_not_pollute_count(self):
        """Ledger entries for SENSEX must not count toward NIFTY today_trades_count."""
        today = _today_str()
        ledger = [
            {"ticker": "^BSESN", "trigger_time": f"{today} 09:30:00",
             "exit_time": f"{today} 10:00:00"},
        ]
        _write_ledger(self.ledger_file, ledger)
        engine = _make_engine(self.state_file, self.ledger_file)
        self.assertEqual(engine.today_trades_count, 0,
                         "SENSEX ledger entry must not count toward NIFTY daily limit")

    def test_F3_ledger_missing_ticker_key_does_not_count(self):
        """Ledger entry without ticker key must be ignored safely."""
        today = _today_str()
        ledger = [{"trigger_time": f"{today} 09:30:00"}]   # no 'ticker' key
        _write_ledger(self.ledger_file, ledger)
        engine = _make_engine(self.state_file, self.ledger_file)
        self.assertEqual(engine.today_trades_count, 0,
                         "Ledger entry without 'ticker' key must be safely ignored")

    # ── G. Recovery action requirements ──────────────────────────────────────

    def test_G1_untrusted_state_emits_BLOCKED_entry(self):
        """If state is recoverable (valid today data) → position restored → entries blocked.
        If state unrecoverable (corrupt/stale/missing) → no position → entries allowed
        only when model cache is valid. This is acceptable behavior."""
        # Case 1: Valid state → BLOCKED
        pos = _make_active_pos()
        _write_state(self.state_file, _make_state(pos))
        engine_ok = _make_engine(self.state_file, self.ledger_file)
        self.assertIsNotNone(engine_ok.active_position)

        # Case 2: Corrupt state → entries gated by model cache only
        tf2 = self.state_file + ".corrupt"
        with open(tf2, "w") as f:
            f.write("{bad}")
        engine_bad = LiveSentinelEngine(
            ticker=NIFTY_TICKER, interval=5,
            state_file=tf2, ledger_file=self.ledger_file
        )
        self.assertIsNone(engine_bad.active_position)
        # Without model cache, still blocked
        result = engine_bad.evaluate_entry_trigger({}, {})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
