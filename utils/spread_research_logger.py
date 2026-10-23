"""
utils/spread_research_logger.py
───────────────────────────────
Prospective Shadow Logger for Spread Guard Research & Outcome Tracking.

Purpose:
  - Logs EVERY candidate opportunity detected by System 2 alongside its bid/ask depth,
    quote provenance, spread metrics, baseline decision, and counterfactual spread guard decision.
  - Dynamically updates the matching record with identical baseline exit outcomes
    (including for guard-rejected candidates) once positions exit.
  - Distinguishes observed quotes vs missing quotes and measured vs unmeasurable outcomes.
  - Pure non-live observation: does NOT alter live alerts or execute broker orders.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADOW_LOG_FILE = os.path.join(PROJECT_ROOT, "spread_guard_shadow_log.json")


class SpreadResearchLogger:
    """Non-live prospective logger for empirical spread distribution and counterfactual analysis."""

    def __init__(self, log_file: str = SHADOW_LOG_FILE):
        self.log_file = log_file

    def _load_log(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _save_log(self, records: List[Dict[str, Any]]) -> bool:
        """Atomic write to disk to prevent partial or corrupt logs."""
        target_dir = os.path.dirname(os.path.abspath(self.log_file))
        os.makedirs(target_dir, exist_ok=True)
        temp_file = f"{self.log_file}.tmp_{os.getpid()}_{time.time()}"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, self.log_file)
            return True
        except Exception:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            return False

    def log_opportunity(
        self,
        candidate_opportunity: Dict[str, Any],
        spread_eval: Dict[str, Any],
        baseline_decision: str = "EXECUTE"
    ) -> Dict[str, Any]:
        """
        Record a candidate observation at trigger time.
        
        Records:
          - candidate_id: unique tracking ID
          - Timestamp & Asset info
          - Bid, Ask, Spread, Spread %, Quote Age, Quote Provenance
          - has_observed_depth: True if authentic bid/ask exists, False if missing
          - Baseline decision vs Spread-Guard decision (PASS or REJECT)
          - Guard rejection reason if filtered
          - outcome_status: 'PENDING_EXIT' initially
        """
        timestamp_str = candidate_opportunity.get("trigger_time")
        if timestamp_str:
            cid_time_part = timestamp_str.replace("-", "").replace(":", "").replace(" ", "_")
        else:
            now = datetime.now(IST)
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            cid_time_part = now.strftime("%Y%m%d_%H%M%S")

        contract = candidate_opportunity.get("contract", "")
        cid = f"{cid_time_part}_{contract.replace(' ', '_')}"

        bid = float(candidate_opportunity.get("bid", 0.0) or 0.0)
        ask = float(candidate_opportunity.get("ask", 0.0) or 0.0)
        has_observed_depth = bool(bid > 0.0 and ask > 0.0 and ask >= bid)

        guard_passed = bool(spread_eval.get("is_tradable", False))

        # Set research result to UNOBSERVABLE if authentic bid/ask depth is absent
        if not has_observed_depth or spread_eval.get("spread_research_result") == "UNOBSERVABLE":
            spread_research_result = "UNOBSERVABLE"
            guard_decision = "UNOBSERVABLE"
            guard_reason = spread_eval.get("rejection_reason") or "UNOBSERVABLE (Authentic top-of-book bid/ask absent from feed)"
        else:
            spread_research_result = spread_eval.get("spread_research_result", "OBSERVED_VALID")
            guard_decision = "PASS" if guard_passed else "REJECT"
            guard_reason = spread_eval.get("rejection_reason")

        record = {
            "candidate_id": cid,
            "timestamp": timestamp_str,
            "ticker": candidate_opportunity.get("ticker", ""),
            "ticker_name": candidate_opportunity.get("ticker_name", ""),
            "contract": contract,
            "atm_strike": candidate_opportunity.get("atm_strike", 0),
            "option_type": candidate_opportunity.get("option_type", ""),
            "direction": candidate_opportunity.get("direction", ""),
            "regime": candidate_opportunity.get("regime", "UNKNOWN"),
            "conviction": candidate_opportunity.get("conviction", 0.0),
            "spot": candidate_opportunity.get("spot", 0.0),
            "vwap": candidate_opportunity.get("vwap", 0.0),
            "ltp": candidate_opportunity.get("entry_premium", 0.0),
            "bid": bid if bid > 0 else None,
            "ask": ask if ask > 0 else None,
            "has_observed_depth": has_observed_depth,
            "spread_research_result": spread_research_result,
            "quote_provenance": candidate_opportunity.get("quote_provenance", "UNKNOWN_FEED"),
            "quote_freshness": spread_eval.get("quote_freshness", "UNKNOWN"),
            "spread_abs": spread_eval.get("spread_abs"),
            "spread_pct": spread_eval.get("spread_pct"),
            "quote_age_sec": spread_eval.get("quote_age_sec"),
            "trigger_price": candidate_opportunity.get("entry_premium", 0.0),
            "subsequent_observed_ltps": [],
            "baseline_decision": baseline_decision,
            "guard_decision": guard_decision,
            "guard_rejection_reason": guard_reason,
            "outcome_status": "PENDING_EXIT",
            "exit_timestamp": None,
            "subsequent_exit_price": None,
            "subsequent_exit_reason": None,
            "actual_pnl_rs": None,
            "actual_pnl_pct": None,
        }

        records = self._load_log()
        records.append(record)
        self._save_log(records)
        return record

    def record_subsequent_ltp(self, candidate_id: str, timestamp: str, ltp: float) -> bool:
        """Appends contemporaneous subsequent observed LTP to candidate record for persistence research."""
        records = self._load_log()
        updated = False
        for r in records:
            if r.get("candidate_id") == candidate_id:
                if "subsequent_observed_ltps" not in r:
                    r["subsequent_observed_ltps"] = []
                r["subsequent_observed_ltps"].append({
                    "timestamp": timestamp,
                    "ltp": round(float(ltp), 2)
                })
                updated = True
                break
        if updated:
            self._save_log(records)
        return updated

    def update_outcome(
        self,
        candidate_id: str,
        exit_price: float,
        exit_reason: str,
        actual_pnl_rs: float,
        actual_pnl_pct: Optional[float] = None,
        exit_timestamp: Optional[str] = None,
        outcome_status: str = "MEASURED"
    ) -> bool:
        """
        Updates the identical baseline exit outcome for a previously recorded candidate.
        Works for both guard-accepted and guard-rejected candidates.
        """
        records = self._load_log()
        updated = False
        now_str = exit_timestamp or datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        for r in records:
            if r.get("candidate_id") == candidate_id or (not r.get("subsequent_exit_price") and r.get("contract") in candidate_id):
                r["outcome_status"] = outcome_status
                r["exit_timestamp"] = now_str
                r["subsequent_exit_price"] = round(float(exit_price), 2)
                r["subsequent_exit_reason"] = exit_reason
                r["actual_pnl_rs"] = round(float(actual_pnl_rs), 2)
                if actual_pnl_pct is not None:
                    r["actual_pnl_pct"] = round(float(actual_pnl_pct), 2)
                elif r.get("ltp") and r.get("ltp") > 0:
                    r["actual_pnl_pct"] = round(((exit_price - r["ltp"]) / r["ltp"]) * 100.0, 2)
                updated = True
                break

        if updated:
            self._save_log(records)
        return updated

    def mark_unmeasurable(self, candidate_id: str, reason: str = "SESSION_CLOSED_NO_CONTEMPORANEOUS_QUOTE") -> bool:
        """Marks an open opportunity as unmeasurable if contemporaneous quotes are unavailable."""
        records = self._load_log()
        updated = False
        for r in records:
            if r.get("candidate_id") == candidate_id:
                r["outcome_status"] = "UNMEASURABLE"
                r["unmeasurable_reason"] = reason
                updated = True
                break
        if updated:
            self._save_log(records)
        return updated
