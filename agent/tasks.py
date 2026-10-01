import os
import sys
import logging
import datetime as dt
from typing import Dict, Any, List, Optional

# Ensure parent directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.base import BaseTask, TaskResult
from utils.prediction_audit import load_saved_audit_history
from utils.data_loader import load_saved_watchlist, get_stock_data
from utils.ml_model import train_and_predict
from config import STOCK_NAME_MAP

logger = logging.getLogger("StockScoutAgent.Tasks")

class ModelMonitoringTask(BaseTask):
    """
    Evaluates real-world prediction audit performance, checks for model drift,
    accuracy degradation, divergence patterns, and data freshness.
    """
    def __init__(self, default_threshold_pct: float = 55.0, min_eval_samples: int = 5):
        super().__init__(
            name="model_monitoring",
            description="Evaluates prediction audits, detects accuracy drift, and checks model health."
        )
        self.default_threshold_pct = default_threshold_pct
        self.min_eval_samples = min_eval_samples

    def execute(self, context: Dict[str, Any]) -> TaskResult:
        symbols = context.get("symbols") or load_saved_watchlist()
        threshold_pct = context.get("threshold_pct", self.default_threshold_pct)
        min_samples = context.get("min_eval_samples", self.min_eval_samples)

        audit_history = load_saved_audit_history()
        health_reports = {}
        all_recommendations = []
        retrain_needed_symbols = []

        for sym in symbols:
            name = STOCK_NAME_MAP.get(sym, sym)
            # Filter audit records for symbol
            records = [r for r in audit_history if r.get("symbol") == sym]
            completed = [r for r in records if r.get("is_correct") is not None]
            pending = [r for r in records if r.get("is_correct") is None]

            total_records = len(records)
            completed_count = len(completed)

            if completed_count == 0:
                health_reports[sym] = {
                    "name": name,
                    "symbol": sym,
                    "status": "INSUFFICIENT_DATA",
                    "status_badge": "🟡 Insufficient Data",
                    "overall_accuracy_pct": None,
                    "rolling_5_acc_pct": None,
                    "rolling_10_acc_pct": None,
                    "completed_samples": 0,
                    "pending_samples": len(pending),
                    "last_prediction_date": records[-1].get("target_date") if records else None,
                    "divergence_highlights": [],
                    "retrain_recommended": False,
                    "reason": "Not enough audited historical trade sessions to compute drift."
                }
                continue

            # Calculate accuracy metrics
            correct_count = sum(1 for r in completed if r.get("is_correct") is True)
            overall_acc = round((correct_count / completed_count) * 100.0, 1)

            # Rolling recent performance (last 5 & last 10 completed)
            last_5 = completed[-5:]
            last_10 = completed[-10:]
            acc_last_5 = round((sum(1 for r in last_5 if r.get("is_correct") is True) / len(last_5)) * 100.0, 1) if last_5 else overall_acc
            acc_last_10 = round((sum(1 for r in last_10 if r.get("is_correct") is True) / len(last_10)) * 100.0, 1) if last_10 else overall_acc

            # Extract recent divergence reasons
            divergences = []
            for r in reversed(completed[-10:]):
                if r.get("is_correct") is False and r.get("divergence_reasons"):
                    for dr in r["divergence_reasons"]:
                        if dr not in divergences and "Pending" not in dr:
                            divergences.append(dr)

            # Drift & Health Evaluation
            needs_retrain = False
            status = "HEALTHY"
            status_badge = "🟢 Healthy"
            reason = "Model performance is stable and meets target accuracy."

            if completed_count >= min_samples:
                # Drift condition: recent 10-trade accuracy drops below threshold or significantly below overall
                if acc_last_10 < threshold_pct:
                    status = "NEEDS_RETRAINING"
                    status_badge = "🔴 Accuracy Drift (Retrain Needed)"
                    needs_retrain = True
                    reason = f"Recent 10-session win rate ({acc_last_10}%) dropped below threshold ({threshold_pct}%)."
                    retrain_needed_symbols.append(sym)
                    all_recommendations.append(f"Retrain model for {sym}: win rate degraded to {acc_last_10}%.")
                elif acc_last_10 < (overall_acc - 15.0):
                    status = "PERFORMANCE_DIP"
                    status_badge = "🟠 Performance Dip"
                    reason = f"Recent accuracy ({acc_last_10}%) is 15%+ lower than historical average ({overall_acc}%)."
                    all_recommendations.append(f"Monitor {sym} closely: recent performance is dipping.")
            else:
                status = "CALIBRATING"
                status_badge = "🔵 Calibrating"
                reason = f"Audited sample count ({completed_count}) is below minimum threshold ({min_samples})."

            health_reports[sym] = {
                "name": name,
                "symbol": sym,
                "status": status,
                "status_badge": status_badge,
                "overall_accuracy_pct": overall_acc,
                "rolling_5_acc_pct": acc_last_5,
                "rolling_10_acc_pct": acc_last_10,
                "completed_samples": completed_count,
                "pending_samples": len(pending),
                "total_samples": total_records,
                "last_prediction_date": records[-1].get("target_date") if records else None,
                "divergence_highlights": divergences[:3],
                "retrain_recommended": needs_retrain,
                "reason": reason
            }

        return TaskResult(
            task_name=self.name,
            success=True,
            message=f"Monitored {len(symbols)} symbols. {len(retrain_needed_symbols)} models flagged for retraining.",
            data={"health_reports": health_reports, "retrain_needed_symbols": retrain_needed_symbols},
            metrics={
                "symbols_count": len(symbols),
                "retrain_needed_count": len(retrain_needed_symbols),
                "audit_records_analyzed": len(audit_history)
            },
            recommendations=all_recommendations
        )


class ModelTrainingTask(BaseTask):
    """
    Executes ML model training for selected symbols using utils.ml_model.train_and_predict.
    Clears cache, evaluates validation metrics, and captures top feature drivers.
    """
    def __init__(self, default_period: str = "2y"):
        super().__init__(
            name="model_training",
            description="Trains directional ensemble models and logs validation metrics."
        )
        self.default_period = default_period

    def execute(self, context: Dict[str, Any]) -> TaskResult:
        symbols = context.get("symbols")
        if not symbols:
            symbols = load_saved_watchlist()
        elif isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]

        period = context.get("period", self.default_period)

        # Clear Streamlit cache if applicable
        try:
            if hasattr(train_and_predict, "clear"):
                train_and_predict.clear()
        except Exception as e:
            logger.debug(f"Could not clear cache: {e}")

        training_results = {}
        successful_trains = 0
        failed_trains = 0
        recommendations = []

        for sym in symbols:
            name = STOCK_NAME_MAP.get(sym, sym)
            logger.info(f"Training model for {sym} ({period})...")
            try:
                res = train_and_predict(sym, period=period)
                if res.get("status") == "success":
                    successful_trains += 1
                    top_feats = list(res.get("feature_importances", {}).items())[:5]
                    training_results[sym] = {
                        "symbol": sym,
                        "name": name,
                        "status": "success",
                        "test_accuracy_pct": res.get("test_accuracy_pct"),
                        "precision_pct": res.get("precision_pct"),
                        "recall_pct": res.get("recall_pct"),
                        "direction": res.get("direction"),
                        "probability_up_pct": res.get("probability_up_pct"),
                        "confidence": res.get("confidence"),
                        "sample_count": res.get("sample_count"),
                        "test_sample_count": res.get("test_sample_count"),
                        "top_features": top_feats,
                        "feedback_offset_pct": res.get("feedback_offset_pct"),
                        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    recommendations.append(
                        f"Trained {sym}: Test Accuracy = {res.get('test_accuracy_pct')}%, "
                        f"Bias = {res.get('direction')} ({res.get('probability_up_pct')}%)"
                    )
                else:
                    failed_trains += 1
                    training_results[sym] = {
                        "symbol": sym,
                        "name": name,
                        "status": "failed",
                        "error": res.get("message", "Unknown training error"),
                        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            except Exception as e:
                failed_trains += 1
                logger.error(f"Error executing training for {sym}: {e}")
                training_results[sym] = {
                    "symbol": sym,
                    "name": name,
                    "status": "failed",
                    "error": str(e),
                    "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

        is_success = successful_trains > 0 or len(symbols) == 0
        return TaskResult(
            task_name=self.name,
            success=is_success,
            message=f"Trained {successful_trains}/{len(symbols)} models successfully.",
            data={"training_results": training_results},
            metrics={
                "requested": len(symbols),
                "successful": successful_trains,
                "failed": failed_trains
            },
            recommendations=recommendations
        )


class AutoAuditAndRetrainTask(BaseTask):
    """
    Automated orchestration pipeline:
    1. Inspects prediction audit for drift/degradation.
    2. Identifies models requiring retraining.
    3. Retrains models automatically and updates state.
    """
    def __init__(self, default_threshold_pct: float = 55.0):
        super().__init__(
            name="auto_audit_and_retrain",
            description="Autonomous health audit and automatic retraining for drifted models."
        )
        self.default_threshold_pct = default_threshold_pct
        self.monitor_task = ModelMonitoringTask(default_threshold_pct=default_threshold_pct)
        self.train_task = ModelTrainingTask()

    def execute(self, context: Dict[str, Any]) -> TaskResult:
        threshold_pct = context.get("threshold_pct", self.default_threshold_pct)
        
        # 1. Run monitoring
        monitor_res = self.monitor_task.execute(context)
        retrain_symbols = monitor_res.data.get("retrain_needed_symbols", [])

        if not retrain_symbols:
            return TaskResult(
                task_name=self.name,
                success=True,
                message="Audit completed. All models meet target performance; no retraining required.",
                data={
                    "monitoring_summary": monitor_res.data,
                    "retrained_symbols": []
                },
                metrics={"drifted_count": 0, "retrained_count": 0},
                recommendations=["All models healthy. No retraining needed at this time."]
            )

        # 2. Retrain drifted models
        logger.info(f"Auto-triggering retrain for drifted models: {retrain_symbols}")
        train_context = dict(context)
        train_context["symbols"] = retrain_symbols
        train_res = self.train_task.execute(train_context)

        recommendations = [
            f"Auto-retrained {len(retrain_symbols)} models due to accuracy drift below {threshold_pct}%."
        ] + train_res.recommendations

        return TaskResult(
            task_name=self.name,
            success=train_res.success,
            message=f"Audit completed: {len(retrain_symbols)} models drifted and were retrained.",
            data={
                "monitoring_summary": monitor_res.data,
                "retrained_results": train_res.data.get("training_results", {}),
                "retrained_symbols": retrain_symbols
            },
            metrics={
                "drifted_count": len(retrain_symbols),
                "retrained_count": train_res.metrics.get("successful", 0)
            },
            recommendations=recommendations
        )


class CustomTaskTemplate(BaseTask):
    """
    Template for users to easily create and register their own custom tasks.
    Example: Custom feature analysis, volume anomaly detection, or external alerting.
    """
    def __init__(self, name: str = "custom_user_task", description: str = "User customized task"):
        super().__init__(name=name, description=description)

    def execute(self, context: Dict[str, Any]) -> TaskResult:
        symbols = context.get("symbols") or load_saved_watchlist()
        # User adds custom logic here
        return TaskResult(
            task_name=self.name,
            success=True,
            message=f"Executed custom task for {len(symbols)} symbols.",
            data={"symbols": symbols},
            recommendations=["Custom task completed successfully."]
        )
