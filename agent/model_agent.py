import os
import sys
import logging
from typing import Dict, Any, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.base import BaseTask, TaskResult, AgentStateStore
from agent.tasks import (
    ModelMonitoringTask,
    ModelTrainingTask,
    AutoAuditAndRetrainTask
)
from utils.data_loader import load_saved_watchlist

logger = logging.getLogger("StockScoutAgent")

DEFAULT_STATE_FILE = os.path.join(PROJECT_ROOT, "agent_state.json")

class StockScoutModelAgent:
    """
    Autonomous Model Monitoring & Training Agent for Stock Scout.
    
    Capabilities:
    - Monitors real-time prediction audits and detects performance degradation/drift.
    - Evaluates rolling win rate against customizable accuracy thresholds.
    - Trains and retrains ensemble directional models on-demand or automatically.
    - Persists agent state, model health profiles, and execution history.
    - Highly extensible: users can register custom tasks with `register_task(...)`.
    """
    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or DEFAULT_STATE_FILE
        self.state_store = AgentStateStore(self.state_file)
        self.tasks: Dict[str, BaseTask] = {}
        
        # Register standard default tasks
        self._register_default_tasks()

    def _register_default_tasks(self):
        self.register_task("monitor", ModelMonitoringTask())
        self.register_task("train", ModelTrainingTask())
        self.register_task("auto_retrain", AutoAuditAndRetrainTask())

    def register_task(self, name: str, task: BaseTask):
        """Register a new task or custom capability to the agent."""
        if not isinstance(task, BaseTask):
            raise TypeError(f"Task '{name}' must inherit from BaseTask")
        self.tasks[name] = task
        logger.info(f"Registered agent task: '{name}' ({task.__class__.__name__})")

    def run_task(self, task_name: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """Execute a registered task by name and log into history."""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' is not registered with the agent.")

        ctx = context or {}
        logger.info(f"Executing task '{task_name}' with context: {ctx}")
        result = self.tasks[task_name].execute(ctx)

        # Update persistent state
        self._update_state_from_result(result)
        return result

    def _update_state_from_result(self, result: TaskResult):
        """Persist task output, update model health, and append history."""
        state = self.state_store.load()
        monitored_models = state.setdefault("monitored_models", {})

        # If monitoring task updated health reports
        if "health_reports" in result.data:
            for sym, report in result.data["health_reports"].items():
                entry = monitored_models.setdefault(sym, {})
                entry.update({
                    "name": report.get("name"),
                    "status": report.get("status"),
                    "status_badge": report.get("status_badge"),
                    "overall_accuracy_pct": report.get("overall_accuracy_pct"),
                    "rolling_10_acc_pct": report.get("rolling_10_acc_pct"),
                    "completed_samples": report.get("completed_samples"),
                    "last_monitored": result.timestamp
                })

        # If training task updated results
        if "training_results" in result.data:
            for sym, t_res in result.data["training_results"].items():
                if t_res.get("status") == "success":
                    entry = monitored_models.setdefault(sym, {})
                    entry.update({
                        "test_accuracy_pct": t_res.get("test_accuracy_pct"),
                        "last_direction": t_res.get("direction"),
                        "last_confidence": t_res.get("confidence"),
                        "last_trained": t_res.get("trained_at"),
                        "top_features": t_res.get("top_features", [])[:3]
                    })

        self.state_store.save(state)
        self.state_store.append_history(
            event_type=result.task_name,
            details={
                "message": result.message,
                "success": result.success,
                "metrics": result.metrics,
                "recommendations": result.recommendations
            }
        )

    # Convenience API methods
    def monitor_models(
        self,
        symbols: Optional[List[str]] = None,
        threshold_pct: float = 55.0,
        min_eval_samples: int = 5
    ) -> TaskResult:
        """Run health audit across watchlist models."""
        return self.run_task("monitor", {
            "symbols": symbols,
            "threshold_pct": threshold_pct,
            "min_eval_samples": min_eval_samples
        })

    def train_models(
        self,
        symbols: Optional[List[str]] = None,
        period: str = "2y"
    ) -> TaskResult:
        """Train or retrain ML directional models."""
        return self.run_task("train", {
            "symbols": symbols,
            "period": period
        })

    def auto_evaluate_and_retrain(
        self,
        symbols: Optional[List[str]] = None,
        threshold_pct: float = 55.0
    ) -> TaskResult:
        """Autonomous check: monitor models and retrain any that show performance drift."""
        return self.run_task("auto_retrain", {
            "symbols": symbols,
            "threshold_pct": threshold_pct
        })

    def get_state(self) -> Dict[str, Any]:
        """Get current persisted agent state."""
        return self.state_store.load()

    def get_history(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Get recent agent execution history entries."""
        state = self.get_state()
        return state.get("history", [])[:limit]
