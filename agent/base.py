import os
import json
import logging
import datetime as dt
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StockScoutAgent")

class TaskResult:
    """Standardized response from any agent task execution."""
    def __init__(
        self,
        task_name: str,
        success: bool,
        message: str = "",
        data: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None
    ):
        self.task_name = task_name
        self.success = success
        self.message = message
        self.data = data or {}
        self.metrics = metrics or {}
        self.recommendations = recommendations or []
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp
        }

class BaseTask(ABC):
    """
    Abstract base class for all pluggable agent tasks.
    Users can inherit from BaseTask to add new model operations,
    evaluators, backtesters, or alerting mechanisms.
    """
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> TaskResult:
        """
        Execute task logic.
        :param context: Shared dictionary containing runtime configurations,
                        symbols, thresholds, and references.
        :return: TaskResult
        """
        pass

class AgentStateStore:
    """Manages persistent state, audit metrics, and execution history for the agent."""
    def __init__(self, state_file_path: str):
        self.state_file_path = state_file_path

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load agent state from {self.state_file_path}: {e}")
        return {
            "agent_name": "StockScoutModelAgent",
            "version": "1.0.0",
            "last_run": None,
            "monitored_models": {},
            "history": []
        }

    def save(self, state: Dict[str, Any]):
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent state to {self.state_file_path}: {e}")

    def append_history(self, event_type: str, details: Dict[str, Any], max_history: int = 100):
        state = self.load()
        history = state.get("history", [])
        entry = {
            "id": len(history) + 1,
            "event_type": event_type,
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "details": details
        }
        history.insert(0, entry)
        state["history"] = history[:max_history]
        state["last_run"] = entry["timestamp"]
        self.save(state)
