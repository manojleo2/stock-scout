"""
Stock Scout Model Monitoring & Training Agent Framework.
Extensible agent package for autonomous ML health monitoring and retraining.
"""

from agent.base import BaseTask, TaskResult, AgentStateStore
from agent.tasks import (
    ModelMonitoringTask,
    ModelTrainingTask,
    AutoAuditAndRetrainTask,
    CustomTaskTemplate
)
from agent.model_agent import StockScoutModelAgent

__all__ = [
    "BaseTask",
    "TaskResult",
    "AgentStateStore",
    "ModelMonitoringTask",
    "ModelTrainingTask",
    "AutoAuditAndRetrainTask",
    "CustomTaskTemplate",
    "StockScoutModelAgent"
]
