"""
Pydantic models for the On-Call Incident Response Environment.

Defines typed Action and Observation schemas that conform to the OpenEnv spec.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# Helper models (used as fields inside the Observation)
# ---------------------------------------------------------------------------

class AlertInfo(BaseModel):
    alert_id: str
    service: str
    severity: str = Field(..., description="critical | warning | info")
    message: str
    timestamp: str
    acknowledged: bool = False


class LogEntry(BaseModel):
    timestamp: str
    level: str = Field(..., description="DEBUG | INFO | WARN | ERROR | FATAL")
    service: str
    message: str


class MetricPoint(BaseModel):
    timestamp: str
    value: float


class MetricsData(BaseModel):
    service: str
    metric_name: str
    unit: str
    data_points: List[MetricPoint]


class ServiceStatusInfo(BaseModel):
    service: str
    status: str = Field(..., description="healthy | degraded | down")
    uptime_seconds: int
    last_restart: Optional[str] = None
    version: str


class DeployInfo(BaseModel):
    deploy_id: str
    service: str
    timestamp: str
    version: str
    description: str
    status: str = Field(..., description="success | failed | rolled_back")
    changed_configs: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = [
    "check_alerts",
    "query_logs",
    "check_metrics",
    "inspect_config",
    "check_dependencies",
    "check_status",
    "check_deploy_history",
    "restart_service",
    "rollback_deploy",
    "update_config",
    "scale_service",
    "submit_diagnosis",
]

INVESTIGATION_ACTIONS = {
    "check_alerts",
    "query_logs",
    "check_metrics",
    "inspect_config",
    "check_dependencies",
    "check_status",
    "check_deploy_history",
}

REMEDIATION_ACTIONS = {
    "restart_service",
    "rollback_deploy",
    "update_config",
    "scale_service",
}


class OnCallAction(Action):
    """Agent action in the On-Call environment."""

    action_type: str = Field(
        ...,
        description=(
            "One of: check_alerts, query_logs, check_metrics, inspect_config, "
            "check_dependencies, check_status, check_deploy_history, "
            "restart_service, rollback_deploy, update_config, scale_service, "
            "submit_diagnosis"
        ),
    )
    service: Optional[str] = Field(
        default=None,
        description="Target service name (required for most actions except check_alerts)",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Action-specific parameters (e.g. level, metric_name, key/value, root_cause_service)",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class OnCallObservation(Observation):
    """Observation returned by the On-Call environment after each step."""

    alerts: Optional[List[AlertInfo]] = Field(default=None, description="Active alerts")
    logs: Optional[List[LogEntry]] = Field(default=None, description="Log entries")
    metrics: Optional[MetricsData] = Field(default=None, description="Metric time-series")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Service configuration")
    dependencies: Optional[List[str]] = Field(default=None, description="Service dependency list")
    service_status: Optional[ServiceStatusInfo] = Field(default=None, description="Service health info")
    deploy_history: Optional[List[DeployInfo]] = Field(default=None, description="Recent deploys")
    action_result: str = Field(default="", description="Human-readable result of the last action")
    available_services: List[str] = Field(default_factory=list, description="All service names in the system")
    incident_summary: str = Field(default="", description="Initial incident report shown at reset")
    step_number: int = Field(default=0)
    max_steps: int = Field(default=0)
