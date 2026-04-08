"""
On-Call Incident Response Environment.

Implements the OpenEnv Environment interface for a production incident
response simulation.  An agent investigates a microservices system by
reading logs, metrics, and configs, then applies a fix and submits a
diagnosis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import (
        AlertInfo,
        DeployInfo,
        LogEntry,
        MetricPoint,
        MetricsData,
        OnCallAction,
        OnCallObservation,
        ServiceStatusInfo,
        INVESTIGATION_ACTIONS,
        REMEDIATION_ACTIONS,
        VALID_ACTION_TYPES,
    )
except ImportError:
    from ..models import (
        AlertInfo,
        DeployInfo,
        LogEntry,
        MetricPoint,
        MetricsData,
        OnCallAction,
        OnCallObservation,
        ServiceStatusInfo,
        INVESTIGATION_ACTIONS,
        REMEDIATION_ACTIONS,
        VALID_ACTION_TYPES,
    )

from .infra import SERVICE_NAMES
from .scenarios import (
    STEP_PENALTY,
    TASK_NAMES,
    Scenario,
    compute_investigation_reward,
    compute_remediation_reward,
    grade_diagnosis,
    load_scenario,
)


class OnCallEnvironment(Environment):
    """
    On-Call SRE environment.

    The agent investigates a production incident by querying services
    (logs, metrics, configs, dependencies, deploys, status) then
    applies remediation (restart, rollback, config update, scale)
    and finally submits a root-cause diagnosis.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Scenario] = None
        self._fix_applied = False
        self._fix_is_correct = False
        self._fix_is_partial = False
        self._collateral_count = 0
        self._accumulated_reward = 0.0
        self._done = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> OnCallObservation:
        task_name = task_name or "service_down"
        scenario_seed = seed if seed is not None else 42

        self._scenario = load_scenario(task_name, seed=scenario_seed)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._fix_applied = False
        self._fix_is_correct = False
        self._fix_is_partial = False
        self._collateral_count = 0
        self._accumulated_reward = 0.0
        self._done = False

        return OnCallObservation(
            done=False,
            reward=0.0,
            action_result="Environment reset. Incident reported — begin investigation.",
            available_services=list(SERVICE_NAMES),
            incident_summary=self._scenario.incident_summary,
            step_number=0,
            max_steps=self._scenario.max_steps,
            metadata={"task_name": task_name, "episode_id": self._state.episode_id},
        )

    def step(self, action: OnCallAction, **kwargs: Any) -> OnCallObservation:  # type: ignore[override]
        if self._scenario is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            return self._terminal_obs("Episode already finished.")

        self._state.step_count += 1
        at = action.action_type
        svc = action.service
        params = action.params or {}

        if at not in VALID_ACTION_TYPES:
            return self._step_obs(
                reward=-0.01,
                action_result=f"Invalid action_type '{at}'. Valid: {VALID_ACTION_TYPES}",
            )

        if at != "check_alerts" and at != "submit_diagnosis" and svc is None:
            return self._step_obs(
                reward=-0.01,
                action_result=f"Action '{at}' requires a 'service' field.",
            )

        if svc and svc not in SERVICE_NAMES:
            return self._step_obs(
                reward=-0.01,
                action_result=f"Unknown service '{svc}'. Available: {SERVICE_NAMES}",
            )

        # Dispatch
        if at in INVESTIGATION_ACTIONS:
            return self._handle_investigation(at, svc, params)
        if at in REMEDIATION_ACTIONS:
            return self._handle_remediation(at, svc, params)
        if at == "submit_diagnosis":
            return self._handle_submit(params)

        return self._step_obs(reward=-0.01, action_result=f"Unhandled action '{at}'.")

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Investigation handlers
    # ------------------------------------------------------------------

    def _handle_investigation(
        self, action_type: str, service: Optional[str], params: Dict[str, Any]
    ) -> OnCallObservation:
        gt = self._scenario.ground_truth  # type: ignore[union-attr]
        critical: Set[str] = gt.get("critical_path_services", set())
        reward = compute_investigation_reward(action_type, service, critical)

        obs_kwargs: Dict[str, Any] = {}

        if action_type == "check_alerts":
            alerts_raw = self._scenario.alerts  # type: ignore[union-attr]
            obs_kwargs["alerts"] = [AlertInfo(**a) for a in alerts_raw]
            obs_kwargs["action_result"] = f"Showing {len(alerts_raw)} active alert(s)."

        elif action_type == "query_logs":
            sd = self._scenario.services[service]  # type: ignore[index]
            level_filter = params.get("level")
            logs = sd.logs
            if level_filter:
                level_filter = level_filter.upper()
                logs = [l for l in logs if l["level"] == level_filter]
            obs_kwargs["logs"] = [LogEntry(**l) for l in logs[-30:]]
            obs_kwargs["action_result"] = (
                f"Showing {len(obs_kwargs['logs'])} log entries for {service}"
                + (f" (level={level_filter})" if level_filter else "")
                + "."
            )

        elif action_type == "check_metrics":
            sd = self._scenario.services[service]  # type: ignore[index]
            metric_name = params.get("metric_name", "request_latency_ms")
            if metric_name not in sd.metrics:
                return self._step_obs(
                    reward=0.0,
                    action_result=(
                        f"Unknown metric '{metric_name}' for {service}. "
                        f"Available: {list(sd.metrics.keys())}"
                    ),
                )
            from .infra import _METRIC_UNITS
            points = sd.metrics[metric_name]
            obs_kwargs["metrics"] = MetricsData(
                service=service,  # type: ignore[arg-type]
                metric_name=metric_name,
                unit=_METRIC_UNITS.get(metric_name, ""),
                data_points=[MetricPoint(**p) for p in points],
            )
            obs_kwargs["action_result"] = (
                f"Showing {metric_name} for {service} ({len(points)} data points)."
            )

        elif action_type == "inspect_config":
            sd = self._scenario.services[service]  # type: ignore[index]
            obs_kwargs["config"] = sd.config
            obs_kwargs["action_result"] = f"Showing configuration for {service}."

        elif action_type == "check_dependencies":
            sd = self._scenario.services[service]  # type: ignore[index]
            obs_kwargs["dependencies"] = sd.dependencies
            obs_kwargs["action_result"] = (
                f"{service} depends on: {sd.dependencies or '(none)'}."
            )

        elif action_type == "check_status":
            sd = self._scenario.services[service]  # type: ignore[index]
            obs_kwargs["service_status"] = ServiceStatusInfo(
                service=sd.name,
                status=sd.status,
                uptime_seconds=sd.uptime_seconds,
                last_restart=sd.last_restart,
                version=sd.version,
            )
            obs_kwargs["action_result"] = (
                f"{service}: status={sd.status}, version={sd.version}, "
                f"uptime={sd.uptime_seconds}s."
            )

        elif action_type == "check_deploy_history":
            sd = self._scenario.services[service]  # type: ignore[index]
            obs_kwargs["deploy_history"] = [DeployInfo(**d) for d in sd.deploy_history]
            obs_kwargs["action_result"] = (
                f"Showing {len(sd.deploy_history)} deploy(s) for {service}."
            )

        return self._step_obs(reward=reward, **obs_kwargs)

    # ------------------------------------------------------------------
    # Remediation handlers
    # ------------------------------------------------------------------

    def _handle_remediation(
        self, action_type: str, service: Optional[str], params: Dict[str, Any]
    ) -> OnCallObservation:
        gt = self._scenario.ground_truth  # type: ignore[union-attr]
        reward = compute_remediation_reward(action_type, service, params, gt)

        is_correct = reward >= 0.15
        is_partial = 0 < reward < 0.15

        if is_correct:
            self._fix_applied = True
            self._fix_is_correct = True
            msg = f"Remediation applied: {action_type} on {service}. Service recovering."
            sd = self._scenario.services.get(service, None)  # type: ignore[union-attr]
            if sd:
                sd.status = "healthy"
        elif is_partial:
            self._fix_applied = True
            self._fix_is_partial = True
            msg = f"Remediation applied: {action_type} on {service}. Partial improvement observed."
            sd = self._scenario.services.get(service, None)  # type: ignore[union-attr]
            if sd and sd.status == "down":
                sd.status = "degraded"
        else:
            self._collateral_count += 1
            msg = (
                f"Remediation applied: {action_type} on {service}. "
                "No improvement observed — the issue persists."
            )

        if action_type == "update_config" and service:
            key = params.get("key", "")
            value = params.get("value", "")
            sd = self._scenario.services.get(service, None)  # type: ignore[union-attr]
            if sd and key:
                sd.config[key] = value
            msg += f" Config '{key}' updated to '{value}'."

        return self._step_obs(reward=reward, action_result=msg)

    # ------------------------------------------------------------------
    # Submit diagnosis
    # ------------------------------------------------------------------

    def _handle_submit(self, params: Dict[str, Any]) -> OnCallObservation:
        gt = self._scenario.ground_truth  # type: ignore[union-attr]

        submitted_service = params.get("root_cause_service")
        submitted_desc = params.get("root_cause_description", "")
        submitted_remediation = params.get("remediation_action", "")

        final_score = grade_diagnosis(
            ground_truth=gt,
            submitted_service=submitted_service,
            submitted_description=submitted_desc,
            fix_applied=self._fix_applied,
            fix_is_correct=self._fix_is_correct,
            fix_is_partial=self._fix_is_partial,
            steps_taken=self._state.step_count,
            max_steps=self._scenario.max_steps,  # type: ignore[union-attr]
            collateral_damage_count=self._collateral_count,
        )

        self._done = True

        return OnCallObservation(
            done=True,
            reward=final_score,
            action_result=(
                f"Diagnosis submitted. Final score: {final_score:.4f}. "
                f"Root cause service: {submitted_service} "
                f"(expected: {gt['root_cause_service']}). "
                f"Fix applied: {self._fix_applied} "
                f"(correct: {self._fix_is_correct}, partial: {self._fix_is_partial})."
            ),
            available_services=list(SERVICE_NAMES),
            incident_summary=self._scenario.incident_summary,  # type: ignore[union-attr]
            step_number=self._state.step_count,
            max_steps=self._scenario.max_steps,  # type: ignore[union-attr]
            metadata={
                "final_score": final_score,
                "task_name": self._scenario.name,  # type: ignore[union-attr]
                "steps_taken": self._state.step_count,
            },
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _step_obs(self, reward: float, **extra: Any) -> OnCallObservation:
        """Build an in-progress observation, applying the step penalty."""
        total_reward = reward + STEP_PENALTY
        self._accumulated_reward += total_reward

        done = self._state.step_count >= self._scenario.max_steps  # type: ignore[union-attr]
        if done:
            self._done = True

        defaults: Dict[str, Any] = {
            "done": done,
            "reward": round(total_reward, 4),
            "available_services": list(SERVICE_NAMES),
            "incident_summary": self._scenario.incident_summary,  # type: ignore[union-attr]
            "step_number": self._state.step_count,
            "max_steps": self._scenario.max_steps,  # type: ignore[union-attr]
            "action_result": "",
        }
        defaults.update(extra)

        if done and "action_result" in defaults:
            defaults["action_result"] += " Max steps reached — episode ended."

        return OnCallObservation(**defaults)

    def _terminal_obs(self, msg: str) -> OnCallObservation:
        return OnCallObservation(
            done=True,
            reward=0.0,
            action_result=msg,
            available_services=list(SERVICE_NAMES),
            incident_summary=self._scenario.incident_summary if self._scenario else "",
            step_number=self._state.step_count,
            max_steps=self._scenario.max_steps if self._scenario else 0,
        )
