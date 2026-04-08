"""
On-Call Environment Client.

Typed ``EnvClient`` wrapper for the On-Call Incident Response Environment.
Communicates with the environment server over WebSocket.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import OnCallAction, OnCallObservation


class OnCallEnv(EnvClient[OnCallAction, OnCallObservation, State]):
    """
    Client for the On-Call environment.

    Example::

        env = OnCallEnv(base_url="http://localhost:7860")
        result = env.reset(task_name="service_down")
        print(result.observation.incident_summary)

        result = env.step(OnCallAction(action_type="check_alerts"))
        for alert in result.observation.alerts:
            print(alert.severity, alert.message)

        env.close()
    """

    def _step_payload(self, action: OnCallAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.service is not None:
            payload["service"] = action.service
        if action.params is not None:
            payload["params"] = action.params
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[OnCallObservation]:
        obs_data = payload.get("observation", {})
        observation = OnCallObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
