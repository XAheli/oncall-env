"""
Baseline inference script for the On-Call Incident Response Environment.

Uses the OpenAI client to run an LLM agent against the environment,
producing reproducible baseline scores on all three tasks.

Required env vars:
    HF_TOKEN           – API key (mandatory, no default)
    API_BASE_URL       – LLM endpoint (default: https://api.openai.com/v1)
    MODEL_NAME         – model id  (default: gpt-4.1-mini)
Optional:
    LOCAL_IMAGE_NAME   – Docker image to launch via from_docker_image()
    ENV_URL            – URL of an already-running env (fallback)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables (per hackathon spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "oncall"
TASKS = ["service_down", "cascading_failure", "phantom_bottleneck"]
MAX_STEPS = {"service_down": 30, "cascading_failure": 50, "phantom_bottleneck": 75}
TEMPERATURE = 0.2
MAX_TOKENS = 1024
SUCCESS_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Import typed client
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import OnCallEnv  # noqa: E402
from models import OnCallAction  # noqa: E402

# ---------------------------------------------------------------------------
# Structured logging (matches spec EXACTLY)
#   [START] task=<task_name> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rw = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} score={score:.2f} rewards={rw}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) responding to a production incident.

You interact with a simulated microservices environment. At each step you must
output EXACTLY ONE JSON action object (no markdown fences, no extra text).

Available actions:

INVESTIGATION (read-only):
  {"action_type": "check_alerts"}
  {"action_type": "query_logs", "service": "<name>", "params": {"level": "ERROR"}}
  {"action_type": "check_metrics", "service": "<name>", "params": {"metric_name": "<metric>"}}
  {"action_type": "inspect_config", "service": "<name>"}
  {"action_type": "check_dependencies", "service": "<name>"}
  {"action_type": "check_status", "service": "<name>"}
  {"action_type": "check_deploy_history", "service": "<name>"}

REMEDIATION (modifies state):
  {"action_type": "restart_service", "service": "<name>"}
  {"action_type": "rollback_deploy", "service": "<name>"}
  {"action_type": "update_config", "service": "<name>", "params": {"key": "<k>", "value": "<v>"}}
  {"action_type": "scale_service", "service": "<name>", "params": {"replicas": <n>}}

SUBMIT (terminal):
  {"action_type": "submit_diagnosis", "params": {"root_cause_service": "<name>", "root_cause_description": "<text>", "remediation_action": "<what you did>"}}

Services: api-gateway, auth-service, user-service, order-service, payment-service, notification-service, cache-service
Metrics: cpu_usage, memory_usage, request_latency_ms, error_rate, request_count

Strategy:
1. Start by checking alerts.
2. Investigate the most suspicious services by reading logs and metrics.
3. Trace through dependencies to find the root cause.
4. Apply a fix BEFORE submitting your diagnosis.
5. Submit your diagnosis with the root-cause service, description, and what you fixed.

Be efficient — every step has a small cost. Do NOT explore services that look healthy unless you have reason to suspect them.
"""

# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump(exclude_none=True)
    return dict(obs) if obs else {}


def build_user_prompt(step: int, obs_dict: Dict[str, Any], history: List[str]) -> str:
    parts = [f"Step {step}/{obs_dict.get('max_steps', '?')}"]

    if obs_dict.get("incident_summary") and step == 1:
        parts.append(f"\n--- INCIDENT REPORT ---\n{obs_dict['incident_summary']}")

    if obs_dict.get("action_result"):
        parts.append(f"\nLast action result: {obs_dict['action_result']}")

    for field in ("alerts", "logs", "metrics", "config", "dependencies",
                   "service_status", "deploy_history"):
        val = obs_dict.get(field)
        if val is not None:
            parts.append(f"\n{field}: {json.dumps(val, indent=2, default=str)}")

    if history:
        parts.append("\n--- Recent history ---")
        for h in history[-8:]:
            parts.append(h)

    parts.append("\nOutput your next action as a single JSON object:")
    return "\n".join(parts)


def get_model_action(
    client: OpenAI,
    step: int,
    obs_dict: Dict[str, Any],
    history: List[str],
) -> str:
    prompt = build_user_prompt(step, obs_dict, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"action_type": "check_alerts"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "check_alerts"}'


def parse_action(raw: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"action_type": "check_alerts"}


def dict_to_action(d: Dict[str, Any]) -> OnCallAction:
    return OnCallAction(
        action_type=d.get("action_type", "check_alerts"),
        service=d.get("service"),
        params=d.get("params"),
    )

# ---------------------------------------------------------------------------
# Per-task episode runner
# ---------------------------------------------------------------------------

async def run_task(llm_client: OpenAI, env: OnCallEnv, task_name: str) -> float:
    max_steps = MAX_STEPS[task_name]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(task_name=task_name)
        obs_dict = _obs_to_dict(result.observation)
        done = result.done

        for step in range(1, max_steps + 1):
            if done:
                break

            raw_action = get_model_action(llm_client, step, obs_dict, history)
            action_dict = parse_action(raw_action)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            action = dict_to_action(action_dict)
            result = await env.step(action)

            obs_dict = _obs_to_dict(result.observation)
            reward = result.reward or 0.0
            done = result.done
            error = obs_dict.get("metadata", {}).get("error")

            rewards.append(float(reward))
            steps_taken = step

            log_step(step=step, action=action_str, reward=float(reward), done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        if rewards:
            score = min(sum(rewards), 0.99)
        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if LOCAL_IMAGE_NAME:
        env = await OnCallEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = OnCallEnv(base_url=ENV_URL)
        await env.connect()

    try:
        for task in TASKS:
            score = await run_task(llm_client, env, task)
            print(f"[DEBUG] {task} score: {score:.4f}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
