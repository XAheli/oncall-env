---
title: On-Call Incident Response Env
sdk: docker
app_port: 7860
tags:
  - openenv
---

# On-Call: Production Incident Response Environment

An OpenEnv-compliant reinforcement learning environment that simulates
production incident response in a microservices system. An AI agent must
investigate alerts, read logs, check metrics, trace dependency chains,
apply remediations, and submit a root-cause diagnosis — just like a
real on-call engineer.

## Motivation

Production incident response is one of the most cognitively demanding tasks in
software engineering. Every engineer at scale rotates on-call, yet no
RL/agent environment exists for training or evaluating diagnostic reasoning
in this domain. This environment fills that gap by providing deterministic,
reproducible incident scenarios graded against planted faults with known root
causes.

## Architecture

A simulated platform of **7 microservices**:

| Service                | Role                          |
| ---------------------- | ----------------------------- |
| `api-gateway`          | Frontend proxy, routes traffic |
| `auth-service`         | Authentication & sessions      |
| `user-service`         | User profiles & accounts       |
| `order-service`        | Order processing pipeline      |
| `payment-service`      | Payment transactions           |
| `notification-service` | Email / SMS / push             |
| `cache-service`        | Shared Redis-like cache        |

Each service exposes logs, metrics, configuration, deploy history,
dependency graph, and health status that the agent can query.

## Action Space

| Action Type            | Category      | Description                                  |
| ---------------------- | ------------- | -------------------------------------------- |
| `check_alerts`         | Investigation | View all active alerts                       |
| `query_logs`           | Investigation | Read logs (optional level filter)            |
| `check_metrics`        | Investigation | Get metric time-series for a service         |
| `inspect_config`       | Investigation | View full configuration JSON                 |
| `check_dependencies`   | Investigation | View dependency graph                        |
| `check_status`         | Investigation | Service health, version, uptime              |
| `check_deploy_history` | Investigation | Recent deployments and config changes        |
| `restart_service`      | Remediation   | Restart a service (risky if wrong)           |
| `rollback_deploy`      | Remediation   | Roll back most recent deploy                 |
| `update_config`        | Remediation   | Change a config key-value pair               |
| `scale_service`        | Remediation   | Change replica count                         |
| `submit_diagnosis`     | Terminal      | Submit root cause + description + fix        |

Metrics available: `cpu_usage`, `memory_usage`, `request_latency_ms`,
`error_rate`, `request_count`.

## Observation Space

Each step returns:

- **alerts** — active alert objects (id, service, severity, message, timestamp)
- **logs** — log entries (timestamp, level, service, message)
- **metrics** — time-series data points for a given metric
- **config** — service configuration dict
- **dependencies** — list of upstream dependencies
- **service_status** — health status, version, uptime
- **deploy_history** — recent deploys with changed configs
- **action_result** — human-readable outcome of the last action
- **incident_summary** — the initial incident report
- **step_number / max_steps** — progress tracking

## Tasks

### Task 1: `service_down` (Easy, 30 steps)

Payment failures across the platform. Root cause: a recent deploy set the
payment-service database connection pool to 1, causing connection
exhaustion under normal load. Direct single-service investigation.

### Task 2: `cascading_failure` (Medium, 50 steps)

Platform-wide latency degradation. Root cause: cache-service OOM killed
after a memory-leaking deploy, causing auth-service to fall back to
direct database queries, overloading the DB and cascading latency to all
services. Requires tracing through the dependency chain.

### Task 3: `phantom_bottleneck` (Hard, 75 steps)

Intermittent ~20% order failures with no obvious pattern. Root cause:
a notification-service deploy added a synchronous external webhook call
with a 30s timeout and undersized connection pool, causing head-of-line
blocking that intermittently stalls order-service. Includes red herrings
(unrelated deploys, downstream error noise).

## Reward Design

**Per-step signals:**

- Investigation of a critical-path service: **+0.02**
- Investigation of an unrelated service: **+0.005**
- Checking alerts: **+0.01**
- Correct remediation: **+0.15**
- Partial remediation: **+0.05**
- Wrong remediation (collateral): **-0.05**
- Step penalty: **-0.003** per step

**Final score** (on `submit_diagnosis`, 0.0–1.0):

| Component                    | Weight |
| ---------------------------- | ------ |
| Root-cause service correct   | 30%    |
| Root-cause description (keywords) | 25% |
| Correct fix applied          | 25%    |
| Investigation efficiency     | 10%    |
| No collateral damage         | 10%    |

## Setup

### Docker (recommended)

```bash
docker build -t oncall-env .
docker run -p 7860:7860 oncall-env
```

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run inference

```bash
export HF_TOKEN=your-api-key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export ENV_URL=http://localhost:7860

python inference.py
```

## API Endpoints

| Method | Path     | Description               |
| ------ | -------- | ------------------------- |
| POST   | /reset   | Reset env (pass `task_name`) |
| POST   | /step    | Execute an action         |
| GET    | /state   | Current episode state     |
| GET    | /schema  | Action/observation schemas |
| GET    | /health  | Health check              |

### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "service_down"}'
```

### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "check_alerts"}}'
```

## Baseline Scores

| Task                  | Difficulty | Expected Range |
| --------------------- | ---------- | -------------- |
| `service_down`        | Easy       | 0.70 – 0.90   |
| `cascading_failure`   | Medium     | 0.45 – 0.65   |
| `phantom_bottleneck`  | Hard       | 0.25 – 0.45   |

Scores depend on the model used. Ranges above are estimates for
frontier-class models (GPT-4 tier).
