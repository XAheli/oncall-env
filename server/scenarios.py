"""
Incident scenario definitions and deterministic grading for the On-Call env.

Each scenario specifies the incident narrative, pre-built infrastructure state,
ground-truth root cause, accepted fixes, and a scoring function that maps agent
actions to a [0.0, 1.0] score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .infra import (
    ServiceData,
    build_cascading_failure,
    build_phantom_bottleneck,
    build_service_down,
)


# ---------------------------------------------------------------------------
# Scenario descriptor
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    display_name: str
    incident_summary: str
    services: Dict[str, ServiceData]
    alerts: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    max_steps: int


TASK_NAMES = ["service_down", "cascading_failure", "phantom_bottleneck"]

_DIFFICULTY_ALIASES = {
    "easy": "service_down",
    "medium": "cascading_failure",
    "hard": "phantom_bottleneck",
}

_INCIDENT_SUMMARIES = {
    "service_down": (
        "INCIDENT REPORT — Severity: HIGH\n"
        "Reported at: 2024-01-15 10:08 UTC\n\n"
        "Summary: A critical service is degraded. Users are experiencing errors "
        "when performing core operations. Support tickets are spiking.\n\n"
        "Your task: Investigate the production environment, identify the root "
        "cause, apply an appropriate fix, and submit your diagnosis."
    ),
    "cascading_failure": (
        "INCIDENT REPORT — Severity: CRITICAL\n"
        "Reported at: 2024-01-15 13:45 UTC\n\n"
        "Summary: Platform-wide degradation detected. Users are experiencing "
        "extremely slow page loads and intermittent 503 errors across the "
        "entire application. Multiple services appear affected. The issue "
        "started approximately 30 minutes ago.\n\n"
        "Your task: Investigate the cascading failure, trace the root cause "
        "through the service dependency chain, apply a fix, and submit your "
        "diagnosis."
    ),
    "phantom_bottleneck": (
        "INCIDENT REPORT — Severity: MEDIUM\n"
        "Reported at: 2024-01-15 15:50 UTC\n\n"
        "Summary: Intermittent processing failures reported. "
        "Approximately 20% of requests are failing with 503 errors, but the "
        "failures have no obvious pattern — some succeed, others fail. "
        "The issue appears to have started about 30 minutes ago. No recent "
        "changes to the affected pipeline are known.\n\n"
        "Your task: Investigate the intermittent failures, identify why only "
        "a fraction of requests are affected, apply a fix, and submit your "
        "diagnosis."
    ),
}

_BUILDERS = {
    "service_down": build_service_down,
    "cascading_failure": build_cascading_failure,
    "phantom_bottleneck": build_phantom_bottleneck,
}


def load_scenario(name: str, seed: int = 42) -> Scenario:
    """Load a scenario, optionally with a seed for variant selection.

    ``name`` can be a task name (``service_down``) or a difficulty alias
    (``easy``).  The ``seed`` controls which variant of the scenario is
    generated — same seed always yields the same scenario.
    """
    name = _DIFFICULTY_ALIASES.get(name, name)
    if name not in _BUILDERS:
        all_names = list(_BUILDERS) + list(_DIFFICULTY_ALIASES)
        raise ValueError(
            f"Unknown scenario '{name}'. Choose from: {', '.join(all_names)}"
        )
    services, alerts, ground_truth = _BUILDERS[name](seed=seed)
    return Scenario(
        name=name,
        display_name=name.replace("_", " ").title(),
        incident_summary=_INCIDENT_SUMMARIES[name],
        services=services,
        alerts=alerts,
        ground_truth=ground_truth,
        max_steps=ground_truth["max_steps"],
    )


# ---------------------------------------------------------------------------
# Reward helpers (called per-step by the environment)
# ---------------------------------------------------------------------------

STEP_PENALTY = -0.003


def compute_investigation_reward(
    action_type: str,
    service: Optional[str],
    critical_path: Set[str],
) -> float:
    """Reward for a read-only investigation action."""
    if action_type == "check_alerts":
        return 0.01
    if service is None:
        return 0.0
    if service in critical_path:
        return 0.02
    return 0.005


def compute_remediation_reward(
    action_type: str,
    service: Optional[str],
    params: Optional[Dict[str, Any]],
    ground_truth: Dict[str, Any],
) -> float:
    """Reward for a state-modifying remediation action."""
    if service is None:
        return -0.05

    for fix in ground_truth.get("correct_fixes", []):
        if fix["action"] == action_type and fix["service"] == service:
            if "params_key" in fix and params:
                if fix["params_key"] in (params.get("key", ""), str(params)):
                    return 0.15
            elif "params_key" not in fix:
                return 0.15

    for fix in ground_truth.get("partial_fixes", []):
        if fix["action"] == action_type and fix["service"] == service:
            return 0.05

    return -0.05


# ---------------------------------------------------------------------------
# Final grading (called when agent submits diagnosis)
# ---------------------------------------------------------------------------

def grade_diagnosis(
    ground_truth: Dict[str, Any],
    submitted_service: Optional[str],
    submitted_description: Optional[str],
    fix_applied: bool,
    fix_is_correct: bool,
    fix_is_partial: bool,
    steps_taken: int,
    max_steps: int,
    collateral_damage_count: int,
) -> float:
    """
    Compute the final episode score in [0.0, 1.0].

    Components (weights):
        root_cause_service_correct  0.30
        root_cause_description      0.25
        correct_fix_applied         0.25
        investigation_efficiency    0.10
        no_collateral_damage        0.10
    """
    score = 0.0

    # 1. Root-cause service identification (30%)
    if submitted_service and submitted_service == ground_truth["root_cause_service"]:
        score += 0.30

    # 2. Root-cause description quality — keyword matching (25%)
    desc_score = _keyword_score(
        submitted_description or "",
        ground_truth["root_cause_keywords"],
    )
    score += 0.25 * desc_score

    # 3. Fix applied (25%)
    if fix_is_correct:
        score += 0.25
    elif fix_is_partial:
        score += 0.125

    # 4. Investigation efficiency (10%)
    if max_steps > 0:
        efficiency = max(0.0, 1.0 - steps_taken / max_steps)
        score += 0.10 * efficiency

    # 5. No collateral damage (10%)
    if collateral_damage_count == 0:
        score += 0.10
    else:
        damage_penalty = min(collateral_damage_count * 0.03, 0.10)
        score += max(0.0, 0.10 - damage_penalty)

    return round(min(max(score, 0.0), 1.0), 4)


def _keyword_score(text: str, keywords: List[str]) -> float:
    """Fraction of keywords found in *text* (case-insensitive)."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)
