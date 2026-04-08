"""
Simulated microservices infrastructure for the On-Call environment.

Scenarios are generated from parameterised *fault templates* controlled
by a seed.  Same seed → same scenario (deterministic).  Different seeds
→ different service / fault combinations for the same difficulty level.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


# ───────────────────────────────────────────────────────────────────
# Core data structures
# ───────────────────────────────────────────────────────────────────

@dataclass
class ServiceData:
    name: str
    status: str
    dependencies: List[str]
    config: Dict[str, Any]
    logs: List[Dict[str, str]]
    metrics: Dict[str, List[Dict[str, Any]]]
    deploy_history: List[Dict[str, Any]]
    version: str
    uptime_seconds: int
    last_restart: Optional[str] = None


# ───────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────

SERVICE_NAMES = [
    "api-gateway", "auth-service", "user-service", "order-service",
    "payment-service", "notification-service", "cache-service",
]

SERVICE_DEPENDENCIES: Dict[str, List[str]] = {
    "api-gateway": ["auth-service", "user-service", "order-service"],
    "auth-service": ["cache-service"],
    "user-service": [],
    "order-service": ["payment-service", "notification-service"],
    "payment-service": [],
    "notification-service": [],
    "cache-service": [],
}

REVERSE_DEPS: Dict[str, List[str]] = {}
for _consumer, _providers in SERVICE_DEPENDENCIES.items():
    for _p in _providers:
        REVERSE_DEPS.setdefault(_p, []).append(_consumer)

DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "api-gateway": {
        "service_name": "api-gateway", "port": 8000,
        "upstream_services": ["auth-service", "user-service", "order-service"],
        "rate_limit_rpm": 10000, "request_timeout_ms": 30000,
        "max_connections": 1000, "ssl_enabled": True,
        "cors_origins": ["https://app.example.com"], "log_level": "INFO",
    },
    "auth-service": {
        "service_name": "auth-service", "port": 8001,
        "session_ttl_seconds": 3600, "cache_enabled": True,
        "cache_host": "cache-service", "cache_port": 6379,
        "cache_ttl_seconds": 300, "db_host": "db.internal",
        "db_port": 5432, "db_name": "auth",
        "db_max_connections": 20, "db_connection_timeout_ms": 30000,
        "jwt_secret_key": "********", "jwt_expiry_seconds": 900,
        "max_retry_attempts": 3, "log_level": "INFO",
    },
    "user-service": {
        "service_name": "user-service", "port": 8002,
        "db_host": "db.internal", "db_port": 5432, "db_name": "users",
        "db_max_connections": 20, "db_connection_timeout_ms": 30000,
        "max_retry_attempts": 3,
        "profile_image_bucket": "s3://user-profiles",
        "max_upload_size_mb": 10, "log_level": "INFO",
    },
    "order-service": {
        "service_name": "order-service", "port": 8003,
        "db_host": "db.internal", "db_port": 5432, "db_name": "orders",
        "db_max_connections": 30, "db_connection_timeout_ms": 30000,
        "max_retry_attempts": 3,
        "payment_service_url": "http://payment-service:8004",
        "notification_service_url": "http://notification-service:8005",
        "order_timeout_ms": 30000, "log_level": "INFO",
    },
    "payment-service": {
        "service_name": "payment-service", "port": 8004,
        "db_host": "db.internal", "db_port": 5432, "db_name": "payments",
        "db_max_connections": 50, "db_connection_timeout_ms": 30000,
        "max_retry_attempts": 3,
        "stripe_api_url": "https://api.stripe.com/v1",
        "stripe_webhook_secret": "********",
        "idempotency_ttl_seconds": 86400, "log_level": "INFO",
    },
    "notification-service": {
        "service_name": "notification-service", "port": 8005,
        "email_provider": "sendgrid", "email_api_key": "********",
        "sms_provider": "twilio", "sms_api_key": "********",
        "max_outbound_connections": 10, "notification_retry_attempts": 2,
        "template_dir": "/app/templates", "log_level": "INFO",
    },
    "cache-service": {
        "service_name": "cache-service", "port": 6379,
        "max_memory_mb": 512, "eviction_policy": "allkeys-lru",
        "persistence_enabled": True, "snapshot_interval_seconds": 300,
        "max_clients": 500, "log_level": "INFO",
    },
}

DEFAULT_VERSIONS: Dict[str, str] = {
    "api-gateway": "3.1.0", "auth-service": "2.4.2",
    "user-service": "1.8.0", "order-service": "2.1.3",
    "payment-service": "2.3.0", "notification-service": "3.1.0",
    "cache-service": "1.4.2",
}

_METRIC_UNITS = {
    "cpu_usage": "percent", "memory_usage": "percent",
    "request_latency_ms": "ms", "error_rate": "percent",
    "request_count": "req/min",
}


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

def _ts(base: str, offset_minutes: float) -> str:
    dt = datetime.fromisoformat(base.replace("Z", "+00:00"))
    dt += timedelta(minutes=offset_minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(base: str, offset: float, level: str, svc: str, msg: str) -> Dict[str, str]:
    return {"timestamp": _ts(base, offset), "level": level,
            "service": svc, "message": msg}


def _mp(base: str, offset: float, val: float) -> Dict[str, Any]:
    return {"timestamp": _ts(base, offset), "value": round(val, 2)}


def _noisy(rng: random.Random, base: List[float], noise: float = 0.15) -> List[float]:
    """Add seeded noise to a list of metric values."""
    return [max(0.0, v * (1 + rng.uniform(-noise, noise))) for v in base]


# ───────────────────────────────────────────────────────────────────
# Healthy baseline generators
# ───────────────────────────────────────────────────────────────────

_HEALTHY_CPU = [14, 16, 13, 15, 14, 17, 15, 14, 16, 13, 15, 14, 16, 15, 14]
_HEALTHY_MEM = [42, 43, 42, 44, 43, 43, 42, 44, 43, 42, 43, 44, 43, 42, 43]
_HEALTHY_LAT = [45, 52, 38, 67, 43, 55, 41, 48, 62, 39, 44, 51, 47, 42, 50]
_HEALTHY_ERR = [0.1, 0.0, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1]
_HEALTHY_RPS = [340, 355, 362, 348, 371, 359, 345, 367, 352, 341, 358, 366, 349, 354, 360]


def _build_healthy_metrics(base: str, rng: random.Random) -> Dict[str, List[Dict]]:
    return {
        "cpu_usage":          [_mp(base, -14 + i, v) for i, v in enumerate(_noisy(rng, _HEALTHY_CPU))],
        "memory_usage":       [_mp(base, -14 + i, v) for i, v in enumerate(_noisy(rng, _HEALTHY_MEM))],
        "request_latency_ms": [_mp(base, -14 + i, v) for i, v in enumerate(_noisy(rng, _HEALTHY_LAT))],
        "error_rate":         [_mp(base, -14 + i, v) for i, v in enumerate(_noisy(rng, _HEALTHY_ERR, 0.3))],
        "request_count":      [_mp(base, -14 + i, v) for i, v in enumerate(_noisy(rng, _HEALTHY_RPS))],
    }


def _build_healthy_logs(base: str, svc: str, rng: random.Random) -> List[Dict[str, str]]:
    """A handful of generic healthy INFO logs for any service."""
    ops = [
        "Health check passed",
        f"Request processed successfully",
        f"Routine maintenance completed",
        f"Connection pool status: healthy",
    ]
    logs = []
    for i in range(rng.randint(6, 10)):
        offset = -30 + i * 3 + rng.random()
        logs.append(_log(base, offset, "INFO", svc, rng.choice(ops)))
    return logs


def _build_default_deploys(base: str, svc: str) -> List[Dict[str, Any]]:
    return [
        {"deploy_id": f"deploy-{hash(svc) % 900 + 100:03d}", "service": svc,
         "timestamp": _ts(base, -10080), "version": DEFAULT_VERSIONS[svc],
         "description": "Routine maintenance release", "status": "success"},
    ]


def build_base_services(base: str, rng: random.Random) -> Dict[str, ServiceData]:
    services: Dict[str, ServiceData] = {}
    for name in SERVICE_NAMES:
        services[name] = ServiceData(
            name=name, status="healthy",
            dependencies=list(SERVICE_DEPENDENCIES[name]),
            config=copy.deepcopy(DEFAULT_CONFIGS[name]),
            logs=_build_healthy_logs(base, name, rng),
            metrics=_build_healthy_metrics(base, rng),
            deploy_history=_build_default_deploys(base, name),
            version=DEFAULT_VERSIONS[name],
            uptime_seconds=7 * 86400,
        )
    return services


# ═══════════════════════════════════════════════════════════════════
# Fault variant definitions
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConfigBugVariant:
    """Parameters for an easy (single-service config bug) scenario."""
    service: str
    config_key: str
    bad_value: Any
    good_value: Any
    error_keyword: str
    deploy_desc: str
    bumped_version: str

@dataclass
class CascadeVariant:
    """Parameters for a medium (cascading failure) scenario."""
    crashed_service: str
    crash_reason: str
    fallback_service: str
    fallback_message: str
    cascade_keyword: str
    bumped_version: str
    deploy_desc: str

@dataclass
class IntermittentVariant:
    """Parameters for a hard (intermittent external timeout) scenario."""
    root_service: str
    affected_service: str
    external_url: str
    pool_key: str
    pool_bad: int
    bumped_version: str
    deploy_desc: str


_EASY_VARIANTS = [
    ConfigBugVariant("payment-service", "db_max_connections", 1, 50,
                     "ConnectionPoolExhausted",
                     "Updated database connection pool settings for resource optimization", "2.3.1"),
    ConfigBugVariant("auth-service", "db_connection_timeout_ms", 100, 30000,
                     "ConnectionTimeoutError",
                     "Tuned database connection timeout for faster failure detection", "2.4.3"),
    ConfigBugVariant("order-service", "max_retry_attempts", 0, 3,
                     "NoRetryConfigured",
                     "Disabled retry logic to reduce duplicate processing", "2.1.4"),
    ConfigBugVariant("user-service", "db_max_connections", 1, 20,
                     "ConnectionPoolExhausted",
                     "Reduced connection pool for memory optimization", "1.8.1"),
]

_MEDIUM_VARIANTS = [
    CascadeVariant("cache-service", "Out of memory: process killed by OOM killer",
                   "auth-service",
                   "Cache unavailable, falling back to direct database query",
                   "oom", "1.5.0",
                   "Upgraded caching algorithm to adaptive-lru for improved hit rates"),
    CascadeVariant("payment-service", "Unhandled exception in payment processor thread",
                   "order-service",
                   "Payment service unavailable, order processing blocked",
                   "crash", "2.3.1",
                   "Refactored payment processing pipeline for async support"),
]

_HARD_VARIANTS = [
    IntermittentVariant("notification-service", "order-service",
                        "https://hooks.partner-api.com/orders",
                        "max_outbound_connections", 2, "3.2.0",
                        "Added external webhook integration for partner order notifications"),
    IntermittentVariant("payment-service", "order-service",
                        "https://fraud-check.example.com/v2/verify",
                        "max_outbound_connections", 2, "2.3.1",
                        "Added synchronous fraud-check API call before payment capture"),
]


# ═══════════════════════════════════════════════════════════════════
# EASY scenario generator
# ═══════════════════════════════════════════════════════════════════

def build_service_down(seed: int = 42) -> Tuple[Dict[str, ServiceData], List[Dict], Dict[str, Any]]:
    rng = random.Random(seed)
    variant = _EASY_VARIANTS[seed % len(_EASY_VARIANTS)]
    base = "2024-01-15T10:15:00Z"
    services = build_base_services(base, rng)

    svc = services[variant.service]
    svc.status = "degraded"
    svc.version = variant.bumped_version
    svc.config[variant.config_key] = variant.bad_value

    svc.logs = [
        _log(base, -20, "INFO", variant.service, f"Starting {variant.service} {variant.bumped_version}"),
        _log(base, -19, "INFO", variant.service, f"Config loaded: {variant.config_key}={variant.bad_value}"),
        _log(base, -18, "INFO", variant.service, "Health check passed"),
    ]
    for i, offset in enumerate([-12, -10, -8, -6, -4, -2.5, -1.5]):
        oid = f"REQ-{rng.randint(10000, 99999)}"
        if i < 2:
            svc.logs.append(_log(base, offset, "WARN", variant.service,
                f"{variant.config_key} limit reached: {variant.error_keyword} "
                f"({variant.bad_value}/{variant.bad_value} in use), request {oid} waiting"))
        else:
            svc.logs.append(_log(base, offset, "ERROR", variant.service,
                f"Request {oid} failed: {variant.error_keyword} — "
                f"{variant.config_key}={variant.bad_value} is insufficient under current load"))
    svc.logs.append(_log(base, -0.5, "WARN", variant.service,
        f"Health check degraded: {variant.error_keyword}"))

    svc.metrics["error_rate"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [0, 0, 0, 0, 0, 0, 2.1, 18.5, 42.3, 55.8, 61.2, 58.7, 63.1, 60.5, 62.8])]
    svc.metrics["request_latency_ms"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [85, 88, 82, 90, 86, 84, 920, 15200, 28500, 30000, 30000, 30000, 30000, 30000, 30000])]

    svc.deploy_history = [
        {"deploy_id": f"deploy-{rng.randint(100,999)}", "service": variant.service,
         "timestamp": _ts(base, -20), "version": variant.bumped_version,
         "description": variant.deploy_desc, "status": "success",
         "changed_configs": {variant.config_key: f"{variant.good_value} -> {variant.bad_value}"}},
    ] + _build_default_deploys(base, variant.service)

    downstream = REVERSE_DEPS.get(variant.service, [])
    for ds in downstream:
        services[ds].logs.extend([
            _log(base, -7, "ERROR", ds,
                 f"Upstream {variant.service} returned 500"),
            _log(base, -4, "ERROR", ds,
                 f"Upstream {variant.service} timeout"),
        ])
        services[ds].logs.sort(key=lambda x: x["timestamp"])

    alerts = [
        {"alert_id": f"ALT-{rng.randint(1000,9999)}", "service": variant.service,
         "severity": "critical",
         "message": f"{variant.service} error rate exceeded 50% (current: 62.8%)",
         "timestamp": _ts(base, -7), "acknowledged": False},
        {"alert_id": f"ALT-{rng.randint(1000,9999)}", "service": variant.service,
         "severity": "warning",
         "message": f"{variant.service} p99 latency exceeded 5000ms",
         "timestamp": _ts(base, -8), "acknowledged": False},
    ]
    if downstream:
        alerts.append({"alert_id": f"ALT-{rng.randint(1000,9999)}",
            "service": downstream[0], "severity": "warning",
            "message": f"{downstream[0]} experiencing elevated error rates",
            "timestamp": _ts(base, -5), "acknowledged": False})

    ground_truth: Dict[str, Any] = {
        "root_cause_service": variant.service,
        "root_cause_keywords": [
            variant.error_keyword.lower(), variant.config_key,
            "connection", "exhausted", "timeout", "limit",
        ],
        "correct_fixes": [
            {"action": "rollback_deploy", "service": variant.service},
            {"action": "update_config", "service": variant.service,
             "params_key": variant.config_key},
        ],
        "partial_fixes": [
            {"action": "restart_service", "service": variant.service},
        ],
        "critical_path_services": {variant.service},
        "max_steps": 30,
    }
    return services, alerts, ground_truth


# ═══════════════════════════════════════════════════════════════════
# MEDIUM scenario generator
# ═══════════════════════════════════════════════════════════════════

def build_cascading_failure(seed: int = 42) -> Tuple[Dict[str, ServiceData], List[Dict], Dict[str, Any]]:
    rng = random.Random(seed)
    variant = _MEDIUM_VARIANTS[seed % len(_MEDIUM_VARIANTS)]
    base = "2024-01-15T14:00:00Z"
    services = build_base_services(base, rng)

    # --- crashed service ---
    cs = services[variant.crashed_service]
    cs.status = "down"
    cs.version = variant.bumped_version
    cs.uptime_seconds = 0
    cs.logs = [
        _log(base, -35, "INFO", cs.name, f"Starting {cs.name} {variant.bumped_version}"),
        _log(base, -34, "INFO", cs.name, f"Deploy applied: {variant.deploy_desc}"),
        _log(base, -28, "WARN", cs.name, "Resource usage growing faster than expected"),
        _log(base, -22, "WARN", cs.name, "Resource usage approaching critical threshold"),
        _log(base, -18, "ERROR", cs.name, "Critical resource threshold reached"),
        _log(base, -16, "FATAL", cs.name, variant.crash_reason),
    ]
    cs.metrics["error_rate"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [0, 0, 0.5, 2.1, 5.8, 12.4, 28.3, 55.0, 100, 100, 100, 100, 100, 100, 100])]
    cs.metrics["cpu_usage"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [4, 8, 15, 25, 38, 52, 68, 78, 85, 92, 95, 0, 0, 0, 0])]
    cs.metrics["memory_usage"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [25, 31, 39, 48, 58, 67, 78, 84, 89, 93, 97, 0, 0, 0, 0])]
    cs.deploy_history = [
        {"deploy_id": f"deploy-{rng.randint(100,999)}", "service": cs.name,
         "timestamp": _ts(base, -35), "version": variant.bumped_version,
         "description": variant.deploy_desc, "status": "success"},
    ] + _build_default_deploys(base, cs.name)

    # --- fallback service ---
    fb = services[variant.fallback_service]
    fb.status = "degraded"
    fb.logs = [
        _log(base, -17, "WARN", fb.name,
             f"{cs.name} connection refused"),
        _log(base, -16, "WARN", fb.name, variant.fallback_message),
        _log(base, -14, "WARN", fb.name, "Database query time elevated"),
        _log(base, -10, "ERROR", fb.name, "Request timeout: overloaded"),
        _log(base, -6, "ERROR", fb.name, "Connection pool under pressure"),
        _log(base, -2, "ERROR", fb.name, "Connection pool exhausted"),
        _log(base, -1, "WARN", fb.name,
             f"Health check degraded: {cs.name} unavailable"),
    ]
    fb.metrics["request_latency_ms"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [22, 24, 25, 23, 180, 420, 680, 890, 1200, 1450, 1800, 2100, 2400, 2200, 2350])]
    fb.metrics["error_rate"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [0, 0, 0, 0, 0.5, 2.1, 5.8, 8.4, 12.3, 15, 18.2, 16.5, 19.1, 17.8, 18.5])]

    # --- gateway / upstream callers show latency ---
    for caller in REVERSE_DEPS.get(variant.fallback_service, []):
        gw = services[caller]
        gw.status = "degraded"
        gw.logs.extend([
            _log(base, -12, "WARN", caller,
                 f"Slow upstream response: {fb.name}"),
            _log(base, -6, "ERROR", caller,
                 f"Request timeout from {fb.name}"),
        ])
        gw.logs.sort(key=lambda x: x["timestamp"])
        gw.metrics["request_latency_ms"] = [
            _mp(base, -14 + i, v) for i, v in enumerate(
                [45, 48, 52, 44, 280, 550, 820, 1100, 1500, 1850, 2200, 2450, 2600, 2400, 2500])]

    alerts = [
        {"alert_id": f"ALT-{rng.randint(1000,9999)}", "service": fb.name,
         "severity": "critical",
         "message": f"{fb.name} error rate exceeded 15%",
         "timestamp": _ts(base, -8), "acknowledged": False},
        {"alert_id": f"ALT-{rng.randint(1000,9999)}", "service": cs.name,
         "severity": "critical",
         "message": f"{cs.name} health check failed — service unreachable",
         "timestamp": _ts(base, -16), "acknowledged": False},
    ]
    for caller in REVERSE_DEPS.get(variant.fallback_service, []):
        alerts.append({"alert_id": f"ALT-{rng.randint(1000,9999)}",
            "service": caller, "severity": "critical",
            "message": f"{caller} p99 latency exceeded 5000ms",
            "timestamp": _ts(base, -10), "acknowledged": False})

    ground_truth: Dict[str, Any] = {
        "root_cause_service": variant.crashed_service,
        "root_cause_keywords": [
            variant.cascade_keyword, "memory", "killed", "crash",
            "unavailable", "down", variant.crashed_service,
        ],
        "correct_fixes": [
            {"action": "rollback_deploy", "service": variant.crashed_service},
        ],
        "partial_fixes": [
            {"action": "restart_service", "service": variant.crashed_service},
        ],
        "critical_path_services": {variant.crashed_service, variant.fallback_service},
        "max_steps": 50,
    }
    return services, alerts, ground_truth


# ═══════════════════════════════════════════════════════════════════
# HARD scenario generator
# ═══════════════════════════════════════════════════════════════════

def build_phantom_bottleneck(seed: int = 42) -> Tuple[Dict[str, ServiceData], List[Dict], Dict[str, Any]]:
    rng = random.Random(seed)
    variant = _HARD_VARIANTS[seed % len(_HARD_VARIANTS)]
    base = "2024-01-15T16:00:00Z"
    services = build_base_services(base, rng)

    # --- root cause service (looks mostly healthy) ---
    rs = services[variant.root_service]
    rs.version = variant.bumped_version
    rs.config["external_webhook_url"] = variant.external_url
    rs.config["external_webhook_timeout_ms"] = 30000
    rs.config[variant.pool_key] = variant.pool_bad

    rs.logs = []
    for i in range(12):
        oid = f"REQ-{rng.randint(10000, 99999)}"
        offset = -30 + i * 2.5
        rs.logs.append(_log(base, offset, "INFO", rs.name,
            f"Processing request {oid}"))
        rs.logs.append(_log(base, offset + 0.5, "INFO", rs.name,
            f"Calling external endpoint {variant.external_url} for {oid}"))
        latency = rng.choice([950, 1200, 1800, 12450, 25890, 28543, 30000])
        if latency > 20000:
            rs.logs.append(_log(base, offset + 1, "WARN", rs.name,
                f"External call slow: {latency}ms for {oid}"))
        if latency >= 30000:
            rs.logs.append(_log(base, offset + 1.2, "ERROR", rs.name,
                f"External call timeout after 30000ms for {oid}"))
        if rng.random() < 0.3:
            rs.logs.append(_log(base, offset + 1.5, "WARN", rs.name,
                f"Outbound connection pool saturated "
                f"({variant.pool_bad}/{variant.pool_bad} in use), request queued"))
    rs.logs.append(_log(base, -1, "INFO", rs.name, "Health check passed"))

    rs.metrics["request_latency_ms"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [45, 48, rng.choice([1200, 52]), 52, rng.choice([12450, 48]),
             48, 980, 44, rng.choice([28543, 50]), 50,
             rng.choice([30000, 46]), 46, 2100, rng.choice([25890, 44]),
             rng.choice([30000, 48])])]
    rs.metrics["error_rate"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [0, 0, 0, 0, 0, 0, 0, 0,
             rng.uniform(5, 12), 0, rng.uniform(10, 20), 0,
             0, rng.uniform(8, 18), rng.uniform(12, 22)])]

    rs.deploy_history = [
        {"deploy_id": f"deploy-{rng.randint(100,999)}", "service": rs.name,
         "timestamp": _ts(base, -32), "version": variant.bumped_version,
         "description": variant.deploy_desc, "status": "success",
         "changed_configs": {
             "external_webhook_url": f"null -> {variant.external_url}",
             "external_webhook_timeout_ms": "null -> 30000",
             variant.pool_key: f"10 -> {variant.pool_bad}",
         }},
    ] + _build_default_deploys(base, rs.name)

    # --- affected service (intermittent failures) ---
    af = services[variant.affected_service]
    af.status = "degraded"
    af.logs = []
    for i in range(14):
        oid = f"ORD-{rng.randint(80000, 99999)}"
        offset = -28 + i * 2
        af.logs.append(_log(base, offset, "INFO", af.name,
            f"Processing {oid}"))
        if rng.random() < 0.25:
            af.logs.append(_log(base, offset + 1, "ERROR", af.name,
                f"Timeout calling {rs.name} for {oid} after 30000ms"))
            af.logs.append(_log(base, offset + 1.2, "ERROR", af.name,
                f"Failed to complete {oid}: upstream {rs.name} timeout"))
        else:
            af.logs.append(_log(base, offset + 1, "INFO", af.name,
                f"{oid} completed successfully"))
    af.logs.append(_log(base, -1, "WARN", af.name,
        f"Health check degraded: intermittent upstream {rs.name} timeouts"))

    af.metrics["error_rate"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [0.1, 0, 0.2, 0.1, rng.uniform(3, 8), 0.5, 0.2, 0.1,
             rng.uniform(15, 22), 0.3, rng.uniform(18, 25), 0.2,
             0.5, rng.uniform(12, 20), rng.uniform(16, 24)])]
    af.metrics["request_latency_ms"] = [
        _mp(base, -14 + i, v) for i, v in enumerate(
            [120, 130, 125, 118, rng.uniform(5000, 12000), 135, 122, 128,
             rng.uniform(20000, 30000), 140, 30000, 126,
             2200, rng.uniform(18000, 28000), 30000])]

    # --- red herring: gateway error bump ---
    gw = services["api-gateway"]
    gw.logs.extend([
        _log(base, -10, "WARN", "api-gateway",
             f"Upstream {af.name} returned 503"),
        _log(base, -1.5, "WARN", "api-gateway",
             f"Elevated 5xx rate from {af.name}"),
    ])
    gw.logs.sort(key=lambda x: x["timestamp"])

    # --- red herring: unrelated deploy on user-service ---
    us = services["user-service"]
    us.version = "1.8.1"
    us.deploy_history = [
        {"deploy_id": f"deploy-{rng.randint(100,999)}", "service": "user-service",
         "timestamp": _ts(base, -60), "version": "1.8.1",
         "description": "Added email verification resend endpoint",
         "status": "success"},
    ] + us.deploy_history

    alerts = [
        {"alert_id": f"ALT-{rng.randint(1000,9999)}", "service": af.name,
         "severity": "warning",
         "message": f"{af.name} error rate elevated (intermittent, ~20%)",
         "timestamp": _ts(base, -10), "acknowledged": False},
        {"alert_id": f"ALT-{rng.randint(1000,9999)}", "service": "api-gateway",
         "severity": "warning",
         "message": "api-gateway 503 error rate elevated",
         "timestamp": _ts(base, -5), "acknowledged": False},
        {"alert_id": f"ALT-{rng.randint(1000,9999)}", "service": "user-service",
         "severity": "info",
         "message": "user-service deploy completed: v1.8.1",
         "timestamp": _ts(base, -60), "acknowledged": True},
    ]

    ground_truth: Dict[str, Any] = {
        "root_cause_service": variant.root_service,
        "root_cause_keywords": [
            "webhook", "external", "timeout", "outbound",
            "connection pool", variant.pool_key, "synchronous",
            variant.external_url.split("/")[2],
        ],
        "correct_fixes": [
            {"action": "rollback_deploy", "service": variant.root_service},
        ],
        "partial_fixes": [
            {"action": "update_config", "service": variant.root_service,
             "params_key": variant.pool_key},
            {"action": "restart_service", "service": variant.root_service},
        ],
        "critical_path_services": {variant.root_service, variant.affected_service},
        "max_steps": 75,
    }
    return services, alerts, ground_truth
