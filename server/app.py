"""
FastAPI application for the On-Call Incident Response Environment.

Exposes the OnCallEnvironment over HTTP endpoints compatible with
the OpenEnv spec (``/reset``, ``/step``, ``/state``, ``/schema``,
``/health``).

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from openenv.core.env_server.http_server import create_app

try:
    from models import OnCallAction, OnCallObservation
except ImportError:
    from ..models import OnCallAction, OnCallObservation

from .oncall_env import OnCallEnvironment

app = create_app(
    OnCallEnvironment,
    OnCallAction,
    OnCallObservation,
    env_name="oncall",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for ``uv run --project . server``."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
