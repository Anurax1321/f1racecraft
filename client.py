"""
F1 Strategist — EnvClient
==========================
Thin subclass of openenv.core.env_client.EnvClient wired for F1 Strategist types.
Direct port of OpsTwin's client.py shape.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import F1Action, F1Observation, F1State


class F1StrategistEnv(EnvClient[F1Action, F1Observation, F1State]):
    """Client for the F1 Strategist server."""

    def _step_payload(self, action: F1Action) -> dict:
        return {"command": action.command}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {}) or {}
        return StepResult(
            observation=F1Observation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                **{
                    k: v
                    for k, v in obs_data.items()
                    if k in F1Observation.model_fields and k not in ("done", "reward")
                },
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> F1State:
        return F1State(**{k: v for k, v in payload.items() if k in F1State.model_fields})
