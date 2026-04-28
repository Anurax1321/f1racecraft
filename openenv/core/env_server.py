"""
Compatibility shim for OpenEnv.

The installed openenv-core package on this machine does not expose:
    openenv.core.env_server

This project imports:
    from openenv.core.env_server import Action, Observation, State, Environment, create_fastapi_app

This shim provides the small API surface the F1 simulator needs.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ConfigDict


A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


class Action(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class Observation(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class State(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class Environment(Generic[A, O, S]):
    def reset(self, *args: Any, **kwargs: Any) -> O:
        raise NotImplementedError

    def step(self, action: A) -> O:
        raise NotImplementedError


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def create_app(
    env_factory: Callable[[], Environment],
    action_model: type[Action],
    observation_model: type[Observation] | None = None,
    **kwargs: Any,
) -> "FastAPI":
    """Alias kept for server/app.py compatibility."""
    return create_fastapi_app(env_factory, action_model, observation_model)


def create_fastapi_app(
    env_factory: Callable[[], Environment],
    action_model: type[Action],
    observation_model: type[Observation] | None = None,
):
    from fastapi import FastAPI

    app = FastAPI()
    env = env_factory()

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/reset")
    def reset(payload: dict[str, Any] | None = None):
        payload = payload or {}
        try:
            obs = env.reset(**payload)
        except TypeError:
            try:
                obs = env.reset(payload)
            except TypeError:
                obs = env.reset()
        return _dump(obs)

    @app.post("/step")
    def step(payload: dict[str, Any]):
        action = action_model(**payload)
        obs = env.step(action)
        return _dump(obs)

    return app
