"""Pytest: F1StrategistEnvironment basic invariants."""

from models import F1Action, F1Observation
from server.environment import F1StrategistEnvironment


def test_reset_returns_valid_observation():
    env = F1StrategistEnvironment()
    obs = env.reset()
    assert isinstance(obs, F1Observation)
    assert obs.total_laps > 0
    assert obs.total_issues_count > 0


def test_step_unknown_command_penalises():
    env = F1StrategistEnvironment()
    env.reset()
    obs = env.step(F1Action(command="ZZZZ"))
    assert obs.reward == -0.02
    assert obs.done is False


def test_step_done_flips_done_true():
    env = F1StrategistEnvironment()
    env.reset()
    obs = env.step(F1Action(command="DONE"))
    assert obs.done is True
    assert "weighted_final" in obs.multi_objective_scores


def test_episode_terminates_at_total_laps():
    env = F1StrategistEnvironment()
    obs = env.reset()
    for _ in range(obs.total_laps + 2):
        obs = env.step(F1Action(command="STAY_OUT"))
        if obs.done:
            break
    assert obs.done is True
