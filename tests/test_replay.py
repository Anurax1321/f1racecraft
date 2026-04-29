"""Tests for replay mode (sub-step 2.5).

Skips if there's no extracted race data (clean clone before pull/extract).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models import F1Action


def _have_extracted_data() -> bool:
    return Path("data/grounded/raw").exists() and any(
        Path("data/grounded/raw").rglob("*.json")
    )


pytestmark = pytest.mark.skipif(
    not _have_extracted_data(),
    reason="data/grounded/raw/ empty — run scripts/fastf1_extract_all.py first",
)


def test_list_replays_returns_year_gp_pairs():
    from server.replay import list_replays
    rs = list_replays()
    assert rs, "no replays found"
    assert all(isinstance(y, int) and isinstance(g, str) for y, g in rs)


def test_load_monaco_2024_lec():
    from server.replay import load_replay_scenario
    scenario, actual = load_replay_scenario(2024, "monaco", ego_driver="LEC")
    assert scenario["task_name"] == "replay_2024_monaco_LEC"
    assert scenario["total_laps"] == 78  # real Monaco GP length
    assert scenario["use_grounded_calibration"] is True
    assert scenario["grounded_track_key"] == "monaco"
    assert len(scenario["opponents"]) <= 5
    # Opponent driver_numbers must be ints (env requires this)
    for o in scenario["opponents"]:
        assert isinstance(o["driver_number"], int)
    # LEC actually pitted at Monaco 2024 (1-stop)
    assert actual, "LEC must have at least one pit decision"
    assert actual[0]["action"].startswith("PIT_NOW")


def test_replay_scenario_runs_in_env():
    """Replay scenario must successfully reset and step the env."""
    from server.environment import F1StrategistEnvironment
    from server.replay import load_replay_scenario

    scenario, _ = load_replay_scenario(2024, "monaco", ego_driver="LEC")
    env = F1StrategistEnvironment()
    obs = env.reset(seed=7, options={"scenario": scenario})
    assert obs.total_laps == 78
    assert env._grounded_factors is not None  # opt-in is on for replay
    # Step a few times
    for _ in range(5):
        obs = env.step(F1Action(command="STAY_OUT"))
        if obs.done:
            break
    # Either still running or terminated cleanly
    assert obs.current_lap >= 1


def test_unknown_replay_raises():
    from server.replay import load_replay_scenario
    with pytest.raises(FileNotFoundError):
        load_replay_scenario(1999, "nonexistent_gp")


def test_auto_pick_ego_when_driver_omitted():
    """Without an explicit driver, replay picks one near target_finish."""
    from server.replay import load_replay_scenario
    scenario, actual = load_replay_scenario(2024, "monaco")
    assert "ego_driver" in scenario["_replay_meta"]
    assert len(scenario["_replay_meta"]["ego_driver"]) == 3  # 3-letter code


def test_replay_uses_grounded_calibration():
    """Replay scenarios must opt in — that's the whole point of replay."""
    from server.replay import load_replay_scenario
    scenario, _ = load_replay_scenario(2024, "monaco", ego_driver="LEC")
    assert scenario.get("use_grounded_calibration") is True
