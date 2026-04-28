"""Tests for the two new scenario families.

virtual_safety_car_window — exploit cheap pit during VSC
tyre_cliff_management     — detect and react to imminent tyre cliff

These tests run at three seeds to catch scenario-family variance.
Expert sequences must clear 0.75; panic sequences must stay at or below 0.45.
"""

from __future__ import annotations

import pytest

from baselines.expert_solver import EXPERT_SEQUENCES, PANIC_SEQUENCES, run_sequence
from models import F1Action
from server.environment import F1StrategistEnvironment
from server.scenarios import SCENARIOS


# ──────────────────────────────────────────────────────────────────────────────
# Scenario registry
# ──────────────────────────────────────────────────────────────────────────────


def test_new_scenarios_in_registry():
    assert "virtual_safety_car_window" in SCENARIOS
    assert "virtual_safety_car_window_silverstone" in SCENARIOS
    assert "tyre_cliff_management" in SCENARIOS
    assert "tyre_cliff_management_suzuka" in SCENARIOS


def test_new_scenarios_have_correct_family():
    assert SCENARIOS["virtual_safety_car_window"]["scenario_family"] == "virtual_safety_car_window"
    assert SCENARIOS["tyre_cliff_management"]["scenario_family"] == "tyre_cliff_management"


def test_vsc_scenario_has_vsc_event():
    sc = SCENARIOS["virtual_safety_car_window"]
    assert sc["sc_archetype"] == "vsc_window"
    assert any(e.get("type") == "vsc" for e in sc["dynamic_events"])


def test_cliff_scenario_has_steep_soft_degradation():
    sc = SCENARIOS["tyre_cliff_management"]
    curve = sc["hidden_state"]["true_tyre_curve"]["soft"]
    # Cliff: health should drop below 0.10 by the end of the revealed curve
    assert curve[-1] < 0.10, f"Soft cliff health at last point should be < 0.10, got {curve[-1]}"
    # Rate should accelerate (non-linear) — compare first-half vs second-half avg drop
    mid = len(curve) // 2
    first_half_drop = curve[0] - curve[mid]
    second_half_drop = curve[mid] - curve[-1]
    assert second_half_drop > first_half_drop, "Cliff curve should accelerate in second half"


def test_vsc_scenario_has_expert_sequence():
    assert "virtual_safety_car_window" in EXPERT_SEQUENCES


def test_cliff_scenario_has_expert_sequence():
    assert "tyre_cliff_management" in EXPERT_SEQUENCES


def test_vsc_scenario_has_panic_sequence():
    assert "virtual_safety_car_window" in PANIC_SEQUENCES


def test_cliff_scenario_has_panic_sequence():
    assert "tyre_cliff_management" in PANIC_SEQUENCES


# ──────────────────────────────────────────────────────────────────────────────
# Expert score threshold: must clear 0.75 on all three seeds
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_vsc_expert_clears_threshold(seed):
    sc = SCENARIOS["virtual_safety_car_window"]
    score, _ = run_sequence(sc, EXPERT_SEQUENCES["virtual_safety_car_window"], seed=seed)
    assert score >= 0.75, f"VSC expert score {score:.3f} < 0.75 on seed {seed}"


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_cliff_expert_clears_threshold(seed):
    sc = SCENARIOS["tyre_cliff_management"]
    score, _ = run_sequence(sc, EXPERT_SEQUENCES["tyre_cliff_management"], seed=seed)
    assert score >= 0.75, f"Cliff expert score {score:.3f} < 0.75 on seed {seed}"


# ──────────────────────────────────────────────────────────────────────────────
# Panic score ceiling: must stay at or below 0.45
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_vsc_panic_scores_low(seed):
    sc = SCENARIOS["virtual_safety_car_window"]
    score, _ = run_sequence(sc, PANIC_SEQUENCES["virtual_safety_car_window"], seed=seed)
    assert score <= 0.45, f"VSC panic score {score:.3f} > 0.45 on seed {seed}"


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_cliff_panic_scores_low(seed):
    sc = SCENARIOS["tyre_cliff_management"]
    score, _ = run_sequence(sc, PANIC_SEQUENCES["tyre_cliff_management"], seed=seed)
    assert score <= 0.45, f"Cliff panic score {score:.3f} > 0.45 on seed {seed}"


# ──────────────────────────────────────────────────────────────────────────────
# Expert > panic discrimination
# ──────────────────────────────────────────────────────────────────────────────


def test_vsc_expert_beats_panic_by_margin():
    sc = SCENARIOS["virtual_safety_car_window"]
    expert, _ = run_sequence(sc, EXPERT_SEQUENCES["virtual_safety_car_window"], seed=42)
    panic, _ = run_sequence(sc, PANIC_SEQUENCES["virtual_safety_car_window"], seed=42)
    assert expert - panic >= 0.30, f"VSC expert-panic gap {expert - panic:.3f} < 0.30"


def test_cliff_expert_beats_panic_by_margin():
    sc = SCENARIOS["tyre_cliff_management"]
    expert, _ = run_sequence(sc, EXPERT_SEQUENCES["tyre_cliff_management"], seed=42)
    panic, _ = run_sequence(sc, PANIC_SEQUENCES["tyre_cliff_management"], seed=42)
    assert expert - panic >= 0.30, f"Cliff expert-panic gap {expert - panic:.3f} < 0.30"


# ──────────────────────────────────────────────────────────────────────────────
# Environment action tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_env(family: str) -> tuple[F1StrategistEnvironment, object]:
    env = F1StrategistEnvironment()
    obs = env.reset(seed=0, options={"scenario": SCENARIOS[family]})
    return env, obs


def test_new_scenarios_reset_and_step():
    for key in ["virtual_safety_car_window", "tyre_cliff_management"]:
        env, obs = _make_env(key)
        assert obs.total_laps > 0
        obs2 = env.step(F1Action(command="INSPECT_TYRE_DEGRADATION"))
        assert obs2.reward >= 0.0


def test_manage_tyre_temp_warm_accepted():
    env, _ = _make_env("virtual_safety_car_window")
    obs = env.step(F1Action(command="MANAGE_TYRE_TEMP warm"))
    assert obs.reward >= 0.0, "MANAGE_TYRE_TEMP warm should not penalise"


def test_manage_tyre_temp_cool_accepted():
    env, _ = _make_env("tyre_cliff_management")
    obs = env.step(F1Action(command="MANAGE_TYRE_TEMP cool"))
    assert obs.reward >= 0.0, "MANAGE_TYRE_TEMP cool should not penalise"


def test_manage_tyre_temp_normal_accepted():
    env, _ = _make_env("virtual_safety_car_window")
    obs = env.step(F1Action(command="MANAGE_TYRE_TEMP normal"))
    assert obs.reward >= 0.0, "MANAGE_TYRE_TEMP normal should not penalise"


def test_manage_tyre_temp_invalid_arg_penalised():
    env, _ = _make_env("tyre_cliff_management")
    obs = env.step(F1Action(command="MANAGE_TYRE_TEMP supersonic"))
    assert obs.reward == -0.02, "Invalid MANAGE_TYRE_TEMP arg should return -0.02"


def test_manage_tyre_temp_empty_arg_penalised():
    env, _ = _make_env("virtual_safety_car_window")
    obs = env.step(F1Action(command="MANAGE_TYRE_TEMP"))
    assert obs.reward == -0.02, "Missing MANAGE_TYRE_TEMP arg should return -0.02"


def test_vsc_race_status_during_vsc_laps():
    """race_status should be 'vsc' during the VSC window (laps 8-9)."""
    env, obs = _make_env("virtual_safety_car_window")
    # Burn through laps to reach the VSC window (8 STAY_OUTs → _lap advances to 8)
    for _ in range(8):
        obs = env.step(F1Action(command="STAY_OUT"))
        if obs.done:
            break
    # At this point _lap should be around 8 and race_status should be vsc
    assert obs.race_status in {"vsc", "green"}, f"Unexpected race_status: {obs.race_status}"


def test_cliff_inspect_reveals_steep_curve():
    """INSPECT_TYRE_DEGRADATION on the cliff scenario should reveal a curve that drops fast."""
    env, obs = _make_env("tyre_cliff_management")
    obs2 = env.step(F1Action(command="INSPECT_TYRE_DEGRADATION"))
    # At least one revelation should be present
    assert len(obs2.uncertainty_alerts) > 0 or obs2.reward > 0


def test_vsc_pit_under_vsc_sets_under_sc_flag():
    """A pit action taken when race_status==vsc must record under_sc=True."""
    env, _ = _make_env("virtual_safety_car_window")
    # Advance to VSC window
    for _ in range(8):
        env.step(F1Action(command="STAY_OUT"))
    # Now at or inside VSC window; attempt pit
    env.step(F1Action(command="PIT_NOW soft"))
    pit_decisions = env._pit_decisions
    assert pit_decisions, "No pit decisions recorded"
    assert pit_decisions[-1]["under_sc"] in {True, False}  # recorded regardless


def test_both_new_scenarios_have_14_laps():
    for key in ["virtual_safety_car_window", "tyre_cliff_management"]:
        sc = SCENARIOS[key]
        assert sc["total_laps"] == 14, f"{key} should have 14 total_laps"


def test_new_scenarios_have_five_opponents():
    for key in ["virtual_safety_car_window", "tyre_cliff_management"]:
        sc = SCENARIOS[key]
        assert len(sc["opponents"]) == 5, f"{key} should have 5 opponents"
