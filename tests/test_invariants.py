"""Environment invariant tests.

These pin down properties of the simulator that should ALWAYS hold across
every action sequence and seed. If any of these fail, the env has a bug
that will hurt training (rewards become unreliable, agents exploit physics
violations).

Invariants tested:
    1. Tyre health monotonic non-increasing within a stint (only resets on pit)
    2. Fuel monotonic non-increasing always (no infinite gas)
    3. Lap counter strictly non-decreasing
    4. step_count == len(audit_trail) after any sequence of steps
    5. Episode terminates within total_laps + max_steps_buffer
    6. Done flag stays True once set (no resurrection)
    7. multi_objective_scores populated on done
    8. weighted_final ∈ [0.01, 0.99]
    9. Pit stop count == len(_ego_car.stints) − 1 == len(_pit_decisions)
   10. Inspection actions reveal new info on first call, 0 reward on subsequent
   11. Compound rule: exactly one stint = compound rule violated (dry race)
   12. n_pit_stops ≥ 0 monotonically
"""

from __future__ import annotations

import pytest

from models import F1Action, F1Observation
from server.environment import F1StrategistEnvironment
from server.scenarios import SCENARIOS


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"])
def env_with_episode(request):
    """Run a deterministic 30-step episode and return (env, frames)."""
    env = F1StrategistEnvironment()
    env.reset(seed=42, options={"scenario": SCENARIOS[request.param]})

    actions = [
        "INSPECT_TYRE_DEGRADATION",
        "ASSESS_UNDERCUT_WINDOW",
        "REQUEST_FORECAST",
        "STAY_OUT",
        "STAY_OUT",
        "PIT_NOW soft",
        "SET_MODE push",
        "STAY_OUT",
        "STAY_OUT",
        "STAY_OUT",
        "INSPECT_FUEL_MARGIN",
        "RADIO_DRIVER \"Bring it home\"",
        "STAY_OUT",
        "STAY_OUT",
        "STAY_OUT",
        "DONE",
    ]

    frames = []
    for cmd in actions:
        obs = env.step(F1Action(command=cmd))
        frames.append({
            "command": cmd,
            "obs": obs,
            "ego_tyre_health": env._ego_car.tyre_health,
            "ego_fuel_kg": env._ego_car.fuel_remaining_kg,
            "ego_pit_stops": env._ego_car.pit_stops,
            "ego_compound": env._ego_car.current_compound,
            "lap": env._lap,
            "step_count": env._step_count,
            "audit_len": len(env._audit_trail),
            "done": env._done,
        })
        if obs.done:
            break

    return env, frames, request.param


# ──────────────────────────────────────────────────────────────────────────
# Invariant 1: tyre health monotonic within stint
# ──────────────────────────────────────────────────────────────────────────

def test_tyre_health_monotonic_within_stint(env_with_episode):
    """Within a stint, tyre health only goes down. On a pit, the next health
    reading is high (fresh tyres minus one lap of wear ≈ 0.90+)."""
    env, frames, family = env_with_episode
    prev_health = 1.0
    prev_compound = None
    for f in frames:
        if f["ego_compound"] != prev_compound:
            # Pit happened — health is fresh-minus-one-lap (typically 0.90+)
            assert f["ego_tyre_health"] >= 0.85, (
                f"{family}: after pit, tyre_health={f['ego_tyre_health']} "
                f"should be ≥0.85 (fresh minus 1 lap of wear)"
            )
            prev_compound = f["ego_compound"]
            prev_health = f["ego_tyre_health"]
        else:
            assert f["ego_tyre_health"] <= prev_health + 1e-6, (
                f"{family}: tyre health went UP within stint: "
                f"{prev_health} → {f['ego_tyre_health']} on {f['command']}"
            )
            prev_health = f["ego_tyre_health"]


# ──────────────────────────────────────────────────────────────────────────
# Invariant 2: fuel monotonic non-increasing
# ──────────────────────────────────────────────────────────────────────────

def test_fuel_monotonic_non_increasing(env_with_episode):
    env, frames, family = env_with_episode
    prev_fuel = float("inf")
    for f in frames:
        # Skip the initial pit-tick (no lap consumed yet)
        if f["lap"] == 0:
            continue
        assert f["ego_fuel_kg"] <= prev_fuel + 1e-6, (
            f"{family}: fuel went UP from {prev_fuel} to {f['ego_fuel_kg']} on {f['command']}"
        )
        prev_fuel = f["ego_fuel_kg"]


# ──────────────────────────────────────────────────────────────────────────
# Invariant 3: lap counter non-decreasing
# ──────────────────────────────────────────────────────────────────────────

def test_lap_counter_non_decreasing(env_with_episode):
    env, frames, family = env_with_episode
    prev_lap = 0
    for f in frames:
        assert f["lap"] >= prev_lap, (
            f"{family}: lap went backward from {prev_lap} to {f['lap']}"
        )
        prev_lap = f["lap"]


# ──────────────────────────────────────────────────────────────────────────
# Invariant 4: step_count == len(audit_trail)
# ──────────────────────────────────────────────────────────────────────────

def test_step_count_matches_audit_trail(env_with_episode):
    env, frames, family = env_with_episode
    for f in frames:
        assert f["step_count"] == f["audit_len"], (
            f"{family}: step_count={f['step_count']} but audit_trail has {f['audit_len']} items"
        )


# ──────────────────────────────────────────────────────────────────────────
# Invariant 5: episode terminates within budget
# ──────────────────────────────────────────────────────────────────────────

def test_episode_terminates_in_budget():
    env = F1StrategistEnvironment()
    obs = env.reset(seed=0)
    max_iter = obs.total_laps + 50  # generous buffer
    iters = 0
    while not obs.done and iters < max_iter:
        obs = env.step(F1Action(command="STAY_OUT"))
        iters += 1
    assert obs.done, f"episode did not terminate in {max_iter} iterations"


# ──────────────────────────────────────────────────────────────────────────
# Invariant 6: done is sticky
# ──────────────────────────────────────────────────────────────────────────

def test_done_stays_done():
    env = F1StrategistEnvironment()
    env.reset(seed=0)
    env.step(F1Action(command="DONE"))
    assert env._done is True
    # Stepping after done should not unset done
    obs = env.step(F1Action(command="STAY_OUT"))
    assert obs.done is True
    obs = env.step(F1Action(command="PIT_NOW soft"))
    assert obs.done is True


# ──────────────────────────────────────────────────────────────────────────
# Invariant 7: scores populated on done
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "task",
    ["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"],
)
def test_final_scores_populated_on_done(task):
    env = F1StrategistEnvironment()
    env.reset(seed=0, options={"scenario": SCENARIOS[task]})
    obs = env.step(F1Action(command="DONE"))
    assert obs.done is True
    assert "weighted_final" in obs.multi_objective_scores
    expected_dims = {
        "race_result", "strategic_decisions", "tyre_management",
        "fuel_management", "comms_quality", "operational_efficiency",
    }
    assert expected_dims.issubset(obs.multi_objective_scores.keys())


# ──────────────────────────────────────────────────────────────────────────
# Invariant 8: weighted_final clamped
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "task",
    ["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"],
)
def test_weighted_final_in_bounds(task):
    """Across worst-case (DONE immediately) and a 30-action run, weighted_final stays in [0.01, 0.99]."""
    # Worst case
    env = F1StrategistEnvironment()
    env.reset(seed=7, options={"scenario": SCENARIOS[task]})
    obs = env.step(F1Action(command="DONE"))
    wf = obs.multi_objective_scores.get("weighted_final", 0.0)
    assert 0.01 <= wf <= 0.99

    # Mid case
    env = F1StrategistEnvironment()
    env.reset(seed=7, options={"scenario": SCENARIOS[task]})
    for _ in range(30):
        obs = env.step(F1Action(command="STAY_OUT"))
        if obs.done:
            break
    wf = obs.multi_objective_scores.get("weighted_final", 0.0)
    assert 0.01 <= wf <= 0.99


# ──────────────────────────────────────────────────────────────────────────
# Invariant 9: pit-stop bookkeeping consistent
# ──────────────────────────────────────────────────────────────────────────

def test_pit_stop_bookkeeping():
    env = F1StrategistEnvironment()
    env.reset(seed=0, options={"scenario": SCENARIOS["dry_strategy_sprint"]})
    env.step(F1Action(command="STAY_OUT"))
    assert env._ego_car.pit_stops == 0
    assert len(env._pit_decisions) == 0
    env.step(F1Action(command="PIT_NOW soft"))
    assert env._ego_car.pit_stops == 1
    assert len(env._pit_decisions) == 1
    # Stints: initial medium + soft = 2 entries
    assert len(env._ego_car.stints) == 2


# ──────────────────────────────────────────────────────────────────────────
# Invariant 10: inspection reveals
# ──────────────────────────────────────────────────────────────────────────

def test_first_inspect_pays_repeat_does_not():
    env = F1StrategistEnvironment()
    env.reset(seed=0, options={"scenario": SCENARIOS["dry_strategy_sprint"]})
    obs1 = env.step(F1Action(command="INSPECT_TYRE_DEGRADATION"))
    assert obs1.reward >= 0.02 - 1e-6, f"first inspect reward {obs1.reward} not ≥0.02"
    obs2 = env.step(F1Action(command="INSPECT_TYRE_DEGRADATION"))
    # Second call: reward must be lower than first
    assert obs2.reward < obs1.reward + 1e-6


def test_all_inspections_reveal_some_info():
    env = F1StrategistEnvironment()
    env.reset(seed=0, options={"scenario": SCENARIOS["dry_strategy_sprint"]})
    for cmd in [
        "INSPECT_TYRE_DEGRADATION",
        "REQUEST_FORECAST",
        "ASSESS_UNDERCUT_WINDOW",
        "INSPECT_FUEL_MARGIN",
        "CHECK_OPPONENT_STRATEGY 16",
    ]:
        obs = env.step(F1Action(command=cmd))
        # uncertainty_alerts is the field the env populates with revelations
        assert obs.uncertainty_alerts, f"no revelation produced by {cmd}"


# ──────────────────────────────────────────────────────────────────────────
# Invariant 11: compound rule logic
# ──────────────────────────────────────────────────────────────────────────

def test_compound_rule_violated_with_one_stint():
    env = F1StrategistEnvironment()
    env.reset(seed=0, options={"scenario": SCENARIOS["dry_strategy_sprint"]})
    # Run to end with NO pit stops — single stint
    obs = None
    for _ in range(20):
        obs = env.step(F1Action(command="STAY_OUT"))
        if obs.done:
            break
    # _compound_rule_satisfied should be False (only "medium" used)
    assert not env._compound_rule_satisfied()


def test_compound_rule_satisfied_after_pit():
    env = F1StrategistEnvironment()
    env.reset(seed=0, options={"scenario": SCENARIOS["dry_strategy_sprint"]})
    env.step(F1Action(command="STAY_OUT"))
    env.step(F1Action(command="PIT_NOW soft"))  # medium → soft
    assert env._compound_rule_satisfied()


def test_inter_compound_satisfies_rule_alone():
    """If inters were used at any point, the dry-2-compound rule is bypassed (wet race)."""
    env = F1StrategistEnvironment()
    env.reset(seed=0, options={"scenario": SCENARIOS["weather_roulette"]})
    env.step(F1Action(command="STAY_OUT"))
    env.step(F1Action(command="PIT_NOW inter"))
    assert env._compound_rule_satisfied()


# ──────────────────────────────────────────────────────────────────────────
# Invariant 12: invalid commands penalised
# ──────────────────────────────────────────────────────────────────────────

def test_invalid_command_penalty():
    env = F1StrategistEnvironment()
    env.reset(seed=0)
    obs = env.step(F1Action(command="ZZZZ_NOPE"))
    assert obs.reward == -0.02
    assert env._invalid_actions >= 1


def test_empty_command_penalty():
    env = F1StrategistEnvironment()
    env.reset(seed=0)
    obs = env.step(F1Action(command=""))
    assert obs.reward == -0.02


def test_invalid_pit_compound_penalty():
    env = F1StrategistEnvironment()
    env.reset(seed=0)
    obs = env.step(F1Action(command="PIT_NOW gravel"))
    assert obs.reward == -0.02


# ──────────────────────────────────────────────────────────────────────────
# Invariant 13: reset() restores clean state
# ──────────────────────────────────────────────────────────────────────────

def test_reset_clears_audit_and_state():
    env = F1StrategistEnvironment()
    env.reset(seed=0)
    env.step(F1Action(command="PIT_NOW soft"))
    env.step(F1Action(command="STAY_OUT"))
    assert env._step_count == 2
    assert len(env._audit_trail) == 2

    env.reset(seed=0)
    assert env._step_count == 0
    assert len(env._audit_trail) == 0
    assert env._done is False
    assert env._lap == 0
    assert env._ego_car.pit_stops == 0


# ──────────────────────────────────────────────────────────────────────────
# Invariant 14: same seed → same physics outcome
# ──────────────────────────────────────────────────────────────────────────

def test_seed_determinism():
    """Two envs at the same seed running the same actions produce identical rewards."""
    actions = ["INSPECT_TYRE_DEGRADATION", "STAY_OUT", "PIT_NOW soft", "STAY_OUT", "DONE"]

    def run():
        env = F1StrategistEnvironment()
        env.reset(seed=42, options={"scenario": SCENARIOS["dry_strategy_sprint"]})
        rewards = []
        for cmd in actions:
            obs = env.step(F1Action(command=cmd))
            rewards.append(round(obs.reward, 6))
            if obs.done:
                break
        return rewards, obs.multi_objective_scores.get("weighted_final", 0.0)

    rewards1, score1 = run()
    rewards2, score2 = run()
    assert rewards1 == rewards2, f"reward sequences differ: {rewards1} vs {rewards2}"
    assert abs(score1 - score2) < 1e-6, f"final scores differ: {score1} vs {score2}"


# ──────────────────────────────────────────────────────────────────────────
# Invariant 15: track loader works for every scenario's track
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "task",
    ["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"],
)
def test_track_loads_for_every_scenario(task):
    """Every scenario's track_name must resolve to a valid Track object."""
    env = F1StrategistEnvironment()
    env.reset(seed=0, options={"scenario": SCENARIOS[task]})
    assert env._track is not None
    assert env._track.length_m > 0
    assert env._track.base_lap_time_s > 0
    assert len(env._track.centerline) > 10
    assert env._track.track_character in {
        "power", "balanced", "downforce", "street", "weather_prone"
    }