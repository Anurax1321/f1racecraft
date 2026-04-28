"""Tests for sub-step 1.1 — race-length parametrization.

What we verify:
  - Backward compat: generate(family, seed) without race_length is unchanged.
  - Scaling: generate(family, seed, race_length=N) returns total_laps=N and
    scales lap-dependent fields proportionally.
  - Boundaries: lap fields always land in [1, total_laps].
  - Weather array length matches total_laps when present.
  - Physics-span fields (duration_laps, undercut_threshold_laps) DO NOT scale.
  - All 6 families scale to short/medium/long without errors.
  - Env.reset() honors race_length option and produces a coherent observation.
"""

from __future__ import annotations

import pytest

from server.environment import F1StrategistEnvironment
from server.generator import (
    RACE_LENGTHS,
    generate,
    scale_scenario_to_length,
)
from server.scenarios import SCENARIOS


FAMILIES = [
    "dry_strategy_sprint",
    "weather_roulette",
    "late_safety_car",
    "championship_decider",
    "virtual_safety_car_window",
    "tyre_cliff_management",
]


# ─────────────────────────────────────────────────────────────────────────
# Backward compatibility — race_length=None preserves old behavior
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("family", FAMILIES)
def test_generate_without_race_length_unchanged(family):
    """generate() without race_length keeps the family's native total_laps."""
    base = SCENARIOS[family]
    out = generate(family, seed=7)
    assert out["total_laps"] == base["total_laps"], \
        f"{family}: native total_laps changed (expected {base['total_laps']}, got {out['total_laps']})"


# ─────────────────────────────────────────────────────────────────────────
# Scaling correctness — total_laps lands exactly
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("length_key", ["short", "medium", "long"])
def test_generate_with_named_length(family, length_key):
    out = generate(family, seed=7, race_length=length_key)
    assert out["total_laps"] == RACE_LENGTHS[length_key], \
        f"{family}@{length_key}: total_laps mismatch"
    assert out["max_steps"] == RACE_LENGTHS[length_key] + 4, \
        f"{family}@{length_key}: max_steps must be total_laps + 4"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("target", [10, 25, 40, 55, 70])
def test_generate_with_int_length(family, target):
    out = generate(family, seed=3, race_length=target)
    assert out["total_laps"] == target


# ─────────────────────────────────────────────────────────────────────────
# Lap fields stay in bounds after scaling
# ─────────────────────────────────────────────────────────────────────────

def _all_lap_values(scenario: dict) -> list[tuple[str, int]]:
    """Walk a scenario and yield (label, lap_value) for every lap-bearing field."""
    out: list[tuple[str, int]] = []
    for opp in scenario.get("opponents", []):
        for stint in opp.get("planned_strategy", []):
            if "planned_end_lap" in stint:
                out.append((f"opponent#{opp.get('driver_number')}.planned_end_lap", stint["planned_end_lap"]))
    for issue in scenario.get("issues", []):
        if "valid_window" in issue and len(issue.get("valid_window", [])) == 2:
            out.append(("issue.valid_window[0]", issue["valid_window"][0]))
            out.append(("issue.valid_window[1]", issue["valid_window"][1]))
    sc = scenario.get("success_criteria", {})
    if "optimal_pit_window" in sc and len(sc.get("optimal_pit_window", [])) == 2:
        out.append(("sc.optimal_pit_window[0]", sc["optimal_pit_window"][0]))
        out.append(("sc.optimal_pit_window[1]", sc["optimal_pit_window"][1]))
    if "rain_peak_lap" in sc:
        out.append(("sc.rain_peak_lap", sc["rain_peak_lap"]))
    for ev in (scenario.get("weather_seed_overrides") or {}).get("sc_events", []) or []:
        if "lap" in ev:
            out.append(("weather.sc_event.lap", ev["lap"]))
    return out


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("length_key", ["short", "medium", "long"])
def test_lap_fields_in_bounds(family, length_key):
    target = RACE_LENGTHS[length_key]
    out = generate(family, seed=11, race_length=length_key)
    for label, lap in _all_lap_values(out):
        assert 1 <= lap <= target, \
            f"{family}@{length_key}: {label}={lap} out of bounds [1, {target}]"


# ─────────────────────────────────────────────────────────────────────────
# Weather per-lap array length tracks total_laps
# ─────────────────────────────────────────────────────────────────────────

def test_weather_array_length_matches_target():
    out = generate("weather_roulette", seed=5, race_length="long")
    per_lap = (out.get("weather_seed_overrides") or {}).get("per_lap", [])
    assert len(per_lap) == RACE_LENGTHS["long"], \
        f"weather per_lap length {len(per_lap)} != {RACE_LENGTHS['long']}"


def test_weather_array_preserves_arc_extremes():
    """Stretched weather should still start dry and peak at rain (Spa archetype)."""
    out = generate("weather_roulette", seed=5, race_length="long")
    per_lap = (out.get("weather_seed_overrides") or {}).get("per_lap", [])
    assert per_lap[0].get("rain_intensity", 0.0) <= 0.05  # opens dry
    peak = max(p.get("rain_intensity", 0.0) for p in per_lap)
    assert peak >= 0.3  # rain still happens somewhere in the stretched race


# ─────────────────────────────────────────────────────────────────────────
# Physics-span fields are NOT scaled
# ─────────────────────────────────────────────────────────────────────────

def test_sc_duration_does_not_scale():
    """SC duration in laps is a physics span — should stay constant when stretched."""
    base_short = generate("late_safety_car", seed=2, race_length="short")
    base_long = generate("late_safety_car", seed=2, race_length="long")
    short_dur = base_short["weather_seed_overrides"]["sc_events"][0]["duration_laps"]
    long_dur = base_long["weather_seed_overrides"]["sc_events"][0]["duration_laps"]
    assert short_dur == long_dur, \
        f"SC duration changed under scaling: {short_dur} -> {long_dur}"


# ─────────────────────────────────────────────────────────────────────────
# Direct scale_scenario_to_length() smoke
# ─────────────────────────────────────────────────────────────────────────

def test_scale_to_same_length_is_identity_for_total_laps():
    base = SCENARIOS["dry_strategy_sprint"]
    out = scale_scenario_to_length(base, base["total_laps"])
    assert out["total_laps"] == base["total_laps"]


def test_scale_invalid_target_raises():
    with pytest.raises(ValueError):
        scale_scenario_to_length(SCENARIOS["dry_strategy_sprint"], 0)


def test_starting_fuel_scales_with_race_length():
    """Starting fuel must scale up so long races don't always run out."""
    base = SCENARIOS["weather_roulette"]
    base_fuel = base["starting_fuel_kg"]
    base_laps = base["total_laps"]
    long_out = scale_scenario_to_length(base, RACE_LENGTHS["long"])
    expected = round(base_fuel * RACE_LENGTHS["long"] / base_laps, 2)
    assert long_out["starting_fuel_kg"] == expected, \
        f"starting_fuel didn't scale: {base_fuel} -> {long_out['starting_fuel_kg']}, expected {expected}"


# ─────────────────────────────────────────────────────────────────────────
# Env integration — reset() with race_length
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("length_key", ["short", "medium", "long"])
def test_env_reset_honors_race_length(family, length_key):
    target = RACE_LENGTHS[length_key]
    env = F1StrategistEnvironment()
    obs = env.reset(seed=7, options={"task": family, "race_length": length_key})
    assert obs.total_laps == target, \
        f"{family}@{length_key}: reset returned total_laps={obs.total_laps}, expected {target}"
    # current_lap is 0 right after reset (env starts at lap 0, advances on first step)
    assert 0 <= obs.current_lap < target
    assert obs.done is False


def test_env_reset_with_int_race_length():
    env = F1StrategistEnvironment()
    obs = env.reset(seed=1, options={"task": "weather_roulette", "race_length": 30})
    assert obs.total_laps == 30
