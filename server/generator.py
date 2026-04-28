"""Procedural scenario variants for training coverage."""

import copy

import numpy as np

from server.scenarios import SCENARIOS


def generate(family: str, seed: int, difficulty: str = "medium") -> dict:
    """Return a seed-deterministic variant of a hand-authored scenario."""
    if family not in SCENARIOS:
        raise ValueError(f"Unknown family: {family}")

    rng = np.random.default_rng(seed)
    base = copy.deepcopy(SCENARIOS[family])
    base["seed"] = int(seed)
    base["task_name"] = f"{base['scenario_family']}_{seed}"

    difficulty_spread = {"easy": 0.05, "medium": 0.12, "hard": 0.22}.get(difficulty, 0.12)
    base["starting_position"] = int(max(1, base.get("starting_position", 4) + rng.integers(-1, 2)))
    base["starting_fuel_kg"] = round(
        float(base.get("starting_fuel_kg", 90.0)) + rng.normal(0.0, 1.0), 2
    )

    for opponent in base.get("opponents", []):
        opponent["pace_offset_s"] = round(
            float(opponent.get("pace_offset_s", 0.0)) + rng.normal(0.0, difficulty_spread), 3
        )
        opponent["aggression"] = round(
            float(
                np.clip(float(opponent.get("aggression", 0.5)) + rng.normal(0.0, 0.04), 0.1, 0.95)
            ),
            3,
        )

    family_name = base.get("scenario_family", family)
    if family_name == "weather_roulette":
        overrides = base.setdefault("weather_seed_overrides", {})
        shift = int(rng.integers(-1, 2))
        for row in overrides.get("per_lap", []):
            if row.get("rain_intensity", 0.0) > 0:
                row["rain_intensity"] = float(
                    np.clip(row["rain_intensity"] + rng.normal(0.0, 0.03), 0.0, 0.8)
                )
        criteria = base.setdefault("success_criteria", {})
        if "rain_peak_lap" in criteria:
            criteria["rain_peak_lap"] = int(
                np.clip(criteria["rain_peak_lap"] + shift, 5, base["total_laps"])
            )
    elif family_name == "late_safety_car":
        overrides = base.setdefault("weather_seed_overrides", {})
        events = overrides.setdefault(
            "sc_events", [{"lap": 8, "sc_type": "full_sc", "duration_laps": 3}]
        )
        events[0]["lap"] = int(np.clip(8 + rng.integers(-1, 2), 7, 10))
        base.setdefault("success_criteria", {})["optimal_pit_window"] = [
            events[0]["lap"],
            min(base["total_laps"], events[0]["lap"] + 2),
        ]
    elif family_name == "dry_strategy_sprint":
        window = base.setdefault("success_criteria", {}).setdefault("optimal_pit_window", [4, 7])
        shift = int(rng.integers(-1, 2))
        base["success_criteria"]["optimal_pit_window"] = [
            max(3, window[0] + shift),
            min(base["total_laps"], window[1] + shift),
        ]

    return base
