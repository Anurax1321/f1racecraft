"""Procedural scenario variants for training coverage.

Two roles:
  - generate(family, seed, ...) — seed-deterministic variant of a hand-authored scenario.
  - scale_scenario_to_length(scenario, target_laps) — stretch/shrink a scenario
    to a different race length while preserving its strategy archetype.

Race lengths:
  short  =  12 laps  (sprint format, current default)
  medium =  25 laps  (mid-distance)
  long   =  55 laps  (full grand prix)
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from server.scenarios import SCENARIOS


RACE_LENGTHS: dict[str, int] = {
    "short": 12,
    "medium": 25,
    "long": 55,
}


# ---------------------------------------------------------------------------
# Race-length scaling
# ---------------------------------------------------------------------------

# Story-relative single-lap fields scale linearly with race length.
_LAP_FIELDS_SCALE = (
    "rain_peak_lap",
    "rain_start_lap",
    "vsc_lap",
    "planned_end_lap",
)

# Two-element [start, end] windows that scale.
_WINDOW_FIELDS_SCALE = (
    "optimal_pit_window",
    "valid_window",
)

# Physics-relative spans — tied to compound/stint physics, not race progression.
# Deliberately do NOT scale (here for documentation; not actively read).
_PHYSICS_SPAN_FIELDS_KEEP = (
    "duration_laps",
    "undercut_threshold_laps",
    "cliff_lap_soft",
)


def _rescale_lap(lap: int | float, ratio: float, target_laps: int) -> int:
    """Map a source-lap number into the target race length, clipped to [1, target]."""
    return max(1, min(target_laps, int(round(float(lap) * ratio))))


def _stretch_per_lap_array(per_lap: list[dict], target_laps: int) -> list[dict]:
    """Stretch a per-lap array (e.g. weather) to ``target_laps`` entries.

    Maps target lap index ``t`` back to a source index proportionally, so the
    weather arc preserves its overall shape regardless of new race length.
    Entries are deep-copied so callers can mutate independently.
    """
    if not per_lap:
        return per_lap
    src_n = len(per_lap)
    if target_laps <= 0:
        return []
    out: list[dict] = []
    for t in range(target_laps):
        if target_laps == 1:
            src_idx = 0
        else:
            src_idx = min(src_n - 1, int(round(t * (src_n - 1) / (target_laps - 1))))
        out.append(copy.deepcopy(per_lap[src_idx]))
    return out


def scale_scenario_to_length(scenario: dict, target_laps: int) -> dict:
    """Return a deep-copied scenario scaled to ``target_laps``.

    Fields that scale (linearly with race length): pit windows, rain
    peak/start, VSC lap, opponent ``planned_end_lap``, scripted event lap
    numbers, weather per-lap arrays, dynamic_events lap triggers.

    Fields that do NOT scale: tyre cliff lap (compound physics), SC duration,
    undercut threshold (race physics).
    """
    if target_laps < 1:
        raise ValueError(f"target_laps must be >= 1, got {target_laps}")

    scaled = copy.deepcopy(scenario)
    src_laps = int(scenario.get("total_laps", 12))
    if src_laps == target_laps:
        return scaled
    ratio = target_laps / src_laps

    def rescale(lap: int | float) -> int:
        return _rescale_lap(lap, ratio, target_laps)

    scaled["total_laps"] = target_laps
    # max_steps grows with the race; keep the +4 buffer the env expects.
    scaled["max_steps"] = target_laps + 4

    # Starting fuel scales with race length (more laps → more fuel needed).
    # Without this, a 55-lap race scaled from a 12-lap scenario runs out at lap ~50.
    if "starting_fuel_kg" in scaled:
        scaled["starting_fuel_kg"] = round(float(scaled["starting_fuel_kg"]) * ratio, 2)

    # Opponents — planned stint endings.
    for opp in scaled.get("opponents", []):
        for stint in opp.get("planned_strategy", []):
            if "planned_end_lap" in stint:
                stint["planned_end_lap"] = rescale(stint["planned_end_lap"])

    # Issues — pit windows and any scaling lap fields.
    for issue in scaled.get("issues", []):
        for f in _WINDOW_FIELDS_SCALE:
            if f in issue and isinstance(issue[f], (list, tuple)) and len(issue[f]) == 2:
                issue[f] = [rescale(issue[f][0]), rescale(issue[f][1])]
        for f in _LAP_FIELDS_SCALE:
            if f in issue and isinstance(issue[f], (int, float)):
                issue[f] = rescale(issue[f])

    # Success criteria.
    sc = scaled.get("success_criteria", {})
    for f in _WINDOW_FIELDS_SCALE:
        if f in sc and isinstance(sc[f], (list, tuple)) and len(sc[f]) == 2:
            sc[f] = [rescale(sc[f][0]), rescale(sc[f][1])]
    for f in _LAP_FIELDS_SCALE:
        if f in sc and isinstance(sc[f], (int, float)):
            sc[f] = rescale(sc[f])

    # Hidden state — opponent action overrides reference lap numbers too.
    hidden = scaled.get("hidden_state", {})
    for stints in hidden.get("opponent_action_overrides", {}).values():
        if isinstance(stints, list):
            for stint in stints:
                if isinstance(stint, dict) and "planned_end_lap" in stint:
                    stint["planned_end_lap"] = rescale(stint["planned_end_lap"])

    # Weather overrides — per-lap array + sc_events list.
    overrides = scaled.get("weather_seed_overrides") or {}
    if "per_lap" in overrides and isinstance(overrides["per_lap"], list):
        overrides["per_lap"] = _stretch_per_lap_array(overrides["per_lap"], target_laps)
    for ev in overrides.get("sc_events", []) or []:
        if isinstance(ev, dict) and "lap" in ev:
            ev["lap"] = rescale(ev["lap"])
        # duration_laps stays as-is (physics span)
    if overrides:
        scaled["weather_seed_overrides"] = overrides

    # Dynamic events — generic lap-triggered events at top level.
    for ev in scaled.get("dynamic_events", []) or []:
        if isinstance(ev, dict) and "lap" in ev:
            ev["lap"] = rescale(ev["lap"])

    return scaled


# ---------------------------------------------------------------------------
# Procedural scenario generation
# ---------------------------------------------------------------------------

def generate(
    family: str,
    seed: int,
    difficulty: str = "medium",
    race_length: Optional[int | str] = None,
) -> dict:
    """Return a seed-deterministic variant of a hand-authored scenario.

    Args:
        family: scenario family key (e.g. ``"weather_roulette"``).
        seed: seed for procedural perturbations.
        difficulty: ``"easy"`` | ``"medium"`` | ``"hard"`` — opponent-pace spread.
        race_length: if set, scale the result to this length. Accepts an int
            (target laps) or a string key from ``RACE_LENGTHS`` (``"short"``,
            ``"medium"``, ``"long"``). ``None`` keeps the family's native length.
    """
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

    if race_length is not None:
        target = RACE_LENGTHS[race_length] if isinstance(race_length, str) else int(race_length)
        base = scale_scenario_to_length(base, target)

    return base
