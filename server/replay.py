"""Replay-mode scenario builder (sub-step 2.5).

Converts a real FastF1 race (already extracted to
``data/grounded/raw/<year>/<round>_<gp>.json``) into a scenario dict the env
can run. The model strategist races *against the real conditions of that GP*
— same total laps, same weather arc, same SC events, same opponent stint
plans, same starting compounds.

Goal: a side-by-side comparison of "what the model called" vs "what the team
actually did." Demo / sanity-check material; not a training pipeline.

Design — what this module does and doesn't:
  - DOES: read extracted-race JSON, pick an ego driver, assemble a scenario
    dict the env can consume.
  - DOES: opt-in to grounded calibration so physics matches real F1 wear.
  - DOES NOT: modify the env. The replay scenario flows through the same
    reset/step path as any other scenario.
  - DOES NOT: simulate sector-by-sector telemetry. We use lap-aggregate
    stints and event timing.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _REPO_ROOT / "data" / "grounded" / "raw"


def _short_name_to_path(year: int, gp_short: str) -> Optional[Path]:
    """Find ``data/grounded/raw/<year>/<round>_<gp_short>.json``."""
    year_dir = _RAW_DIR / str(year)
    if not year_dir.exists():
        return None
    for f in year_dir.glob(f"*_{gp_short}.json"):
        return f
    return None


def list_replays() -> list[tuple[int, str]]:
    """Return (year, gp_short) for every extracted race available."""
    out = []
    for year_dir in _RAW_DIR.glob("*"):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        for f in year_dir.glob("*.json"):
            m = re.match(r"\d+_(.+)\.json", f.name)
            if m:
                out.append((year, m.group(1)))
    return sorted(out)


# ---------------------------------------------------------------------------
# Driver selection
# ---------------------------------------------------------------------------

def _ego_candidates_by_position(race: dict, target_finish: int = 5) -> list[str]:
    """Drivers ranked by closeness to a target finishing position.

    Replay is more interesting when ego doesn't start P1 (no decisions to
    make) or P20 (no chance). Mid-pack is where strategy matters.
    """
    final_positions: dict[str, int] = {}
    for r in race["lap_records"]:
        if r["position"] > 0:
            final_positions[r["driver"]] = r["position"]  # last seen position wins
    return sorted(
        final_positions.keys(),
        key=lambda d: abs(final_positions[d] - target_finish),
    )


def _ego_actual_actions(race: dict, driver: str) -> list[dict]:
    """The actual lap-by-lap actions for a driver — for comparison output."""
    actions: list[dict] = []
    pits = sorted(
        (p for p in race["pit_decisions"] if p["driver"] == driver),
        key=lambda p: p["lap"],
    )
    for p in pits:
        actions.append({
            "lap": p["lap"],
            "action": f"PIT_NOW {p['compound_out']}",
            "context": (
                "under_sc" if p["under_sc"]
                else "under_vsc" if p["under_vsc"]
                else "green"
            ),
        })
    return actions


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def _build_opponents(race: dict, ego: str, max_n: int = 5) -> list[dict]:
    """Build opponent dicts mimicking real F1 driver stint plans.

    Picks top finishers other than ego, since they're the strategically
    relevant ones. Each opponent's planned_strategy reflects their real
    sequence of compound→end_lap.
    """
    final_positions: dict[str, int] = {}
    teams: dict[str, str] = {}
    driver_numbers: dict[str, int] = {}
    for r in race["lap_records"]:
        if r["position"] > 0:
            final_positions[r["driver"]] = r["position"]
            teams[r["driver"]] = r["team"]
            try:
                driver_numbers[r["driver"]] = int(r.get("driver_number") or 0)
            except (TypeError, ValueError):
                driver_numbers.setdefault(r["driver"], 0)

    # Drivers nearest to ego's expected finishing band (roughly P1–P10)
    drivers_sorted = sorted(
        [d for d in final_positions if d != ego],
        key=lambda d: final_positions[d],
    )[:max_n]

    # Group race stints by driver
    stints_by_driver: dict[str, list[dict]] = {}
    for stint in race["stints"]:
        if stint["driver"] in drivers_sorted:
            stints_by_driver.setdefault(stint["driver"], []).append(stint)

    opponents: list[dict] = []
    for d in drivers_sorted:
        stints = sorted(stints_by_driver.get(d, []), key=lambda s: s["stint"])
        if not stints:
            continue
        first_stint = stints[0]
        opponents.append({
            "driver_number": int(driver_numbers.get(d) or 99),
            "team": teams.get(d, "Unknown"),
            "starting_position": final_positions[d],  # use final as proxy for grid
            "starting_compound": first_stint["compound"],
            "pace_offset_s": -0.30 if final_positions[d] <= 3 else 0.0,
            "aggression": 0.65,
            "planned_strategy": [
                {"compound": s["compound"], "planned_end_lap": s["end_lap"]}
                for s in stints
            ],
        })
    return opponents


def _build_weather_overrides(race: dict) -> dict:
    """Hand-shaped weather isn't available; use the per-race aggregate.

    Future enhancement: align real per-lap weather samples to the lap
    count. For now we use the mean track temp from the weather summary.
    """
    w = race.get("weather_summary", {}) or {}
    air = w.get("air_temp_c_mean", 22.0)
    track = w.get("track_temp_c_mean", 35.0)
    rain_any = w.get("rainfall_any", False)
    n_laps = race["metadata"]["total_laps"]
    return {
        "per_lap": [
            {
                "air_temp_c": air,
                "track_temp_c": track,
                "rain_intensity": 0.5 if rain_any else 0.0,
                "surface_state": "damp" if rain_any else "dry",
            }
            for _ in range(n_laps)
        ]
    }


_TRACK_KEY_FROM_COUNTRY: dict[str, str] = {
    "monaco": "monaco",
    "italy": "monza",
    "belgium": "spa",
    "spain": "catalunya",
    "japan": "suzuka",
    "great britain": "silverstone",
    "united kingdom": "silverstone",
    "bahrain": "bahrain",
    "azerbaijan": "baku",
    "miami": "miami",
    "emilia-romagna": "imola",
    "canada": "montreal",
    "austria": "spielberg",
    "hungary": "hungaroring",
    "netherlands": "zandvoort",
    "singapore": "singapore",
    "united states": "austin",
    "mexico": "mexico_city",
    "brazil": "interlagos",
    "qatar": "lusail",
    "abu dhabi": "abu_dhabi",
    "china": "shanghai",
    "australia": "melbourne",
    "saudi arabia": "jeddah",
}


def load_replay_scenario(
    year: int,
    gp_short: str,
    ego_driver: Optional[str] = None,
    *,
    target_finish: int = 5,
) -> tuple[dict, list[dict]]:
    """Build a replay scenario for `year`-`gp_short` and return
    (scenario, ego_actual_actions).

    Args:
        year: e.g. 2024
        gp_short: short key like "monaco" or "italian_grand_prix" — see
            ``list_replays()``.
        ego_driver: 3-letter driver code (e.g. "LEC"). If None, picks the
            driver who finished closest to ``target_finish``.
        target_finish: position to bias ego selection toward (default P5).

    Returns:
        scenario_dict: ready for ``env.reset(options={"scenario": ...})``
        ego_actual_actions: list of {lap, action, context} for comparison.
    """
    path = _short_name_to_path(year, gp_short)
    if not path:
        raise FileNotFoundError(
            f"No extracted race data for {year} {gp_short}. "
            f"Run scripts/fastf1_extract_all.py first. "
            f"Available: {list_replays()[:5]}..."
        )

    race = json.loads(path.read_text())
    candidates = _ego_candidates_by_position(race, target_finish)
    ego = ego_driver if ego_driver in candidates else (candidates[0] if candidates else None)
    if ego is None:
        raise ValueError(f"Could not determine ego driver for {year} {gp_short}")

    # Find ego's first stint for starting compound
    ego_stints = sorted(
        (s for s in race["stints"] if s["driver"] == ego),
        key=lambda s: s["stint"],
    )
    if not ego_stints:
        raise ValueError(f"Ego {ego} has no stints in {year} {gp_short}")
    starting_compound = ego_stints[0]["compound"]

    # Find ego's starting position (best position in first 3 laps)
    early_positions = [
        r["position"] for r in race["lap_records"]
        if r["driver"] == ego and r["position"] > 0 and r["lap_number"] <= 3
    ]
    starting_position = min(early_positions) if early_positions else 5

    meta = race["metadata"]
    country = meta["country"].lower()
    track_short = _TRACK_KEY_FROM_COUNTRY.get(country, country.replace(" ", "_"))
    # Pick a track_name the env's track loader understands. Default mapping
    # uses the country-name proxy; track loader has its own fallbacks.
    track_name = {
        "monaco": "Monaco",
        "monza": "Monza",
        "spa": "Spa",
        "catalunya": "Catalunya",
        "silverstone": "Silverstone",
        "suzuka": "Suzuka",
    }.get(track_short, "Catalunya")  # Catalunya is a balanced fallback

    scenario = {
        "task_name": f"replay_{year}_{gp_short}_{ego}",
        "scenario_family": "replay",
        "description": (
            f"Replay: {meta['event_name']} {year} as {ego}. "
            f"Real conditions, real opponent strategies, your call."
        ),
        "track_name": track_name,
        "total_laps": int(meta["total_laps"]),
        "max_steps": int(meta["total_laps"]) + 8,
        "max_score": 1.0,
        "seed": int(year * 100 + (race.get("metadata", {}).get("round_number") or 0)),
        "starting_position": int(starting_position),
        "starting_compound": str(starting_compound),
        "starting_fuel_kg": 110.0,
        "starting_drive_mode": "race",
        "opponents": _build_opponents(race, ego),
        "weather_archetype": "rain_window" if meta["wet_race"] else "dry_warm",
        "weather_seed_overrides": _build_weather_overrides(race),
        "sc_archetype": "midrace_likely" if any(
            "safety_car" in (e.get("status_decoded") or [])
            for e in race.get("track_status_events", [])
        ) else "none",
        "issues": {
            "race_result": [
                {"goal": f"finish_at_least_p{starting_position}",
                 "points": 0.30}
            ],
            "tyre_management": [
                {"constraint": "use_two_dry_compounds_and_health_gt_30",
                 "points": 0.20}
            ],
            "fuel_management": [
                {"constraint": "finish_with_0_5kg_margin", "points": 0.10}
            ],
            "strategic_decisions": [
                {"decision": "complete_the_race",
                 "valid_window": [1, int(meta["total_laps"])],
                 "points": 0.30}
            ],
            "pending_comms": [
                {"trigger": "pit_call", "audience": "driver",
                 "required": True, "points": 0.10}
            ],
        },
        "success_criteria": {
            "target_position": int(starting_position),
            "bonus_position": max(1, int(starting_position) - 1),
            "target_n_pits": max(1, len(ego_stints) - 1),
            "fuel_margin_kg": 0.5,
            "tyre_health_min": 0.30,
        },
        "hidden_state": {
            "fuel_burn_actual": 1.95,
            "undercut_threshold_laps": 2,
        },
        "dynamic_events": [
            {"lap": p["lap"], "type": "opponent_pit",
             "desc": f"#{p['driver']} pits for {p['compound_out']}"}
            for p in race["pit_decisions"]
            if p["driver"] != ego
        ][:10],  # cap to 10 for readability
        "radio_inbox": [
            {
                "id": "REPLAY-001",
                "from": "race_engineer",
                "message": (
                    f"This is the {meta['event_name']} {year}, replayed. "
                    f"You're {ego}. Make your strategic calls; we'll see how "
                    f"you stack up against what {ego} actually did on the day."
                ),
            }
        ],
        "memory_hint_tags": ["replay", track_short, str(year)],
        # Sub-step 2.4 opt-in: use real F1 tire wear factors when available.
        "use_grounded_calibration": True,
        "grounded_track_key": track_short,
        # Replay-specific metadata (not consumed by env, useful for tooling)
        "_replay_meta": {
            "year": year,
            "gp_short": gp_short,
            "event_name": meta["event_name"],
            "ego_driver": ego,
            "ego_n_pits": max(0, len(ego_stints) - 1),
            "ego_compounds": [s["compound"] for s in ego_stints],
        },
    }

    actual_actions = _ego_actual_actions(race, ego)
    return scenario, actual_actions
