"""Long-race scenario families (sub-step 1.2).

Three hand-authored 55-lap full-grand-prix scenarios for Phase 1. Each is
designed for **multi-stop strategy** (2-stop optimal) with race-length-realistic
fuel and opponent plans that 1.1's stretching cannot produce mechanically.

Why a separate file:
  scenarios.py is already ~800 lines with the six 12–15 lap families. Long-race
  variants add ~300 more lines — keeping them isolated makes both files
  navigable. Both are imported into `SCENARIOS` from `scenarios.py`.

Design notes per family:
  - 4 opponents (vs 5 in short scenarios) — keeps weather + opponent state
    digestible at 55-lap horizons.
  - Weather per_lap array hand-shaped at key inflection points; flat phases
    use a single value repeated.
  - Two-stop pit windows (early ~18–22, late ~38–42).
  - starting_fuel_kg = ~110 (real F1 maximum), burn rate ~1.95 kg/lap.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Weather builders — compact construction of per-lap weather arrays
# ---------------------------------------------------------------------------

def _dry_warm(n: int) -> list[dict]:
    return [
        {"air_temp_c": 28, "track_temp_c": 42, "rain_intensity": 0.0, "surface_state": "dry"}
        for _ in range(n)
    ]


def _dry_cool(n: int) -> list[dict]:
    return [
        {"air_temp_c": 18, "track_temp_c": 26, "rain_intensity": 0.0, "surface_state": "dry"}
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# MONACO_FULL_GP — 55 laps, street circuit, track position is everything
# ---------------------------------------------------------------------------
# Strategy: undercut/overcut critical because passing is near-impossible.
# Tyre cliffs hit late. SC probable mid-race (35% lap 28–42). Pit under SC if
# fired. Optimal: 2-stop with mediums then softs, or 1-stop if SC saves a
# stop.

MONACO_FULL_GP: dict = {
    "task_name": "monaco_full_gp",
    "scenario_family": "monaco_full_gp",
    "description": "Monaco GP: track position over raw pace. Cover the SC window.",
    "track_name": "Monaco",
    "total_laps": 55,
    "max_steps": 59,
    "max_score": 1.0,
    "seed": 78,
    "starting_position": 4,
    "starting_compound": "medium",
    "starting_fuel_kg": 110.0,
    "starting_drive_mode": "race",
    "opponents": [
        {
            "driver_number": 1,
            "team": "Red Bull",
            "starting_position": 1,
            "starting_compound": "medium",
            "pace_offset_s": -0.30,
            "aggression": 0.65,
            "planned_strategy": [
                {"compound": "hard", "planned_end_lap": 22},
                {"compound": "soft", "planned_end_lap": 42},
            ],
        },
        {
            "driver_number": 16,
            "team": "Ferrari",
            "starting_position": 2,
            "starting_compound": "medium",
            "pace_offset_s": -0.10,
            "aggression": 0.70,
            "planned_strategy": [
                {"compound": "hard", "planned_end_lap": 20},
                {"compound": "medium", "planned_end_lap": 40},
            ],
        },
        {
            "driver_number": 44,
            "team": "Mercedes",
            "starting_position": 3,
            "starting_compound": "medium",
            "pace_offset_s": 0.05,
            "aggression": 0.55,
            "planned_strategy": [
                {"compound": "hard", "planned_end_lap": 25},
                {"compound": "soft", "planned_end_lap": 45},
            ],
        },
        {
            "driver_number": 4,
            "team": "McLaren",
            "starting_position": 5,
            "starting_compound": "hard",
            "pace_offset_s": 0.12,
            "aggression": 0.50,
            "planned_strategy": [
                {"compound": "soft", "planned_end_lap": 38},
            ],
        },
    ],
    "weather_archetype": "dry_hot",
    "weather_seed_overrides": {"per_lap": _dry_warm(55)},
    "sc_archetype": "midrace_likely",
    "issues": {
        "race_result": [{"goal": "finish_at_least_p4", "points": 0.25}],
        "tyre_management": [
            {"constraint": "use_two_dry_compounds_and_health_gt_30", "points": 0.15}
        ],
        "fuel_management": [{"constraint": "finish_with_0_5kg_margin", "points": 0.05}],
        "strategic_decisions": [
            {
                "decision": "pit_during_sc_or_window_18_24",
                "valid_window": [18, 24],
                "preconditions": ["ASSESS_UNDERCUT_WINDOW"],
                "points": 0.20,
            },
            {
                "decision": "second_stop_in_window_38_44",
                "valid_window": [38, 44],
                "points": 0.15,
            },
            {"decision": "exactly_two_stops", "points": 0.10},
        ],
        "pending_comms": [
            {"trigger": "pit_call", "audience": "driver", "required": True, "points": 0.10}
        ],
    },
    "success_criteria": {
        "target_position": 4,
        "bonus_position": 3,
        "optimal_pit_window": [18, 24],
        "second_pit_window": [38, 44],
        "target_n_pits": 2,
        "required_inspections": ["ASSESS_UNDERCUT_WINDOW", "CHECK_OPPONENT_STRATEGY"],
        "required_comms": ["pit"],
        "fuel_margin_kg": 0.5,
        "tyre_health_min": 0.30,
    },
    "hidden_state": {
        "true_tyre_curve": {
            "hard":   [round(1.0 - 0.012 * i, 3) for i in range(30)],
            "medium": [round(1.0 - 0.020 * i, 3) for i in range(25)],
            "soft":   [round(1.0 - 0.035 * i, 3) for i in range(15)],
        },
        "opponent_strategies": {
            "1":  [{"compound": "hard", "planned_end_lap": 22},
                   {"compound": "soft", "planned_end_lap": 42}],
            "16": [{"compound": "hard", "planned_end_lap": 20},
                   {"compound": "medium", "planned_end_lap": 40}],
        },
        "fuel_burn_actual": 1.95,
        "undercut_threshold_laps": 2,
    },
    "dynamic_events": [
        {"lap": 12, "type": "info", "desc": "Tyre window opens for the front-runners."},
        {"lap": 22, "type": "opponent_pit", "desc": "#1 boxes for hards."},
        {"lap": 35, "type": "info", "desc": "SC window approaching."},
        {"lap": 42, "type": "opponent_pit", "desc": "#1 boxes for softs (final stint)."},
    ],
    "radio_inbox": [
        {"id": "R-M01", "from": "race_engineer",
         "message": "P4. Long race. Cover #16 — undercut threat real. SC chance lap 28–42."},
        {"id": "R-M02", "from": "team_principal",
         "message": "Track position is the whole race here. Don't overdrive the tyres."},
    ],
    "memory_hint_tags": ["monaco", "track_position", "two_stop", "long_race"],
}


# ---------------------------------------------------------------------------
# SILVERSTONE_FULL_GP — 55 laps, fast and flowing, high tyre wear
# ---------------------------------------------------------------------------
# Strategy: 2-stop is optimal. Tyre management is the key. VSC events common
# (60% lap 25–35) — using a VSC pit saves ~10s.

SILVERSTONE_FULL_GP: dict = {
    "task_name": "silverstone_full_gp",
    "scenario_family": "silverstone_full_gp",
    "description": "Silverstone GP: high tyre wear, two-stop optimal, VSC may fire.",
    "track_name": "Silverstone",
    "total_laps": 55,
    "max_steps": 59,
    "max_score": 1.0,
    "seed": 70,
    "starting_position": 5,
    "starting_compound": "medium",
    "starting_fuel_kg": 110.0,
    "starting_drive_mode": "race",
    "opponents": [
        {
            "driver_number": 4,
            "team": "McLaren",
            "starting_position": 1,
            "starting_compound": "medium",
            "pace_offset_s": -0.40,
            "aggression": 0.70,
            "planned_strategy": [
                {"compound": "medium", "planned_end_lap": 20},
                {"compound": "soft", "planned_end_lap": 40},
            ],
        },
        {
            "driver_number": 1,
            "team": "Red Bull",
            "starting_position": 2,
            "starting_compound": "medium",
            "pace_offset_s": -0.20,
            "aggression": 0.60,
            "planned_strategy": [
                {"compound": "hard", "planned_end_lap": 25},
                {"compound": "soft", "planned_end_lap": 45},
            ],
        },
        {
            "driver_number": 63,
            "team": "Mercedes",
            "starting_position": 4,
            "starting_compound": "medium",
            "pace_offset_s": 0.00,
            "aggression": 0.55,
            "planned_strategy": [
                {"compound": "medium", "planned_end_lap": 22},
                {"compound": "medium", "planned_end_lap": 42},
            ],
        },
        {
            "driver_number": 81,
            "team": "McLaren",
            "starting_position": 6,
            "starting_compound": "hard",
            "pace_offset_s": 0.15,
            "aggression": 0.50,
            "planned_strategy": [
                {"compound": "soft", "planned_end_lap": 35},
            ],
        },
    ],
    "weather_archetype": "dry_warm",
    "weather_seed_overrides": {"per_lap": _dry_warm(55)},
    "sc_archetype": "vsc_likely",
    "issues": {
        "race_result": [{"goal": "finish_at_least_p5", "points": 0.20}],
        "tyre_management": [
            {"constraint": "use_two_dry_compounds_and_health_gt_30", "points": 0.20}
        ],
        "fuel_management": [{"constraint": "finish_with_0_5kg_margin", "points": 0.05}],
        "strategic_decisions": [
            {
                "decision": "first_stop_in_window_18_24",
                "valid_window": [18, 24],
                "points": 0.20,
            },
            {
                "decision": "second_stop_in_window_38_44",
                "valid_window": [38, 44],
                "points": 0.15,
            },
            {"decision": "exactly_two_stops", "points": 0.10},
        ],
        "pending_comms": [
            {"trigger": "pit_call", "audience": "driver", "required": True, "points": 0.10}
        ],
    },
    "success_criteria": {
        "target_position": 5,
        "bonus_position": 4,
        "optimal_pit_window": [18, 24],
        "second_pit_window": [38, 44],
        "target_n_pits": 2,
        "required_inspections": ["INSPECT_TYRE_DEGRADATION"],
        "required_comms": ["pit"],
        "fuel_margin_kg": 0.5,
        "tyre_health_min": 0.30,
    },
    "hidden_state": {
        "true_tyre_curve": {
            "hard":   [round(1.0 - 0.014 * i, 3) for i in range(30)],
            "medium": [round(1.0 - 0.022 * i, 3) for i in range(22)],
            "soft":   [round(1.0 - 0.040 * i, 3) for i in range(14)],
        },
        "opponent_strategies": {
            "4":  [{"compound": "medium", "planned_end_lap": 20},
                   {"compound": "soft", "planned_end_lap": 40}],
            "63": [{"compound": "medium", "planned_end_lap": 22},
                   {"compound": "medium", "planned_end_lap": 42}],
        },
        "fuel_burn_actual": 1.95,
        "undercut_threshold_laps": 2,
    },
    "dynamic_events": [
        {"lap": 14, "type": "info", "desc": "Front tyres approaching cliff for mediums."},
        {"lap": 20, "type": "opponent_pit", "desc": "#4 boxes for softs."},
        {"lap": 30, "type": "info", "desc": "VSC window — debris reported."},
        {"lap": 40, "type": "opponent_pit", "desc": "#4 boxes for the final stint."},
    ],
    "radio_inbox": [
        {"id": "R-S01", "from": "race_engineer",
         "message": "P5. Two-stop is the call. Watch for VSC around lap 30."},
        {"id": "R-S02", "from": "team_principal",
         "message": "Tyres are the limit here. Don't push beyond what they'll give."},
    ],
    "memory_hint_tags": ["silverstone", "two_stop", "vsc", "tyre_wear", "long_race"],
}


# ---------------------------------------------------------------------------
# SPA_FULL_WET — 55 laps, mixed conditions, rain mid-race
# ---------------------------------------------------------------------------
# Strategy: rain hits laps 20–35. Must switch to inters at the right lap.
# Compound discipline: medium → inter → medium-or-soft. Forecast helps;
# forecast is uncertain.

def _spa_weather() -> list[dict]:
    """Hand-shaped 55-lap weather: dry start, rain peak around lap 27, dry end."""
    out: list[dict] = []
    for lap in range(55):
        if lap < 18:
            rain = 0.0
            surface = "dry"
        elif lap < 22:
            rain = 0.15 + 0.05 * (lap - 18)  # 0.15 → 0.30
            surface = "damp"
        elif lap < 32:
            # peak rain phase
            rain = 0.50 if lap == 27 else 0.40
            surface = "damp"
        elif lap < 38:
            rain = 0.50 - 0.07 * (lap - 32)  # tapering
            surface = "damp"
        else:
            rain = 0.0
            surface = "dry"
        out.append({"air_temp_c": 18, "track_temp_c": 24,
                    "rain_intensity": round(rain, 2), "surface_state": surface})
    return out


SPA_FULL_WET: dict = {
    "task_name": "spa_full_wet",
    "scenario_family": "spa_full_wet",
    "description": "Spa GP with mid-race rain: time the inter switch and the slick comeback.",
    "track_name": "Spa",
    "total_laps": 55,
    "max_steps": 59,
    "max_score": 1.0,
    "seed": 99,
    "starting_position": 5,
    "starting_compound": "medium",
    "starting_fuel_kg": 110.0,
    "starting_drive_mode": "race",
    "opponents": [
        {
            "driver_number": 1,
            "team": "Red Bull",
            "starting_position": 1,
            "starting_compound": "medium",
            "pace_offset_s": -0.35,
            "aggression": 0.70,
            "planned_strategy": [
                {"compound": "inter", "planned_end_lap": 22},
                {"compound": "medium", "planned_end_lap": 38},
                {"compound": "soft", "planned_end_lap": 55},
            ],
        },
        {
            "driver_number": 63,
            "team": "Mercedes",
            "starting_position": 3,
            "starting_compound": "medium",
            "pace_offset_s": 0.05,
            "aggression": 0.55,
            "planned_strategy": [
                {"compound": "inter", "planned_end_lap": 24},
                {"compound": "medium", "planned_end_lap": 55},
            ],
        },
        {
            "driver_number": 16,
            "team": "Ferrari",
            "starting_position": 2,
            "starting_compound": "medium",
            "pace_offset_s": -0.05,
            "aggression": 0.65,
            "planned_strategy": [
                {"compound": "inter", "planned_end_lap": 23},
                {"compound": "soft", "planned_end_lap": 55},
            ],
        },
        {
            "driver_number": 4,
            "team": "McLaren",
            "starting_position": 4,
            "starting_compound": "medium",
            "pace_offset_s": 0.00,
            "aggression": 0.55,
            "planned_strategy": [
                {"compound": "inter", "planned_end_lap": 22},
                {"compound": "medium", "planned_end_lap": 55},
            ],
        },
    ],
    "weather_archetype": "rain_window",
    "weather_seed_overrides": {
        "forecast_uncertainty": 0.30,
        "per_lap": _spa_weather(),
    },
    "sc_archetype": "none",
    "issues": {
        "race_result": [{"goal": "finish_at_least_p5", "points": 0.20}],
        "tyre_management": [
            {"constraint": "use_inters_during_peak_rain", "points": 0.20},
            {"constraint": "switch_back_to_slick_when_dry", "points": 0.15},
        ],
        "fuel_management": [{"constraint": "finish_with_0_5kg_margin", "points": 0.05}],
        "strategic_decisions": [
            {
                "decision": "request_forecast_before_lap_18",
                "preconditions": ["REQUEST_FORECAST"],
                "valid_window": [1, 18],
                "points": 0.10,
            },
            {
                "decision": "pit_for_inters_in_window_19_23",
                "valid_window": [19, 23],
                "points": 0.15,
            },
            {
                "decision": "switch_back_to_slick_in_window_36_40",
                "valid_window": [36, 40],
                "points": 0.10,
            },
        ],
        "pending_comms": [
            {"trigger": "pit_call", "audience": "driver", "required": True, "points": 0.05}
        ],
    },
    "success_criteria": {
        "target_position": 5,
        "bonus_position": 3,
        "optimal_pit_window": [19, 23],
        "second_pit_window": [36, 40],
        "target_n_pits": 2,
        "required_inspections": ["REQUEST_FORECAST"],
        "required_comms": ["pit"],
        "rain_peak_lap": 27,
        "fuel_margin_kg": 0.5,
        "tyre_health_min": 0.30,
    },
    "hidden_state": {
        "true_tyre_curve": {
            "medium": [round(1.0 - 0.022 * i, 3) for i in range(22)],
            "soft":   [round(1.0 - 0.040 * i, 3) for i in range(14)],
            "inter":  [round(1.0 - 0.025 * i, 3) for i in range(20)],
        },
        "opponent_strategies": {
            "1":  [{"compound": "inter", "planned_end_lap": 22},
                   {"compound": "medium", "planned_end_lap": 38},
                   {"compound": "soft", "planned_end_lap": 55}],
            "16": [{"compound": "inter", "planned_end_lap": 23},
                   {"compound": "soft", "planned_end_lap": 55}],
        },
        "fuel_burn_actual": 1.95,
        "undercut_threshold_laps": 2,
        "true_rain_peak_lap": 27,
    },
    "dynamic_events": [
        {"lap": 15, "type": "info", "desc": "Cloud line approaching from the west."},
        {"lap": 19, "type": "info", "desc": "First drops on lap 19."},
        {"lap": 22, "type": "opponent_pit", "desc": "#1 boxes for inters."},
        {"lap": 32, "type": "info", "desc": "Rain easing. Drying line forming."},
        {"lap": 38, "type": "opponent_pit", "desc": "#1 boxes for slicks."},
    ],
    "radio_inbox": [
        {"id": "R-W01", "from": "race_engineer",
         "message": "Watching the radar — rain coming in the next 5 laps. REQUEST_FORECAST."},
        {"id": "R-W02", "from": "team_principal",
         "message": "Get the inter call right and we podium. Get it wrong and we DNF."},
    ],
    "memory_hint_tags": ["spa", "rain", "inter_window", "long_race", "weather"],
}


# ---------------------------------------------------------------------------
# Public registry — imported into scenarios.SCENARIOS
# ---------------------------------------------------------------------------

LONG_SCENARIOS: dict[str, dict] = {
    "monaco_full_gp": MONACO_FULL_GP,
    "silverstone_full_gp": SILVERSTONE_FULL_GP,
    "spa_full_wet": SPA_FULL_WET,
}
