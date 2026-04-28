"""Pytest: six-dimension scorer is pure and deterministic."""

import pytest

from server import scoring


def _kwargs():
    return {
        "scenario_family": "dry_strategy_sprint",
        "success_criteria": {
            "target_position": 4,
            "optimal_pit_window": [4, 7],
            "target_n_pits": 1,
            "required_compound": "soft",
            "required_inspections": ["ASSESS_UNDERCUT_WINDOW"],
            "required_comms": ["pit"],
            "fuel_margin_kg": 0.5,
        },
        "starting_position": 4,
        "finishing_position": 3,
        "pit_decisions": [{"lap": 6, "compound": "soft", "under_sc": False}],
        "inspection_calls": {"ASSESS_UNDERCUT_WINDOW": [3], "INSPECT_FUEL_MARGIN": [9]},
        "final_tyre_health": 0.70,
        "compound_rule_satisfied": True,
        "stints": [{"compound": "medium"}, {"compound": "soft"}],
        "track_character": "power",
        "final_fuel_kg": 1.0,
        "dnf_due_to_fuel": False,
        "radio_calls_made": [{"message": "pit this lap"}],
        "pit_wall_calls": [],
        "n_pit_stops": 1,
        "invalid_actions": 0,
        "harmful_actions": 0,
        "total_laps": 10,
        "steps_used": 12,
        "action_verbs": ["ASSESS_UNDERCUT_WINDOW", "PIT_NOW"],
    }


def test_weighted_final_clamped():
    perfect = scoring.compute_multi_objective_scores(**_kwargs())
    assert perfect["weighted_final"] <= 0.99
    bad = _kwargs()
    bad.update(
        {
            "finishing_position": 10,
            "pit_decisions": [],
            "compound_rule_satisfied": False,
            "final_fuel_kg": -1.0,
            "dnf_due_to_fuel": True,
            "radio_calls_made": [],
            "n_pit_stops": 0,
            "invalid_actions": 8,
            "harmful_actions": 3,
        }
    )
    poor = scoring.compute_multi_objective_scores(**bad)
    assert poor["weighted_final"] >= 0.01


def test_zero_issue_case_scores_near_neutral_baseline():
    neutral = {
        "scenario_family": "neutral",
        "success_criteria": {
            "target_position": 3,
            "optimal_pit_window": [5, 5],
            "target_n_pits": 1,
            "fuel_margin_kg": 0.5,
            "required_comms": [],
        },
        "starting_position": 4,
        "finishing_position": 3,
        "rival_finishing_position": 2,
        "pit_decisions": [{"lap": 5, "compound": "medium", "under_sc": False}],
        "inspection_calls": {},
        "final_tyre_health": 0.0889,
        "compound_rule_satisfied": True,
        "stints": [{"compound": "medium"}],
        "track_character": "balanced",
        "final_fuel_kg": 0.3824,
        "dnf_due_to_fuel": False,
        "radio_calls_made": [],
        "pit_wall_calls": [],
        "n_pit_stops": 1,
        "invalid_actions": 6,
        "harmful_actions": 0,
        "total_laps": 10,
        "steps_used": 10,
        "action_verbs": ["PIT_NOW"],
    }
    scores = scoring.compute_multi_objective_scores(**neutral)
    assert scores["weighted_final"] == pytest.approx(0.65, abs=0.02)


def test_perfect_case_clamps_to_upper_bound():
    perfect = {
        "scenario_family": "neutral",
        "success_criteria": {
            "target_position": 3,
            "optimal_pit_window": [5, 5],
            "target_n_pits": 1,
            "fuel_margin_kg": 0.5,
            "required_comms": ["box"],
        },
        "starting_position": 5,
        "finishing_position": 2,
        "pit_decisions": [{"lap": 5, "compound": "soft", "under_sc": False}],
        "inspection_calls": {"ASSESS_UNDERCUT_WINDOW": [4], "INSPECT_FUEL_MARGIN": [9]},
        "final_tyre_health": 0.95,
        "compound_rule_satisfied": True,
        "stints": [{"compound": "medium"}, {"compound": "soft"}],
        "track_character": "power",
        "final_fuel_kg": 1.2,
        "dnf_due_to_fuel": False,
        "radio_calls_made": [{"message": "box this lap"}],
        "pit_wall_calls": [],
        "n_pit_stops": 1,
        "invalid_actions": 0,
        "harmful_actions": 0,
        "total_laps": 10,
        "steps_used": 10,
        "action_verbs": ["ASSESS_UNDERCUT_WINDOW", "PIT_NOW"],
    }
    scores = scoring.compute_multi_objective_scores(**perfect)
    assert scores["weighted_final"] == pytest.approx(0.99, abs=1e-6)


def test_dnf_case_scores_low():
    bad = _kwargs()
    bad.update(
        {
            "success_criteria": {
                "target_position": 1,
                "optimal_pit_window": [4, 4],
                "target_n_pits": 1,
                "required_compound": "soft",
                "required_comms": ["pit"],
                "fuel_margin_kg": 1.0,
            },
            "finishing_position": 0,
            "rival_finishing_position": None,
            "pit_decisions": [],
            "inspection_calls": {},
            "final_tyre_health": 0.0,
            "compound_rule_satisfied": False,
            "stints": [{"compound": "medium"}],
            "final_fuel_kg": -0.5,
            "dnf_due_to_fuel": True,
            "radio_calls_made": [],
            "pit_wall_calls": [],
            "n_pit_stops": 0,
            "invalid_actions": 4,
            "harmful_actions": 4,
            "action_verbs": [],
        }
    )
    scores = scoring.compute_multi_objective_scores(**bad)
    assert scores["weighted_final"] <= 0.30


def test_each_dimension_can_be_tested_in_isolation():
    assert scoring.compute_race_result(5, 2, 3) == 1.0
    assert scoring.compute_race_result(5, 0, 3) == 0.0

    assert (
        scoring.compute_strategic_decisions(
            [{"lap": 4}], (3, 5), {"ASSESS_UNDERCUT_WINDOW": [3]}, False, True
        )
        == 1.0
    )
    assert scoring.compute_strategic_decisions([], (3, 5), {}, False, False) == 0.0

    assert scoring.compute_tyre_management(1.0, True, [], "balanced") == 1.0
    assert scoring.compute_tyre_management(0.0, False, [], "balanced") == 0.0

    assert scoring.compute_fuel_management(1.0, 0.5, False, True) == 1.0
    assert scoring.compute_fuel_management(-0.1, 0.5, True, False) == 0.0

    assert scoring.compute_comms_quality(["pit"], [{"message": "pit now"}], []) == 1.0
    assert scoring.compute_comms_quality(["pit"], [], []) == 0.0

    assert scoring.compute_operational_efficiency(1, 1, 0, 0, 10, 10) == 1.0
    assert scoring.compute_operational_efficiency(0, 1, 5, 3, 10, 25) == 0.0


def test_pure_function_no_side_effects():
    a = scoring.compute_multi_objective_scores(**_kwargs())
    b = scoring.compute_multi_objective_scores(**_kwargs())
    assert a == b
