"""Regression tests for the hardened scoring layer.

These tests pin down the audit findings so they don't regress:
    * Random policies must NOT score above 0.50 mean across 10 seeds on any family
    * Trained scripted policy must beat random by ≥0.20 mean on every family
    * Expert sequences must clear 0.85 on every family (anchor point)
    * Comms keyword matching is whole-word (no `pit` ⊂ `spit` exploit)
    * Over-pitting caps the strategic-decision dimension
    * Race-result halo gated on comms ≥ 0.5 (random can't bypass it)
    * Pre-pit inspection ordering is enforced (inspection AFTER pit ≠ credit)
"""

from __future__ import annotations

import statistics

import pytest

from server import scoring
from server.scenarios import SCENARIOS


# ──────────────────────────────────────────────────────────────────────────
# 1. Random / panic policies stay capped
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "task",
    ["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"],
)
def test_random_policy_mean_below_threshold(task):
    """Random sampling over 10 seeds must average < 0.50 weighted_final."""
    from evaluate import run_one
    scores = [run_one(task, "random", seed) for seed in range(10)]
    mean = statistics.mean(scores)
    assert mean < 0.50, (
        f"{task}: random mean={mean:.3f} above 0.50 — scoring is exploitable. "
        f"Individual scores: {[round(s, 3) for s in scores]}"
    )


@pytest.mark.parametrize(
    "task",
    ["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"],
)
def test_random_policy_max_below_threshold(task):
    """No single random seed should score above 0.70 — that's near-expert by luck."""
    from evaluate import run_one
    scores = [run_one(task, "random", seed) for seed in range(10)]
    assert max(scores) < 0.70, (
        f"{task}: random max={max(scores):.3f} above 0.70 — accidental near-expert play. "
        f"Individual scores: {[round(s, 3) for s in scores]}"
    )


# ──────────────────────────────────────────────────────────────────────────
# 2. Trained > random gap is real
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "task",
    ["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"],
)
def test_trained_beats_random_by_margin(task):
    """Scripted-trained mean must exceed random mean by ≥ 0.20."""
    from evaluate import run_one
    random_mean = statistics.mean(run_one(task, "random", seed) for seed in range(5))
    trained_mean = statistics.mean(run_one(task, "trained", seed) for seed in range(5))
    delta = trained_mean - random_mean
    assert delta >= 0.20, (
        f"{task}: trained={trained_mean:.3f} vs random={random_mean:.3f} "
        f"(delta {delta:.3f}) — trained policy isn't sufficiently above random."
    )


# ──────────────────────────────────────────────────────────────────────────
# 3. Expert sequences are anchored
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "task,threshold",
    [
        ("dry_strategy_sprint", 0.78),
        ("weather_roulette", 0.85),
        ("late_safety_car", 0.85),
        ("championship_decider", 0.85),
    ],
)
def test_expert_sequence_clears_threshold(task, threshold):
    """Expert sequences must score above threshold averaged across 3 seeds.

    Dry sprint has a slightly lower threshold because the opponent pace seed
    swings the race-result dimension; the expert sequence is solid but
    not bulletproof against a bad opponent draw.
    """
    from evaluate import run_one
    scores = [run_one(task, "expert", seed) for seed in range(3)]
    mean = statistics.mean(scores)
    assert mean >= threshold, (
        f"{task}: expert mean={mean:.3f} below {threshold} — expert sequence is broken. "
        f"Individual scores: {[round(s, 3) for s in scores]}"
    )


# ──────────────────────────────────────────────────────────────────────────
# 4. Comms keyword matching is whole-word
# ──────────────────────────────────────────────────────────────────────────

def test_comms_pit_not_substring_of_spit():
    """`pit` must not match `spit`/`split` — pure substring matching is wrong."""
    assert scoring.compute_comms_quality(["pit"], [{"message": "spit polish"}], []) == 0.0
    assert scoring.compute_comms_quality(["pit"], [{"message": "split decision"}], []) == 0.0


def test_comms_pit_matches_inflections():
    """`pit` should match `pit`, `pits`, `pitting`, `pitlane`."""
    for msg in ["box for pit", "pits this lap", "pitting now", "into pitlane"]:
        score = scoring.compute_comms_quality(["pit"], [{"message": msg}], [])
        assert score == 1.0, f"`{msg}` should match keyword `pit` but scored {score}"


def test_comms_unrelated_message_scores_zero():
    """A `Status update.` radio call must NOT score on a `pit` requirement."""
    score = scoring.compute_comms_quality(["pit"], [{"message": "Status update."}], [])
    assert score == 0.0


def test_comms_empty_required_returns_full():
    """No required triggers → full credit (nothing was needed)."""
    assert scoring.compute_comms_quality([], [], []) == 1.0


# ──────────────────────────────────────────────────────────────────────────
# 5. Over-pitting cap on strategic dimension
# ──────────────────────────────────────────────────────────────────────────

def test_over_pitting_caps_strategic_dim():
    """Pitting twice on a 1-stop scenario must cap strategic at 0.60 even if perfect."""
    base_kwargs = _good_dry_kwargs()
    base_kwargs["n_pit_stops"] = 1
    perfect = scoring.compute_multi_objective_scores(**base_kwargs)
    assert perfect["strategic_decisions"] >= 0.95

    over_pitted = dict(base_kwargs)
    over_pitted["n_pit_stops"] = 2
    over_pitted["pit_decisions"] = base_kwargs["pit_decisions"] + [
        {"lap": 8, "compound": "soft", "under_sc": False}
    ]
    result = scoring.compute_multi_objective_scores(**over_pitted)
    assert result["strategic_decisions"] <= 0.60, (
        f"strategic_decisions={result['strategic_decisions']:.3f} should cap ≤0.60 "
        "when n_pit_stops=2 vs target=1"
    )


def test_double_over_pitting_caps_lower():
    """Three pits on a 1-stop scenario must cap strategic at 0.40."""
    base = _good_dry_kwargs()
    base["n_pit_stops"] = 3
    base["pit_decisions"] = [
        {"lap": 5, "compound": "soft", "under_sc": False},
        {"lap": 7, "compound": "hard", "under_sc": False},
        {"lap": 9, "compound": "soft", "under_sc": False},
    ]
    result = scoring.compute_multi_objective_scores(**base)
    assert result["strategic_decisions"] <= 0.40


# ──────────────────────────────────────────────────────────────────────────
# 6. Race-result halo is comms-gated
# ──────────────────────────────────────────────────────────────────────────

def test_race_halo_requires_comms_above_floor():
    """Strategic=1.0 + comms=0.0 must NOT bump race_result via halo."""
    kwargs = _good_dry_kwargs()
    kwargs["radio_calls_made"] = []  # zero comms
    kwargs["finishing_position"] = 5  # would normally fall below target=4
    kwargs["starting_position"] = 4
    result = scoring.compute_multi_objective_scores(**kwargs)
    # finishing P5 from P4 = lost a position → race_result should NOT be 0.90
    assert result["race_result"] < 0.90, (
        f"race_result={result['race_result']:.3f} got halo bump despite comms=0"
    )


def test_race_halo_applies_when_strategy_and_comms_high():
    """Strategic≥0.95 + comms≥0.5 → race_result halo at 0.90 minimum."""
    kwargs = _good_dry_kwargs()
    kwargs["radio_calls_made"] = [{"message": "Pit this lap for softs."}]
    kwargs["finishing_position"] = 5
    kwargs["starting_position"] = 4
    result = scoring.compute_multi_objective_scores(**kwargs)
    assert result["race_result"] >= 0.90


# ──────────────────────────────────────────────────────────────────────────
# 7. Pre-pit inspection ordering enforced
# ──────────────────────────────────────────────────────────────────────────

def test_inspection_after_pit_does_not_count():
    """ASSESS_UNDERCUT_WINDOW called AFTER first pit → no precondition credit."""
    kwargs = _good_dry_kwargs()
    kwargs["pit_decisions"] = [{"lap": 5, "compound": "soft", "under_sc": False}]
    kwargs["inspection_calls"] = {"ASSESS_UNDERCUT_WINDOW": [8]}  # AFTER lap 5
    result = scoring.compute_multi_objective_scores(**kwargs)
    assert result["strategic_decisions"] <= 0.65, (
        f"strategic_decisions={result['strategic_decisions']:.3f} should be ≤0.65 "
        "when ASSESS_UNDERCUT_WINDOW happened after the pit"
    )


def test_inspection_before_pit_grants_full_credit():
    """ASSESS_UNDERCUT_WINDOW lap 3 → PIT_NOW soft lap 5 = full credit."""
    kwargs = _good_dry_kwargs()
    kwargs["pit_decisions"] = [{"lap": 5, "compound": "soft", "under_sc": False}]
    kwargs["inspection_calls"] = {"ASSESS_UNDERCUT_WINDOW": [3]}
    result = scoring.compute_multi_objective_scores(**kwargs)
    assert result["strategic_decisions"] >= 0.95


# ──────────────────────────────────────────────────────────────────────────
# 8. Family-specific precondition strictness
# ──────────────────────────────────────────────────────────────────────────

def test_weather_family_requires_forecast_not_just_any_inspection():
    """Weather scenario: INSPECT_FUEL_MARGIN before pit ≠ REQUEST_FORECAST."""
    kwargs = _good_weather_kwargs()
    # Replace the forecast with a fuel inspection — should NOT pass
    kwargs["inspection_calls"] = {"INSPECT_FUEL_MARGIN": [3]}
    result = scoring.compute_multi_objective_scores(**kwargs)
    assert result["strategic_decisions"] <= 0.65, (
        "Weather scenario without REQUEST_FORECAST before pit should not score full strategic"
    )


def test_late_sc_family_requires_hold_gap_verb():
    """Late SC scenario: missing HOLD_GAP verb caps strategic at 0.55."""
    kwargs = _good_late_sc_kwargs()
    kwargs["action_verbs"] = ["INSPECT_TYRE_DEGRADATION", "PIT_NOW", "STAY_OUT"]
    # No HOLD_GAP, no required_actions match
    result = scoring.compute_multi_objective_scores(**kwargs)
    assert result["strategic_decisions"] <= 0.55


# ──────────────────────────────────────────────────────────────────────────
# 9. Determinism
# ──────────────────────────────────────────────────────────────────────────

def test_scoring_is_deterministic():
    """Same inputs → identical output every call."""
    kwargs = _good_dry_kwargs()
    a = scoring.compute_multi_objective_scores(**kwargs)
    b = scoring.compute_multi_objective_scores(**kwargs)
    c = scoring.compute_multi_objective_scores(**kwargs)
    assert a == b == c


def test_scoring_clamps_to_001_099():
    """weighted_final must always sit in [0.01, 0.99]."""
    # Worst case
    bad = {
        "scenario_family": "dry_strategy_sprint",
        "success_criteria": {"target_position": 1, "optimal_pit_window": [4, 4]},
        "starting_position": 5, "finishing_position": 0, "pit_decisions": [],
        "inspection_calls": {}, "final_tyre_health": 0.0, "compound_rule_satisfied": False,
        "stints": [], "track_character": "balanced", "final_fuel_kg": -1.0,
        "dnf_due_to_fuel": True, "radio_calls_made": [], "pit_wall_calls": [],
        "n_pit_stops": 0, "invalid_actions": 99, "harmful_actions": 99,
        "total_laps": 10, "steps_used": 100, "action_verbs": [],
    }
    result = scoring.compute_multi_objective_scores(**bad)
    assert 0.01 <= result["weighted_final"] <= 0.99


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _good_dry_kwargs() -> dict:
    """A dry_strategy_sprint episode with all preconditions satisfied."""
    return {
        "scenario_family": "dry_strategy_sprint",
        "success_criteria": {
            "target_position": 4,
            "optimal_pit_window": [4, 7],
            "target_n_pits": 1,
            "required_compound": "soft",
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
        "action_verbs": ["ASSESS_UNDERCUT_WINDOW", "PIT_NOW", "RADIO_DRIVER"],
    }


def _good_weather_kwargs() -> dict:
    return {
        "scenario_family": "weather_roulette",
        "success_criteria": {
            "target_position": 5,
            "optimal_pit_window": [6, 8],
            "target_n_pits": 1,
            "required_compound": "inter",
            "required_comms": ["rain"],
            "fuel_margin_kg": 0.5,
        },
        "starting_position": 5,
        "finishing_position": 3,
        "pit_decisions": [{"lap": 7, "compound": "inter", "under_sc": False}],
        "inspection_calls": {"REQUEST_FORECAST": [3], "INSPECT_FUEL_MARGIN": [10]},
        "final_tyre_health": 0.72,
        "compound_rule_satisfied": True,
        "stints": [{"compound": "medium"}, {"compound": "inter"}],
        "track_character": "weather_prone",
        "final_fuel_kg": 1.5,
        "dnf_due_to_fuel": False,
        "radio_calls_made": [{"message": "rain incoming, box for inters"}],
        "pit_wall_calls": [],
        "n_pit_stops": 1,
        "invalid_actions": 0,
        "harmful_actions": 0,
        "total_laps": 12,
        "steps_used": 14,
        "action_verbs": ["REQUEST_FORECAST", "PIT_NOW", "RADIO_DRIVER"],
    }


def _good_late_sc_kwargs() -> dict:
    return {
        "scenario_family": "late_safety_car",
        "success_criteria": {
            "target_position": 3,
            "optimal_pit_window": [8, 10],
            "target_n_pits": 1,
            "required_compound": "hard",
            "required_actions": ["HOLD_GAP"],
            "required_comms": ["sc"],
            "fuel_margin_kg": 0.5,
        },
        "starting_position": 3,
        "finishing_position": 2,
        "pit_decisions": [{"lap": 8, "compound": "hard", "under_sc": True}],
        "inspection_calls": {"INSPECT_TYRE_DEGRADATION": [1], "CHECK_OPPONENT_STRATEGY": [2]},
        "final_tyre_health": 0.75,
        "compound_rule_satisfied": True,
        "stints": [{"compound": "medium"}, {"compound": "hard"}],
        "track_character": "street",
        "final_fuel_kg": 1.0,
        "dnf_due_to_fuel": False,
        "radio_calls_made": [{"message": "sc out, boxing"}],
        "pit_wall_calls": [],
        "n_pit_stops": 1,
        "invalid_actions": 0,
        "harmful_actions": 0,
        "total_laps": 12,
        "steps_used": 14,
        "action_verbs": ["HOLD_GAP", "PIT_NOW", "RADIO_DRIVER"],
    }