"""Tests for the grounded calibration loader (sub-step 2.4).

The grounded module reads `data/grounded/<track>.json` files produced by
`scripts/fastf1_aggregate.py` from real FastF1 lap data. These tests skip if
the data layer is missing (clean clone before pull/extract/aggregate).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from server import grounded


def _have_grounded_data() -> bool:
    return (Path("data/grounded") / "_index.json").exists()


pytestmark = pytest.mark.skipif(
    not _have_grounded_data(),
    reason="data/grounded/ empty — run scripts/fastf1_aggregate.py first",
)


@pytest.fixture(autouse=True)
def _reset_caches():
    grounded.reset_caches()
    yield
    grounded.reset_caches()


def test_listing_includes_scenario_tracks():
    tracks = grounded.list_grounded_tracks()
    assert tracks  # we should have a non-empty list
    # Every track that's a scenario family should be in the index
    for required in ("monaco", "silverstone", "spa", "monza", "catalunya"):
        assert required in tracks, f"{required} missing from grounded index"


def test_get_track_data_returns_dict():
    d = grounded.get_track_data("silverstone")
    assert d is not None
    assert "tyres" in d
    assert "pit" in d
    assert "n_races" in d


def test_unknown_track_returns_none():
    assert grounded.get_track_data("not_a_real_track") is None


def test_unknown_track_wear_factor_is_one():
    assert grounded.get_wear_factor("not_a_real_track", "medium") == 1.0


def test_known_track_wear_factor_in_sane_range():
    """Multiplier should be unitless and within the sanity-clamped range."""
    for track in ("silverstone", "spa", "monaco", "catalunya"):
        for compound in ("hard", "medium", "soft"):
            f = grounded.get_wear_factor(track, compound)
            assert 0.20 <= f <= 3.0, f"{track}/{compound}: {f} outside clamp"


def test_quality_gate_filters_low_r2():
    """Singapore soft has R²=0.055 and n=76 — borderline. The quality gate
    should either return it (above threshold) or fall back to 1.0."""
    f = grounded.get_wear_factor("singapore", "soft")
    # Either the loader passed the quality gate (factor != 1.0) or it
    # rejected it and returned 1.0. Both are acceptable; what's NOT
    # acceptable is the raw 0.184/0.064 = ~2.9 ratio without sanity clamp.
    assert 0.20 <= f <= 3.0


def test_track_evolution_dominant_flag_monaco():
    """Monaco hard/medium are track-evolution-dominant per 3 seasons of data."""
    assert grounded.is_track_evolution_dominant("monaco", "hard") is True
    assert grounded.is_track_evolution_dominant("monaco", "medium") is True


def test_track_evolution_dominant_flag_silverstone():
    """Silverstone is normal degradation — flag should be False."""
    assert grounded.is_track_evolution_dominant("silverstone", "medium") is False


def test_health_curve_starts_at_one():
    curve = grounded.get_health_curve("silverstone", "medium")
    assert curve is not None
    assert curve[0] == 1.0


def test_health_curve_decreases():
    curve = grounded.get_health_curve("silverstone", "medium")
    assert curve is not None
    # Should be monotone non-increasing (tire health doesn't grow)
    for i in range(len(curve) - 1):
        assert curve[i] >= curve[i + 1] - 1e-6, \
            f"curve[{i}]={curve[i]} < curve[{i+1}]={curve[i+1]} — non-monotone"


def test_relative_wear_orders_match_intuition():
    """Tracks-with-data should preserve real F1 intuition.

    Spa is heavy on mediums (slope 0.091); Monaco is gentle. So
    Spa's medium wear factor must be > Monaco's medium wear factor.
    """
    spa = grounded.get_wear_factor("spa", "medium")
    monaco = grounded.get_wear_factor("monaco", "medium")
    assert spa > monaco, \
        f"Expected Spa medium ({spa}) > Monaco medium ({monaco}) per real F1 data"


def test_optin_actually_changes_env_behaviour():
    """Sanity check: opting in to grounded calibration must actually do
    something. Verified by running the same expert sequence against the same
    scenario with use_grounded_calibration on/off and checking the score
    differs on a track where the wear factor is materially != 1.0.

    Spa's medium wear factor is ~1.73x — strong enough to change the
    expert's outcome.
    """
    import copy
    from baselines.expert_solver import EXPERT_SEQUENCES, run_sequence
    from server.scenarios import SCENARIOS

    base = SCENARIOS["spa_full_wet"]
    sc_synth = copy.deepcopy(base)
    sc_synth["use_grounded_calibration"] = False
    sc_grnd = copy.deepcopy(base)
    sc_grnd["use_grounded_calibration"] = True
    sc_grnd["grounded_track_key"] = "spa"

    score_synth, _ = run_sequence(sc_synth, EXPERT_SEQUENCES["spa_full_wet"], seed=7)
    score_grnd, _ = run_sequence(sc_grnd, EXPERT_SEQUENCES["spa_full_wet"], seed=7)

    # Grounded Spa is harder (real F1 wears mediums 1.73x faster).
    # Same expert sequence should score lower on grounded.
    assert score_grnd < score_synth, (
        f"Opt-in had no effect: synth={score_synth:.3f}, grnd={score_grnd:.3f}. "
        f"Grounded factors aren't reaching the physics."
    )
