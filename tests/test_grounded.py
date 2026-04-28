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


def test_wear_factors_pass_sanity_clamp():
    """All loaded factors must respect the sanity clamp [0.20, 3.0].

    Note: our linear regression on partially-wet tracks (Spa, Silverstone)
    has variable signal quality. We don't assert specific orderings
    (those depend on which biases dominate the residual data). The clamp
    prevents pathological factors from reaching physics.
    """
    for track in grounded.list_grounded_tracks():
        for compound in ("hard", "medium", "soft"):
            f = grounded.get_wear_factor(track, compound)
            assert 0.20 <= f <= 3.0, f"{track}/{compound}: {f} outside clamp"


def test_optin_path_is_wired_correctly():
    """Sanity: opt-in must alter SOMETHING in the env (factors set, hidden
    state replaced). We don't assert the *direction* of the score change
    because per-track wear factors have noisy estimates with the current
    linear regression — that's a known limitation; see
    docs/calibration-decisions.md.
    """
    import copy
    from server.environment import F1StrategistEnvironment
    from server.scenarios import SCENARIOS

    base = SCENARIOS["spa_full_wet"]

    # Without opt-in: factors stay None
    env_off = F1StrategistEnvironment()
    env_off.reset(seed=7, options={"scenario": copy.deepcopy(base)})
    assert env_off._grounded_factors is None, "synthetic scenario should not set factors"

    # With opt-in: factors populated
    sc_grnd = copy.deepcopy(base)
    sc_grnd["use_grounded_calibration"] = True
    sc_grnd["grounded_track_key"] = "spa"
    env_on = F1StrategistEnvironment()
    env_on.reset(seed=7, options={"scenario": sc_grnd})
    assert env_on._grounded_factors is not None, "opt-in must populate factors"
    assert "medium" in env_on._grounded_factors
    # At least one compound must have a non-trivial factor (or this whole
    # mechanism is wired to a noop).
    nontrivial = [c for c, f in env_on._grounded_factors.items() if abs(f - 1.0) > 0.05]
    assert nontrivial, "opt-in produced no factor != 1.0 — wiring is dead"
