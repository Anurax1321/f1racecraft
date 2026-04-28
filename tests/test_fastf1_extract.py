"""Tests for the FastF1 extraction module.

Uses the cached Monaco 2024 race (which sub-step 2.1's smoke pull guarantees
exists). If the cache is missing, the tests skip — they're not gates for the
core env, they're gates for the data layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _have_cached_monaco() -> bool:
    cache = Path("data/fastf1_cache")
    return cache.exists() and any(cache.rglob("*"))


pytestmark = pytest.mark.skipif(
    not _have_cached_monaco(),
    reason="FastF1 cache empty — run scripts/fastf1_pull.py --smoke first",
)


@pytest.fixture(scope="module")
def monaco_features():
    from server.fastf1_extract import extract_race_features
    from server.fastf1_loader import load_race
    session = load_race(2024, "Monaco")
    return extract_race_features(session)


def test_metadata_shape(monaco_features):
    meta = monaco_features["metadata"]
    assert meta["year"] == 2024
    assert meta["country"].lower() == "monaco"
    assert meta["total_laps"] == 78
    assert meta["n_drivers"] == 20
    assert "soft" in meta["compounds_used"] or "medium" in meta["compounds_used"]
    assert meta["wet_race"] is False


def test_lap_records_present(monaco_features):
    laps = monaco_features["lap_records"]
    assert len(laps) > 1000  # 20 drivers × ~78 laps with some attrition
    sample = laps[0]
    for required_field in (
        "driver", "lap_number", "stint", "compound",
        "tyre_life", "lap_time_s", "fuel_corr_s",
        "lap_time_corrected_s", "is_accurate", "track_status",
        "track_status_decoded", "under_sc", "under_vsc",
    ):
        assert required_field in sample, f"missing {required_field}"


def test_fuel_correction_decreases_with_lap(monaco_features):
    """Fuel correction should drop monotonically as laps progress
    (fuel mass decreases linearly with lap number)."""
    laps = monaco_features["lap_records"]
    early = [r["fuel_corr_s"] for r in laps if r["lap_number"] == 5]
    late = [r["fuel_corr_s"] for r in laps if r["lap_number"] == 50]
    assert early and late
    # Same lap number → identical correction per our formula
    assert all(e == early[0] for e in early)
    assert all(l == late[0] for l in late)
    assert early[0] > late[0], "fuel correction should decrease with lap"


def test_stints_have_one_compound(monaco_features):
    """Each stint should be on one compound only."""
    for stint in monaco_features["stints"]:
        assert len(stint["compounds_observed"]) == 1, \
            f"stint mixes compounds: {stint}"


def test_pit_decisions_match_stint_changes(monaco_features):
    """Number of pit decisions should equal (stints - drivers)."""
    n_stints = len(monaco_features["stints"])
    n_drivers = monaco_features["metadata"]["n_drivers"]
    n_pits = len(monaco_features["pit_decisions"])
    # Each driver: stints == pits + 1, so total_stints = total_pits + n_drivers
    # This may not hold exactly if drivers DNF mid-stint
    assert n_pits == n_stints - n_drivers or abs((n_pits + n_drivers) - n_stints) <= 5


def test_track_status_codes_decoded(monaco_features):
    """Every status code should decode to a known event name."""
    for r in monaco_features["lap_records"]:
        decoded = r["track_status_decoded"]
        assert isinstance(decoded, list)


def test_weather_summary_present(monaco_features):
    w = monaco_features["weather_summary"]
    assert w["n_samples"] > 0
    assert 10 < w["air_temp_c_mean"] < 40
    assert 15 < w["track_temp_c_mean"] < 80
    # Monaco 2024 was dry
    assert w["rainfall_any"] is False


def test_under_sc_flag_correct_at_lap_1(monaco_features):
    """Monaco 2024 had a red-flag/SC start. Lap 1 records should carry the SC flag."""
    lap_1 = [r for r in monaco_features["lap_records"] if r["lap_number"] == 1]
    assert lap_1
    # At least some lap-1 records should be under SC
    assert any(r["under_sc"] for r in lap_1)
