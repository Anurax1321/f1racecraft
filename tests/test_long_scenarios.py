"""Tests for sub-step 1.2 — long-race scenario families.

Verifies:
  - All 3 long families register in SCENARIOS.
  - Each loads via env.reset() and produces a coherent observation.
  - total_laps == 55 and weather per_lap array length matches.
  - Each can run end-to-end without crashing.
  - Each has at least one 2-stop strategic-decision issue.
  - Realistic starting fuel (90–115 kg).
"""

from __future__ import annotations

import pytest

from models import F1Action
from server.environment import F1StrategistEnvironment
from server.scenarios import SCENARIOS
from server.scenarios_long import LONG_SCENARIOS


LONG_FAMILIES = ["monaco_full_gp", "silverstone_full_gp", "spa_full_wet"]


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_family_registered(family):
    assert family in SCENARIOS, f"{family} missing from SCENARIOS registry"
    assert family in LONG_SCENARIOS


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_total_laps_is_55(family):
    assert SCENARIOS[family]["total_laps"] == 55


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_weather_array_length_matches(family):
    sc = SCENARIOS[family]
    per_lap = (sc.get("weather_seed_overrides") or {}).get("per_lap", [])
    assert len(per_lap) == sc["total_laps"], \
        f"{family}: weather per_lap is {len(per_lap)} entries, expected {sc['total_laps']}"


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_realistic_starting_fuel(family):
    fuel = SCENARIOS[family]["starting_fuel_kg"]
    assert 90 <= fuel <= 115, \
        f"{family}: starting_fuel={fuel} outside realistic [90, 115] for full GP"


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_has_at_least_two_pit_decisions(family):
    """Long races require multi-stop strategy — verify ≥2 pit-related decisions."""
    issues = SCENARIOS[family].get("issues", {}).get("strategic_decisions", [])
    pit_decisions = [
        iss for iss in issues
        if any(k in iss.get("decision", "")
               for k in ("pit", "stop", "switch", "inter", "slick"))
    ]
    assert len(pit_decisions) >= 2, \
        f"{family}: only {len(pit_decisions)} pit-related decisions (need >=2 for multi-stop)"


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_env_reset_loads_family(family):
    env = F1StrategistEnvironment()
    obs = env.reset(seed=7, options={"task": family})
    assert obs.total_laps == 55
    assert obs.race_status == "green"
    assert obs.done is False


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_end_to_end_does_not_crash(family):
    """A naive STAY_OUT-then-pit-twice run should complete without errors."""
    env = F1StrategistEnvironment()
    obs = env.reset(seed=11, options={"task": family})

    steps = 0
    while not obs.done and steps < obs.total_laps + 8:
        if steps == 5:
            cmd = "REQUEST_FORECAST"
        elif steps == 20:
            cmd = "PIT_NOW medium"
        elif steps == 40:
            cmd = "PIT_NOW soft"
        else:
            cmd = "STAY_OUT"
        obs = env.step(F1Action(command=cmd))
        steps += 1

    assert obs.done, f"{family}: race did not terminate within {obs.total_laps + 8} steps"
    # Any score in the valid clamp is acceptable here — we're just checking no crashes.
    assert 0.0 <= obs.score <= 1.0


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_has_4_or_more_opponents(family):
    opps = SCENARIOS[family].get("opponents", [])
    assert len(opps) >= 4, f"{family}: only {len(opps)} opponents (expected >=4)"


def test_spa_full_wet_has_rain_phase():
    """Spa scenario must include a wet phase mid-race."""
    per_lap = SCENARIOS["spa_full_wet"]["weather_seed_overrides"]["per_lap"]
    peak = max(p.get("rain_intensity", 0.0) for p in per_lap)
    assert peak >= 0.4, f"spa_full_wet rain peak only {peak}"
    # Race opens dry
    assert per_lap[0]["rain_intensity"] == 0.0
    # Race closes dry (after rain passes)
    assert per_lap[-1]["rain_intensity"] == 0.0


def test_monaco_has_sc_archetype():
    """Monaco's whole strategic shape depends on SC probability."""
    assert SCENARIOS["monaco_full_gp"]["sc_archetype"] in ("midrace_likely", "late_likely")


def test_silverstone_has_vsc_archetype():
    assert SCENARIOS["silverstone_full_gp"]["sc_archetype"] in ("vsc_likely", "midrace_likely")


# ─────────────────────────────────────────────────────────────────────────
# Sub-step 1.4 gates: expert sequences score ≥0.85 on every long family
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_expert_sequence_scores_above_floor(family):
    from baselines.expert_solver import EXPERT_SEQUENCES, run_sequence
    assert family in EXPERT_SEQUENCES, f"{family}: no expert sequence registered"
    scores = []
    for seed in [7, 11, 42]:
        score, _ = run_sequence(SCENARIOS[family], EXPERT_SEQUENCES[family], seed=seed)
        scores.append(score)
    avg = sum(scores) / len(scores)
    assert avg >= 0.85, \
        f"{family}: expert avg score {avg:.3f} below 0.85 floor (seeds 7/11/42: {scores})"


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_expert_beats_panic(family):
    """Expert sequence must beat panic by >=0.20 — confirms the scorer
    discriminates good play from bad on long-race families."""
    from baselines.expert_solver import EXPERT_SEQUENCES, PANIC_SEQUENCES, run_sequence
    e_score, _ = run_sequence(SCENARIOS[family], EXPERT_SEQUENCES[family], seed=7)
    p_score, _ = run_sequence(SCENARIOS[family], PANIC_SEQUENCES[family], seed=7)
    assert e_score - p_score >= 0.20, \
        f"{family}: expert {e_score:.3f} - panic {p_score:.3f} = {e_score-p_score:+.3f} (need >=0.20)"


@pytest.mark.parametrize("family", LONG_FAMILIES)
def test_overpit_does_not_score_higher_than_untrained(family):
    """Sub-step 1.5 hardening: a pit-spam policy must NOT outscore the
    no-op (untrained) baseline. Found during 1.5 baseline measurement —
    random pit-spam scored 0.66 on Silverstone before the over-pit cap.
    """
    from baselines.expert_solver import run_sequence
    pit_spam = ["PIT_NOW soft"] * 25 + ["DONE"]  # 25 pits — egregious over-pit
    no_op = ["STAY_OUT"] * 55 + ["DONE"]
    overpit_score, _ = run_sequence(SCENARIOS[family], pit_spam, seed=7)
    noop_score, _ = run_sequence(SCENARIOS[family], no_op, seed=7)
    assert overpit_score <= noop_score + 0.10, (
        f"{family}: pit-spam {overpit_score:.3f} > no-op {noop_score:.3f} + 0.10 — "
        f"over-pit cap not working"
    )
