"""Deterministic six-dimension scorer for F1 strategist episodes.

Hardened version (audit fix v2):
    - compute_strategic_decisions enforces *family-correct* preconditions
      (REQUEST_FORECAST for weather, ASSESS_UNDERCUT_WINDOW for dry, HOLD_GAP
      for SC). Generic "any inspection happened" is no longer enough.
    - The pit-window check requires precondition BEFORE first qualifying pit,
      not just both somewhere in the run.
    - Comms-quality keyword matching is whole-word and inflection-aware so a
      "Status update" no longer triggers credit on a rain-pit requirement.
    - Random/panic policies should plateau ≤0.40 weighted_final on every
      family; expert sequences keep clearing 0.85+. Verified by
      tests/smoke_all_scenarios.py and tests/test_scoring_strict.py.
"""

from __future__ import annotations

import re

WEIGHTS = {
    "race_result": 0.35,
    "strategic_decisions": 0.20,
    "tyre_management": 0.15,
    "fuel_management": 0.10,
    "comms_quality": 0.10,
    "operational_efficiency": 0.10,
}

DRY_COMPOUNDS = {"soft", "medium", "hard"}

# Per-family bookkeeping. The set of inspections that MUST be called before
# the optimal pit window credit is awarded, and the verb the agent must have
# used at least once to demonstrate situational awareness.
FAMILY_PRECONDITIONS = {
    "dry_strategy_sprint": {
        "required_pre_pit_inspection": "ASSESS_UNDERCUT_WINDOW",
        "required_action_verb": None,
    },
    "weather_roulette": {
        "required_pre_pit_inspection": "REQUEST_FORECAST",
        "required_action_verb": None,
    },
    "late_safety_car": {
        "required_pre_pit_inspection": None,
        "required_action_verb": "HOLD_GAP",
    },
    "championship_decider": {
        "required_pre_pit_inspection": "REQUEST_FORECAST",
        "required_action_verb": "CHECK_OPPONENT_STRATEGY",
    },
    # New families
    "virtual_safety_car_window": {
        "required_pre_pit_inspection": "ASSESS_UNDERCUT_WINDOW",
        "required_action_verb": None,
    },
    "tyre_cliff_management": {
        "required_pre_pit_inspection": "INSPECT_TYRE_DEGRADATION",
        "required_action_verb": None,
    },
}


def compute_race_result(
    starting_position: int,
    finishing_position: int,
    target_position: int,
    rival_finishing_position: int | None = None,
) -> float:
    if finishing_position <= 0:
        return 0.0
    score = 1.0 if finishing_position <= target_position else 0.0
    if not score and finishing_position <= starting_position:
        score = 0.55
    if not score and finishing_position == target_position + 1:
        score = 0.35
    if rival_finishing_position is not None:
        score = (
            min(score, 0.65) if finishing_position > rival_finishing_position else max(score, 0.95)
        )
    return _clamp01(score)


def compute_strategic_decisions(
    pit_decisions: list,
    optimal_pit_window: tuple[int, int],
    inspection_calls: dict,
    forecast_called_before_pit: bool,
    undercut_assessed_before_pit: bool,
) -> float:
    """Generic strategic-decision dim. Keeps the simple shape used by the
    isolation tests in test_scoring.py; the family-specific tightening lives
    in `_scenario_strategy_adjustment`.
    """
    if not pit_decisions:
        return 0.0
    lo, hi = optimal_pit_window
    in_window = any(lo <= int(p.get("lap", 0)) <= hi for p in pit_decisions)
    if not in_window:
        return 0.25
    precondition = forecast_called_before_pit or undercut_assessed_before_pit
    return 1.0 if precondition else 0.65


def compute_tyre_management(
    final_tyre_health: float,
    compound_rule_satisfied: bool,
    stints: list,
    track_character: str,
) -> float:
    health_floor = 0.30 if track_character == "street" else 0.40
    health_score = (
        1.0 if final_tyre_health >= health_floor else max(0.0, final_tyre_health / health_floor)
    )
    compound_score = 1.0 if compound_rule_satisfied else 0.0
    return _clamp01(0.55 * compound_score + 0.45 * health_score)


def compute_fuel_management(
    final_fuel_kg: float,
    target_margin_kg: float,
    dnf_due_to_fuel: bool,
    inspect_fuel_called: bool,
) -> float:
    if dnf_due_to_fuel or final_fuel_kg < 0:
        return 0.0
    margin_score = min(1.0, final_fuel_kg / max(0.1, target_margin_kg))
    return _clamp01(0.85 * margin_score + (0.15 if inspect_fuel_called else 0.0))


def compute_comms_quality(
    triggered_comms_events: list,
    radio_calls_made: list,
    pit_wall_calls: list,
) -> float:
    """Whole-word keyword match against radio + pit-wall messages.

    A required trigger like "pit" only counts if a message contains the word
    "pit" (or a plausible inflection like "pits"/"pitting"/"pitlane"); a
    generic "Status update." no longer scores. Empty trigger list returns 1.0
    (nothing was required).
    """
    required = [str(x).lower().strip() for x in triggered_comms_events if x]
    required = [r for r in required if r]
    if not required:
        return 1.0
    text = " ".join(
        str(call.get("message", call)).lower() for call in radio_calls_made + pit_wall_calls
    )
    hits = sum(1 for needle in required if _keyword_hits(needle, text))
    return _clamp01(hits / len(required))


def compute_operational_efficiency(
    n_pit_stops: int,
    target_n_pits: int,
    invalid_actions: int,
    harmful_actions: int,
    total_laps: int,
    steps_used: int,
) -> float:
    pit_score = (
        1.0
        if n_pit_stops == target_n_pits
        else max(0.0, 1.0 - 0.45 * abs(n_pit_stops - target_n_pits))
    )
    penalty = 0.12 * invalid_actions + 0.18 * harmful_actions
    if steps_used > total_laps + 8:
        penalty += min(0.20, 0.02 * (steps_used - total_laps - 8))
    return _clamp01(pit_score - penalty)


def compute_multi_objective_scores(**kwargs) -> dict:
    """Compute all dimensions and a weighted final score clamped to [0.01, 0.99]."""
    pit_decisions = kwargs.get("pit_decisions", [])
    inspection_calls = kwargs.get("inspection_calls", {})
    scenario_family = kwargs.get("scenario_family", "")
    criteria = kwargs.get("success_criteria", {}) or {}
    optimal_window = tuple(
        criteria.get("optimal_pit_window", kwargs.get("optimal_pit_window", (1, 999)))
    )
    required_compound = criteria.get("required_compound")
    required_actions = set(criteria.get("required_actions", []))
    action_verbs = set(kwargs.get("action_verbs", []))

    first_pit_lap = min((int(p.get("lap", 999)) for p in pit_decisions), default=999)
    forecast_before = any(
        lap <= first_pit_lap for lap in inspection_calls.get("REQUEST_FORECAST", [])
    )
    undercut_before = any(
        lap <= first_pit_lap for lap in inspection_calls.get("ASSESS_UNDERCUT_WINDOW", [])
    )

    race = compute_race_result(
        kwargs.get("starting_position", 99),
        kwargs.get("finishing_position", 99),
        criteria.get("target_position", kwargs.get("target_position", 99)),
        kwargs.get("rival_finishing_position"),
    )

    strategic = compute_strategic_decisions(
        pit_decisions,
        optimal_window,
        inspection_calls,
        forecast_before,
        undercut_before,
    )
    strategic = _scenario_strategy_adjustment(
        scenario_family,
        strategic,
        pit_decisions,
        optimal_window,
        required_compound,
        forecast_before,
        undercut_before,
        required_actions,
        action_verbs,
        inspection_calls,
        n_pit_stops=int(kwargs.get("n_pit_stops", len(pit_decisions))),
        target_n_pits=int(criteria.get("target_n_pits", kwargs.get("target_n_pits", 1))),
    )

    tyre = compute_tyre_management(
        kwargs.get("final_tyre_health", 0.0),
        kwargs.get("compound_rule_satisfied", False),
        kwargs.get("stints", []),
        kwargs.get("track_character", "balanced"),
    )
    if required_compound and not any(
        stint.get("compound") == required_compound for stint in kwargs.get("stints", [])
    ):
        tyre = min(tyre, 0.20)
    fuel = compute_fuel_management(
        kwargs.get("final_fuel_kg", 0.0),
        criteria.get("fuel_margin_kg", kwargs.get("target_margin_kg", 0.5)),
        kwargs.get("dnf_due_to_fuel", False),
        bool(inspection_calls.get("INSPECT_FUEL_MARGIN")),
    )
    comms = compute_comms_quality(
        criteria.get("required_comms", kwargs.get("triggered_comms_events", [])),
        kwargs.get("radio_calls_made", []),
        kwargs.get("pit_wall_calls", []),
    )
    ops = compute_operational_efficiency(
        kwargs.get("n_pit_stops", 0),
        criteria.get("target_n_pits", kwargs.get("target_n_pits", 1)),
        kwargs.get("invalid_actions", 0),
        kwargs.get("harmful_actions", 0),
        kwargs.get("total_laps", 0),
        kwargs.get("steps_used", 0),
    )

    # Race-result halo: only if strategic dim is high AND comms quality is at
    # least half-way decent. Random agents that pit lucky but never radio the
    # driver get strategic=0.95 but comms=0 → no halo. Real expert play gets
    # both, so the bonus still applies.
    if strategic >= 0.95 and comms >= 0.5:
        race = max(race, 0.90)

    dims = {
        "race_result": _clamp01(race),
        "strategic_decisions": _clamp01(strategic),
        "tyre_management": _clamp01(tyre),
        "fuel_management": _clamp01(fuel),
        "comms_quality": _clamp01(comms),
        "operational_efficiency": _clamp01(ops),
    }
    weighted = sum(dims[k] * WEIGHTS[k] for k in WEIGHTS)
    dims["weighted_final"] = min(0.99, max(0.01, round(weighted, 4)))
    return dims


def _scenario_strategy_adjustment(
    scenario_family: str,
    base: float,
    pit_decisions: list,
    optimal_window: tuple[int, int],
    required_compound: str | None,
    forecast_before: bool,
    undercut_before: bool,
    required_actions: set[str],
    action_verbs: set[str],
    inspection_calls: dict,
    n_pit_stops: int = 0,
    target_n_pits: int = 1,
) -> float:
    """Family-specific strict scoring of the strategic dimension.

    Every family now requires:
      1. A pit decision in (or within tight tolerance of) the optimal window
      2. The pit uses the required compound when one is specified
      3. The family-specific precondition fires BEFORE the first qualifying pit
      4. n_pit_stops is at most target_n_pits + 0 (extra pits cap the dim)
    Missing any → caps at 0.30–0.65 (rewards partial play but stays clear of
    the 0.95 strategic-halo trigger).
    """
    lo, hi = optimal_window
    tolerance = 2 if scenario_family in {"dry_strategy_sprint", "weather_roulette"} else 0

    matching_pits = [
        p
        for p in pit_decisions
        if lo <= int(p.get("lap", 0)) <= hi + tolerance
        and (required_compound is None or p.get("compound") == required_compound)
    ]
    if not matching_pits:
        return {
            "dry_strategy_sprint": min(base, 0.30),
            "weather_roulette": 0.20,
            "late_safety_car": 0.15,
            "championship_decider": min(base, 0.35),
            "virtual_safety_car_window": 0.20,
            "tyre_cliff_management": 0.20,
        }.get(scenario_family, base)

    # Over-pitting cap: stratagy was at best partially correct if the agent
    # pitted more than target. One extra pit → cap at 0.60. Two+ → 0.40.
    over_pit_cap = 1.0
    if n_pit_stops > target_n_pits:
        extra = n_pit_stops - target_n_pits
        over_pit_cap = 0.60 if extra == 1 else 0.40

    first_qualifying_pit_lap = min(int(p.get("lap", 999)) for p in matching_pits)
    rules = FAMILY_PRECONDITIONS.get(scenario_family, {})
    insp_required = rules.get("required_pre_pit_inspection")
    verb_required = rules.get("required_action_verb")

    insp_ok = True
    if insp_required:
        insp_laps = inspection_calls.get(insp_required, [])
        insp_ok = any(int(lap) <= first_qualifying_pit_lap for lap in insp_laps)

    verb_ok = True
    if verb_required:
        verb_ok = verb_required in action_verbs

    if scenario_family == "dry_strategy_sprint":
        result = 1.0 if insp_ok else 0.55
        return min(result, over_pit_cap)

    if scenario_family == "weather_roulette":
        result = 1.0 if insp_ok else 0.45
        return min(result, over_pit_cap)

    if scenario_family == "late_safety_car":
        under_sc = any(p.get("under_sc") for p in matching_pits)
        if required_actions and not (required_actions & action_verbs):
            return min(0.45, over_pit_cap)
        if not verb_ok:
            return min(0.55, over_pit_cap)
        result = 1.0 if under_sc else 0.65
        return min(result, over_pit_cap)

    if scenario_family == "championship_decider":
        result = 1.0 if (insp_ok and verb_ok) else 0.55
        return min(result, over_pit_cap)

    if scenario_family == "virtual_safety_car_window":
        # Reward pitting under VSC (cheap pit) as the highest-quality play.
        under_vsc = any(p.get("under_sc") for p in matching_pits)
        result = 1.0 if (insp_ok and under_vsc) else (0.70 if insp_ok else 0.45)
        return min(result, over_pit_cap)

    if scenario_family == "tyre_cliff_management":
        # Reward inspecting tyre degradation before the cliff pit.
        result = 1.0 if insp_ok else 0.45
        return min(result, over_pit_cap)

    return min(base, over_pit_cap)


_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _keyword_hits(needle: str, haystack: str) -> bool:
    """True iff `needle` (already lowercased) appears as a whole word or a
    plausible inflection in `haystack`. Matches `pit`, `pits`, `pitting`,
    `pitted`, `pitlane`; does NOT match `spit` or `split`. Handles common
    English doubled-consonant inflections (pit → pitting).
    """
    needle = needle.strip()
    if not needle:
        return False
    pattern = _WORD_RE_CACHE.get(needle)
    if pattern is None:
        # Allow:
        #   - bare needle             (pit)
        #   - +s/+es                  (pits)
        #   - +ed / +ted              (pitted, called)
        #   - +ing / +ting            (pitting, calling)
        #   - +lane / +line           (pitlane, finishline)
        #   - +ner / +ter             (cornering)
        last = re.escape(needle[-1]) if needle else ""
        pattern = re.compile(
            rf"\b{re.escape(needle)}(?:{last}?ing|{last}?ed|s|es|lane|line)?\b"
        )
        _WORD_RE_CACHE[needle] = pattern
    return bool(pattern.search(haystack))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))