"""
Self-contained helpers for the /simulate endpoint.

All logic is inline — no root-level imports (inference.py / models.py) so this
module works reliably when imported from within server/app.py on any deployment.
"""

from __future__ import annotations

import re
import random as _random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import F1Observation


SYSTEM_PROMPT = """\
You are an F1 race strategist. Respond with exactly one command.
Commands: PIT_NOW <compound> | STAY_OUT | SET_MODE <mode> | RADIO_DRIVER "<msg>" |
REQUEST_FORECAST | INSPECT_TYRE_DEGRADATION | ASSESS_UNDERCUT_WINDOW |
INSPECT_FUEL_MARGIN | CHECK_OPPONENT_STRATEGY <num> | MANAGE_TYRE_TEMP <warm|cool|normal> |
DEFEND_POSITION | ATTACK_AHEAD | HOLD_GAP <num> | LET_BY <num> | DONE
"""

VALID_VERBS = {
    "PIT_NOW", "STAY_OUT", "RECOMMEND_PIT", "SET_MODE", "DRS_PERMISSION",
    "DEFEND_POSITION", "ATTACK_AHEAD", "HOLD_GAP", "LET_BY",
    "INSPECT_TYRE_DEGRADATION", "CHECK_OPPONENT_STRATEGY", "REQUEST_FORECAST",
    "ASSESS_UNDERCUT_WINDOW", "INSPECT_FUEL_MARGIN", "MANAGE_TYRE_TEMP",
    "RADIO_DRIVER", "DRAFT_PIT_WALL", "BRIEF_MEDIA", "REQUEST_INFO",
    "ESCALATE_TO_PRINCIPAL", "DONE",
}

RANDOM_ACTIONS = [
    "STAY_OUT", "INSPECT_TYRE_DEGRADATION", "REQUEST_FORECAST",
    "ASSESS_UNDERCUT_WINDOW", "INSPECT_FUEL_MARGIN",
    "PIT_NOW soft", "PIT_NOW medium", "PIT_NOW hard",
    "SET_MODE push", "SET_MODE conserve", "SET_MODE race",
    "DEFEND_POSITION", "HOLD_GAP 1",
]

DIM_KEYS = [
    "race_result", "strategic_decisions", "tyre_management",
    "fuel_management", "comms_quality", "operational_efficiency",
]


def parse_action(text: str) -> str:
    """Return the first valid command found in text, falling back to STAY_OUT."""
    text = re.sub(r"</?think>", "", text or "", flags=re.IGNORECASE).strip()
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if not line:
            continue
        verb = line.split(None, 1)[0].upper()
        if verb in VALID_VERBS:
            return line
    m = re.search(
        r"\b(" + "|".join(sorted(VALID_VERBS, key=len, reverse=True)) + r")\b(.*)",
        text, re.I,
    )
    if m:
        return (m.group(1).upper() + m.group(2)).strip()
    return "STAY_OUT"


def surface_label(weather_dict: dict | None) -> str:
    """Convert WeatherState dict to a human-readable condition string."""
    if not weather_dict:
        return "dry"
    rain = weather_dict.get("rain_intensity", 0.0)
    surface = weather_dict.get("surface_state", "dry")
    if rain > 0.5 or surface == "wet":
        return "heavy rain"
    if rain > 0.15 or surface == "damp":
        return "light rain"
    return "dry"


# ---------------------------------------------------------------------------
# Self-contained heuristic policy
# ---------------------------------------------------------------------------

def heuristic(family: str, lap: int, total_laps: int, obs, seen: set) -> str:
    """
    Rule-based policy for all six scenario families.

    ``seen`` is a mutable set that accumulates short string tokens for actions
    that have already been taken, allowing stateful decisions without full
    history tracking.
    """
    rain = (obs.weather_current or {}).get("rain_intensity", 0.0)
    compound = obs.ego_tyre_compound
    health = obs.ego_tyre_health_pct

    # ── Weather Roulette ───────────────────────────────────────────────────
    if family == "weather_roulette":
        if "forecast" not in seen:
            seen.add("forecast")
            return "REQUEST_FORECAST"
        if lap < 5:
            return "STAY_OUT"
        if compound != "inter" and rain >= 0.10 and "radio_rain" not in seen:
            seen.add("radio_rain")
            return f'RADIO_DRIVER "Box now for inters — rain at {rain*100:.0f}%."'
        if compound != "inter" and (rain >= 0.10 or lap >= 7):
            return "PIT_NOW inter"
        return "DONE" if lap >= total_laps else "STAY_OUT"

    # ── Late Safety Car ────────────────────────────────────────────────────
    if family == "late_safety_car":
        sc = obs.race_status in {"sc", "vsc"}
        if "holdgap" not in seen and lap < 7:
            seen.add("holdgap")
            return "HOLD_GAP 4"
        if sc and compound != "hard" and "radio_sc" not in seen:
            seen.add("radio_sc")
            return 'RADIO_DRIVER "Safety car — box this lap for hards."'
        if sc and compound != "hard":
            return "PIT_NOW hard"
        return "DONE" if lap >= total_laps else "STAY_OUT"

    # ── Championship Decider ───────────────────────────────────────────────
    if family == "championship_decider":
        if "checkop" not in seen:
            seen.add("checkop")
            return "CHECK_OPPONENT_STRATEGY 10"
        if "forecast" not in seen:
            seen.add("forecast")
            return "REQUEST_FORECAST"
        if "radio_rival" not in seen and lap >= 3:
            seen.add("radio_rival")
            return 'RADIO_DRIVER "Covering the rival — mirror their strategy."'
        if lap >= 7 and compound == "hard" and "radio_pit2" not in seen:
            seen.add("radio_pit2")
            return 'RADIO_DRIVER "Pitting for mediums — rival covered."'
        if lap >= 7 and compound == "hard" and "radio_pit2" in seen:
            return "PIT_NOW medium"
        return "DONE" if lap >= total_laps else "STAY_OUT"

    # ── VSC Window ─────────────────────────────────────────────────────────
    if family == "virtual_safety_car_window":
        if "undercut" not in seen:
            seen.add("undercut")
            return "ASSESS_UNDERCUT_WINDOW"
        vsc = obs.race_status == "vsc"
        if vsc and compound != "soft" and "radio_vsc" not in seen:
            seen.add("radio_vsc")
            return 'RADIO_DRIVER "VSC deployed — box now for softs."'
        if vsc and compound != "soft":
            return "PIT_NOW soft"
        if lap < 8:
            return "STAY_OUT"
        return "DONE" if lap >= total_laps else "STAY_OUT"

    # ── Tyre Cliff Management ──────────────────────────────────────────────
    if family == "tyre_cliff_management":
        if "inspect" not in seen:
            seen.add("inspect")
            return "INSPECT_TYRE_DEGRADATION"
        if health < 60 and "cool" not in seen:
            seen.add("cool")
            return "MANAGE_TYRE_TEMP cool"
        if lap >= 4 and health < 55 and "radio_cliff" not in seen:
            seen.add("radio_cliff")
            return 'RADIO_DRIVER "Tyre cliff incoming — boxing for mediums."'
        if compound == "soft" and health < 50 and "radio_cliff" in seen:
            return "PIT_NOW medium"
        if lap >= 7 and compound == "soft" and "radio_clff2" not in seen:
            seen.add("radio_clff2")
            return 'RADIO_DRIVER "Tyre cliff — committing pit for mediums."'
        if lap >= 7 and compound == "soft":
            return "PIT_NOW medium"
        return "STAY_OUT"

    # ── Dry Strategy Sprint (default) ──────────────────────────────────────
    if "undercut" not in seen:
        seen.add("undercut")
        return "ASSESS_UNDERCUT_WINDOW"
    if lap >= 4 and "radio_undercut" not in seen:
        seen.add("radio_undercut")
        return 'RADIO_DRIVER "Undercut window open — pit call coming next lap."'
    if lap >= 5 and compound != "soft":
        if "radio_pit" not in seen:
            seen.add("radio_pit")
            return 'RADIO_DRIVER "Pit call — boxing for softs."'
        return "PIT_NOW soft"
    return "DONE" if lap >= total_laps else "STAY_OUT"


def random_action(lap: int, total_laps: int) -> str:
    """Return a random valid action."""
    if lap >= total_laps:
        return "DONE"
    return _random.choice(RANDOM_ACTIONS)


def get_scenario_family(task: str) -> str:
    """Return the scenario family for a task name, falling back to the task itself."""
    try:
        from server.scenarios import SCENARIOS
        return SCENARIOS.get(task, {}).get("scenario_family", task)
    except Exception:
        return task
