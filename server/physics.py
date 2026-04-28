"""
F1 Strategist physics model.

The functions here intentionally model strategy-scale effects rather than car
dynamics: compound pace, degradation, fuel mass, dirty air, weather crossover,
and small deterministic noise. Constants mirror docs/physics-model.md.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid runtime import cycle
    from server.track import Track
    from server.weather import WeatherState


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TYRE_BASELINE_PATH = _REPO_ROOT / "data" / "tyre_compound_baseline.json"


# ──────────────────────────────────────────────────────────────────────────────
# Calibration tables
# ──────────────────────────────────────────────────────────────────────────────

# Drive modes (hand-curated; same numbers as docs/physics-model.md).
# pace_delta_s : seconds added/subtracted from a "race"-mode base lap
# tyre_mult    : multiplier on tyre wear rate
# fuel_mult    : multiplier on fuel burn rate
MODE_TABLE: dict[str, dict[str, float]] = {
    "push": {"pace_delta_s": -0.4, "tyre_mult": 1.5, "fuel_mult": 1.20},
    "race": {"pace_delta_s": 0.0, "tyre_mult": 1.0, "fuel_mult": 1.00},
    "conserve": {"pace_delta_s": +0.6, "tyre_mult": 0.7, "fuel_mult": 0.90},
    "fuel_save": {"pace_delta_s": +0.4, "tyre_mult": 0.85, "fuel_mult": 0.70},
    "tyre_save": {"pace_delta_s": +0.5, "tyre_mult": 0.6, "fuel_mult": 0.95},
}
DEFAULT_MODE = "race"


# Track-character bucket → tyre-wear multiplier (hand-curated).
TRACK_TYRE_MULTIPLIER: dict[str, float] = {
    "power": 1.0,
    "balanced": 1.1,
    "downforce": 1.3,
    "street": 0.7,
    "weather_prone": 1.2,
}


# Per-track fuel burn (kg/lap) at race-mode pace. Hand-curated approximations;
# Person 2's calibration step refines via Kaggle later.
FUEL_BURN_PER_LAP: dict[str, float] = {
    "Austin": 1.85,
    "BrandsHatch": 1.55,
    "Budapest": 1.65,
    "Catalunya": 1.75,
    "Hockenheim": 1.70,
    "IMS": 1.75,
    "Melbourne": 1.75,
    "MexicoCity": 1.55,  # high altitude → less air → less fuel
    "Montreal": 1.85,
    "Monza": 1.95,
    "MoscowRaceway": 1.75,
    "Norisring": 1.30,  # very short lap
    "Nuerburgring": 1.85,
    "Oschersleben": 1.65,
    "Sakhir": 1.85,
    "SaoPaulo": 1.70,
    "Sepang": 1.80,
    "Shanghai": 1.85,
    "Silverstone": 1.85,
    "Sochi": 1.80,
    "Spa": 1.95,  # long full-throttle straights
    "Spielberg": 1.55,
    "Suzuka": 1.85,
    "YasMarina": 1.70,
    "Zandvoort": 1.55,
}
_DEFAULT_FUEL_BURN = 1.75

# In server/physics.py
EGO_CONSTRUCTORS = {
    "Red Bull":    {"pace_delta_s": -1.5, "degradation_factor": 0.92},  # fastest, easiest on tyres
    "Ferrari":     {"pace_delta_s": -0.8, "degradation_factor": 0.96},
    "Mercedes":    {"pace_delta_s": -0.5, "degradation_factor": 0.95},
    "McLaren":     {"pace_delta_s":  0.0, "degradation_factor": 1.00},  # reference (default)
    "Aston Martin":{"pace_delta_s": +0.5, "degradation_factor": 1.03},
    "Williams":    {"pace_delta_s": +2.5, "degradation_factor": 1.08},  # hardest mode
}

@lru_cache(maxsize=1)
def _load_tyre_baseline() -> dict:
    """Load data/tyre_compound_baseline.json. Provides defaults for compounds."""
    if _TYRE_BASELINE_PATH.exists():
        with open(_TYRE_BASELINE_PATH) as f:
            return json.load(f)
    # Fallback (also serves as the schema reference)
    return {
        "soft": {"pace_delta_s": -0.6, "wear_rate": 0.07, "wear_penalty_s_per_unit": 1.8},
        "medium": {"pace_delta_s": 0.0, "wear_rate": 0.045, "wear_penalty_s_per_unit": 1.5},
        "hard": {"pace_delta_s": +0.4, "wear_rate": 0.030, "wear_penalty_s_per_unit": 1.2},
        "inter": {"pace_delta_s": +1.5, "wear_rate": 0.060, "wear_penalty_s_per_unit": 2.0},
        "wet": {"pace_delta_s": +3.5, "wear_rate": 0.050, "wear_penalty_s_per_unit": 2.5},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — minimal lap-time model (v0)
# ──────────────────────────────────────────────────────────────────────────────


def compute_lap_time_v0(
    track: "Track",
    compound: str = "medium",
    fuel_kg: float = 0.0,
    mode: str = DEFAULT_MODE,
) -> float:
    """Baseline lap-time without tyre-wear, weather, or dirty air.

    Used during Phase 1 to get the env skeleton booting end-to-end before
    Phase 2 wires the full model. Phase 2 replaces calls to this with
    ``compute_lap_time(...)``.

    Args:
        track:    Track instance (provides base_lap_time_s)
        compound: tyre compound name (soft/medium/hard/inter/wet)
        fuel_kg:  current fuel mass in kg
        mode:     drive mode key in MODE_TABLE

    Returns:
        Lap time in seconds (deterministic — no noise in v0).
    """
    tyres = _load_tyre_baseline()
    compound_delta = tyres.get(compound, tyres.get("medium", {})).get("pace_delta_s", 0.0)
    mode_delta = MODE_TABLE.get(mode, MODE_TABLE[DEFAULT_MODE])["pace_delta_s"]
    fuel_penalty = 0.035 * max(0.0, float(fuel_kg))
    return float(track.base_lap_time_s + compound_delta + mode_delta + fuel_penalty)


# ──────────────────────────────────────────────────────────────────────────────
# Full physics model
# ──────────────────────────────────────────────────────────────────────────────


def compute_lap_time(
    track: "Track",
    compound: str,
    tyre_health: float,
    fuel_kg: float,
    drive_mode: str,
    dirty_air_factor: float,
    weather: "WeatherState",
    noise_seed: int,
) -> float:
    """Full lap-time function from docs/physics-model.md."""
    tyres = _load_tyre_baseline()
    compound_row = tyres.get(compound, tyres["medium"])
    mode_row = MODE_TABLE.get(drive_mode, MODE_TABLE[DEFAULT_MODE])
    health = min(1.0, max(0.0, float(tyre_health)))
    fuel = max(0.0, float(fuel_kg))
    dirty = min(1.0, max(0.0, float(dirty_air_factor)))

    compound_delta = float(compound_row.get("pace_delta_s", 0.0))
    deg_penalty = (1.0 - health) * float(compound_row.get("wear_penalty_s_per_unit", 1.5))
    fuel_penalty = 0.035 * fuel
    mode_delta = float(mode_row["pace_delta_s"])
    dirty_air_penalty = dirty * 0.7
    weather_penalty = compute_weather_penalty(weather, compound, track.track_character)
    noise = float(np.random.default_rng(int(noise_seed)).normal(0.0, 0.15))

    return float(
        track.base_lap_time_s
        + compound_delta
        + deg_penalty
        + fuel_penalty
        + mode_delta
        + dirty_air_penalty
        + weather_penalty
        + noise
    )


def step_tyre(
    compound: str,
    current_health: float,
    drive_mode: str,
    track_character: str,
    base_temp_c: float = 30.0,
) -> float:
    """Return tyre health after one lap."""
    tyres = _load_tyre_baseline()
    compound_row = tyres.get(compound, tyres["medium"])
    rate = float(compound_row.get("wear_rate", 0.045))
    track_mult = TRACK_TYRE_MULTIPLIER.get(track_character, 1.1)
    mode_mult = MODE_TABLE.get(drive_mode, MODE_TABLE[DEFAULT_MODE])["tyre_mult"]
    temp_mult = max(0.5, 1.0 + 0.01 * (float(base_temp_c) - 30.0))
    delta = rate * track_mult * mode_mult * temp_mult
    return max(0.0, min(1.0, float(current_health)) - delta)


def step_fuel(
    track_name: str,
    drive_mode: str,
    current_fuel_kg: float,
    noise_seed: int,
) -> float:
    """Return the new fuel level in kg after one lap."""
    mode_mult = MODE_TABLE.get(drive_mode, MODE_TABLE[DEFAULT_MODE])["fuel_mult"]
    base_burn = FUEL_BURN_PER_LAP.get(track_name, _DEFAULT_FUEL_BURN)
    noise = float(np.random.default_rng(int(noise_seed)).normal(0.0, 0.05))
    burn = max(0.0, base_burn * mode_mult + noise)
    return float(current_fuel_kg) - burn


def compute_dirty_air_factor(gap_to_car_ahead_s: float | None) -> float:
    """0.0 (clear air) to 1.0 (full DRS-train dirty air).

    Implemented in Phase 1 because it's pure arithmetic and the env skeleton
    can call it once opponent gaps come online. Returns 0.0 if gap is None
    (no car ahead).
    """
    if gap_to_car_ahead_s is None:
        return 0.0
    g = float(gap_to_car_ahead_s)
    if g > 1.5:
        return 0.0
    if g > 0.6:
        return (1.5 - g) / 0.9
    return 1.0


def compute_weather_penalty(
    weather: "WeatherState",
    compound: str,
    track_character: str,
) -> float:
    """Lap-time delta from weather/compound mismatch."""
    rain = min(1.0, max(0.0, float(getattr(weather, "rain_intensity", 0.0))))
    compound = compound.lower()

    if compound in ("inter", "wet"):
        if rain < 0.1:
            penalty = 8.0
        elif rain < 0.3:
            penalty = 1.5 if compound == "inter" else 3.0
        elif rain < 0.6:
            penalty = -2.0 if compound == "inter" else 1.0
        else:
            penalty = 0.0 if compound == "wet" else 3.0
    else:
        if rain < 0.1:
            penalty = 0.0
        elif rain < 0.3:
            penalty = 1.0
        elif rain < 0.6:
            penalty = 5.0
        else:
            penalty = 12.0

    if track_character == "weather_prone" and rain >= 0.3:
        penalty *= 1.1
    return float(penalty)
