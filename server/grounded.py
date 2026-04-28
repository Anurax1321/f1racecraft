"""Per-track grounded calibration loader.

Reads `data/grounded/<track>.json` (produced by `scripts/fastf1_aggregate.py`)
and exposes helpers that the env opts into via the
``scenario["use_grounded_calibration"]`` flag.

Design — what this module does and doesn't do:
  - DOES: load and cache `data/grounded/<track>.json`.
  - DOES: convert grounded slopes into wear-rate multipliers, normalised
    against the median across all tracks per compound.
  - DOES: surface the ``track_evolution_dominant`` flag.
  - DOES: provide the grounded ``health_curve`` for hidden-state injection.
  - DOES NOT: modify physics on its own. Physics opt-in is gated by env code
    consulting these helpers explicitly.
  - DOES NOT: enable on any existing scenario by default. Opt-in only.

Why opt-in: existing scenarios + their expert sequences are calibrated to
synthetic physics. Switching wholesale would shift expected scores; we'd
have to re-author every scenario. Opt-in lets us calibrate one scenario at
a time and verify the expert still scores ≥0.85 before locking it in.

Why per-compound normalisation: a raw `grounded_slope_s_per_lap` isn't a
``wear_rate`` multiplier — different units. Dividing by the median grounded
slope (across tracks) for that compound gives a unit-free ratio: "how much
faster/slower than average does this track wear this compound?"

References
----------
- Beatson 2025 (state-space tire model): arXiv:2512.00640
- Mercedes-AMG tyre energy paper: arXiv:2501.04067
"""

from __future__ import annotations

import json
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent
_GROUNDED_DIR = _REPO_ROOT / "data" / "grounded"

# Quality gate — only use grounded data when the regression is *useful*.
# Below these thresholds we fall back to the synthetic default (factor=1.0).
MIN_R_SQUARED = 0.05
MIN_SAMPLES = 50


@lru_cache(maxsize=1)
def _load_index() -> dict[str, dict]:
    """Load every `data/grounded/<track>.json` once."""
    out: dict[str, dict] = {}
    if not _GROUNDED_DIR.exists():
        return out
    for f in _GROUNDED_DIR.glob("*.json"):
        if f.name == "_index.json":
            continue
        try:
            d = json.loads(f.read_text())
            out[d["track"]] = d
        except Exception:
            continue
    return out


@lru_cache(maxsize=1)
def _median_slopes_per_compound() -> dict[str, float]:
    """Median of `slope_s_per_lap_effective` per compound across all tracks
    that meet the quality gate. Used to normalise per-track multipliers.
    """
    by_compound: dict[str, list[float]] = {"hard": [], "medium": [], "soft": []}
    for track_data in _load_index().values():
        for compound, fit in (track_data.get("tyres") or {}).items():
            if compound not in by_compound:
                continue
            if fit.get("r_squared", 0) < MIN_R_SQUARED:
                continue
            if fit.get("n_samples", 0) < MIN_SAMPLES:
                continue
            by_compound[compound].append(fit["slope_s_per_lap_effective"])
    medians = {}
    for c, slopes in by_compound.items():
        if slopes:
            medians[c] = statistics.median(slopes)
    return medians


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_track_data(track_short_name: str) -> Optional[dict]:
    """Return the loaded grounded JSON for a track, or None if not available."""
    return _load_index().get(track_short_name)


def get_wear_factor(track_short_name: str, compound: str) -> float:
    """Return a multiplier on the synthetic tire wear rate for this
    track + compound combination.

    Returns 1.0 (no change) when:
      - track has no grounded data
      - compound has no fit in this track's data
      - fit fails the R² or sample-size quality gate
      - compound has no median to normalise against
    """
    track_data = get_track_data(track_short_name)
    if not track_data:
        return 1.0
    fit = (track_data.get("tyres") or {}).get(compound)
    if not fit:
        return 1.0
    if fit.get("r_squared", 0) < MIN_R_SQUARED:
        return 1.0
    if fit.get("n_samples", 0) < MIN_SAMPLES:
        return 1.0
    medians = _median_slopes_per_compound()
    median = medians.get(compound)
    if not median or median <= 0:
        return 1.0
    factor = fit["slope_s_per_lap_effective"] / median
    # Track-evolution dominant: cap at 0.5 — even when raw slope is tiny,
    # tires still wear physically. Don't let the env think tires are
    # invincible.
    if fit.get("track_evolution_dominant"):
        factor = max(0.5, factor)
    # Sanity clamp: real F1 tire wear ranges within ~5x of average.
    return max(0.20, min(3.0, factor))


def get_health_curve(track_short_name: str, compound: str) -> Optional[list[float]]:
    """Return the per-stint-age health curve [1.0, 0.95, ...] from grounded
    data, or None if unavailable."""
    track_data = get_track_data(track_short_name)
    if not track_data:
        return None
    fit = (track_data.get("tyres") or {}).get(compound)
    if not fit:
        return None
    if fit.get("r_squared", 0) < MIN_R_SQUARED or fit.get("n_samples", 0) < MIN_SAMPLES:
        return None
    return list(fit.get("health_curve") or [])


def is_track_evolution_dominant(track_short_name: str, compound: str) -> bool:
    track_data = get_track_data(track_short_name)
    if not track_data:
        return False
    fit = (track_data.get("tyres") or {}).get(compound)
    return bool(fit and fit.get("track_evolution_dominant"))


def list_grounded_tracks() -> list[str]:
    return sorted(_load_index().keys())


def reset_caches() -> None:
    """Test helper — call when grounded data on disk has changed mid-run."""
    _load_index.cache_clear()
    _median_slopes_per_compound.cache_clear()
