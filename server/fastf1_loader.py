"""FastF1 data loader — pull, filter, and feature-extract real F1 race data.

This module is the *only* place that talks to FastF1. Scripts go through it.

Conventions:
  - Cache lives in `data/fastf1_cache/` (gitignored).
  - We pull seasons 2023-2025 (current ground-effect-car era; 2022 has different
    tyre allocations and is excluded for stability).
  - Only Race ('R') sessions are pulled. Sprints ('S'), quali ('Q'), and
    practice ('FP1/FP2/FP3') are ignored — they don't reflect race-strategy
    behavior.
  - Wet/intermediate-dominated races are kept (we want weather data) but
    flagged so tire-degradation aggregation can exclude them per the Frontiers
    AI 2025 pit-stop paper recommendation.

Sources
-------
- FastF1 docs: https://docs.fastf1.dev/
- Beatson (2025) state-space tire model: arXiv:2512.00640
- Mercedes-AMG tyre energy paper: arXiv:2501.04067
- Frontiers AI Bi-LSTM pit decision paper:
  https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full

See `docs/data-sources.md` for the full provenance list.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# fastf1 emits chatty INFO-level logs by default; quiet them unless verbose.
logging.getLogger("fastf1").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent.parent / "data" / "fastf1_cache"
RAW_OUT_DIR = Path(__file__).parent.parent / "data" / "grounded" / "raw"
GROUNDED_OUT_DIR = Path(__file__).parent.parent / "data" / "grounded"

RACE_YEARS: tuple[int, ...] = (2023, 2024, 2025)

# FastF1 TrackStatus codes — combined codes (e.g. "12") mean two flags
# active simultaneously.
TRACK_STATUS_CODES: dict[str, str] = {
    "1": "green",
    "2": "yellow",
    "3": "sc_ending",
    "4": "safety_car",
    "5": "red_flag",
    "6": "vsc_deployed",
    "7": "vsc_ending",
}

# Compound mapping FastF1 → our env naming
COMPOUND_MAP: dict[str, str] = {
    "SOFT": "soft",
    "MEDIUM": "medium",
    "HARD": "hard",
    "INTERMEDIATE": "inter",
    "WET": "wet",
}

# Tracks we care about (subset of the F1 calendar). Maps env scenario family
# names to FastF1 GP names. Add more as we expand scenario coverage.
TRACK_TO_GP: dict[str, str] = {
    "monaco": "Monaco",
    "silverstone": "British",     # FastF1 names this "British Grand Prix"
    "spa": "Belgian",              # "Belgian Grand Prix"
    "monza": "Italian",
    "catalunya": "Spanish",
    "suzuka": "Japanese",
    "bahrain": "Bahrain",
    "imola": "Emilia Romagna",
    "miami": "Miami",
    "austin": "United States",
    "vegas": "Las Vegas",
    "abu_dhabi": "Abu Dhabi",
    "interlagos": "Brazilian",     # "São Paulo Grand Prix"
    "zandvoort": "Dutch",
}


# ---------------------------------------------------------------------------
# Cache initialisation
# ---------------------------------------------------------------------------

def enable_cache() -> Path:
    """Idempotent — call before any FastF1 operation. Returns the cache path."""
    import fastf1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    return CACHE_DIR


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

@dataclass
class RaceMetadata:
    year: int
    round: int
    gp_name: str
    track_name: str        # short key matching TRACK_TO_GP
    total_laps: int
    n_drivers: int
    compounds_used: list[str]
    n_track_status_events: int
    weather_rows: int
    wet_race: bool         # True if INTERMEDIATE or WET tyres were used
    error: Optional[str] = None


def load_race(year: int, gp: str, *, telemetry: bool = False) -> "fastf1.core.Session":
    """Load one race session from cache (downloads if not cached).

    Args:
        year: season year (e.g. 2024).
        gp: GP name as FastF1 expects (e.g. "Monaco", "British"). See
            ``TRACK_TO_GP`` for our short-key → FastF1-name mapping.
        telemetry: pull per-tick car/position telemetry (large; ~10× cache
            size). Default False — we only need lap-level data for tyre
            degradation.

    Returns:
        Loaded ``fastf1.core.Session`` with laps, weather, track_status, and
        race_control_messages populated.
    """
    import fastf1

    enable_cache()
    session = fastf1.get_session(year, gp, "R")
    session.load(
        laps=True,
        telemetry=telemetry,
        weather=True,
        messages=True,
    )
    return session


def session_metadata(session) -> RaceMetadata:
    """Summarise a loaded session into a portable metadata record."""
    laps = session.laps
    compounds = sorted({str(c) for c in laps["Compound"].dropna().unique()})
    is_wet = any(c in {"INTERMEDIATE", "WET"} for c in compounds)
    event = session.event
    return RaceMetadata(
        year=int(event.get("EventDate").year if event.get("EventDate") is not None else 0),
        round=int(event.get("RoundNumber", 0)),
        gp_name=str(event.get("EventName", "")),
        track_name=str(event.get("Country", "")).lower().replace(" ", "_"),
        total_laps=int(laps["LapNumber"].max() or 0),
        n_drivers=int(laps["Driver"].nunique()),
        compounds_used=compounds,
        n_track_status_events=int(len(session.track_status)),
        weather_rows=int(len(session.weather_data)),
        wet_race=is_wet,
    )


# ---------------------------------------------------------------------------
# Calendar enumeration
# ---------------------------------------------------------------------------

def list_races(year: int) -> list[dict]:
    """Return one entry per race in ``year``.

    Each dict has: ``round_number``, ``gp_name``, ``country``, ``date``.
    Skips testing events. Used by ``scripts/fastf1_pull.py`` to walk the
    full calendar.
    """
    import fastf1

    enable_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    out: list[dict] = []
    for _, row in schedule.iterrows():
        out.append({
            "round_number": int(row["RoundNumber"]),
            "gp_name": str(row["EventName"]),
            "country": str(row["Country"]),
            "date": str(row["EventDate"]),
        })
    return out
