"""Feature extraction from cached FastF1 sessions.

This module turns a loaded ``fastf1.core.Session`` into a structured dict of
per-race features that downstream aggregation can fit curves over.

What we extract per race
------------------------
1. **lap_records** — per-lap rows for tire-degradation modeling
   (lap_time, tyre_life, compound, fuel-corrected lap time, sector times,
   track_status, is_accurate).
2. **stints** — per-driver stint summaries (compound, n_laps, start/end laps).
3. **pit_decisions** — every pit stop with timing, compound change,
   under_sc flag, estimated pit-lane loss.
4. **track_status_events** — SC/VSC/Yellow timeline with lap ranges.
5. **weather_summary** — temp distribution, rainfall stats per race.
6. **position_changes** — overtake count + distribution (overtaking-difficulty
   proxy per track).

Filtering rules (informed by Frontiers AI 2025 Bi-LSTM paper, ASU FastF1
practitioner notes):
  - Drop rows with ``IsAccurate=False`` for tire curves (in/out laps, SC laps).
  - Drop rows where ``LapTime`` is null (driver retired mid-lap).
  - Mark wet races for separate handling — wet stints have erratic tyre wear
    that shouldn't pollute dry-compound curves.

Fuel correction
---------------
Per Beatson 2025 (arXiv:2512.00640), lap times decompose as:
    lap_time = base_time + fuel_correction(fuel_kg) + tyre_pace(stint_age, compound) + noise

We don't know real per-race starting fuel, so we approximate:
    starting_fuel_kg = 110 (FIA maximum)
    burn_rate_kg_per_lap = 1.95 (typical F1 average)
    fuel_kg(lap_n) = max(0, 110 - 1.95 * lap_n)
    fuel_correction_s = fuel_kg(lap_n) * 0.030

Subtracting fuel correction from raw lap_time isolates tire+pace effects.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from server.fastf1_loader import COMPOUND_MAP, TRACK_STATUS_CODES


# Constants for fuel correction (Beatson 2025 + F1 telemetry conventions)
STARTING_FUEL_KG = 110.0
BURN_RATE_KG_PER_LAP = 1.95
FUEL_S_PER_KG = 0.030


def _fuel_kg_at_lap(lap_n: int) -> float:
    return max(0.0, STARTING_FUEL_KG - BURN_RATE_KG_PER_LAP * lap_n)


def _fuel_correction_s(lap_n: int) -> float:
    return _fuel_kg_at_lap(lap_n) * FUEL_S_PER_KG


def _td_to_seconds(td) -> Optional[float]:
    """Convert a pandas Timedelta to seconds, or None if NaT."""
    if td is None or (hasattr(td, "isna") and td.isna()):
        return None
    try:
        return float(td.total_seconds())
    except Exception:
        return None


def _decode_track_status(code: str) -> list[str]:
    """Decode a FastF1 TrackStatus combo code (e.g. '14', '127') into named events."""
    code = str(code).strip()
    if not code:
        return []
    return [TRACK_STATUS_CODES.get(c, f"unknown_{c}") for c in code]


def _is_under_sc_or_vsc(track_status: str) -> tuple[bool, bool]:
    """Return (under_sc, under_vsc) flags for a TrackStatus combo code."""
    flags = _decode_track_status(track_status)
    return ("safety_car" in flags), ("vsc_deployed" in flags)


# ---------------------------------------------------------------------------
# Per-lap extraction
# ---------------------------------------------------------------------------

def _safe_int(v, default: int = 0) -> int:
    """int() but tolerant of NaN, None, and non-numeric values."""
    if v is None:
        return default
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        return int(f)
    except (TypeError, ValueError):
        return default


def extract_lap_records(session) -> list[dict]:
    """One record per lap per driver, with fuel-corrected timing."""
    laps = session.laps
    out: list[dict] = []
    for _, row in laps.iterrows():
        lap_time_s = _td_to_seconds(row.get("LapTime"))
        if lap_time_s is None:
            continue
        lap_n = _safe_int(row.get("LapNumber"))
        if lap_n <= 0:
            continue  # skip rows without a valid lap number
        compound_raw = str(row.get("Compound") or "").upper()
        compound = COMPOUND_MAP.get(compound_raw, compound_raw.lower() or "unknown")
        track_status = str(row.get("TrackStatus") or "1")
        under_sc, under_vsc = _is_under_sc_or_vsc(track_status)
        fuel_corr = _fuel_correction_s(lap_n)
        out.append({
            "driver": str(row["Driver"]),
            "driver_number": str(row.get("DriverNumber") or ""),
            "team": str(row.get("Team") or ""),
            "lap_number": lap_n,
            "stint": _safe_int(row.get("Stint")),
            "compound": compound,
            "compound_raw": compound_raw,
            "tyre_life": _safe_int(row.get("TyreLife")),
            "fresh_tyre": bool(row.get("FreshTyre") or False),
            "lap_time_s": round(lap_time_s, 3),
            "fuel_corr_s": round(fuel_corr, 3),
            "lap_time_corrected_s": round(lap_time_s - fuel_corr, 3),
            "sector1_s": _round(_td_to_seconds(row.get("Sector1Time"))),
            "sector2_s": _round(_td_to_seconds(row.get("Sector2Time"))),
            "sector3_s": _round(_td_to_seconds(row.get("Sector3Time"))),
            "speed_i1": _round(row.get("SpeedI1")),
            "speed_i2": _round(row.get("SpeedI2")),
            "speed_fl": _round(row.get("SpeedFL")),
            "speed_st": _round(row.get("SpeedST")),
            "pit_in": row.get("PitInTime") is not None and not _is_na(row.get("PitInTime")),
            "pit_out": row.get("PitOutTime") is not None and not _is_na(row.get("PitOutTime")),
            "is_accurate": bool(row.get("IsAccurate") or False),
            "position": _safe_int(row.get("Position")),
            "track_status": track_status,
            "track_status_decoded": _decode_track_status(track_status),
            "under_sc": under_sc,
            "under_vsc": under_vsc,
        })
    return out


def _round(v, ndigits: int = 3):
    if v is None or _is_na(v):
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


def _is_na(v) -> bool:
    try:
        return bool(v != v)  # NaN check
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stints
# ---------------------------------------------------------------------------

def extract_stints(lap_records: list[dict]) -> list[dict]:
    """Aggregate per-driver stint summaries from lap records."""
    by_driver_stint: dict[tuple[str, int], list[dict]] = {}
    for r in lap_records:
        key = (r["driver"], r["stint"])
        by_driver_stint.setdefault(key, []).append(r)
    out: list[dict] = []
    for (driver, stint), rows in by_driver_stint.items():
        rows.sort(key=lambda r: r["lap_number"])
        compounds = sorted({r["compound"] for r in rows})
        out.append({
            "driver": driver,
            "team": rows[0]["team"],
            "stint": stint,
            "compound": rows[0]["compound"],
            "compounds_observed": compounds,  # should be 1
            "start_lap": rows[0]["lap_number"],
            "end_lap": rows[-1]["lap_number"],
            "n_laps": len(rows),
            "fresh_tyre_at_start": rows[0]["fresh_tyre"],
            "n_accurate_laps": sum(1 for r in rows if r["is_accurate"]),
        })
    out.sort(key=lambda s: (s["driver"], s["stint"]))
    return out


# ---------------------------------------------------------------------------
# Pit decisions
# ---------------------------------------------------------------------------

def extract_pit_decisions(lap_records: list[dict], stints: list[dict]) -> list[dict]:
    """Detect pit stops as compound/stint transitions and annotate context."""
    by_driver: dict[str, list[dict]] = {}
    for r in lap_records:
        by_driver.setdefault(r["driver"], []).append(r)

    out: list[dict] = []
    for driver, rows in by_driver.items():
        rows.sort(key=lambda r: r["lap_number"])
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            cur = rows[i]
            # A pit stop happened between prev and cur if the stint changed
            if cur["stint"] > prev["stint"]:
                out.append({
                    "driver": driver,
                    "lap": cur["lap_number"],
                    "compound_in": prev["compound"],
                    "compound_out": cur["compound"],
                    "stint_in": prev["stint"],
                    "stint_out": cur["stint"],
                    "under_sc": cur["under_sc"] or prev["under_sc"],
                    "under_vsc": cur["under_vsc"] or prev["under_vsc"],
                })
    out.sort(key=lambda p: (p["driver"], p["lap"]))
    return out


# ---------------------------------------------------------------------------
# Track status timeline
# ---------------------------------------------------------------------------

def extract_track_status_events(session) -> list[dict]:
    """List SC/VSC/Yellow events with start/end times and decoded names."""
    ts = session.track_status
    out: list[dict] = []
    for _, row in ts.iterrows():
        status = str(row.get("Status") or "")
        time_s = _td_to_seconds(row.get("Time"))
        message = str(row.get("Message") or "")
        out.append({
            "time_s": _round(time_s),
            "status": status,
            "status_decoded": _decode_track_status(status),
            "message": message,
        })
    return out


# ---------------------------------------------------------------------------
# Weather summary
# ---------------------------------------------------------------------------

def extract_weather_summary(session) -> dict:
    """Per-race weather distribution."""
    w = session.weather_data
    if len(w) == 0:
        return {}
    return {
        "n_samples": int(len(w)),
        "air_temp_c_mean": _round(w["AirTemp"].mean()),
        "air_temp_c_min": _round(w["AirTemp"].min()),
        "air_temp_c_max": _round(w["AirTemp"].max()),
        "track_temp_c_mean": _round(w["TrackTemp"].mean()),
        "track_temp_c_min": _round(w["TrackTemp"].min()),
        "track_temp_c_max": _round(w["TrackTemp"].max()),
        "humidity_mean": _round(w["Humidity"].mean()),
        "rainfall_any": bool((w["Rainfall"] > 0).any()) if "Rainfall" in w else False,
        "rainfall_frac_samples": _round((w["Rainfall"] > 0).mean()) if "Rainfall" in w else 0.0,
        "wind_speed_mean": _round(w["WindSpeed"].mean()),
    }


# ---------------------------------------------------------------------------
# Position changes (overtaking difficulty proxy)
# ---------------------------------------------------------------------------

def extract_position_changes(lap_records: list[dict]) -> dict:
    """Total overtakes across all driver pairs. Approx — counts position diffs."""
    by_driver: dict[str, list[dict]] = {}
    for r in lap_records:
        by_driver.setdefault(r["driver"], []).append(r)

    total_position_changes = 0
    for driver, rows in by_driver.items():
        rows.sort(key=lambda r: r["lap_number"])
        for i in range(1, len(rows)):
            if rows[i]["position"] != 0 and rows[i - 1]["position"] != 0:
                total_position_changes += abs(rows[i]["position"] - rows[i - 1]["position"])
    # Each overtake is counted twice (overtaker + overtaken)
    return {
        "total_position_changes": total_position_changes // 2,
    }


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_race_features(session) -> dict[str, Any]:
    """Run all extractors on one race session and return a single dict."""
    laps = session.laps
    event = session.event
    lap_records = extract_lap_records(session)
    stints = extract_stints(lap_records)
    pit_decisions = extract_pit_decisions(lap_records, stints)
    track_status_events = extract_track_status_events(session)
    weather = extract_weather_summary(session)
    pos = extract_position_changes(lap_records)

    compounds = sorted({r["compound"] for r in lap_records})
    is_wet = any(c in {"inter", "wet"} for c in compounds)

    return {
        "metadata": {
            "year": int(event.get("EventDate").year if event.get("EventDate") is not None else 0),
            "round_number": int(event.get("RoundNumber", 0)),
            "event_name": str(event.get("EventName", "")),
            "country": str(event.get("Country", "")),
            "official_event_name": str(event.get("OfficialEventName", "")),
            "session_date": str(event.get("Session5Date", "")),
            "total_laps": int(laps["LapNumber"].max() or 0),
            "n_drivers": int(laps["Driver"].nunique()),
            "compounds_used": compounds,
            "wet_race": is_wet,
        },
        "lap_records": lap_records,
        "stints": stints,
        "pit_decisions": pit_decisions,
        "track_status_events": track_status_events,
        "weather_summary": weather,
        "position_changes": pos,
    }
