#!/usr/bin/env python
"""Aggregate per-race extracted features into per-track grounded constants.

Reads from `data/grounded/raw/<year>/*.json`, groups by country (track), and
produces `data/grounded/<track>.json` with:

  - tyres: per-compound degradation curve (lap_time_corrected vs tyre_life)
  - pit: avg pit lane loss, typical pit windows, pit-window clustering
  - sc: SC/VSC frequency (fraction of races with SC), lap distribution
  - strategy: distribution of n_pit_stops, common compound sequences
  - weather: temp distribution, rainfall probability
  - dirty_air: position-change rate per lap (overtaking-difficulty proxy)

The fitting is deliberately simple (linear regression on filtered laps) — we
want grounded numbers, not a Bayesian state-space model. The Beatson 2025
state-space approach is earmarked for a future enhancement if our linear
curves don't fit the data well.

Usage:
    python scripts/fastf1_aggregate.py             # all tracks
    python scripts/fastf1_aggregate.py --track monaco
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.fastf1_loader import GROUNDED_OUT_DIR, RAW_OUT_DIR


# Compounds we model
DRY_COMPOUNDS = ("hard", "medium", "soft")
WET_COMPOUNDS = ("inter", "wet")

# Top teams — used to filter tire-degradation curves to consistent-pace cars.
# Inter-team pace variance dominates the regression; restricting to top
# teams isolates the actual tire signal. Same approach as Mercedes-AMG paper
# (arXiv:2501.04067) which uses Mercedes-only data.
TOP_TEAMS_2023_2024 = {"Red Bull Racing", "Mercedes", "McLaren", "Ferrari"}
TOP_TEAMS_2025 = {"McLaren", "Ferrari", "Mercedes", "Red Bull Racing"}
TOP_TEAMS = TOP_TEAMS_2023_2024 | TOP_TEAMS_2025


def _country_to_track_key(country: str) -> str:
    """Country → short track key. 'United States' → 'austin' etc."""
    c = country.lower().strip()
    return {
        "monaco": "monaco",
        "great britain": "silverstone",
        "united kingdom": "silverstone",
        "belgium": "spa",
        "italy": "monza",
        "spain": "catalunya",
        "japan": "suzuka",
        "bahrain": "bahrain",
        "saudi arabia": "jeddah",
        "australia": "melbourne",
        "azerbaijan": "baku",
        "miami": "miami",
        "emilia-romagna": "imola",
        "canada": "montreal",
        "austria": "spielberg",
        "hungary": "hungaroring",
        "netherlands": "zandvoort",
        "singapore": "singapore",
        "united states": "austin",
        "mexico": "mexico_city",
        "brazil": "interlagos",
        "qatar": "lusail",
        "united arab emirates": "abu_dhabi",
        "abu dhabi": "abu_dhabi",
        "china": "shanghai",
    }.get(c, c.replace(" ", "_"))


def _load_all_races(year_filter: int | None = None) -> list[dict]:
    out: list[dict] = []
    for year_dir in sorted(RAW_OUT_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        if year_filter and str(year_filter) != year_dir.name:
            continue
        for f in sorted(year_dir.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except Exception as exc:
                print(f"  ⚠️ couldn't load {f}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Tire degradation curve fitting
# ---------------------------------------------------------------------------

def _fit_compound_curve(samples: list[tuple[int, float]], min_samples: int = 20) -> dict | None:
    """Fit a linear lap-time-vs-tyre-life curve.

    Args:
        samples: list of (tyre_life, lap_time_corrected_s) tuples.
        min_samples: minimum data points required.

    Returns dict with: slope_s_per_lap, intercept_s, n_samples, r_squared,
    health_curve (list of health values 0..max_tyre_life). Or None if too few.
    """
    if len(samples) < min_samples:
        return None
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    slope = num / den
    intercept = my - slope * mx
    # R²
    ss_tot = sum((y - my) ** 2 for y in ys) or 1e-9
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = max(0.0, 1.0 - ss_res / ss_tot)

    # Convert linear lap-time degradation to a tyre-health curve our env
    # consumes. Health drops from 1.0 to ~0.0 over the typical stint life.
    # Define "spent" as +1.5s relative to fresh; clamp [0.05, 1.0].
    #
    # Slope floor (0.005 s/lap): some tracks (Monaco, Singapore — street
    # circuits with strong track evolution) show NEGATIVE raw slopes because
    # track grip improvement outpaces tire degradation lap-on-lap. Real F1
    # phenomenon, but it produces health curves that go UP (nonsense). Apply
    # a minimum degradation floor so the env still simulates tire wear.
    SLOPE_FLOOR = 0.005
    effective_slope = max(SLOPE_FLOOR, slope)
    track_evolution_dominant = slope < SLOPE_FLOOR

    max_age = max(xs) if xs else 30
    health_curve = []
    for age in range(0, int(max_age) + 5):
        delta_s = effective_slope * age
        health = max(0.05, min(1.0, 1.0 - delta_s / 1.5))
        health_curve.append(round(health, 3))

    return {
        "slope_s_per_lap_raw": round(slope, 4),
        "slope_s_per_lap_effective": round(effective_slope, 4),
        "intercept_s": round(intercept, 3),
        "n_samples": n,
        "r_squared": round(r2, 3),
        "max_tyre_life_observed": int(max_age),
        "track_evolution_dominant": track_evolution_dominant,
        "health_curve": health_curve,
    }


def aggregate_tyres(races: list[dict], *, top_teams_only: bool = True) -> dict:
    """Per-compound tire degradation curves with per-(driver, stint, race)
    normalization + optional top-team filter.

    Filters applied:
      1. Wet races skipped entirely.
      2. is_accurate=False, under_sc, under_vsc rows skipped (per Frontiers
         AI 2025 Bi-LSTM paper — wet/SC laps are noise).
      3. tyre_life > 1 (skips out-lap warmup).
      4. ``top_teams_only=True`` restricts to top-4 constructors. Removes
         inter-team pace variance per the Mercedes-AMG paper approach
         (arXiv:2501.04067). Disable for full-field aggregation.

    Per-stint normalization: each (race, driver, stint) gets its mean lap
    time subtracted, isolating the *slope* component of degradation from
    absolute pace differences.
    """
    by_stint: dict[tuple, list[dict]] = defaultdict(list)
    for race_idx, race in enumerate(races):
        # Per-stint weather: a "wet race" can still have plenty of dry
        # stints (e.g., Britain 2024 had laps 1-26 dry, then rain). We
        # *don't* skip the whole race here — we filter per-lap below using
        # the compound the driver was actually on. Soft/medium/hard imply
        # the driver believed conditions were dry; INTERMEDIATE / WET
        # would be filtered by the compound check.
        for r in race["lap_records"]:
            if not r["is_accurate"]:
                continue
            if r["under_sc"] or r["under_vsc"]:
                continue
            if r["compound"] not in DRY_COMPOUNDS:
                continue  # also drops INTERMEDIATE and WET stints
            if r["tyre_life"] <= 1 or r["lap_time_corrected_s"] <= 0:
                continue
            if top_teams_only and r["team"] not in TOP_TEAMS:
                continue
            key = (race_idx, r["driver"], r["stint"], r["compound"])
            by_stint[key].append(r)

    # For each stint, subtract its mean lap time → keeps only the slope vs age
    by_compound: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (race_idx, driver, stint, compound), rows in by_stint.items():
        if len(rows) < 4:
            continue  # need ≥4 laps in stint for meaningful slope
        mean_time = statistics.mean(r["lap_time_corrected_s"] for r in rows)
        for r in rows:
            relative_lap_time = r["lap_time_corrected_s"] - mean_time
            by_compound[compound].append((r["tyre_life"], relative_lap_time))

    out = {}
    for compound, samples in by_compound.items():
        fit = _fit_compound_curve(samples)
        if fit:
            out[compound] = fit
    return out


# ---------------------------------------------------------------------------
# Pit lane loss + pit windows
# ---------------------------------------------------------------------------

def aggregate_pit(races: list[dict]) -> dict:
    """Pit lane loss estimate + clustering of typical pit windows."""
    pit_windows: list[int] = []
    n_pits_per_race: list[int] = []
    pit_under_sc = 0
    total_pits = 0

    for race in races:
        race_pits = race["pit_decisions"]
        n_pits_per_race.append(len(race_pits) // race["metadata"]["n_drivers"]
                                if race["metadata"]["n_drivers"] else 0)
        for p in race_pits:
            pit_windows.append(p["lap"])
            total_pits += 1
            if p["under_sc"] or p["under_vsc"]:
                pit_under_sc += 1

    if not pit_windows:
        return {}

    # Histogram of pit lap distribution
    counts = Counter(pit_windows)
    sorted_laps = sorted(counts.items())

    return {
        "n_races_in_corpus": len(races),
        "total_pit_decisions": total_pits,
        "avg_pits_per_driver_per_race": round(statistics.mean(n_pits_per_race), 2)
                                         if n_pits_per_race else 0,
        "fraction_pits_under_sc_or_vsc": round(pit_under_sc / max(1, total_pits), 3),
        "pit_lap_p10": _percentile(pit_windows, 10),
        "pit_lap_p25": _percentile(pit_windows, 25),
        "pit_lap_p50": _percentile(pit_windows, 50),
        "pit_lap_p75": _percentile(pit_windows, 75),
        "pit_lap_p90": _percentile(pit_windows, 90),
    }


def _percentile(xs: list[int], p: int) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[k]


# ---------------------------------------------------------------------------
# SC / VSC frequency
# ---------------------------------------------------------------------------

def aggregate_sc(races: list[dict]) -> dict:
    """Fraction of races with SC/VSC events; lap distribution of when they fire."""
    n_races = len(races)
    n_with_sc = 0
    n_with_vsc = 0
    sc_lap_distribution: list[int] = []

    for race in races:
        had_sc = had_vsc = False
        for r in race["lap_records"]:
            if r["under_sc"]:
                had_sc = True
                sc_lap_distribution.append(r["lap_number"])
            if r["under_vsc"]:
                had_vsc = True
        if had_sc:
            n_with_sc += 1
        if had_vsc:
            n_with_vsc += 1

    return {
        "n_races_in_corpus": n_races,
        "fraction_with_sc": round(n_with_sc / max(1, n_races), 3),
        "fraction_with_vsc": round(n_with_vsc / max(1, n_races), 3),
        "sc_lap_p25": _percentile(sc_lap_distribution, 25) if sc_lap_distribution else 0,
        "sc_lap_p50": _percentile(sc_lap_distribution, 50) if sc_lap_distribution else 0,
        "sc_lap_p75": _percentile(sc_lap_distribution, 75) if sc_lap_distribution else 0,
    }


# ---------------------------------------------------------------------------
# Strategy distribution
# ---------------------------------------------------------------------------

def aggregate_strategy(races: list[dict]) -> dict:
    """Distribution of n_pit_stops per driver, common compound sequences."""
    pits_per_driver: list[int] = []
    sequences: Counter[tuple[str, ...]] = Counter()

    for race in races:
        if race["metadata"]["wet_race"]:
            continue
        # Group stints by driver
        by_driver: dict[str, list[dict]] = defaultdict(list)
        for stint in race["stints"]:
            by_driver[stint["driver"]].append(stint)
        for driver, stints in by_driver.items():
            stints.sort(key=lambda s: s["stint"])
            n_pits = len(stints) - 1
            pits_per_driver.append(n_pits)
            seq = tuple(s["compound"] for s in stints if s["compound"] in DRY_COMPOUNDS)
            if seq and len(seq) >= 2:
                sequences[seq] += 1

    if not pits_per_driver:
        return {}

    pit_count_dist = Counter(pits_per_driver)
    most_common_seq = sequences.most_common(5)

    return {
        "n_drivers_in_corpus": len(pits_per_driver),
        "pit_count_distribution": dict(pit_count_dist),
        "median_n_pits": int(statistics.median(pits_per_driver)),
        "most_common_compound_sequences": [
            {"sequence": list(seq), "count": cnt} for seq, cnt in most_common_seq
        ],
    }


# ---------------------------------------------------------------------------
# Weather distribution
# ---------------------------------------------------------------------------

def aggregate_weather(races: list[dict]) -> dict:
    air_temps = [r["weather_summary"].get("air_temp_c_mean")
                 for r in races if r["weather_summary"]]
    track_temps = [r["weather_summary"].get("track_temp_c_mean")
                   for r in races if r["weather_summary"]]
    air_temps = [t for t in air_temps if t is not None]
    track_temps = [t for t in track_temps if t is not None]
    n_wet = sum(1 for r in races if r["metadata"]["wet_race"])

    return {
        "n_races_in_corpus": len(races),
        "fraction_wet_races": round(n_wet / max(1, len(races)), 3),
        "air_temp_c_mean": round(statistics.mean(air_temps), 1) if air_temps else None,
        "air_temp_c_min": round(min(air_temps), 1) if air_temps else None,
        "air_temp_c_max": round(max(air_temps), 1) if air_temps else None,
        "track_temp_c_mean": round(statistics.mean(track_temps), 1) if track_temps else None,
        "track_temp_c_min": round(min(track_temps), 1) if track_temps else None,
        "track_temp_c_max": round(max(track_temps), 1) if track_temps else None,
    }


# ---------------------------------------------------------------------------
# Position change rate (overtaking difficulty proxy)
# ---------------------------------------------------------------------------

def aggregate_dirty_air(races: list[dict]) -> dict:
    """Average overtakes per race, normalised by race length."""
    rates = []
    for race in races:
        n_changes = race["position_changes"]["total_position_changes"]
        n_laps = race["metadata"]["total_laps"]
        if n_laps > 0:
            rates.append(n_changes / n_laps)
    if not rates:
        return {}
    return {
        "n_races_in_corpus": len(rates),
        "overtakes_per_lap_mean": round(statistics.mean(rates), 3),
        "overtakes_per_lap_min": round(min(rates), 3),
        "overtakes_per_lap_max": round(max(rates), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default=None)
    args = parser.parse_args()

    races = _load_all_races()
    print(f"Loaded {len(races)} races")

    # Group by track
    by_track: dict[str, list[dict]] = defaultdict(list)
    for race in races:
        track_key = _country_to_track_key(race["metadata"]["country"])
        by_track[track_key].append(race)

    GROUNDED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for track, track_races in sorted(by_track.items()):
        if args.track and args.track != track:
            continue
        agg = {
            "track": track,
            "country": track_races[0]["metadata"]["country"],
            "event_names": sorted({r["metadata"]["event_name"] for r in track_races}),
            "years_covered": sorted({r["metadata"]["year"] for r in track_races}),
            "n_races": len(track_races),
            "tyres": aggregate_tyres(track_races),
            "pit": aggregate_pit(track_races),
            "sc": aggregate_sc(track_races),
            "strategy": aggregate_strategy(track_races),
            "weather": aggregate_weather(track_races),
            "dirty_air": aggregate_dirty_air(track_races),
        }
        out_path = GROUNDED_OUT_DIR / f"{track}.json"
        out_path.write_text(json.dumps(agg, indent=1))
        compounds_modeled = list(agg["tyres"].keys()) if agg["tyres"] else []
        print(f"  {track:<18} {agg['n_races']} races · "
              f"compounds={','.join(compounds_modeled) or 'none'} · "
              f"sc={agg['sc'].get('fraction_with_sc', 0):.2f} · "
              f"wet={agg['weather'].get('fraction_wet_races', 0):.2f}")
        index.append({"track": track, "country": agg["country"],
                      "n_races": agg["n_races"], "compounds": compounds_modeled,
                      "file": f"{track}.json"})

    (GROUNDED_OUT_DIR / "_index.json").write_text(json.dumps(index, indent=1))
    print(f"\nWrote {len(index)} track files + _index.json to {GROUNDED_OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
