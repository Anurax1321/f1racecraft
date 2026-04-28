"""Build a frozen historical-race scenario from FastF1.

POST-TRAINING TASK — do not run during the hackathon's main training window.
Run only after the GRPO checkpoint exists, so we can compare the trained
strategist's calls against what actually happened in the real race.

Output: data/historical_replays/<event>_<year>.json with the same shape as a
scenario from server/scenarios.py — the trained agent runs on it, we score
the calls, and compare against the actual race outcome (winner's pit lap,
final positions, etc.).

Usage (offline, run once locally):
    pip install fastf1
    python scripts/build_historical_replay.py --year 2024 --event Spa --session R \\
        --output data/historical_replays/spa_2024.json

Schema:
{
  "event": "Spa",
  "year": 2024,
  "session": "R",
  "track_name": "Spa",
  "total_laps": 44,
  "starting_grid": [{"driver_number": 1, "grid_position": 1, ...}, ...],
  "weather_per_lap": [{"lap": 1, "rain_intensity": 0.0, ...}, ...],
  "safety_car_laps": [3, 4, 5],
  "actual_pit_decisions": {
      "1": [{"lap": 12, "compound_in": "medium", ...}],
      "44": [...]
  },
  "actual_final_positions": {"1": 3, "44": 1, ...}
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True, help="e.g. 2024")
    parser.add_argument("--event", required=True, help="e.g. Spa, Monaco, Monza")
    parser.add_argument("--session", default="R", help="R | Q | FP1 | FP2 | FP3")
    parser.add_argument("--output", required=True, help="Path to write the JSON")
    parser.add_argument("--cache", default=os.path.expanduser("~/.cache/fastf1"))
    args = parser.parse_args()

    try:
        import fastf1
    except ImportError:
        print("Install fastf1 first: pip install fastf1", file=sys.stderr)
        return 2

    Path(args.cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)

    print(f"Loading {args.year} {args.event} {args.session} …")
    session = fastf1.get_session(args.year, args.event, args.session)
    session.load(weather=True, laps=True, telemetry=False)

    laps = session.laps
    weather = session.weather_data
    results = session.results

    starting_grid: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        starting_grid.append({
            "driver_number": int(row.get("DriverNumber", 0)),
            "driver_code": str(row.get("Abbreviation", "")),
            "team": str(row.get("TeamName", "")),
            "grid_position": int(row.get("GridPosition", 0)),
        })

    weather_per_lap: list[dict[str, Any]] = []
    if weather is not None and not weather.empty:
        for _, w in weather.iterrows():
            weather_per_lap.append({
                "time": str(w.get("Time", "")),
                "air_temp_c": float(w.get("AirTemp", 0)),
                "track_temp_c": float(w.get("TrackTemp", 0)),
                "rain_intensity": 1.0 if w.get("Rainfall", False) else 0.0,
                "humidity": float(w.get("Humidity", 0)),
                "wind_speed": float(w.get("WindSpeed", 0)),
            })

    pit_decisions: dict[str, list[dict[str, Any]]] = {}
    safety_car_laps: list[int] = []
    if laps is not None and not laps.empty:
        for drv in laps["Driver"].unique():
            drv_laps = laps.pick_driver(drv)
            stops = []
            for _, lap in drv_laps.iterrows():
                if lap.get("PitInTime") is not None and not _isna(lap.get("PitInTime")):
                    stops.append({
                        "lap": int(lap.get("LapNumber", 0)),
                        "compound_out": str(lap.get("Compound", "")),
                        "stint": int(lap.get("Stint", 0)),
                    })
            if stops:
                pit_decisions[str(drv)] = stops
            for _, lap in drv_laps.iterrows():
                if str(lap.get("TrackStatus", "1")) in ("4", "6", "7"):
                    safety_car_laps.append(int(lap.get("LapNumber", 0)))

    final_positions: dict[str, int] = {}
    for _, row in results.iterrows():
        if row.get("Position"):
            final_positions[str(row.get("Abbreviation", ""))] = int(row["Position"])

    record = {
        "event": args.event,
        "year": args.year,
        "session": args.session,
        "track_name": args.event,
        "total_laps": int(laps["LapNumber"].max()) if laps is not None and not laps.empty else 0,
        "starting_grid": starting_grid,
        "weather_per_lap": weather_per_lap,
        "safety_car_laps": sorted(set(safety_car_laps)),
        "actual_pit_decisions": pit_decisions,
        "actual_final_positions": final_positions,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


def _isna(x: Any) -> bool:
    try:
        import pandas as pd
        return pd.isna(x)
    except Exception:
        return x is None


if __name__ == "__main__":
    sys.exit(main())
