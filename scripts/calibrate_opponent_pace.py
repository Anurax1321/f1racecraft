"""Calibrate per-track lap-time distributions from the Kaggle F1 archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CIRCUIT_TO_TRACK = {
    "monza": "Monza",
    "monaco": "Monaco",
    "spa": "Spa",
    "catalunya": "Catalunya",
    "silverstone": "Silverstone",
    "sakhir": "Sakhir",
    "bahrain": "Sakhir",
    "red bull ring": "Spielberg",
    "spielberg": "Spielberg",
    "suzuka": "Suzuka",
    "hungaroring": "Budapest",
    "budapest": "Budapest",
    "interlagos": "SaoPaulo",
    "sao paulo": "SaoPaulo",
    "americas": "Austin",
    "austin": "Austin",
    "yas marina": "YasMarina",
    "zandvoort": "Zandvoort",
    "shanghai": "Shanghai",
    "montreal": "Montreal",
    "villeneuve": "Montreal",
    "mexico": "MexicoCity",
    "melbourne": "Melbourne",
    "albert park": "Melbourne",
    "sepang": "Sepang",
    "sochi": "Sochi",
    "hockenheim": "Hockenheim",
    "nurburgring": "Nuerburgring",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="archive")
    parser.add_argument("--output", default="data/opponent_pace_calibration.json")
    parser.add_argument("--year-min", type=int, default=2014)
    parser.add_argument("--year-max", type=int, default=2024)
    args = parser.parse_args()
    src = Path(args.source)
    required = ["circuits.csv", "races.csv", "lap_times.csv", "pit_stops.csv"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise SystemExit(f"Missing archive files under {src}: {missing}")

    circuits = pd.read_csv(src / "circuits.csv")
    races = pd.read_csv(src / "races.csv")
    laps = pd.read_csv(src / "lap_times.csv")
    pits = pd.read_csv(src / "pit_stops.csv")

    races = races[(races["year"] >= args.year_min) & (races["year"] <= args.year_max)]
    races = races[["raceId", "circuitId", "year"]].merge(
        circuits[["circuitId", "name"]], on="circuitId"
    )
    laps = laps[laps["raceId"].isin(set(races["raceId"]))]
    laps = laps.merge(races[["raceId", "name"]], on="raceId")
    laps["lap_time_s"] = laps["milliseconds"] / 1000.0
    laps = laps[(laps["lap_time_s"] > 45.0) & (laps["lap_time_s"] < 180.0)]
    pits = pits[pits["raceId"].isin(set(races["raceId"]))].merge(
        races[["raceId", "name"]], on="raceId"
    )
    pits["pit_loss_s"] = pits["milliseconds"] / 1000.0

    out = {}
    for circuit_name, group in laps.groupby("name"):
        track = _map_track(circuit_name)
        if not track:
            continue
        pit_group = pits[pits["name"] == circuit_name]
        out[track] = {
            "base_lap_time_s": round(float(group["lap_time_s"].median()), 3),
            "lap_time_p25": round(float(group["lap_time_s"].quantile(0.25)), 3),
            "lap_time_p75": round(float(group["lap_time_s"].quantile(0.75)), 3),
            "median_pit_loss_s": round(float(pit_group["pit_loss_s"].median()), 3)
            if not pit_group.empty
            else 22.0,
            "n_samples": int(len(group)),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(sorted(out.items())), indent=2), encoding="utf-8")
    print(f"wrote {output} with {len(out)} tracks")


def _map_track(circuit_name: str) -> str | None:
    low = circuit_name.lower()
    for needle, track in CIRCUIT_TO_TRACK.items():
        if needle in low:
            return track
    return None


if __name__ == "__main__":
    main()
