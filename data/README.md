# data/

Track geometry and calibration constants. Files here are committed so the local
environment and HF Space can boot without runtime downloads.

## Contents

```text
data/
├── README.md
├── track_metadata.json
├── tracks/                              # one CSV per track
│   ├── README.md
│   ├── Monza.csv
│   ├── Spa.csv
│   └── ...
├── opponent_pace_calibration.json       # generated from archive.zip
└── tyre_compound_baseline.json          # generated tyre pace/wear constants
```

## Generation

The calibration JSON files are generated, not hand-edited:

```bash
python scripts/calibrate_opponent_pace.py --output data/opponent_pace_calibration.json
python scripts/calibrate_tyre_baseline.py --output data/tyre_compound_baseline.json
```

Track CSVs are extracted from the racetrack database archive:

```bash
python scripts/extract_track_csvs.py --source racetrack-database-master
```

## Provenance

- Kaggle Formula 1 World Championship dataset: source for opponent pace and
  pit-loss calibration.
- TUMFTM racetrack-database: source for `tracks/*.csv`.
- OpenF1/FastF1 references: used for vocabulary and validation only; not runtime
  dependencies.

## Runtime Notes

`server/track.py` loads per-track metadata from `track_metadata.json`, computes
centerline length from the CSV geometry, detects corners from curvature, and
applies calibrated base lap times when `opponent_pace_calibration.json` exists.

See `data/tracks/README.md` for the CSV schema, supported circuits, validation
commands, and the Monaco/BrandsHatch substitution note.
