# data/tracks/

This folder contains one lightweight CSV centerline per track used by the F1 Strategist simulator.

## Supported CSV schemas

`server/track.py` supports both of these formats:

```csv
# x_m,y_m,w_tr_right_m,w_tr_left_m
-0.320123,1.087714,5.739,5.932
```

and simpler hand-added centerlines:

```csv
# x_m,y_m
-2.794502,3.849079
```

When width columns are missing or blank, the loader fills `w_tr_right_m` and `w_tr_left_m` with a safe default of `6.0` metres. This is intentional so manually added tracks like Monaco work in the Day-0 simulator without needing full racing-line data.

## What the simulator derives

The CSVs do **not** need precomputed curvature or velocity columns. The loader computes:

- closed-loop track length,
- cumulative distance around the lap,
- smoothed curvature,
- approximate corner clusters,
- width arrays for future visualisation/track-limit work.

The strategy simulator uses metadata from `data/track_metadata.json` for base lap time, pit-lane loss, safety-car pit loss, track character, and F1/non-F1 flags.

## Current coverage

The current project folder contains 26 tracks, including more than the requested 10-track minimum:

- Austin
- BrandsHatch
- Budapest
- Catalunya
- Hockenheim
- IMS
- Melbourne
- MexicoCity
- Monaco
- Montreal
- Monza
- MoscowRaceway
- Norisring
- Nuerburgring
- Oschersleben
- Sakhir
- SaoPaulo
- Sepang
- Shanghai
- Silverstone
- Sochi
- Spa
- Spielberg
- Suzuka
- YasMarina
- Zandvoort

`server/generator.py` can procedurally generate variants from the hand-authored families; Person 2 will use those generated scenarios for GRPO/SFT coverage across many tracks.

## Quick validation

From the repo root after installing dependencies:

```powershell
python - <<'PY'
from server.track import list_available_tracks, load_track
for name in list_available_tracks():
    t = load_track(name)
    print(f"{name:<16} length={t.length_m:7.1f}m corners={len(t.corners):2d} character={t.track_character}")
PY
```
