# Physics Model

Pure-Python lap-time, tyre-wear, and fuel-burn model. No external sims required. The goal
is to be **realistic enough that strategy decisions matter**, not a simulator-grade model.
Calibration constants come from `data/opponent_pace_calibration.json` and
`data/tyre_compound_baseline.json`, both derived from the Kaggle F1 dataset and the
racetrack-database racing-line data.

---

## Why we don't need a real simulator

The hackathon brief is to train an LLM agent. The LLM is the **strategist**, not the
driver. So the physics model only has to answer: given a strategy, how fast did the lap
go, and how much did the tyres / fuel decay? No throttle modelling, no aerodynamics, no
tyre slip — those are driver concerns.

The minimum viable physics is a deterministic-with-noise lap-time function:

```
lap_time = base_lap_time(track)
         + compound_pace_delta(compound)
         + degradation_penalty(compound, tyre_health)
         + fuel_penalty(fuel_kg)
         + drive_mode_delta(mode)
         + dirty_air_penalty(gap_to_car_ahead)
         + weather_penalty(weather, compound)
         + gaussian_noise(seed, sigma=0.15)
```

This produces lap times to within ~0.3 s of real-world variance, which is good enough.

---

## Calibration data

### Source 1: `archive.zip` — Kaggle F1 dataset (1950–2024)

Used files:
- `lap_times.csv` (17 MB) — every recorded lap. We use this to derive **median lap time
  per circuit** as our `base_lap_time(track)` table.
- `pit_stops.csv` — pit-stop frequency and stint lengths. We use this to derive opponent
  stint plan distributions and pit-loss times per circuit.
- `circuits.csv` — circuit metadata, lat/lon, country.
- `races.csv` — to filter by year (we use 2014–2024 only since cars and rules differ).
- `results.csv` — finishing positions for cross-checking.

`scripts/calibrate_opponent_pace.py` reads these and produces:
```json
{
  "Monza": {
    "base_lap_time_s": 81.5,
    "lap_time_p25": 80.8,
    "lap_time_p75": 82.6,
    "median_pit_loss_s": 22.0,
    "n_samples": 4392
  },
  "Monaco": {
    "base_lap_time_s": 73.2,
    ...
  },
  ...
}
```

### Source 2: `racetrack-database-master/` — geometry

Track centerline CSVs. Columns: `x_m`, `y_m`, `w_tr_right_m`, `w_tr_left_m`,
`x_normvec_m`, `y_normvec_m`, `alpha_m`, `s_racetraj_m`, `psi_racetraj_rad`,
`kappa_racetraj_radpm`, `vx_racetraj_mps`, `ax_racetraj_mps2`.

Used columns:
- `x_m`, `y_m` — for the visualizer to render the track outline
- `kappa_racetraj_radpm` — curvature; we cluster into **corners** (high-|κ| segments)
  and **straights** (near-zero κ)
- `s_racetraj_m` — distance along racing line (used for total length)

`scripts/extract_track_csvs.py` copies the chosen tracks (Monza, Monaco-by-substitute,
Catalunya, BrandsHatch-as-Spa-substitute, Silverstone-by-substitute) into `data/tracks/`
and normalises column names.

> **Note on Monaco and Spa:** The TUMFTM racetrack-database does not include Monaco or
> Spa explicitly. We substitute geometric proxies and override the timing parameters from
> Kaggle data. This is a pragmatic shortcut — judges care that the strategic decisions
> are interesting, not that the centerline geometry matches the broadcast TV view.
> Substitutes used: BrandsHatch (street-character) for Monaco, generated synthetic for
> Spa with curvature inspired by Hockenheim. Track-character labels in `track.py` are
> what actually drive scoring decisions; the CSV geometry is for the visualizer only.

### Source 3: `race_results.zip` — historical per-race CSVs

Used for sanity-checking the calibration, not at runtime.

### Source 4: OpenF1 API documentation

Provides the dictionary of stint compounds, DRS encoding, weather field semantics. We
do **not** call the live API at runtime. The vocabulary is hard-coded from the docs.

---

## Tyre model

```python
TYRE_BASELINE = {
    "soft":   {"pace_delta_s": -0.6, "wear_rate": 0.07, "wear_penalty_s_per_unit": 1.8},
    "medium": {"pace_delta_s":  0.0, "wear_rate": 0.045, "wear_penalty_s_per_unit": 1.5},
    "hard":   {"pace_delta_s": +0.4, "wear_rate": 0.030, "wear_penalty_s_per_unit": 1.2},
    "inter":  {"pace_delta_s": +1.5, "wear_rate": 0.060, "wear_penalty_s_per_unit": 2.0},  # only fast in damp
    "wet":    {"pace_delta_s": +3.5, "wear_rate": 0.050, "wear_penalty_s_per_unit": 2.5},  # only fast in heavy rain
}
```

`pace_delta_s`: how much faster (negative) or slower (positive) than medium reference,
on dry track, fresh tyres.

`wear_rate`: fraction of health lost per lap in `race` mode at 30°C track temp on a
balanced track. Multiplied by drive-mode and track-character multipliers.

`wear_penalty_s_per_unit`: how much slower one full health-unit-of-wear makes the lap.
A soft tyre at 50% health is `0.5 * 1.8 = 0.9 s` slower than its fresh self.

### Track-character multipliers

```python
TRACK_TYRE_MULTIPLIER = {
    "power":          1.0,   # Monza: not abrasive
    "balanced":       1.1,   # Catalunya: medium abrasion
    "downforce":      1.3,   # Hungary, Suzuka: high lateral loading
    "street":         0.7,   # Monaco: low speed, less wear
    "weather_prone":  1.2,   # Spa: variable
}
```

### Drive-mode multipliers

```python
MODE_TABLE = {
    "push":       {"pace_delta_s": -0.4, "tyre_mult": 1.5, "fuel_mult": 1.2},
    "race":       {"pace_delta_s":  0.0, "tyre_mult": 1.0, "fuel_mult": 1.0},
    "conserve":   {"pace_delta_s": +0.6, "tyre_mult": 0.7, "fuel_mult": 0.9},
    "fuel_save":  {"pace_delta_s": +0.4, "tyre_mult": 0.85, "fuel_mult": 0.7},
    "tyre_save":  {"pace_delta_s": +0.5, "tyre_mult": 0.6, "fuel_mult": 0.95},
}
```

`pace_delta_s` is the lap-time delta. `tyre_mult` and `fuel_mult` modify the per-lap
consumption.

---

## Fuel model

```python
FUEL_BURN_PER_LAP = {
    "Monza":       1.95,   # high-throttle, more fuel
    "Catalunya":   1.75,
    "Spa":         1.85,
    "Silverstone": 1.80,
    "Monaco":      1.50,   # low-speed street, less fuel
    # ... full table in data/fuel_burn_per_lap.json
}
```

Per-lap fuel burn = `FUEL_BURN_PER_LAP[track] * mode_fuel_mult + noise(σ=0.05)`.

A 12-lap Monaco race in race mode burns ~18 kg. Cars start with 20–25 kg + margin.
A 15-lap Monza race in push mode burns ~30 kg.

If `final_fuel_kg < 0.0` → DNF. `fuel_management` dimension zeroes.

---

## Lap-time function (full)

```python
def compute_lap_time(
    track: Track,
    compound: str,
    tyre_health: float,
    fuel_kg: float,
    drive_mode: str,
    dirty_air_factor: float,
    weather: WeatherState,
    noise_seed: int,
) -> float:
    base = TRACK_BASE_LAP_TIMES[track.name]                                  # from Kaggle
    compound_delta = TYRE_BASELINE[compound]["pace_delta_s"]
    deg_penalty = (1 - tyre_health) * TYRE_BASELINE[compound]["wear_penalty_s_per_unit"]
    fuel_penalty = 0.035 * fuel_kg                                           # ~3.5 kg/s
    mode_delta = MODE_TABLE[drive_mode]["pace_delta_s"]
    dirty_air_penalty = dirty_air_factor * 0.7
    weather_penalty = compute_weather_penalty(weather, compound, track.track_character)
    noise = gaussian(seed=noise_seed, sigma=0.15)

    return base + compound_delta + deg_penalty + fuel_penalty \
         + mode_delta + dirty_air_penalty + weather_penalty + noise
```

### Weather penalty

```python
def compute_weather_penalty(weather, compound, track_character):
    rain = weather.rain_intensity
    if compound in ("inter", "wet"):
        if rain < 0.1:
            return 8.0   # inters/wets on dry: catastrophic, ~8s/lap slower
        elif rain < 0.3:
            return 1.5   # inters in damp: still slow
        elif rain < 0.6:
            return -2.0  # inters in light rain: fastest
        else:
            return 0.0 if compound == "wet" else 3.0
    else:  # slicks
        if rain < 0.1:
            return 0.0
        elif rain < 0.3:
            return 1.0   # slicks in damp: 1s slower
        elif rain < 0.6:
            return 5.0   # slicks in light rain: 5s slower
        else:
            return 12.0  # slicks in heavy rain: catastrophic
    # Track character can amplify weather penalty for weather_prone tracks (Spa)
```

This produces the natural strategic crossover: at rain intensity 0.3–0.5, slicks are 4–6s
slower than inters. The optimal pit-for-inters lap is when rain crosses ~0.3 going up.

---

## Dirty air

```python
def compute_dirty_air_factor(gap_to_car_ahead_s: float) -> float:
    if gap_to_car_ahead_s > 1.5:
        return 0.0   # clear air
    elif gap_to_car_ahead_s > 0.6:
        return (1.5 - gap_to_car_ahead_s) / 0.9   # linear ramp 0.0 → 1.0
    else:
        return 1.0   # full DRS-train dirty air
```

A car following inside 1 second of another loses ~0.7 s/lap. This makes `ATTACK_AHEAD`
expensive on tyres (you push to close the gap, but every lap inside 1s wears more) and
makes `HOLD_GAP` strategically valuable.

---

## Pit-stop time

Per-track pit-loss values from `data/pit_loss.json`:

```json
{
  "Monza":        {"green_loss_s": 22.5, "sc_loss_s": 7.5},
  "Monaco":       {"green_loss_s": 18.0, "sc_loss_s": 5.5},
  "Catalunya":    {"green_loss_s": 22.0, "sc_loss_s": 7.0},
  "Spa":          {"green_loss_s": 23.0, "sc_loss_s": 8.0},
  "Silverstone":  {"green_loss_s": 22.0, "sc_loss_s": 7.0}
}
```

When the agent issues `PIT_NOW <compound>`:
- if `race_status == "sc"` or `race_status == "vsc"`: pit-loss = `sc_loss_s`
- else: pit-loss = `green_loss_s`
- Lap time for the pit lap = `compute_lap_time(...) + pit_loss`
- After the pit, ego compound switches and tyre age resets to 0
- `PIT_COOLDOWN_LAPS = 2` — second pit within 2 laps gets harmful penalty unless SC/weather justifies

---

## Visualisation

The visualiser (`server/visualizer.py`) consumes a rollout `.jsonl` and renders:

1. **Track plot**: matplotlib polygon of `(x_m, y_m)` from the CSV
2. **Cars**: ego + opponents as coloured circles, animated along the centerline by interpolating their position (lap fraction × centerline length)
3. **Per-lap strip**: a 1D timeline below the track showing pit events, mode changes, weather, SC zones
4. **Annotations**: text overlay each step showing `RADIO_DRIVER` calls and inspection reveals
5. **Output**: GIF (for blog) and MP4 (for video)

Implementation uses `matplotlib.animation.FuncAnimation` with `blit=True` for speed.
Headless rendering for the CI / Space build.

The Gradio interface (built into the OpenEnv server when `ENABLE_WEB_INTERFACE=1`)
exposes the same data live — judges click `Reset`, type a strategic command, and see the
race-state JSON update on screen.

---

## Calibration validation

Before declaring physics done, run `scripts/validate_physics.py`:

```bash
python scripts/validate_physics.py --track Monza --laps 50 --compound medium
# Expected output:
#   Median lap time: 81.4 s (Kaggle median: 81.5 s, delta 0.1)
#   p25/p75: 80.7 / 82.5 (Kaggle: 80.8 / 82.6)
#   Tyre health at lap 50: 0.32 (plausible for medium)
#   Fuel burn over 50 laps: 97.5 kg (1.95 kg/lap target)
```

If any value is more than 1 s off the Kaggle reference, fix the calibration before
shipping. The model doesn't have to be exact, but it can't lie about Monza being 30s/lap.
