# Architecture Spec

This document covers: Environment loop, Hidden State, Action reference, Observation schema,
Track and Physics interface, and Models. All components are direct transplants of the
OpsTwin Recovery primitives with domain changes.

---

## 1. Environment Loop (`server/environment.py`)

Direct port of `OpsTwinEnvironment`. Domain changes:

| OpsTwin | F1 Strategist |
|---------|---------------|
| `_services` | `_ego_car`, `_opponents` |
| `_pipelines` | `_track`, `_pit_lane` |
| `_alerts` | `_weather_alerts`, `_sc_alerts` |
| `_tickets` | `_radio_messages` |
| `_issues["service_outages"]` | `_issues["race_result"]` (positions retained/gained) |
| `_issues["ticket_escalations"]` | `_issues["tyre_management"]` (compound rules satisfied) |
| `_issues["approval_blocks"]` | `_issues["fuel_management"]` (no DNF, margin held) |
| `_issues["mandatory_rollbacks"]` | `_issues["strategic_decisions"]` (pit-window calls) |
| `_issues["pending_comms"]` | `_issues["pending_comms"]` (radio messages required) |
| `MINUTES_PER_STEP = 5` | `LAPS_PER_STEP = 1` |
| `_clock_minutes` | `_lap` |
| `GATE_COOLDOWN_STEPS` | `PIT_COOLDOWN_LAPS` |

Keep unchanged: `reset()/step()/_obs()/_exec()/_load()` signatures, `_done`, `_audit_trail`,
`_fired_events`, `_dynamic_events`, `_pending_rewards`.

The `_obs()` method returns a filtered view of state — opponent strategies are hidden
unless `CHECK_OPPONENT_STRATEGY` was called for that opponent; weather forecast is hidden
unless `REQUEST_FORECAST` was called this stint; true tyre wear is hidden unless
`INSPECT_TYRE_DEGRADATION` was called this stint.

### Step lifecycle

```
client sends F1Action(command="PIT_NOW soft")
  ↓
environment._exec(action) → (immediate_reward, message)
  ↓
environment.advance_lap() → run physics, run opponents, fire dynamic events
  ↓
environment._obs() → filtered F1Observation
  ↓
return StepResult(observation, reward, done)
```

### Episode termination

Episode terminates when any of:
- `lap >= total_laps` (race finished)
- agent issues `DONE` action
- ego car DNFs (fuel ran out, or terminal tyre failure on a 2-lap-old soft)
- step budget exceeded (`max_steps`, defaults to `total_laps + 4` to allow inspection actions to count)

On termination, `compute_multi_objective_scores()` runs and the result lands in
`observation.multi_objective_scores`. `observation.score` holds the weighted final.
Postmortem is generated and persisted.

---

## 2. Hidden State (`server/hidden_state.py`)

Replaces OpsTwin's `hidden_state.py`. **This is the signature primitive — do not cut.**

### Hidden Variables Per Episode

Each episode initialises these latent values. They are set in `_load()` but NOT exposed
in `_obs()`:

| Key | Type | What it represents |
|-----|------|---------------------|
| `true_tyre_curve` | `dict[compound, list[float]]` | actual per-lap health values; displayed dashboard health is a noisy estimate |
| `opponent_strategies` | `dict[driver_number, list[Stint]]` | each opponent's planned stint chain |
| `weather_evolution` | `list[tuple[lap, rain_intensity, surface_state]]` | pre-rolled per-lap weather state |
| `safety_car_schedule` | `list[tuple[lap, sc_type, duration]]` | pre-rolled SC events |
| `fuel_burn_actual` | `float` | per-lap kg burned, may differ from displayed if mode mismatched |
| `undercut_threshold_laps` | `int` | how many laps fresher tyres dominate at this track this scenario |

### Inspection Actions → Revelation

Each inspection action reveals one hidden variable if it hasn't been revealed yet
(or hasn't been re-revealed this stint). New revelation: `+0.02` reward (same as OpsTwin's
inspect pattern). Already-revealed: `+0.0` reward, but info is returned again.

| Action | Reveals |
|--------|---------|
| `INSPECT_TYRE_DEGRADATION` | true_tyre_curve for current compound, current lap onwards |
| `CHECK_OPPONENT_STRATEGY <driver_number>` | opponent's next planned pit lap (±1 noise) and target compound |
| `REQUEST_FORECAST` | weather_evolution for next 5 laps as probabilities |
| `ASSESS_UNDERCUT_WINDOW` | undercut_threshold_laps + current viability calc |
| `INSPECT_FUEL_MARGIN` | fuel_burn_actual vs displayed; flags if margin is dangerous |

### Delayed Reward Pattern

Some actions score fully ONLY after hidden state is confirmed. Example:

- `PIT_NOW soft` at lap 5 → immediate `+0.20` (mandatory_pit issue partially resolved)
- If `weather_evolution` shows rain at lap 6 (which the agent could have known via
  `REQUEST_FORECAST`), full reward is withheld and a `-0.10` modifier applies
- If `ASSESS_UNDERCUT_WINDOW` confirmed the undercut was viable AND the rival had longer
  stint plan, the remaining `+0.10` is released

This forces genuine investigation rather than greedy action. Implementation lives in
`environment._pending_rewards` — a dict keyed by `(action_signature, hidden_state_key)`,
finalised on each subsequent step as more state becomes visible.

---

## 3. Track and Physics Interface (`server/track.py`, `server/physics.py`)

### Track

```python
@dataclass(frozen=True)
class Corner:
    name: str
    distance_m: float        # along centerline
    radius_m: float          # tightness; lower = slower
    direction: str           # "L" / "R"

@dataclass
class Track:
    name: str
    centerline: np.ndarray   # shape (N, 2), metres in x/y
    length_m: float
    corners: list[Corner]
    pit_lane_loss_s: float   # estimated total time loss for a green-flag stop
    sc_pit_loss_s: float     # under SC the loss is ~30% of green-flag
    drs_zones: list[tuple[float, float]]  # (start_distance_m, end_distance_m)
    track_character: str     # "power" | "balanced" | "downforce" | "street" | "weather_prone"
```

`load_track(name) -> Track` reads the corresponding CSV from `data/tracks/`, runs
curvature analysis on the racing line, and labels corners by curvature buckets.

### Physics — lap time model

```python
def compute_lap_time(
    track: Track,
    compound: str,
    tyre_age_laps: int,
    tyre_health: float,        # 0.0–1.0
    fuel_kg: float,
    drive_mode: str,           # "push" | "race" | "conserve" | "fuel_save" | "tyre_save"
    dirty_air_factor: float,   # 0.0 (clear) – 1.0 (dirty)
    weather_state: WeatherState,
    noise_seed: int,
) -> float:
    base = TRACK_BASE_LAP_TIMES[track.name]              # data/track_base_times.json
    compound_delta = TYRE_BASELINE[compound]["pace_delta_s"]   # data/tyre_compound_baseline.json
    deg_penalty = (1 - tyre_health) * TYRE_BASELINE[compound]["wear_penalty_s_per_unit"]
    fuel_penalty = 0.035 * fuel_kg                       # ~3.5 kg/s rule of thumb
    mode_delta = MODE_TABLE[drive_mode]["pace_delta_s"]  # push: −0.4s, conserve: +0.6s
    dirty_air_penalty = dirty_air_factor * 0.7           # following car loses ~0.7s/lap
    weather_penalty = WEATHER_PENALTY[weather_state.compound_match(compound)]
    noise = gaussian(seed=noise_seed, sigma=0.15)
    return base + compound_delta + deg_penalty + fuel_penalty + mode_delta + dirty_air_penalty + weather_penalty + noise
```

Calibration constants live in `data/`. The full equation derivation is in
[`physics-model.md`](physics-model.md).

### Physics — tyre wear

```python
def compute_tyre_wear(compound, lap_in_stint, drive_mode, track_character, base_temp):
    base_rate = TYRE_BASELINE[compound]["wear_rate"]
    track_mult = TRACK_TYRE_MULTIPLIER[track_character]
    mode_mult = MODE_TABLE[drive_mode]["tyre_mult"]
    temp_mult = 1.0 + 0.01 * (base_temp - 30)            # warmer = more wear
    delta = base_rate * track_mult * mode_mult * temp_mult
    return delta                                          # subtract from health each lap
```

Compound legality (FIA rule: every dry race must use ≥2 distinct dry compounds) is
enforced in scoring, not physics — agent gets a `tyre_management` penalty if they
finish a dry race on one compound.

---

## 4. Opponents (`server/opponents.py`)

```python
@dataclass
class Stint:
    compound: str
    planned_end_lap: int     # rule-based, may be revised on dynamic events

@dataclass
class Opponent:
    driver_number: int
    team: str
    pace_offset_s: float          # vs ego baseline; sampled per scenario
    aggression: float             # 0.0–1.0; affects defend/attack behaviour
    planned_strategy: list[Stint]
    current_compound: str
    current_tyre_age: int
    current_position: int
    fuel_remaining_kg: float
```

`step_opponents(state, lap)` advances every opponent by one lap:
- runs physics for each
- if `lap == stint.planned_end_lap` and pit window matches, executes pit
- if SC is active, may opportunistically pit
- updates positions based on relative lap times
- writes back into `state` in place

Opponent pace offsets are sampled from `opponent_pace_calibration.json` derived from
the Kaggle dataset (1950–2024 lap-time distributions per circuit).

Opponents can be **revealed** or **hidden**:
- Position, last lap time, current compound, stint age → always visible (TV-feed level)
- Planned next pit lap, target compound for next stint → hidden, revealed by `CHECK_OPPONENT_STRATEGY`

---

## 5. Weather (`server/weather.py`)

```python
@dataclass
class WeatherState:
    air_temp_c: float
    track_temp_c: float
    rain_intensity: float    # 0.0 = dry, 0.3 = light, 0.7 = heavy
    surface_state: str       # "dry" | "damp" | "wet"

@dataclass
class WeatherSchedule:
    per_lap: list[WeatherState]   # ground truth, hidden
    forecast_uncertainty: float   # 0.0–1.0, how noisy the agent's forecast is

    @classmethod
    def from_seed(cls, seed: int, scenario: dict) -> "WeatherSchedule": ...
    def observe(self, current_lap: int, k_laps_ahead: int) -> list[ProbCone]: ...
```

Each scenario specifies a weather _archetype_:
- `clear_dry` — no rain, no SC
- `light_rain_window` — rain 0.2–0.4 between laps X and Y
- `heavy_rain_lottery` — rain ramps from 0.0 → 0.8 between laps X and Y
- `safety_car_prone` — 35% SC at laps X..Y range
- `mixed_conditions` — combination

The seed determines actual values within the archetype band.

---

## 6. Models (`models.py`)

```python
class F1Action(Action):
    """Agent sends a single text command each step."""
    command: str

class F1Observation(Observation):
    """Full F1 race state returned after reset() or step(). Filtered by what
    has been revealed via inspection actions."""

    # Race clock
    current_lap: int = 0
    total_laps: int = 0
    race_phase: str = ""              # "start" | "mid" | "end"
    race_status: str = ""             # "green" | "sc" | "vsc" | "red" | "finished"

    # Ego car
    ego_position: int = 0
    ego_tyre_compound: str = ""
    ego_tyre_age_laps: int = 0
    ego_tyre_health_pct: float = 0.0  # noisy estimate; INSPECT reveals true
    ego_fuel_remaining_kg: float = 0.0
    ego_drive_mode: str = ""
    ego_last_lap_time_s: float = 0.0
    ego_gap_to_leader_s: Optional[float] = None
    ego_gap_ahead_s: Optional[float] = None
    ego_gap_behind_s: Optional[float] = None

    # Opponents (TV-feed level visibility)
    opponents: list[dict] = []        # each: driver_number, team, position, last_lap_time_s, gap_s, current_compound, stint_age

    # Weather (forecast cone if requested, else just current)
    weather_current: dict = {}
    weather_forecast: list[dict] = [] # populated only after REQUEST_FORECAST

    # Alerts and dynamic events
    pit_window_alerts: list[dict] = []
    cascade_alerts: list[dict] = []
    uncertainty_alerts: list[dict] = []  # hidden state revealed this step
    radio_inbox: list[dict] = []         # incoming team comms

    # Issue tracking
    pending_issues_count: int = 0
    resolved_issues_count: int = 0
    total_issues_count: int = 0

    # Step-level signal
    message: str = ""
    available_commands: list[str] = []
    multi_objective_scores: dict[str, float] = {}
    score: float = 0.0
    hint: str = ""
    memory_hints: list[str] = []         # postmortem injections

class F1State(State):
    """Episode metadata. Inherits episode_id + step_count from base."""
    task_name: str = ""
    scenario_family: str = ""
    track_name: str = ""
    total_laps: int = 0
    resolved_issues: int = 0
    total_issues: int = 0
    max_steps: int = 0
    seed: int = 0
```

---

## 7. Full Action Reference

```
# Pit & Tyre
PIT_NOW <compound:soft|medium|hard|inter|wet>
STAY_OUT
RECOMMEND_PIT <next_lap:int>          # signals to driver, no immediate pit, +0.005 if matches optimum

# Drive mode
SET_MODE <push|race|conserve|fuel_save|tyre_save>
DRS_PERMISSION <on|off>

# Strategic positioning
DEFEND_POSITION                       # vs attacker behind
ATTACK_AHEAD                          # spend tyre/fuel to chase ahead
HOLD_GAP                              # maintain delta to specific car
LET_BY <driver_number>                # team orders

# Hidden-state inspection (small reward for new revelations)
INSPECT_TYRE_DEGRADATION
CHECK_OPPONENT_STRATEGY <driver_number>
REQUEST_FORECAST
ASSESS_UNDERCUT_WINDOW
INSPECT_FUEL_MARGIN

# Communications
RADIO_DRIVER <message>                # to driver — required at certain dynamic events
DRAFT_PIT_WALL <message>              # internal engineering comms
BRIEF_MEDIA <message>                 # post-race only

# Info queries
REQUEST_INFO <opponents|weather|tyres|fuel|pit_window|standings|audit>

# Control
ESCALATE_TO_PRINCIPAL                 # once per episode, strategic hint
DONE
```

### Reward Values per Action Category

| Category | Reward on correct resolution |
|----------|------------------------------|
| Race result issue (position retained / gained) | `issue["points"]` |
| Tyre management issue (compound rule satisfied) | `issue["points"]` |
| Fuel management issue (target margin held) | `issue["points"]` |
| Strategic-decision issue (pit-window correct) | `issue["points"]` |
| Pending comms sent | `issue["points"]` |
| Hidden state revealed (new) | `+0.02` |
| Drive-mode match phase | `+0.005` |
| Invalid command | `-0.02` |
| Unknown command | `-0.02` |
| Harmful action (panic pit, unnecessary defend > 4 s) | `-0.05` |
| Efficiency bonus (finish on minimum mandatory pits) | `min(0.10, ...)` |

All `issue["points"]` values within a scenario sum to 1.0 across the six categories.

---

## 8. Scoring Glue (`server/scoring.py` interface)

The scorer is invoked once at episode end (and queryable any time via `REQUEST_INFO scoring`)
and produces a dict matching:

```python
{
  "race_result":          float in [0,1],
  "tyre_management":      float in [0,1],
  "fuel_management":      float in [0,1],
  "strategic_decisions":  float in [0,1],
  "comms_quality":        float in [0,1],
  "operational_efficiency": float in [0,1],
  "weighted_final":       float in [0.01, 0.99],
}
```

Detailed formulas in [`reward-model.md`](reward-model.md).
