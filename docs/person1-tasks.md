# Person 1 Queue — Shashwat Rajan

**Owner of:** Environment + Physics + Tracks + Opponents + Weather + Scenarios + Hidden state + Visualizer + Postmortem.

You build everything inside `server/` *except* `scoring.py` and `postmortem.py` (Person 2 wires those, but you implement them — see merge points).

You are the world-builder. The trainer (Person 2) consumes your env over HTTP and grades it via Person 2's scorer. Your contract is: the env conforms to OpenEnv 0.2.3, returns valid `F1Observation` schemas, and the rewards in those observations come from Person 2's scoring functions which YOU import and call.

This document is your day-by-day work queue. Tick the boxes as you go.

---

## Day 0 — Bootstrap (parallel with Person 2)

- [x] Clone repo, `pip install -e .` works
- [x] Read [`docs/architecture.md`](architecture.md) end-to-end
- [x] Read [`docs/physics-model.md`](physics-model.md) end-to-end
- [x] Read [`docs/scenarios.md`](scenarios.md) end-to-end
- [x] **Run `scripts/extract_track_csvs.py`** — populates `data/tracks/Monza.csv`, etc. Validate each loads in a Python REPL via `pandas.read_csv`.
- [x] Spot-check track lengths against published F1 figures. If any CSV is in km, add a unit-conversion in `track.py`.
- [x] Commit `data/tracks/*.csv` and `data/tracks/README.md` to git.

---

## Day 1 — Phase 1 (Environment Skeleton)

Goal: by end of day, `python -m server.app` starts a server and `tests/smoke_http.py` passes 5 checks.

### 1.1 Models — `models.py`

Mirror OpsTwin's `models.py`. Define three classes:

```python
from typing import Dict, List, Optional
from openenv.core.env_server import Action, Observation, State

class F1Action(Action):
    command: str

class F1Observation(Observation):
    # See docs/architecture.md §6 for the full field list.
    current_lap: int = 0
    total_laps: int = 0
    # ...

class F1State(State):
    task_name: str = ""
    scenario_family: str = ""
    track_name: str = ""
    # ...
```

- All fields default to safe empty values
- Use `Optional[float] = None` for gap fields (None = no car ahead)
- Validate that `from openenv.core.env_server import Action, Observation, State` imports correctly

### 1.2 Client — `client.py`

Direct port from OpsTwin. Replace `Ops` with `F1`:

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import F1Action, F1Observation, F1State

class F1StrategistEnv(EnvClient[F1Action, F1Observation, F1State]):
    def _step_payload(self, action): return {"command": action.command}
    def _parse_result(self, payload): ...
    def _parse_state(self, payload): ...
```

### 1.3 Server — `server/app.py`, `server/__init__.py`

Direct port:
- `server/__init__.py` exports `OpsTwinEnvironment` → rename to `F1StrategistEnvironment`
- `server/app.py` uses `create_fastapi_app(lambda: _shared_env, F1Action, F1Observation)` with a single shared singleton

### 1.4 Track loader — `server/track.py`

```python
@dataclass
class Track: ...

def load_track(name: str) -> Track:
    df = pd.read_csv(f"data/tracks/{name}.csv")
    centerline = np.column_stack([df["x_m"], df["y_m"]])
    length_m = float(df["s_racetraj_m"].max())
    corners = detect_corners(df["kappa_racetraj_radpm"], df["s_racetraj_m"])
    track_character = TRACK_CHARACTER_TABLE[name]
    pit_loss = PIT_LOSS_TABLE[name]
    drs_zones = DRS_ZONES.get(name, [])
    return Track(name, centerline, length_m, corners, pit_loss, drs_zones, track_character)
```

`detect_corners`: simple curvature threshold (`abs(kappa) > 0.005`), cluster contiguous segments, take centroid of each cluster.

### 1.5 Physics v0 — `server/physics.py`

Just the lap-time function with constant base time, no wear yet:

```python
def compute_lap_time_v0(track, compound, fuel_kg, mode):
    base = TRACK_BASE_LAP_TIMES[track.name]
    return base + MODE_TABLE[mode]["pace_delta_s"] + 0.035 * fuel_kg
```

Wear and weather are wired in Phase 2.

### 1.6 Environment skeleton — `server/environment.py`

Stub all action handlers to return `(0.0, "stub")`. Get `reset()` returning a valid `F1Observation`, `step(F1Action(command="STAY_OUT"))` returning a valid `StepResult`, and `step(F1Action(command="DONE"))` flipping `done=True`.

```python
class F1StrategistEnvironment(Environment[F1Action, F1Observation, F1State]):
    SUPPORTS_CONCURRENT_SESSIONS = True
    LAPS_PER_STEP = 1
    PIT_COOLDOWN_LAPS = 2

    def __init__(self):
        # ... initialise default scenario
        self._track = None
        self._lap = 0
        self._done = False
        self._audit_trail = []
        self._fired_events = set()
        self._dynamic_events = []
        self._pending_rewards = {}
        self._scenario = None
        self._reset_internal()

    def reset(self, options=None) -> F1Observation:
        # Pick scenario from options, load it, return _obs()
        ...

    def step(self, action: F1Action) -> tuple[F1Observation, float, bool]:
        # 1. _exec(action) → (immediate_reward, message)
        # 2. advance_lap()
        # 3. finalize pending rewards
        # 4. return _obs(), reward, _done
        ...

    def _exec(self, action) -> tuple[float, str]: ...
    def _obs(self) -> F1Observation: ...
    def _load(self, scenario_dict): ...
    def state(self) -> F1State: ...
```

### 1.7 Smoke tests — `tests/`

- [x] `tests/test_environment.py` — pytest, asserts `reset()` returns valid `F1Observation`, `step` works
- [x] `tests/smoke_http.py` — boots a server in a subprocess, hits `/health`, `/reset`, `/step`, asserts 200s

**Phase 1 exit criteria:** Both smoke tests pass. Server boots cleanly. One full episode runs to `done=True`.

Commit and push so Person 2 can start writing `inference.py` against your env.

---

## Day 2 morning — Phase 2 (Physics + Scenarios)

Goal: physics is real, three scenarios are loadable, reward signal is discriminative.

### 2.1 Physics v1 — `server/physics.py`

Implement the full lap-time function from `physics-model.md` §lap-time-function-(full).
Add tyre wear:

```python
def step_tyre(compound, current_health, drive_mode, track_character, base_temp):
    rate = TYRE_BASELINE[compound]["wear_rate"]
    delta = rate * TRACK_TYRE_MULTIPLIER[track_character] * MODE_TABLE[drive_mode]["tyre_mult"] * (1 + 0.01 * (base_temp - 30))
    return max(0.0, current_health - delta)
```

Add fuel burn:
```python
def step_fuel(track_name, drive_mode, current_fuel):
    burn = FUEL_BURN_PER_LAP[track_name] * MODE_TABLE[drive_mode]["fuel_mult"] + gauss(0, 0.05)
    return current_fuel - burn
```

### 2.2 Opponents — `server/opponents.py`

```python
@dataclass
class Stint: compound: str; planned_end_lap: int

@dataclass
class Opponent:
    driver_number: int; team: str; pace_offset_s: float; aggression: float
    planned_strategy: list[Stint]
    current_compound: str; current_tyre_age: int
    current_position: int; fuel_remaining_kg: float
    cumulative_time_s: float = 0.0    # for ranking

def step_opponents(opponents, track, weather, sc_active, lap):
    for opp in opponents:
        # 1. compute their lap time (same physics as ego, with their pace_offset)
        lap_time = compute_lap_time(track, opp.current_compound, ...) + opp.pace_offset_s
        # 2. apply pit if planned for this lap
        if opp.planned_strategy and opp.planned_strategy[0].planned_end_lap == lap:
            stint = opp.planned_strategy.pop(0)
            lap_time += track.pit_lane_loss_s if not sc_active else track.sc_pit_loss_s
            opp.current_compound = opp.planned_strategy[0].compound if opp.planned_strategy else opp.current_compound
            opp.current_tyre_age = 0
        # 3. update wear, fuel, time
        opp.cumulative_time_s += lap_time
        opp.current_tyre_age += 1
    # 4. recompute positions by cumulative_time
    ranked = sorted(opponents + [ego], key=lambda c: c.cumulative_time_s)
    for i, c in enumerate(ranked, 1): c.current_position = i
```

### 2.3 Weather — `server/weather.py`

```python
@dataclass
class WeatherState:
    air_temp_c: float; track_temp_c: float
    rain_intensity: float; surface_state: str

@dataclass
class WeatherSchedule:
    per_lap: list[WeatherState]
    forecast_uncertainty: float

    @classmethod
    def from_seed(cls, seed: int, archetype: str, n_laps: int):
        rng = np.random.default_rng(seed)
        if archetype == "clear_dry":
            per_lap = [WeatherState(...) for _ in range(n_laps)]
        elif archetype == "light_rain_window":
            # roll start lap (5–8), peak intensity (0.3–0.5), duration (3–5 laps)
            ...
        # etc

    def observe(self, current_lap, k_laps_ahead) -> list[ProbCone]:
        # Return upper/lower cone for each future lap based on forecast_uncertainty
        ...
```

### 2.4 Scenarios — `server/scenarios.py`

Three hand-authored dicts conforming to `docs/scenarios.md` §schema:

```python
DRY_STRATEGY_SPRINT = {
    "task_name": "dry_strategy_sprint_monza",
    "scenario_family": "dry_strategy_sprint",
    "track_name": "Monza",
    "total_laps": 10,
    # ...
}

WEATHER_ROULETTE = {
    "task_name": "weather_roulette_spa",
    "scenario_family": "weather_roulette",
    "track_name": "Spa",
    # ...
}

LATE_SAFETY_CAR = {
    "task_name": "late_safety_car_monaco",
    "scenario_family": "late_safety_car",
    "track_name": "Monaco",
    # ...
}

SCENARIOS = {
    "dry_strategy_sprint": DRY_STRATEGY_SPRINT,
    "weather_roulette":    WEATHER_ROULETTE,
    "late_safety_car":     LATE_SAFETY_CAR,
}
```

For each, manually trace the optimal sequence from `docs/scenarios.md` and verify it scores ≥ 0.85 once Person 2's scorer is in.

### 2.5 Hidden state — `server/hidden_state.py`

Direct port of OpsTwin's hidden_state.py. New keys per architecture.md §2.

```python
class HiddenStateLayer:
    def __init__(self, scenario_dict): ...
    def reveal(self, key, sub_key=None) -> tuple[bool, dict]:
        # Returns (is_new_revelation, info_dict)
        ...
```

### 2.6 Wire it all in `environment.py`

`_load(scenario_dict)`:
1. Load track via `load_track(scenario_dict["track_name"])`
2. Initialise `_ego_car` from `starting_*` fields
3. Initialise `_opponents` list
4. Initialise `_weather = WeatherSchedule.from_seed(seed, archetype, total_laps)`
5. Initialise `_hidden_state = HiddenStateLayer(scenario_dict)`
6. Reset `_audit_trail`, `_fired_events`, `_pending_rewards`

`_exec(action)`:
- Parse the command, route to one of ~20 handler methods
- Each handler returns `(immediate_reward, message)`
- Append to `_audit_trail`

`step(action)`:
1. `immediate_reward, message = self._exec(action)`
2. `self.advance_lap()` — runs physics, runs opponents, fires dynamic events, decrements fuel/wear
3. `delayed_reward = self._finalize_pending_rewards()`
4. `self._check_done()`
5. Build observation, attach `score = sum(immediate, delayed)`, return

**Phase 2 exit:** `python tests/smoke_all_scenarios.py` passes — all three scenarios load, expert sequences score ≥ 0.85, panic sequences score ≤ 0.40.

---

## Day 2 afternoon — Phase 4 (Postmortem) + Phase 5 (Polish)

### 4.1 Postmortem — `server/postmortem.py`

Direct port of OpsTwin's. Storage: append-only `.jsonl` at `baselines/trajectories/postmortems.jsonl`.

```python
class PostmortemMemory:
    PATH = Path("baselines/trajectories/postmortems.jsonl")

    @classmethod
    def record(cls, summary: dict): ...

    @classmethod
    def retrieve(cls, scenario_family: str, k: int = 2) -> list[dict]:
        # Return top-k lowest-scoring postmortems for this family
        ...

    @classmethod
    def classify_failure(cls, audit_trail, final_scores) -> str:
        # Rules:
        # - if rain peak passed without PIT_NOW inter → "late_weather_call"
        # - if ASSESS_UNDERCUT_WINDOW never called → "missed_undercut"
        # - if final_fuel < 0 → "fuel_underburn"
        # - if 2 pits within 3 laps → "panic_pit"
        # - if same action 3+ times → "thrashing"
        # - if pending_comms unresolved → "comms_forgotten"
        # - else → "other"
        ...
```

In `environment.py`, on `_done = True`, call:
```python
PostmortemMemory.record({
    "scenario_family": self._scenario["scenario_family"],
    "failure_category": PostmortemMemory.classify_failure(...),
    "first_bad_action": self._find_first_bad_action(),
    "missed_signal": self._diagnose_missed_signal(),
    "preferred_intervention_order": self._suggest_order(),
    "final_score": self._final_scores["weighted_final"],
    "episode_id": self._episode_id,
    "timestamp": now_iso(),
})
```

In `reset()`, call `hints = PostmortemMemory.retrieve(family, k=2)` and inject as text into `obs.memory_hints` list.

### 5.1 Visualizer — `server/visualizer.py`

```python
def render_rollout(rollout_jsonl_path: Path, output_path: Path, fmt: str = "gif"):
    """
    Reads a .jsonl rollout, renders each step's race-state, animates.
    Layout:
    +-----------------------------------------+
    |          Track polygon + cars           |
    |                                         |
    +-----------------------------------------+
    | Lap timeline | weather strip | pit events|
    +-----------------------------------------+
    """
    fig, (ax_track, ax_strip) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1]})
    # ... matplotlib animation
```

### 5.2 Procedural generator — `server/generator.py`

```python
def generate(family: str, seed: int, difficulty: str = "medium") -> dict:
    rng = np.random.default_rng(seed)
    base = copy.deepcopy(SCENARIOS[family])
    # Vary: starting_position, opponent pace_offsets, weather seed,
    # SC schedule, total_laps within the family's typical range
    ...
    return base
```

Validate: `python -c "from server.generator import generate; from baselines.expert_solver import solve; assert solve(generate('weather_roulette', i)) > 0.85" for i in range(10)`.

### 5.3 Gradio web interface

Add to `Dockerfile`:
```dockerfile
ENV ENABLE_WEB_INTERFACE=1
```

Test: hit `/web` on the running Space and confirm the Gradio panel works.

---

## Merge points with Person 2

| When | What you provide | What Person 2 needs |
|------|-----------------|---------------------|
| End of Day 1 | `models.py`, `client.py`, working `server/app.py` | imports `F1Action` etc. for `inference.py` |
| Day 2 morning | `server/scoring.py` skeleton (signatures only — Person 2 fills bodies) | scorer is callable from `environment.step()` |
| Day 2 noon | All scenarios load, expert sequences exist in commit | Person 2 runs `train.py` against the env via Docker |
| Day 2 afternoon | Postmortem records and retrieves | Person 2 runs the ablation eval |

---

## Pitfalls to avoid

- **Mutating scenario templates.** Always `copy.deepcopy(SCENARIOS[name])` before mutating. We saw this bite OpsTwin in stage 2 and lost a day debugging.
- **Per-lap noise too high.** σ = 0.15 s is intentional. σ = 0.5 s makes the optimal strategy too random and the agent can't learn.
- **Opponent pace too random.** Pace offsets must be sampled at scenario load and frozen for the episode. Re-sampling per lap → same problem as above.
- **Centerline orientation.** Some racetrack-database CSVs are clockwise, some anti-clockwise. Visualiser will look wrong if you mix. Normalise direction in `track.py`.
- **Track-character mislabelling.** Monaco labelled `power` instead of `street` → Monaco scenarios become unsolvable. Test `track_character` lookups before scenarios.

---

## Definition of done (Person 1)

- [x] All Phase 1 boxes ticked
- [x] All Phase 2 boxes ticked
- [x] Postmortem implemented and tested
- [x] Visualizer produces a watchable GIF for at least one rollout
- [ ] Gradio web UI works on the live HF Space
- [x] No unresolved TODO comments in `server/`
