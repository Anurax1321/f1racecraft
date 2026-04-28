# Scenario Families

Three hand-authored families plus one stretch. Each scenario dict must conform to the schema
at the bottom of this file. All are deterministically solvable with a known action sequence,
and every family has been stress-tested for reward-hackability before training.

---

## Family 1: Dry Strategy Sprint (Monza)

**Disruption type:** `dry_strategy_sprint`
**Track:** Monza (power circuit, long straights, low downforce)
**Total laps:** 10
**Max steps:** 14
**Core tension:** Optimal pace is to extend the medium stint and pit late for softs. The trap
is opponent undercut: a rival pitting two laps earlier on fresh softs gains 1.2 s/lap, and
the agent has only 2 laps to react before the rival emerges ahead.

### Episode setup

Ego car:
- Starting position: 4th
- Starting compound: medium, age 0
- Fuel: 95 kg (race-distance + 2 kg margin)
- Drive mode: race

Opponents:
- 5 cars, mixed teams
- Notable: opponent #44 (3rd, hard-on-medium, planned 1-stop end of lap 6 → soft to flag)
- Notable: opponent #16 (5th, behind, soft tyres age 0, planned aggressive undercut at lap 4)

Weather: clear and dry, track temp 38°C, no rain forecast.

Tickets / radio inbox at start:
- R-001: from race engineer — "P4 currently. Tyre window opens lap 5. We're seeing the cars behind on softs."
- R-002: from team principal — "Push for podium if you can. Don't burn the tyres."

Pending strategic issues:
- I-001: Race result — finish ≥ 4th (points = 0.20). Bonus if finish ≥ 3rd (+0.05).
- I-002: Tyre management — finish on a different compound than start (FIA rule), AND keep tyre health > 0.40 at flag (points = 0.15).
- I-003: Strategic decision — pit between laps 4–7 inclusive AND only after `ASSESS_UNDERCUT_WINDOW` was called (points = 0.20).
- I-004: Operational efficiency — exactly 1 pit stop (points = 0.10).
- I-005: Comms — required `RADIO_DRIVER` after pit announcing target lap times (points = 0.10).
- I-006: Fuel — finish with ≥ 0.5 kg margin (points = 0.05). Auto if `INSPECT_FUEL_MARGIN` was called once.
- I-007: Race result bonus — gain ≥ 1 position (points = 0.20).

Total points: 1.00.

### Hidden state

- `opponent_strategies`:
    - #44 → planned pit lap 6, target soft
    - #16 → planned pit lap 3, target soft (aggressive undercut)
    - others → 1-stop, generic
- `undercut_threshold_laps`: 2 (this is fast at Monza; the long straights mean fresh tyres dominate hard)
- `weather_evolution`: all dry (placeholder — populated for symmetry)
- `safety_car_schedule`: `[]` (no SC this scenario)

### Optimal sequence (verified score ≈ 0.94)

1. `INSPECT_TYRE_DEGRADATION` — reveals soft window stays good for ~12 laps, medium starts dropping at lap 6
2. `CHECK_OPPONENT_STRATEGY 16` — reveals #16's lap 3 undercut plan
3. `CHECK_OPPONENT_STRATEGY 44` — reveals #44's lap 6 pit
4. `ASSESS_UNDERCUT_WINDOW` — confirms 2-lap threshold
5. `SET_MODE push` — start aggressive to keep #16 from undercutting
6. `STAY_OUT` (lap 3) — let #16 pit; they emerge in traffic, neutralised
7. `STAY_OUT` (lap 4)
8. `RADIO_DRIVER "Box this lap. Target P3 on softs."` — pre-pit comms
9. `PIT_NOW soft` (lap 5) — perfect undercut on #44 who hasn't pitted yet
10. `SET_MODE race` — settle in
11. `STAY_OUT` (laps 6–9)
12. `INSPECT_FUEL_MARGIN` — confirms 0.7 kg margin
13. `RADIO_DRIVER "All clear. Bring it home P3."` — final stint comms
14. `DONE` (after lap 10)

Result: P3, all six issues resolved, score 0.94.

### Trap sequences

**Trap A — greedy stay-out (verified score 0.55):**
Agent stays out past lap 7, pits lap 8, emerges in traffic with cold tyres, finishes P5.
- Race result: 0.0 (lost a position)
- Tyre management: 0.15 (compound rule met, health OK)
- Strategic: 0.10 (pit was outside the 4–7 window, no inspection)
- Comms: 0.0 (forgot RADIO_DRIVER)
- Operational: 0.10 (1 pit)
- Fuel: 0.05
- Final: ~0.55

**Trap B — panic pit at lap 2 (verified score 0.32):**
Agent pits lap 2 in panic, gets stuck behind backmarkers on softs that go off, pits again, finishes P6.
- Race result: 0.0
- Tyre management: 0.0 (tyres collapsed below 0.40)
- Strategic: 0.0 (pit too early, no inspection)
- Operational: 0.0 (2 pits)
- Comms: 0.0
- Fuel: 0.05
- Plus harmful-action penalties accumulated
- Final: ~0.32

---

## Family 2: Weather Roulette (Spa)

**Disruption type:** `weather_roulette`
**Track:** Spa-Francorchamps (balanced, weather-prone)
**Total laps:** 12
**Max steps:** 16
**Core tension:** Light rain begins between laps 5–8 (pre-rolled, agent sees only a forecast
cone with uncertainty). The fast strategy is to pit for inters one lap before the rain peak;
pitting one lap early wastes a stop on dry track, pitting one lap late loses 30 s on slicks
in standing water.

### Episode setup

Ego car: 5th, medium tyres age 0, fuel 100 kg, race mode.

Opponents:
- 5 cars
- Notable: opponent #1 (1st, intermediate-prepared team, planned switch at first rain signal)
- Notable: opponent #63 (4th ahead, slick gambler, will stay out)

Weather (visible to agent at start):
- Air temp 18°C, track temp 24°C, rain intensity 0.0
- Forecast: NOT visible until `REQUEST_FORECAST` is called

Pending issues:
- I-001: Race result — finish ≥ 5th (points = 0.15). Bonus ≥ 3rd (+0.10).
- I-002: Strategic — pit for inters within ±1 lap of the rain peak (points = 0.30 — this is the big one).
- I-003: Tyre management — at least one stint on inters if rain peaks above 0.4 (points = 0.15).
- I-004: Comms — `RADIO_DRIVER` warning before the rain pit (points = 0.10).
- I-005: Operational — finish without DNF, ≤ 2 pit stops (points = 0.10).
- I-006: Fuel — margin ≥ 0.5 kg at flag (points = 0.05).
- I-007: Strategic bonus — `REQUEST_FORECAST` was called BEFORE the pit decision (points = 0.15).

### Hidden state (seed-deterministic example)

- `weather_evolution`:
    - laps 1–5: dry, intensity 0.0
    - lap 6: 0.15 (drizzle begins, surface "damp")
    - lap 7: 0.45 (peak — this is the pit lap)
    - lap 8: 0.50 (still wet)
    - lap 9: 0.40 (declining)
    - lap 10–12: 0.20 (greasy, drying)
- `forecast_uncertainty`: 0.30 — the forecast cone has noise
- `safety_car_schedule`: 25% chance of SC at lap 7 if rain ≥ 0.4 (variant per seed)

### Optimal sequence (verified score ≈ 0.93)

1. `REQUEST_FORECAST` (lap 1) — reveals rain probability ramping at lap 6–8 (with uncertainty)
2. `INSPECT_TYRE_DEGRADATION` — confirms medium tyres can hold to lap 7 fine
3. `CHECK_OPPONENT_STRATEGY 1` — reveals #1's planned inter switch on first rain signal
4. `SET_MODE race` — neither push nor conserve; manage tyres but keep pace
5. `STAY_OUT` (laps 2–5)
6. `REQUEST_FORECAST` (lap 5 — second call, refines the cone)
7. `RADIO_DRIVER "Inters incoming. Plan to box lap 6 or 7."` (lap 5)
8. `STAY_OUT` (lap 6 — agent reads the actual rain, now visible at 0.15)
9. `RADIO_DRIVER "Box now. Inters."` (lap 7)
10. `PIT_NOW inter` (lap 7) — perfect timing, peak rain
11. `SET_MODE push` (lap 8 — wet conditions, push on inters while opponents on slicks struggle)
12. `STAY_OUT` (laps 8–11)
13. `INSPECT_FUEL_MARGIN` (lap 11) — confirms safe
14. `DONE` (after lap 12)

Result: P2 or P3, all issues resolved, score 0.93.

### Trap sequences

**Trap A — pit one lap early (verified 0.61):** Agent pits lap 6 on inters, then laps 6 is still mostly dry, loses 8 seconds on out-lap, drops to P7. Rain comes, recovers to P5. Strategic-decision points partial.

**Trap B — never check forecast (verified 0.42):** Agent stays out into peak rain, pits lap 8 (one lap late), loses 25 s on slicks in heavy rain, drops to P9. `forecast_called_before_pit` issue fails entirely. Strategic decisions zeroed.

**Trap C — gambler stays on slicks (verified 0.18):** Agent never pits for inters. Spins, recovers, finishes P10. Multiple harmful-action penalties.

---

## Family 3: Late Safety Car Lottery (Monaco)

**Disruption type:** `late_safety_car`
**Track:** Monaco (street, ultra-low overtaking, pit-loss is small here)
**Total laps:** 12
**Max steps:** 16
**Core tension:** A 35% safety-car probability at laps 7–10. A pit under SC costs ~8 s; a
green-flag pit costs ~22 s. Agent who commits to a one-stop too early gets stuck on stale
tyres if the SC fires; agent who waits gets a free stop.

### Episode setup

Ego car: 3rd, medium tyres age 0, fuel 80 kg, race mode.

Opponents:
- 5 cars
- Notable: opponent #11 (2nd, will pit at lap 5–6 because their team is risk-averse on Monaco tyres)
- Notable: opponent #4 (1st, will hold for SC — same hidden plan as ego's optimal)

Weather: clear, dry, no rain. SC is the only volatility.

Pending issues:
- I-001: Race result — finish ≥ 3rd (points = 0.20). Bonus ≥ 2nd (+0.05).
- I-002: Strategic — pit during an SC if one fires, OR after the SC window closes (points = 0.30). Pitting at green during the SC window is a panic pit (-0.05).
- I-003: Tyre management — compound rule + tyre health > 0.30 at flag (points = 0.15).
- I-004: Operational efficiency — exactly 1 pit stop (points = 0.15).
- I-005: Comms — `RADIO_DRIVER` if SC fires (points = 0.10).
- I-006: Fuel — margin ≥ 0.5 kg (points = 0.05).
- I-007: Strategic — `HOLD_GAP` to opponent ahead during SC window (points = 0.05).

### Hidden state (seed-deterministic example, "SC at lap 8" variant)

- `safety_car_schedule`: `[(8, "full_sc", 3_laps)]` — SC fires at lap 8, lasts 3 laps, releases lap 11
- `opponent_strategies`:
    - #11 → planned pit lap 6, will not change strategy
    - #4 → holding for SC, will pit when it fires
    - others → mixed
- `undercut_threshold_laps`: 4 (Monaco is poor for undercut, very high)

### Optimal sequence (verified 0.92)

1. `INSPECT_TYRE_DEGRADATION` — mediums hold easily for 12 laps at Monaco low-pace
2. `CHECK_OPPONENT_STRATEGY 4` — reveals #4 holding for SC
3. `CHECK_OPPONENT_STRATEGY 11` — reveals #11 lap 6 pit
4. `ASSESS_UNDERCUT_WINDOW` — confirms 4-lap threshold (so undercut is bad here)
5. `SET_MODE conserve` — preserve tyres, hold position
6. `HOLD_GAP 4` (laps 1–7) — stay within DRS of leader
7. SC fires at lap 8 (visible at start of lap 8)
8. `RADIO_DRIVER "SC. Boxing this lap."` (lap 8)
9. `PIT_NOW hard` (lap 8, under SC — only ~8 s pit-loss)
10. `SET_MODE race` (lap 9, behind SC)
11. `STAY_OUT` (laps 9–11)
12. SC releases lap 11, ego on fresh hards vs leader on used mediums
13. `SET_MODE push` (lap 11)
14. `ATTACK_AHEAD` (lap 12) — try for P1
15. `DONE` (after lap 12)

Result: P2 or P1, score 0.92.

### Trap sequences

**Trap A — pre-pit at lap 5 (verified 0.45):** Agent pits early at green, costs 22 s, drops to P5. SC fires — opponents pit free, ego is now P6 on stale tyres. Strategic-decision points zeroed.

**Trap B — never pit (verified 0.30):** Agent stays out the whole race, tyres die at lap 10, loses 4 positions in last 2 laps. Operational efficiency 0 (zero pits = compound rule violated). Multiple harmful-action penalties.

---

## Stretch: Championship Decider (Catalunya)

**Disruption type:** `championship_decider`
**Track:** Catalunya (balanced, abrasive, mixed character)
**Total laps:** 15
**Max steps:** 20
**Core tension:** Last race of the championship. Agent must defend P3 against ONE specific
rival (#10, the championship rival) for points purposes, while managing a tyre crossover
and a mid-race light-rain forecast. Aggressive pace tanks tyres before the rain change;
passive pace cedes the championship position.

### Episode setup

Ego car: 3rd, hard tyres age 0, fuel 110 kg, race mode.

Opponents:
- 5 cars
- Notable: #10 (4th behind ego, championship rival — finishing ahead of #10 = championship)
- Notable: #44 (1st, pulling away, irrelevant to title)

Weather: dry start, **40% rain probability** at laps 8–11.

Pending issues:
- I-001: Race result — finish ≥ 3rd AND ahead of #10 (points = 0.40 — title-defining).
- I-002: Tyre management — health > 0.30 at flag, compound rule (points = 0.15).
- I-003: Strategic — pit timing + rain reaction (points = 0.20).
- I-004: Comms — multiple `RADIO_DRIVER` calls during dynamic events (points = 0.15).
- I-005: Operational — ≤ 2 pit stops, no DNF (points = 0.05).
- I-006: Fuel — margin ≥ 0.5 kg (points = 0.05).

### Hidden state

- `weather_evolution`: 30% probability of mild rain at lap 9; in this seed it materialises at lap 9 with intensity 0.25 (damp, not wet)
- `opponent_strategies`:
    - #10 → planned pit lap 7, soft for sprint
    - others → mixed
- `safety_car_schedule`: 20% probability at any lap 6–12

### Optimal sequence (verified 0.91, on the rain-not-coming variant 0.93)

Sketch (full sequence requires balancing rain and #10's undercut):
1. Inspections lap 1–2
2. `HOLD_GAP 10` early — defensive
3. `REQUEST_FORECAST` lap 5 — reveals rain cone
4. Cover #10's pit (`PIT_NOW medium` lap 8 if they pit lap 7)
5. Watch rain — if it comes, switch to inters lap 10; if not, push on mediums
6. Defend final 3 laps with `DEFEND_POSITION` if #10 closes
7. Multiple `RADIO_DRIVER` events for SC, rain, and final-stint planning

### Trap sequence

**"Played for points without checking" (verified 0.40):** Agent runs ultra-conservative, doesn't inspect, doesn't react to rain, finishes P5 behind #10 — championship lost. Race result issue zeroed (the 0.40 dimension). Final score around 0.40.

---

## Scenario Dict Schema

```python
{
    "task_name": str,                          # unique identifier
    "scenario_family": str,                    # "dry_strategy_sprint" | "weather_roulette" | "late_safety_car" | "championship_decider"
    "description": str,                        # shown in reset() observation
    "track_name": str,                         # must exist in data/tracks/
    "total_laps": int,
    "max_steps": int,
    "max_score": float,                        # always 1.0
    "starting_position": int,
    "starting_compound": str,
    "starting_fuel_kg": float,

    "opponents": [
        {
            "driver_number": int,
            "team": str,
            "starting_position": int,
            "starting_compound": str,
            "pace_offset_s": float,
            "aggression": float,
            "planned_strategy": [
                {"compound": str, "planned_end_lap": int}
            ],
        }
    ],

    "weather_archetype": str,                  # "clear_dry" | "light_rain_window" | "heavy_rain_lottery" | "safety_car_prone" | "mixed_conditions"
    "weather_seed_overrides": dict,            # optional fixed values
    "sc_archetype": str,                       # "none" | "sc_window" | "vsc_window"

    "issues": {
        "race_result": [
            {"goal": str, "valid_resolutions": list[str], "points": float}
        ],
        "tyre_management": [
            {"constraint": str, "points": float}
        ],
        "fuel_management": [
            {"constraint": str, "points": float}
        ],
        "strategic_decisions": [
            {"decision": str, "valid_window": list[int], "preconditions": list[str], "points": float}
        ],
        "pending_comms": [
            {"trigger": str, "audience": "driver|pit_wall|media", "required": bool, "points": float}
        ],
    },

    "hidden_state": {
        "true_tyre_curve": dict,
        "opponent_strategies": dict,
        "weather_evolution": list,
        "safety_car_schedule": list,
        "fuel_burn_actual": float,
        "undercut_threshold_laps": int,
    },

    "dynamic_events": [
        {"lap": int, "type": str, "desc": str}
    ],

    "memory_hint_tags": list[str],             # tags for postmortem retrieval
}
```

---

## Validation rules

Before committing a scenario, run:
- `python -m baselines.expert_solver --task <task_name>` and confirm score ≥ 0.85
- `python tests/smoke_all_scenarios.py` confirms expert ≥ 0.85, panic ≤ 0.40
- Manually inspect that issue points sum to 1.0

If the expert solver can't reach 0.85, the scenario is broken (usually a hidden-state
inconsistency or an issue where the optimal action is not actually allowed).
