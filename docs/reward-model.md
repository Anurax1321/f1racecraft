# Reward Model

Six scoring dimensions, all deterministic — no LLM judge anywhere in the scoring path.
Final scalar = weighted sum, clamped to `[0.01, 0.99]`.

This is a near-direct port of OpsTwin's six-dimension scorer, with race-relevant signals
swapped in. The architectural pattern (independent pure functions, weighted aggregation,
delayed-credit support) is unchanged because it was a strength of the OpsTwin submission.

---

## Dimension Weights

| Dimension | Weight | OpsTwin equivalent |
|-----------|--------|--------------------|
| `race_result` | 0.35 | `service_recovery` (was 0.35) |
| `strategic_decisions` | 0.20 | `customer_outcome` (was 0.20) |
| `tyre_management` | 0.15 | `security_compliance` (was 0.15) |
| `fuel_management` | 0.10 | `change_hygiene` (was 0.10) |
| `comms_quality` | 0.10 | `communication_quality` (was 0.10) |
| `operational_efficiency` | 0.10 | `operational_efficiency` (was 0.10) |

Total: 1.00.

---

## Dimension Specs

### 1. `race_result` (0.35)

```python
def compute_race_result(
    starting_position: int,
    finishing_position: int,
    target_position: int,
    rival_finishing_position: int | None,
    target_rival_position: int | None,
) -> float:
    """
    starting_position: grid slot
    finishing_position: chequered-flag slot
    target_position: scenario-specified target (e.g. P3)
    rival_finishing_position: for championship_decider — must finish ahead of this rival
    target_rival_position: for championship_decider — should be > rival's actual finish
    """
    # Base: did we hit our target?
    if finishing_position <= target_position:
        base = 1.0
    else:
        # Linear penalty per position lost beyond target
        base = max(0.0, 1.0 - 0.20 * (finishing_position - target_position))

    # Position gain bonus
    delta = starting_position - finishing_position
    if delta > 0:
        base = min(1.0, base + 0.05 * delta)

    # Championship-specific: must beat rival
    if rival_finishing_position is not None:
        if finishing_position < rival_finishing_position:
            base = min(1.0, base + 0.10)
        else:
            base = max(0.0, base - 0.30)  # title lost — heavy penalty

    return min(max(base, 0.0), 1.0)
```

Notes:
- DNF auto-zeroes this dimension (caller passes finishing_position = 99)
- Stretch championship_decider scenario wires the rival check; other families pass `None`

---

### 2. `strategic_decisions` (0.20)

```python
def compute_strategic_decisions(
    pit_decisions: list[PitDecision],     # (lap, compound, was_under_sc, preconditions_met)
    optimal_pit_window: tuple[int, int],
    inspection_calls: dict[str, list[int]],   # action_name -> list of laps it was called
    forecast_called_before_pit: bool,
    undercut_assessed_before_pit: bool,
) -> float:
    """
    Rewards (a) pitting in the optimal window, (b) doing inspection BEFORE the pit decision,
    (c) avoiding the panic-pit penalty.
    """
    score = 0.0

    # Pit window adherence (largest term)
    valid_pits = [
        p for p in pit_decisions
        if optimal_pit_window[0] <= p.lap <= optimal_pit_window[1]
    ]
    if valid_pits:
        score += 0.50

    # Preconditions: ASSESS_UNDERCUT_WINDOW before pit
    if undercut_assessed_before_pit:
        score += 0.20

    # REQUEST_FORECAST before any weather-driven pit
    if forecast_called_before_pit:
        score += 0.20

    # Free-pit recognition (under SC)
    sc_pits = [p for p in pit_decisions if p.was_under_sc]
    if sc_pits:
        score += 0.10

    return min(max(score, 0.0), 1.0)
```

Notes:
- This is the dimension that punishes shallow play hardest
- Inspection-without-action is detected: if `INSPECT_*` was called but the next action ignored the revealed info, score is reduced

---

### 3. `tyre_management` (0.15)

```python
def compute_tyre_management(
    final_tyre_health: float,
    compound_rule_satisfied: bool,
    stints: list[Stint],
    track_character: str,
) -> float:
    """
    final_tyre_health: 0.0–1.0 health at chequered flag
    compound_rule_satisfied: ≥2 distinct compounds used in dry race (FIA rule)
    stints: list of stints with compound, length, end_health
    """
    score = 0.0

    # Compound rule (FIA gating)
    if compound_rule_satisfied:
        score += 0.30
    else:
        return 0.0  # full penalty — this dimension fails

    # Final tyre health
    if final_tyre_health > 0.40:
        score += 0.40
    elif final_tyre_health > 0.20:
        score += 0.20
    else:
        score += 0.0

    # Stint discipline: no stint ended with tyres at < 0.10 (death)
    if all(s.end_health > 0.10 for s in stints):
        score += 0.20

    # Bonus: extracted compound-appropriate stint length
    # (e.g. a 12-lap soft stint is wasteful; a 20-lap soft stint is heroic)
    score += 0.10 if all(_stint_is_appropriate(s, track_character) for s in stints) else 0.0

    return min(max(score, 0.0), 1.0)
```

---

### 4. `fuel_management` (0.10)

```python
def compute_fuel_management(
    final_fuel_kg: float,
    target_margin_kg: float,
    dnf_due_to_fuel: bool,
    inspect_fuel_called: bool,
) -> float:
    if dnf_due_to_fuel:
        return 0.0

    score = 0.0
    if final_fuel_kg >= target_margin_kg:
        score += 0.70    # finished with margin — main goal
    elif final_fuel_kg >= 0.0:
        score += 0.30    # finished but tight

    if inspect_fuel_called:
        score += 0.20    # discipline bonus — can't claim margin without measuring
    else:
        score = score * 0.5   # half credit if you got lucky without checking

    if final_fuel_kg > 3.0:
        score -= 0.10    # under-burn penalty — dragged extra weight all race

    return min(max(score, 0.0), 1.0)
```

---

### 5. `comms_quality` (0.10)

```python
def compute_comms_quality(
    triggered_comms_events: list[CommsEvent],   # events that REQUIRED a radio call
    radio_calls_made: list[RadioCall],          # actual RADIO_DRIVER calls
    pit_wall_calls: list[PitWallCall],          # DRAFT_PIT_WALL calls
) -> float:
    if not triggered_comms_events:
        return 1.0    # nothing to do, full credit

    # Coverage: fraction of required comms that got a call within ±1 lap
    covered = 0
    for evt in triggered_comms_events:
        matches = [c for c in radio_calls_made if abs(c.lap - evt.lap) <= 1]
        if matches:
            covered += 1
    base = covered / len(triggered_comms_events)

    # Timeliness: comms in the same lap as the trigger gets a small bonus
    timely = sum(
        1 for evt in triggered_comms_events
        if any(c.lap == evt.lap for c in radio_calls_made)
    )
    timeliness_bonus = 0.10 if timely > 0 else 0.0

    # Pit-wall discipline: at least one DRAFT_PIT_WALL during the race
    pit_wall_bonus = 0.10 if pit_wall_calls else 0.0

    return min(base + timeliness_bonus + pit_wall_bonus, 1.0)
```

Notes:
- "Required" comms events fire from `dynamic_events` in the scenario (e.g. SC trigger,
  rain start, first pit announcement)
- Each scenario family specifies which comms are required; `weather_roulette` requires
  the rain warning, `late_safety_car` requires the SC announcement, etc.

---

### 6. `operational_efficiency` (0.10)

```python
def compute_operational_efficiency(
    n_pit_stops: int,
    target_n_pits: int,
    invalid_actions: int,
    harmful_actions: int,
    total_laps: int,
    steps_used: int,
) -> float:
    score = 0.0

    # Exact match on planned pit count
    if n_pit_stops == target_n_pits:
        score += 0.50
    elif n_pit_stops == target_n_pits + 1:
        score += 0.20    # one extra pit, partial
    elif n_pit_stops == target_n_pits - 1:
        score += 0.10    # one fewer (likely compound rule violation)

    # No wasted commands
    score -= 0.05 * invalid_actions
    score -= 0.10 * harmful_actions

    # Step efficiency: fewer steps relative to laps = more decisive
    decisiveness = 1.0 - min(1.0, max(0.0, (steps_used - total_laps) / total_laps))
    score += 0.30 * decisiveness

    # Tail bonus: if total_laps + 1 ≥ steps_used, agent was efficient
    if steps_used <= total_laps + 1:
        score += 0.20

    return min(max(score, 0.0), 1.0)
```

---

## Final Scalar

```python
def compute_multi_objective_scores(...) -> dict:
    scores = {
        "race_result": compute_race_result(...),
        "strategic_decisions": compute_strategic_decisions(...),
        "tyre_management": compute_tyre_management(...),
        "fuel_management": compute_fuel_management(...),
        "comms_quality": compute_comms_quality(...),
        "operational_efficiency": compute_operational_efficiency(...),
    }

    weights = {
        "race_result": 0.35,
        "strategic_decisions": 0.20,
        "tyre_management": 0.15,
        "fuel_management": 0.10,
        "comms_quality": 0.10,
        "operational_efficiency": 0.10,
    }

    final = sum(scores[k] * weights[k] for k in scores)
    scores["weighted_final"] = round(min(max(final, 0.01), 0.99), 4)
    return {k: round(v, 4) for k, v in scores.items()}
```

---

## Per-step shaping reward

Distinct from the final scalar above. The trainer uses per-step rewards to learn faster.

For each step:
```python
per_step_reward = (
    issue_resolution_reward          # +issue["points"] if action resolved an issue
  + new_revelation_bonus             # +0.02 first time an INSPECT reveals new state
  + mode_match_bonus                 # +0.005 if SET_MODE matches phase recommendation
  + invalid_penalty                  # -0.02 for unknown / not-allowed commands
  + harmful_penalty                  # -0.05 for panic_pit / unnecessary defend
  + delayed_credit                   # +/- pending rewards finalised this step
)
```

Final-step reward also adds the `weighted_final - 0.01` (so untrained policy gets ~0
extra at flag while expert gets ~+0.94). This ties the per-step signal to the multi-
objective final score.

---

## Delayed Reward Pattern

Not all good actions reward immediately. Implement this pattern in `step()`:

1. Action is taken → partial reward credited immediately
2. Hidden state revealed later (via inspection) → remaining reward released or withheld

Example: `PIT_NOW soft` at lap 5 in `dry_strategy_sprint`:
- Immediate: `+0.10` (partial credit for pitting in the broad valid window 4–7)
- After `ASSESS_UNDERCUT_WINDOW` confirms 2-lap threshold AND `CHECK_OPPONENT_STRATEGY 16`
  confirms #16 already pitted: `+0.10` released (the pit was the optimal play)
- If `ASSESS_UNDERCUT_WINDOW` was never called: `+0.10` is withheld and the strategic-
  decisions dimension is reduced

Implementation: store `_pending_rewards` dict keyed by `(action_signature, hidden_state_key)`.
On each step, check if any pending rewards can be finalised based on currently-revealed
hidden state.

This delayed pattern is what makes the environment genuinely hard for shallow policies.
A policy that always pits in the right window but never investigates can't unlock the
delayed credits, capping at ~0.55 even on Family 1.

---

## Reward hacking checklist

Before training, validate:

- [ ] **Inspection spam.** Calling `INSPECT_TYRE_DEGRADATION` 14 times still pays only `+0.02` (first reveal) + 0.0 (repeats). Verified.
- [ ] **DRAFT_PIT_WALL spam.** Comms quality bonus caps at 0.10 regardless of message count. Verified.
- [ ] **STAY_OUT spam.** Burns step budget; if `done` triggers via lap limit, operational efficiency drops sharply.
- [ ] **DONE before all issues resolved.** Caps `pending_issues_count > 0` at episode end → final dimensions all reduced.
- [ ] **PIT_NOW twice in two laps.** Second pit triggers harmful-action penalty if not justified by SC or weather change. Operational efficiency drops.
- [ ] **Set mode push every step.** Tyres die early, `tyre_management` zeroes.
- [ ] **Set mode conserve every step.** Race result tanks (positions lost), `race_result` drops.

If any of the above can be exploited to reach ≥0.50 in a normal seed, fix the scoring
before training starts. This is not optional.
