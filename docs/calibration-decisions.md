# Calibration Decisions Log

> **Why this file exists.** When training reveals a parameter is wrong (and
> it will), we need to know *exactly* what we picked, *why* we picked it, and
> what would change if we tweaked it. Don't tune blind. Don't tune from
> memory. Tune against this list.

Every numeric or methodological choice in our calibration layer is logged
here with: **value · rationale · sensitivity · revisit-when**.

---

## Section A — Tire degradation curve fitting

### A1. Linear regression model
- **Choice:** linear `lap_time = slope × tyre_age + intercept`, fit per
  (track, compound).
- **Why:** simplest defensible model that gives a single interpretable
  number per compound per track. Beatson 2025 (arXiv:2512.00640) is more
  rigorous (Bayesian state-space) but a multi-day implementation.
- **Sensitivity:** linear is wrong when degradation has cliffs (sudden
  drop-offs). Compounds at end-of-life behave non-linearly.
- **Revisit when:** R² stays low (<0.20) on tracks where we expect strong
  signal. That's the symptom that linear doesn't fit. Earmarked: Phase
  6.5+ enhancement to swap in the state-space model.

### A2. Per-(race, driver, stint) normalization
- **Choice:** subtract each stint's mean lap-time before fitting.
- **Why:** raw lap times mix "how fast Hamilton's car is" with "how fast
  this tire is degrading." Inter-driver pace dwarfs the tire signal
  (~0.5s vs ~0.05s/lap). Subtracting per-stint means leaves only the
  within-stint trend.
- **Sensitivity:** without this, R² drops from 0.10–0.48 to 0.05–0.14 (we
  measured this).
- **Revisit when:** never — this is core to the methodology.

### A3. Top-team filter (Red Bull, Mercedes, McLaren, Ferrari)
- **Choice:** restrict tire-fitting to laps from these 4 constructors.
- **Why:** their cars run more consistent pace lap-to-lap; their
  drivers manage tires consistently. Smaller teams' data has more
  external noise (bad starts, traffic, etc.).
- **Source:** matches Mercedes-AMG paper (arXiv:2501.04067) which uses
  Mercedes-only data for tire-energy modeling.
- **Sensitivity:** doubles R² but halves sample size. Tradeoff favors
  cleanness over volume given our lap-time variance.
- **Revisit when:** if some compound has too few samples per track to fit
  reliably (n<50), consider opening to all teams.

### A4. Quality gate: R² ≥ 0.05 AND n ≥ 50
- **Choice:** below either threshold, fall back to factor 1.0.
- **Why:** R² < 0.05 means fit explains <5% of variance — basically noise.
  n < 50 means too few laps to trust the slope. Either alone is grounds
  to discard.
- **Sensitivity:** if too lenient → noise contaminates physics; too
  strict → fewer tracks have grounded data.
- **Revisit when:** training reveals grounded-calibrated tracks behave
  weirdly. Tighten gate (e.g., R² ≥ 0.15) and re-evaluate.

### A5. Slope floor for track-evolution-dominant tracks (0.005 s/lap)
- **Choice:** if raw fitted slope < 0.005, use 0.005 as the effective slope.
- **Why:** Monaco/Singapore show negative raw slopes (track grip improves
  more than tires wear). Real tires DO physically wear; the env shouldn't
  let them be immortal.
- **Sensitivity:** if too low → tires effectively don't wear; too high →
  street circuits feel like Spa.
- **Revisit when:** scenario authors note a street-circuit scenario is
  unwinnable because tires wear too fast.

### A6. Wear-factor sanity clamp [0.20, 3.0]
- **Choice:** the per-track-per-compound multiplier is clamped to this
  range before reaching physics.
- **Why:** a noisy fit could produce factor 0 (tires never wear) or factor
  10 (instant DNF). Real F1 wear varies maybe 5× across tracks. Clamping
  prevents pathological values.
- **Sensitivity:** values near the clamp boundaries are likely artifacts
  of low-sample fits.
- **Revisit when:** more than ~3 tracks hit the clamp boundaries — that
  indicates the underlying fits are unreliable.

### A7. Track-evolution floor 0.5
- **Choice:** if track has `track_evolution_dominant: True` AND raw factor
  < 0.5, force factor = 0.5.
- **Why:** track-evolution tracks (Monaco, Singapore) wear tires gently
  but not zero. Floor 0.5 says "even at the most gentle track, tires wear
  half as fast as average — not 10× slower."
- **Sensitivity:** lowering to 0.3 would make Monaco effectively
  consequence-free for tires.
- **Revisit when:** Monaco strategy in trained models becomes "stay out
  forever" — that means the floor is too low.

### A8. "1.5s = spent" constant in health curve generation
- **Choice:** `health = 1 - (slope × age) / 1.5` — tires are "spent"
  (health → 0.05) after the lap-time penalty reaches 1.5s.
- **Why:** real F1 tires fall off ~1–2s slower than fresh before being
  changed. 1.5 is the midpoint.
- **Sensitivity:** using 1.0 → faster wear; using 2.0 → slower wear.
  Affects only the `health_curve` exposed to the model via INSPECT,
  NOT the physics.
- **Revisit when:** the model's belief about tire life (from INSPECT)
  doesn't match what physics actually does. That's an inconsistency
  signal.

---

## Section B — Wet-race / weather handling

### B1. Wet-race definition: any INTERMEDIATE or WET compound used
- **Choice:** classify a race as `wet_race=True` if any driver used inter
  or wet tires.
- **Why:** crude but practical. FastF1 doesn't expose "race officially
  declared wet" cleanly.
- **Sensitivity:** false positives (one driver gambled on inters in light
  drizzle) flag the race as wet when most stints were dry.
- **Revisit when:** loss of useful dry stints from these false-positive
  races is hurting calibration.

### B2. Per-stint dry-compound filter (current, post per-lap upgrade)
- **Choice:** include all stints where the compound is hard/medium/soft,
  even if the race was nominally wet. Drop intermediate/wet stints.
- **Why:** Britain 2024 had laps 1–26 dry on slicks then rain. The dry
  stint data is valid — there's no reason to exclude it. Earlier
  whole-race-skip discarded these.
- **Effect when introduced:** Silverstone medium slope went 0.024 → 0.086
  (n=215 → 436). Closer to F1 wisdom of "Silverstone is high-wear."
  Spa medium slope went 0.091 → 0.011 (revealed the medium > soft
  pattern WAS contamination).
- **Sensitivity:** mixes dry stints from races with very different track
  conditions (cool damp surface after rain ≠ dry warm surface). May
  introduce noise where it removes contamination.
- **Revisit when:** B3 (per-lap weather filter) supersedes this.

### B3. Per-lap weather filter (FUTURE — not implemented)
- **Status:** not yet applied. Earmarked.
- **What it would do:** for each lap record, look up `Rainfall` from
  `session.weather_data` at `lap.LapStartTime`. Drop laps where rainfall
  > 0 even if the compound is dry.
- **Why future-work:** requires weather-timestamp lookup against lap-time
  index. Couple hours of extraction code change.
- **Revisit when:** R² stays low on tracks with frequent partial-rain
  races (Britain, Belgium, Brazil, Zandvoort, Monaco).

---

## Section C — Fuel correction

### C1. Constants: 110 kg start, 1.95 kg/lap burn, 0.030 s/kg lap-time effect
- **Source:** Beatson 2025 (arXiv:2512.00640) decomposition; canonical F1
  values.
- **Why:** allows isolating tire-pace effect from fuel-mass effect on
  raw lap times.
- **Sensitivity:** wrong burn rate (1.95 should be 1.7 at MexicoCity due
  to altitude, 2.1 at Monza due to long throttle) introduces systematic
  bias. Per-track fuel correction would be more accurate.
- **Revisit when:** lap-time degradation curves systematically mis-fit
  early/late laps (suggests fuel correction is wrong).

---

## Section D — Scoring (sub-step 1.3 + 1.5 hardenings)

### D1. Operational efficiency penalty per missed pit (0.30, was 0.45)
- **Choice:** `pit_score = 1 - 0.30 × |n_pits - target|`.
- **Why:** 0.45 over-penalized; combined with strategic_decisions
  penalty for missed pit windows, double-counted the same mistake.
- **Sensitivity:** too low → over-pitting becomes safe; too high → small
  pit-count errors crater the score.
- **Revisit when:** trained model strategy distribution starts looking
  pit-heavy or pit-shy.

### D2. Egregious over-pit hard cap (n_pits > target + 3)
- **Choice:** when n_pits exceeds target by >3, cap race_result and
  tyre_management at 0.30 each.
- **Why:** 1.5 baseline measurement found random pit-spam (19 pits)
  scored 0.66 because position-floor + tire-refresh hid the cost. Would
  poison GRPO training.
- **Sensitivity:** if too lenient → pit-spam still wins; too strict →
  legitimate 4-stop strategies (rare wet races) get capped.
- **Revisit when:** training reveals weird pit-count behavior.

### D3. Multi-window strategic scoring (sub-step 1.3)
- **Choice:** for `target_n_pits ≥ 2`, require pit in BOTH windows
  (`optimal_pit_window` AND `second_pit_window`) for full credit.
- **Why:** otherwise 1-stop incomplete play scored same as clean 2-stop.
- **Revisit when:** scenarios with 3+ stops are added (would need
  `third_pit_window`).

---

## Section F — Grounded calibration mechanism (sub-step 2.4)

### F1. Default-off opt-in (no scenario opts in by default)
- **Choice:** existing scenarios run synthetic physics. Grounded
  calibration activates only when scenario sets
  `use_grounded_calibration: True`.
- **Why:** turning it on for Spa shifted the expert sequence score from
  0.99 → 0.46. Existing experts are tuned to synthetic physics; flipping
  them wholesale would break the ≥0.85 expert floor on every scenario.
  Re-authoring is a separate task per scenario.
- **Sensitivity:** scenarios that opt in WILL behave differently. Each
  needs its own expert sequence verification.
- **Revisit when:** a scenario is intentionally re-authored to use grounded
  calibration. At that point, the opt-in flag becomes truthful.

### F2. Per-compound factors populated for all dry compounds at reset
- **Choice:** when opt-in is true, env populates factors for hard, medium,
  soft, inter, wet — not just compounds the scenario uses.
- **Why:** the model can pit to ANY compound mid-race. If we only
  populated factors for the starting compound, post-pit physics would
  silently fall back to synthetic. Inconsistency = bad signal.

### F3. Test rewrite — no cross-track ordering assertions
- **Choice:** removed `test_relative_wear_orders_match_intuition` and
  `test_optin_actually_changes_env_behaviour` (Spa-specific score-drop).
  Replaced with `test_wear_factors_pass_sanity_clamp` and
  `test_optin_path_is_wired_correctly`.
- **Why:** the old tests asserted that Spa medium > Monaco medium and
  that opting Spa in would crash the expert. Both were *encoding the
  wet-race contamination as ground truth*. After per-stint filtering
  removed the contamination, the tests correctly broke. The new tests
  verify the *mechanism* (clamp, wiring) without depending on contaminated
  ordering.
- **Lesson:** when calibration changes, tests that pin specific numbers
  may be testing artifacts, not behavior.

---

## Section E — Reward shaping (per `docs/reward-philosophy.md`)

### E1. Per-step shaping rewards
- **Inspections (first-time only):** `+0.02`
- **PIT_NOW, HOLD_GAP:** `+0.01`
- **SET_MODE, RADIO_DRIVER, etc:** `+0.005`
- **Invalid command:** `-0.02`
- **Harmful action:** `-0.05`

These are deliberately small so terminal score dominates. Total per-episode
shaping bounded at ~0.20. Don't add dense per-step shaping; see reward
philosophy doc.

---

## How to use this file when something looks wrong

1. **Find the symptom** in the "Revisit when" line of each entry.
2. **Read the rationale** before changing the value.
3. **Change one thing at a time.** Multiple simultaneous changes make
   diagnosis impossible.
4. **Re-run** `scripts/reward_audit.py` and `scripts/fastf1_aggregate.py`
   to see what propagates.
5. **Update this file** with the new value AND the reason for the change.
