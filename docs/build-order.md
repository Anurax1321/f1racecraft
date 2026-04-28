# Build Order

Follow this phase sequence strictly. Do not advance to the next phase if the current one is
incomplete. The training evidence in Phase 3 is non-negotiable — it is 20% of the judging
score, and judges will refuse to engage seriously with a project that has none.

---

## Phase 0 — Bootstrap (parallelisable)

Goal: shared scaffolding both authors can pull from on day one.

**Both:**
1. Clone the repo, run `pip install -e .` (or `uv pip install -e .`) to install in editable mode.
2. Read [`docs/architecture.md`](architecture.md) end-to-end. Read your own person-tasks doc.

**Person 1 (Shashwat):**
3. Run `scripts/extract_track_csvs.py` to copy the five chosen tracks from `racetrack-database-master.zip` into `data/tracks/` with normalised column names. Verify each CSV loads in `server/track.py` and produces a sane closed-loop centerline.
4. Spot-check track lengths against published F1 figures (Monza 5.793 km, Monaco 3.337 km, Catalunya 4.657 km, Spa 7.004 km, Silverstone 5.891 km). If a CSV is in metres, leave it; if it's in kilometres, convert to metres in `track.py`. Write the chosen unit into `data/tracks/README.md`.

**Person 2 (Tanish):**
5. Run `scripts/calibrate_opponent_pace.py` against `archive.zip` (Kaggle Formula 1 dataset) and `race_results.zip`. Output: `data/opponent_pace_calibration.json` containing per-track median lap-time, p25, p75, and stint-length distribution per compound (when stint data is available).
6. Run `scripts/calibrate_tyre_baseline.py` to produce `data/tyre_compound_baseline.json` — soft/medium/hard pace deltas (s/lap) and degradation rates (s/lap²) derived from observed stint pace patterns.
7. Create the empty HF Space (`huggingface-cli repo create f1-strategist --type space --space_sdk docker --organization Deltasthic`). Push README only at this stage to confirm metadata renders correctly.

**Exit criteria:** `data/` is populated, both authors can `python -c "from server.track import load_track; print(load_track('Monza'))"` without error, the HF Space exists and shows the README.

---

## Phase 1 — Environment Skeleton (Person 1 owns)

Goal: `reset()/step()` loop works end-to-end with stub physics. No real scenarios yet.

8. Implement `models.py`. Define `F1Action`, `F1Observation`, `F1State` using OpsTwin
   models as template. Replace operations fields with race fields. Use the schema
   in [`docs/architecture.md`](architecture.md) §6.
9. Implement `client.py`. Thin subclass of `EnvClient[F1Action, F1Observation, F1State]`.
   Mirrors `OpsTwinRecoveryEnv` structure exactly.
10. Port `server/app.py` from OpsTwin. Single shared env singleton, FastAPI on 8000.
11. Implement `server/environment.py`:
    - Class `F1StrategistEnvironment`
    - Properties: `_track`, `_ego_car`, `_opponents`, `_weather`, `_lap`, `_total_laps`, `_done`, `_audit_trail`, `_fired_events`, `_dynamic_events`, `_pending_rewards`
    - Keep `reset()`/`step()`/`_obs()`/`_exec()`/`_load()` signatures identical to OpsTwin
    - Wire all action handlers as stubs returning `(0.0, "stub")`
    - Add `LAPS_PER_STEP = 1` (default; overridable per scenario)
12. Implement `server/track.py`:
    - `load_track(name) -> Track`
    - `Track` exposes `centerline: np.ndarray (N, 2)`, `length_m: float`, `corners: list[Corner]`, `pit_lane_offset_m: float`
    - Corner detection: simple curvature threshold on the racetrack-database racing-line CSVs
13. Implement `server/physics.py` v0:
    - `compute_lap_time(track, compound, tyre_age, fuel_kg, mode, dirty_air, weather, noise_seed) -> float`
    - For Phase 1 just return a constant base time per track; Phase 2 adds wear and fuel.
14. Smoke test: `tests/smoke_http.py` passes 5 checks (`/health` 200, `/reset` returns valid F1Observation, `/step` with `STAY_OUT` returns reward 0.0, `/step` with garbage returns reward -0.02, `done` flips after `total_laps`).

**Exit criteria:** Server starts, one episode completes, no exceptions, observation
schema validates.

---

## Phase 2 — Physics + Scenarios + Reward (split work)

Goal: deterministic scoring discriminates good from bad strategy.

**Person 1:**

15. `server/physics.py` v1:
    - Tyre wear: `health(t) = 1 - degradation_rate * t - delta_for_compound(c) * t²`
    - Lap time: `base + a * (1 - health) + b * fuel_kg + dirty_air_penalty + mode_delta + weather_penalty`
    - Calibrate constants against `data/tyre_compound_baseline.json`
    - Per-lap Gaussian noise (σ ≈ 0.15 s) so the optimal strategy isn't deterministic
16. `server/opponents.py`:
    - `Opponent` dataclass with `driver_number`, `team`, `pace_offset_s`, `aggression`, `planned_strategy: list[Stint]`
    - Each opponent has a hidden stint plan (stint = compound + planned end lap)
    - `step_opponents(state, lap)` advances each opponent by one lap, applies physics, executes their plan, returns updated standings
    - Pace offsets sampled from `opponent_pace_calibration.json`
17. `server/weather.py`:
    - `WeatherSchedule.from_seed(seed)` rolls a deterministic per-lap rain probability and one-of-many SC events
    - `WeatherSchedule.observe(current_lap, k_laps_ahead)` returns the **forecast** (a probability cone), not ground truth
    - Ground truth is never exposed except via the SC actually firing
18. `server/scenarios.py`:
    - Three hand-authored scenario dicts: `dry_strategy_sprint`, `weather_roulette`, `late_safety_car`
    - Each conforms to the schema in [`docs/scenarios.md`](scenarios.md) §schema
    - Each has a manually-traced optimal sequence verified to score ≥0.85
    - Each has a panic / suboptimal sequence verified to score ≤0.40
19. `server/hidden_state.py`:
    - Port from OpsTwin's `hidden_state.py` — same `HiddenStateLayer` API, new domain keys
    - Inspection actions reveal one variable each, +0.02 reward on first reveal, 0.0 on repeat
20. Wire all the above into `_load()` and `_exec()` in `environment.py`.

**Person 2:**

21. `server/scoring.py`:
    - Six pure functions, one per dimension, signatures in [`docs/reward-model.md`](reward-model.md)
    - `compute_multi_objective_scores(...)` returns a dict including `weighted_final` clamped to `[0.01, 0.99]`
    - No imports beyond stdlib + numpy. No LLM, no random.
22. `baselines/expert_solver.py`:
    - Rule-based solver that reads scenario state and emits the optimal action sequence
    - Uses inspection actions strategically (it knows the hidden-state schema; the solver runs at "god mode" with full information)
    - Run against all three scenarios + the stretch; verify score ≥0.92 on each
    - Save traces as `baselines/trajectories/expert_<scenario>_seed<n>.jsonl`
23. `tests/smoke_all_scenarios.py`:
    - For each scenario family + the stretch:
        - run expert solver, assert final score ≥ 0.85
        - run a deliberately-bad scripted policy (e.g. always STAY_OUT, never inspect), assert score ≤ 0.40
        - assert `total_issues > 0` after `reset()`, `done is False` after first non-DONE step
24. Generate the SFT seed dataset: `python capture_everything.py --n-seeds 50 --output sft_dataset.jsonl`. This will be the warm-start data for Phase 3.

**Exit criteria:** Three scenarios + stretch, expert solver works, reward signal
is clearly discriminative (≥0.45 spread between expert and panic policies on every
scenario).

---

## Phase 3 — Training (HIGHEST PRIORITY)

Goal: at least one GRPO reward curve showing improvement. This is what judges will ask
about.

25. `train.py` — TRL with environment_factory pattern:
    ```python
    from trl import GRPOTrainer, GRPOConfig
    from openenv import OpenEnvClient
    config = GRPOConfig(
        model_name="Qwen/Qwen3-4B",
        reward_funcs=["environment"],
        max_steps=500,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=10,
    )
    ```
    Use Unsloth on the RTX 5090 path (Qwen3-4B + LoRA, batch 1, grad-accum 32).
    Use the smaller Qwen3-0.6B path with QLoRA for Colab smoke.

26. Smoke run on the 5090: 50 steps, confirm reward curve trends up, `training_log.txt`
    is captured. If reward is flat for 50 steps, **stop** and inspect rollouts manually
    — there's almost certainly a reward bug, not a training bug.

27. Run optional warm-start SFT first if needed (curriculum learning):
    - Stage 1: SFT on expert trajectories (broad coverage)
    - Stage 2: GRPO with shaped rewards (refines decisions)
    - Stage 3: GRPO with sparse final-only rewards (long-horizon polish)
    See [`TRAINING.md`](../TRAINING.md) for exact recipes.

28. Full run on the 5090: 500 GRPO steps minimum. Aim for ≥0.20 average improvement
    over untrained Qwen3-4B on held-out seeds.

29. `evaluate.py`:
    - Modes: `--mode random`, `--mode untrained`, `--mode trained`, `--mode expert`
    - Inputs: `--n-seeds`, `--tasks`, `--output`
    - Outputs: `results/eval_summary.json`, `results/eval_curve.png`
    - Bar chart with four bars per scenario; error bars from std across seeds.

30. `notebooks/f1_strategist_training_colab.ipynb`:
    - Cell 1: install OpenEnv from PyPI + TRL + Unsloth + transformers
    - Cell 2: clone the env via `from_env("Deltasthic/f1-strategist")` (Docker pulled into Colab)
    - Cell 3: tiny GRPO smoke run on Qwen3-0.6B, 30 steps
    - Cell 4: print loss + reward curves
    - Cell 5: print one rollout end-to-end as a chat transcript
    - Should run in ~7 minutes on a Colab Free T4.

31. Push trained checkpoint to HF Hub: `Deltasthic/f1-strategist-qwen3-4b-grpo`.

32. Produce the evidence package (minimum for demo):
    - `results/training_loss_curve.png` — loss + reward over training steps
    - `results/eval_curve.png` — random vs untrained vs trained vs expert bar chart
    - `results/eval_summary.json` — canonical numbers
    - `results/FINAL_RESULTS.md` — narrative table + methodology notes
    - One qualitative example: a trajectory where trained policy avoids a mistake
      that untrained makes. Render it via the visualiser.

**Exit criteria:** Reward curve exists. Trained policy measurably outperforms
untrained on held-out seeds (≥0.20 average improvement). HF checkpoint published.

---

## Phase 4 — Self-Improvement Story (Person 1)

Goal: postmortem memory loop produces a measurable additional improvement delta.

33. Implement `server/postmortem.py`:
    ```python
    {
      "scenario_family": str,
      "failure_category": str,   # missed_undercut | late_weather_call | fuel_underburn |
                                 # cascade_ignored | comms_forgotten | panic_pit | thrashing
      "first_bad_action": str,
      "missed_signal": str,
      "preferred_intervention_order": list[str],
      "final_score": float,
      "episode_id": str,
      "timestamp": str,
    }
    ```
    Append-only `.jsonl` at `baselines/trajectories/postmortems.jsonl`.

34. Hook generation into `environment.py`: when `_done = True`, classify the failure
    via simple rules over the audit trail and call `PostmortemMemory.record(summary)`.

35. In `reset()`, retrieve top-2 most similar past postmortems (same scenario family,
    lowest scores) and inject into observation under `memory_hints: list[str]`:
    ```
    [MEMORY] Last similar incident: failure_category=late_weather_call.
    First bad action: STAY_OUT lap 7. Missed signal: REQUEST_FORECAST not called.
    Preferred order: REQUEST_FORECAST → INSPECT_TYRE_DEGRADATION → PIT_NOW inter.
    ```

36. **Person 2** then runs the ablation: trained policy with vs without postmortem hints
    on 10 held-out seeds. Show improvement delta in `results/ablation.json`.

**Exit criteria:** Memory retrieval works. Augmented policy shows ≥0.05 average
improvement over base trained.

---

## Phase 5 — Polish + Demo + Deploy

37. **Person 1**: `server/visualizer.py` — matplotlib animator that takes a rollout
    `.jsonl` and emits a top-down GIF/MP4. Track centerline, ego car position, opponent
    positions, weather state overlay, pit-window indicators, lap counter. Used for the
    blog and video.

38. **Person 1**: `server/generator.py` — procedural seed-deterministic variants of the
    three families. `generate(family, seed, difficulty) -> ScenarioDict`. Each must be
    solvable: verify by running the expert solver and asserting score ≥ 0.85.

39. **Person 1**: enable Gradio web UI in the Space. Add `ENABLE_WEB_INTERFACE=1` to the
    Dockerfile env. Test the `/web` route works on the live Space.

40. **Person 2**: `demo-assets/blog-post.md` — final polish, target 600–900 words,
    Hugging Face blog format. Embed `eval_curve.png` and one visualiser GIF.

41. **Person 2**: `demo-assets/video-script.md` — record video < 2 min. Open with the
    untrained agent making a panic pit, cut to the trained agent calling the rain
    correctly, end on the bar chart.

42. **Person 2**: deploy HF Space (see [`DEPLOY_SPACE.md`](../DEPLOY_SPACE.md)).

43. **Person 2**: smoke-test the live Space. Save URL to `demo-assets/hf-space-link.txt`.

44. **Person 2**: publish blog on HF Hub, video on YouTube. Update README links.

45. **Both**: walk through [`PRE_PUSH_CHECKLIST.md`](../PRE_PUSH_CHECKLIST.md) item by item before final tag.

**Exit criteria:** Demo-ready. Blog post published. README complete. All four W-bullets
in [`TODO.md`](../TODO.md) ticked.

---

## Pre-Onsite Checklist (must complete before April 25)

- [ ] Phases 0–2 complete locally (no GPU needed)
- [ ] `train.py` written and unit-smoke-tested on Qwen3-0.6B
- [ ] Expert trajectories generated and stored
- [ ] At least 1 short reward curve from a 30-step smoke run on the 5090
- [ ] `evaluate.py` works against random + untrained + expert
- [ ] `data/tracks/*.csv` and `data/*.json` populated and committed
- [ ] HF Space exists and serves a hello-world `/reset`

Phase 3 full training run happens **on-site (Apr 25–26)** with HuggingFace compute
credits and the personal RTX 5090 server.
