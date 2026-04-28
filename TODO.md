# F1 Strategist — Checkpoint Tracker

Submission: **Meta PyTorch OpenEnv Hackathon Grand Finale**, Bangalore, April 25–26, 2026.

Authors: Shashwat Rajan, Tanish Shitanshu.
Repo: `Deltasthicc/f1-strategist`.
HF Space: `Deltasthic/f1-strategist`.

> Person 1 = Shashwat (Environment / Physics / Visualisation).
> Person 2 = Tanish (Training / Evaluation / Demo).

---

## Rubric: Minimum Requirements (gating)

- [x] **W1** Uses latest OpenEnv (`openenv-core>=0.2.3`).
- [x] **W1** Minimal TRL training script runnable in Colab — `notebooks/f1_strategist_training_colab.ipynb`.
- [ ] **W2** Environment hosted on HuggingFace Spaces. Push with `git push hfspace main:main`.
- [ ] **W3** Mini-blog on HuggingFace OR YouTube video <2 min. Drafts in `demo-assets/`.
- [ ] **W4** README links to all of: HF Space, Colab notebook, blog, video, training plots, results table.

---

## Current Scoring Estimate

Code complete + tested: **~50 / 100**.
After training curve exists: target **70 / 100**.
After full GRPO + postmortem ablation + demo polish: target **85–90 / 100**.

---

## Phase Tracker

### Phase 0 — Repo Bootstrap (COMPLETE)

- [x] Layout decided (mirrors OpsTwin)
- [x] Theme aligned (#2 Long-Horizon primary, #3.1 Professional Tasks secondary)
- [x] Action space decided (strategic only, no driving)
- [x] Six scoring dimensions decided
- [x] 25 tracks loaded from racetrack-database (`data/tracks/`)
- [x] `data/opponent_pace_calibration.json` populated
- [x] `data/tyre_compound_baseline.json` populated
- [x] `data/track_metadata.json` populated

### Phase 1 — Environment Skeleton (COMPLETE)

- [x] `models.py` — `F1Action`, `F1Observation`, `F1State`
- [x] `server/app.py` — FastAPI singleton via `create_app`
- [x] `client.py` — `F1StrategistEnv(EnvClient)` subclass
- [x] `server/environment.py` — `reset/step/_obs/_exec/_load` fully implemented
- [x] `server/track.py` — CSV loader, curvature, corner detection (25 tracks)
- [x] `server/physics.py` — tyre wear, fuel burn, dirty-air, lap-time formula
- [x] `tests/test_environment.py` — 4 invariant tests
- [x] `tests/smoke_http.py` — HTTP `/reset`, `/step`, `/health` assertions

### Phase 2 — Physics + Scenarios + Reward (COMPLETE)

- [x] `server/physics.py` v1 — full tyre-wear curves per compound, fuel burn, dirty-air
- [x] `server/opponents.py` — rule-based opponents with pace + stint plans
- [x] `server/weather.py` — pre-rolled weather, SC events from seed
- [x] `server/scenarios.py` — 4 hand-authored scenarios (dry sprint, weather, late SC, championship)
- [x] `server/hidden_state.py` — reveal-once accounting
- [x] `server/generator.py` — seed-deterministic procedural variants
- [x] `server/scoring.py` — 6 pure scoring functions, tightened against random exploitation
- [x] `baselines/expert_solver.py` — rule-based solver scoring ≥0.88 on all families
- [x] `baselines/trajectories/*.jsonl` — expert traces generated
- [x] `tests/smoke_all_scenarios.py` — expert ≥0.85, panic ≤0.40
- [x] `tests/test_scoring.py` — 6 unit tests
- [x] `tests/test_scoring_strict.py` — 30 regression tests (random can't exceed ~0.45 mean)
- [x] `tests/test_inference.py` — 47 parse_action robustness tests
- [x] `tests/test_invariants.py` — 41 environment physical/logical invariant tests
- [x] **Total: 130 / 130 tests pass**

### Phase 3 — Training (READY TO RUN)

These require the RTX 5090 server. All code is written and validated.

- [x] `train.py` written — `local-smoke` and `trl` (GRPO) backends complete
- [x] `evaluate.py` written — random / untrained / trained / expert modes distinct
- [x] `inference.py` written — heuristic + transformers backends, robust `parse_action`
- [x] `capture_everything.py` written — SFT seed data generator
- [x] `notebooks/f1_strategist_training_colab.ipynb` — Colab-runnable smoke, correct CLI
- [ ] **Smoke run on RTX 5090** — 50 steps Qwen3-0.6B, confirm curve trends up
- [ ] **Full GRPO run** — 500 steps Qwen3-4B, Unsloth + LoRA
- [ ] **Push checkpoint** to `Deltasthic/f1-strategist-qwen3-4b-grpo`
- [ ] **Produce** `results/training_loss_curve.png`, `results/eval_curve.png`, `results/eval_summary.json`

**Exit criteria:** Reward curve exists. Trained ≥ untrained + 0.20 avg. Checkpoint published.

### Phase 4 — Self-Improvement Story (stretch)

- [x] `server/postmortem.py` — append-only JSONL, failure classification, top-k retrieval
- [x] Postmortem hooks in `environment.py` — appended on `_done = True`
- [x] Memory hints injected into `reset()` observation
- [ ] Ablation eval: base trained vs postmortem-augmented (10 held-out seeds)

### Phase 5 — Polish + Demo + Deploy

- [x] `server/visualizer.py` — matplotlib top-down replay + GIF export + Gradio panel
- [x] `rollout.py` — injects `track_name` into trace so visualizer renders the correct track
- [x] `server/generator.py` — procedural scenario variants for training diversity
- [x] Dockerfile — imageio + pillow + gradio included, `client.py` copied
- [ ] **Deploy HF Space** — `git push hfspace main:main`
- [ ] Smoke-test live Space: `python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space`
- [ ] Record before/after rollout GIFs (`rollout.py --render`)
- [ ] `demo-assets/blog-post.md` — final polish (≥600 words, real numbers)
- [ ] Record video <2 min (see `demo-assets/video-script.md`)
- [ ] Publish blog on HF, video on YouTube; update README links
- [ ] Walk through `PRE_PUSH_CHECKLIST.md`

---

## What to Run Right Now (priority order)

1. **Deploy the HF Space** (see `VERIFY_AND_DEPLOY.md` Phase B):
   ```bash
   huggingface-cli login
   huggingface-cli repo create f1-strategist --type space --space_sdk docker --organization Deltasthic
   git remote add hfspace https://huggingface.co/spaces/Deltasthic/f1-strategist
   git push hfspace main:main
   ```

2. **SSH to the 5090 and start training** (see `VERIFY_AND_DEPLOY.md` Phase C + D):
   ```bash
   # Smoke first
   python train.py --backend trl --model Qwen/Qwen3-0.6B --task dry_strategy_sprint \
       --max-steps 50 --batch-size 1 --grad-accum 8 --output-dir ./grpo_smoke

   # Full run
   python train.py --backend trl --model Qwen/Qwen3-4B --task multi \
       --max-steps 500 --batch-size 1 --grad-accum 32 --output-dir ./grpo_v1
   ```

3. **Evaluate the checkpoint and produce demo assets** (Phase E):
   ```bash
   python evaluate.py --model Deltasthic/f1-strategist-qwen3-4b-grpo \
       --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
       --n-seeds 5 --modes random untrained trained expert \
       --output-json results/eval_summary.json --output-png results/eval_curve.png
   ```

---

## Known Risks

- **Reward hacking via excessive INSPECT actions** — revelation reward fires only on *new* hidden info; fixed.
- **Lap-time model overfit** — per-lap noise + per-seed opponent pace jitter prevents strategy collapse.
- **Long-horizon credit assignment** — dense per-step shaping rewards + delayed-credit pattern address this.
- **Colab T4 OOM on Qwen3-4B** — notebook uses `local-smoke` backend; real training is 5090-only.

---

## AUDIT_NOTES

See `AUDIT_NOTES.md` for full details on every file reviewed, bugs found, fixes applied, and test coverage.
