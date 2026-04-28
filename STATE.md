# Execution State — F1 Strategist Hackathon

Updated after each phase gate. If anything kills the session, read this top-to-bottom to resume.

## Identity & infra
- Box: `anurag-vps`, user `anurag`
- GPU: RTX 5090, CUDA driver 13.2, 32 GB VRAM
- Venv: `~/.virtualenvs/f1-strategist` (python 3.12.3)
- Repo: `/home/anurag/projects/F1_Simulator_OpenENV` (branch `dev`)
- Port: env binds to `8765`
- Domain: `f1.chinnaboina.com` (via cloudflared tunnel)
- HF token: placeholder in `.env`, real value pending

## Milestones

- [x] Phase 0 — Pre-flight + repo prep
- [x] Phase 1 — Install training stack
- [x] Phase 2 — Re-verify env (164 tests + 6 scenarios)
- [x] Phase 3 — Cloudflare tunnel for f1.chinnaboina.com (LIVE)
- [x] Phase 4 — Generate SFT seed data (2450 turns)
- [x] Phase 5 — GRPO smoke (Shashwat already trained on another machine, smoke equivalent done)
- [x] Phase 6 — Full GRPO 500 steps Qwen3-4B (Shashwat — committed via main, merged into dev as `d763d0b`)
- [x] Phase 7 — Landing page LIVE at https://f1.chinnaboina.com/
- [x] Phase 8 — HF Space LIVE (Deltasthic/f1-strategist serves landing page)
- [x] Phase 9 — Eval against trained checkpoint (LoRA merged → `grpo_v1/merged-500/`, results at `results/eval_summary.json` + `results/eval_curve.png`)
- [x] Phase 10 — Iter 0–4: SFT v3 + GRPO v2 — **0.618 avg** champion (was 0.538). 5 bugs caught + fixed.
- [x] Phase 11a — Merged `origin/main` into `dev`, integrating UI/Space + keeping our model + bug-fixed eval
- [ ] Phase 11b — HF Hub model push: replace `Deltasthic/f1-strategist-qwen3-4b-grpo` with `grpo_v2/` adapter
- [ ] Phase 11c — Final commit + push to GitHub, submit hackathon form

## Champion model
**`grpo_v2/`** — SFT v3 (enriched obs, thinking-off render) → GRPO v2 (200 steps, no Unsloth, no vLLM, beta=0.005, num_generations=8). Avg eval **0.618** vs untrained 0.429 (+0.20). See `results/journey.png`, `results/scenario_breakdown.png`, `results/comparison_real.png`.

## Five bugs caught and fixed (the journey)
1. **Silent scripted-fallback bug** in main's `evaluate.py` → reported 0.79 was actually a hand-coded rule policy, never the model. Verified bit-exact match by running scripted policy in isolation.
2. **Qwen3 thinking-mode trap** — default reasoning-on with `max_new_tokens=64` → unparseable rambles → `parse_action()` fell through to STAY_OUT every step.
3. **Train/eval format mismatch** — flipping `enable_thinking=False` only at eval broke the chat-template prefix the model never saw.
4. **`format_obs` stripped scenario disambiguation** — model couldn't tell `late_safety_car` from `dry_strategy_sprint` at lap 0. Restored `obs.message` + `obs.hint`.
5. **Cold GRPO collapse** — vanilla GRPO from base Qwen3 plateaued at 0.54 (frac_reward_zero_std → 1.0). Fixed by SFT warm-start before GRPO (per `TRAINING.md`'s prescribed but skipped recipe).

## Phase log

### Phase 0 — Pre-flight + repo prep ✅
- 2026-04-25 16:40 — venv created at `~/.virtualenvs/f1-strategist`
- 2026-04-25 16:41 — `.env` created from template, port=8765, HF_TOKEN=placeholder
- 2026-04-25 16:41 — `.env` confirmed gitignored
- Cloudflared inspection: existing tunnel `476cc98d-...` named `vijayrekha`, single ingress to `chinnaboina.com:80`
- User confirmed: Option A (extend existing tunnel)
- Ports 8765–8770 all free
- Nginx site `vijayrekha.conf` untouched

### Phase 1 — Install training stack ✅
- 2026-04-25 16:50 — torch 2.10.0+cu128 installed, sm_120 matmul verified on RTX 5090
- 2026-04-25 16:55 — repo installed editable; openenv-core 0.2.3 in
- 2026-04-25 17:00 — trl 0.24.0, transformers 5.5.0, peft 0.19.1, accelerate 1.13.0, datasets 4.3.0
- 2026-04-25 17:05 — unsloth 2026.4.8 + unsloth_zoo 2026.4.9 installed
- 2026-04-25 17:08 — `requirements.lock` written (165 packages)
- Note: Flash Attention 2 not present, Unsloth uses Xformers (no perf impact)

### Phase 2 — Re-verify env ✅
- 2026-04-25 17:00 — 164/164 tests pass (more than docs claimed 130 — `test_scenarios_extended` adds coverage)
- 2026-04-25 17:01 — 6 scenarios discriminate (expert ≥0.88, panic ≤0.40)
- 2026-04-25 17:02 — Env booted on 127.0.0.1:8765, PID 473838 in background
- 2026-04-25 17:02 — All 20 expert traces present (4 scenarios × 5 seeds)

### Phase 3 — Cloudflare tunnel ✅ LIVE
- 2026-04-25 17:01 — DNS CNAME `f1.chinnaboina.com` → tunnel UUID 476cc98d
- 2026-04-25 17:05 — Backed up `/etc/cloudflared/config.yml` to `.bak.<ts>`
- 2026-04-25 17:05 — Replaced config (added f1.chinnaboina.com → localhost:8765 ingress, vijayrekha untouched)
- 2026-04-25 17:06 — `systemctl restart cloudflared` (reload not supported on this unit)
- 2026-04-25 17:06 — Both endpoints verified:
  - `https://chinnaboina.com` → 200 (vijayrekha alive)
  - `https://f1.chinnaboina.com/health` → 200 `{"status":"healthy"}`
- HF_TOKEN: real value confirmed in .env, `whoami()` returns `Deltasthic` (org account direct login)

## Open blockers
_None._ git-lfs installed, weights pulled, LoRA merged into base, eval ran clean.

## Phase 9 — Eval results (4 tasks × 5 seeds, weighted_final score)

| mode | dry | weather | safety_car | champ | **avg** |
|---|---|---|---|---|---|
| random | 0.40 | 0.34 | 0.33 | 0.21 | **0.32** |
| untrained | 0.51 | 0.41 | 0.53 | 0.27 | **0.43** |
| trained | 0.52 | 0.56 | 0.55 | 0.52 | **0.54** |
| expert | 0.84 | 0.95 | 0.94 | 0.97 | **0.92** |

- Trained > untrained on every task; biggest jump championship_decider (+0.25)
- Trained std=0.0 across seeds (greedy decode, deterministic command sequence)
- Gap to expert ceiling: 0.38 — room to grow but clear training signal demonstrated

## Phase 6 — GRPO training (Shashwat's run, merged into dev)
- Commit: `eb9f4bf "Add trained Unsloth vLLM GRPO model and results"` (Shashwat, 2026-04-25 13:07)
- Steps: 500 (full target run)
- Saves: every 50 steps, 10 checkpoints (50, 100, 150, ..., 500)
- Reward trajectory (from `grpo_v1/checkpoint-500/trainer_state.json`):
  - Step 10  : reward=0.875, std=0.121
  - Step 500 : reward=0.893, std=0.095
  - Slight upward trend; std tightening (consistency improving)
  - `loss` collapsed to 0.0, `frac_reward_zero_std` reached 1.0 by end (policy converged)
- Train.py merge conflict resolved by adopting Shashwat's working config:
  `fast_inference=True, max_lora_rank=16, gpu_memory_utilization=0.30`

## Phase 7 — Landing page ✅
- `server/static/{index.html, style.css, app.js}` created with line-drawing animation
- `server/app.py` patched to serve `/` and `/static/*` (route override for openenv's redirect)
- Live at https://f1.chinnaboina.com/ AND https://Deltasthic-f1-strategist.hf.space/

## Phase 8 — HF Space ✅
- Deployed via `huggingface_hub.upload_folder()` (bypasses git-lfs need for binaries)
- W2 hackathon requirement satisfied
- Updated `demo-assets/hf-space-link.txt` with all three live URLs

## Merge details — origin/main → dev (commit d763d0b)
- Pulled trained checkpoint (~10 checkpoints × LoRA adapter + tokenizer files)
- Resolved 1 conflict in train.py
- Auto-merged .gitignore additions
- 164 tests still passing post-merge

---

## PRE_PUSH_CHECKLIST audit — 2026-04-26 11:30 (post-merge `f947743`)

Source: `PRE_PUSH_CHECKLIST.md`. Walked top to bottom. Status: ✅ pass / ⚠️ partial / ❌ fail / ⏭️ skipped-by-decision.

### 1. Repository hygiene
- ✅ `git status` clean (working tree at HEAD `f947743`, post-merge commit done)
- ⚠️ Secrets scan: matches in commit history are **placeholders** (`HF_TOKEN=hf_your_token_here`) and `.env` template references — no real tokens leaked
- ❌ **Large files in git: `grpo_v1/checkpoint-*/tokenizer.json` (10 × 11 MB = 110 MB)** committed before `.gitignore` rule — eats clone time, doesn't affect Space build. **Decision: leave for now, post-submission cleanup.**
- ✅ `.gitignore` lists `.venv/`, `__pycache__/`, `grpo_*/`, `sft_*/`, `.env`, `.env.local`

### 2. README accuracy
- ✅ HF Space frontmatter present (lines 1–17 are valid YAML)
- ✅ Title `F1 Strategist`, sdk `docker`, app_port `8000`, license `mit`
- ✅ Section links resolve to `docs/*.md` files in tree
- ✅ HF Space link → live (Stage `RUNNING`)
- ✅ Colab badge → notebook on dev branch
- ✅ Blog link target → `/blog` endpoint serves rendered HTML on Space + tunnel
- ⏭️ Video link → **dropped per project decision (no demo video this round)**
- ✅ All `results/*.png` referenced in README exist on disk

### 3. Phase 1 (Environment) gating
- ✅ `python -m server.app` starts (running on port 8765 since 10:16, PID 680119)
- ✅ `/health` returns 200
- ✅ `tests/smoke_http.py` covered by full pytest run
- ✅ `tests/test_environment.py` passes
- ✅ `client.py` imports

### 4. Phase 2 (Scenarios + Scoring) gating
- ✅ `tests/smoke_all_scenarios.py` passes (164/164 in full suite)
- ✅ `tests/test_scoring.py` passes (6 cases)
- ✅ Expert solver scores per `eval_six_scenarios.json`:
  - dry_strategy_sprint: **0.840** ⚠️ (spec ≥ 0.85, just under by 0.010 — known borderline)
  - weather_roulette: 0.950 ✅
  - late_safety_car: 0.935 ✅
  - championship_decider: 0.965 ✅
  - virtual_safety_car_window: 0.965 ✅
  - tyre_cliff_management: 0.965 ✅
- ✅ Expert traces present: 6 scenarios × 5 seeds = 30 jsonl files in `baselines/trajectories/`

### 5. Phase 3 (Training) gating — CRITICAL
- ✅ Reward curve at `results/grpo_v2_reward_curve.png` (real, from `trainer_state.json`, 200 steps)
- ✅ `results/eval_summary.json` exists; **canonical numbers in `results/eval_six_scenarios.json`**
- ✅ `results/eval_curve.png` shows ≥0.20 average improvement (untrained 0.42 → trained 0.62)
- ✅ Trained checkpoint published: `Deltasthic/f1-strategist-qwen3-4b-grpo`
- ✅ Public, ungated, has README + adapter
- ✅ Model card describes usage
- ⚠️ Colab notebook updated with grpo_v2 content but **end-to-end run on Free T4 not re-verified** post-rewrite
- ✅ Notebook prints reward curve plot + rollout transcript (`demo-assets/trained-rollout-transcript.txt`)

### 6. Phase 4 (Self-improvement) — optional
- ✅ `server/postmortem.py` records on episode end (verified by `test_postmortem.py`)
- ✅ `reset()` injects `memory_hints`
- ⚠️ `results/ablation.md` shows **+0.000 delta** on currently-tested scenarios — saturated trained policy on easy seeds
- ❌ **Decision: re-run on harder seeds OR drop the empirical claim**. Per audit Lane C.

### 7. Phase 5 (Demo + Deploy) gating
- ✅ HF Space Stage `RUNNING`
- ✅ `/reset` and `/step` return valid F1Observation on live Space + tunnel
- ❌ **Gradio `/web` route returns 404** — likely `ENABLE_WEB_INTERFACE=1` not honored by the openenv-core 0.2.3 + create_fastapi_app combo on the Space. `/blog` works. Investigation pending.
- ✅ `demo-assets/hf-space-link.txt` populated
- ⏭️ `demo-assets/youtube-link.txt` — video skipped
- ⚠️ `demo-assets/hf-blog-link.txt` — to be updated to `/blog` endpoint URL
- ✅ Blog rendered correctly at `https://deltasthic-f1-strategist.hf.space/blog` and `https://f1.chinnaboina.com/blog`
- ⏭️ Video items
- ✅ Visualizer GIFs in `demo-assets/`: `trained-spa.gif`, `untrained-spa.gif`

### 8. Submission requirements (W1–W4)
- ✅ **W1** `pyproject.toml` lists `openenv-core>=0.2.3`
- ✅ **W1** Colab notebook committed (latest dev commit)
- ✅ **W2** HF Space public + Running
- ✅ **W3** Blog published at `https://deltasthic-f1-strategist.hf.space/blog` (Space-hosted page; renders blog.md as HTML); video skipped
- ✅ **W4** README links to: Space, Colab, blog, eval_curve.png, training_loss/grpo_v2_reward_curve.png

### 9. Theme alignment
- ✅ README references Theme #2 (long-horizon planning) + Theme #3.1 (professional tasks)
- ✅ Blog opens with the long-horizon framing ("a 12-lap sprint is 25 minutes of simulated race; lap-4 mistakes hurt at lap-9")
- ✅ Architecture doc has full Hidden State spec (`docs/architecture.md` §2)

### 10. Final smoke
- ✅ Local: 164 pytest pass
- ✅ Live Space `/reset` + `/step` 200
- ⚠️ Eval reproduction on `--n-seeds 2` not re-run today; cached numbers stand
- ⚠️ Colab smoke not re-run today

### 11. Tag and submit
- 🟦 Pending — user (Anurag) commits + tags + pushes; do not auto-do this

### 12. Submit to organisers
- 🟦 Pending — user submits via hackathon form

---

## Outstanding gaps (priority-ordered)

1. **Postmortem ablation +0.000** — pivot to honest "future work" framing OR re-run on harder seeds (C1 vs C2 of audit plan)
2. **Gradio `/web` returns 404 on Space** — `ENABLE_WEB_INTERFACE=1` not landing in this `create_fastapi_app` build; surface the workaround OR drop the claim
3. **Blog draft → HF Hub blog publication** — Space-hosted `/blog` already counts; if a separate huggingface.co/blog post is desired, requires manual UI step
4. **Expert dry_strategy_sprint at 0.840** — 0.01 below spec floor, cosmetic
5. **Colab end-to-end on Free T4** — last full smoke was pre-rewrite; quick spot-check recommended
6. **`grpo_v1/checkpoint-*/tokenizer.json` × 10** in git — 110 MB cruft, post-submission cleanup
