# GPU Handoff: Tasks for the RTX 5090 Server

This file lets a fresh Claude Code session (or a teammate) on the GPU-enabled Arch Linux
server pick up F1 Strategist training / evaluation work. Read this end-to-end before
starting.

## Project context (one-minute read)

F1 Strategist is an OpenEnv environment for training LLM agents as Formula 1 race
strategists. Submission: **Meta PyTorch OpenEnv Hackathon Grand Finale**, Bangalore,
April 25–26, 2026. Theme #2 (Long-Horizon) primary, Theme #3.1 (Professional Tasks)
secondary. Authors: Shashwat Rajan, Tanish Shitanshu. Repo: `Deltasthicc/f1-strategist`
(GitHub) and `Deltasthic/f1-strategist` (HF Space).

Rubric weights: Environment Innovation 40%, Storytelling 30%, Reward Improvement 20%,
Training Pipeline 10%. Minimums: OpenEnv latest (`openenv-core>=0.2.3`), Colab-runnable
training script, HF Space hosting, blog OR video <2 min.

Target numbers from [`TRAINING.md`](TRAINING.md):

| scenario | random | untrained | trained (target) | expert |
|---|---|---|---|---|
| dry_strategy_sprint | ~0.20 | ~0.35 | **≥0.70** | 0.95 |
| weather_roulette | ~0.20 | ~0.30 | **≥0.65** | 0.93 |
| late_safety_car | ~0.20 | ~0.40 | **≥0.65** | 0.92 |
| championship_decider | ~0.18 | ~0.30 | **≥0.55** | 0.90 |
| **average** | ~0.20 | ~0.34 | **≥0.65** | 0.92 |

## Hardware target

- **OS:** Arch Linux
- **GPU:** 1× RTX 5090 32 GB VRAM (Blackwell, sm_120)
- **CUDA:** 12.8 (required for sm_120 PyTorch wheels)
- **RAM:** ≥ 64 GB recommended for full Qwen3-4B + dataset in memory
- **Disk:** ≥ 100 GB free (checkpoints + logs)

The training stack (Python + Torch + TRL + Unsloth + transformers) is identical to
OpsTwin V3 — same pins, same gotchas. We tested this exact stack on the 5090 in OpsTwin
V3 and it produced the 0.956-average checkpoint.

## Environment setup on the GPU server

```bash
# Clone (or pull if already present)
git clone git@github.com:Deltasthicc/f1-strategist.git
cd f1-strategist

# Python 3.12 is preferred. Arch defaults to 3.13; install 3.12 via pyenv or AUR if needed.
python3 --version    # should be 3.12.x
python3 -m venv .venv
source .venv/bin/activate

# Install torch with CUDA 12.8 wheels FIRST (Blackwell sm_120 requirement)
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch

# Install training extras
pip install -e ".[train,inference,eval]"

# Or, if pyproject doesn't have those extras yet, install manually:
pip install \
  'trl==0.14.0' 'transformers>=4.55.4' 'peft>=0.13.2' \
  'datasets>=3.2.0' 'accelerate>=1.3' \
  'liger-kernel<0.6' 'bitsandbytes>=0.49' \
  matplotlib protobuf python-dotenv

pip install "unsloth[cu128] @ git+https://github.com/unslothai/unsloth.git"

# Hugging Face auth (needed to push the trained model)
huggingface-cli login
# OR: export HF_TOKEN=hf_xxxx

# Sanity-check GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 5090
```

## Pre-flight checks

```bash
# 1. The OpenEnv server starts on this machine
python -m server.app &
sleep 3
curl http://localhost:8000/health
# Expected: {"status": "ok"}
kill %1

# 2. The smoke tests pass
python tests/smoke_http.py
python tests/smoke_all_scenarios.py
python tests/test_environment.py
python tests/test_scoring.py

# 3. The expert solver scores ≥ 0.85 on every scenario
python -m baselines.expert_solver --task dry_strategy_sprint --seed 0
python -m baselines.expert_solver --task weather_roulette --seed 0
python -m baselines.expert_solver --task late_safety_car --seed 0
python -m baselines.expert_solver --task championship_decider --seed 0

# 4. Inference works on Qwen3-0.6B
python inference.py --model Qwen/Qwen3-0.6B --task dry_strategy_sprint --n-episodes 1
```

If any of the above fail, **fix locally first** before opening tmux for training. A failed
smoke test mid-training run is the most common reason for wasted GPU time.

## Training playbook

Use a tmux session so the run survives SSH disconnects.

```bash
tmux new -s train

# Inside tmux:
cd /path/to/f1-strategist
source .venv/bin/activate
set -a && source .env && set +a    # exports HF_TOKEN

# Smoke run first — 50 steps, ~5 min
python train.py \
    --model Qwen/Qwen3-0.6B \
    --task dry_strategy_sprint \
    --max-steps 50 \
    --batch-size 1 --grad-accum 8 \
    --output-dir ./grpo_smoke 2>&1 | tee training_smoke.log

# Inspect smoke
python scripts/plot_training_curve.py --run-dir ./grpo_smoke

# If smoke is good, full run — ~3-4h
python train.py \
    --model Qwen/Qwen3-4B \
    --task multi \
    --max-steps 500 \
    --batch-size 1 --grad-accum 32 \
    --warm-start-sft ./sft_checkpoints_v1/checkpoint-200 \
    --output-dir ./grpo_v1 2>&1 | tee training_v1.log

# Detach: Ctrl-b d
# Reattach: tmux attach -t train
```

Use `nvidia-smi -l 1` in another terminal to monitor. Expected utilisation: 90-100% on
the 5090, ~28 GB VRAM in use, ~50 W idle / 580 W peak power.

## Push checkpoint

```bash
python scripts/push_checkpoint.py \
    --checkpoint ./grpo_v1/checkpoint-best \
    --repo Deltasthic/f1-strategist-qwen3-4b-grpo
```

The push uploads ~8 GB; allow 5-10 min depending on uplink speed. The HF Hub UI will
show progress.

## Evaluate

```bash
python evaluate.py \
    --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
    --n-seeds 5 \
    --modes random untrained trained expert \
    --output-json results/eval_summary.json \
    --output-png results/eval_curve.png
```

Should take ~30-45 min for the full grid (4 scenarios × 4 modes × 5 seeds × ~30 s/episode).

Inspect the bar chart:
```bash
xdg-open results/eval_curve.png
# Or scp to your laptop
```

If the trained bars don't beat the untrained bars by ≥ 0.20 average, **iterate**:
1. Check evaluation logs for malformed actions
2. Inspect a couple of trained-mode rollouts via `python rollout.py --mode trained --verbose`
3. If the model is emitting valid actions but losing on strategy, more training steps may help
4. If the model is emitting malformed actions, fix the chat template or `max_new_tokens`

## Postmortem ablation (Phase 4)

Once Person 1 has the postmortem retrieval wired:

```bash
# Run 1: trained policy without memory injection
python evaluate.py --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car \
    --n-seeds 10 --no-memory \
    --output-json results/ablation_no_memory.json

# Run 2: same model with memory hints injected
python evaluate.py --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car \
    --n-seeds 10 --use-memory \
    --output-json results/ablation_with_memory.json

# Diff
python scripts/diff_ablation.py \
    results/ablation_no_memory.json \
    results/ablation_with_memory.json \
    --output results/ablation.md
```

Expected: memory-augmented policy averages 0.05–0.10 higher than base.

## Final-day commands (April 26 morning)

```bash
# 1. Re-run eval to lock numbers
python evaluate.py ... --output-json results/eval_summary.json

# 2. Generate the visualizer GIFs
python rollout.py --task weather_roulette --seed 7 --mode untrained --render
python rollout.py --task weather_roulette --seed 7 --mode trained --render
mv captures/*.gif demo-assets/

# 3. Render the bar chart with final numbers
python scripts/plot_training_curve.py --output results/training_loss_curve.png

# 4. Commit results
git add results/ demo-assets/
git commit -m "final: lock eval numbers and demo GIFs"
git push origin main

# 5. Push to HF Space (triggers rebuild)
git push hfspace main:main

# 6. Smoke-test the live Space
python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space

# 7. Walk through PRE_PUSH_CHECKLIST.md item by item
```

## What can go wrong on the 5090 + Arch combo

- **`nvidia-smi` reports the GPU but `torch.cuda.is_available()` is False.** Almost always
  the wrong CUDA wheel. Reinstall via `pip install --extra-index-url https://download.pytorch.org/whl/cu128 --force-reinstall torch`.
- **Kernel module mismatch after a system update.** Arch ships fresh nvidia drivers
  weekly. `pacman -Syu` can break the user-space CUDA toolkit. Pin `nvidia` and
  `nvidia-utils` to a known-good version, OR run training inside a Docker container
  with `nvidia/cuda:12.8.0-devel-ubuntu22.04` as base.
- **OOM at step 1 with batch=1, grad_accum=32.** Verify Unsloth gradient checkpointing
  is on (`use_gradient_checkpointing="unsloth"`). If still OOMing, drop the model to
  Qwen3-1.7B for the run.
- **`pip install unsloth` fails on Arch.** Use `pip install "unsloth[cu128]
  @ git+https://github.com/unslothai/unsloth.git"`. The PyPI version sometimes lags the
  GH version on CUDA support.
- **Training freezes for 30+ seconds at startup.** Normal. TRL pre-compiles vLLM rollout
  workers; first step is always slow.

## Logs and artefacts

Persist everything:
```bash
# Always tee to a log file
python train.py ... 2>&1 | tee training_v1.log

# Persist intermediate checkpoints separately
mv ./grpo_v1/checkpoint-* /backup/grpo_v1/

# After successful publish, keep ONLY the final checkpoint locally
rm -rf grpo_v1/checkpoint-* && cp -r /backup/grpo_v1/checkpoint-best ./grpo_v1/
```

## Pre-flight cheatsheet (laminate this)

```bash
# Resume work
tmux attach -t train
source .venv/bin/activate
set -a && source .env && set +a

# Quick health check
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python tests/smoke_http.py

# Plot whatever's running
python scripts/plot_training_curve.py --run-dir ./grpo_v1

# Push when ready
python scripts/push_checkpoint.py
```
