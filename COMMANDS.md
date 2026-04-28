# Command Reference

All commands run from the repo root. On the SSH server, prefix with the correct
virtualenv activation (`source .venv/bin/activate` or `conda activate f1`).

---

## Inference (local, no GPU needed)

Run the heuristic/expert policy on any scenario and see lap-by-lap decisions:

```bash
# Default: 3 episodes of dry_strategy_sprint, verbose
python inference.py

# Change scenario
python inference.py --task weather_roulette
python inference.py --task late_safety_car
python inference.py --task championship_decider
python inference.py --task virtual_safety_car_window
python inference.py --task tyre_cliff_management

# More episodes, different seed
python inference.py --task weather_roulette --n-episodes 5 --seed 7

# Quiet mode (just final scores, no lap-by-lap)
python inference.py --task dry_strategy_sprint --quiet

# Save the trace to a JSONL file
python inference.py --task weather_roulette --save captures/my_run.jsonl

# Run your actual trained GRPO checkpoint (SSH server)
python inference.py --model ./grpo_v1 --task weather_roulette --n-episodes 5
python inference.py --model ./grpo_v2 --task championship_decider --n-episodes 5
```

---

## Evaluation

Run the full held-out evaluation across all scenario families and generate charts.

```bash
# Quick eval — heuristic baseline, 3 seeds per task
python evaluate.py

# Full eval with all modes, 5 seeds (takes a few minutes locally)
python evaluate.py --n-seeds 5

# Evaluate only specific tasks
python evaluate.py --tasks dry_strategy_sprint weather_roulette

# Evaluate your trained checkpoint (do this on SSH server)
python evaluate.py --model ./grpo_v1 --n-seeds 5

# Compare grpo_v2 vs v1 vs expert
python evaluate.py --model ./grpo_v2 --n-seeds 5 \
  --output-json results/eval_grpo_v2.json \
  --output-png results/eval_curve_v2.png

# Run only specific modes
python evaluate.py --modes random untrained trained expert
```

---

## Visualisations

```bash
# Race story chart — before/after for one scenario+seed
python scripts/plot_race_story.py --task weather_roulette --seed 7
python scripts/plot_race_story.py --task late_safety_car --seed 3
python scripts/plot_race_story.py --task championship_decider --seed 0

# Track grid — expert policy across 8 circuits
python scripts/render_track_grid.py

# Side-by-side GIF comparison (untrained vs trained)
python scripts/render_compare.py --task weather_roulette --seed 7

# Training loss curve
python scripts/plot_training_curve.py

# Run everything at once (regenerates all results/ and captures/)
python scripts/run_full_demo.py
```

---

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_scoring_strict.py -v
python -m pytest tests/test_invariants.py -v
python -m pytest tests/test_inference.py -v

# Run fast (no output capture)
python -m pytest tests/ -x --tb=short

# Count: should be 164 passed
python -m pytest tests/ -q
```

---

## Server (local API)

```bash
# Start the FastAPI server on port 8000
uvicorn server.app:app --port 8000 --reload

# In another terminal: smoke test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"task": "weather_roulette", "seed": 7}'
```

---

## Training on SSH server (RTX 5090)

Copy-paste these exactly — no inline comments to cause bash errors.

### grpo_v2: fine-tune from v1 checkpoint, 800 steps

```bash
python train.py \
  --model ./grpo_v1 \
  --backend trl \
  --max-steps 800 \
  --batch-size 2 \
  --grad-accum 16 \
  --learning-rate 2e-6 \
  --output-dir ./grpo_v2 \
  --task multi
```

### grpo_v1 from scratch (original run, for reference)

```bash
python train.py \
  --model Qwen/Qwen3-4B \
  --backend trl \
  --max-steps 500 \
  --batch-size 1 \
  --grad-accum 32 \
  --learning-rate 5e-6 \
  --output-dir ./grpo_v1 \
  --task multi
```

### Smoke test the training loop without a GPU (no model weights needed)

```bash
python train.py --backend local-smoke --max-steps 20
```

---

## Hugging Face deployment

```bash
# Push model checkpoint to HF Hub
huggingface-cli upload Deltasthic/f1-strategist ./grpo_v1 grpo_v1 --repo-type model

# Push the full Space (includes server + README)
# Make sure HF_TOKEN is set in your environment
huggingface-cli login
git remote add hf https://huggingface.co/spaces/Deltasthic/f1-strategist
git push hf main

# Or if the Space already exists and you just want to push latest code
git push hf main --force
```

---

## Evaluate from the deployed HF Space

```bash
# Run inference against the live HF Space endpoint
python inference.py \
  --base-url https://deltasthic-f1-strategist.hf.space \
  --model heuristic \
  --task weather_roulette \
  --n-episodes 3
```

---

## Common one-liners

```bash
# Quick sanity check: does the env even work?
python -c "
from server.environment import F1StrategistEnvironment
env = F1StrategistEnvironment()
obs = env.reset(task='dry_strategy_sprint', seed=0)
print('reset OK, lap', obs.current_lap, 'P', obs.ego_position)
from models import F1Action
obs2 = env.step(F1Action(command='STAY_OUT'))
print('step OK, lap', obs2.current_lap, 'score', obs2.score)
"

# Check all scenario names
python -c "from server.scenarios import SCENARIOS; print(list(SCENARIOS))"

# Check track names
python -c "from server.track import list_tracks; print(list_tracks())"
```
