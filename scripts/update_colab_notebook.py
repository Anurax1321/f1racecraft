"""Rewrite the F1 strategist Colab notebook in-place with grpo_v2 content.

Idempotent: matches cells by id and replaces source. Preserves cell IDs and
notebook metadata.
"""

from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "f1_strategist_training_colab.ipynb"

UPDATES: dict[str, str] = {}

UPDATES["cell_title"] = """\
# F1 Strategist — SFT → GRPO Training & Evaluation

**Hackathon:** Meta PyTorch OpenEnv Grand Finale · April 25–26 2026 · Bangalore
**Team:** Shashwat Rajan · Tanish Shitanshu · Org: [Deltasthic](https://huggingface.co/Deltasthic)
**Space:** [Deltasthic/f1-strategist](https://huggingface.co/spaces/Deltasthic/f1-strategist)
**Model:** [Deltasthic/f1-strategist-qwen3-4b-grpo](https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo)
**GitHub:** [Deltasthicc/F1_Simulator_OpenENV](https://github.com/Deltasthicc/F1_Simulator_OpenENV) (branch `dev`)
**Live:** https://f1.chinnaboina.com/

---

## What this notebook does

| Section | What it does | GPU needed? |
|---------|-------------|-------------|
| 1 | Install deps + clone repo (`dev` branch) | No |
| 2 | Environment smoke test (reset → step → score) | No |
| 3 | 6-scenario eval results (random / untrained / **grpo_v2** / expert) | No |
| 4 | Real GRPO v2 training reward curve from `trainer_state.json` | No |
| 5 | Iteration journey: random → SFT → GRPO (the 5-bug story) | No |
| 6 | Track generalization grid (8 circuits) | No |
| 7 | *(Optional)* Reproduce the SFT → GRPO recipe yourself | Yes — T4/A100 |

Sections 1–6 run on Colab CPU. Section 7 needs a T4+ (Section 7 reproduces the
SFT warm-start + GRPO recipe that produced the shipped `grpo_v2` checkpoint).

---

## What the model does

The LLM is the **race engineer on the pit wall**. Each step is one strategic
decision (typically one lap). It sees the current race state — lap, position,
tyre health, fuel, weather (probability cone, not point estimate), opponent
gaps, scenario briefing — and emits one command from a 21-command vocabulary.

**Six-dimensional reward (deterministic Python, no LLM judge):**
- Race result (×0.35): final position vs. grid + target
- Strategic decisions (×0.20): pit-window timing, investigation discipline
- Tyre management (×0.15): FIA two-compound rule, cliff avoidance
- Fuel management (×0.10): finishing without running dry or wasting kg
- Comms quality (×0.10): radio call before key actions
- Operational efficiency (×0.10): no invalid commands, no panic pits

**Long-horizon design.** A bad lap-4 pit call manifests in a lap-9 outcome —
opponents catching up, tyre cliff hitting before the chequered. Theme #2.

**Honest headline.** Trained model averages **0.62** across 6 scenarios × 5 seeds
(+0.20 over untrained Qwen3-4B). On weather (0.97) it edges the rule-based
expert (0.95). Full bug-discovery story in [`blog.md`](https://github.com/Deltasthicc/F1_Simulator_OpenENV/blob/dev/blog.md).
"""

UPDATES["cell_sec1_header"] = """\
## Section 1 — Install dependencies and clone the repo
"""

UPDATES["cell_install"] = """\
# Install runtime dependencies (CPU-only path — Sections 1–6)
%pip install -q "openenv-core>=0.2.3" fastapi uvicorn "pydantic>=2.0" \\
                numpy pandas matplotlib pillow imageio tqdm \\
                "huggingface_hub>=0.24" peft transformers accelerate

import pathlib, subprocess, sys, os

REPO_URL    = "https://github.com/Deltasthicc/F1_Simulator_OpenENV.git"
REPO_NAME   = "F1_Simulator_OpenENV"
REPO_BRANCH = "dev"   # current ship branch

if not pathlib.Path(REPO_NAME).exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL],
        check=True,
    )
    print(f"Cloned {REPO_NAME} ({REPO_BRANCH})")
else:
    subprocess.run(["git", "-C", REPO_NAME, "fetch", "--depth", "1", "origin", REPO_BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_NAME, "checkout", REPO_BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_NAME, "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
    print(f"Repo present — synced to origin/{REPO_BRANCH}")

ROOT = pathlib.Path(REPO_NAME).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print(f"Working directory: {ROOT}")
"""

UPDATES["cell_sec2_header"] = """\
## Section 2 — Environment smoke test

Verify `reset()`, `step()`, and the deterministic reward signal work end-to-end
without booting the FastAPI server.
"""

UPDATES["cell_smoke_test"] = """\
from server.environment import F1StrategistEnvironment
from models import F1Action

env = F1StrategistEnvironment()
obs = env.reset(task="weather_roulette", seed=0)

print(f"Reset — task=weather_roulette  seed=0")
print(f"  Total laps : {obs.total_laps}")
print(f"  Start pos  : P{obs.ego_position}")
print(f"  Compound   : {obs.ego_tyre_compound}")
print(f"  Fuel       : {obs.ego_fuel_remaining_kg:.1f} kg")
print()

# A short scripted sequence to demonstrate stepping
DEMO_ACTIONS = [
    "REQUEST_FORECAST",
    "STAY_OUT",
    "STAY_OUT",
    'RADIO_DRIVER "Box now for inters — rain coming."',
    "PIT_NOW inter",
]
print(f"{'Lap':>4}  {'Pos':>3}  {'Compound':>8}  {'Health':>7}  {'Fuel':>6}  Action")
print("-" * 78)
for act in DEMO_ACTIONS:
    print(f"{obs.current_lap:>4}  P{obs.ego_position:<2}  {obs.ego_tyre_compound:>8}"
          f"  {obs.ego_tyre_health_pct:>6.1f}%  {obs.ego_fuel_remaining_kg:>5.1f}kg  {act}")
    obs = env.step(F1Action(command=act))

print(f"\\nAfter 5 laps: P{obs.ego_position}  compound={obs.ego_tyre_compound}  "
      f"health={obs.ego_tyre_health_pct:.1f}%  score={obs.score:.3f}")
print("\\nSmoke test PASSED.")
"""

UPDATES["cell_run_full_episode"] = """\
# Run a full episode with the rule-based expert ('heuristic') and print the lap log.
# Heuristic is fine for the smoke walkthrough; the LLM model is loaded in Section 7.
from inference import run_inference

scores = run_inference(
    model="heuristic",
    task="weather_roulette",
    n_episodes=1,
    verbose=True,
    seed=0,
)
print(f"\\nFinal score: {scores[0]:.3f}")
"""

UPDATES["cell_sec3_header"] = """\
## Section 3 — Held-out evaluation: 4 policies × 6 scenarios

Real LLM forward pass through the env at every step. The trained model is
**`grpo_v2`** — Qwen3-4B + LoRA, SFT v3 warm-start (4,900 expert turns) then
200 GRPO steps with `num_generations=8`, `beta=0.005`, `enable_thinking=False`.

`results/eval_six_scenarios.json` is the canonical numbers file.
"""

UPDATES["cell_eval_table"] = """\
import json, pathlib

# Prefer the 6-scenario file; fall back to the 4-scenario one if missing
six_path  = pathlib.Path("results/eval_six_scenarios.json")
four_path = pathlib.Path("results/eval_summary.json")
summary_path = six_path if six_path.exists() else four_path

with summary_path.open() as f:
    summary = json.load(f)

policies  = ["random", "untrained", "trained", "expert"]
scenarios = list(next(iter(summary.values())).keys())
print(f"Loaded {summary_path.name} — {len(scenarios)} scenarios × {len(policies)} policies\\n")

col_w = 14
header = f"{'Scenario':<32}" + "".join(f"{p.upper():>{col_w}}" for p in policies)
print(header)
print("-" * len(header))
for sc in scenarios:
    row = f"{sc:<32}"
    for p in policies:
        v = summary.get(p, {}).get(sc, {}).get("mean", 0.0)
        row += f"{v:>{col_w}.3f}"
    print(row)

print()
print(f"{'AVERAGE':<32}" + "".join(
    f"{(sum(summary.get(p, {}).get(sc, {}).get('mean', 0.0) for sc in scenarios)/len(scenarios)):>{col_w}.3f}"
    for p in policies
))
print()
print("Delta (trained − untrained):")
for sc in scenarios:
    tr = summary.get("trained",   {}).get(sc, {}).get("mean", 0.0)
    un = summary.get("untrained", {}).get(sc, {}).get("mean", 0.0)
    print(f"  {sc:<32} +{tr - un:+.3f}")
"""

UPDATES["cell_eval_chart"] = """\
import matplotlib.pyplot as plt
import numpy as np

# F1 palette (matches landing-page CSS)
PALETTE = {
    "random":    "#666666",
    "untrained": "#ef8a17",
    "trained":   "#e10600",
    "expert":    "#ffd60a",
}
SC_LABELS = {
    "dry_strategy_sprint":           "Dry sprint",
    "weather_roulette":              "Weather",
    "late_safety_car":               "Safety car",
    "championship_decider":          "Champ",
    "virtual_safety_car_window":     "VSC win",
    "tyre_cliff_management":         "Tyre cliff",
}

x = np.arange(len(scenarios))
w = 0.18
fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="#0a0a0a")
ax.set_facecolor("#0f0f0f")

for i, pol in enumerate(policies):
    vals = [summary.get(pol, {}).get(sc, {}).get("mean", 0.0) for sc in scenarios]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=pol, color=PALETTE[pol],
                  alpha=0.92, edgecolor="#0a0a0a", linewidth=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                color="#e8e8e8", fontfamily="monospace")

ax.set_xticks(x)
ax.set_xticklabels([SC_LABELS.get(sc, sc) for sc in scenarios], color="#e8e8e8", fontsize=10)
ax.set_ylim(0, 1.12)
ax.axhline(0.50, color="#666", ls="--", lw=0.6, alpha=0.6, zorder=1, label="0.50 baseline")
ax.set_ylabel("Mean weighted_final score (0–1)", color="#888", fontsize=10)
ax.set_title("F1 Strategist — Held-out eval (5 seeds × 6 scenarios)",
             color="#e8e8e8", fontsize=12, pad=14)
ax.tick_params(colors="#666")
for spine in ax.spines.values():
    spine.set_edgecolor("#1e1e1e")
ax.yaxis.grid(True, color="#1e1e1e", zorder=0)
leg = ax.legend(fontsize=9, framealpha=0, labelcolor="#e8e8e8",
                loc="upper right", facecolor="none")
plt.tight_layout()
plt.savefig("results/colab_eval_curve.png", dpi=140, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Saved to results/colab_eval_curve.png")
"""

UPDATES["cell_sec4_header"] = """\
## Section 4 — GRPO v2 training reward curve

The reward curve from the actual `grpo_v2` training run — 200 GRPO steps from
the SFT v3 base, plotted from `grpo_v2/checkpoint-200/trainer_state.json`. Every
point is a real logging step. Peak ~0.93 around step 30, then plateau.
"""

UPDATES["cell_training_curve"] = """\
from IPython.display import Image as IPyImage, display
import pathlib, subprocess

# Prefer the new grpo_v2 curve; if not present (older clone) regenerate
v2_curve = pathlib.Path("results/grpo_v2_reward_curve.png")
old_curve = pathlib.Path("results/training_loss_curve.png")

if not v2_curve.exists() and pathlib.Path("scripts/make_grpo_reward_curve.py").exists():
    subprocess.run(["python", "scripts/make_grpo_reward_curve.py"], check=False)

if v2_curve.exists():
    display(IPyImage(str(v2_curve), width=920))
    print(f"Real GRPO v2 reward curve: {v2_curve}")
elif old_curve.exists():
    display(IPyImage(str(old_curve), width=920))
    print(f"Showing legacy curve from {old_curve} (run scripts/make_grpo_reward_curve.py for the v2 curve)")
else:
    print("No training curve found. Run Section 7 to generate one.")
"""

UPDATES["cell_sec5_header"] = """\
## Section 5 — Iteration journey & before/after rollout

We didn't get to the final number on the first try. Five distinct bugs hid the
model's true performance — including one that proved the originally-reported
0.79 average was a **scripted-policy fallback**, not the LLM. The journey graph
below is every iteration we ran.

**Before/after rollout** is a real `grpo_v2` run on `weather_roulette` seed=0:
trained scores **0.985** vs untrained random scores **0.347**. The trained model
calls `REQUEST_FORECAST` early, then `PIT_NOW inter` at lap 6, one lap before
rain peak. Full transcript: `demo-assets/trained-rollout-transcript.txt`.
"""

UPDATES["cell_race_story"] = """\
from IPython.display import Image as IPyImage, display
import pathlib

journey = pathlib.Path("results/journey.png")
breakdown = pathlib.Path("results/scenario_breakdown.png")

if journey.exists():
    print("Iteration journey (avg across scenarios per stage):")
    display(IPyImage(str(journey), width=920))
if breakdown.exists():
    print("Per-scenario breakdown across iterations:")
    display(IPyImage(str(breakdown), width=920))

# Print rollout transcript for the 'show, don't tell' moment
transcript = pathlib.Path("demo-assets/trained-rollout-transcript.txt")
if transcript.exists():
    print("\\n--- Real rollout transcript (weather_roulette, seed=0) ---")
    print(transcript.read_text())
"""

UPDATES["cell_sec6_header"] = """\
## Section 6 — Track generalization grid

Expert policy across 8 F1 circuits — Monza / Spa / Monaco / Catalunya / Silverstone /
Suzuka / Zandvoort / Hungaroring. Confirms the env generalises across full
circuit layouts, not just one hand-tuned track.
"""

UPDATES["cell_track_grid"] = """\
from IPython.display import Image as IPyImage, display
import pathlib, subprocess

grid_path = pathlib.Path("results/track_grid.png")
if not grid_path.exists() and pathlib.Path("scripts/render_track_grid.py").exists():
    subprocess.run(["python", "scripts/render_track_grid.py", "--out", str(grid_path)], check=False)

if grid_path.exists():
    display(IPyImage(str(grid_path), width=980))
    print(f"Track grid: {grid_path}")
else:
    print("Track grid not yet rendered.")
"""

UPDATES["cell_sec7_header"] = """\
## Section 7 — Reproduce the SFT → GRPO recipe yourself *(GPU required)*

This section reproduces the `grpo_v2` checkpoint. Recipe (matches what we
actually shipped):

1. **SFT warm-start** — 3 epochs on `sft_dataset_v2.jsonl` (4,900 expert turns
   with enriched obs and `enable_thinking=False` rendering). ~12 min on a 5090.
2. **GRPO refine** — 200 steps on top of the SFT base, `num_generations=8`,
   `beta=0.005`. ~22 min on a 5090 (no Unsloth, no vLLM — see compat notes).

Approximate Colab times:
- T4 (16 GB): SFT ~30 min, GRPO ~60 min
- A100 (40 GB): SFT ~10 min, GRPO ~20 min

**Compatibility caveats (read before running):**
- `trl==0.18.2` is pinned because newer transformers/trl versions had unrelated
  import breaks at the time of this run (`TRANSFORMERS_CACHE` etc).
- Unsloth's compiled GRPO trainer assumes `trl >= 0.22` (calls
  `truncate_with_protected_tokens`). On `trl 0.18.2` it raises NameError, so
  Section 7's GRPO cell sets `--no-unsloth`. SFT still uses Unsloth (works fine).
- vLLM is disabled in `train.py` (`use_vllm=False`) for the same reason.
"""

UPDATES["cell_gpu_check"] = """\
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}  ({vram:.1f} GB VRAM)")
    if vram < 14.5:
        print("⚠  This GPU may not have enough VRAM for Qwen3-4B + LoRA. "
              "T4 (16 GB) is the bare minimum; A100 (40 GB) recommended.")
else:
    print("No GPU detected. Section 7 needs a GPU runtime.")
    print("Colab → Runtime → Change runtime type → T4 GPU (or A100 if available)")
"""

UPDATES["cell_train_deps"] = """\
# Training stack pinned to versions our recipe is tested against
%pip install -q \\
    "torch>=2.4" \\
    "trl==0.18.2" \\
    "transformers==4.57.6" \\
    "peft>=0.13.2" \\
    "datasets>=3.2.0" \\
    "accelerate>=1.3" \\
    "bitsandbytes>=0.49" \\
    "unsloth==2026.4.8" "unsloth_zoo==2026.4.9"

print("Training deps installed.")
print("Note: GRPO step uses --no-unsloth flag (TRL 0.18 compat).")
"""

UPDATES["cell_sft_dataset"] = """\
# Step 7a — Generate the enriched SFT dataset (4,900 expert turns)
import subprocess, pathlib

ds = pathlib.Path("sft_dataset_v2.jsonl")
if not ds.exists():
    res = subprocess.run(
        ["python", "capture_everything.py",
         "--tasks", "dry_strategy_sprint", "weather_roulette",
                    "late_safety_car", "championship_decider",
         "--n-seeds", "100",
         "--output", str(ds)],
        capture_output=True, text=True
    )
    if res.returncode == 0:
        print(res.stdout.strip().splitlines()[-1])
        print(f"Created: {ds} ({ds.stat().st_size // 1024} KB)")
    else:
        print("ERR:", res.stderr[-600:])
else:
    print(f"Dataset already exists: {ds} ({ds.stat().st_size // 1024} KB)")
"""

UPDATES["cell_grpo_smoke"] = """\
# Step 7b — SFT warm-start (3 epochs, ~12 min on RTX 5090, ~30 min on T4)
# Outputs: sft_checkpoints_v1/final/ (LoRA adapter)
import subprocess

res = subprocess.run(
    ["python", "train_sft_v1.py",
     "--model", "unsloth/Qwen3-4B",
     "--dataset", "sft_dataset_v2.jsonl",
     "--output-dir", "./sft_checkpoints_v1",
     "--epochs", "3",
     "--batch-size", "1",
     "--grad-accum", "32",
     "--lr", "1e-5"],
    capture_output=True, text=True
)
print(res.stdout[-2000:])
if res.returncode != 0:
    print("STDERR:", res.stderr[-1000:])

# Merge SFT LoRA -> standalone checkpoint usable as GRPO base
if subprocess.run(["python", "scripts/merge_lora.py",
                   "--adapter", "sft_checkpoints_v1/final",
                   "--out",     "sft_checkpoints_v1/merged"]).returncode == 0:
    print("\\nSFT base merged at sft_checkpoints_v1/merged/")
"""

UPDATES["cell_grpo_full"] = """\
# Step 7c — GRPO refinement on top of the SFT base (200 steps, ~22 min on 5090, ~60 min on T4)
# Note: --no-unsloth is required because Unsloth's GRPO trainer assumes trl >= 0.22.
import subprocess

res = subprocess.run(
    ["python", "train.py",
     "--base-checkpoint", "sft_checkpoints_v1/merged",
     "--task",   "multi",
     "--max-steps",  "200",
     "--batch-size", "1",
     "--grad-accum", "16",
     "--reward-mode", "shaped",
     "--output-dir", "./grpo_v2_colab",
     "--backend",   "trl",
     "--no-unsloth"],
    capture_output=True, text=True
)
print(res.stdout[-3000:])
if res.returncode != 0:
    print("STDERR:", res.stderr[-1000:])

# Merge GRPO LoRA -> the final inference-ready checkpoint
if subprocess.run(["python", "scripts/merge_lora.py",
                   "--adapter", "grpo_v2_colab",
                   "--out",     "grpo_v2_colab/merged"]).returncode == 0:
    print("\\nFinal model merged at grpo_v2_colab/merged/")
"""

UPDATES["cell_eval_trained"] = """\
# Step 7d — Evaluate your freshly trained checkpoint vs the shipped one
# Replace './grpo_v2_colab/merged' with the HF Hub repo to eval the official model:
#   checkpoint = "Deltasthic/f1-strategist-qwen3-4b-grpo"
import subprocess

checkpoint = "./grpo_v2_colab/merged"   # or "Deltasthic/f1-strategist-qwen3-4b-grpo"
print(f"Evaluating: {checkpoint}")
res = subprocess.run(
    ["python", "evaluate.py",
     "--model", checkpoint,
     "--tasks", "dry_strategy_sprint", "weather_roulette",
                "late_safety_car", "championship_decider",
                "virtual_safety_car_window", "tyre_cliff_management",
     "--n-seeds", "5",
     "--modes",   "trained",
     "--output-json", "results/eval_colab.json",
     "--output-png",  "results/eval_colab.png"],
    capture_output=True, text=True
)
print(res.stdout[-2000:])
if res.returncode != 0:
    print("STDERR:", res.stderr[-800:])
"""

UPDATES["cell_summary"] = """\
## Summary

| What we showed | Evidence |
|---|---|
| Environment works end-to-end | Section 2 smoke test: reset → step → score |
| `grpo_v2` lifts +0.20 over untrained | Section 3 bar chart, 6 scenarios × 5 seeds |
| Real training, real reward | Section 4 curve from `trainer_state.json`, peak 0.93 |
| Five bugs caught + fixed | Section 5 journey + before/after rollout transcript |
| Generalises across 8 circuits | Section 6 track grid |
| Recipe is reproducible | Section 7: SFT v3 → GRPO 200 steps, ~90 min on T4 |

**Best single result:** weather scenario, trained 0.965 > rule-based expert 0.950.
The model learned to call `REQUEST_FORECAST` early and pit for inters one lap
before rain peak — investigation discipline + correct timing.

**Honest gap:** dry sprint stays at 0.52 because the model rarely picks
`PIT_NOW`. PIT events are 8% of training tokens — under-represented. Future work
in [`blog.md`](https://github.com/Deltasthicc/F1_Simulator_OpenENV/blob/dev/blog.md#whats-next-concrete-starting-points) tier list.

**Links:**
- Model:  https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo
- Space:  https://huggingface.co/spaces/Deltasthic/f1-strategist
- Live:   https://f1.chinnaboina.com/
- Blog:   https://github.com/Deltasthicc/F1_Simulator_OpenENV/blob/dev/blog.md
- Repo:   https://github.com/Deltasthicc/F1_Simulator_OpenENV (branch `dev`)
"""


def main() -> None:
    nb = json.loads(NB.read_text())
    n_updated = 0
    for cell in nb["cells"]:
        cid = cell.get("id")
        if cid in UPDATES:
            new_src = UPDATES[cid].splitlines(keepends=True)
            cell["source"] = new_src
            cell.pop("outputs", None)
            cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs", [])
            cell["execution_count"] = None if cell["cell_type"] == "code" else cell.get("execution_count")
            n_updated += 1
    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Updated {n_updated}/{len(UPDATES)} cells in {NB.name}")


if __name__ == "__main__":
    main()
