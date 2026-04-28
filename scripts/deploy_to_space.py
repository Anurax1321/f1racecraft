"""Deploy the current dev tree to the HF Space Deltasthic/f1-strategist.

Excludes large/training-only artifacts (grpo_v2 weights, .venv, .git, datasets,
results/*.png that aren't shown by the site, etc.). Uses HfApi.upload_folder so
git-lfs budget isn't touched.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

REPO = "Deltasthic/f1-strategist"
ROOT = Path(__file__).resolve().parent.parent

# What we actually want on the Space.
ALLOW = [
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "uv.lock",
    "models.py",
    "client.py",
    "inference.py",
    "evaluate.py",
    "train.py",
    "server/**",
    "openenv/**",
    "data/**",
    "blog.md",
    "model_card.md",
    "STATE.md",
    "results/grpo_v2_reward_curve.png",
    "results/journey.png",
    "results/scenario_breakdown.png",
    "results/comparison_real.png",
    "results/race_story.png",
    "results/track_grid.png",
    "results/eval_six_scenarios.json",
    "results/eval_six_scenarios.png",
    "demo-assets/**",
]

IGNORE = [
    ".git/**",
    ".github/**",
    ".venv*/**",
    ".virtualenvs/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    "grpo_v2/**",
    "grpo_v1/**",
    "sft_v*/**",
    "rft_*/**",
    "outputs/**",
    "wandb/**",
    "datasets/**",
    "tests/**",
    "notebooks/**",
    "scripts/**",
    "baselines/trajectories/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "results/eval_*.png",
    "results/eval_*.json",
    "results/ablation_*.png",
    "results/eval_curve.png",
]
# But re-allow the specific eval files we *do* want
KEEP = [
    "results/eval_six_scenarios.json",
    "results/eval_six_scenarios.png",
]


def main() -> None:
    api = HfApi()
    print(f"Uploading {ROOT} -> {REPO} ...")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=REPO,
        repo_type="space",
        allow_patterns=ALLOW + KEEP,
        ignore_patterns=IGNORE,
        commit_message="deploy: dev sync — grpo_v2 numbers, recommendations, hover fix",
    )
    print("Done.")


if __name__ == "__main__":
    main()
