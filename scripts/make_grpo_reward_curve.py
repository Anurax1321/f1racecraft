"""Generate a real GRPO v2 reward curve PNG from trainer_state.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TS_PATH = ROOT / "grpo_v2" / "checkpoint-200" / "trainer_state.json"
OUT_PATH = ROOT / "results" / "grpo_v2_reward_curve.png"


def main() -> None:
    state = json.loads(TS_PATH.read_text())
    log = [e for e in state["log_history"] if "reward" in e and "step" in e]
    steps = [e["step"] for e in log]
    rewards = [e["reward"] for e in log]
    reward_stds = [e.get("reward_std", 0.0) for e in log]
    kls = [e.get("kl", 0.0) for e in log]
    fzeros = [e.get("frac_reward_zero_std", 0.0) for e in log]

    EXPERT_AVG = 0.937
    UNTRAINED_AVG = 0.415
    RANDOM_AVG = 0.303

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0a0a0a")

    rewards_arr = np.array(rewards)
    stds_arr = np.array(reward_stds)
    ax.fill_between(steps, rewards_arr - stds_arr, rewards_arr + stds_arr,
                    alpha=0.18, color="#00d2be", linewidth=0)
    ax.plot(steps, rewards, marker="o", color="#00d2be", linewidth=2.2,
            markersize=5, markeredgecolor="#0a0a0a", markeredgewidth=0.8,
            label="GRPO v2 reward (per logging step)", zorder=3)

    ax.axhline(y=EXPERT_AVG, color="#ffd60a", linestyle="--", linewidth=1.0,
               alpha=0.85, label=f"expert ceiling {EXPERT_AVG:.2f}")
    ax.axhline(y=UNTRAINED_AVG, color="#ef8a17", linestyle="--", linewidth=1.0,
               alpha=0.85, label=f"untrained Qwen3-4B {UNTRAINED_AVG:.2f}")
    ax.axhline(y=RANDOM_AVG, color="#666666", linestyle=":", linewidth=1.0,
               alpha=0.7, label=f"random {RANDOM_AVG:.2f}")

    peak_i = int(rewards_arr.argmax())
    ax.scatter([steps[peak_i]], [rewards[peak_i]], s=140, color="#e10600",
               edgecolor="#0a0a0a", linewidth=1.2, zorder=4)
    ax.annotate(f"peak {rewards[peak_i]:.3f} @ step {steps[peak_i]}",
                xy=(steps[peak_i], rewards[peak_i]),
                xytext=(steps[peak_i] - 35, rewards[peak_i] + 0.04),
                fontsize=9, color="#e8e8e8",
                arrowprops=dict(arrowstyle="-", color="#e10600", lw=0.8))

    ax.set_xlabel("training step", color="#888", fontsize=10)
    ax.set_ylabel("reward (per-step weighted_final score)", color="#888", fontsize=10)
    ax.set_title("GRPO v2 — real training reward (200 steps · Qwen3-4B + LoRA · RTX 5090)",
                 color="#e8e8e8", fontsize=12, pad=12)
    ax.tick_params(colors="#666", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e1e1e")
    ax.grid(True, axis="y", linestyle="--", alpha=0.18, color="#444")
    ax.set_ylim(0.30, 1.00)
    ax.set_xlim(0, max(steps) + 10)
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor="#e8e8e8")

    avg_kl = sum(kls) / len(kls) if kls else 0.0
    avg_fzero = sum(fzeros) / len(fzeros) if fzeros else 0.0
    fig.text(0.13, 0.025,
             f"From SFT v3 base · num_generations=8 · beta=0.005 · "
             f"avg KL {avg_kl:.4f} · avg frac_reward_zero_std {avg_fzero:.2f}",
             fontsize=8, color="#666")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=140, facecolor="#0a0a0a", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH} ({len(log)} log entries, peak {max(rewards):.3f})")


if __name__ == "__main__":
    main()
