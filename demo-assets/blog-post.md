# Training an LLM to call F1 race strategy

> Draft. Target 600–900 words. Final version goes on Hugging Face Hub blog.
> Embed: `results/eval_curve.png` and one visualizer GIF showing the trained
> agent calling the rain pit correctly.
> Authors: Shashwat Rajan, Tanish Shitanshu (`Deltasthic`).

---

It's lap 8 of 12 at Spa. Light rain has started. The leader ahead just stayed out on slicks. The car behind boxed two laps ago for inters and is setting purple sectors. Your tyres have four laps of grip left. Your pit window closes in three. The board on the pit wall is asking what to call.

That's the world we built for the **Meta PyTorch OpenEnv Hackathon Grand Finale**. F1 Strategist is an OpenEnv environment for training LLM agents on real Formula 1 race-strategy decisions. The model doesn't drive the car — a built-in physics simulator handles laps, tyres, fuel, and opponents. The model is the **race engineer** making strategic calls.

## Why F1 strategy is a good RL benchmark

Three things matter for a long-horizon LLM environment, and race strategy hits all three:

- **Verifiable but not scriptable.** Final position, FIA compound rules, fuel margin — all programmable Python checks. No LLM judge in the reward path. But the optimal sequence depends on hidden state (opponent stint plans, weather evolution, true tyre degradation) that the agent must investigate before deciding.
- **Genuinely long-horizon.** A 12-lap sprint is 12 strategic decision points across 25 minutes of simulated race time. A wrong call at lap 4 only manifests in lap 9. This is the failure mode current LLMs handle worst — and it's exactly what Theme #2 of the hackathon asks for.
- **Multi-objective without being subjective.** Six independent reward dimensions (race result, tyre management, fuel discipline, strategic timing, comms quality, operational efficiency) combine into a single weighted scalar. The model trains against a stable signal; judges read a coherent breakdown.

## The environment

Each episode is one race. Each step is one lap. The agent sees the current race state (positions, weather, tyres, fuel, opponents within TV-feed visibility) and emits one of ~20 strategic commands:

```
PIT_NOW <soft|medium|hard|inter|wet>
SET_MODE <push|race|conserve|fuel_save|tyre_save>
DEFEND_POSITION  /  ATTACK_AHEAD  /  HOLD_GAP <n>
INSPECT_TYRE_DEGRADATION
CHECK_OPPONENT_STRATEGY <n>
REQUEST_FORECAST
ASSESS_UNDERCUT_WINDOW
RADIO_DRIVER "<message>"
DONE
```

The signature primitive is **hidden state revealed by inspection**. Each episode has a true tyre wear curve, opponent stint plans, a pre-rolled weather schedule, and a pre-rolled safety-car schedule. None of it is in the observation by default. The agent investigates by issuing inspection commands, each of which pays a small `+0.02` shaping reward the first time it returns new information.

This forces the agent to *plan* — not just react. We saw this work in our Round-1 OpsTwin submission and we ported the entire architectural pattern into the F1 domain.

## The training story

The repo is wired for the Qwen3-4B + TRL GRPO run on a single RTX 5090. The local smoke
artifact included here uses the same environment/evaluation path with a scripted checkpoint
policy so we can verify the end-to-end plumbing before spending GPU time. The intended full
run is:

- Stage 1: SFT warm-start on rule-based expert trajectories
- Stage 2: GRPO with shaped per-step rewards
- Stage 3: GRPO with sparse final-only rewards (long-horizon polish)

> [Embed: `results/eval_curve.png`]

In the local smoke evaluation, the checkpoint policy beats the shallow untrained baseline by a wide margin across all four scenario families. The single most-vivid example is the **weather roulette** scenario at Spa: the untrained baseline stays out into the rain and lands around 0.38. Same seed, same opponents, the checkpoint policy calls `REQUEST_FORECAST`, inspects tyres, sends the radio call, then pits for inters before the rain peak. P2 finish, around 0.95 score.

> [Embed: visualizer GIF — 4 seconds, untrained vs trained side-by-side]

## The self-improvement loop

After each episode, the env classifies the failure (missed undercut, late weather call, fuel underburn, panic pit, ...) and writes a structured postmortem to a JSONL file. On the next reset for the same scenario family, the top-2 lowest-scoring past postmortems are injected into the initial observation as memory hints:

```
[MEMORY] Last similar incident: late_weather_call.
First bad action: STAY_OUT lap 7. Missed signal: REQUEST_FORECAST not called.
Preferred order: REQUEST_FORECAST → INSPECT_TYRE_DEGRADATION → PIT_NOW inter.
```

Memory retrieval is implemented and measurable via `results/ablation.md`. The local scripted smoke policy is already near ceiling, so its ablation delta is expectedly small; the meaningful ablation should be rerun after the real GRPO checkpoint is trained.

## Try it

- **HF Space (interactive demo):** [`Deltasthic/f1-strategist`](https://huggingface.co/spaces/Deltasthic/f1-strategist)
- **Trained checkpoint:** [`Deltasthic/f1-strategist-qwen3-4b-grpo`](https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo)
- **Code:** [github.com/Deltasthicc/f1-strategist](https://github.com/Deltasthicc/f1-strategist)
- **Colab smoke:** open `notebooks/f1_strategist_training_colab.ipynb` (~7 min on a Free T4)

The Space exposes the OpenEnv HTTP API plus a Gradio panel at `/web` where you can issue commands manually and watch the simulated race progress.

## Acknowledgments

Architectural primitives ported from our Round-1 submission OpsTwin Recovery Arena. Track geometry from the open-source [racetrack-database](https://github.com/TUMFTM/racetrack-database). Historical pace calibration from the Kaggle Formula 1 World Championship dataset. Thanks to Meta PyTorch, Hugging Face, Unsloth, and Scaler AI Labs for organising the hackathon.
