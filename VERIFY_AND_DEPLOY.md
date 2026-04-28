# Verify, Deploy, Train — Step by Step

This is the runbook from "fresh clone" through "trained checkpoint published".
Every command is meant to be copy-pasted. Expected outputs are listed so you
know whether each step worked. Stop and fix the moment something diverges
from the expected range.

## 0. Conventions

- Repo root in this guide is referred to as `Project/`. Run all commands
  from there unless stated otherwise.
- Where you see `Deltasthic/...` (HF) or `Deltasthicc/...` (GitHub), substitute
  your actual handle if you've changed it since.
- The 5090 is on a remote Arch-Linux SSH box; everything else runs on the
  laptop or the HF Space.
- Time budgets in parentheses are wall-clock estimates; budget 2× for first runs.

---

## Phase A — Local smoke checks (no GPU, ~5 min)

These confirm the environment, scoring, scenarios, and evaluation all work
before you spend GPU time. Run **every step** the first time and re-run
A.1 + A.2 + A.4 after any code change.

### A.1 Install the package

```bash
cd Project
python3 -m venv .venv && source .venv/bin/activate
pip install -e . --break-system-packages --ignore-installed PyJWT 2>&1 | tail -1
pip install pytest --break-system-packages 2>&1 | tail -1
```

Expected: `Successfully installed f1-strategist-1.0.0` (or a more recent
version) and `Successfully installed iniconfig-... pluggy-... pytest-...`.

> If `pip` complains about externally-managed Python, the `--break-system-packages`
> flag is intentional on Debian/Ubuntu/Arch. On macOS this flag is unnecessary.

### A.2 Run the full test sweep (~15 s)

```bash
python3 -m pytest tests/ -q
```

Expected output:
```
tests/test_environment.py ....                                           [  3%]
tests/test_inference.py ...............................................  [ 39%]
tests/test_invariants.py .........................................       [ 70%]
tests/test_postmortem.py ..                                              [ 72%]
tests/test_scoring.py ......                                             [ 76%]
tests/test_scoring_strict.py ..............................              [100%]
============================= 130 passed in 13.26s =============================
```

If anything **other than** `130 passed` shows up, do not proceed. Read
`AUDIT_NOTES.md` to understand what each suite covers, then triage.

### A.3 Run the scenario discrimination smoke (~5 s)

```bash
python3 tests/smoke_all_scenarios.py
```

Expected:
```
dry_strategy_sprint      expert=0.885 panic=0.322
weather_roulette         expert=0.950 panic=0.378
late_safety_car          expert=0.935 panic=0.365
championship_decider     expert=0.965 panic=0.170
All scenario checks passed.
```

Expert must be ≥ 0.85, panic must be ≤ 0.40 on every family. If any number
lands outside that band, scoring or scenarios drifted — investigate before
moving on.

### A.4 Run the eval baseline (~30 s)

```bash
python3 evaluate.py \
    --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
    --n-seeds 5 \
    --modes random untrained trained expert \
    --output-json results/eval_summary.json \
    --output-png results/eval_curve.png
```

Expected (numbers ±0.05):
```
random     dry_strategy_sprint      mean=0.40
untrained  dry_strategy_sprint      mean=0.51
trained    dry_strategy_sprint      mean=0.75
expert     dry_strategy_sprint      mean=0.84
random     weather_roulette         mean=0.34
untrained  weather_roulette         mean=0.41
trained    weather_roulette         mean=0.94
expert     weather_roulette         mean=0.95
random     late_safety_car          mean=0.33
untrained  late_safety_car          mean=0.53
trained    late_safety_car          mean=0.94
expert     late_safety_car          mean=0.94
random     championship_decider     mean=0.21
untrained  championship_decider     mean=0.27
trained    championship_decider     mean=0.54
expert     championship_decider     mean=0.97
```

Open `results/eval_curve.png` to confirm the bar chart looks right (clear
ordering: random < untrained < trained ≤ expert on every family). This
becomes the **before-training baseline** against which the GRPO checkpoint
will be compared.

### A.5 Render a sample rollout (~10 s)

```bash
python3 rollout.py --task weather_roulette --seed 7 --mode trained --render --verbose
```

Expected: prints lap-by-lap commands, ends with `score=0.~ trace=captures/...`,
writes `captures/weather_roulette_trained_seed7.gif`. Open the GIF — you should
see the track polygon, ego car circle, opponent dots, and a per-lap weather
strip. This is the asset format that ships with the demo blog and video.

### A.6 Heuristic inference end-to-end (~10 s)

```bash
python3 inference.py --model heuristic --task weather_roulette --n-episodes 1 --verbose
```

Expected: prints `L00 -> REQUEST_FORECAST`, `L01 -> STAY_OUT`, ..., ends
with `Episode 0: score=0.9~`. Confirms the heuristic policy + parser work
without any LLM checkpoint loaded.

### A.7 (Optional) Live HTTP smoke

```bash
python3 -m server.app &
SERVER_PID=$!
sleep 3
python3 tests/smoke_http.py
kill $SERVER_PID
```

Expected: 5 checks pass, server logs show clean startup. This is the same
test you'll run against the deployed HF Space in Phase B.5.

---

## Phase B — HuggingFace Space deployment (~20 min, mostly waiting)

The Space serves the OpenEnv HTTP API plus an optional Gradio panel. Judges
hit it directly, and the Colab smoke notebook can pull the env from it.
Required for hackathon W2.

### B.1 HuggingFace authentication (one-time)

```bash
pip install -U "huggingface_hub[cli]" --break-system-packages
huggingface-cli login
```

Paste a **write** token from https://huggingface.co/settings/tokens. Token
is stored in `~/.cache/huggingface/token`. Verify:

```bash
huggingface-cli whoami
```

Expected: prints your username + the orgs you belong to (look for `Deltasthic`).

### B.2 Create the Space (one-time)

```bash
huggingface-cli repo create f1-strategist \
    --type space \
    --space_sdk docker \
    --organization Deltasthic
```

Expected: `Your space has been created at https://huggingface.co/spaces/Deltasthic/f1-strategist`

If the org doesn't exist yet or you're not a member, drop `--organization
Deltasthic` to push to your personal account. The README references can be
updated later.

### B.3 Add the Space as a git remote

```bash
git remote add hfspace https://huggingface.co/spaces/Deltasthic/f1-strategist
git remote -v
```

You should see both `origin` (GitHub) and `hfspace` (HF) listed. If the
remote already exists from a prior deploy, drop the add line.

### B.4 Push the code (~3 min build)

```bash
git push hfspace main:main
```

Push triggers an automatic build. Watch progress:

```
https://huggingface.co/spaces/Deltasthic/f1-strategist?logs=build
```

Build phases:
1. **Resolving** — git checkout (~10 s)
2. **Building** — `docker build` (~2-3 min, dominated by `pip install openenv-core`)
3. **Running** — container starts; `uvicorn server.app:app` listens on port 8000

Wait for the **Running** badge in the Space header. The status badge is also
visible at `https://huggingface.co/api/spaces/Deltasthic/f1-strategist`.

> **Common build failures:**
> - "Failed to resolve dependency" → check `pyproject.toml` and `Dockerfile`
>   pin set; usually a transient PyPI issue, retry the push
> - "ModuleNotFoundError: openenv" → make sure `openenv-core>=0.2.3` is in
>   the `Dockerfile`'s `pip install` line
> - "App could not be loaded" → click **Logs → Container** in the Space UI;
>   look for the import error, fix locally, push again

### B.5 Smoke-test the live Space

```bash
python3 tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space
```

Expected: same 5 checks pass as in A.7. **The URL pattern is `https://<org>-<repo>.hf.space`** —
slashes become hyphens. If the Space is private or paused, you'll get
HTTP 404; check the Space settings.

Save the URL to the demo asset file:

```bash
echo "https://huggingface.co/spaces/Deltasthic/f1-strategist" > demo-assets/hf-space-link.txt
git add demo-assets/hf-space-link.txt
git commit -m "demo: lock HF Space URL"
git push origin main
```

### B.6 (Optional) Manual interactive test

Visit `https://Deltasthic-f1-strategist.hf.space/web` in a browser. The
Gradio panel renders if `ENABLE_WEB_INTERFACE=1` is set in the Dockerfile
(it already is). Click **Reset**, type `INSPECT_TYRE_DEGRADATION` in the
command box, click **Step**, and watch the JSON observation update.

### B.7 (Optional) Eval against the live Space

You can run `evaluate.py` against the deployed Space rather than a local
import — useful for validating that the deployed code matches local. The
current `evaluate.py` always uses a local env; if you need remote eval,
substitute the env line with:

```python
from client import F1StrategistEnv
env = F1StrategistEnv(base_url="https://Deltasthic-f1-strategist.hf.space")
```

This is **not required** for the hackathon submission — local eval against
the same code is sufficient.

---

## Phase C — RTX 5090 server setup (~15 min, one-time)

All training happens on the 5090. The 32 GB system RAM is fine for the
SFT dataset (~50 MB) and Qwen3-4B + LoRA (~12 GB VRAM).

### C.1 SSH in and verify GPU

```bash
ssh user@your-5090-host
nvidia-smi
```

Expected: shows the RTX 5090, driver version, 32 GB VRAM total, ~0% utilisation.
If `nvidia-smi` doesn't work but `lsmod | grep nvidia` shows the driver loaded,
the user-space CUDA toolkit is misconfigured — see GPU_HANDOFF.md §troubleshooting.

### C.2 Clone and install

```bash
git clone https://github.com/Deltasthicc/f1-strategist.git
cd f1-strategist
python3 -m venv .venv && source .venv/bin/activate
```

Install torch with **CUDA 12.8 wheels** (REQUIRED for the 5090's Blackwell sm_120 architecture):

```bash
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch
```

Verify:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True NVIDIA GeForce RTX 5090`.

If `torch.cuda.is_available()` returns False but `nvidia-smi` works, you have
the wrong CUDA wheel. Force-reinstall:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 --force-reinstall torch
```

### C.3 Install training stack

```bash
pip install -e .
pip install \
    'trl==0.14.0' 'transformers>=4.55.4' 'peft>=0.13.2' \
    'datasets>=3.2.0' 'accelerate>=1.3' \
    'liger-kernel<0.6' 'bitsandbytes>=0.49' \
    matplotlib protobuf python-dotenv
pip install "unsloth[cu128] @ git+https://github.com/unslothai/unsloth.git"
```

> Pin notes (lessons from Round 1):
> - `transformers >= 4.51` is required for Qwen3 architecture recognition
> - `accelerate >= 1.3` is required for `transformers >= 4.55` (older versions
>   miss the `keep_torch_compile` kwarg)
> - `liger-kernel < 0.6` if you stay on transformers 4.51–4.55; pin >= 0.6
>   only if you upgrade transformers to >= 4.52

### C.4 HF auth on the server

```bash
huggingface-cli login
echo "HF_TOKEN=hf_xxx" > .env       # for scripts that load via dotenv
```

Verify with `huggingface-cli whoami`.

### C.5 Re-run the local smoke checks

Repeat all of Phase A from the 5090. **Both** environments must pass the 130
tests; otherwise something in the install diverged.

```bash
python3 -m pytest tests/ -q
python3 tests/smoke_all_scenarios.py
python3 evaluate.py --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider --n-seeds 3 --modes random untrained trained expert --output-json /tmp/eval_5090.json --output-png /tmp/eval_5090.png
```

If everything matches Phase A's expected output ±0.05, you're ready to train.

---

## Phase D — Training (~4 hours full run)

Three sub-phases: SFT seed dataset → SFT warm-start → GRPO. SFT warm-start
is recommended but optional — you can go straight to GRPO if time is tight.

### D.1 Generate SFT seed data (~3 min)

```bash
python3 capture_everything.py \
    --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
    --n-seeds 50 \
    --output sft_dataset_v1.jsonl
```

Expected: `wrote ~3000 SFT turns to sft_dataset_v1.jsonl`. Each line is a
chat-format `(system, user, assistant)` triple where the assistant message
is the expert action for that observation.

Verify the file is well-formed:

```bash
wc -l sft_dataset_v1.jsonl
head -1 sft_dataset_v1.jsonl | python3 -m json.tool | head -20
```

### D.2 GRPO smoke run (~10 min, the canary)

Always run a 50-step smoke on Qwen3-0.6B before the full run. This is the
gate that catches reward bugs before you waste 4 hours.

```bash
tmux new -s smoke
source .venv/bin/activate

python3 train.py \
    --backend trl \
    --model Qwen/Qwen3-0.6B \
    --task dry_strategy_sprint \
    --max-steps 50 \
    --batch-size 1 --grad-accum 8 \
    --logging-steps 5 \
    --save-steps 25 \
    --output-dir ./grpo_smoke 2>&1 | tee training_smoke.log
```

What you should see in `training_smoke.log`:

- Step 1–10: rewards in the 0.0–0.5 range, fairly noisy
- Step 10–30: average reward starts trending upward
- Step 30–50: average reward at 0.5+ on most steps

Detach from tmux with `Ctrl-b d`. Reattach with `tmux attach -t smoke`.

After the run finishes, plot the curve:

```bash
python3 scripts/plot_training_curve.py --run-dir ./grpo_smoke --output /tmp/smoke_curve.png
scp user@5090:/tmp/smoke_curve.png .   # if you want to view from your laptop
```

If the curve is flat or trending **down**, **stop** — you have a reward
bug, not a training bug. Read `AUDIT_NOTES.md` and inspect a couple of
rollouts manually:

```bash
python3 rollout.py --task dry_strategy_sprint --seed 0 --mode trained --model ./grpo_smoke/checkpoint-25 --verbose
```

### D.3 Full GRPO run (~3-4 hours)

Once smoke is green, kick off the real run:

```bash
tmux new -s grpo

python3 train.py \
    --backend trl \
    --model Qwen/Qwen3-4B \
    --task multi \
    --max-steps 500 \
    --batch-size 1 --grad-accum 32 \
    --logging-steps 10 \
    --save-steps 50 \
    --learning-rate 5e-6 \
    --output-dir ./grpo_v1 2>&1 | tee training_v1.log
```

Detach and let it run. Expected wall-clock: ~30 s/step × 500 steps = ~4.2 hours.
Memory: ~28 GB VRAM. Logs print one line per `--logging-steps`.

Useful side-terminals:
- `nvidia-smi -l 1` — live GPU utilisation (target ≥ 90%)
- `tail -f training_v1.log | grep reward` — reward trend
- `du -sh grpo_v1/` — checkpoint disk usage (will grow to ~20 GB)

If reward plateaus before step 200, **don't** push more steps — the model
isn't learning. Drop `--learning-rate` to 2e-6 and restart.

### D.4 Plot the training curve

```bash
python3 scripts/plot_training_curve.py --run-dir ./grpo_v1 --output results/training_loss_curve.png
```

Open `results/training_loss_curve.png`. You should see loss trending down,
reward trending up. Two distinct trends visible in the same chart (twin axes).

### D.5 Push the trained checkpoint to HF Hub

```bash
python3 scripts/push_checkpoint.py \
    --checkpoint ./grpo_v1 \
    --repo Deltasthic/f1-strategist-qwen3-4b-grpo
```

Expected: ~8 GB upload, takes 5-10 min depending on your uplink. Verify
on https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo that the
files appear (config.json, adapter_model.safetensors, tokenizer files, etc).

---

## Phase E — Final eval and ablation (~30 min)

### E.1 Evaluate the published checkpoint

```bash
python3 evaluate.py \
    --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
    --n-seeds 5 \
    --modes random untrained trained expert \
    --output-json results/eval_summary.json \
    --output-png results/eval_curve.png
```

This run will load the actual GRPO checkpoint into the `trained` column
(not the scripted fallback). Expected outcome:

- **trained mean ≥ 0.65 averaged across all four families** — the demo
  threshold; below this, the training pipeline didn't deliver a useful policy
- **trained > untrained by ≥ 0.20 on every family** — the headline number
  for the blog post
- **trained ≤ expert on every family** — sanity (if trained beats expert,
  scoring is broken or the expert sequence has regressed)

### E.2 Postmortem ablation

Test whether memory hints help the trained policy:

```bash
# Without memory injection
python3 evaluate.py \
    --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
    --n-seeds 10 \
    --modes trained \
    --no-memory \
    --output-json results/ablation_no_memory.json

# With memory injection (default)
python3 evaluate.py \
    --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
    --n-seeds 10 \
    --modes trained \
    --output-json results/ablation_with_memory.json

# Diff
python3 scripts/diff_ablation.py \
    results/ablation_no_memory.json \
    results/ablation_with_memory.json \
    --output results/ablation.md
```

Expected: memory-augmented mean is 0.03–0.10 higher than no-memory. If the
delta is negative, the memory hints are noise rather than signal — that's
fine for the demo but don't claim it as a feature.

### E.3 Render demo-quality rollouts

```bash
# The "before" — untrained Qwen on a tough scenario
python3 rollout.py --task weather_roulette --seed 7 --mode untrained --render --verbose

# The "after" — trained model, same seed
python3 rollout.py --task weather_roulette --seed 7 --mode trained \
    --model Deltasthic/f1-strategist-qwen3-4b-grpo --render --verbose

mv captures/weather_roulette_untrained_seed7.gif demo-assets/untrained-spa.gif
mv captures/weather_roulette_trained_seed7.gif   demo-assets/trained-spa.gif
```

Open both GIFs — the untrained should fumble the rain pit, the trained
should call it correctly. Side-by-side these become the centrepiece of
the video.

---

## Phase F — Demo finalization and submission (~1 hour)

### F.1 Update results in the README

The README references `results/eval_curve.png` and the trained-vs-untrained
delta. Refresh the numbers from `results/eval_summary.json`:

```bash
cat results/eval_summary.json | python3 -m json.tool
```

Update the table in `README.md` and `demo-assets/blog-post.md` with the
real numbers.

### F.2 Finalize the blog post

Edit `demo-assets/blog-post.md`:
- Replace placeholder GIF paths with the actual files in `demo-assets/`
- Replace the bar chart placeholder with `results/eval_curve.png`
- Replace the qualitative comparison numbers with real figures from `eval_summary.json`

Publish on HF Hub:
1. Go to https://huggingface.co/blog
2. Click **New Article**
3. Paste the content of `blog-post.md`
4. Upload images from `demo-assets/` and `results/`
5. Publish; copy the URL

```bash
echo "https://huggingface.co/blog/your-handle/f1-strategist" > demo-assets/hf-blog-link.txt
```

### F.3 Record and upload the video

Follow the storyboard in `demo-assets/video-script.md`:
1. Open `demo-assets/untrained-spa.gif` — record the screen at 1080p
2. Cut to `demo-assets/trained-spa.gif`
3. Cut to `results/eval_curve.png`
4. Voice-over per the script

Tools: OBS, QuickTime, DaVinci Resolve free. Target ≤1:50 runtime.

Upload to YouTube unlisted-or-public:

```bash
echo "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" > demo-assets/youtube-link.txt
```

### F.4 Update README links

The README has placeholders for the Space, Colab, blog, and video links.
Replace them with the real URLs and commit.

```bash
git add demo-assets/ results/ README.md
git commit -m "demo: lock final assets and links"
git push origin main
git push hfspace main:main
```

### F.5 Walk PRE_PUSH_CHECKLIST.md

Open `PRE_PUSH_CHECKLIST.md` and tick every item. The minimum required:

- [ ] All 130 tests pass
- [ ] `tests/smoke_all_scenarios.py` passes
- [ ] `tests/smoke_http.py` against the live Space passes
- [ ] `results/eval_summary.json`, `eval_curve.png`, `training_loss_curve.png` all exist
- [ ] `Deltasthic/f1-strategist-qwen3-4b-grpo` is public on HF Hub
- [ ] `demo-assets/hf-space-link.txt`, `hf-blog-link.txt`, `youtube-link.txt` populated
- [ ] README links resolve in incognito
- [ ] No secrets committed (`git log -p | grep -E "hf_[a-zA-Z0-9]{30,}"` returns empty)

### F.6 Tag the submission

```bash
git tag -a v1.0-finale -m "Meta PyTorch OpenEnv Hackathon Grand Finale submission"
git push origin main --tags
```

Submit the four URLs (GitHub, HF Space, blog, video) through the official
hackathon form. Screenshot the confirmation:

```bash
# Save to demo-assets/submission-confirmation.png
```

You're done.

---

## Quick reference: copy-paste verification command

If anything looks wrong at any stage, run this single command and compare:

```bash
cd Project && \
    python3 -m pytest tests/ -q 2>&1 | tail -3 && \
    echo --- && \
    python3 tests/smoke_all_scenarios.py 2>&1 | tail -6 && \
    echo --- && \
    python3 evaluate.py \
        --tasks dry_strategy_sprint weather_roulette late_safety_car championship_decider \
        --n-seeds 3 \
        --modes random untrained trained expert \
        --output-json /tmp/sanity.json \
        --output-png /tmp/sanity.png 2>&1 | tail -16
```

Expected last lines:
```
130 passed in ~13s
---
dry_strategy_sprint      expert=0.885 panic=0.322
weather_roulette         expert=0.950 panic=0.378
late_safety_car          expert=0.935 panic=0.365
championship_decider     expert=0.965 panic=0.170
All scenario checks passed.
---
random     dry_strategy_sprint      mean=0.~
untrained  dry_strategy_sprint      mean=0.~
trained    dry_strategy_sprint      mean=0.~
expert     dry_strategy_sprint      mean=0.~
... (same for all four families)
```

Numbers should be within ±0.05 of the values in section A.4.