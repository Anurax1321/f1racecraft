# Pre-Push Checklist

Walk through this **in order** before tagging the final submission commit. Do not skip
items. Each takes 30 seconds to verify and saves a hour of regret if missed.

---

## 1. Repository hygiene

- [ ] `git status` is clean. No uncommitted changes.
- [ ] No accidentally-committed secrets. Run:
  ```bash
  git log -p | grep -E "(hf_[a-zA-Z0-9]{30,}|sk-[a-zA-Z0-9]{30,}|API_KEY=|TOKEN=)" || echo "clean"
  ```
- [ ] No accidentally-committed large files. Run:
  ```bash
  git ls-files | xargs du -h 2>/dev/null | sort -h | tail -10
  ```
  Anything > 10 MB is suspect. Checkpoints, datasets, and venvs go in `.gitignore`,
  not in git.
- [ ] `.gitignore` includes `.venv/`, `__pycache__/`, `*.pyc`, `sft_dataset_*.jsonl`,
  `grpo_*/`, `sft_checkpoints_*/`, `.env`, `captures/*.gif`, `captures/*.mp4`.

## 2. README accuracy

- [ ] HF Space frontmatter at top (lines 1–17 are YAML)
- [ ] Title, emoji, sdk, app_port match the live Space
- [ ] All section links resolve (no 404s on `docs/*.md`)
- [ ] HF Space link points to a Space that actually loads
- [ ] Colab badge / link works (open it in incognito to verify auth-free access)
- [ ] Blog post link points to a published HF blog
- [ ] Video link points to a public-or-unlisted YouTube
- [ ] All `results/*.png` referenced in README actually exist

## 3. Phase 1 (Environment) gating

- [ ] `python -m server.app` starts without errors
- [ ] `curl http://localhost:8000/health` → `{"status": "ok"}`
- [ ] `python tests/smoke_http.py` passes all 5 checks
- [ ] `python tests/test_environment.py` passes
- [ ] `client.py` imports without errors

## 4. Phase 2 (Scenarios + Scoring) gating

- [ ] `python tests/smoke_all_scenarios.py` passes — expert ≥ 0.85, panic ≤ 0.40 on each
- [ ] `python tests/test_scoring.py` passes — six-dim scorer is pure and deterministic
- [ ] `python -m baselines.expert_solver --task dry_strategy_sprint` scores ≥ 0.92
- [ ] `python -m baselines.expert_solver --task weather_roulette` scores ≥ 0.92
- [ ] `python -m baselines.expert_solver --task late_safety_car` scores ≥ 0.92
- [ ] `python -m baselines.expert_solver --task championship_decider` scores ≥ 0.85
- [ ] All four expert traces in `baselines/trajectories/expert_*.jsonl` are committed

## 5. Phase 3 (Training) gating — THE CRITICAL ONE

- [ ] At least one GRPO reward curve exists in `results/training_loss_curve.png`
- [ ] `results/eval_summary.json` has numbers for random / untrained / trained / expert
- [ ] `results/eval_curve.png` exists and shows ≥ 0.20 average improvement of trained over untrained
- [ ] Trained checkpoint is published to `Deltasthic/f1-strategist-qwen3-4b-grpo` (or chosen name)
- [ ] The published checkpoint is **public** (not gated, not private — judges must access it)
- [ ] Model card on the HF repo describes how to use it
- [ ] `notebooks/f1_strategist_training_colab.ipynb` runs end-to-end on Colab Free T4 in < 10 min
- [ ] The notebook prints both a reward-curve plot and one full rollout transcript

## 6. Phase 4 (Self-improvement) — optional but score-boosting

- [ ] `server/postmortem.py` records postmortems on episode end
- [ ] `reset()` injects `memory_hints` from past postmortems
- [ ] `results/ablation.md` shows memory-augmented vs base trained delta
- [ ] Ablation delta is positive (≥ 0.03 average) — if not, omit the claim

## 7. Phase 5 (Demo + Deploy) gating

- [ ] HF Space is **Running** (not building, not paused, not error)
- [ ] `python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space` passes
- [ ] Gradio `/web` route works (if enabled)
- [ ] `demo-assets/hf-space-link.txt` populated
- [ ] `demo-assets/youtube-link.txt` populated
- [ ] `demo-assets/hf-blog-link.txt` populated
- [ ] Blog post rendered correctly on HF Hub (open in incognito to verify)
- [ ] Video has audio AND captions
- [ ] Video is < 2:00 runtime (W3 requirement)
- [ ] At least one visualizer GIF in `demo-assets/` shows the trained agent making a good call

## 8. Submission requirements (W1–W4)

- [ ] **W1** — `pyproject.toml` lists `openenv-core>=0.2.3` as a dep. Verify with `pip show openenv-core`.
- [ ] **W1** — Colab notebook is committed and runnable
- [ ] **W2** — HF Space exists, is public, is Running
- [ ] **W3** — Blog OR video published (we have both, ideally)
- [ ] **W4** — README links to all of: HF Space, Colab, blog, video, eval_curve.png, training_loss_curve.png

## 9. Theme alignment

- [ ] README §"Theme alignment" references Theme #2 and Theme #3.1 explicitly
- [ ] Blog post mentions long-horizon planning as the core motivation
- [ ] Architecture doc demonstrates partial-observability via hidden state

## 10. Final smoke

```bash
# Smoke the env locally one last time
python tests/smoke_http.py
python tests/smoke_all_scenarios.py

# Smoke the live Space
python tests/smoke_http.py --base-url https://Deltasthic-f1-strategist.hf.space

# Smoke the eval reproduction
python evaluate.py --model Deltasthic/f1-strategist-qwen3-4b-grpo \
    --tasks dry_strategy_sprint --n-seeds 2 --modes trained
# Should print numbers matching eval_summary.json (within seed noise)

# Smoke the Colab notebook
# (open it in Colab, run all cells, verify reward curve appears)

# Verify all artifacts exist
for f in results/eval_summary.json results/eval_curve.png \
         results/training_loss_curve.png demo-assets/blog-post.md \
         demo-assets/video-script.md demo-assets/hf-space-link.txt; do
  test -s "$f" && echo "  ok: $f" || echo "MISSING: $f"
done
```

If any line says `MISSING:`, fix before tagging.

## 11. Tag and submit

```bash
git tag -a v1.0-finale -m "Meta PyTorch OpenEnv Hackathon Grand Finale submission"
git push origin main --tags
git push hfspace main:main

# Confirm tag visible on GitHub
echo "https://github.com/Deltasthicc/f1-strategist/releases/tag/v1.0-finale"
```

## 12. Submit to organisers

Submit through the official hackathon form (link in the hackathon Slack / Discord):
- GitHub URL: `https://github.com/Deltasthicc/f1-strategist`
- HF Space URL: `https://huggingface.co/spaces/Deltasthic/f1-strategist`
- Blog URL: from `demo-assets/hf-blog-link.txt`
- Video URL: from `demo-assets/youtube-link.txt`
- Theme: #2 Long-Horizon (primary), #3.1 Professional Tasks (secondary)
- Authors: Shashwat Rajan, Tanish Shitanshu

Take a screenshot of the submission confirmation. Save it to `demo-assets/submission-confirmation.png`.

Done. Now go and demo. Good luck.
