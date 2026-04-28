# F1 LLM Racing — Current Cursor

> **How to use this file:** if you (or a fresh Claude session) are picking
> up this project, read this top-to-bottom. It tells you exactly where to
> resume. Update the **Cursor** block every time you finish a sub-step.

---

## Cursor

```
Phase:        0 (Foundation cleanup)
Sub-step:     0.4 (next — strip 110 MB tokenizer cruft from git)
Last commit:  38883f4 (submitted) — pre-roadmap baseline
Next action:  After GitHub repo move (Anurax1321/F1_LLM_Racing), do 0.4
Blocked on:   user to create empty repo at github.com/Anurax1321/F1_LLM_Racing
Last session: 2026-04-28
Owner:        Anurag (solo personal project, post-hackathon)
```

---

## Where we are in the big picture

| Phase | Status |
|---|---|
| 0 — Foundation cleanup | 🟡 in progress |
| 1 — Full race length | ⬜ not started |
| 2 — FastF1 grounding | ⬜ not started |
| 3 — Bigger model (14B) | ⬜ not started |
| 4 — 5 personas | ⬜ not started |
| 5 — 20 cars LLM-driven | ⬜ not started |
| 6 — Self-play GRPO | ⬜ not started |
| 7 — Live race UI | ⬜ not started |
| 8 — Commentary track | ⬜ not started |
| 9 — Polish & ship | ⬜ not started |

For the full plan see [`ROADMAP.md`](ROADMAP.md).

---

## Phase 0 sub-step status

| # | Sub-step | Status |
|---|---|---|
| 0.1 | Write `ROADMAP.md` and `STATUS.md` | ✅ done |
| 0.2 | Build `/f1_LLM_racing` preview page | ✅ done (needs server restart to go live) |
| 0.3 | Postmortem ablation on harder seeds | ✅ done — verdict: keep, +0.019 avg ([docs/postmortem-ablation.md](docs/postmortem-ablation.md)) |
| 0.4 | Strip 110 MB tokenizer cruft from git | ⬜ |
| 0.5 | Decide & act on Gradio `/web` | ⬜ |
| 0.6 | Re-run Colab on T4 | ⬜ |
| 0.7 | Add `scripts/` wrappers | ⬜ |
| 0.8 | Tag `v0.2-foundation` | ⬜ |

---

## Last shipped (hackathon — pre-roadmap)

- Champion: `grpo_v2/` — Qwen3-4B + LoRA, GRPO 200 steps, beta=0.005
- Eval: 0.618 weighted avg (untrained 0.415, expert 0.925)
- HF Space: `Deltasthic/f1-strategist` (RUNNING)
- Tunnel: `https://f1.chinnaboina.com/` (LIVE)
- HF Hub model: `Deltasthic/f1-strategist-qwen3-4b-grpo`

Full hackathon log lives in `STATE.md` (frozen — do not edit).

---

## Hardware (single 5090)

- GPU: RTX 5090, 32 GB VRAM, CUDA 13.2
- RAM: 30 GB + 15 GB swap
- Disk: 875 GB free at `/`
- Python venv: `~/.virtualenvs/f1-strategist`
- Repo: `/home/anurag/projects/F1_Simulator_OpenENV` (branch `dev`)
- Port: `8765` (cloudflared → `f1.chinnaboina.com`)

---

## Routes

| Route | Purpose | Status |
|---|---|---|
| `/` | Hackathon submission landing (frozen) | live |
| `/f1_LLM_racing` | NEW — vision/roadmap/progress preview page | being built (0.2) |
| `/blog` | Submission blog | live |
| `/reset`, `/step`, `/health` | OpenEnv API | live |
| `/race/live/<id>` | Phase 7 live race viewer | not built |

When `v1.0` ships, `/f1_LLM_racing` is promoted to `/`. The current `/` becomes
`/submission` and is preserved for posterity.

---

## How to resume

1. Read this file top-to-bottom.
2. `git log --oneline -5` to see recent commits.
3. `git status` to see uncommitted work.
4. Look at the **Cursor** block above — that's literally where to start typing.
5. If `Blocked on:` is non-empty, resolve that first.

For higher-level context, read [`ROADMAP.md`](ROADMAP.md) and [`CLAUDE.md`](CLAUDE.md).
For frozen hackathon history, read [`STATE.md`](STATE.md).
