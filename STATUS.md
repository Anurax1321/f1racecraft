# F1 LLM Racing — Current Cursor

> **How to use this file:** if you (or a fresh Claude session) are picking
> up this project, read this top-to-bottom. It tells you exactly where to
> resume. Update the **Cursor** block every time you finish a sub-step.

---

## Cursor

```
Phase:        0 (Foundation cleanup)
Sub-step:     0.8 (next — tag v0.2-foundation; close out Phase 0)
Last commit:  a18aa20 (updated and step 0 in progress) on origin/dev
Next action:  Commit pending changes, then `git tag v0.2-foundation && git push --tags`
Blocked on:   nothing
Last session: 2026-04-28
Owner:        Anurag (solo personal project, post-hackathon)
```

## Server ops — dev reload + production auto-restart

**Dev mode (auto-reload on file change):**
```bash
F1_DEV_MODE=1 /home/anurag/.virtualenvs/f1-strategist/bin/python -m uvicorn \
  server.app:app --host 127.0.0.1 --port 8765 --reload
```
`F1_DEV_MODE=1` skips the 30s Qwen3-0.6B preload (each reload would re-trigger it).

**Production (auto-restart if it dies, auto-start on boot):**

A systemd unit lives at `deploy/f1-strategist.service`. Install once:
```bash
sudo cp deploy/f1-strategist.service /etc/systemd/system/f1racecraft.service
sudo systemctl daemon-reload
sudo systemctl enable --now f1racecraft.service
sudo systemctl status f1racecraft.service     # check it's running
journalctl -u f1racecraft.service -f          # follow logs
```

After install, kill the manually-launched uvicorn so systemd takes over:
```bash
pkill -f "uvicorn server.app:app"             # systemd will respawn from its own ExecStart
```

The unit has `Restart=always` (3-second backoff) and `ExecStartPre=git pull` so
deploys = `git push` + a systemd restart. To turn that off, edit the unit and
remove the two `ExecStartPre=...git...` lines.

## Pending commit (Claude has staged changes; user to commit)

Author attribution updated everywhere: **Anurag Chinnaboina, Shashwat Rajan**
(Tanish removed). README rewritten for readability. `hfspace` remote removed.
HF Space discoverability decision deferred to Phase 9.6.

Files touched this session (not yet committed):
- `LICENSE` — copyright line
- `pyproject.toml` — authors + URLs
- `README.md` — full rewrite, much shorter
- `blog.md` — Authors line
- `openenv.yaml` — author line
- `CLAUDE.md` — Team line
- `server/static/index.html` — frontend authors (footer + team section)
- `ROADMAP.md` — added Phase 6.5 (Continual Learning) + Phase 9.6 (HF Space decision)
- `STATUS.md` — this file
- `docs/postmortem-ablation.md` — sub-step 0.3 verdict (new file)
- `server/preview.py` — preview page module (new file)
- `server/app.py` — `/f1_LLM_racing` route mounted; dropped `/web` redirect, `/demo` Gradio mount, `ENABLE_WEB_INTERFACE` block; added `F1_DEV_MODE=1` to skip Qwen3 preload during reload-dev mode
- `deploy/f1-strategist.service` — description rebranded to F1Racecraft
- `.gitignore` — model checkpoint dirs excluded for clean LFS-free repo
- `results/ablation_hard_*.json/png` — sub-step 0.3 artifacts

Files **not** touched (frozen historical):
`STATE.md`, `TODO.md`, `GPU_HANDOFF.md`, `PRE_PUSH_CHECKLIST.md`,
`demo-assets/*`, `docs/person*-tasks.md`, `docs/build-order.md`, the Colab
notebook. These document the hackathon-as-it-was; don't rewrite history.

Suggested commit message:
> `chore: rebrand to F1Racecraft; rewrite README; remove hfspace remote`

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
| 0.4 | Strip 110 MB tokenizer cruft from git | ✅ skipped — new repo never committed it (Option C migration) |
| 0.5 | Decide & act on Gradio `/web` | ✅ done — dropped (openenv shim doesn't support it; SSE `/dashboard` planned for Phase 7 instead). Code cleanup applied to `server/app.py`. |
| 0.6 | Re-run Colab on T4 | ⏭️ deferred — hackathon artifact, not core to the vision. URLs patched so it's not broken; verification parked in `Deferred / Parking Lot` (ROADMAP.md). |
| 0.7 | Add `scripts/` wrappers | ✅ done — `serve.sh`, `eval.sh`, `train.sh`, `race.sh`. Idempotent. README updated. |
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
| `/` | Hackathon submission landing (frozen, attribution updated) | live |
| `/f1_LLM_racing` | Vision/roadmap/progress preview page | live ✅ |
| `/blog` | Submission blog | live |
| `/reset`, `/step`, `/health` | OpenEnv API | live |
| `/race/live/<id>` | Phase 7 live race viewer | not built |

## Remotes

| Name | URL | Purpose |
|---|---|---|
| `origin` | `git@github.com:Anurax1321/f1racecraft.git` | active personal repo |
| `deltasthicc` | `Deltasthicc/F1_Simulator_OpenENV` | frozen hackathon archive |
| ~~`hfspace`~~ | ~~`Deltasthic/f1-strategist` HF Space~~ | **removed 2026-04-28** — Space frozen as hackathon artifact, not pushed to from this repo anymore. Revisit at Phase 9.6 (own Space at v1.0). |

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
