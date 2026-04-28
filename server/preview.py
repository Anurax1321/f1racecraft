"""
Preview page for the F1 LLM Racing personal project.

Mounted at `/f1_LLM_racing` while the project is under construction. When the
project ships v1.0, the same render function is mounted at `/` instead — see
`server/app.py` for the one-line swap.

The page reads the live phase status from `STATUS.md` so the public-facing
progress is always in sync with the cursor file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from fastapi.responses import HTMLResponse


_REPO_ROOT = Path(__file__).parent.parent
_STATUS_PATH = _REPO_ROOT / "STATUS.md"
_ROADMAP_PATH = _REPO_ROOT / "ROADMAP.md"


# ---------------------------------------------------------------------------
# Status parsing — pull the cursor + phase table out of STATUS.md
# ---------------------------------------------------------------------------

def _parse_cursor() -> dict[str, str]:
    """Extract the ```cursor``` fenced block from STATUS.md."""
    if not _STATUS_PATH.exists():
        return {}
    text = _STATUS_PATH.read_text(encoding="utf-8")
    m = re.search(r"```\n(Phase:.*?)\n```", text, re.DOTALL)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _parse_phase_table() -> List[Tuple[str, str, str]]:
    """Pull (number, name, status_emoji) for each phase from STATUS.md."""
    if not _STATUS_PATH.exists():
        return []
    text = _STATUS_PATH.read_text(encoding="utf-8")
    rows: List[Tuple[str, str, str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Phase | Status |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) >= 2 and "—" in cols[0]:
                num, name = cols[0].split("—", 1)
                rows.append((num.strip(), name.strip(), cols[1]))
    return rows


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------

def render_preview_page() -> HTMLResponse:
    cursor = _parse_cursor()
    phases = _parse_phase_table()

    current_phase = cursor.get("phase", "0")
    current_substep = cursor.get("sub-step", "0.1")
    next_action = cursor.get("next action", "—")

    phase_rows_html = ""
    for num, name, status in phases:
        emoji_class = "done" if "✅" in status or "🟢" in status \
            else "active" if "🟡" in status \
            else "pending"
        emoji = "●" if emoji_class == "done" else ("◐" if emoji_class == "active" else "○")
        phase_rows_html += (
            f'<div class="phase-row {emoji_class}">'
            f'<span class="phase-num">{num}</span>'
            f'<span class="phase-dot">{emoji}</span>'
            f'<span class="phase-name">{name}</span>'
            f'</div>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>F1 LLM Racing — Vision & Roadmap</title>
<style>
  :root {{
    --bg:        #0a0a0a;
    --bg-2:      #111;
    --ink:       #eee;
    --dim:       #888;
    --dimmer:    #555;
    --red:       #e10600;
    --teal:      #00d2be;
    --gold:      #ffd700;
    --rule:      #1e1e1e;
    --rule-2:    #2a2a2a;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: var(--bg); color: var(--ink); }}
  body {{
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
    line-height: 1.65;
    max-width: 960px;
    margin: 0 auto;
    padding: 56px 28px 120px;
  }}

  /* ── Header ────────────────────────────── */
  .preview-banner {{
    display: inline-block;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    border: 1px solid var(--gold);
    padding: 4px 10px;
    margin-bottom: 24px;
  }}
  h1.title {{
    font-size: 38px;
    color: var(--red);
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 8px;
  }}
  h1.title .accent {{ color: var(--ink); }}
  .tagline {{
    font-size: 15px;
    color: var(--dim);
    margin-bottom: 40px;
  }}

  /* ── Sections ────────────────────────── */
  h2.section {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--ink);
    margin: 56px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--rule);
  }}
  h2.section .num {{ color: var(--red); margin-right: 12px; }}
  p {{ font-size: 14px; color: #ccc; margin-bottom: 12px; }}
  p strong {{ color: var(--ink); }}
  p code {{ background: var(--bg-2); color: var(--teal); padding: 2px 6px; font-size: 12px; }}

  /* ── Vision card ─────────────────────── */
  .vision-card {{
    background: linear-gradient(135deg, #131313 0%, #0c0c0c 100%);
    border: 1px solid var(--rule-2);
    border-left: 3px solid var(--red);
    padding: 24px 28px;
    margin: 24px 0;
  }}
  .vision-card .quote {{
    font-size: 17px;
    color: var(--ink);
    font-style: italic;
    line-height: 1.55;
  }}
  .vision-card .by {{
    font-size: 11px;
    color: var(--dim);
    margin-top: 12px;
    letter-spacing: 0.05em;
  }}

  /* ── Pillars grid ───────────────────── */
  .pillars {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 24px 0;
  }}
  .pillar {{
    background: var(--bg-2);
    border: 1px solid var(--rule);
    padding: 18px 20px;
  }}
  .pillar .icon {{
    color: var(--teal);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .pillar .title {{ color: var(--ink); font-size: 15px; margin-bottom: 6px; }}
  .pillar .desc {{ color: var(--dim); font-size: 13px; line-height: 1.5; }}

  /* ── Roadmap ────────────────────────── */
  .roadmap {{ margin: 16px 0; }}
  .phase-row {{
    display: grid;
    grid-template-columns: 40px 30px 1fr 100px;
    align-items: center;
    padding: 10px 14px;
    border-left: 2px solid var(--rule);
    margin-bottom: 2px;
    transition: background 0.15s;
  }}
  .phase-row:hover {{ background: var(--bg-2); }}
  .phase-row.done {{ border-left-color: var(--teal); }}
  .phase-row.active {{ border-left-color: var(--gold); background: #181410; }}
  .phase-row.pending {{ border-left-color: var(--rule); }}
  .phase-num {{ color: var(--dim); font-size: 12px; }}
  .phase-dot {{ font-size: 14px; }}
  .phase-row.done .phase-dot {{ color: var(--teal); }}
  .phase-row.active .phase-dot {{ color: var(--gold); }}
  .phase-row.pending .phase-dot {{ color: var(--dimmer); }}
  .phase-name {{ font-size: 13.5px; color: var(--ink); }}
  .phase-row.pending .phase-name {{ color: var(--dim); }}

  /* ── Progress ───────────────────────── */
  .progress-card {{
    background: var(--bg-2);
    border: 1px solid var(--rule-2);
    padding: 24px 28px;
    margin: 16px 0;
    font-size: 13px;
  }}
  .progress-row {{ display: flex; gap: 12px; margin-bottom: 8px; }}
  .progress-row .label {{
    color: var(--dim);
    width: 110px;
    flex-shrink: 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 10.5px;
    padding-top: 2px;
  }}
  .progress-row .value {{ color: var(--ink); }}
  .progress-row .value.highlight {{ color: var(--teal); }}

  /* ── Footer / links ─────────────────── */
  .links {{
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 48px;
    padding-top: 32px;
    border-top: 1px solid var(--rule);
    font-size: 12px;
  }}
  .links a {{
    color: var(--dim);
    text-decoration: none;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transition: color 0.15s;
  }}
  .links a:hover {{ color: var(--red); }}

  .footnote {{
    margin-top: 48px;
    color: var(--dimmer);
    font-size: 11px;
    text-align: center;
    letter-spacing: 0.05em;
  }}
</style>
</head>
<body>

<span class="preview-banner">▲ Preview · Under construction</span>

<h1 class="title">F1 LLM Racing<br><span class="accent">— a complete race, simulated.</span></h1>
<p class="tagline">A personal project to simulate a full F1 grand prix where every car is a language model with its own personality.</p>

<!-- ── VISION ───────────────────────────── -->
<h2 class="section"><span class="num">01</span>The vision</h2>

<div class="vision-card">
  <div class="quote">
    "Twenty cars. Twenty AI strategists. One race. Each car has its own personality, makes its own calls, races on real F1 data, and you can watch it happen live with commentary. The race is complete: 55 laps, real weather, real tyre cliffs, real safety cars. Nothing scripted, everything emergent."
  </div>
  <div class="by">— the goal</div>
</div>

<div class="pillars">
  <div class="pillar">
    <div class="icon">▲ Multi-agent</div>
    <div class="title">All 20 cars are LLMs</div>
    <div class="desc">Five distinct personas (Aggressor, Cautious, Veteran, Rookie, Wildcard) sharing one base model, swapped via LoRA per car.</div>
  </div>
  <div class="pillar">
    <div class="icon">▲ Real data</div>
    <div class="title">Grounded in FastF1</div>
    <div class="desc">Tyre curves, fuel burn, weather, safety-car frequency — all calibrated against 2023–2025 telemetry.</div>
  </div>
  <div class="pillar">
    <div class="icon">▲ Long-horizon</div>
    <div class="title">Full grand prix length</div>
    <div class="desc">50–70 laps per race. Multi-stop strategy. Lap-4 mistakes show up at lap 49. The kind of horizon LLMs handle worst.</div>
  </div>
  <div class="pillar">
    <div class="icon">▲ Watchable</div>
    <div class="title">Live UI + commentary</div>
    <div class="desc">Track map, live position deltas, tyre wear bars. A commentator LLM narrates each lap. Optional TTS for a true broadcast feel.</div>
  </div>
</div>

<!-- ── ROADMAP ──────────────────────────── -->
<h2 class="section"><span class="num">02</span>The roadmap</h2>

<p>Ten phases. Each one ends with a working, shippable thing. Full plan in <code>ROADMAP.md</code>.</p>

<div class="roadmap">
{phase_rows_html}</div>

<!-- ── PROGRESS ─────────────────────────── -->
<h2 class="section"><span class="num">03</span>Where we are right now</h2>

<div class="progress-card">
  <div class="progress-row">
    <div class="label">Phase</div>
    <div class="value highlight">{current_phase}</div>
  </div>
  <div class="progress-row">
    <div class="label">Sub-step</div>
    <div class="value">{current_substep}</div>
  </div>
  <div class="progress-row">
    <div class="label">Next</div>
    <div class="value">{next_action}</div>
  </div>
  <div class="progress-row">
    <div class="label">Hardware</div>
    <div class="value">RTX 5090 · 32 GB VRAM · single box</div>
  </div>
  <div class="progress-row">
    <div class="label">Owner</div>
    <div class="value">Anurag Chinnaboina · solo</div>
  </div>
</div>

<!-- ── HOW IT WORKS ─────────────────────── -->
<h2 class="section"><span class="num">04</span>How it works, briefly</h2>

<p>Every car reads the race state, picks one of ~20 strategic commands (<code>PIT_NOW</code>, <code>SET_MODE push</code>, <code>RADIO_DRIVER</code>, etc.), and the simulator advances the world. The model doesn't drive the car — it's the strategist on the pit wall. A deterministic Python scorer rates the race across six dimensions (final position, strategic decisions, tyre management, fuel, comms, operational efficiency).</p>

<p>The model learns in three stages: <strong>imitate</strong> (SFT on expert traces) → <strong>trial &amp; reward</strong> (GRPO on simulated races) → <strong>race each other</strong> (self-play, all five personas co-evolving).</p>

<p>All five personas share the same base brain (one ~8 GB Qwen3-14B in 4-bit). Each persona is a tiny ~30 MB LoRA patch on top — same eyes, different lens.</p>

<!-- ── LINKS ────────────────────────────── -->
<div class="links">
  <a href="/blog">Blog</a>
  <a href="/">Hackathon submission</a>
  <a href="https://github.com/Deltasthicc/F1_Simulator_OpenENV" target="_blank">GitHub</a>
  <a href="https://huggingface.co/Deltasthic/f1-strategist-qwen3-4b-grpo" target="_blank">Model</a>
  <a href="https://huggingface.co/spaces/Deltasthic/f1-strategist" target="_blank">HF Space</a>
</div>

<div class="footnote">
  This preview page lives at <code>/f1_LLM_racing</code>. When v1.0 ships it gets promoted to <code>/</code>.<br>
  Last updated from STATUS.md · {len(phases)} phases tracked
</div>

</body>
</html>
"""
    return HTMLResponse(content=html)
