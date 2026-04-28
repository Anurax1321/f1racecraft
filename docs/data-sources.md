# Data Sources & Provenance

Complete provenance for every data input to the F1Racecraft environment.
Updated whenever a new source is added. Used for credibility, reproducibility,
and license compliance.

> **One-line rule:** every numeric calibration in the env should trace back to
> a primary source documented here. If you can't cite it, don't ship it.

---

## Primary data sources

### 1. FastF1 (Python library, MIT)

- **Repo:** [theOehrly/Fast-F1](https://github.com/theOehrly/Fast-F1)
- **Docs:** [docs.fastf1.dev](https://docs.fastf1.dev/)
- **License:** MIT — non-commercial / educational use is fine; commercial usage
  needs F1's blessing through proper licensing
- **What we pull:** lap timing, tire compound, stint, pit in/out, track status,
  weather, sector times, position
- **Coverage:** seasons 2018+ (we use **2023–2025** to match the current
  ground-effect era)
- **Cache:** `data/fastf1_cache/` (gitignored — ~500 MB for full 3-season pull)

### 2. racetrack-database (TUM, MIT)

- **Repo:** [TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database)
- **What we use:** track centerline CSVs for Monaco, Spa, Silverstone, Monza,
  Catalunya, Suzuka. See `data/tracks/`.
- **License:** MIT, attribution required.

### 3. Kaggle Formula 1 World Championship dataset

- **Source:** [Kaggle: Formula 1 World Championship 1950–2024](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- **What we use:** historical pace and stint calibration baselines (initial
  hackathon-era values; FastF1 will supersede these).
- **License:** CC0 (public domain).

---

## Academic priors driving our env design

These shape *what we extract* from FastF1 and *how we calibrate* the env. Every
calibration constant in `data/grounded/` should map back to one of these.

### A. Tire degradation modeling

- **Beatson (2025).** *A State-Space Approach to Modeling Tire Degradation in
  Formula 1 Racing.* [arXiv:2512.00640](https://arxiv.org/abs/2512.00640).
  - **Use:** lap times = f(fuel mass, latent tire pace); pit stops = state
    resets. We adopt this decomposition for fuel correction in
    `scripts/fastf1_extract.py`.
  - **Why it fits:** uses publicly available FastF1 timing data directly.

- **Mugge & Zandieh (ASU 2024).** *Using Python and Fast F1 to Pull and
  Analyze Data on Tire Degradation.*
  [PDF](https://cisa.asu.edu/sites/g/files/litvpz691/files/2024-04/Mugge_E_Zandieh.pdf).
  - **Use:** practical FastF1 extraction patterns; stint segmentation via the
    `Stint` column.

### B. Tire energy and feature importance

- **Aggarwal et al., Mercedes-AMG PETRONAS F1 (2025).** *Explainable Time
  Series Prediction of Tyre Energy in Formula One Race Strategy.*
  [arXiv:2501.04067](https://arxiv.org/abs/2501.04067) ·
  [ACM SIGAPP](https://dl.acm.org/doi/10.1145/3672608.3707765).
  - **Use:** feature importance ranking. Steering wheel angle is the #1
    predictor of tire energy; pit stop status is #2; speed-after-apex matters.
  - **Direct implication:** when we extract telemetry (Phase 2 stretch), we
    prioritize steering and throttle/brake over engine RPM.
  - **Note:** clockwise tracks burn front-left more than front-right. We don't
    model per-corner tire wear yet, but flagged for a future enhancement.

### C. Pit-stop decision modeling

- **Frontiers AI (2025).** *Data-driven pit stop decision support for
  Formula 1 using deep learning models.*
  [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full) ·
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12626961/).
  - **Use:** the feature set we'll mirror in extraction:
    `LapTime, Δlaptime, TyreLife, Stint, Compound, Position, RaceProgress,
    DriverAheadPit, DriverBehindPit, CumulativeTimeStint, TrackStatus`.
  - **Bi-LSTM result:** F1=0.81 on real F1 races for predicting pit decisions.
  - **Key empirical finding:** pit probability **2.6% → 5.7%** when the car
    *ahead* pits. We expose `DriverAheadPit` to the model so it can learn the
    undercut response.

### D. F1 reinforcement learning prior art

- **Thomas et al., Mercedes-AMG PETRONAS F1 (ACM SIGAPP 2025).**
  *Explainable Reinforcement Learning for Formula One Race Strategy.*
  [arXiv:2501.04068](https://arxiv.org/abs/2501.04068).
  - **Use:** confirms our terminal-reward design philosophy
    ([`docs/reward-philosophy.md`](reward-philosophy.md)).
  - Quote: *"Reward shaping is not used because we are unable to determine
    midway through a race whether a decision was good or not."*
  - Their action space: 4 discrete (no-pit, pit-S, pit-M, pit-H). Ours is
    richer (~20 commands including investigations + comms).

- **Optimum Racing (IJRASET 2025).** *F1 Strategy Predictor using
  Reinforcement Learning.*
  [paper](https://www.ijraset.com/research-paper/optimum-racing-a-f1-strategy-predictor-using-reinforcement-learning).
  - **Use:** confirms starting compound is fixed by the simulator, not the
    policy (we do the same — see ROADMAP Phase 5+ enhancement).

### E. Long-horizon LLM RL (broader context)

- **HCAPO (2026).** *Hindsight Credit Assignment for Long-Horizon LLM
  Agents.* [arXiv:2603.08754](https://arxiv.org/abs/2603.08754).
  - **Use:** earmarked Phase 3 upgrade path if 14B GRPO plateaus on long
    races (see [`ROADMAP.md`](../ROADMAP.md)).

### F. Reward hacking immunity

- **Anthropic (2025).** *Natural Emergent Misalignment from Reward Hacking
  in Production RL.* [PDF](https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf).
  - **Use:** "immunity by construction" stance; informed our over-pit cap fix
    in sub-step 1.5 and the one-shot revelation reward design.

---

## Domain references (qualitative grounding)

- **Slipstream vs. dirty air physics.** [Raceteq](https://www.raceteq.com/articles/2025/07/slipstream-vs-dirty-air-explained) ·
  [Halleys Clinic on Medium](https://medium.com/@halleysclinic/differences-between-slipstream-and-dirty-air-in-motorsports-complete-physics-c38a01b7f10f).
  - Used for: `compute_dirty_air_factor` calibration in `server/physics.py`.
  - Key fact: cars within 1s of the car ahead lose ~5–15% downforce.

- **DRS rules and timing.** [F1 Chronicle DRS guide](https://f1chronicle.com/what-is-formula-1-drs-formula-1-technology/).
  - Used for: when DRS becomes available (only after lap 2, only when within
    1s of car ahead, only in DRS zones).

- **Safety Car / VSC procedures.** [FIA Sporting Regulations](https://www.fia.com/regulation/category/110)
  (Article 55 covers SC, Article 56 covers VSC).
  - Used for: SC pit-loss reduction model (~5.5s SC pit vs ~22s green-flag).
  - Open source summary: [F1 Wiki: Virtual Safety Car](https://f1.fandom.com/wiki/Virtual_safety_car).

---

## What this gives us in `data/grounded/`

Every file in `data/grounded/<track>/` traces back to the sources above:

| File | What | Source |
|---|---|---|
| `tyres.json` | Per-compound degradation curves | FastF1 lap data + Beatson state-space framing |
| `pit.json` | Avg pit lane loss, typical pit windows, strategy distribution | FastF1 PitInTime/PitOutTime/Stint |
| `sc.json` | Real SC/VSC frequency + lap distribution | FastF1 track_status timeline |
| `weather.json` | Temp distribution, rainfall probability | FastF1 weather_data |
| `dirty_air.json` | Position-change rate (overtaking difficulty proxy) | FastF1 Position diffs per lap |

`data/grounded/_index.json` lists tracks with full coverage.

---

## License + attribution at deploy

When the env is deployed publicly (HF Space at v1.0), the README must include:

> Lap timing and telemetry sourced from
> [FastF1](https://github.com/theOehrly/Fast-F1) (MIT). F1 race data is the
> property of Formula One Management; this project is non-commercial /
> educational use only.

Track centerlines from [racetrack-database](https://github.com/TUMFTM/racetrack-database) (MIT, TUM Institute of Automotive Technology).

---

## Update protocol

Adding a new data source:
1. Add it to the relevant section above with a primary URL.
2. Note the license.
3. State **what we extract from it** and **where it lands in the env**.
4. If it's a paper: include the citation key and direct quote that justifies
   our design choice.
5. If it's regulatory/qualitative: include the regulation number or page.
