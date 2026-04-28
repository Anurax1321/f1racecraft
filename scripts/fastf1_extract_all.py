#!/usr/bin/env python
"""Extract per-race features from every cached FastF1 session.

Reads from `data/fastf1_pull_manifest.json` (written by `fastf1_pull.py`),
loads each cached race, runs `extract_race_features()`, and writes one JSON
per race to `data/grounded/raw/<year>/<round>_<short>.json`.

Idempotent — skips races whose output JSON already exists. Use `--force` to
re-extract everything.

Usage:
    python scripts/fastf1_extract_all.py             # all cached races
    python scripts/fastf1_extract_all.py --year 2024 # one season
    python scripts/fastf1_extract_all.py --force     # re-extract everything
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.fastf1_extract import extract_race_features
from server.fastf1_loader import RAW_OUT_DIR, load_race


def _short_name(gp_name: str) -> str:
    """Convert 'Monaco Grand Prix' → 'monaco'."""
    name = gp_name.lower().replace("grand prix", "").strip()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = Path("data/fastf1_pull_manifest.json")
    if not manifest_path.exists():
        print(f"❌ {manifest_path} missing — run scripts/fastf1_pull.py first")
        return 1
    manifest = json.loads(manifest_path.read_text())
    summaries = [s for s in manifest["summaries"] if s.get("ok")]

    if args.year:
        summaries = [s for s in summaries if s["year"] == args.year]

    print(f"Plan: extract features from {len(summaries)} races")
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = n_fail = 0
    for i, s in enumerate(summaries, 1):
        year = s["year"]
        gp = s["gp"]
        round_n = s.get("round", 0)
        short = _short_name(gp)
        out_dir = RAW_OUT_DIR / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{round_n:02d}_{short}.json"

        if out_path.exists() and not args.force:
            print(f"[{i:>3}/{len(summaries)}] ⏭ {year} {gp:<25} (already extracted)")
            n_skip += 1
            continue

        t0 = time.time()
        try:
            session = load_race(year, gp)
            features = extract_race_features(session)
            out_path.write_text(json.dumps(features, indent=1))
            elapsed = time.time() - t0
            n_laps = len(features["lap_records"])
            n_pits = len(features["pit_decisions"])
            wet = "WET" if features["metadata"]["wet_race"] else "dry"
            size_mb = out_path.stat().st_size / 1024**2
            print(f"[{i:>3}/{len(summaries)}] ✓ {year} {gp:<25} {elapsed:>4.1f}s "
                  f"laps={n_laps:>4} pits={n_pits:>2} {wet} ({size_mb:.2f} MB)")
            n_done += 1
        except Exception as exc:
            print(f"[{i:>3}/{len(summaries)}] ✗ {year} {gp:<25} {type(exc).__name__}: {exc}")
            n_fail += 1

    total_size_mb = sum(p.stat().st_size for p in RAW_OUT_DIR.rglob("*.json")) / 1024**2
    print()
    print(f"Done. extracted={n_done}  skipped={n_skip}  failed={n_fail}")
    print(f"Total raw JSON size: {total_size_mb:.1f} MB at {RAW_OUT_DIR}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
