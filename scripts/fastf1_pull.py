#!/usr/bin/env python
"""Walk the F1 calendars for 2023-2025 and populate the FastF1 cache.

Idempotent — FastF1's own cache layer dedupes. Skips races that already have
cached data. Resumable: kill it any time and rerun.

Usage:
    python scripts/fastf1_pull.py                # all years, all races
    python scripts/fastf1_pull.py --year 2024    # one season
    python scripts/fastf1_pull.py --smoke        # 1 race only, for verification
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.fastf1_loader import (
    RACE_YEARS,
    enable_cache,
    list_races,
    load_race,
    session_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None, help="Limit to one season")
    parser.add_argument("--smoke", action="store_true", help="Pull a single race for verification")
    parser.add_argument("--telemetry", action="store_true", help="Also pull tick-level telemetry (10x cache size)")
    args = parser.parse_args()

    cache_dir = enable_cache()
    print(f"FastF1 cache: {cache_dir}")

    if args.smoke:
        plan = [(2024, "Monaco")]
    else:
        years = (args.year,) if args.year else RACE_YEARS
        plan: list[tuple[int, str]] = []
        for year in years:
            try:
                races = list_races(year)
                for race in races:
                    plan.append((year, race["gp_name"]))
            except Exception as exc:
                print(f"  ⚠️  could not list races for {year}: {exc}")

    print(f"Plan: {len(plan)} races")
    print()

    summaries: list[dict] = []
    for i, (year, gp) in enumerate(plan, 1):
        t0 = time.time()
        try:
            session = load_race(year, gp, telemetry=args.telemetry)
            meta = session_metadata(session)
            elapsed = time.time() - t0
            mark = "✓"
            note = (
                f"laps={meta.total_laps} drivers={meta.n_drivers} "
                f"compounds={','.join(meta.compounds_used)} "
                f"{'WET' if meta.wet_race else 'dry'} "
                f"sc_events={meta.n_track_status_events}"
            )
            summaries.append({
                "year": year, "gp": gp, "ok": True, "elapsed_s": round(elapsed, 1),
                **{k: v for k, v in meta.__dict__.items() if k != "error"},
            })
        except Exception as exc:
            elapsed = time.time() - t0
            mark = "✗"
            note = f"FAILED: {type(exc).__name__}: {exc}"
            summaries.append({
                "year": year, "gp": gp, "ok": False, "elapsed_s": round(elapsed, 1),
                "error": str(exc),
            })
        print(f"[{i:>3}/{len(plan)}] {mark} {year} {gp:<25} {elapsed:>5.1f}s  {note}")

    # Cache size
    cache_size_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1024**2
    n_ok = sum(1 for s in summaries if s["ok"])
    n_fail = len(summaries) - n_ok

    print()
    print(f"Done. {n_ok}/{len(plan)} races cached. {n_fail} failed.")
    print(f"Cache size: {cache_size_mb:.1f} MB")

    # Save manifest for downstream extraction scripts
    manifest_path = cache_dir.parent / "fastf1_pull_manifest.json"
    manifest_path.write_text(json.dumps({
        "summaries": summaries,
        "cache_size_mb": round(cache_size_mb, 1),
        "n_ok": n_ok,
        "n_fail": n_fail,
    }, indent=2))
    print(f"Manifest: {manifest_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
