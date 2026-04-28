"""
Copy all 25 tracks from racetrack-database-master/tracks/ into data/tracks/.

Owner: Person 1 (Phase 0).

The TUMFTM racetrack-database has TWO data dirs per track:
    - tracks/<n>.csv     — centerline + widths (4 cols: x_m, y_m, w_tr_right_m, w_tr_left_m)
    - racelines/<n>.csv  — optimised racing line (2 cols: x_m, y_m)

We copy from `tracks/` because the widths let the visualiser render the
actual track polygon (not just a centerline). Header line `# x_m,...` is
preserved in the destination — server/track.py uses pandas.read_csv with
comment='#'.

License: LGPL-3.0 (see racetrack-database-master/LICENSE).

The 25 tracks split as:
    18 F1 venues:  Austin, Catalunya, Hockenheim, Melbourne, MexicoCity, Montreal,
                   Monza, Sakhir, SaoPaulo, Sepang, Shanghai, Silverstone, Sochi,
                   Spa, Spielberg, Suzuka, YasMarina, Zandvoort
    7 non-F1:      BrandsHatch, Budapest (F1 — Hungary), IMS (IndyCar),
                   MoscowRaceway, Norisring, Nuerburgring, Oschersleben
                   (Budapest IS F1 - Hungaroring; the 6 others are DTM/IndyCar)
    MISSING:       Monaco (no equivalent in dataset; use BrandsHatch as street proxy)

CLI:
    python scripts/extract_track_csvs.py
    python scripts/extract_track_csvs.py --source /path/to/racetrack-database-master
    python scripts/extract_track_csvs.py --only Monza Spa Silverstone   # subset
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="racetrack-database-master",
        help="Path to extracted racetrack-database-master/ directory",
    )
    parser.add_argument(
        "--dest",
        default="data/tracks",
        help="Destination directory for normalised CSVs",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of track names to extract (default: all)",
    )
    args = parser.parse_args()

    src_dir = Path(args.source) / "tracks"
    dst_dir = Path(args.dest)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"ERROR: {src_dir} not found.", file=sys.stderr)
        print(
            f"Hint: unzip racetrack-database-master.zip in the repo root so "
            f"that {args.source}/tracks/ exists.",
            file=sys.stderr,
        )
        sys.exit(1)

    available = sorted(p.stem for p in src_dir.glob("*.csv"))
    if not available:
        print(f"ERROR: no CSVs found under {src_dir}", file=sys.stderr)
        sys.exit(1)

    if args.only:
        missing = [n for n in args.only if n not in available]
        if missing:
            print(f"ERROR: requested tracks not in dataset: {missing}", file=sys.stderr)
            print(f"Available: {', '.join(available)}", file=sys.stderr)
            sys.exit(1)
        targets = args.only
    else:
        targets = available

    n_copied = 0
    for name in targets:
        src = src_dir / f"{name}.csv"
        dst = dst_dir / f"{name}.csv"
        shutil.copy2(src, dst)
        n_copied += 1
        print(f"  ok: {name}.csv")

    print(f"\nCopied {n_copied} track(s) into {dst_dir}/")
    print(
        'Next: python -c "from server.track import load_track; '
        "print(load_track('Monza').length_m)\""
    )


if __name__ == "__main__":
    main()
