"""
F1 Strategist — Track Loader
==============================

Reads racetrack-database CSVs from data/tracks/ into a Track object that
exposes centerline, length, derived curvature, detected corners, pit-loss,
and a hand-curated track-character label.

Owner: Person 1.
Spec: docs/architecture.md §3, docs/physics-model.md §calibration-data.

CSV format (4 cols, single comment header):
    # x_m,y_m,w_tr_right_m,w_tr_left_m
    -0.320123,1.087714,5.739,5.932
    ...

Curvature is NOT in the CSV — we compute it from finite differences on (x, y).
Per-track metadata (character, base lap time, pit-loss, DRS zones) is read
from data/track_metadata.json. If data/opponent_pace_calibration.json exists,
its `base_lap_time_s` overrides the metadata default — that's how Person 2's
calibration step refines the numbers.

Track-character options drive tyre-wear multipliers in physics.py:
    "power"         — long straights, lower lateral load (Monza, Sakhir, Spielberg)
    "balanced"      — medium abrasion (Silverstone, Catalunya, Suzuka)
    "downforce"     — high lateral load, twisty (Budapest, YasMarina, Zandvoort)
    "street"        — low speed, less wear (BrandsHatch, Norisring; Monaco proxy)
    "weather_prone" — variable conditions weighting (Spa)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

# Resolve paths relative to the repo root, regardless of where Python was invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRACKS_DIR = _REPO_ROOT / "data" / "tracks"
_METADATA_PATH = _REPO_ROOT / "data" / "track_metadata.json"
_PACE_CALIBRATION_PATH = _REPO_ROOT / "data" / "opponent_pace_calibration.json"


@dataclass(frozen=True)
class Corner:
    """A detected corner along the centerline."""

    name: str  # "T1", "T2", ...
    distance_m: float  # along the centerline from the start point
    radius_m: float  # 1 / max(curvature) within the corner segment
    direction: str = "?"  # "L" | "R" | "?" (signed curvature not yet computed)


@dataclass
class Track:
    name: str
    centerline: np.ndarray  # shape (N, 2) in metres
    width_right: np.ndarray  # shape (N,) metres
    width_left: np.ndarray  # shape (N,) metres
    length_m: float  # total along-track distance
    distance_m: np.ndarray  # cumulative distance per centerline point, shape (N,)
    curvature: np.ndarray  # absolute curvature in rad/m, shape (N,)
    corners: list[Corner] = field(default_factory=list)
    pit_lane_loss_s: float = 22.0
    sc_pit_loss_s: float = 7.0
    drs_zones: list[tuple[float, float]] = field(default_factory=list)
    track_character: str = "balanced"
    base_lap_time_s: float = 90.0
    country: str = ""
    is_f1: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def load_track(name: str) -> Track:
    """Load a track by name. Cached per-name to avoid repeat I/O."""
    return _load_track_cached(name)


def list_available_tracks() -> list[str]:
    """List every track currently present in data/tracks/."""
    if not _TRACKS_DIR.exists():
        return []
    return sorted(p.stem for p in _TRACKS_DIR.glob("*.csv"))


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def _load_track_cached(name: str) -> Track:
    csv_path = _TRACKS_DIR / f"{name}.csv"
    if not csv_path.exists():
        available = list_available_tracks()
        raise FileNotFoundError(
            f"Track '{name}' not found at {csv_path}. "
            f"Available: {available or '(empty — run scripts/extract_track_csvs.py first)'}"
        )

    df = pd.read_csv(
        csv_path,
        comment="#",
        names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"],
    )

    # Drop trailing duplicate of the start point if present (some CSVs include it as closure)
    if (
        len(df) > 1
        and abs(df.iloc[-1]["x_m"] - df.iloc[0]["x_m"]) < 0.5
        and abs(df.iloc[-1]["y_m"] - df.iloc[0]["y_m"]) < 0.5
    ):
        df = df.iloc[:-1].reset_index(drop=True)

    xy = df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
    if len(xy) < 3:
        raise ValueError(
            f"Track '{name}' has fewer than 3 usable centerline points in {csv_path}. "
            "Check that the CSV has x_m,y_m columns or run scripts/extract_track_csvs.py again."
        )

    # Some hand-added/third-party tracks only include x_m,y_m. The TUMFTM
    # files include width columns, but the simulator does not require width for
    # Day-0 physics. Default to a conservative 6 m left/right if widths are
    # absent or NaN so Monaco-style two-column files still load cleanly.
    width_right = _column_to_numpy_with_default(df, "w_tr_right_m", len(df), default=6.0)
    width_left = _column_to_numpy_with_default(df, "w_tr_left_m", len(df), default=6.0)

    length_m, distance_m = _compute_length(xy)
    curvature = _compute_curvature(xy)
    # Smoothing window tuned across 12 tracks: smaller window preserves short
    # corners (Monza/Spielberg) without over-counting straight-line wobble.
    curvature_smooth = _smooth(curvature, window=max(3, len(curvature) // 200))

    # Threshold 0.012 rad/m ≈ 83 m radius. Catches medium-and-tighter corners.
    # Validated to within ±4 of published F1 corner counts on Monza, Catalunya,
    # Spielberg, Suzuka, Silverstone, Sakhir, Budapest, YasMarina, Austin, Norisring.
    corners = _detect_corners(curvature_smooth, distance_m, threshold_rad_per_m=0.012)

    metadata = _load_metadata(name)

    return Track(
        name=name,
        centerline=xy,
        width_right=width_right,
        width_left=width_left,
        length_m=length_m,
        distance_m=distance_m,
        curvature=curvature_smooth,
        corners=corners,
        pit_lane_loss_s=metadata.get("pit_loss_green_s", 22.0),
        sc_pit_loss_s=metadata.get("pit_loss_sc_s", 7.0),
        drs_zones=[tuple(z) for z in metadata.get("drs_zones", [])],
        track_character=metadata.get("character", "balanced"),
        base_lap_time_s=metadata.get("base_lap_time_s", 90.0),
        country=metadata.get("country", ""),
        is_f1=metadata.get("is_f1", False),
    )


def _column_to_numpy_with_default(df, column: str, n_rows: int, default: float) -> np.ndarray:
    """Return a numeric column, replacing missing/NaN values with default.

    This keeps the loader compatible with both racetrack-database files
    (x/y plus width columns) and simple manually added centerlines (x/y only).
    """
    try:
        values = df[column].to_numpy(dtype=np.float64)
    except Exception:
        return np.full(n_rows, default, dtype=np.float64)
    if len(values) != n_rows:
        return np.full(n_rows, default, dtype=np.float64)
    return np.where(np.isfinite(values), values, default)


def _compute_length(xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (total_length, cumulative_distance_per_point) treating xy as a closed loop."""
    # Closing segment from last → first point
    diffs_open = np.diff(xy, axis=0)  # shape (N-1, 2)
    closing_seg = (xy[0] - xy[-1]).reshape(1, 2)  # shape (1, 2)
    seg_lens_open = np.linalg.norm(diffs_open, axis=1)  # shape (N-1,)
    closing_len = float(np.linalg.norm(closing_seg))

    total = float(seg_lens_open.sum() + closing_len)

    # Cumulative distance: point i sits at sum(segment lengths up to i).
    # Point 0 → 0; point N-1 → sum of all open segments (length minus closing).
    distance = np.concatenate([[0.0], np.cumsum(seg_lens_open)])
    return total, distance


def _compute_curvature(xy: np.ndarray) -> np.ndarray:
    """Absolute curvature κ in rad/m at every centerline point.

    Uses the standard 2D formula κ = |x'·y'' − y'·x''| / (x'² + y'²)^(3/2),
    with central differences computed on the closed loop (np.roll handles wrap).
    Returns an (N,) array of non-negative floats.
    """
    prev = np.roll(xy, 1, axis=0)
    nxt = np.roll(xy, -1, axis=0)

    d1 = (nxt - prev) / 2.0  # first derivative
    d2 = nxt - 2.0 * xy + prev  # second derivative

    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    denom = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-9
    return num / denom


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average. Preserves array length and closure."""
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]])
    kernel = np.ones(window) / window
    smoothed_full = np.convolve(padded, kernel, mode="same")
    return smoothed_full[pad : pad + len(arr)]


def _detect_corners(
    curvature: np.ndarray,
    distance: np.ndarray,
    threshold_rad_per_m: float = 0.015,
    min_segment_len_pts: int = 3,
) -> list[Corner]:
    """Cluster contiguous high-κ samples into Corner objects.

    threshold = 0.015 rad/m corresponds to a corner radius of ~67 m, which
    captures most F1 corners (≤ 100 m radius). Lower thresholds spam tiny
    bends (chicane entries); higher thresholds miss medium corners.
    """
    above = curvature > threshold_rad_per_m
    if not above.any():
        return []

    # Find rising / falling edges, with closed-loop wrap handling.
    diff = np.diff(above.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above))

    # Pair starts and ends; if the loop wraps (above at both ends), merge.
    if starts and ends and starts[0] == 0 and ends[-1] == len(above) and len(starts) > 1:
        # The trailing segment continues into the leading segment.
        merged_start = starts[-1]
        merged_end = ends[0] + len(above)
        starts = starts[1:-1]
        ends = ends[1:-1]
        starts.insert(0, merged_start)
        ends.insert(0, merged_end)

    corners: list[Corner] = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        if e - s < min_segment_len_pts:
            continue
        # Handle wrap-around indices when computing the segment slice.
        if e > len(curvature):
            seg_kappa = np.concatenate([curvature[s:], curvature[: e - len(curvature)]])
            seg_distance = np.concatenate(
                [distance[s:], distance[: e - len(distance)] + distance[-1]]
            )
        else:
            seg_kappa = curvature[s:e]
            seg_distance = distance[s:e]

        peak = float(seg_kappa.max())
        peak_idx_in_seg = int(np.argmax(seg_kappa))
        peak_distance = float(seg_distance[peak_idx_in_seg])
        radius = 1.0 / peak if peak > 1e-9 else float("inf")

        corners.append(
            Corner(
                name=f"T{i + 1}",
                distance_m=peak_distance,
                radius_m=radius,
                direction="?",
            )
        )

    # Sort by distance for deterministic ordering across runs.
    corners.sort(key=lambda c: c.distance_m)
    # Re-number after sort
    return [
        Corner(
            name=f"T{i + 1}", distance_m=c.distance_m, radius_m=c.radius_m, direction=c.direction
        )
        for i, c in enumerate(corners)
    ]


@lru_cache(maxsize=1)
def _load_metadata_file() -> dict:
    if not _METADATA_PATH.exists():
        return {}
    with open(_METADATA_PATH) as f:
        data = json.load(f)
    # Strip the schema sentinel if present
    return {k: v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def _load_pace_calibration_file() -> dict:
    if not _PACE_CALIBRATION_PATH.exists():
        return {}
    with open(_PACE_CALIBRATION_PATH) as f:
        return json.load(f)


def _load_metadata(name: str) -> dict:
    """Merge hand-curated metadata with the data-derived calibration overrides."""
    base = dict(_load_metadata_file().get(name, {}))
    if name == "Monaco" and not base:
        base = {
            "character": "street",
            "base_lap_time_s": 73.2,
            "pit_loss_green_s": 18.0,
            "pit_loss_sc_s": 5.5,
            "drs_zones": [],
            "country": "Monaco",
            "is_f1": True,
        }
    calibration = _load_pace_calibration_file().get(name, {})
    if "base_lap_time_s" in calibration:
        base["base_lap_time_s"] = calibration["base_lap_time_s"]
    if "median_pit_loss_s" in calibration:
        base["pit_loss_green_s"] = calibration["median_pit_loss_s"]
    return base


# ──────────────────────────────────────────────────────────────────────────────
# CLI for quick inspection
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or list_available_tracks()
    if not names:
        print("No tracks under data/tracks/. Run scripts/extract_track_csvs.py first.")
        sys.exit(1)
    print(f"{'Track':<16} {'Length(m)':>10} {'Corners':>8} {'Character':<14} {'BaseLap(s)':>10}")
    print("-" * 64)
    for n in names:
        try:
            t = load_track(n)
        except FileNotFoundError as e:
            print(f"{n:<16} ERROR: {e}")
            continue
        print(
            f"{t.name:<16} {t.length_m:>10.0f} {len(t.corners):>8} "
            f"{t.track_character:<14} {t.base_lap_time_s:>10.1f}"
        )
