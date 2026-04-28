"""Append-only postmortem memory for completed episodes."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


class PostmortemMemory:
    PATH = Path("baselines/trajectories/postmortems.jsonl")

    @classmethod
    def record(cls, summary: dict) -> None:
        cls.PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(summary)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with cls.PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    @classmethod
    def retrieve(cls, scenario_family: str, k: int = 2) -> list[dict]:
        if not cls.PATH.exists():
            return []
        rows: list[dict] = []
        with cls.PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("scenario_family") == scenario_family:
                    rows.append(row)
        rows.sort(key=lambda r: float(r.get("final_score", 1.0)))
        return rows[:k]

    @classmethod
    def classify_failure(cls, audit_trail: list, final_scores: dict) -> str:
        actions = [str(item.get("action", "")).strip() for item in audit_trail]
        verbs = [action.split()[0].upper() for action in actions if action]
        pit_laps = [
            int(item.get("lap", 0))
            for item in audit_trail
            if str(item.get("action", "")).upper().startswith("PIT_NOW")
        ]

        if (
            final_scores.get("fuel_management", 1.0) <= 0.05
            or final_scores.get("final_fuel_kg", 1.0) < 0
        ):
            return "fuel_underburn"
        if _has_panic_pit(pit_laps):
            return "panic_pit"
        if _has_thrashing(verbs):
            return "thrashing"
        if final_scores.get("comms_quality", 1.0) < 0.5:
            return "comms_forgotten"
        if "REQUEST_FORECAST" in verbs and not any(
            a.upper().startswith("PIT_NOW INTER") for a in actions
        ):
            return "late_weather_call"
        if "ASSESS_UNDERCUT_WINDOW" not in verbs:
            return "missed_undercut"
        if final_scores.get("strategic_decisions", 1.0) < 0.5:
            return "other"
        return "other"


def _has_panic_pit(pit_laps: list[int]) -> bool:
    pit_laps = sorted(lap for lap in pit_laps if lap > 0)
    return any(b - a <= 3 for a, b in zip(pit_laps, pit_laps[1:]))


def _has_thrashing(verbs: list[str]) -> bool:
    if len(verbs) < 3:
        return False
    for i in range(len(verbs) - 2):
        if verbs[i] == verbs[i + 1] == verbs[i + 2]:
            return True
    return any(count >= 5 for count in Counter(verbs).values())
