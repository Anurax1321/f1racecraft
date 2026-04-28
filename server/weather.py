"""Seeded weather and safety-car schedules for the race strategist env."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WeatherState:
    air_temp_c: float = 25.0
    track_temp_c: float = 35.0
    rain_intensity: float = 0.0  # 0.0=dry, 0.3=light, 0.7=heavy
    surface_state: str = "dry"  # "dry" | "damp" | "wet"


@dataclass
class ProbCone:
    """Forecast for a future lap. Both bounds are rain_intensity values."""

    lap: int
    lower: float
    upper: float
    median: float


@dataclass
class SCEvent:
    lap: int
    sc_type: str  # "full_sc" | "vsc"
    duration_laps: int


@dataclass
class WeatherSchedule:
    per_lap: list[WeatherState] = field(default_factory=list)  # ground truth, hidden
    sc_events: list[SCEvent] = field(default_factory=list)
    forecast_uncertainty: float = 0.2

    @classmethod
    def from_seed(
        cls,
        seed: int,
        archetype: str,
        n_laps: int,
        overrides: dict | None = None,
        sc_archetype: str = "none",
    ) -> "WeatherSchedule":
        rng = np.random.default_rng(seed)
        overrides = overrides or {}
        n_laps = max(1, int(n_laps))

        if "per_lap" in overrides:
            per_lap = [_coerce_weather_state(item) for item in overrides["per_lap"]]
            if len(per_lap) < n_laps:
                per_lap.extend([per_lap[-1]] * (n_laps - len(per_lap)))
            per_lap = per_lap[:n_laps]
        elif archetype == "clear_dry":
            air = float(overrides.get("air_temp_c", rng.normal(24.0, 1.5)))
            track = float(overrides.get("track_temp_c", air + rng.normal(12.0, 1.0)))
            per_lap = [WeatherState(air, track, 0.0, "dry") for _ in range(n_laps)]
        elif archetype == "light_rain_window":
            start = int(overrides.get("rain_start_lap", rng.integers(5, 8)))
            peak = int(overrides.get("rain_peak_lap", min(n_laps, start + rng.integers(1, 3))))
            duration = int(overrides.get("rain_duration_laps", rng.integers(4, 6)))
            peak_intensity = float(overrides.get("rain_peak_intensity", rng.uniform(0.38, 0.52)))
            per_lap = []
            for lap in range(1, n_laps + 1):
                if lap < start or lap > start + duration:
                    rain = 0.0 if lap < start else 0.18
                elif lap <= peak:
                    span = max(1, peak - start + 1)
                    rain = peak_intensity * (lap - start + 1) / span
                else:
                    span = max(1, start + duration - peak)
                    rain = peak_intensity - (peak_intensity - 0.18) * (lap - peak) / span
                per_lap.append(WeatherState(18.0, 24.0, max(0.0, rain), _surface(max(0.0, rain))))
        elif archetype == "heavy_rain_lottery":
            start = int(overrides.get("rain_start_lap", rng.integers(4, 8)))
            per_lap = []
            for lap in range(1, n_laps + 1):
                rain = 0.0 if lap < start else min(0.85, 0.12 * (lap - start + 1))
                per_lap.append(WeatherState(17.0, 22.0, rain, _surface(rain)))
        elif archetype == "mixed_conditions":
            start = int(overrides.get("rain_start_lap", rng.integers(8, 11)))
            peak = float(overrides.get("rain_peak_intensity", 0.28))
            per_lap = []
            for lap in range(1, n_laps + 1):
                rain = peak if start <= lap <= start + 2 else 0.0
                per_lap.append(WeatherState(21.0, 29.0, rain, _surface(rain)))
        else:
            per_lap = [WeatherState(25.0, 35.0, 0.0, "dry") for _ in range(n_laps)]

        sc_events = [_coerce_sc_event(item) for item in overrides.get("sc_events", [])]
        if not sc_events and sc_archetype in {"sc_window", "vsc_window"}:
            if "sc_lap" in overrides:
                sc_lap = int(overrides["sc_lap"])
            elif archetype == "safety_car_prone":
                sc_lap = int(rng.integers(7, min(n_laps, 10) + 1))
            else:
                sc_lap = int(rng.integers(max(2, n_laps // 2), n_laps + 1))
            sc_type = "vsc" if sc_archetype == "vsc_window" else "full_sc"
            duration = int(overrides.get("sc_duration_laps", 3 if sc_type == "full_sc" else 2))
            sc_events.append(SCEvent(sc_lap, sc_type, duration))

        return cls(
            per_lap=per_lap,
            sc_events=sc_events,
            forecast_uncertainty=float(overrides.get("forecast_uncertainty", 0.2)),
        )

    def observe(self, current_lap: int, k_laps_ahead: int = 5) -> list[ProbCone]:
        """Return forecast cone for the next k laps."""
        cones: list[ProbCone] = []
        for lap in range(current_lap + 1, current_lap + k_laps_ahead + 1):
            state = self.at_lap(lap)
            spread = self.forecast_uncertainty * (0.12 + 0.03 * max(0, lap - current_lap - 1))
            median = state.rain_intensity
            cones.append(
                ProbCone(
                    lap=lap,
                    lower=max(0.0, median - spread),
                    upper=min(1.0, median + spread),
                    median=median,
                )
            )
        return cones

    def at_lap(self, lap: int) -> WeatherState:
        if not self.per_lap:
            return WeatherState()
        idx = max(0, min(len(self.per_lap) - 1, int(lap) - 1))
        return self.per_lap[idx]

    def is_sc_active(self, lap: int) -> bool:
        for evt in self.sc_events:
            if evt.lap <= lap < evt.lap + evt.duration_laps:
                return True
        return False

    def sc_type_at(self, lap: int) -> str:
        for evt in self.sc_events:
            if evt.lap <= lap < evt.lap + evt.duration_laps:
                return "vsc" if evt.sc_type == "vsc" else "sc"
        return "green"

    def to_hidden_weather(self) -> list[dict]:
        return [
            {
                "lap": i + 1,
                "air_temp_c": w.air_temp_c,
                "track_temp_c": w.track_temp_c,
                "rain_intensity": w.rain_intensity,
                "surface_state": w.surface_state,
            }
            for i, w in enumerate(self.per_lap)
        ]

    def to_hidden_sc(self) -> list[dict]:
        return [
            {"lap": evt.lap, "sc_type": evt.sc_type, "duration_laps": evt.duration_laps}
            for evt in self.sc_events
        ]


def _surface(rain: float) -> str:
    if rain >= 0.55:
        return "wet"
    if rain >= 0.1:
        return "damp"
    return "dry"


def _coerce_weather_state(item) -> WeatherState:
    if isinstance(item, WeatherState):
        return item
    if isinstance(item, dict):
        rain = float(item.get("rain_intensity", item.get("rain", 0.0)))
        return WeatherState(
            air_temp_c=float(item.get("air_temp_c", 25.0)),
            track_temp_c=float(item.get("track_temp_c", 35.0)),
            rain_intensity=rain,
            surface_state=str(item.get("surface_state", _surface(rain))),
        )
    if isinstance(item, (list, tuple)):
        rain = float(item[1] if len(item) > 1 else 0.0)
        surface = str(item[2]) if len(item) > 2 else _surface(rain)
        return WeatherState(25.0, 35.0, rain, surface)
    return WeatherState()


def _coerce_sc_event(item) -> SCEvent:
    if isinstance(item, SCEvent):
        return item
    if isinstance(item, dict):
        return SCEvent(
            lap=int(item.get("lap", 1)),
            sc_type=str(item.get("sc_type", item.get("type", "full_sc"))),
            duration_laps=int(item.get("duration_laps", item.get("duration", 2))),
        )
    if isinstance(item, (list, tuple)):
        return SCEvent(int(item[0]), str(item[1]), int(item[2] if len(item) > 2 else 2))
    return SCEvent(1, "full_sc", 2)
