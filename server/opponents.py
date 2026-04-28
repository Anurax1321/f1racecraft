"""Rule-based opponent cars and stint execution."""

from dataclasses import dataclass, field

import numpy as np

from server.physics import compute_lap_time, step_fuel, step_tyre


@dataclass
class Stint:
    compound: str
    planned_end_lap: int


@dataclass
class Opponent:
    driver_number: int
    team: str
    pace_offset_s: float
    aggression: float
    planned_strategy: list[Stint] = field(default_factory=list)
    current_compound: str = ""
    current_tyre_age: int = 0
    current_position: int = 0
    fuel_remaining_kg: float = 0.0
    cumulative_time_s: float = 0.0
    tyre_health: float = 1.0
    last_lap_time_s: float = 0.0
    pit_stops: int = 0


def step_opponents(opponents: list[Opponent], **ctx) -> None:
    """Advance opponents by one lap and recompute positions with ego if supplied."""
    track = ctx["track"]
    weather = ctx["weather"]
    lap = int(ctx["lap"])
    sc_active = bool(ctx.get("sc_active", False))
    seed = int(ctx.get("seed", 0))
    ego = ctx.get("ego")

    for idx, opp in enumerate(opponents):
        due_to_pit = bool(opp.planned_strategy and opp.planned_strategy[0].planned_end_lap == lap)
        opportunistic_sc_pit = bool(
            sc_active
            and opp.planned_strategy
            and 0 <= opp.planned_strategy[0].planned_end_lap - lap <= 2
        )

        if due_to_pit or opportunistic_sc_pit:
            stint = opp.planned_strategy.pop(0)
            pit_loss = track.sc_pit_loss_s if sc_active else track.pit_lane_loss_s
            opp.current_compound = stint.compound
            opp.current_tyre_age = 0
            opp.tyre_health = 1.0
            opp.pit_stops += 1
        else:
            pit_loss = 0.0

        lap_time = compute_lap_time(
            track=track,
            compound=opp.current_compound or "medium",
            tyre_health=opp.tyre_health,
            fuel_kg=opp.fuel_remaining_kg,
            drive_mode=_mode_for_opponent(opp),
            dirty_air_factor=0.15 if opp.current_position > 1 else 0.0,
            weather=weather,
            noise_seed=seed + lap * 101 + idx,
        )
        opp.last_lap_time_s = lap_time + opp.pace_offset_s + pit_loss
        opp.cumulative_time_s += opp.last_lap_time_s
        opp.fuel_remaining_kg = step_fuel(
            track.name,
            _mode_for_opponent(opp),
            opp.fuel_remaining_kg,
            seed + lap * 113 + idx,
        )
        opp.tyre_health = step_tyre(
            opp.current_compound or "medium",
            opp.tyre_health,
            _mode_for_opponent(opp),
            track.track_character,
            weather.track_temp_c,
        )
        opp.current_tyre_age += 1

    ranked = list(opponents)
    if ego is not None:
        ranked.append(ego)
    ranked.sort(
        key=lambda car: (
            getattr(car, "cumulative_time_s", 0.0),
            getattr(car, "current_position", 99),
        )
    )
    for pos, car in enumerate(ranked, 1):
        car.current_position = pos


def build_opponents_for_scenario(scenario: dict, seed: int) -> list[Opponent]:
    """Construct the opponent list from scenario dict + RNG seed."""
    rng = np.random.default_rng(seed)
    opponents: list[Opponent] = []
    for raw in scenario.get("opponents", []):
        pace = float(raw.get("pace_offset_s", rng.normal(0.0, 0.35)))
        stints = [
            Stint(str(item.get("compound", "medium")), int(item.get("planned_end_lap", 0)))
            for item in raw.get("planned_strategy", [])
        ]
        opponents.append(
            Opponent(
                driver_number=int(raw["driver_number"]),
                team=str(raw.get("team", "")),
                pace_offset_s=pace,
                aggression=float(raw.get("aggression", 0.5)),
                planned_strategy=stints,
                current_compound=str(raw.get("starting_compound", "medium")),
                current_tyre_age=int(raw.get("starting_tyre_age", 0)),
                current_position=int(raw.get("starting_position", 99)),
                fuel_remaining_kg=float(
                    raw.get("starting_fuel_kg", scenario.get("starting_fuel_kg", 90.0))
                ),
                cumulative_time_s=0.0,
            )
        )
    opponents.sort(key=lambda opp: opp.current_position)
    return opponents


def _mode_for_opponent(opp: Opponent) -> str:
    if opp.aggression >= 0.72:
        return "push"
    if opp.aggression <= 0.30:
        return "conserve"
    return "race"
