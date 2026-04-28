"""Hidden-state revelation layer for inspection actions."""

from dataclasses import dataclass, field


@dataclass
class HiddenStateLayer:
    true_tyre_curve: dict = field(default_factory=dict)
    opponent_strategies: dict = field(default_factory=dict)
    weather_evolution: list = field(default_factory=list)
    safety_car_schedule: list = field(default_factory=list)
    fuel_burn_actual: float = 0.0
    undercut_threshold_laps: int = 0

    revealed: set[str] = field(default_factory=set)

    @classmethod
    def from_scenario(cls, scenario: dict) -> "HiddenStateLayer":
        """Initialise from scenario["hidden_state"] dict."""
        hidden = dict(scenario.get("hidden_state", {}))
        return cls(
            true_tyre_curve=dict(hidden.get("true_tyre_curve", {})),
            opponent_strategies=dict(hidden.get("opponent_strategies", {})),
            weather_evolution=list(hidden.get("weather_evolution", [])),
            safety_car_schedule=list(hidden.get("safety_car_schedule", [])),
            fuel_burn_actual=float(hidden.get("fuel_burn_actual", 0.0)),
            undercut_threshold_laps=int(hidden.get("undercut_threshold_laps", 0)),
        )

    def reveal(self, key: str, sub_key: str | None = None) -> tuple[bool, dict]:
        """Reveal a hidden variable. Returns (is_new_revelation, info_dict)."""
        full_key = f"{key}:{sub_key}" if sub_key else key
        is_new = full_key not in self.revealed
        self.revealed.add(full_key)
        info: dict
        if key == "true_tyre_curve":
            if sub_key:
                info = {str(sub_key): self.true_tyre_curve.get(str(sub_key), [])}
            else:
                info = dict(self.true_tyre_curve)
        elif key == "opponent_strategies":
            raw = self.opponent_strategies.get(str(sub_key), [])
            if raw is None:
                raw = []
            info = {"driver_number": sub_key, "planned_strategy": raw}
        elif key == "weather_evolution":
            info = {"per_lap": list(self.weather_evolution)}
        elif key == "safety_car_schedule":
            info = {"events": list(self.safety_car_schedule)}
        elif key == "fuel_burn_actual":
            info = {"fuel_burn_actual": self.fuel_burn_actual}
        elif key == "undercut_threshold_laps":
            info = {"undercut_threshold_laps": self.undercut_threshold_laps}
        else:
            info = {}
        return is_new, info

    def is_revealed(self, key: str, sub_key: str | None = None) -> bool:
        full_key = f"{key}:{sub_key}" if sub_key else key
        return full_key in self.revealed
