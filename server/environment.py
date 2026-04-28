"""Core OpenEnv environment for F1 race strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from datetime import datetime, timezone
import uuid
from typing import Any

from openenv.core.env_server import Environment

from models import F1Action, F1Observation, F1State
from server.hidden_state import HiddenStateLayer
from server.opponents import Opponent, build_opponents_for_scenario, step_opponents
from server.physics import (
    MODE_TABLE,
    compute_dirty_air_factor,
    compute_lap_time,
    step_fuel,
    step_tyre,
)
from server.postmortem import PostmortemMemory
from server.scenarios import SCENARIOS
from server.scoring import compute_multi_objective_scores
from server.track import Track, load_track
from server.weather import WeatherSchedule


@dataclass
class EgoCar:
    driver_number: int = 99
    team: str = "Player"
    current_position: int = 0
    current_compound: str = "medium"
    current_tyre_age: int = 0
    tyre_health: float = 1.0
    fuel_remaining_kg: float = 0.0
    drive_mode: str = "race"
    cumulative_time_s: float = 0.0
    last_lap_time_s: float = 0.0
    pit_stops: int = 0
    stints: list[dict] = field(default_factory=list)


class F1StrategistEnvironment(Environment[F1Action, F1Observation, F1State]):
    SUPPORTS_CONCURRENT_SESSIONS = True
    LAPS_PER_STEP = 1
    PIT_COOLDOWN_LAPS = 2

    def __init__(self) -> None:
        super().__init__()
        self._track: Track | None = None
        self._ego_car = EgoCar()
        self._opponents: list[Opponent] = []
        self._weather: WeatherSchedule | None = None
        self._scenario: dict | None = None
        self._hidden_state: HiddenStateLayer | None = None
        self._memory_hints: list[str] = []
        self._last_revelations: list[dict] = []
        self._last_message = ""
        self._final_scores: dict[str, float] = {}
        self._reset_internal()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        options: dict | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> F1Observation:
        options = dict(options or {})
        options.update(kwargs)
        if "scenario" in options and isinstance(options["scenario"], dict):
            scenario = copy.deepcopy(options["scenario"])
        else:
            family = (
                task
                or options.get("task")
                or options.get("scenario_family")
                or "dry_strategy_sprint"
            )
            if family not in SCENARIOS:
                family = "dry_strategy_sprint"
            scenario = copy.deepcopy(SCENARIOS[family])
        if seed is not None:
            scenario["seed"] = seed
        if options.get("seed") is not None:
            scenario["seed"] = int(options["seed"])
        self._reset_internal(episode_id=episode_id)
        self._load(scenario)
        hints = PostmortemMemory.retrieve(self._scenario["scenario_family"], k=2)
        self._memory_hints = [_format_memory_hint(item) for item in hints]
        obs = self._obs()
        obs.message = self._scenario.get("description", "")
        return obs

    def step(
        self,
        action: F1Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> F1Observation:
        if self._done:
            obs = self._obs()
            obs.done = True
            obs.reward = 0.0
            obs.message = "Episode already complete."
            return obs

        reward, message = self._exec(action)
        self._step_count += 1
        if not self._done:
            self._advance_lap()
        delayed = self._finalize_pending_rewards()
        total_reward = reward + delayed

        self._audit_trail.append(
            {
                "step": self._step_count,
                "lap": self._lap,
                "action": action.command,
                "reward": total_reward,
                "message": message,
            }
        )
        self._check_done()
        obs = self._obs()
        obs.reward = total_reward
        obs.done = self._done
        obs.message = message
        obs.score = (
            total_reward
            if not self._done
            else self._final_scores.get("weighted_final", total_reward)
        )
        return obs

    @property
    def state(self) -> F1State:
        return F1State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._scenario.get("task_name", "") if self._scenario else "",
            scenario_family=self._scenario.get("scenario_family", "") if self._scenario else "",
            track_name=self._scenario.get("track_name", "") if self._scenario else "",
            total_laps=self._total_laps,
            resolved_issues=self._resolved_count(),
            total_issues=self._total_issue_count(),
            max_steps=self._max_steps,
            seed=self._scenario.get("seed", 0) if self._scenario else 0,
        )

    def _reset_internal(self, episode_id: str | None = None) -> None:
        self._lap = 0
        self._total_laps = 0
        self._max_steps = 0
        self._step_count = 0
        self._done = False
        self._episode_id = episode_id or uuid.uuid4().hex[:12]
        self._audit_trail: list[dict] = []
        self._fired_events: set[str] = set()
        self._dynamic_events: list[dict] = []
        self._pending_rewards: dict = {}
        self._pit_cooldown_remaining = 0
        self._inspection_calls: dict[str, list[int]] = {}
        self._pit_decisions: list[dict] = []
        self._radio_calls: list[dict] = []
        self._pit_wall_calls: list[dict] = []
        self._action_verbs: list[str] = []
        self._invalid_actions = 0
        self._harmful_actions = 0
        self._race_status = "green"
        self._memory_hints = []
        self._last_revelations = []
        self._final_scores = {}
        self._postmortem_recorded = False

    def _load(self, scenario_dict: dict) -> None:
        self._scenario = copy.deepcopy(scenario_dict)
        if self._scenario["track_name"] == "Monaco":
            try:
                self._track = load_track("Monaco")
            except FileNotFoundError:
                self._track = load_track("BrandsHatch")
                self._track.name = "Monaco"
                self._track.track_character = "street"
                self._track.base_lap_time_s = 73.2
                self._track.pit_lane_loss_s = 18.0
                self._track.sc_pit_loss_s = 5.5
        else:
            self._track = load_track(self._scenario["track_name"])

        self._total_laps = int(self._scenario.get("total_laps", 10))
        self._max_steps = int(self._scenario.get("max_steps", self._total_laps + 4))
        seed = int(self._scenario.get("seed", 0))
        self._weather = WeatherSchedule.from_seed(
            seed=seed,
            archetype=self._scenario.get("weather_archetype", "clear_dry"),
            n_laps=self._total_laps,
            overrides=self._scenario.get("weather_seed_overrides", {}),
            sc_archetype=self._scenario.get("sc_archetype", "none"),
        )
        self._scenario.setdefault("hidden_state", {})
        self._scenario["hidden_state"]["weather_evolution"] = self._weather.to_hidden_weather()
        self._scenario["hidden_state"]["safety_car_schedule"] = self._weather.to_hidden_sc()
        self._hidden_state = HiddenStateLayer.from_scenario(self._scenario)
        self._opponents = build_opponents_for_scenario(self._scenario, seed)
        self._ego_car = EgoCar(
            current_position=int(self._scenario.get("starting_position", 1)),
            current_compound=str(self._scenario.get("starting_compound", "medium")),
            fuel_remaining_kg=float(self._scenario.get("starting_fuel_kg", 90.0)),
            drive_mode=str(self._scenario.get("starting_drive_mode", "race")),
        )
        self._ego_car.stints = [
            {"compound": self._ego_car.current_compound, "start_lap": 0, "end_lap": None}
        ]
        self._dynamic_events = list(self._scenario.get("dynamic_events", []))
        self._last_message = self._scenario.get("description", "")

    def _exec(self, action: F1Action) -> tuple[float, str]:
        self._last_revelations = []
        cmd = (action.command or "").strip()
        if not cmd:
            self._invalid_actions += 1
            return -0.02, "Empty command."
        verb, *rest = cmd.split(None, 1)
        verb = verb.upper()
        arg = rest[0] if rest else ""
        handlers = {
            "PIT_NOW": self._h_pit_now,
            "STAY_OUT": self._h_stay_out,
            "RECOMMEND_PIT": self._h_recommend_pit,
            "SET_MODE": self._h_set_mode,
            "DRS_PERMISSION": self._h_drs_permission,
            "DEFEND_POSITION": self._h_defend_position,
            "ATTACK_AHEAD": self._h_attack_ahead,
            "HOLD_GAP": self._h_hold_gap,
            "LET_BY": self._h_let_by,
            "INSPECT_TYRE_DEGRADATION": self._h_inspect_tyre,
            "CHECK_OPPONENT_STRATEGY": self._h_check_opponent,
            "REQUEST_FORECAST": self._h_request_forecast,
            "ASSESS_UNDERCUT_WINDOW": self._h_assess_undercut,
            "INSPECT_FUEL_MARGIN": self._h_inspect_fuel,
            "RADIO_DRIVER": self._h_radio_driver,
            "DRAFT_PIT_WALL": self._h_draft_pit_wall,
            "BRIEF_MEDIA": self._h_brief_media,
            "REQUEST_INFO": self._h_request_info,
            "ESCALATE_TO_PRINCIPAL": self._h_escalate,
            "MANAGE_TYRE_TEMP": self._h_manage_tyre_temp,
            "DONE": self._h_done,
        }
        handler = handlers.get(verb)
        if handler is None:
            self._invalid_actions += 1
            return -0.02, f"Unknown command: {verb}"
        self._action_verbs.append(verb)
        return handler(arg)

    def _advance_lap(self) -> None:
        self._lap += 1
        if not self._track or not self._weather:
            return
        weather = self._weather.at_lap(max(1, self._lap))
        self._race_status = self._weather.sc_type_at(self._lap)
        dirty = compute_dirty_air_factor(self._gap_ahead())
        lap_time = compute_lap_time(
            self._track,
            self._ego_car.current_compound,
            self._ego_car.tyre_health,
            self._ego_car.fuel_remaining_kg,
            self._ego_car.drive_mode,
            dirty,
            weather,
            int(self._scenario.get("seed", 0)) + self._lap * 17,
        )
        self._ego_car.last_lap_time_s = lap_time
        self._ego_car.cumulative_time_s += lap_time
        self._ego_car.fuel_remaining_kg = step_fuel(
            self._track.name,
            self._ego_car.drive_mode,
            self._ego_car.fuel_remaining_kg,
            int(self._scenario.get("seed", 0)) + self._lap * 19,
        )
        self._ego_car.tyre_health = step_tyre(
            self._ego_car.current_compound,
            self._ego_car.tyre_health,
            self._ego_car.drive_mode,
            self._track.track_character,
            weather.track_temp_c,
        )
        self._ego_car.current_tyre_age += 1
        step_opponents(
            self._opponents,
            track=self._track,
            weather=weather,
            sc_active=self._race_status in {"sc", "vsc"},
            lap=self._lap,
            seed=int(self._scenario.get("seed", 0)),
            ego=self._ego_car,
        )
        if self._pit_cooldown_remaining > 0:
            self._pit_cooldown_remaining -= 1
        for event in self._dynamic_events:
            if int(event.get("lap", -1)) == self._lap:
                self._fired_events.add(str(event.get("type", "event")))

    def _finalize_pending_rewards(self) -> float:
        return 0.0

    def _check_done(self) -> None:
        if self._ego_car.fuel_remaining_kg < 0:
            self._done = True
        if self._ego_car.tyre_health <= 0.02 and self._lap > 1:
            self._done = True
            self._harmful_actions += 1
        if self._lap >= self._total_laps:
            self._done = True
        if self._step_count >= self._max_steps:
            self._done = True
        if self._done and not self._final_scores:
            self._finalize_episode()

    def _finalize_episode(self) -> None:
        self._final_scores = compute_multi_objective_scores(
            scenario_family=self._scenario.get("scenario_family", ""),
            success_criteria=self._scenario.get("success_criteria", {}),
            starting_position=int(self._scenario.get("starting_position", 99)),
            finishing_position=self._ego_car.current_position,
            rival_finishing_position=self._rival_position(),
            pit_decisions=self._pit_decisions,
            inspection_calls=self._inspection_calls,
            final_tyre_health=self._ego_car.tyre_health,
            compound_rule_satisfied=self._compound_rule_satisfied(),
            stints=self._ego_car.stints,
            track_character=self._track.track_character if self._track else "balanced",
            final_fuel_kg=self._ego_car.fuel_remaining_kg,
            dnf_due_to_fuel=self._ego_car.fuel_remaining_kg < 0,
            radio_calls_made=self._radio_calls,
            pit_wall_calls=self._pit_wall_calls,
            n_pit_stops=self._ego_car.pit_stops,
            invalid_actions=self._invalid_actions,
            harmful_actions=self._harmful_actions,
            total_laps=self._total_laps,
            steps_used=self._step_count,
            action_verbs=self._action_verbs,
        )
        self._final_scores["final_fuel_kg"] = self._ego_car.fuel_remaining_kg
        self._record_postmortem()

    def _record_postmortem(self) -> None:
        if self._postmortem_recorded or not self._scenario:
            return
        PostmortemMemory.record(
            {
                "scenario_family": self._scenario["scenario_family"],
                "failure_category": PostmortemMemory.classify_failure(
                    self._audit_trail, self._final_scores
                ),
                "first_bad_action": self._find_first_bad_action(),
                "missed_signal": self._diagnose_missed_signal(),
                "preferred_intervention_order": self._suggest_order(),
                "final_score": self._final_scores.get("weighted_final", 0.0),
                "episode_id": self._episode_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._postmortem_recorded = True

    def _obs(self) -> F1Observation:
        weather = self._weather.at_lap(max(1, self._lap)) if self._weather else None
        scores = dict(self._final_scores)
        current_score = scores.get("weighted_final", 0.0)
        return F1Observation(
            done=self._done,
            reward=None,
            current_lap=self._lap,
            total_laps=self._total_laps,
            race_phase=self._race_phase(),
            race_status="finished" if self._done else self._race_status,
            ego_position=self._ego_car.current_position,
            ego_tyre_compound=self._ego_car.current_compound,
            ego_tyre_age_laps=self._ego_car.current_tyre_age,
            ego_tyre_health_pct=round(self._ego_car.tyre_health * 100.0, 1),
            ego_fuel_remaining_kg=round(self._ego_car.fuel_remaining_kg, 2),
            ego_drive_mode=self._ego_car.drive_mode,
            ego_last_lap_time_s=round(self._ego_car.last_lap_time_s, 3),
            ego_gap_to_leader_s=self._gap_to_leader(),
            ego_gap_ahead_s=self._gap_ahead(),
            ego_gap_behind_s=self._gap_behind(),
            opponents=self._visible_opponents(),
            weather_current=_weather_dict(weather),
            weather_forecast=self._visible_forecast(),
            pit_window_alerts=self._pit_window_alerts(),
            cascade_alerts=self._cascade_alerts(),
            uncertainty_alerts=list(self._last_revelations),
            radio_inbox=list(self._scenario.get("radio_inbox", [])) if self._scenario else [],
            pending_issues_count=max(0, self._total_issue_count() - self._resolved_count()),
            resolved_issues_count=self._resolved_count(),
            total_issues_count=self._total_issue_count(),
            message=self._last_message,
            available_commands=self._available_commands(),
            multi_objective_scores=scores,
            score=current_score,
            hint=self._hint(),
            memory_hints=list(self._memory_hints),
        )

    def _h_pit_now(self, arg: str) -> tuple[float, str]:
        compound = arg.strip().split()[0].lower() if arg.strip() else ""
        if compound not in {"soft", "medium", "hard", "inter", "wet"}:
            self._invalid_actions += 1
            return -0.02, "Invalid pit compound."
        if self._pit_cooldown_remaining > 0:
            self._harmful_actions += 1
        if self._ego_car.stints:
            self._ego_car.stints[-1]["end_lap"] = self._lap
        self._ego_car.current_compound = compound
        self._ego_car.current_tyre_age = 0
        self._ego_car.tyre_health = 1.0
        self._ego_car.pit_stops += 1
        self._ego_car.stints.append({"compound": compound, "start_lap": self._lap, "end_lap": None})
        pit_loss = (
            self._track.sc_pit_loss_s
            if self._race_status in {"sc", "vsc"}
            else self._track.pit_lane_loss_s
        )
        self._ego_car.cumulative_time_s += pit_loss
        self._pit_cooldown_remaining = self.PIT_COOLDOWN_LAPS
        self._pit_decisions.append(
            {"lap": self._lap, "compound": compound, "under_sc": self._race_status in {"sc", "vsc"}}
        )
        return 0.01, f"Box confirmed for {compound}; pit loss {pit_loss:.1f}s."

    def _h_stay_out(self, arg: str) -> tuple[float, str]:
        return 0.0, "Staying out."

    def _h_recommend_pit(self, arg: str) -> tuple[float, str]:
        return 0.005, f"Pit recommendation logged: {arg.strip() or 'next viable lap'}."

    def _h_set_mode(self, arg: str) -> tuple[float, str]:
        mode = arg.strip().lower()
        if mode not in MODE_TABLE:
            self._invalid_actions += 1
            return -0.02, f"Invalid drive mode: {mode}"
        self._ego_car.drive_mode = mode
        return 0.005, f"Drive mode set to {mode}."

    def _h_drs_permission(self, arg: str) -> tuple[float, str]:
        return 0.0, f"DRS permission noted: {arg.strip() or 'unchanged'}."

    def _h_defend_position(self, arg: str) -> tuple[float, str]:
        self._ego_car.drive_mode = "push"
        return 0.005, "Defending position; driver may spend tyre to hold track position."

    def _h_attack_ahead(self, arg: str) -> tuple[float, str]:
        self._ego_car.drive_mode = "push"
        return 0.005, "Attack mode enabled."

    def _h_hold_gap(self, arg: str) -> tuple[float, str]:
        return 0.01, f"Gap target held against {arg.strip() or 'car ahead'}."

    def _h_let_by(self, arg: str) -> tuple[float, str]:
        self._harmful_actions += 1
        return -0.01, f"Team-order let-by logged for {arg.strip() or 'unknown car'}."

    def _h_manage_tyre_temp(self, arg: str) -> tuple[float, str]:
        """Adjust tyre temperature management mode.

        warm   — increases grip window but adds marginal wear (out-lap/push scenarios)
        cool   — reduces wear at cost of grip (overheating/cliff scenarios)
        normal — return to balanced default
        """
        target = arg.strip().lower()
        if target not in {"warm", "cool", "normal"}:
            self._invalid_actions += 1
            return -0.02, f"Invalid MANAGE_TYRE_TEMP target '{target}'. Use warm|cool|normal."
        if target == "warm":
            # Slight push mode to generate tyre heat
            if self._ego_car.drive_mode not in {"push"}:
                self._ego_car.drive_mode = "push"
        elif target == "cool":
            # Drop to conserve to reduce surface temp
            if self._ego_car.drive_mode not in {"conserve", "tyre_save"}:
                self._ego_car.drive_mode = "tyre_save"
        else:
            if self._ego_car.drive_mode not in {"race"}:
                self._ego_car.drive_mode = "race"
        return 0.005, f"Tyre temperature mode: {target}. Drive mode adjusted."

    def _h_inspect_tyre(self, arg: str) -> tuple[float, str]:
        is_new, info = self._hidden_state.reveal("true_tyre_curve", self._ego_car.current_compound)
        self._record_inspection("INSPECT_TYRE_DEGRADATION")
        self._last_revelations.append({"key": "true_tyre_curve", "info": info})
        return (0.02 if is_new else 0.0), "Tyre degradation curve revealed."

    def _h_check_opponent(self, arg: str) -> tuple[float, str]:
        driver = arg.strip().split()[0] if arg.strip() else ""
        is_new, info = self._hidden_state.reveal("opponent_strategies", driver)
        self._record_inspection("CHECK_OPPONENT_STRATEGY")
        self._last_revelations.append({"key": f"opponent_strategies:{driver}", "info": info})
        return (0.02 if is_new else 0.0), f"Opponent #{driver} strategy revealed."

    def _h_request_forecast(self, arg: str) -> tuple[float, str]:
        is_new, info = self._hidden_state.reveal("weather_evolution")
        self._record_inspection("REQUEST_FORECAST")
        self._last_revelations.append({"key": "weather_evolution", "info": info})
        return (0.02 if is_new else 0.0), "Forecast cone updated."

    def _h_assess_undercut(self, arg: str) -> tuple[float, str]:
        is_new, info = self._hidden_state.reveal("undercut_threshold_laps")
        self._record_inspection("ASSESS_UNDERCUT_WINDOW")
        self._last_revelations.append({"key": "undercut_threshold_laps", "info": info})
        return (0.02 if is_new else 0.0), "Undercut window assessed."

    def _h_inspect_fuel(self, arg: str) -> tuple[float, str]:
        is_new, info = self._hidden_state.reveal("fuel_burn_actual")
        self._record_inspection("INSPECT_FUEL_MARGIN")
        self._last_revelations.append({"key": "fuel_burn_actual", "info": info})
        return (0.02 if is_new else 0.0), "Fuel margin inspected."

    def _h_radio_driver(self, arg: str) -> tuple[float, str]:
        self._radio_calls.append({"lap": self._lap, "message": arg.strip()})
        return 0.005, "Driver radio sent."

    def _h_draft_pit_wall(self, arg: str) -> tuple[float, str]:
        self._pit_wall_calls.append({"lap": self._lap, "message": arg.strip()})
        return 0.0, "Pit wall draft logged."

    def _h_brief_media(self, arg: str) -> tuple[float, str]:
        return 0.0, "Media briefing saved."

    def _h_request_info(self, arg: str) -> tuple[float, str]:
        topic = arg.strip().lower() or "summary"
        if topic == "scoring" and self._done:
            return 0.0, f"Scores: {self._final_scores}"
        return 0.0, f"Info available for {topic}."

    def _h_escalate(self, arg: str) -> tuple[float, str]:
        return 0.005, self._hint()

    def _h_done(self, arg: str) -> tuple[float, str]:
        self._done = True
        self._check_done()
        return 0.0, "Episode ended."

    def _record_inspection(self, name: str) -> None:
        self._inspection_calls.setdefault(name, []).append(self._lap)

    def _visible_opponents(self) -> list[dict]:
        rows = []
        for opp in sorted(self._opponents, key=lambda o: o.current_position):
            rows.append(
                {
                    "driver_number": opp.driver_number,
                    "team": opp.team,
                    "position": opp.current_position,
                    "last_lap_time_s": round(opp.last_lap_time_s, 3),
                    "gap_s": round(opp.cumulative_time_s - self._ego_car.cumulative_time_s, 3),
                    "current_compound": opp.current_compound,
                    "stint_age": opp.current_tyre_age,
                }
            )
        return rows

    def _visible_forecast(self) -> list[dict]:
        if "REQUEST_FORECAST" not in self._inspection_calls or not self._weather:
            return []
        return [cone.__dict__ for cone in self._weather.observe(self._lap, 5)]

    def _pit_window_alerts(self) -> list[dict]:
        criteria = self._scenario.get("success_criteria", {}) if self._scenario else {}
        lo, hi = criteria.get("optimal_pit_window", [0, 0])
        if lo <= self._lap <= hi:
            return [
                {"type": "pit_window", "message": f"Optimal pit window active: laps {lo}-{hi}."}
            ]
        return []

    def _cascade_alerts(self) -> list[dict]:
        return [e for e in self._dynamic_events if int(e.get("lap", -1)) <= self._lap]

    def _available_commands(self) -> list[str]:
        return [
            "PIT_NOW soft",
            "PIT_NOW medium",
            "PIT_NOW hard",
            "PIT_NOW inter",
            "STAY_OUT",
            "SET_MODE race",
            "SET_MODE push",
            "SET_MODE conserve",
            "HOLD_GAP <driver_number>",
            "ATTACK_AHEAD",
            "DEFEND_POSITION",
            "INSPECT_TYRE_DEGRADATION",
            "CHECK_OPPONENT_STRATEGY <driver_number>",
            "REQUEST_FORECAST",
            "ASSESS_UNDERCUT_WINDOW",
            "INSPECT_FUEL_MARGIN",
            "RADIO_DRIVER <message>",
            "REQUEST_INFO scoring",
            "DONE",
        ]

    def _race_phase(self) -> str:
        if not self._total_laps:
            return "start"
        frac = self._lap / self._total_laps
        if frac < 0.33:
            return "start"
        if frac < 0.67:
            return "mid"
        return "end"

    def _gap_to_leader(self) -> float | None:
        leader = min([self._ego_car, *self._opponents], key=lambda c: c.current_position)
        if leader is self._ego_car:
            return 0.0
        return round(max(0.0, self._ego_car.cumulative_time_s - leader.cumulative_time_s), 3)

    def _gap_ahead(self) -> float | None:
        ahead = [
            c for c in self._opponents if c.current_position == self._ego_car.current_position - 1
        ]
        if not ahead:
            return None
        return round(max(0.0, self._ego_car.cumulative_time_s - ahead[0].cumulative_time_s), 3)

    def _gap_behind(self) -> float | None:
        behind = [
            c for c in self._opponents if c.current_position == self._ego_car.current_position + 1
        ]
        if not behind:
            return None
        return round(max(0.0, behind[0].cumulative_time_s - self._ego_car.cumulative_time_s), 3)

    def _compound_rule_satisfied(self) -> bool:
        compounds = {s["compound"] for s in self._ego_car.stints}
        if any(c in {"inter", "wet"} for c in compounds):
            return True
        return len(compounds & {"soft", "medium", "hard"}) >= 2

    def _rival_position(self) -> int | None:
        rival = self._scenario.get("championship_rival") if self._scenario else None
        if rival is None:
            return None
        for opp in self._opponents:
            if opp.driver_number == int(rival):
                return opp.current_position
        return None

    def _total_issue_count(self) -> int:
        if not self._scenario:
            return 0
        return sum(len(v) for v in self._scenario.get("issues", {}).values())

    def _resolved_count(self) -> int:
        if not self._final_scores:
            return 0
        return sum(1 for k, v in self._final_scores.items() if k != "weighted_final" and v >= 0.8)

    def _hint(self) -> str:
        family = self._scenario.get("scenario_family", "") if self._scenario else ""
        if family == "weather_roulette":
            return "Forecast first, then time the inter switch around the rain peak."
        if family == "late_safety_car":
            return "At Monaco, track position and a cheap SC stop beat an early green stop."
        if family == "championship_decider":
            return "Cover #10 and keep the weather forecast fresh before committing."
        return "Assess the undercut before boxing for the soft stint."

    def _find_first_bad_action(self) -> str:
        for item in self._audit_trail:
            if item.get("reward", 0.0) < 0:
                return str(item.get("action", ""))
        return ""

    def _diagnose_missed_signal(self) -> str:
        if (
            "REQUEST_FORECAST" not in self._inspection_calls
            and self._scenario.get("scenario_family") == "weather_roulette"
        ):
            return "weather_forecast"
        if (
            "ASSESS_UNDERCUT_WINDOW" not in self._inspection_calls
            and self._scenario.get("scenario_family") == "dry_strategy_sprint"
        ):
            return "undercut_window"
        if (
            self._scenario.get("scenario_family") == "late_safety_car"
            and "HOLD_GAP" not in self._action_verbs
        ):
            return "safety_car_gap"
        return ""

    def _suggest_order(self) -> list[str]:
        family = self._scenario.get("scenario_family", "") if self._scenario else ""
        if family == "weather_roulette":
            return ["REQUEST_FORECAST", "RADIO_DRIVER", "PIT_NOW inter", "INSPECT_FUEL_MARGIN"]
        if family == "late_safety_car":
            return ["HOLD_GAP 4", "RADIO_DRIVER", "PIT_NOW hard", "SET_MODE push"]
        return [
            "CHECK_OPPONENT_STRATEGY 16",
            "ASSESS_UNDERCUT_WINDOW",
            "RADIO_DRIVER",
            "PIT_NOW soft",
        ]


def _weather_dict(weather) -> dict:
    if weather is None:
        return {}
    return {
        "air_temp_c": weather.air_temp_c,
        "track_temp_c": weather.track_temp_c,
        "rain_intensity": weather.rain_intensity,
        "surface_state": weather.surface_state,
    }


def _format_memory_hint(item: dict) -> str:
    category = item.get("failure_category", "other")
    score = item.get("final_score", 0.0)
    missed = item.get("missed_signal") or "strategy signal"
    return f"Prior {category} episode scored {score:.2f}; watch {missed}."
