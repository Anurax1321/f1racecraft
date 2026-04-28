"""
F1 Strategist — Pydantic Type Definitions
==========================================

Action, Observation, and State inherit from openenv.core.env_server base classes.
Observation already provides `done: bool` and `reward: Optional[float]`;
State already provides `episode_id` and `step_count`.

Spec: docs/architecture.md §6.

Person 1 owns this file. Fill in the field list during Phase 1.
"""

from typing import Dict, List, Optional
from openenv.core.env_server import Action, Observation, State


class F1Action(Action):
    """Agent sends a single text strategic command each step.

    Examples:
        F1Action(command="PIT_NOW soft")
        F1Action(command="STAY_OUT")
        F1Action(command="REQUEST_FORECAST")
        F1Action(command="RADIO_DRIVER Box this lap. Soft tyres.")

    Full action grammar: docs/architecture.md §7.
    """

    command: str


class F1Observation(Observation):
    """Full F1 race state returned after reset() or step().

    Filtered by what has been revealed via inspection actions:
    - opponent_strategies hidden until CHECK_OPPONENT_STRATEGY
    - weather_forecast hidden until REQUEST_FORECAST
    - true tyre wear hidden until INSPECT_TYRE_DEGRADATION
    - true fuel margin hidden until INSPECT_FUEL_MARGIN
    """

    # Race clock
    current_lap: int = 0
    total_laps: int = 0
    race_phase: str = ""  # "start" | "mid" | "end"
    race_status: str = ""  # "green" | "sc" | "vsc" | "red" | "finished"

    # Ego car (always visible — TV-feed level)
    ego_position: int = 0
    ego_tyre_compound: str = ""
    ego_tyre_age_laps: int = 0
    ego_tyre_health_pct: float = 0.0  # noisy estimate; INSPECT reveals true curve
    ego_fuel_remaining_kg: float = 0.0
    ego_drive_mode: str = ""
    ego_last_lap_time_s: float = 0.0
    ego_gap_to_leader_s: Optional[float] = None
    ego_gap_ahead_s: Optional[float] = None
    ego_gap_behind_s: Optional[float] = None

    # Opponents (TV-feed level)
    opponents: List[Dict] = []

    # Weather (forecast cone if REQUEST_FORECAST was called, else just current)
    weather_current: Dict = {}
    weather_forecast: List[Dict] = []  # populated only after REQUEST_FORECAST

    # Alerts and dynamic events
    pit_window_alerts: List[Dict] = []
    cascade_alerts: List[Dict] = []
    uncertainty_alerts: List[Dict] = []  # hidden state revealed this step
    radio_inbox: List[Dict] = []  # incoming team comms

    # Issue tracking
    pending_issues_count: int = 0
    resolved_issues_count: int = 0
    total_issues_count: int = 0

    # Step-level signal
    message: str = ""
    available_commands: List[str] = []
    multi_objective_scores: Dict[str, float] = {}
    score: float = 0.0
    hint: str = ""
    memory_hints: List[str] = []  # postmortem injections from prior episodes


class F1State(State):
    """Episode metadata. Inherits episode_id + step_count from base."""

    task_name: str = ""
    scenario_family: str = ""
    track_name: str = ""
    total_laps: int = 0
    resolved_issues: int = 0
    total_issues: int = 0
    max_steps: int = 0
    seed: int = 0
