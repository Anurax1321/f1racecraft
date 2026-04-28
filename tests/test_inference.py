"""Tests for the inference layer: parse_action against noisy LLM-style output.

The parse_action function in inference.py has to handle:
    - Bare commands: "PIT_NOW soft"
    - <think>...</think> blocks (Qwen3-style reasoning tokens)
    - Code-fenced commands: "```\nPIT_NOW soft\n```"
    - Multi-line responses where the command is later in the body
    - Lowercased / mixed-case variants
    - Embedded explanations: "I think we should PIT_NOW soft because..."
    - Empty/whitespace input → safe default (STAY_OUT)
    - Unknown verbs → safe default
"""

from __future__ import annotations

import pytest

from inference import parse_action, format_obs, VALID_VERBS
from models import F1Action, F1Observation


# ──────────────────────────────────────────────────────────────────────────
# Bare commands
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text,expected_verb",
    [
        ("PIT_NOW soft", "PIT_NOW"),
        ("STAY_OUT", "STAY_OUT"),
        ("REQUEST_FORECAST", "REQUEST_FORECAST"),
        ("ASSESS_UNDERCUT_WINDOW", "ASSESS_UNDERCUT_WINDOW"),
        ("DONE", "DONE"),
        ("SET_MODE push", "SET_MODE"),
        ("HOLD_GAP 4", "HOLD_GAP"),
        ("CHECK_OPPONENT_STRATEGY 16", "CHECK_OPPONENT_STRATEGY"),
        ('RADIO_DRIVER "Box now"', "RADIO_DRIVER"),
    ],
)
def test_parse_action_bare(text, expected_verb):
    action = parse_action(text)
    assert action.command.split()[0] == expected_verb


# ──────────────────────────────────────────────────────────────────────────
# Qwen3 <think>...</think> reasoning tokens
# ──────────────────────────────────────────────────────────────────────────

def test_parse_action_strips_think_block():
    text = "<think>The driver is on softs and the rain is coming.</think>\nPIT_NOW inter"
    assert parse_action(text).command == "PIT_NOW inter"


def test_parse_action_with_close_tag_only():
    text = "Reasoning was inline.</think>\nSTAY_OUT"
    assert parse_action(text).command == "STAY_OUT"


def test_parse_action_with_capitalised_think():
    text = "<Think>Hmm</Think>\nDONE"
    assert parse_action(text).command == "DONE"


# ──────────────────────────────────────────────────────────────────────────
# Code fences and markdown
# ──────────────────────────────────────────────────────────────────────────

def test_parse_action_code_fence_simple():
    text = "```\nPIT_NOW soft\n```"
    assert parse_action(text).command == "PIT_NOW soft"


def test_parse_action_code_fence_with_language():
    text = "```text\nREQUEST_FORECAST\n```"
    assert parse_action(text).command == "REQUEST_FORECAST"


def test_parse_action_inline_backticks():
    # Single-tick wrapping
    text = "`PIT_NOW hard`"
    assert parse_action(text).command.startswith("PIT_NOW")


# ──────────────────────────────────────────────────────────────────────────
# Embedded in prose
# ──────────────────────────────────────────────────────────────────────────

def test_parse_action_embedded_in_sentence():
    text = "I think we should PIT_NOW soft because the rain is coming."
    action = parse_action(text)
    assert action.command.startswith("PIT_NOW")


def test_parse_action_picks_first_valid_verb():
    text = "Looking at the data: PIT_NOW soft\nAlso, STAY_OUT might work."
    # First valid line is the bare PIT_NOW
    assert parse_action(text).command.startswith("PIT_NOW")


def test_parse_action_lower_case_recovered():
    text = "stay_out"
    # Falls through to regex, which is case-insensitive
    result = parse_action(text)
    assert result.command.upper().startswith("STAY_OUT")


# ──────────────────────────────────────────────────────────────────────────
# Safe defaults
# ──────────────────────────────────────────────────────────────────────────

def test_parse_action_empty_returns_stay_out():
    assert parse_action("").command == "STAY_OUT"


def test_parse_action_whitespace_returns_stay_out():
    assert parse_action("   \n\n   ").command == "STAY_OUT"


def test_parse_action_garbage_returns_stay_out():
    assert parse_action("hsdfjkl pneumonoultramicroscopic").command == "STAY_OUT"


def test_parse_action_none_safe():
    assert parse_action(None).command == "STAY_OUT"


# ──────────────────────────────────────────────────────────────────────────
# All VALID_VERBS round-trip cleanly
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("verb", sorted(VALID_VERBS))
def test_all_valid_verbs_parse_themselves(verb):
    """Every verb in VALID_VERBS must parse back to itself."""
    if verb == "PIT_NOW":
        text = "PIT_NOW soft"
    elif verb == "SET_MODE":
        text = "SET_MODE race"
    elif verb in {"HOLD_GAP", "LET_BY", "CHECK_OPPONENT_STRATEGY"}:
        text = f"{verb} 4"
    elif verb == "RECOMMEND_PIT":
        text = "RECOMMEND_PIT 7"
    elif verb == "DRS_PERMISSION":
        text = "DRS_PERMISSION on"
    elif verb in {"RADIO_DRIVER", "DRAFT_PIT_WALL", "BRIEF_MEDIA"}:
        text = f'{verb} "test message"'
    elif verb == "REQUEST_INFO":
        text = "REQUEST_INFO weather"
    else:
        text = verb
    action = parse_action(text)
    assert action.command.split()[0].upper() == verb


# ──────────────────────────────────────────────────────────────────────────
# format_obs sanity
# ──────────────────────────────────────────────────────────────────────────

def test_format_obs_does_not_crash_on_empty_observation():
    obs = F1Observation()
    text = format_obs(obs)
    assert "Lap 0/0" in text
    assert "Return one command only" in text


def test_format_obs_includes_required_fields():
    obs = F1Observation(
        current_lap=5,
        total_laps=12,
        ego_position=3,
        ego_tyre_compound="medium",
        ego_tyre_age_laps=5,
        ego_fuel_remaining_kg=42.5,
    )
    text = format_obs(obs)
    assert "Lap 5/12" in text
    assert "P3" in text
    assert "medium" in text
    assert "42.5" in text


def test_format_obs_with_memory_hints():
    obs = F1Observation(
        current_lap=1,
        total_laps=10,
        memory_hints=["Prior late_weather_call episode scored 0.32; watch forecast"],
    )
    text = format_obs(obs)
    assert "late_weather_call" in text


# ──────────────────────────────────────────────────────────────────────────
# Heuristic generator end-to-end (does not require a transformers checkpoint)
# ──────────────────────────────────────────────────────────────────────────

def test_heuristic_inference_runs_to_completion():
    """`inference.run_inference("heuristic", ...)` must complete an episode without errors."""
    from inference import run_inference
    scores = run_inference(model="heuristic", task="dry_strategy_sprint", n_episodes=1, seed=0)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0


def test_heuristic_inference_multi_task():
    """Heuristic inference must work across all four families without error."""
    from inference import run_inference
    for task in ["dry_strategy_sprint", "weather_roulette", "late_safety_car", "championship_decider"]:
        scores = run_inference(model="heuristic", task=task, n_episodes=1, seed=0)
        assert len(scores) == 1