"""Configuration loading and validation for Track A."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "track_a_pilot.yaml"


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_config(config: Dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Track A config schema_version must be 1")
    hz = float(config["clock"]["hz"])
    if hz <= 0:
        raise ValueError("clock.hz must be positive")
    fps = [int(value) for value in config["actions"]["fps"]]
    if fps != sorted(set(fps)) or not fps or max(fps) > hz:
        raise ValueError("actions.fps must be unique, sorted, and no greater than clock.hz")
    if config["actions"]["preferred_core_kib"] not in (90, 129):
        raise ValueError("preferred_core_kib must be 90 or 129")
    if float(config["safety"]["epsilon_m"]) <= 0:
        raise ValueError("safety.epsilon_m must be positive")
    weights = config["reward"]["task_metric_weights"]
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("task_metric_weights must sum to 1")
    rungs = config["channel"]["rungs"]
    matrix = config["channel"]["transition_matrix"]
    if set(rungs) != set(matrix):
        raise ValueError("channel rung and transition row names differ")
    for name, row in matrix.items():
        if set(row) != set(rungs) or abs(sum(float(v) for v in row.values()) - 1.0) > 1e-9:
            raise ValueError(f"invalid transition probabilities for {name}")
    ratios = config["replay"]["split_ratios"]
    if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise ValueError("replay.split_ratios must sum to 1")
    calibration = config.get("safety_calibration")
    if calibration is not None:
        ucb_values = [float(value) for value in calibration["ucb_k_values"]]
        c1_values = [float(value) for value in calibration["c1_pessimism_factor_values"]]
        if ucb_values != sorted(set(ucb_values)) or any(value < 0.0 for value in ucb_values):
            raise ValueError("safety_calibration.ucb_k_values must be unique, sorted, and nonnegative")
        if c1_values != sorted(set(c1_values)) or any(not 0.0 < value <= 1.0 for value in c1_values):
            raise ValueError(
                "safety_calibration.c1_pessimism_factor_values must be unique, sorted, and in (0, 1]"
            )
        if 0.0 not in ucb_values or 1.0 not in ucb_values or 0.7 not in c1_values:
            raise ValueError("safety calibration must retain ucb_k=0/1 and c1=0.7 reference cells")
        fixed = calibration["fixed_point"]
        if (
            float(fixed["epsilon_m"]) != 2.0
            or int(fixed["preferred_core_kib"]) != 90
            or float(fixed["range_m"]) != 25.0
        ):
            raise ValueError("Track A safety calibration fixed point must remain epsilon=2, core=90, range=25")

    def validate_fixed_point(name: str, fixed: Dict[str, Any]) -> None:
        if (
            float(fixed["epsilon_m"]) != 2.0
            or int(fixed["preferred_core_kib"]) != 90
            or float(fixed["range_m"]) != 25.0
            or float(fixed["ucb_k"]) != 0.0
            or float(fixed["c1_pessimism_factor"]) != 0.7
        ):
            raise ValueError(f"{name}.fixed_point must remain epsilon=2, core=90, range=25, ucb=0, c1=0.7")

    estimator = config["estimator_sensitivity"]
    validate_fixed_point("estimator_sensitivity", estimator["fixed_point"])
    lag_values = [int(value) for value in estimator["telemetry_lag_steps_values"]]
    noise_values = [float(value) for value in estimator["estimate_noise_fraction_values"]]
    if lag_values != sorted(set(lag_values)) or any(value < 0 for value in lag_values):
        raise ValueError("estimator_sensitivity telemetry lags must be unique, sorted, and nonnegative")
    if noise_values != sorted(set(noise_values)) or any(value < 0.0 for value in noise_values):
        raise ValueError("estimator_sensitivity noise fractions must be unique, sorted, and nonnegative")
    if 2 not in lag_values or 0.05 not in noise_values or 0 not in lag_values or 0.0 not in noise_values:
        raise ValueError("estimator grid must retain both the baseline (2, 0.05) and idealized (0, 0) cells")

    reward_sensitivity = config["reward_sensitivity"]
    validate_fixed_point("reward_sensitivity", reward_sensitivity["fixed_point"])
    expected_reward_knobs = {"w_error", "lambda_prb", "w_task"}
    reward_knobs = set(config["reward"]["one_at_a_time_sensitivity"])
    if reward_knobs != expected_reward_knobs:
        raise ValueError("reward one-at-a-time sensitivity must contain w_error, lambda_prb, and w_task")
    if any(len(config["reward"]["one_at_a_time_sensitivity"][name]) != 2 for name in reward_knobs):
        raise ValueError("each reward one-at-a-time knob must declare exactly low/high values")

    advisor = config["advisor_sweep"]
    epsilon_values = [float(value) for value in advisor["epsilon_m_values"]]
    core_values = [int(value) for value in advisor["preferred_core_kib_values"]]
    range_values = [float(value) for value in advisor["range_m_values"]]
    if epsilon_values != [1.5, 2.0, 2.5]:
        raise ValueError("advisor_sweep epsilon grid must be [1.5, 2.0, 2.5]")
    if core_values != [90, 129]:
        raise ValueError("advisor_sweep preferred-core grid must be [90, 129]")
    if range_values != [25.0, 40.0]:
        raise ValueError("advisor_sweep range grid must be [25, 40]")
    fixed_shield = advisor["fixed_shield"]
    if float(fixed_shield["ucb_k"]) != 0.0 or float(fixed_shield["c1_pessimism_factor"]) != 0.7:
        raise ValueError("advisor_sweep fixed shield must remain ucb=0, c1=0.7")

    for section in ("pilot", "safety_calibration", "estimator_sensitivity", "reward_sensitivity", "advisor_sweep"):
        if not isinstance(config[section]["common_random_latency_by_tick"], bool):
            raise ValueError(f"{section}.common_random_latency_by_tick must be boolean")


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    resolved = copy.deepcopy(config)
    resolved["_meta"] = {
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "resolved_sha256": hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest(),
        "repo_root": str(REPO_ROOT),
    }
    return resolved


def public_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON/YAML-safe copy without machine-specific absolute paths."""
    result = copy.deepcopy(config)
    if "_meta" in result:
        result["_meta"].pop("repo_root", None)
    return result
