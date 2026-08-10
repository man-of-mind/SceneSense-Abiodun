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
