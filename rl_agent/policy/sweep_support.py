"""Shared replay and execution helpers for immutable Track A sensitivity runs."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .catalog import flatten_actions, load_profile_catalog
from .channel import ChannelProcess, ChannelSurface
from .config import REPO_ROOT
from .env import SurrogateEnv
from .oracles import run_oracle
from .replay import TraceRecord, discover_trace_registry, load_trace_episode
from .run_pilot import _select_episodes


Replay = Tuple[TraceRecord, list]


def markdown_table(frame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def prepare_replays(config: Mapping[str, object], ranges_m: Iterable[float]) -> tuple[list, Dict[float, List[Replay]]]:
    """Select one held-out episode set, then load that same set at every requested range."""
    registry = discover_trace_registry(config)
    selected_at_default = _select_episodes(config, registry)
    records = [record for record, _ in selected_at_default]
    by_range: Dict[float, List[Replay]] = {}
    for range_m in sorted({float(value) for value in ranges_m}):
        loaded: List[Replay] = []
        for record in records:
            frames = load_trace_episode(
                record,
                config,
                range_m=range_m,
                max_steps=int(config["replay"]["max_episode_steps"]),
            )
            if not frames:
                raise ValueError(f"selected replay {record.episode_id} is empty at range {range_m:g} m")
            loaded.append((record, frames))
        by_range[range_m] = loaded
    expected_ids = [record.episode_id for record in records]
    for range_m, loaded in by_range.items():
        if [record.episode_id for record, _ in loaded] != expected_ids:
            raise AssertionError(f"replay identity changed at range {range_m:g} m")
    return registry, by_range


def selected_manifest(replays_by_range: Mapping[float, Sequence[Replay]]) -> list:
    rows = []
    for range_m, selected in sorted(replays_by_range.items()):
        for record, frames in selected:
            rows.append(
                {
                    "range_m": float(range_m),
                    "episode_id": record.episode_id,
                    "run_group": record.run_group,
                    "scenario_family": record.scenario_family,
                    "split": record.split,
                    "ground_truth_path": str(record.ground_truth_path.relative_to(REPO_ROOT)),
                    "ground_truth_sha256": record.ground_truth_sha256,
                    "prediction_path": str(record.prediction_path.relative_to(REPO_ROOT)),
                    "prediction_sha256": record.prediction_sha256,
                    "frame_count": len(frames),
                }
            )
    return rows


def source_hashes(surface: ChannelSurface) -> dict:
    metadata_path = REPO_ROOT / "rl_agent" / "policy" / "data" / "action_catalog.meta.json"
    return {**surface.source_hashes, **json.loads(metadata_path.read_text(encoding="utf-8"))}


def run_cell(
    config: Mapping[str, object],
    surface: ChannelSurface,
    selected: Sequence[Replay],
    controllers: Sequence[str],
    common_random_latency_by_tick: bool,
    cell_fields: Mapping[str, object],
) -> list[dict]:
    """Run one fully resolved cell with paired channel and per-tick latency randomness."""
    profiles = load_profile_catalog(config["actions"]["catalog_csv"])
    actions = flatten_actions(
        profiles,
        config["actions"]["fps"],
        int(config["actions"]["preferred_core_kib"]),
    )
    channel_seeds = [int(value) for value in config["pilot"]["channel_seeds"]]
    rows: list[dict] = []
    for episode_index, (record, frames) in enumerate(selected):
        seed = channel_seeds[episode_index % len(channel_seeds)]
        for controller in controllers:
            channel = ChannelProcess(config, surface, seed)
            env = SurrogateEnv(
                config,
                frames,
                actions,
                channel,
                surface,
                seed + 10_000,
                latency_mode="sample",
                latency_crn_by_tick=common_random_latency_by_tick,
            )
            result = run_oracle(env, controller)
            for row in result.rows:
                row.update(cell_fields)
                row["scenario"] = record.run_group
                row["scenario_family"] = record.scenario_family
                row["replay_split"] = record.split
                rows.append(row)
    return rows
