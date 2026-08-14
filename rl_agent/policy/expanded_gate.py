"""Desk-only expanded SPLIT+SKIP feasibility and oracle headroom gate.

This module deliberately reuses the accepted single-UE event surrogate.  It
does not emulate a shared network queue; aggregate C1 shaping and a fail-closed
miss-rate gate define the narrow validity envelope documented in the spec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from rl_agent.multiue_oai.endpoint import frame_onwire_bytes

from .catalog import Action, flatten_actions, load_profile_catalog
from .channel import ChannelProcess, ChannelSurface, RUNG_BY_MCS
from .config import REPO_ROOT, validate_config
from .env import SurrogateEnv
from .replay import TraceRecord, load_trace_episode
from .shield import ActionEvaluation


CONTROLLERS = (
    "expanded_decentralized_greedy",
    "expanded_joint_true_state_one_step_oracle",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_gate_spec(path: str | Path) -> Dict[str, object]:
    config_path = _resolve(path)
    with config_path.open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if spec.get("schema") in {
        "scenesense.policy.expanded_action_gate.v2",
        "scenesense.policy.expanded_action_gate.v3",
    }:
        base_entry = spec["base_config"]
        base_path = _resolve(base_entry["path"])
        actual_base_hash = sha256_file(base_path)
        if actual_base_hash != str(base_entry["sha256"]):
            raise ValueError("expanded-action v2 base-config hash mismatch")
        inherited = load_gate_spec(base_path)
        inherited.pop("_meta", None)
        inherited["schema"] = spec["schema"]
        inherited["correction"] = dict(spec["correction"])
        if "oracle_truth_scope" in spec["correction"]:
            inherited["evaluation"]["oracle_truth_scope"] = str(
                spec["correction"]["oracle_truth_scope"]
            )
        for key in (
            "evaluation_mode",
            "system_degradation",
            "primary_frame_filter",
            "oracle_interpretation",
        ):
            if key in spec["correction"]:
                inherited["evaluation"][key] = str(spec["correction"][key])
        spec = inherited
    elif spec.get("schema") != "scenesense.policy.expanded_action_gate.v1":
        raise ValueError("unsupported expanded-action gate schema")
    if list(spec["evaluation"]["controllers"]) != list(CONTROLLERS):
        raise ValueError("expanded-action gate must retain the frozen two controllers")
    if list(spec["evaluation"]["action_modes"]) != ["SPLIT", "SKIP"]:
        raise ValueError("v1 action modes must be exactly SPLIT and SKIP")
    if spec["evaluation"]["local_status"] != "excluded_uncalibrated":
        raise ValueError("LOCAL must remain excluded until calibrated")
    forbidden = set(spec["authorization"]["forbidden"])
    if not {"OAI", "CARLA", "LOCAL", "MPC", "RL"}.issubset(forbidden):
        raise ValueError("desk-only authorization guard is incomplete")
    spec["_meta"] = {
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
    }
    return spec


def verify_frozen_sources(spec: Mapping[str, object]) -> Dict[str, str]:
    verified: Dict[str, str] = {}
    for name, entry in spec["sources"].items():
        if name == "accepted_controller_run":
            continue
        path = _resolve(entry["path"])
        actual = sha256_file(path)
        expected = str(entry["sha256"])
        if actual != expected:
            raise ValueError(f"frozen source hash mismatch for {name}: {actual} != {expected}")
        verified[str(path.relative_to(REPO_ROOT))] = actual
    return verified


def load_accepted_config(spec: Mapping[str, object]) -> Dict[str, object]:
    path = _resolve(spec["sources"]["resolved_config"]["path"])
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    if int(config["reward"]["formulation_version"]) != 5:
        raise ValueError("accepted replay config is not reward v5")
    return config


def load_frozen_registry(spec: Mapping[str, object]) -> Dict[str, TraceRecord]:
    path = _resolve(spec["sources"]["replay_registry"]["path"])
    frame = pd.read_csv(path)
    if frame["episode_id"].astype(str).duplicated().any():
        raise ValueError("accepted replay registry has duplicate episode IDs")
    records: Dict[str, TraceRecord] = {}
    for row in frame.itertuples(index=False):
        gt_path = _resolve(str(row.ground_truth_path))
        prediction_value = str(row.prediction_path).strip()
        prediction_path = _resolve(prediction_value) if prediction_value else None
        if sha256_file(gt_path) != str(row.ground_truth_sha256):
            raise ValueError(f"ground-truth hash drift for {row.episode_id}")
        if prediction_path is not None and sha256_file(prediction_path) != str(row.prediction_sha256):
            raise ValueError(f"prediction hash drift for {row.episode_id}")
        records[str(row.episode_id)] = TraceRecord(
            episode_id=str(row.episode_id),
            run_group=str(row.run_group),
            scenario_family=str(row.scenario_family),
            split=str(row.split),
            ground_truth_path=gt_path,
            prediction_path=prediction_path,
            ground_truth_sha256=str(row.ground_truth_sha256),
            prediction_sha256=str(row.prediction_sha256) if prediction_path is not None else None,
        )
    return records


def load_actions(config: Mapping[str, object], spec: Mapping[str, object]) -> List[Action]:
    catalog = load_profile_catalog(spec["sources"]["action_catalog"]["path"])
    actions = flatten_actions(
        catalog,
        config["actions"]["fps"],
        int(config["actions"]["preferred_core_kib"]),
    )
    if len(actions) != 36 or {action.mode for action in actions} != {"SPLIT", "SKIP"}:
        raise ValueError("expanded v1 action catalog must contain 35 SPLIT actions plus SKIP")
    return actions


def _profile_rows(config: Mapping[str, object], spec: Mapping[str, object]) -> List[Dict[str, object]]:
    catalog = load_profile_catalog(spec["sources"]["action_catalog"]["path"])
    rows = []
    reference_compute = float(config["latency_projection"]["reference_front_plus_back_ms"])
    for row in catalog.to_dict(orient="records"):
        rows.append(
            {
                "profile_id": str(row["profile_id"]),
                "payload_kib": float(row["payload_kib"]),
                "compute_ms": float(row["front_ms"]) + float(row["back_ms"]),
                "source": "measured_catalog",
            }
        )
    for payload in spec["frontier"]["stress_payload_kib"]:
        rows.append(
            {
                "profile_id": f"stress_{float(payload):g}k_reference_compute",
                "payload_kib": float(payload),
                "compute_ms": reference_compute,
                "source": "stress_payload_reference_compute",
            }
        )
    return rows


def compute_feasibility_frontier(
    config: Mapping[str, object], spec: Mapping[str, object]
) -> pd.DataFrame:
    combined = pd.read_csv(_resolve(spec["sources"]["combined_surface"]["path"]))
    combined["rung"] = combined["mcs"].map(RUNG_BY_MCS)
    capacities = combined.groupby("rung", as_index=False)["sched_ul_mbps"].max()
    capacity_by_rung = dict(zip(capacities["rung"], capacities["sched_ul_mbps"]))
    nonuplink_ms = float(config["latency_projection"]["fast_full_pipeline_p95_ms"]) - float(
        config["latency_projection"]["fast_uplink_p95_ms"]
    )
    reference_compute = float(config["latency_projection"]["reference_front_plus_back_ms"])
    c1 = float(spec["frontier"]["c1_factor"])
    chunk_bytes = int(spec["frontier"]["chunk_bytes"])
    output: List[Dict[str, object]] = []
    for profile in _profile_rows(config, spec):
        payload_bytes = int(round(float(profile["payload_kib"]) * 1024.0))
        onwire_bytes = frame_onwire_bytes(payload_bytes, chunk_bytes)
        overhead_bytes = onwire_bytes - payload_bytes
        nonnetwork_p95_ms = nonuplink_ms + float(profile["compute_ms"]) - reference_compute
        for ue_count in spec["frontier"]["ue_counts"]:
            ue_count = int(ue_count)
            for rung_name in config["channel"]["rungs"]:
                capacity = float(capacity_by_rung[rung_name])
                shares = {
                    "whole_cell_optimistic": capacity,
                    "equal_raw": capacity / ue_count,
                    "equal_c1": c1 * capacity / ue_count,
                }
                for deadline_s in spec["frontier"]["deadlines_s"]:
                    for fps in spec["frontier"]["fps"]:
                        onwire_offered_mbps = onwire_bytes * 8.0 * int(fps) / 1_000_000.0
                        c1_rate_budget_mbps = c1 * capacity / ue_count
                        rate_feasible = onwire_offered_mbps <= c1_rate_budget_mbps + 1e-12
                        for envelope, share_mbps in shares.items():
                            serialization_ms = onwire_bytes * 8.0 / (share_mbps * 1_000_000.0) * 1000.0
                            queue_free_p95_ms = nonnetwork_p95_ms + serialization_ms
                            latency_feasible = queue_free_p95_ms <= float(deadline_s) * 1000.0
                            output.append(
                                {
                                    "profile_id": profile["profile_id"],
                                    "profile_source": profile["source"],
                                    "payload_kib": profile["payload_kib"],
                                    "payload_bytes": payload_bytes,
                                    "onwire_bytes": onwire_bytes,
                                    "protocol_overhead_bytes": overhead_bytes,
                                    "ue_count": ue_count,
                                    "rung": rung_name,
                                    "mcs": int(config["channel"]["rungs"][rung_name]["mcs"]),
                                    "measured_cell_ceiling_mbps": capacity,
                                    "share_envelope": envelope,
                                    "share_mbps": share_mbps,
                                    "deadline_s": float(deadline_s),
                                    "target_fps": int(fps),
                                    "onwire_offered_mbps_per_ue": onwire_offered_mbps,
                                    "c1_rate_budget_mbps_per_ue": c1_rate_budget_mbps,
                                    "rate_feasible": bool(rate_feasible),
                                    "nonnetwork_p95_ms": nonnetwork_p95_ms,
                                    "serialization_ms": serialization_ms,
                                    "queue_free_p95_ms": queue_free_p95_ms,
                                    "latency_necessary_feasible": bool(latency_feasible),
                                    "joint_necessary_feasible": bool(rate_feasible and latency_feasible),
                                    "queue_sufficiency_claimed": False,
                                }
                            )
    return pd.DataFrame(output)


def _candidate_evaluations(
    env: SurrogateEnv,
    actions: Sequence[Action],
    *,
    truth: bool,
    external_rate_budget_mbps: float,
    truth_scope: str = "all_truth_objects",
) -> Tuple[List[ActionEvaluation], List[ActionEvaluation], bool]:
    if truth and truth_scope == "matched_deployable_track_keys_with_true_kinematics_and_capacity":
        observation = env.matched_truth_observation()
    elif truth_scope == "all_truth_objects" or not truth:
        observation = env.observation(truth=truth)
    else:
        raise ValueError(f"unsupported oracle truth scope: {truth_scope}")
    rung = env.last_channel_snapshot.rung if truth else observation.observed_channel_rung
    true_capacity = env.last_channel_snapshot.true_capacity_mbps if truth else None
    evaluations = [
        env.shield.evaluate(
            action,
            observation,
            rung,
            env.time_to_next_capture(action),
            true_capacity_mbps=true_capacity,
        )
        for action in actions
    ]
    admitted = [
        item
        for item in evaluations
        if (item.action.mode == "SKIP" or item.action.offered_mbps <= external_rate_budget_mbps + 1e-12)
        and not item.out_of_support
    ]
    if not admitted:
        raise RuntimeError("expanded external C1 gate removed every action including SKIP")
    skip = next(item for item in admitted if item.action.mode == "SKIP")
    epsilon = float(env.config["safety"]["epsilon_m"])
    safe = [item for item in admitted if item.bound_m <= epsilon]
    if safe:
        candidates = list(safe)
        if skip not in candidates:
            candidates.append(skip)
        return evaluations, candidates, True
    best_bound = min(item.bound_m for item in admitted)
    delta = float(env.config["safety"]["delta_loc_m"])
    degraded = [item for item in admitted if item.bound_m <= best_bound + delta]
    if skip not in degraded:
        degraded.append(skip)
    return evaluations, degraded, False


def _select_best(candidates: Sequence[ActionEvaluation]) -> ActionEvaluation:
    return max(candidates, key=lambda item: (item.expected_reward, -item.bound_m, item.action.action_id))


def _joint_oracle(
    candidate_sets: Sequence[Sequence[ActionEvaluation]], budget_mbps: float
) -> List[ActionEvaluation]:
    budget_bps = int(np.floor(budget_mbps * 1_000_000.0 + 1e-9))
    # rate -> (reward, action evaluations); lower-rate equal-reward states dominate.
    states: Dict[int, Tuple[float, Tuple[ActionEvaluation, ...]]] = {0: (0.0, tuple())}
    for candidates in candidate_sets:
        expanded: Dict[int, Tuple[float, Tuple[ActionEvaluation, ...]]] = {}
        for used_bps, (reward, chosen) in states.items():
            for item in candidates:
                rate_bps = int(round(item.action.offered_mbps * 1_000_000.0))
                total_bps = used_bps + rate_bps
                if total_bps > budget_bps:
                    continue
                proposal = (reward + item.expected_reward, chosen + (item,))
                incumbent = expanded.get(total_bps)
                proposal_ids = tuple(value.action.action_id for value in proposal[1])
                incumbent_ids = (
                    tuple(value.action.action_id for value in incumbent[1]) if incumbent is not None else tuple()
                )
                if incumbent is None or proposal[0] > incumbent[0] + 1e-12 or (
                    abs(proposal[0] - incumbent[0]) <= 1e-12 and proposal_ids > incumbent_ids
                ):
                    expanded[total_bps] = proposal
        if not expanded:
            raise RuntimeError("joint oracle found no aggregate-C1 action combination")
        pruned: Dict[int, Tuple[float, Tuple[ActionEvaluation, ...]]] = {}
        best_reward = -float("inf")
        for rate_bps in sorted(expanded):
            value = expanded[rate_bps]
            if value[0] > best_reward + 1e-12:
                pruned[rate_bps] = value
                best_reward = value[0]
        states = pruned
    _, best = max(
        states.items(),
        key=lambda pair: (
            pair[1][0],
            -pair[0],
            tuple(item.action.action_id for item in pair[1][1]),
        ),
    )
    return list(best[1])


@dataclass(frozen=True)
class GroupRun:
    rows: List[Dict[str, object]]
    summary: Dict[str, object]


def _build_envs(
    config: Mapping[str, object],
    actions: Sequence[Action],
    surface: ChannelSurface,
    episodes: Sequence[Sequence[object]],
    channel_seed: int,
) -> List[SurrogateEnv]:
    envs = []
    for ue_index, frames in enumerate(episodes):
        channel = ChannelProcess(config, surface, seed=channel_seed)
        envs.append(
            SurrogateEnv(
                config,
                frames,
                actions,
                channel,
                surface,
                seed=channel_seed * 1000 + ue_index + 17,
                latency_mode="sample",
                latency_crn_by_tick=True,
            )
        )
    return envs


def run_group(
    config: Mapping[str, object],
    actions: Sequence[Action],
    surface: ChannelSurface,
    group_id: str,
    episode_ids: Sequence[str],
    episodes: Sequence[Sequence[object]],
    channel_seed: int,
    controller: str,
    oracle_truth_scope: str,
) -> GroupRun:
    if controller not in CONTROLLERS:
        raise ValueError(f"unsupported expanded controller: {controller}")
    lengths = {len(frames) for frames in episodes}
    if len(lengths) != 1:
        raise ValueError(f"group {group_id} episode lengths differ: {sorted(lengths)}")
    envs = _build_envs(config, actions, surface, episodes, channel_seed)
    c1 = float(config["safety"]["c1_pessimism_factor"])
    rows: List[Dict[str, object]] = []
    aggregate_miss_frames = 0
    total_steps = next(iter(lengths))
    for group_step in range(total_steps):
        snapshots = [env.last_channel_snapshot for env in envs]
        if len({(snap.rung, round(snap.true_capacity_mbps, 12)) for snap in snapshots}) != 1:
            raise RuntimeError("paired UEs lost the common cell channel realization")
        true_capacity = snapshots[0].true_capacity_mbps
        if controller == "expanded_decentralized_greedy":
            selected = []
            feasible_flags = []
            for env in envs:
                budget = c1 * env.last_channel_snapshot.estimated_capacity_mbps / len(envs)
                _, candidates, feasible = _candidate_evaluations(
                    env,
                    actions,
                    truth=False,
                    external_rate_budget_mbps=budget,
                )
                selected.append(_select_best(candidates))
                feasible_flags.append(feasible)
        else:
            candidate_sets = []
            feasible_flags = []
            joint_budget = c1 * true_capacity
            for env in envs:
                _, candidates, feasible = _candidate_evaluations(
                    env,
                    actions,
                    truth=True,
                    external_rate_budget_mbps=joint_budget,
                    truth_scope=oracle_truth_scope,
                )
                candidate_sets.append(candidates)
                feasible_flags.append(feasible)
            selected = _joint_oracle(candidate_sets, joint_budget)

        aggregate_offered = sum(item.action.offered_mbps for item in selected)
        aggregate_miss = aggregate_offered > c1 * true_capacity + 1e-12
        aggregate_miss_frames += int(aggregate_miss)
        for ue_index, (env, item) in enumerate(zip(envs, selected)):
            matched = env.matched_truth_evaluation(item.action)
            result = env.step(item.action)
            latency_ms = result["actual_latency_ms"]
            result.update(
                {
                    "controller": controller,
                    "group_id": group_id,
                    "group_step": group_step,
                    "channel_seed": int(channel_seed),
                    "ue_index": ue_index,
                    "assigned_episode_id": episode_ids[ue_index],
                    "group_ue_count": len(envs),
                    "expanded_candidate_feasible": bool(feasible_flags[ue_index]),
                    "selected_expected_reward_information_set": item.expected_reward,
                    "matched_truth_reward_v5": matched.expected_reward,
                    "matched_truth_g_m": matched.expected_g_m,
                    "matched_truth_safe": matched.risk_p95_m
                    <= float(config["safety"]["epsilon_m"]),
                    "group_aggregate_offered_mbps": aggregate_offered,
                    "group_c1_true_budget_mbps": c1 * true_capacity,
                    "group_aggregate_c1_miss": aggregate_miss,
                    "within_250ms": bool(latency_ms is not None and float(latency_ms) <= 250.0),
                    "within_500ms": bool(latency_ms is not None and float(latency_ms) <= 500.0),
                }
            )
            rows.append(result)
    frame = pd.DataFrame(rows)
    ue_rewards = frame.groupby("ue_index")["matched_truth_reward_v5"].mean()
    captures = frame[frame["captured"]]
    summary = {
        "controller": controller,
        "group_id": group_id,
        "channel_seed": int(channel_seed),
        "ue_count": len(envs),
        "steps": total_steps,
        "ue_frame_count": len(frame),
        "mean_reward_v5": float(frame["matched_truth_reward_v5"].mean()),
        "worst_ue_mean_reward_v5": float(ue_rewards.min()),
        "mean_g_m": float(frame["matched_truth_g_m"].replace([np.inf], np.nan).mean()),
        "safe_fraction": float(frame["matched_truth_safe"].mean()),
        "split_fraction": float((frame["mode"] == "SPLIT").mean()),
        "attempt_count": int(len(captures)),
        "within_250ms_fraction": float(captures["within_250ms"].mean()) if len(captures) else 0.0,
        "within_500ms_fraction": float(captures["within_500ms"].mean()) if len(captures) else 0.0,
        "aggregate_c1_miss_frames": aggregate_miss_frames,
        "aggregate_c1_miss_fraction": aggregate_miss_frames / total_steps,
    }
    return GroupRun(rows=rows, summary=summary)


def _system_oracle_decision(
    envs: Sequence[SurrogateEnv],
    actions: Sequence[Action],
    joint_budget_mbps: float,
    truth_scope: str,
) -> Tuple[List[ActionEvaluation], bool]:
    """Exact common-state choice with degradation applied after joint feasibility.

    The v1/v2 local ``best_bound + delta`` filter is intentionally absent here.
    It is not valid to remove a lower-rate action before knowing whether the
    locally best actions fit together under the aggregate cell budget.
    """

    admitted_sets: List[List[ActionEvaluation]] = []
    safe_sets: List[List[ActionEvaluation]] = []
    for env in envs:
        if truth_scope != "matched_deployable_track_keys_with_true_kinematics_and_capacity":
            raise ValueError(f"unsupported v3 oracle truth scope: {truth_scope}")
        observation = env.matched_truth_observation()
        evaluations = [
            env.shield.evaluate(
                action,
                observation,
                env.last_channel_snapshot.rung,
                env.time_to_next_capture(action),
                true_capacity_mbps=env.last_channel_snapshot.true_capacity_mbps,
            )
            for action in actions
        ]
        admitted = [
            item
            for item in evaluations
            if not item.out_of_support
            and (
                item.action.mode == "SKIP"
                or item.action.offered_mbps <= joint_budget_mbps + 1e-12
            )
        ]
        if not admitted or not any(item.action.mode == "SKIP" for item in admitted):
            raise RuntimeError("system oracle lost the mandatory SKIP action")
        admitted_sets.append(admitted)
        epsilon = float(env.config["safety"]["epsilon_m"])
        safe_sets.append([item for item in admitted if item.bound_m <= epsilon])

    if all(safe_sets):
        try:
            return _joint_oracle(safe_sets, joint_budget_mbps), True
        except RuntimeError:
            pass
    return _joint_oracle(admitted_sets, joint_budget_mbps), False


def _counterfactual_summary(
    frame: pd.DataFrame,
    controller: str,
    group_id: str,
    channel_seed: int,
    ue_count: int,
    total_steps: int,
) -> Dict[str, object]:
    selected = frame[frame["controller"] == controller]
    eligible = selected[selected["primary_eligible"]]
    if eligible.empty:
        raise RuntimeError(f"no primary-eligible frames for {controller}/{group_id}/{channel_seed}")
    ue_rewards = eligible.groupby("ue_index")["matched_truth_reward_v5"].mean()
    captures = eligible[eligible["captured"]]
    return {
        "controller": controller,
        "group_id": group_id,
        "channel_seed": int(channel_seed),
        "ue_count": int(ue_count),
        "steps": int(total_steps),
        "ue_frame_count": int(len(selected)),
        "primary_eligible_ue_frames": int(len(eligible)),
        "primary_eligible_fraction": float(len(eligible) / len(selected)),
        "mean_reward_v5": float(eligible["matched_truth_reward_v5"].mean()),
        "worst_ue_mean_reward_v5": float(ue_rewards.min()),
        "mean_g_m": float(eligible["matched_truth_g_m"].replace([np.inf], np.nan).mean()),
        "safe_fraction": float(eligible["matched_truth_safe"].mean()),
        "split_fraction": float((eligible["mode"] == "SPLIT").mean()),
        "attempt_count": int(len(captures)),
        "within_250ms_fraction": float(captures["within_250ms"].mean()) if len(captures) else 0.0,
        "within_500ms_fraction": float(captures["within_500ms"].mean()) if len(captures) else 0.0,
        "aggregate_c1_miss_frames": int(selected["group_aggregate_c1_miss"].sum() / ue_count),
        "aggregate_c1_miss_fraction": float(
            selected.groupby("group_step")["group_aggregate_c1_miss"].first().mean()
        ),
    }


def run_common_state_group(
    config: Mapping[str, object],
    actions: Sequence[Action],
    surface: ChannelSurface,
    group_id: str,
    episode_ids: Sequence[str],
    episodes: Sequence[Sequence[object]],
    channel_seed: int,
    oracle_truth_scope: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Advance greedy only; score greedy and the exact oracle on each common state."""

    lengths = {len(frames) for frames in episodes}
    if len(lengths) != 1:
        raise ValueError(f"group {group_id} episode lengths differ: {sorted(lengths)}")
    total_steps = next(iter(lengths))
    envs = _build_envs(config, actions, surface, episodes, channel_seed)
    c1 = float(config["safety"]["c1_pessimism_factor"])
    epsilon = float(config["safety"]["epsilon_m"])
    rows: List[Dict[str, object]] = []

    for group_step in range(total_steps):
        snapshots = [env.last_channel_snapshot for env in envs]
        if len({(snap.rung, round(snap.true_capacity_mbps, 12)) for snap in snapshots}) != 1:
            raise RuntimeError("paired UEs lost the common cell channel realization")
        true_capacity = snapshots[0].true_capacity_mbps
        greedy_items: List[ActionEvaluation] = []
        for env in envs:
            per_ue_budget = c1 * env.last_channel_snapshot.estimated_capacity_mbps / len(envs)
            _, candidates, _ = _candidate_evaluations(
                env,
                actions,
                truth=False,
                external_rate_budget_mbps=per_ue_budget,
            )
            greedy_items.append(_select_best(candidates))
        greedy_truth = [
            env.matched_truth_evaluation(item.action)
            for env, item in zip(envs, greedy_items)
        ]
        oracle_items, joint_safe_exists = _system_oracle_decision(
            envs,
            actions,
            c1 * true_capacity,
            oracle_truth_scope,
        )
        greedy_aggregate = sum(item.action.offered_mbps for item in greedy_items)
        greedy_c1_miss = greedy_aggregate > c1 * true_capacity + 1e-12
        greedy_all_safe = all(item.bound_m <= epsilon for item in greedy_truth)
        primary_eligible = not greedy_c1_miss and (not joint_safe_exists or greedy_all_safe)

        # Counterfactual fields are frozen before stepping the common greedy state.
        oracle_rows: List[Dict[str, object]] = []
        oracle_aggregate = sum(item.action.offered_mbps for item in oracle_items)
        for ue_index, (env, item) in enumerate(zip(envs, oracle_items)):
            capture_due = item.action.mode == "SPLIT" and env.time_to_next_capture(item.action) <= 1e-12
            if item.action.mode == "SPLIT":
                latency = env.latency.estimate(item.action, env.last_channel_snapshot.rung)
                predicted_p95_ms = latency.p95_ms
            else:
                predicted_p95_ms = None
            oracle_rows.append(
                {
                    "episode_id": env.frame.episode_id,
                    "step_index": env.step_index,
                    "timestamp_s": env.frame.timestamp_s,
                    "action_id": item.action.action_id,
                    "mode": item.action.mode,
                    "profile_id": item.action.profile_id or "",
                    "target_fps": item.action.target_fps,
                    "channel_rung_true": env.last_channel_snapshot.rung,
                    "channel_rung_observed": env.last_channel_snapshot.observed_rung,
                    "true_capacity_mbps": true_capacity,
                    "estimated_capacity_mbps": env.last_channel_snapshot.estimated_capacity_mbps,
                    "offered_mbps": item.action.offered_mbps,
                    "captured": capture_due,
                    "actual_delivery": None,
                    "actual_latency_ms": None,
                    "counterfactual_p95_latency_ms": predicted_p95_ms,
                    "controller": CONTROLLERS[1],
                    "counterfactual_common_state": True,
                    "group_id": group_id,
                    "group_step": group_step,
                    "channel_seed": int(channel_seed),
                    "ue_index": ue_index,
                    "assigned_episode_id": episode_ids[ue_index],
                    "group_ue_count": len(envs),
                    "system_joint_safe_combination_exists": joint_safe_exists,
                    "selected_expected_reward_information_set": item.expected_reward,
                    "matched_truth_reward_v5": item.expected_reward,
                    "matched_truth_g_m": item.expected_g_m,
                    "matched_truth_safe": item.bound_m <= epsilon,
                    "group_aggregate_offered_mbps": oracle_aggregate,
                    "group_c1_true_budget_mbps": c1 * true_capacity,
                    "group_aggregate_c1_miss": False,
                    "greedy_reference_aggregate_c1_miss": greedy_c1_miss,
                    "greedy_reference_all_matched_safe": greedy_all_safe,
                    "primary_eligible": primary_eligible,
                    "within_250ms": bool(
                        capture_due and predicted_p95_ms is not None and predicted_p95_ms <= 250.0
                    ),
                    "within_500ms": bool(
                        capture_due and predicted_p95_ms is not None and predicted_p95_ms <= 500.0
                    ),
                    "truth_object_count": len(env.frame.truth_objects),
                    "observed_object_count": len(env.frame.observed_objects),
                }
            )

        for ue_index, (env, selected, matched) in enumerate(
            zip(envs, greedy_items, greedy_truth)
        ):
            actual = env.step(selected.action)
            actual.update(
                {
                    "counterfactual_p95_latency_ms": None,
                    "controller": CONTROLLERS[0],
                    "counterfactual_common_state": False,
                    "group_id": group_id,
                    "group_step": group_step,
                    "channel_seed": int(channel_seed),
                    "ue_index": ue_index,
                    "assigned_episode_id": episode_ids[ue_index],
                    "group_ue_count": len(envs),
                    "system_joint_safe_combination_exists": joint_safe_exists,
                    "selected_expected_reward_information_set": selected.expected_reward,
                    "matched_truth_reward_v5": matched.expected_reward,
                    "matched_truth_g_m": matched.expected_g_m,
                    "matched_truth_safe": matched.bound_m <= epsilon,
                    "group_aggregate_offered_mbps": greedy_aggregate,
                    "group_c1_true_budget_mbps": c1 * true_capacity,
                    "group_aggregate_c1_miss": greedy_c1_miss,
                    "greedy_reference_aggregate_c1_miss": greedy_c1_miss,
                    "greedy_reference_all_matched_safe": greedy_all_safe,
                    "primary_eligible": primary_eligible,
                    "within_250ms": bool(
                        actual["actual_latency_ms"] is not None
                        and float(actual["actual_latency_ms"]) <= 250.0
                    ),
                    "within_500ms": bool(
                        actual["actual_latency_ms"] is not None
                        and float(actual["actual_latency_ms"]) <= 500.0
                    ),
                }
            )
            rows.append(actual)
        rows.extend(oracle_rows)

    frame = pd.DataFrame(rows)
    summaries = [
        _counterfactual_summary(
            frame,
            controller,
            group_id,
            channel_seed,
            len(envs),
            total_steps,
        )
        for controller in CONTROLLERS
    ]
    return rows, summaries


def run_expanded_evaluation(
    config: Mapping[str, object],
    spec: Mapping[str, object],
    actions: Sequence[Action],
    surface: ChannelSurface,
    registry: Mapping[str, TraceRecord],
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = {
        str(episode_id)
        for group in spec["evaluation"]["groups"]
        for episode_id in group["episodes"]
    }
    missing = selected_ids - set(registry)
    if missing:
        raise ValueError(f"frozen evaluation episodes missing from registry: {sorted(missing)}")
    frames_by_episode = {}
    for episode_id in sorted(selected_ids):
        record = registry[episode_id]
        if record.split != spec["evaluation"]["split"]:
            raise ValueError(f"episode {episode_id} is not in frozen test split")
        frames_by_episode[episode_id] = load_trace_episode(record, config)
    rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for group in spec["evaluation"]["groups"]:
        episode_ids = [str(value) for value in group["episodes"]]
        if int(group["ue_count"]) != len(episode_ids):
            raise ValueError(f"group {group['group_id']} UE count does not match episode list")
        episodes = [frames_by_episode[episode_id] for episode_id in episode_ids]
        for channel_seed in spec["evaluation"]["channel_seeds"]:
            if spec["evaluation"].get("evaluation_mode") == "common_greedy_state_counterfactual":
                group_rows, group_summaries = run_common_state_group(
                    config,
                    actions,
                    surface,
                    str(group["group_id"]),
                    episode_ids,
                    episodes,
                    int(channel_seed),
                    str(spec["evaluation"]["oracle_truth_scope"]),
                )
                rows.extend(group_rows)
                summaries.extend(group_summaries)
                if progress_callback is not None:
                    for controller in CONTROLLERS:
                        progress_callback(
                            {
                                "event": "group_controller_complete",
                                "group_id": str(group["group_id"]),
                                "channel_seed": int(channel_seed),
                                "controller": controller,
                                "completed_cells": len(summaries),
                                "total_cells": len(spec["evaluation"]["groups"])
                                * len(spec["evaluation"]["channel_seeds"])
                                * len(CONTROLLERS),
                            }
                        )
                continue
            for controller in CONTROLLERS:
                result = run_group(
                    config,
                    actions,
                    surface,
                    str(group["group_id"]),
                    episode_ids,
                    episodes,
                    int(channel_seed),
                    controller,
                    str(spec["evaluation"].get("oracle_truth_scope", "all_truth_objects")),
                )
                rows.extend(result.rows)
                summaries.append(result.summary)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "group_controller_complete",
                            "group_id": str(group["group_id"]),
                            "channel_seed": int(channel_seed),
                            "controller": controller,
                            "completed_cells": len(summaries),
                            "total_cells": len(spec["evaluation"]["groups"])
                            * len(spec["evaluation"]["channel_seeds"])
                            * len(CONTROLLERS),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def decide_outcome(summary: pd.DataFrame, spec: Mapping[str, object]) -> Dict[str, object]:
    pivot = summary.pivot(
        index=["group_id", "channel_seed", "ue_count"],
        columns="controller",
        values=["mean_reward_v5", "worst_ue_mean_reward_v5", "aggregate_c1_miss_fraction"],
    )
    greedy = CONTROLLERS[0]
    oracle = CONTROLLERS[1]
    reward_diff = pivot["mean_reward_v5"][oracle] - pivot["mean_reward_v5"][greedy]
    worst_diff = (
        pivot["worst_ue_mean_reward_v5"][oracle]
        - pivot["worst_ue_mean_reward_v5"][greedy]
    )
    paired = reward_diff.rename("reward_lift").reset_index()
    group_lifts = paired.groupby(["group_id", "ue_count"], as_index=False)["reward_lift"].mean()
    absolute_lift = float(group_lifts["reward_lift"].mean())
    group_greedy = (
        summary[summary["controller"] == greedy]
        .groupby(["group_id", "ue_count"], as_index=False)["mean_reward_v5"]
        .mean()
    )
    mean_greedy = float(group_greedy["mean_reward_v5"].mean())
    relative_lift = absolute_lift / max(abs(mean_greedy), 0.1)
    rng = np.random.default_rng(int(spec["decision_gate"]["bootstrap_seed"]))
    values = group_lifts["reward_lift"].to_numpy(dtype=float)
    replicates = int(spec["decision_gate"]["bootstrap_replicates"])
    bootstrap = np.mean(rng.choice(values, size=(replicates, len(values)), replace=True), axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    by_n = group_lifts.groupby("ue_count")["reward_lift"].mean().to_dict()
    worst_regression = float(worst_diff.min())
    greedy_miss = float(pivot["aggregate_c1_miss_fraction"][greedy].max())
    max_miss = float(spec["evaluation"]["maximum_greedy_aggregate_true_capacity_miss_fraction"])
    validity_pass = greedy_miss <= max_miss + 1e-12
    checks = {
        "absolute_lift_pass": absolute_lift
        >= float(spec["decision_gate"]["minimum_absolute_reward_lift"]),
        "relative_lift_pass": relative_lift
        >= float(spec["decision_gate"]["minimum_relative_reward_lift"]),
        "bootstrap_lower_pass": float(ci_low) > 0.0,
        "n2_and_n4_positive_pass": float(by_n.get(2, 0.0)) > 0.0
        and float(by_n.get(4, 0.0)) > 0.0,
        "worst_ue_regression_pass": worst_regression
        >= -float(spec["decision_gate"]["maximum_worst_ue_mean_reward_regression"]),
        "queue_free_c1_validity_pass": validity_pass,
    }
    if not validity_pass:
        verdict = str(spec["decision_gate"]["invalid_result"])
    elif all(checks.values()):
        verdict = str(spec["decision_gate"]["positive_result"])
    else:
        verdict = str(spec["decision_gate"]["negative_result"])
    return {
        "schema": "scenesense.policy.expanded_action_gate.decision.v1",
        "gate_schema": spec["schema"],
        "verdict": verdict,
        "primary_metric": spec["evaluation"]["primary_metric"],
        "group_equal_greedy_mean_reward_v5": mean_greedy,
        "group_equal_oracle_mean_reward_v5": mean_greedy + absolute_lift,
        "absolute_reward_lift": absolute_lift,
        "relative_reward_lift": relative_lift,
        "cluster_bootstrap_95ci": [float(ci_low), float(ci_high)],
        "mean_reward_lift_by_ue_count": {str(key): float(value) for key, value in by_n.items()},
        "minimum_paired_worst_ue_reward_lift": worst_regression,
        "maximum_greedy_aggregate_c1_miss_fraction": greedy_miss,
        "checks": checks,
        "interpretation_boundary": (
            "Queue-free held-out replay upper-bound gate; no shared queue, OAI, CARLA, LOCAL, MPC, or RL."
        ),
    }


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
