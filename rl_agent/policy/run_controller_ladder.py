"""Train/evaluate the pre-RL Track A controller ladder on a verified corpus."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import yaml

from .catalog import flatten_actions, load_profile_catalog
from .channel import ChannelProcess, ChannelSurface
from .config import REPO_ROOT, load_controller_ladder_config
from .controllers import DeployableController, build_controller
from .env import SurrogateEnv
from .ladder import run_deployable_controller
from .latency import LatencyProjector
from .replay import TraceRecord, discover_trace_registry, load_trace_episode, registry_frame
from .reporting import new_run_directory, save_mode_and_risk_figure, summarize_frames, write_run_files
from .shield import SharedShield


def _canonical_hash(config: Mapping[str, object]) -> str:
    payload = {key: value for key, value in config.items() if key != "_meta"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _apply_corpus_overrides(
    config: Mapping[str, object],
    replay_roots: Sequence[Path],
    split_manifest: Path | None,
    verification_manifest: Path | None,
) -> Dict[str, object]:
    resolved = copy.deepcopy(dict(config))
    if replay_roots:
        resolved["replay"]["roots"] = [
            str(path if not path.is_absolute() else path.relative_to(REPO_ROOT))
            for path in replay_roots
        ]
    if split_manifest is not None:
        resolved["replay"]["split_manifest_csv"] = str(
            split_manifest
            if not split_manifest.is_absolute()
            else split_manifest.relative_to(REPO_ROOT)
        )
    if verification_manifest is not None:
        verification_path = (
            verification_manifest
            if verification_manifest.is_absolute()
            else REPO_ROOT / verification_manifest
        )
        verification_payload = json.loads(
            verification_path.read_text(encoding="utf-8")
        )
        frozen_thresholds = verification_payload.get(
            "prediction_score_min_by_class", {}
        )
        if frozen_thresholds:
            resolved["replay"]["prediction_score_min_by_class"] = {
                str(class_name): float(threshold)
                for class_name, threshold in frozen_thresholds.items()
            }
        resolved["controller_ladder"]["verification_manifest_json"] = str(
            verification_manifest
            if not verification_manifest.is_absolute()
            else verification_manifest.relative_to(REPO_ROOT)
        )
    resolved["_meta"]["runtime_corpus_override"] = bool(
        replay_roots or split_manifest or verification_manifest
    )
    resolved["_meta"]["resolved_sha256"] = _canonical_hash(resolved)
    return resolved


def _verify_corpus_contract(config: Mapping[str, object]) -> None:
    ladder = config["controller_ladder"]
    if not bool(ladder["require_verified_corpus"]):
        return
    roots = [str(value).strip() for value in config["replay"]["roots"]]
    split_manifest = str(config["replay"].get("split_manifest_csv", "")).strip()
    verification_manifest = str(ladder.get("verification_manifest_json", "")).strip()
    base_roots = {"staleness"}
    if not roots or set(roots) == base_roots:
        raise ValueError(
            "verified corrected-vehicle corpus root is required; pass --replay-root or pin replay_roots"
        )
    if not split_manifest:
        raise ValueError(
            "verified episode-level split manifest is required; pass --split-manifest or pin it in config"
        )
    if not verification_manifest:
        raise ValueError(
            "PASS verification manifest is required; pass --verification-manifest or pin it in config"
        )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    verification_path = resolve(verification_manifest)
    split_path = resolve(split_manifest)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("schema") != "policy_corpus_verification.v1":
        raise ValueError("unexpected corpus verification manifest schema")
    if verification.get("status") != "PASS" or verification.get("gate_failures"):
        raise ValueError("controller ladder requires a PASS corpus verification with no gate failures")
    verified_thresholds = {
        str(class_name): float(threshold)
        for class_name, threshold in verification.get(
            "prediction_score_min_by_class", {}
        ).items()
    }
    replay_thresholds = {
        str(class_name): float(threshold)
        for class_name, threshold in config["replay"].get(
            "prediction_score_min_by_class", {}
        ).items()
    }
    if verified_thresholds and replay_thresholds != verified_thresholds:
        raise ValueError(
            "replay per-class thresholds do not match the PASS verification"
        )
    split_artifact = verification.get("artifacts", {}).get("replay_split_manifest.csv")
    if not split_artifact or split_artifact.get("sha256") != hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest():
        raise ValueError("split manifest does not match the PASS verification artifact hash")

    batch_dir = verification_path.parent.parent.parent
    resolved_roots = {resolve(value).resolve() for value in roots}
    if batch_dir.resolve() not in resolved_roots:
        raise ValueError("replay root must include the batch certified by the verification manifest")
    batch_manifest_path = batch_dir / "batch_manifest.json"
    if hashlib.sha256(batch_manifest_path.read_bytes()).hexdigest() != verification.get(
        "batch_manifest_sha256"
    ):
        raise ValueError("batch manifest hash does not match the PASS verification manifest")
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    if batch_manifest.get("mode") != "full":
        raise ValueError("controller ladder headline input must be a full corpus batch, not smoke")
    collection_config_path = batch_dir / "resolved_collection_config.yaml"
    if hashlib.sha256(collection_config_path.read_bytes()).hexdigest() != verification.get(
        "collection_config_sha256"
    ):
        raise ValueError("resolved collection config hash does not match verification")
    with collection_config_path.open("r", encoding="utf-8") as stream:
        collection_config = yaml.safe_load(stream)
    if str(collection_config.get("experiment_name")) != str(ladder["corpus_id"]):
        raise ValueError("verified collection experiment_name does not match controller corpus_id")


def _load_split(
    config: Mapping[str, object],
    registry: Sequence[TraceRecord],
    split: str,
    episode_count: int | None,
) -> List[Tuple[TraceRecord, list]]:
    selected = []
    for record in registry:
        if record.split != split or record.prediction_path is None:
            continue
        frames = load_trace_episode(
            record,
            config,
            range_m=float(config["safety"]["range_m"]),
            max_steps=int(config["replay"]["max_episode_steps"]),
        )
        if not any(frame.truth_objects for frame in frames):
            continue
        if not any(frame.observed_objects for frame in frames):
            continue
        selected.append((record, frames))
        if episode_count is not None and len(selected) >= int(episode_count):
            break
    if not selected:
        raise ValueError(f"no usable paired episodes found for {split} split")
    if episode_count is not None and len(selected) < int(episode_count):
        raise ValueError(
            f"requested {episode_count} usable {split} episodes, found {len(selected)}"
        )
    return selected


def _new_env(config, frames, actions, surface, channel_seed: int) -> SurrogateEnv:
    channel = ChannelProcess(config, surface, channel_seed)
    return SurrogateEnv(
        config,
        frames,
        actions,
        channel,
        surface,
        channel_seed + 10_000,
        latency_mode="sample",
        latency_crn_by_tick=bool(config["controller_ladder"]["common_random_latency_by_tick"]),
    )


def _manifest_record(record: TraceRecord, frames: Sequence[object]) -> Dict[str, object]:
    return {
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


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def run(
    config_path: Path | None = None,
    *,
    replay_roots: Sequence[Path] = (),
    split_manifest: Path | None = None,
    verification_manifest: Path | None = None,
    scaffold_smoke: bool = False,
) -> Path:
    config = _apply_corpus_overrides(
        load_controller_ladder_config(config_path),
        replay_roots,
        split_manifest,
        verification_manifest,
    )
    _verify_corpus_contract(config)
    ladder = config["controller_ladder"]
    profiles = load_profile_catalog(config["actions"]["catalog_csv"])
    actions = flatten_actions(
        profiles, config["actions"]["fps"], config["actions"]["preferred_core_kib"]
    )
    action_ids = {action.action_id for action in actions}
    fixed_action = str(ladder["controllers"]["fixed"]["action_id"])
    if fixed_action not in action_ids:
        raise ValueError(f"fixed controller action is outside the canonical catalog: {fixed_action}")

    surface = ChannelSurface(config)
    registry = discover_trace_registry(config)
    train_count = 1 if scaffold_smoke else ladder["training_episode_count"]
    evaluation_count = 1 if scaffold_smoke else ladder["evaluation_episode_count"]
    training = _load_split(config, registry, ladder["training_split"], train_count)
    evaluation = _load_split(config, registry, ladder["evaluation_split"], evaluation_count)
    seeds = [int(value) for value in ladder["channel_seeds"]]
    reference_shield = SharedShield(config, LatencyProjector(config, surface))
    controllers: Dict[str, DeployableController] = {
        name: build_controller(name, config, actions, reference_shield, ladder)
        for name in ladder["enabled_controllers"]
    }

    training_rows = []
    if "linucb" in controllers:
        bandit = controllers["linucb"]
        for episode_index, (record, frames) in enumerate(training):
            channel_seed = seeds[episode_index % len(seeds)] + 50_000
            result = run_deployable_controller(
                _new_env(config, frames, actions, surface, channel_seed),
                bandit,
                training=True,
                feedback_source=str(ladder["bandit_feedback_source"]),
            )
            for row in result.rows:
                row.update(
                    {
                        "scenario": record.run_group,
                        "scenario_family": record.scenario_family,
                        "replay_split": record.split,
                    }
                )
                training_rows.append(row)

    rows = []
    for episode_index, (record, frames) in enumerate(evaluation):
        channel_seed = seeds[episode_index % len(seeds)]
        for controller in controllers.values():
            result = run_deployable_controller(
                _new_env(config, frames, actions, surface, channel_seed),
                controller,
                training=False,
                feedback_source=str(ladder["bandit_feedback_source"]),
            )
            for row in result.rows:
                row.update(
                    {
                        "scenario": record.run_group,
                        "scenario_family": record.scenario_family,
                        "replay_split": record.split,
                    }
                )
                rows.append(row)

    metrics = pd.DataFrame(rows)
    training_metrics = pd.DataFrame(training_rows)
    per_scenario = summarize_frames(metrics)
    overall = summarize_frames(metrics.assign(scenario="ALL_VERIFIED_VEHICLE_REPLAY"))
    summary = pd.concat([overall, per_scenario], ignore_index=True)
    kind = "controller_ladder_smoke" if scaffold_smoke else "controller_ladder"
    run_dir = new_run_directory(kind)
    registry_frame(registry).to_csv(run_dir / "replay_registry.csv", index=False)
    training_metrics.to_csv(run_dir / "training_metrics.csv", index=False)
    (run_dir / "controller_states.json").write_text(
        json.dumps(
            {name: controller.state_dict() for name, controller in controllers.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_mode_and_risk_figure(
        metrics,
        run_dir / "figures" / "controller_ladder_mode_and_risk",
        "Track A pre-RL controller ladder on verified vehicle replay",
    )
    status = "scaffold_validation" if scaffold_smoke else "completed_surrogate_controller_evaluation"
    display_columns = [
        "controller",
        "frames",
        "split_pct",
        "skip_pct",
        "capture_attempt_pct",
        "over_budget_pct",
        "selected_matched_true_safe_pct",
        "matched_false_admit_conditional_pct",
        "matched_false_reject_conditional_pct",
        "mean_predicted_reward",
        "mean_matched_true_scored_reward_finite",
        "mean_prb_cost",
    ]
    report = "\n".join(
        [
            "# Track A controller ladder results",
            "",
            f"**Implementation status:** `{status}`.",
            "",
            "This is a table-driven SPLIT+SKIP surrogate comparison on the explicitly pinned corrected-vehicle "
            "corpus. It is not a CARLA/OAI run, LOCAL evaluation, live safety validation, or RL result.",
            "",
            "## Shared comparison contract",
            "",
            "- Fixed, threshold rule, one-step greedy, fitted LinUCB, and shielded MPC use the identical "
            "canonical action catalog and live `A_m -> A_safe` implementation.",
            "- LinUCB trains only on the grouped training split, using matched/tracked environment reward "
            "feedback, and is frozen on the evaluation split.",
            "- MPC replans each tick from observable state with declared Markov-expected capacity, modal-rung "
            "latency, and constant-kinematics projections; it receives neither future replay frames nor true "
            "channel capacity.",
            "- DQN/SAC/PPO are intentionally absent until the simpler ladder is reviewed.",
            "",
            "## Evaluation summary",
            "",
            _markdown_table(overall[display_columns].round(4)),
            "",
            "## Interpretation",
            "",
            "A smoke artifact validates plumbing only. A non-smoke artifact is a completed surrogate "
            "controller evaluation, but adoption still requires held-out anticipatory traces and comparable "
            "safety; these results alone do not justify RL.",
            "",
            "## Artifacts",
            "",
            f"Run directory: `{run_dir.relative_to(REPO_ROOT)}`",
            "",
            "See `per_frame_metrics.csv`, `training_metrics.csv`, `summary.csv`, `controller_states.json`, "
            "`replay_registry.csv`, `resolved_config.yaml`, `manifest.json`, and the figure files.",
            "",
        ]
    )
    (run_dir / "CONTROLLER_LADDER_RESULTS.md").write_text(report, encoding="utf-8")
    selected_records = training + evaluation
    catalog_meta_path = REPO_ROOT / "rl_agent" / "policy" / "data" / "action_catalog.meta.json"
    catalog_meta = json.loads(catalog_meta_path.read_text(encoding="utf-8"))
    write_run_files(
        run_dir,
        config,
        metrics,
        summary,
        {
            "run_type": "track_a_pre_rl_controller_ladder",
            "implementation_status": status,
            "corpus_id": ladder["corpus_id"],
            "controllers": list(controllers),
            "training_feedback_source": ladder["bandit_feedback_source"],
            "training_split": ladder["training_split"],
            "evaluation_split": ladder["evaluation_split"],
            "selected_replays": [
                _manifest_record(record, frames) for record, frames in selected_records
            ],
            "source_hashes": {**surface.source_hashes, **catalog_meta},
            "common_random_latency_by_tick": bool(ladder["common_random_latency_by_tick"]),
            "limitations": [
                "SPLIT+SKIP only; LOCAL table remains pending",
                "synthetic Markov channel composed with real corrected-vehicle replay",
                "MPC uses a declared observable-state modal forecast",
                "no pedestrian claims until Track B passes",
                "surrogate validation only",
            ],
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--replay-root", type=Path, action="append", default=[])
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--verification-manifest", type=Path)
    parser.add_argument(
        "--scaffold-smoke",
        action="store_true",
        help="run one train and one test episode; artifact is labelled plumbing validation",
    )
    args = parser.parse_args()
    print(
        run(
            args.config,
            replay_roots=args.replay_root,
            split_manifest=args.split_manifest,
            verification_manifest=args.verification_manifest,
            scaffold_smoke=args.scaffold_smoke,
        )
    )


if __name__ == "__main__":
    main()
