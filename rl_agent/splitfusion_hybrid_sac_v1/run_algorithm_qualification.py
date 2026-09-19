"""Reproducible renderer for bounded Hybrid-SAC algorithm qualification.

Every emitted record is labelled ``HYBRID_SAC_ALGORITHM_QUALIFICATION_ONLY``.
The analytic task validates training mechanics; it is not SplitFusion evidence
and its scores are not system-performance results.  Its state directly encodes
the target mode and target q, so the fixed evaluation stream is not evidence of
generalization beyond the finite analytic state family.

The default command encodes three fixed seeds and 500 updates per seed, but
importing this module launches nothing.  Output is assembled in a sibling
temporary directory and atomically renamed into place only after all hashes
and checkpoints verify.  Any pre-existing target directory is refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

import rl_agent.splitfusion_hybrid_sac_v1 as package_module

from . import (
    action_contract,
    hybrid_sac_models,
    hybrid_sac_trainer,
    replay_buffer,
    reward_ticket_controller,
    scene_descriptors,
    state_reward_transition_contract,
    transaction_identity,
)
from .hybrid_sac_training_runner import (
    PHASE_LABEL,
    AcceptanceThresholdsV1,
    EvaluationMetricsV1,
    HybridSacAlgorithmQualificationRunnerV1,
    QualificationError,
    QualificationRunnerConfigV1,
)

__all__ = [
    "DEFAULT_SEEDS",
    "QualificationRenderConfigV1",
    "RenderError",
    "build_argument_parser",
    "main",
    "render_algorithm_qualification",
    "verify_artifact_directory",
]


MANIFEST_SCHEMA = "splitfusion.hybrid_sac.algorithm_qualification_manifest.v1"
SUMMARY_SCHEMA = "splitfusion.hybrid_sac.algorithm_qualification_summary.v1"
LEARNING_CURVE_SCHEMA = (
    "splitfusion.hybrid_sac.algorithm_qualification_learning_curve.v1"
)
DEFAULT_SEEDS: Tuple[int, int, int] = (17, 29, 43)


class RenderError(Exception):
    """The renderer refused invalid input, unsafe output, or hash drift."""


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RenderError(f"{name} is not finite: {value!r}")
    return numeric


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _git_output(repo_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def _repository_root() -> Path:
    root = _git_output(Path(__file__).resolve().parent, ["rev-parse", "--show-toplevel"])
    return Path(root.strip()).resolve()


def _dirty_path_names(repo_root: Path) -> List[str]:
    """Return names and status codes only; no diff or dirty content is read."""
    output = _git_output(
        repo_root,
        ["-c", "core.quotepath=false", "status", "--porcelain=v1", "--untracked-files=all"],
    )
    paths: List[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            raise RenderError("git status emitted a malformed porcelain row")
        # Retain status beside the name so staged and unstaged state remain
        # auditable, while deliberately never reading the file's contents.
        paths.append(line)
    return sorted(paths)


def _source_bindings(repo_root: Path) -> Dict[str, str]:
    import rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_training_runner as runner_module

    sources = (
        Path(package_module.__file__).resolve(),
        Path(action_contract.__file__).resolve(),
        Path(__file__).resolve(),
        Path(hybrid_sac_models.__file__).resolve(),
        Path(hybrid_sac_trainer.__file__).resolve(),
        Path(replay_buffer.__file__).resolve(),
        Path(reward_ticket_controller.__file__).resolve(),
        Path(runner_module.__file__).resolve(),
        Path(scene_descriptors.__file__).resolve(),
        Path(state_reward_transition_contract.__file__).resolve(),
        Path(transaction_identity.__file__).resolve(),
    )
    bindings: Dict[str, str] = {}
    for source in sources:
        try:
            relative = source.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise RenderError(f"source lies outside repository: {source}") from exc
        bindings[relative] = _sha256_file(source)
    return dict(sorted(bindings.items()))


def _runtime_versions() -> Dict[str, str]:
    """Return exact interpreter/library versions needed for replay."""
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }


@dataclass(frozen=True, slots=True)
class QualificationRenderConfigV1:
    """Artifact-run configuration; defaults are bounded, fixed hypotheses."""

    output: Path
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    updates: int = 500
    evaluation_interval: int = 25
    evaluation_steps: int = 240
    replay_capacity: int = 8_192
    batch_size: int = 64
    warmup_transitions: int = 256
    collect_per_update: int = 2
    episode_horizon: int = 24
    gamma_per_tensor: float = 0.99
    alpha_d: float = 0.10
    alpha_c: float = 0.05
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    fixed_mode: int = 0
    fixed_q_e4: int = 4_900
    minimum_improvement_over_random: float = 0.05
    maximum_oracle_regret: float = 0.35
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.output, Path):
            raise RenderError("output must be a pathlib.Path")
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise RenderError("at least three unique seeds are required")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise RenderError("seeds must be unique non-negative integers")
        for name in (
            "updates",
            "evaluation_interval",
            "evaluation_steps",
            "replay_capacity",
            "batch_size",
            "warmup_transitions",
            "collect_per_update",
            "episode_horizon",
            "fixed_mode",
            "fixed_q_e4",
            "torch_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise RenderError(f"{name} must be an integer")
        if self.updates < 1:
            raise RenderError("updates must be positive")
        if not 1 <= self.evaluation_interval <= self.updates:
            raise RenderError("evaluation_interval must lie in [1, updates]")
        if self.evaluation_steps < 12:
            raise RenderError("evaluation_steps must be at least 12")
        if self.fixed_mode not in range(12):
            raise RenderError("fixed_mode must lie in [0, 11]")
        if not 0 <= self.fixed_q_e4 <= 9_800:
            raise RenderError("fixed_q_e4 must lie in [0, 9800]")
        if self.torch_threads < 1:
            raise RenderError("torch_threads must be positive")
        for name in (
            "gamma_per_tensor",
            "alpha_d",
            "alpha_c",
            "tau",
            "actor_lr",
            "critic_lr",
            "minimum_improvement_over_random",
            "maximum_oracle_regret",
        ):
            _finite(getattr(self, name), name)
        # Delegate remaining learning configuration invariants to the runner.
        self.runner_config(self.seeds[0])
        self.thresholds()

    def runner_config(self, seed: int) -> QualificationRunnerConfigV1:
        return QualificationRunnerConfigV1(
            seed=seed,
            max_updates=self.updates,
            replay_capacity=self.replay_capacity,
            batch_size=self.batch_size,
            warmup_transitions=self.warmup_transitions,
            collect_per_update=self.collect_per_update,
            episode_horizon=self.episode_horizon,
            gamma_per_tensor=self.gamma_per_tensor,
            alpha_d=self.alpha_d,
            alpha_c=self.alpha_c,
            tau=self.tau,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
        )

    def thresholds(self) -> AcceptanceThresholdsV1:
        return AcceptanceThresholdsV1(
            minimum_improvement_over_random=self.minimum_improvement_over_random,
            maximum_oracle_regret=self.maximum_oracle_regret,
        )

    def canonical_parameters(self) -> Dict[str, Any]:
        value = asdict(self)
        value["output"] = None  # Output location is not part of scientific identity.
        value["seeds"] = list(self.seeds)
        return value


LEARNING_CURVE_FIELDS: Tuple[str, ...] = (
    "phase_label",
    "schema",
    "seed",
    "update",
    "collected_transitions",
    "policy_mean_reward",
    "random_mean_reward",
    "fixed_mean_reward",
    "oracle_mean_reward",
    "improvement_over_random",
    "oracle_regret",
    "policy_mode_accuracy",
    "policy_mean_absolute_q_error",
    "policy_selected_mode_count",
    "policy_interior_q_fraction",
    "accepted",
    "critic_loss_total",
    "actor_loss",
    "discrete_entropy",
)


def _curve_record(
    seed: int,
    update: int,
    collected: int,
    evaluation: EvaluationMetricsV1,
    update_metric: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    evaluation.assert_finite()
    metric = update_metric or {}
    record: Dict[str, Any] = {
        "phase_label": PHASE_LABEL,
        "schema": LEARNING_CURVE_SCHEMA,
        "seed": seed,
        "update": update,
        "collected_transitions": collected,
        "policy_mean_reward": evaluation.policy_mean_reward,
        "random_mean_reward": evaluation.random_mean_reward,
        "fixed_mean_reward": evaluation.fixed_mean_reward,
        "oracle_mean_reward": evaluation.oracle_mean_reward,
        "improvement_over_random": evaluation.improvement_over_random,
        "oracle_regret": evaluation.oracle_regret,
        "policy_mode_accuracy": evaluation.policy_mode_accuracy,
        "policy_mean_absolute_q_error": evaluation.policy_mean_absolute_q_error,
        "policy_selected_mode_count": evaluation.policy_selected_mode_count,
        "policy_interior_q_fraction": evaluation.policy_interior_q_fraction,
        "accepted": evaluation.accepted,
        "critic_loss_total": metric.get("critic_loss_total", ""),
        "actor_loss": metric.get("actor_loss", ""),
        "discrete_entropy": metric.get("discrete_entropy", ""),
    }
    for name in (
        "policy_mean_reward",
        "random_mean_reward",
        "fixed_mean_reward",
        "oracle_mean_reward",
        "improvement_over_random",
        "oracle_regret",
        "policy_mode_accuracy",
        "policy_mean_absolute_q_error",
        "policy_interior_q_fraction",
    ):
        _finite(record[name], f"curve {name}")
    for name in ("critic_loss_total", "actor_loss", "discrete_entropy"):
        if record[name] != "":
            _finite(record[name], f"curve {name}")
    return record


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=LEARNING_CURVE_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row[name] for name in LEARNING_CURVE_FIELDS})
    return output.getvalue().encode("utf-8")


def _evaluation_dict(value: EvaluationMetricsV1) -> Dict[str, Any]:
    value.assert_finite()
    document = value.as_dict()
    if document["phase_label"] != PHASE_LABEL:
        raise RenderError("evaluation phase label differs")
    return document


def _aggregate(seed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not seed_results:
        raise RenderError("no seed results were produced")
    keys = (
        "policy_mean_reward",
        "random_mean_reward",
        "fixed_mean_reward",
        "improvement_over_random",
        "oracle_regret",
        "policy_mode_accuracy",
        "policy_mean_absolute_q_error",
    )
    aggregate: Dict[str, Any] = {}
    for key in keys:
        aggregate[f"post_{key}_mean"] = sum(
            float(seed["post_evaluation"][key]) for seed in seed_results
        ) / len(seed_results)
    aggregate["accepted_seed_count"] = sum(
        bool(seed["post_evaluation"]["accepted"]) for seed in seed_results
    )
    aggregate["seed_count"] = len(seed_results)
    for name, value in aggregate.items():
        if isinstance(value, float):
            _finite(value, f"aggregate {name}")
    return aggregate


def _report_text(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    rows = [
        "# Hybrid-SAC Algorithm Qualification",
        "",
        f"**{PHASE_LABEL}**",
        "",
        "This is a bounded analytic-task qualification of training mechanics. "
        "It is not SplitFusion evidence, not a CARLA/OAI measurement, and not "
        "a deployable-policy or system-performance claim.",
        "",
        "## Configuration",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in summary['parameters']['seeds'])}",
        f"- Updates per seed: {summary['parameters']['updates']}",
        f"- Fixed evaluation steps: {summary['parameters']['evaluation_steps']}",
        f"- Evaluation interval: {summary['parameters']['evaluation_interval']} updates",
        f"- PyTorch intra-op threads: {summary['parameters']['torch_threads']}",
        f"- Python: {summary['runtime_versions']['python']}",
        f"- PyTorch: {summary['runtime_versions']['torch']}",
        "- Evaluation: one fixed analytic stream per seed at every checkpoint",
        "- Analytic-state disclosure: each state directly encodes its target "
        "mode and normalized target q; this evaluates mechanics on the same "
        "finite state family and does not measure generalization",
        "- Baselines: paired random action, fixed action, and analytic oracle",
        "",
        "## Result",
        "",
        f"- Accepted seeds: {aggregate['accepted_seed_count']}/{aggregate['seed_count']}",
        f"- Mean post-training policy reward: {aggregate['post_policy_mean_reward_mean']:.6f}",
        f"- Mean improvement over random: {aggregate['post_improvement_over_random_mean']:.6f}",
        f"- Mean oracle regret: {aggregate['post_oracle_regret_mean']:.6f}",
        "",
        "Acceptance here means only that the implementation met the registered "
        "analytic-task thresholds. It does not establish value on measured "
        "SplitFusion transitions.",
        "",
        "## Reproducibility",
        "",
        "Every seed has a hash-bound checkpoint with the initial fixed-evaluation "
        "RNG state needed to reproduce its reported post-evaluation. "
        "`manifest.json` binds every artifact, the complete local executable "
        "source closure, Python/PyTorch versions, Git HEAD, and dirty-path names. "
        "Dirty file contents are never copied into this artifact.",
        "",
    ]
    return "\n".join(rows)


def _artifact_inventory(root: Path, relative_paths: Iterable[str]) -> Dict[str, Any]:
    inventory: Dict[str, Any] = {}
    for relative in sorted(relative_paths):
        path = root / relative
        inventory[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return inventory


def verify_artifact_directory(output: Path, *, verify_sources: bool = True) -> Dict[str, Any]:
    """Verify the exact artifact inventory and optionally current source pins."""
    root = Path(output)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise RenderError("artifact directory or manifest is absent")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RenderError("manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise RenderError("manifest schema differs")
    if manifest.get("phase_label") != PHASE_LABEL:
        raise RenderError("manifest phase label differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RenderError("manifest artifact inventory is invalid")
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if actual != sorted(artifacts):
        raise RenderError("artifact file inventory differs from manifest")
    for relative, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes"}:
            raise RenderError(f"artifact record differs for {relative}")
        path = root / relative
        if path.stat().st_size != record["size_bytes"]:
            raise RenderError(f"artifact size differs for {relative}")
        if _sha256_file(path) != record["sha256"]:
            raise RenderError(f"artifact SHA-256 differs for {relative}")
    if verify_sources:
        repo_root = _repository_root()
        if manifest.get("source_sha256") != _source_bindings(repo_root):
            raise RenderError("qualification source hashes differ")
        if manifest.get("runtime_versions") != _runtime_versions():
            raise RenderError("qualification runtime versions differ")
        if manifest.get("git_head") != _git_output(repo_root, ["rev-parse", "HEAD"]).strip():
            raise RenderError("Git HEAD differs from manifest")
    return manifest


def render_algorithm_qualification(config: QualificationRenderConfigV1) -> Dict[str, Any]:
    """Run, render, hash, verify, then atomically publish one bounded result."""
    # The networks are tiny; allowing PyTorch to fan each operation across the
    # host's full core inventory makes this bounded qualification dramatically
    # slower through oversubscription.  Inter-op threads are intentionally not
    # changed because PyTorch permits that setting only before parallel work.
    torch.set_num_threads(config.torch_threads)
    target = config.output.resolve()
    if target.exists():
        raise RenderError(f"refusing existing output path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    repo_root = _repository_root()
    provenance = {
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]).strip(),
        "git_dirty_paths": _dirty_path_names(repo_root),
        "source_sha256": _source_bindings(repo_root),
        "runtime_versions": _runtime_versions(),
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        curve: List[Dict[str, Any]] = []
        seed_results: List[Dict[str, Any]] = []
        checkpoint_paths: List[str] = []
        thresholds = config.thresholds()
        for seed in config.seeds:
            runner = HybridSacAlgorithmQualificationRunnerV1(
                config.runner_config(seed)
            )
            # Every evaluation for a seed reuses the same analytic contexts and
            # random-baseline draws.  The contexts come from the same finite,
            # target-bearing state family used in training; this controls Monte
            # Carlo drift and deliberately makes no generalization claim.
            runner.reset_fixed_evaluation_stream()
            pre = runner.evaluate(
                config.evaluation_steps,
                fixed_mode=config.fixed_mode,
                fixed_q_e4=config.fixed_q_e4,
                thresholds=thresholds,
            )
            curve.append(
                _curve_record(seed, 0, runner.collected_transitions, pre, None)
            )
            while runner.update_count < config.updates:
                count = min(
                    config.evaluation_interval,
                    config.updates - runner.update_count,
                )
                metrics = runner.advance(count)
                runner.reset_fixed_evaluation_stream()
                evaluation = runner.evaluate(
                    config.evaluation_steps,
                    fixed_mode=config.fixed_mode,
                    fixed_q_e4=config.fixed_q_e4,
                    thresholds=thresholds,
                )
                curve.append(
                    _curve_record(
                        seed,
                        runner.update_count,
                        runner.collected_transitions,
                        evaluation,
                        metrics[-1].as_dict(),
                    )
                )
            post = evaluation
            checkpoint_relative = f"checkpoints/seed_{seed}.pt"
            checkpoint_path = stage / checkpoint_relative
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_sha256 = runner.save_checkpoint(checkpoint_path)
            if _sha256_file(checkpoint_path) != checkpoint_sha256:
                raise RenderError(f"checkpoint hash disagrees for seed {seed}")
            restored = HybridSacAlgorithmQualificationRunnerV1.load_checkpoint(
                checkpoint_path, expected_sha256=checkpoint_sha256
            )
            if restored.update_count != config.updates:
                raise RenderError(f"checkpoint update count differs for seed {seed}")
            restored.reset_fixed_evaluation_stream()
            restored_post = restored.evaluate(
                config.evaluation_steps,
                fixed_mode=config.fixed_mode,
                fixed_q_e4=config.fixed_q_e4,
                thresholds=thresholds,
            )
            if restored_post.as_dict() != post.as_dict():
                raise RenderError(
                    f"checkpoint cannot reproduce post-evaluation for seed {seed}"
                )
            checkpoint_paths.append(checkpoint_relative)
            seed_results.append(
                {
                    "phase_label": PHASE_LABEL,
                    "seed": seed,
                    "updates": runner.update_count,
                    "collected_transitions": runner.collected_transitions,
                    "pre_evaluation": _evaluation_dict(pre),
                    "post_evaluation": _evaluation_dict(post),
                    "checkpoint": checkpoint_relative,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )

        parameters = config.canonical_parameters()
        summary: Dict[str, Any] = {
            "schema": SUMMARY_SCHEMA,
            "phase_label": PHASE_LABEL,
            "claim_scope": "ALGORITHM_MECHANICS_ONLY_NOT_SPLITFUSION_PERFORMANCE",
            "evaluation_protocol": "PAIRED_FIXED_ANALYTIC_STREAM_PER_SEED",
            "runtime_versions": provenance["runtime_versions"],
            "parameters": parameters,
            "aggregate": _aggregate(seed_results),
            "seeds": seed_results,
        }
        _write_new(stage / "summary.json", _canonical_json_bytes(summary))
        _write_new(stage / "learning_curve.csv", _csv_bytes(curve))
        _write_new(stage / "REPORT.md", _report_text(summary).encode("utf-8"))

        artifact_paths = checkpoint_paths + [
            "REPORT.md",
            "learning_curve.csv",
            "summary.json",
        ]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "phase_label": PHASE_LABEL,
            "claim_scope": "ALGORITHM_MECHANICS_ONLY_NOT_SPLITFUSION_PERFORMANCE",
            "parameters_sha256": _sha256_bytes(_canonical_json_bytes(parameters)),
            "torch_threads": config.torch_threads,
            "artifacts": _artifact_inventory(stage, artifact_paths),
            **provenance,
        }
        _write_new(stage / "manifest.json", _canonical_json_bytes(manifest))
        verify_artifact_directory(stage, verify_sources=True)
        # Recheck immediately before publication.  rename is atomic when the
        # staging and target directories share this parent/filesystem.
        if target.exists():
            raise RenderError(f"output path appeared during render: {target}")
        stage.rename(target)
        return verify_artifact_directory(target, verify_sources=True)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parse_seeds(value: str) -> Tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("at least three unique non-negative seeds are required")
    return seeds


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--evaluation-steps", type=int, default=240)
    parser.add_argument("--replay-capacity", type=int, default=8_192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-transitions", type=int, default=256)
    parser.add_argument("--collect-per-update", type=int, default=2)
    parser.add_argument("--episode-horizon", type=int, default=24)
    parser.add_argument("--minimum-improvement-over-random", type=float, default=0.05)
    parser.add_argument("--maximum-oracle-regret", type=float, default=0.35)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = QualificationRenderConfigV1(
        output=args.output,
        seeds=tuple(args.seeds),
        updates=args.updates,
        evaluation_interval=args.evaluation_interval,
        evaluation_steps=args.evaluation_steps,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_transitions=args.warmup_transitions,
        collect_per_update=args.collect_per_update,
        episode_horizon=args.episode_horizon,
        minimum_improvement_over_random=args.minimum_improvement_over_random,
        maximum_oracle_regret=args.maximum_oracle_regret,
        torch_threads=args.torch_threads,
    )
    manifest = render_algorithm_qualification(config)
    print(json.dumps({
        "phase_label": PHASE_LABEL,
        "manifest_schema": manifest["schema"],
        "output": str(config.output.resolve()),
        "status": "COMPLETE",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
