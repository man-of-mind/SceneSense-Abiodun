from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
CONFIG_PATH = PACKAGE / "recovery_config.json"


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, value: str, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def package_hashes() -> dict[str, str]:
    return {str(path.relative_to(PACKAGE)): sha256(path) for path in sorted(PACKAGE.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix != ".pyc"}


def current_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def load_recovery_config() -> Mapping[str, Any]:
    value = load_json(CONFIG_PATH)
    if value.get("state") != "UNQUALIFIED_IMPLEMENTATION_ONLY":
        raise RuntimeError("immutable implementation config state drift")
    return value


def resolve_repo_path(value: str) -> Path:
    path = (ROOT / value).resolve(strict=True)
    if ROOT not in path.parents and path != ROOT:
        raise RuntimeError(f"path escapes repository: {value}")
    return path


def verify_original_provenance(*, checkpoint_metadata: bool = False) -> dict[str, Any]:
    config = load_recovery_config()
    original = config["original"]
    original_config = resolve_repo_path(original["config"])
    checkpoint = resolve_repo_path(original["checkpoint"])
    source_package = resolve_repo_path(original["package"])
    registration = resolve_repo_path(original["registration"])
    registration_hashes = resolve_repo_path(original["registration_hashes"])
    experiment = resolve_repo_path(original["experiment"])
    if sha256(original_config) != original["config_sha256"]:
        raise RuntimeError("original config hash drift")
    if sha256(checkpoint) != original["checkpoint_sha256"]:
        raise RuntimeError("epoch-9 checkpoint hash drift")
    source_files = {name: sha256(source_package / name) for name in original["source_files_sha256"]}
    if source_files != original["source_files_sha256"] or canonical_hash(source_files) != original["source_canonical_sha256"]:
        raise RuntimeError("original source package hash drift")
    checkpoint_source_files = {**source_files, **original["checkpoint_source_file_overrides"]}
    if canonical_hash(checkpoint_source_files) != original["checkpoint_source_canonical_sha256"]:
        raise RuntimeError("declared epoch-9 checkpoint source manifest hash drift")
    artifact_checks = {
        str(registration): original["registration_sha256"],
        str(registration_hashes): original["registration_hashes_sha256"],
        str(experiment / "QUALIFIED_RUNTIME.json"): original["qualified_runtime_sha256"],
        str(experiment / "LOSS_CALIBRATION.json"): original["loss_calibration_sha256"],
    }
    for path_string, expected_digest in artifact_checks.items():
        if sha256(Path(path_string)) != expected_digest:
            raise RuntimeError(f"original registration/runtime artifact hash drift: {path_string}")
    for relative, expected_digest in original["evaluation_sources_sha256"].items():
        path = resolve_repo_path(relative)
        if sha256(path) != expected_digest:
            raise RuntimeError(f"frozen evaluation source hash drift: {path}")
    expected_base = original["registration_source_canonical_sha256"]
    allowed_scopes = ("diagnostic_runtime_only", "training_runtime_guard_only",
                      "evaluation_undefined_metric_runtime_only")
    for index, (name, expected_digest) in enumerate(original["source_amendments_sha256"].items()):
        path = experiment / name
        if sha256(path) != expected_digest:
            raise RuntimeError(f"source amendment hash drift: {path}")
        amendment = load_json(path)
        base = amendment.get("base_source_state_sha256", amendment.get("base_registration_source_state_sha256"))
        if (base != expected_base or amendment.get("scope") != allowed_scopes[index]
                or amendment.get("scientific_settings_changed") is not False):
            raise RuntimeError(f"source amendment chain drift: {path}")
        expected_base = amendment["amended_source_state_sha256"]
        if index == 1 and expected_base != original["checkpoint_source_canonical_sha256"]:
            raise RuntimeError("epoch-9 checkpoint source does not match amendment 002")
    if expected_base != original["source_canonical_sha256"]:
        raise RuntimeError("source amendment chain does not terminate at checkpoint source")
    report: dict[str, Any] = {"config": str(original_config), "checkpoint": str(checkpoint),
                              "config_sha256": sha256(original_config), "checkpoint_sha256": sha256(checkpoint),
                              "source_package": str(source_package),
                              "source_canonical_sha256": canonical_hash(source_files),
                              "checkpoint_source_canonical_sha256": canonical_hash(checkpoint_source_files),
                              "registration": str(registration), "registration_sha256": sha256(registration),
                              "registration_hashes_sha256": sha256(registration_hashes)}
    if checkpoint_metadata:
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "schema": original["checkpoint_schema"], "epoch": original["checkpoint_epoch"],
            "global_optimizer_update": original["checkpoint_global_optimizer_update"],
            "config_sha256": original["config_sha256"],
        }
        actual = {name: state.get(name) for name in expected}
        if actual != expected:
            raise RuntimeError(f"epoch-9 checkpoint metadata drift: {actual} != {expected}")
        for key in ("model", "optimizer", "scheduler", "rng", "sampler"):
            if key not in state:
                raise RuntimeError(f"epoch-9 recovery state missing {key}")
        sampler_expected = {"length": 16827, "seed": 20260829, "epoch": 9, "start_index": 0}
        scheduler_expected = {"kind": "absolute_epoch_multistep", "lrs": {"pretrained_backbone": 0.001,
            "pretrained_fpn_heads": 0.0025, "new": 0.01}, "milestones_after_epochs": [16, 22], "gamma": 0.1}
        group_names = [group.get("name") for group in state["optimizer"].get("param_groups", [])]
        if state["sampler"] != sampler_expected or state["scheduler"] != scheduler_expected:
            raise RuntimeError("epoch-9 sampler/scheduler recovery state drift")
        if group_names != ["pretrained_backbone", "pretrained_fpn_heads", "new"] or len(state["optimizer"]["state"]) != 158:
            raise RuntimeError("epoch-9 optimizer recovery state drift")
        if set(state["rng"]) != {"python", "numpy", "torch", "cuda"} or state.get("validation_accessed") is not False:
            raise RuntimeError("epoch-9 RNG/validation recovery state drift")
        if (state.get("source_hashes") != checkpoint_source_files
                or state.get("registration_hashes") != load_json(registration_hashes)):
            raise RuntimeError("epoch-9 source/registration binding drift")
        report.update({"metadata": actual, "recovery_keys": ["model", "optimizer", "scheduler", "rng", "sampler"],
                       "sampler": sampler_expected, "scheduler": scheduler_expected,
                       "optimizer_groups": group_names, "optimizer_state_entries": 158,
                       "rng_keys": sorted(state["rng"]), "validation_accessed": False})
    return report


def require_qualified(qualification_dir: Path, authorization: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    root = Path(qualification_dir).resolve(strict=True)
    marker = root / "QUALIFIED_TO_TRAIN"
    qualified_path = root / "QUALIFIED_RECOVERY_CONFIG.json"
    qualification_path = root / "RECOVERY_QUALIFICATION.json"
    review_path = root / "INDEPENDENT_SOURCE_REVIEW.json"
    qualification_review_path = root / "INDEPENDENT_QUALIFICATION_REVIEW.json"
    for path in (marker, qualified_path, qualification_path, review_path, qualification_review_path, Path(authorization)):
        if not path.is_file():
            raise RuntimeError(f"required scientific gate absent: {path}")
    qualified = load_json(qualified_path)
    qualification = load_json(qualification_path)
    review = load_json(review_path)
    qualification_review = load_json(qualification_review_path)
    auth = load_json(Path(authorization))
    hashes = package_hashes()
    package_sha = canonical_hash(hashes)
    commit = current_commit()
    checks = (
        qualified.get("state") == "QUALIFIED_TO_TRAIN",
        qualification.get("pass") is True,
        qualification.get("range", {}).get("aggregate_required_gradient_reachability", {}).get(
            "all_required_trainable_groups_observed_nonzero") is True,
        qualification.get("range", {}).get("aggregate_required_gradient_reachability", {}).get(
            "all_required_trainable_groups_finite_every_update") is True,
        review.get("approved") is True,
        qualification_review.get("approved") is True,
        auth.get("authorized") is True,
        qualified.get("source_commit") == commit,
        qualified.get("source_files_sha256") == package_sha,
        qualification.get("source_files_sha256") == package_sha,
        review.get("source_commit") == commit,
        qualification_review.get("source_commit") == commit,
        qualification_review.get("source_files_sha256") == package_sha,
        qualification_review.get("qualification_sha256") == sha256(qualification_path),
        qualification_review.get("qualified_config_sha256") == sha256(qualified_path),
        auth.get("source_commit") == commit,
        auth.get("qualification_sha256") == sha256(qualification_path),
        auth.get("qualified_config_sha256") == sha256(qualified_path),
        sha256(marker) == qualified.get("marker_sha256"),
    )
    if not all(checks):
        raise RuntimeError("qualification/review/authorization source binding failed")
    validate_selected_config(qualified)
    return qualified, qualification


def validate_selected_config(config: Mapping[str, Any]) -> None:
    immutable = load_recovery_config()
    tau = config.get("selected_tau")
    if tau is None or float(tau) not in tuple(float(x) for x in immutable["yaw"]["candidate_tau"]):
        raise RuntimeError("selected tau absent or outside preregistered candidate set")
    ceilings = config.get("ceilings")
    expected_groups = {"pretrained_backbone", "pretrained_fpn_heads", "new"}
    if (not isinstance(ceilings, Mapping)
            or set(ceilings) != {"gradient_norm", "momentum_norm", "proposed_sgd_update_norm",
                                "max_parameter_relative_update"}
            or set(ceilings.get("gradient_norm", {})) != expected_groups | {"global"}
            or set(ceilings.get("momentum_norm", {})) != expected_groups
            or set(ceilings.get("proposed_sgd_update_norm", {})) != expected_groups):
        raise RuntimeError("qualified pre-step ceilings absent")
    numerical_ceilings = [float(value) for family in ("gradient_norm", "momentum_norm",
                                                       "proposed_sgd_update_norm")
                          for value in ceilings[family].values()]
    numerical_ceilings.append(float(ceilings["max_parameter_relative_update"]))
    if not all(math.isfinite(value) and value >= 0.0 for value in numerical_ceilings):
        raise RuntimeError("qualified pre-step ceiling is null, nonfinite, or negative")
    if config.get("threshold_formula") != "10 * maximum_healthy_value":
        raise RuntimeError("qualification threshold formula drift")
    original = immutable["original"]
    expected = {
        "original_checkpoint_sha256": original["checkpoint_sha256"],
        "original_source_canonical_sha256": original["source_canonical_sha256"],
        "original_checkpoint_source_canonical_sha256": original["checkpoint_source_canonical_sha256"],
        "original_config_sha256": original["config_sha256"],
        "original_registration_sha256": original["registration_sha256"],
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise RuntimeError("qualified config original provenance drift")
