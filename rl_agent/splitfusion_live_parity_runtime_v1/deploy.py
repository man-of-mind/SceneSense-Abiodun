"""Plan or create-only stage the additive repair runtime; never launch services.

All candidate files and existing dependencies are checked before copying any
bytes. Existing identical files are retained. The old Run5 runtime and the
cross-worktree CPU-partition helpers are never deployment targets.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "splitfusion.live_parity_runtime.deployment_plan.v1"
PACKAGES = (
    "rl_agent/splitfusion_live_parity_repair_v1",
    "rl_agent/splitfusion_live_target_parity_v1",
    "rl_agent/splitfusion_live_parity_runtime_v1",
)
RUNTIME_PACKAGE = PACKAGES[-1]
PROFILER_PACKAGE = "rl_agent/splitfusion_single_gpu_profiler_v1"
OLD_RUNTIME = "rl_agent/splitfusion_run5_live_avo_v1"
HISTORICAL_SCRIPTS = (
    "experiments/route_b_v3_vehicle_gt_reconciliation_audit_v1/20260828_000427/scripts/run_vehicle_gt_reconciliation_audit_v1.py",
    "experiments/route_b_v3_static_vehicle_gt_audit_v1/20260827_225706/scripts/run_static_vehicle_audit_v1.py",
)
PIN_DECLARATIONS = (
    ("rl_agent/splitfusion_live_target_parity_v1/adapter.py", "SOURCES"),
    ("rl_agent/splitfusion_live_parity_repair_v1/source_geometry.py", "SOURCES"),
    ("rl_agent/splitfusion_live_parity_repair_v1/perception.py", None),
    ("rl_agent/splitfusion_live_parity_repair_v1/evaluation.py", None),
)

# Finite, explicitly registered CPU fixtures. Native input views in the source
# repository contain directory symlinks; their six PNG source paths below name
# the real in-repository data files. Destinations are ordinary copied files.
# Whole CSVs retain the hashes used by the frozen tests. No corpus directory is
# enumerated and no held/test partition is read.
_FROZEN = "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455/views/val/"
_FINAL = "experiments/route_b_v3_1_clean_base_v1/20260828_012309/contracts/v010/val/"
_STATIC = "experiments/route_b_v3_static_vehicle_gt_audit_v1/20260827_225706/"
_RECON = "experiments/route_b_v3_vehicle_gt_reconciliation_audit_v1/20260828_000427/"
_SID05 = "canonical_v3_05_val_30_30_s601_tm1601_000027_frame357"
_SID06 = "canonical_v3_06_val_50_50_s602_tm1602_001177_frame4935"
_EP05 = "canonical_v3_05_val_30_30_s601_tm1601/"
_EP06 = "canonical_v3_06_val_50_50_s602_tm1602/"
CPU_FIXTURE_FILES = {
    _FROZEN + "manifest.csv": (8262438, "c23220cca825f44416d52ea29771a3b00c2989391cb5013fa0af99e7d09ced27"),
    _FROZEN + "object_boxes_all.csv": (39652979, "cdaae52c400894c9790485ef7196d82cd879a74612f0d970d97575c9ec5da010"),
    _FROZEN + "object_visibility_all.csv": (36904993, "ce8b2c6cbed29ba3827def876c98d0c60b401993d247e1bff38834754d98dca4"),
    _FROZEN + _EP05 + "depth/" + _SID05 + ".png": (1085137, "71c205a4d37e173f1ce1e8c4ab118b345e807694db6e507106787e4afc7a60ad"),
    _FROZEN + _EP05 + "masks/" + _SID05 + ".png": (3087, "7d4be24be5b4af239f8b73bafc0767921fea79b39f2ecc69d553e152674be52b"),
    _FROZEN + _EP05 + "semantic_tags/" + _SID05 + ".png": (57621, "8b61cb928b49c7f8a3c531a03beff828f4b9923a330fdcd474f238d6171dd117"),
    _FROZEN + _EP06 + "depth/" + _SID06 + ".png": (1021027, "3b6c7793748f2956ef9b63349b0edaa7038b1a36b453cc8e9245d120826abd44"),
    _FROZEN + _EP06 + "masks/" + _SID06 + ".png": (5249, "c200fc9a67d01f82959135c2b554a71a68542335f128a1c897d997f201dcf5b5"),
    _FROZEN + _EP06 + "semantic_tags/" + _SID06 + ".png": (48729, "40096667e2b125b27b487bc8bb9e6dac3ed7193288df266067b9e31373c8b1af"),
    _FINAL + "object_boxes.csv": (10913272, "347c908edf92b2723a40ef78c6f6cd30915600df64206675f4dd567b11c231f2"),
    _FINAL + "object_ignore_regions.csv": (17260854, "a48faf9192e0633e37db0120133aaaeeef353c278126c499188f31b2004c3530"),
    _FINAL + "segmentation_masks/" + _SID05 + ".png": (1561, "9dfb833f343adb9ca7810509e78384770f56e1f8162bdbc337e6c6c465ee2480"),
    _FINAL + "segmentation_masks/" + _SID06 + ".png": (2619, "cdc07d4e727e65157683e92ca1d97a5b7a971ae6de165cedd5f9b95eb766344e"),
    _FINAL + "object_ignore_masks/" + _SID05 + ".png": (1413, "d7fddd8bf3de5732969cdbb5ef3d25a06c434fde9329d979dca45d3ccd6cee2d"),
    _FINAL + "object_ignore_masks/" + _SID06 + ".png": (2254, "e1f2291c0eab6daad23b7dc8886b4bbbc70fe1ed556afc8a5c871a717ad590e2"),
    _STATIC + "static_vehicle_visibility.csv": (8391009, "286735f21cccbd93afd49bf6b8e19597728e4c6dd7564978d52f46af47a3bfbc"),
    _STATIC + "static_environment_truth/catalog/static_environment_objects.csv": (71156, "26ee85ea878204bc93e25e9e3439af47bfc863458d4956e626b3918f859f3c8f"),
    _RECON + "manual_review_selection.csv": (14692, "2426ba8a595a5a57c8f9a7b3566df000a924f7c95d69bde51be8aa3d1bfd4322"),
    _RECON + "static_dynamic_conflicts.csv": (2584101, "70d8aaaaffb55403f08448baec141e9b7e7c0723111885d441dbc4ef86514b9e"),
    _RECON + "static_static_conflicts.csv": (41097, "8a50b29416903079d0788642f423dde13e3f01689928806b480e54ba880d6612"),
    "experiments/splitfusion_fcos_service_candidate_v1/predictions/detections.csv": (60328031, "a682a1fc5eabb2e59e07449a8c6b5fc604077b40ef094b57dc30c5a18d7ec260"),
    "experiments/actor_volume_observability_model_comparison_v1/20260901_repaired_tolerance_cpu_once/actor_volume_observability_table.csv": (4504110, "abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386"),
}
CPU_FIXTURE_SOURCE_OVERRIDES = {
    _FROZEN + episode + kind + "/" + sample + ".png":
        "data_collection/experiments/route_b_perception_v3/" + episode + kind + "/" + sample + ".png"
    for episode, sample in ((_EP05, _SID05), (_EP06, _SID06))
    for kind in ("depth", "masks", "semantic_tags")
}
CPU_FIXTURE_BYTE_LIMIT = 200_000_000


class DeploymentError(ValueError):
    pass


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _safe_path(value, *, directory=False):
    """Reject symbolic components, including ones hidden before a '..'."""
    path = Path(value).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current.parent if part == ".." else current / part
        if current.is_symlink():
            raise DeploymentError(f"symbolic path component refused: {current}")
        if current.exists() and current != path and not current.is_dir():
            raise DeploymentError(f"nondirectory path component: {current}")
    path = Path(os.path.abspath(path))
    if directory and not path.is_dir():
        raise DeploymentError(f"required directory missing: {path}")
    return path


def _read_regular(path):
    path = _safe_path(path)
    if not path.is_file():
        raise DeploymentError(f"regular source/dependency missing: {path}")
    return path.read_bytes()


def _package_files(root, relative, *, require_init=True):
    directory = _safe_path(root / relative, directory=True)
    if require_init:
        _read_regular(directory / "__init__.py")
    result = []
    pending = [directory]
    while pending:
        for path in sorted(pending.pop().iterdir()):
            if path.is_symlink():
                raise DeploymentError(f"symbolic package entry refused: {path}")
            if path.name == "__pycache__" or path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                result.append(path.relative_to(root).as_posix())
            else:
                raise DeploymentError(f"nonregular package entry refused: {path}")
    return sorted(result)


def _literal_assignments(path):
    values = {}
    for node in ast.parse(_read_regular(path), filename=str(path)).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values


def _source_pins(root):
    result = {}
    for relative, collection in PIN_DECLARATIONS:
        values = _literal_assignments(root / relative)
        if collection is not None:
            pairs = list(values[collection].values())
        else:
            pairs = []
            for name, value in values.items():
                for suffix in ("_PATH", "_SOURCE"):
                    if name.endswith(suffix):
                        digest = values.get(name[:-len(suffix)] + "_SHA256")
                        if digest is not None:
                            pairs.append((value, digest))
        if not pairs:
            raise DeploymentError(f"source pin declaration empty: {relative}")
        for source, digest in pairs:
            if (not isinstance(source, str) or Path(source).is_absolute()
                    or ".." in Path(source).parts or not isinstance(digest, str)
                    or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise DeploymentError(f"invalid source pin: {relative}")
            if source in result and result[source] != digest:
                raise DeploymentError(f"conflicting source pins: {source}")
            if _sha(_read_regular(root / source)) != digest:
                raise DeploymentError(f"source pin changed: {source}")
            result[source] = digest
    return result


def _old_runtime_files(root):
    directory = _safe_path(root / OLD_RUNTIME, directory=True)
    result = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise DeploymentError(f"symbolic old-runtime entry refused: {path}")
        if path.suffix == ".py" and not path.name.startswith("test_"):
            result.append(path.relative_to(root).as_posix())
    export = _safe_path(directory / "actor_export", directory=True)
    for path in sorted(export.iterdir()):
        _read_regular(path)
        result.append(path.relative_to(root).as_posix())
    if not result or not any("/actor_export/" in name for name in result):
        raise DeploymentError("old Run5 runtime/actor export inventory empty")
    return result


def test_fixture_inventory(source_root=ROOT):
    """Check the exact 22 finite CPU inputs; never follow source-view aliases."""
    root = _safe_path(source_root, directory=True)
    if sum(size for size, _ in CPU_FIXTURE_FILES.values()) > CPU_FIXTURE_BYTE_LIMIT:
        raise DeploymentError("declared CPU fixture byte budget exceeded")
    files = {}
    for relative, (size, digest) in sorted(CPU_FIXTURE_FILES.items()):
        source_relative = CPU_FIXTURE_SOURCE_OVERRIDES.get(relative, relative)
        for name in (relative, source_relative):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise DeploymentError("unsafe registered CPU fixture path")
        path = _safe_path(root / source_relative)
        if not path.is_file() or path.stat().st_size != size:
            raise DeploymentError(f"CPU fixture absent or size changed: {source_relative}")
        if _sha(_read_regular(path)) != digest:
            raise DeploymentError(f"CPU fixture SHA-256 changed: {source_relative}")
        files[relative] = {"source_relative": source_relative, "bytes": size, "sha256": digest}
    body = {"files": files, "count": len(files),
            "total_bytes": sum(row["bytes"] for row in files.values()),
            "maximum_bytes": CPU_FIXTURE_BYTE_LIMIT}
    return {**body, "inventory_sha256": _sha(_canonical(body))}


def deployment_plan(worktree, *, source_root=ROOT, include_test_fixtures=False):
    """Return a sealed read-only inventory after full source/target validation."""
    source_root = _safe_path(source_root, directory=True)
    worktree = _safe_path(worktree, directory=True)
    if source_root == worktree:
        raise DeploymentError("deployment destination must differ from source root")
    _safe_path(worktree / PROFILER_PACKAGE, directory=True)
    pins = _source_pins(source_root)
    items = {}
    for package in PACKAGES:
        for relative in _package_files(source_root, package):
            items[relative] = relative
    for relative in HISTORICAL_SCRIPTS:
        items[relative] = relative
    fixtures = test_fixture_inventory(source_root) if include_test_fixtures else None
    if fixtures is not None:
        items.update({name: row["source_relative"] for name, row in fixtures["files"].items()})
    shims = source_root / RUNTIME_PACKAGE / "runtime_shims"
    _safe_path(shims, directory=True)
    shim_names = sorted(p.name for p in shims.iterdir() if p.suffix == ".py")
    if (len(shim_names) != 2
            or sum(name.startswith("child_v1_") for name in shim_names) != 1
            or sum(name.startswith("edge_profiled_v1_") for name in shim_names) != 1):
        raise DeploymentError("exactly one sampler-visible child and edge shim required")
    for name in shim_names:
        items[f"{PROFILER_PACKAGE}/{name}"] = f"{RUNTIME_PACKAGE}/runtime_shims/{name}"
    for package in PACKAGES:
        target_package = _safe_path(worktree / package)
        if target_package.exists():
            extra = set(_package_files(worktree, package, require_init=False)) - set(items)
            if extra:
                raise DeploymentError(f"foreign destination package inventory: {sorted(extra)}")
    files = {}
    for target_relative, source_relative in sorted(items.items()):
        payload = _read_regular(source_root / source_relative)
        target = _safe_path(worktree / target_relative)
        action = "CREATE"
        if target.exists():
            if _read_regular(target) != payload:
                raise DeploymentError(f"existing destination has foreign bytes: {target_relative}")
            action = "KEEP_IDENTICAL"
        files[target_relative] = {"source_relative": source_relative,
            "sha256": _sha(payload), "bytes": len(payload), "action": action}
    required = dict(pins)
    required.update({relative: _sha(_read_regular(source_root / relative))
                     for relative in _old_runtime_files(source_root)})
    existing = {}
    for relative, digest in sorted(required.items()):
        if relative in files:
            if files[relative]["sha256"] != digest:
                raise DeploymentError(f"copy inventory disagrees with source pin: {relative}")
            continue
        if _sha(_read_regular(worktree / relative)) != digest:
            raise DeploymentError(f"existing runtime dependency differs: {relative}")
        existing[relative] = digest
    body = {"schema": SCHEMA, "source_root": str(source_root), "worktree": str(worktree),
        "files": files, "verified_existing": existing, "source_pins": pins,
        "test_fixtures": fixtures,
        "create_count": sum(row["action"] == "CREATE" for row in files.values()),
        "keep_identical_count": sum(row["action"] == "KEEP_IDENTICAL" for row in files.values()),
        "old_runtime_modified": False, "retained_partition_copied": False,
        "retained_partition_policy": "REUSE_ORIGINAL_LOCATION_WITH_REGISTERED_CROSS_ROOT_DEPENDENCIES",
        "live_launch_performed": False}
    return {**body, "inventory_sha256": _sha(_canonical(body))}


def deploy(worktree, *, source_root=ROOT, plan_only=True, include_test_fixtures=False):
    """Default to a read-only plan; explicit execution only adds missing files."""
    plan = deployment_plan(worktree, source_root=source_root, include_test_fixtures=include_test_fixtures)
    if plan_only:
        return {"status": "PLANNED_NOT_STAGED", "plan": plan}
    root, destination = Path(plan["source_root"]), Path(plan["worktree"])
    # Own every source payload before the first write; reject concurrent drift.
    payloads = {name: _read_regular(root / row["source_relative"])
                for name, row in plan["files"].items()}
    for name, row in plan["files"].items():
        if _sha(payloads[name]) != row["sha256"]:
            raise DeploymentError(f"deployment source changed after planning: {name}")
    # Recheck the complete destination, including existing dependencies, before
    # any copy. The same plan seal means neither inventory nor status changed.
    if deployment_plan(destination, source_root=root, include_test_fixtures=include_test_fixtures) != plan:
        raise DeploymentError("deployment inputs changed after planning")
    created = []
    for name, row in plan["files"].items():
        if row["action"] == "KEEP_IDENTICAL":
            continue
        target = _safe_path(destination / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        _safe_path(target)
        with target.open("xb") as handle:
            handle.write(payloads[name])
            handle.flush()
            os.fsync(handle.fileno())
        if _sha(_read_regular(target)) != row["sha256"]:
            raise DeploymentError(f"deployed content differs: {name}")
        created.append(name)
    verified = deployment_plan(destination, source_root=root, include_test_fixtures=include_test_fixtures)
    if verified["create_count"]:
        raise DeploymentError("deployment left missing files")
    return {"status": "STAGED_ADDITIVELY_NOT_LIVE_QUALIFIED", "plan": plan,
            "created": created, "verification": verified}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true", help="create missing files after all checks")
    parser.add_argument("--report", type=Path, help="create-only durable deployment record")
    parser.add_argument("--include-test-fixtures", action="store_true",
                        help="include the exact hash-pinned 22 CPU fixture files (191,157,429 bytes)")
    args = parser.parse_args(argv)
    if args.report is not None and args.report.exists():
        raise FileExistsError(args.report)
    result = deploy(args.worktree, source_root=args.source_root,
                    plan_only=not args.execute, include_test_fixtures=args.include_test_fixtures)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "report": str(args.report),
                          "created": len(result.get("created", [])),
                          "plan_sha256": result["plan"]["inventory_sha256"]}))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
