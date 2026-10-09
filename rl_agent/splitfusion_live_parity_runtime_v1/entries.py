"""Real sampler-visible entry points, with explicit evidence ownership."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
from . import contract as C


def write_x(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def evidence_scope(attempt):
    """Keep new bundles beside, not inside, the old exact AVO manifest."""
    from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
    from rl_agent.splitfusion_single_gpu_profiler_v1 import seams_ue_v1 as SU
    original, copier = G.preserve_complete_targets, SU.copy_edge_profile
    attempt = Path(attempt)

    def preserve(records, destination):
        expected = attempt / "run5_live_avo_inputs" / "complete_targets"
        if Path(destination) != expected:
            raise ValueError("unexpected complete-target preservation destination")
        return original(records, attempt / "parity_complete_targets")

    def copy(edge_scratch, destination):
        result = copier(edge_scratch, destination)
        if edge_scratch is None:
            return {**result, "pose_archive_preserved": False}
        source = Path(edge_scratch) / "profiler" / "pose_archive"
        if not source.exists():
            return {**result, "pose_archive_preserved": False}
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("symlinked pose archive")
        files = {str(p.relative_to(source)): C.sha(p) for p in source.rglob("*") if p.is_file()}
        if "MANIFEST.json" not in files:
            raise ValueError("edge pose archive did not finish")
        target = Path(destination) / "pose_archive"
        shutil.copytree(source, target)
        if any(C.sha(target / name) != digest for name, digest in files.items()):
            raise ValueError("edge pose copy mismatch")
        return {**result, "pose_archive_preserved": True, "pose_files": files}

    G.preserve_complete_targets, SU.copy_edge_profile = preserve, copy
    try:
        yield
    finally:
        G.preserve_complete_targets, SU.copy_edge_profile = original, copier


@contextmanager
def edge_close_evidence(profile_directory):
    """Preserve the real close/drain result the old edge otherwise discards."""
    from rl_agent.splitfusion_live_parity_repair_v1 import runtime
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    original = runtime.install_edge_evaluation

    @contextmanager
    def install(**kwargs):
        with original(**kwargs) as state:
            cls, old = E.Run4EvaluatorV2, E.Run4EvaluatorV2.close
            def close(self, timeout_s=None):
                result = old(self, timeout_s)
                write_x(Path(profile_directory) / "evaluator_close_result.json", result)
                return result
            cls.close = close
            try:
                yield state
            finally:
                cls.close = old
    runtime.install_edge_evaluation = install
    try:
        yield
    finally:
        runtime.install_edge_evaluation = original


def child(argv=None):
    from rl_agent.splitfusion_live_parity_repair_v1 import composition
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from . import pose
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = BASE.attempt_dir_from_argv(args)
    config = BASE.load_config(attempt)
    if config is None or config.get("qualification_id") != C.ID:
        raise ValueError("enabled, sealed parity profiler configuration required")
    if C.canonical({k: v for k, v in config.items() if k != "sha256"}) != config.get("sha256"):
        raise ValueError("profiler configuration digest mismatch")
    capture = Path(attempt).resolve().parent.parent
    from .host_capture import verify_prospective
    record = verify_prospective(capture.parent)
    if record["budget"] != C.BUDGET:
        raise ValueError("child archive bound differs from registration")
    with pose.install_ue_pose(attempt_dir=attempt, max_frames=C.BUDGET), evidence_scope(attempt):
        return composition.run_ue_child(args, max_frames=C.BUDGET, edge_module=C.EDGE)


def edge(argv=None):
    from rl_agent.splitfusion_live_parity_repair_v1 import composition
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from . import pose
    args = list(sys.argv[1:] if argv is None else argv)
    profile = EP.profile_path(args).parent
    # The partition mount is the sealed host attempt. Do not resolve its host
    # absolute source paths inside the container; parent verified those first.
    registration = json.loads(Path("/partition_output/registration.json").read_text())
    if (registration.get("id") != C.ID or registration.get("transmitted_budget") != C.BUDGET
            or registration.get("live_metric_contract") != C.metrics()
            or registration.get("registration_sha256") != C.canonical(
                {k: v for k, v in registration.items() if k != "registration_sha256"})):
        raise ValueError("edge registration differs from repaired runtime")
    with pose.install_edge_pose(archive_directory=profile/"pose_archive", max_frames=C.BUDGET), \
            edge_close_evidence(profile):
        return composition.run_edge(args, max_frames=C.BUDGET)
