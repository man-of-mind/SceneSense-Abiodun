"""Profile current exact V2 GT on all 48 saved v6 policy frames, CPU only.

One ordinary timing pass and one distinct cProfile pass run the unchanged
five-stage saved-frame pipeline. Private module instances replace only the
reference CELL and scope module; sealed helpers and runtime sources stay intact.
No runtime candidate, service launch, GPU inference, or live timing claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import pstats
import sys
import tempfile
import time

import numpy as np
import torch

from rl_agent.splitfusion_live_gt_optimization_v2 import runtime as V2
from rl_agent.splitfusion_live_parity_runtime_v6 import contract as C6
from rl_agent.splitfusion_live_parity_repair_v1 import readiness as R

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELL = ROOT / "experiments/splitfusion_live_parity_runtime_v6/20261003_paired_quality_v6/qualification_100/capture/cells/a71__favorable_stable"
EXPECTED_FRAMES = 48
PROFILE_HELPER = "rl_agent/splitfusion_live_gt_optimization_v2/profile_saved.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symbolic input/output path")
    return path


def read(path, expected=None):
    return R._json(R._read(safe(path), expected))


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def file_sha(path):
    return C6.sha(safe(path))


def verify_environment():
    require(sys.executable == "/usr/bin/python3", "use the exact retained /usr/bin/python3 environment")
    require(not sys.flags.optimize, "exact retained replay requires assertions enabled")
    require(all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "set all three numerical thread caps to 1 before Python startup")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only profiler requires CUDA_VISIBLE_DEVICES empty")
    require(not torch.cuda.is_initialized(), "CUDA initialized before CPU replay")
    return dict(python=sys.executable, numpy=np.__version__, torch=torch.__version__, **V2.settings())


def private_profile(cell, expected_sources):
    """New module object; never replace globals on the sealed loaded module."""
    path = ROOT / PROFILE_HELPER
    require(file_sha(path) == expected_sources["profile_saved.py"], "sealed profile helper drift")
    spec = importlib.util.spec_from_file_location("_gt_v3_private_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CELL = cell
    module.V1 = V2  # the sealed run_pass now scopes the exact current V2 stack
    return module


def input_binding(cell):
    qualification = cell.parents[2]
    report = read(qualification / "RUN5_POSTFLIGHT.json")
    require(report.get("passed") is True and report.get("errors") == [], "successful retained postflight required")
    manifest = read(qualification / "RUN5_POSTFLIGHT_INPUTS.json", report["input_manifest_sha256"])
    files = manifest["files"]
    registration = read(qualification / "registration.json", files["registration.json"])
    require(registration.get("optimization_sources") == C6.OPTIMIZATION_SOURCES == V2.source_inventory(),
            "current exact V2 differs from pre-capture registration")
    relative = str(cell.relative_to(qualification)) + "/"
    prefixes = tuple(relative + name + "/" for name in (
        "parity_complete_targets", "parity_static_catalogue", "run5_live_avo_inputs"))
    names = {relative + "run5_live_avo_gt_summary.json", relative + "PAIRED_INPUT_SCOPE.json"}
    selected = {name: digest for name, digest in files.items() if name in names or name.startswith(prefixes)}
    require(names <= set(selected), "missing complete saved GT evidence")
    for name, digest in selected.items():
        require(file_sha(qualification / name) == digest, "retained input hash drift: " + name)
    for directory in ("parity_complete_targets", "parity_static_catalogue", "run5_live_avo_inputs"):
        actual = {str(p.relative_to(qualification)) for p in (cell / directory).rglob("*") if p.is_file()}
        expected = {name for name in selected if name.startswith(relative + directory + "/")}
        require(actual == expected, "retained GT archive inventory drift")
    return qualification, registration["optimization_sources"], selected, report["input_manifest_sha256"]


def load_cases(profile, cell):
    reference = profile.R.load_reference()  # source SHA checked by the sealed loader
    reference.CELL = cell
    summary = read(cell / "run5_live_avo_gt_summary.json")
    records = summary["records"]
    require(len(records) == EXPECTED_FRAMES and len({r["frame_id"] for r in records}) == EXPECTED_FRAMES,
            "all 48 distinct saved v6 policy frames required")
    scope = read(cell / "PAIRED_INPUT_SCOPE.json")
    require(scope.get("complete") is True and scope.get("policy_opportunities") == EXPECTED_FRAMES,
            "saved GT population differs from exact-input prepared cohort")
    cases = []
    for record in records:
        bundle, evidence, kwargs = reference.owned(record)
        require(bundle.native_inputs_verified, "full target input verification was omitted")
        avo = read(cell / "run5_live_avo_inputs" / Path(record["path"]).name, record["sha256"])
        people = {int(row["gt_actor_id"]): row for row in avo["raw_pedestrian_rows"]}
        for actor in kwargs["frozen_world"].get_actors():
            if actor.id in people:
                row, rotation = people[actor.id], actor.get_transform().rotation
                rotation.yaw, rotation.pitch, rotation.roll = (row["object_" + k + "_deg"] for k in ("yaw", "pitch", "roll"))
        cases.append((bundle, evidence, kwargs, record, avo))
    return reference, cases


def run(*, cell, output):
    cell, output = safe(cell), safe(output)
    require(not output.exists(), "profiling output is create-only")
    settings = verify_environment()
    qualification, sources, hashes, outer_sha = input_binding(cell)
    profile = private_profile(cell, sources)
    reference, cases = load_cases(profile, cell)
    script_hashes = {name: file_sha(Path(__file__).parent / name) for name in ("__init__.py", "profile_saved.py")}
    reference_hashes = {str(profile.R.REPLAY.relative_to(ROOT)): file_sha(profile.R.REPLAY),
                       PROFILE_HELPER: file_sha(ROOT / PROFILE_HELPER)}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "DESIGN.json", dict(schema="splitfusion.gt_v3.saved_profile_design.v1", implementation="EXACT_RETAINED_V2",
        cell=str(cell), expected_unique_frames=EXPECTED_FRAMES, passes=["unprofiled", "function_profile"],
        stages=list(profile.STAGES), settings=settings, source_inventory_v2=sources,
        profile_sources=script_hashes, sealed_helpers=reference_hashes,
        parent_postflight_input_manifest_sha256=outer_sha, no_candidates=True, no_live_or_gpu=True,
        supplied_historical_clock="SOURCE_EVIDENCE_OBSERVATION_ONLY; no live clock modification"))
    start = time.monotonic()
    summaries = {}
    try:
        for label, profiled in (("unprofiled", False), ("function_profile", True)):
            with tempfile.TemporaryDirectory(prefix="exact-gt-v3-profile-") as temporary:
                rows, profiles = profile.run_pass(reference, cases, Path(temporary), profiled=profiled)
            require(len(rows) == EXPECTED_FRAMES and all(r["exact"] for r in rows), "incomplete exact replay pass")
            write(output / (label + ".json"), rows)
            summaries[label] = dict(frames=len(rows), all_exact=True,
                stage_cost_ms={stage: {key: profile.quantiles([r["timings"][stage][key] for r in rows])
                    for key in ("wall_ms", "thread_cpu_ms")} for stage in profile.STAGES},
                through_ready_ms=profile.quantiles([r["through_ready_ms"] for r in rows]),
                through_archive_ms=profile.quantiles([r["through_archive_ms"] for r in rows]))
            for stage, function_profile in profiles.items():
                write(output / (stage + "_functions.json"), profile.stats(function_profile))
                stream = io.StringIO()
                pstats.Stats(function_profile, stream=stream).sort_stats("cumulative").print_stats(80)
                with (output / (stage + "_profile.txt")).open("x") as target:
                    target.write(stream.getvalue())
            print(json.dumps(dict(pass_name=label, frames=len(rows), all_exact=True,
                                  elapsed_seconds=time.monotonic()-start)), flush=True)
        for name, digest in hashes.items():
            require(file_sha(qualification / name) == digest, "saved GT evidence changed during profiling")
        for name, digest in reference_hashes.items():
            require(file_sha(ROOT / name) == digest, "sealed helper changed during profiling")
        for name, digest in script_hashes.items():
            require(file_sha(Path(__file__).parent / name) == digest, "profiler source changed during profiling")
        V2.verify_sources(sources)
        require(verify_environment() == settings, "retained process settings changed during profiling")
        result = dict(schema="splitfusion.gt_v3.saved_profile_result.v1", status="ALL48_EXACT_CURRENT_V2_CPU_PROFILE_COMPLETE",
            unique_frames=EXPECTED_FRAMES, passes=2, total_frame_replays=2 * EXPECTED_FRAMES,
            elapsed_seconds=time.monotonic()-start, live_started=False, cuda_initialized=False,
            candidate_implemented=False, saved_input_sources_unchanged=True, source_inventory_v2=sources,
            profile_sources=script_hashes, sealed_helpers=reference_hashes, settings=settings, summaries=summaries,
            equality_checks=["AVO and raw pedestrian rows", "complete input provenance/content SHA",
                "complete source canonical evidence", "prepared target content SHA and PNG bytes",
                "all published native/target payload hashes", "complete source archive byte tree"],
            limits=["Offline CPU timings are not live deadline performance.",
                "No new optimization was evaluated; this profiles the current exact V2 implementation.",
                "The base-object collector, queue, network and edge are outside the saved-frame replay.",
                "cProfile instrumentation changes timings; use only the separate unprofiled pass for ordinary costs.",
                "Fresh READY/AVO timestamps differ; numerical results and all target/source payload bytes match."])
        write(output / "RESULT.json", result)
        write(output / "INPUT_SHA256.json", hashes)
        write(output / "OUTPUT_SHA256.json", profile.R.tree(output))
    except BaseException as exc:
        write(output / "FAILURE.json", dict(error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic()-start, completed_passes=list(summaries), cuda_initialized=torch.cuda.is_initialized()))
        raise
    print(json.dumps(result["summaries"]["unprofiled"], indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, default=DEFAULT_CELL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(cell=args.cell, output=args.output)


if __name__ == "__main__":
    main()
