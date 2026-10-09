"""Paired full-GT CPU replay, preserving exact source/target/archive payloads."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from . import runtime

ROOT = Path(__file__).resolve().parents[2]
REPLAYER = ROOT / "experiments/splitfusion_live_parity_repair_v2/20261003_v2/gt_cost_audit/replay_source_cost.py"


def load_cases():
    spec = importlib.util.spec_from_file_location("_sealed_gt_source_replay", REPLAYER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = json.loads((module.CELL / "run5_live_avo_gt_summary.json").read_text())
    return module, [module.owned(record) for record in summary["records"]], summary["records"]


def measure(fn):
    wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
    value = fn()
    return value, {"wall_ms":(time.perf_counter_ns()-wall)/1e6,
                   "cpu_ms":(time.thread_time_ns()-cpu)/1e6}


def quantiles(values):
    return {name:float(np.quantile(values, q)) for name,q in (("p50",.5),("p95",.95),("max",1))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=4, choices=range(2,7))
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = runtime.source_inventory()
    module, cases, records = load_cases()
    if len(cases) != 16:
        raise ValueError("expected the16 retained complete GT frames")
    rows, scopes = [], []
    for cycle in range(args.rounds):
        for arm in (("baseline", "optimized") if cycle % 2 == 0 else ("optimized", "baseline")):
            scope = runtime.install_cpu_gt_optimizations(expected_sources=sources) if arm == "optimized" else nullcontext()
            with scope as state, tempfile.TemporaryDirectory(prefix="exact-gt-replay-") as scratch:
                producer, initialization = measure(lambda: module.producer(*cases[0][:2]))
                for (bundle, evidence, kwargs), record in zip(cases, records):
                    identity = kwargs["identity"]
                    pop = evidence["dynamic_snapshot_evidence"]
                    if producer.population != frozenset(pop["registered_population_ids"]):
                        producer.register_population(pop["registered_population_ids"], pop["population_sha256"],
                                                     before_frame_id=identity["frame_id"])
                    # Replay the saved diagnostic observation stamp only.
                    # Geometry and source evidence are otherwise unchanged;
                    # perf/thread clocks measure the real cost. Live code
                    # always uses its actual observation clock.
                    with patch("rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync.time.clock_gettime_ns",
                               return_value=evidence["source_synchronization"]["observed_raw_ns"]):
                        inputs, source_time = measure(lambda: producer.build(**kwargs))
                    if inputs.provenance != bundle.inputs.provenance:
                        raise AssertionError("source provenance/input content changed")
                    if G.canonical(producer.last_evidence) != G.canonical(evidence):
                        raise AssertionError("source evidence changed")
                    prepared, target_time = measure(lambda: E.prepare_target(inputs))
                    if prepared.content_sha256 != bundle.prepared.content_sha256:
                        raise AssertionError("prepared target content changed")
                    target, publish_time = measure(lambda: T.write_bundle(scratch, identity=identity, inputs=inputs, prepared=prepared))
                    for name, payload in bundle.manifest["payloads"].items():
                        data = (Path(target["directory"])/name).read_bytes()
                        if hashlib.sha256(data).hexdigest() != payload["sha256"]:
                            raise AssertionError(f"target payload changed: {name}")
                    auxiliary, archive_time = measure(lambda: G.write_source_archive(
                        Path(target["directory"]).with_suffix(".sources"), inputs=inputs,
                        evidence=producer.last_evidence, rgb=kwargs["rgb"], radar=kwargs["radar_world_xyz"]))
                    retained_sources = module.CELL/"parity_complete_targets"/Path(record["complete_sources"]["directory"]).name
                    for path in retained_sources.iterdir():
                        if path.read_bytes() != (Path(auxiliary["directory"])/path.name).read_bytes():
                            raise AssertionError(f"auxiliary archive changed: {path.name}")
                    timings = {"source":source_time, "target":target_time, "publish":publish_time, "archive":archive_time}
                    rows.append({"round":cycle, "arm":arm, "frame_id":identity["frame_id"],
                                 "timings":timings, "source_target_archive_exact":True,
                                 "input_content_sha256":inputs.provenance["input_content_sha256"],
                                 "prepared_content_sha256":prepared.content_sha256,
                                 "through_ready_ms":sum(timings[k]["wall_ms"] for k in ("source","target","publish")),
                                 "through_archive_ms":sum(t["wall_ms"] for t in timings.values())})
                scopes.append({"round":cycle, "arm":arm, "initialization":initialization, "scope":state})
            print(json.dumps({"round":cycle, "arm":arm, "frames":len(cases), "exact":True}), flush=True)
    result = {"status":"EXACT_OFFLINE_REPLAY_PASS", "rows":rows, "scopes":scopes,
              "rounds":args.rounds, "frames_per_arm":args.rounds*len(cases), "unique_frames":16,
              "source_inventory":sources, "replayer_sha256":hashlib.sha256(REPLAYER.read_bytes()).hexdigest(),
              "live_qualified":False, "live_route_started":False,
              "scope":"CPU replay of complete saved sources; not live latency or policy performance. Old AVO build/scoring, network, GT queue and edge are not included.",
              "sampling":"Alternating whole causal16-frame sequences; paired frame identities. Repeated scenes are not independent live observations."}
    result["observation_clock"] = "Saved diagnostic source-observation stamp supplied during offline source replay only, for exact evidence bytes; no production clock change."
    result["summary"] = {}
    for arm in ("baseline", "optimized"):
        selected = [r for r in rows if r["arm"]==arm]
        result["summary"][arm] = {k:quantiles([r[k] for r in selected]) for k in ("through_ready_ms","through_archive_ms")}
        result["summary"][arm]["stages"] = {k:quantiles([r["timings"][k]["wall_ms"] for r in selected])
                                            for k in ("source","target","publish","archive")}
    with (args.output/"RESULT.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    runtime.verify_sources(sources)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
