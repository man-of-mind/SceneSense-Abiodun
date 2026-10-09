"""Pilot latency binding: measured v2 + v3 fixed-action chains, their runtime compatibility, and the held-out split.

Sources (FAVORABLE_STABLE, cell a71, one session each, cold decisions excluded by policy):
  screen v2 (2026-10-05 22:07-22:20):  a33 m5/q7000, a35 m5/q9800, a50 m8/q5000, a53 m8/q9800, a67 m11/q3000
  frontier v3 (2026-10-05 23:30-23:39): a67 m11/q3000 (repeat), a69 m11/q7000, off-anchor m11/q8000
Compatibility (checked, not assumed): the exact runtime sources outside the two forced-screen packages (which only
differ in their registration check), actor weights, deadline, profile, CPU partition, exact-GT limits and the edge
image must be identical for every source arm; otherwise the build refuses. Each chain keeps its source id (run
identity), so the two action-67 sessions stay distinguishable and both contribute to the q3000 stratum.

Held-out split (registered here, before any training): per source, decisions in time order are cut into blocks of
3; block index % 3 == 2 -> EVAL, else TRAIN. The same split applies to the chains and to the measured channel trace.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rl_agent.splitfusion_corrected_latency_env_v1 import CANDIDATE_VERSION

from .latency import REGISTERED_REJECTIONS

ROOT = Path(__file__).resolve().parents[2]                    # abiodun/
NEU = ROOT.parent
WT = NEU / "abiodun_run4_single_gpu_profiler_v1"
V2_SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261005_fixed_action_screen_v2"
V3_SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261005_fixed_action_m11_frontier_v3"
V2_ROWS = ROOT / "experiments/splitfusion_fixed_action_screen_v2/results_combined_01/DECISIONS.jsonl"
V3_ROWS = ROOT / "experiments/splitfusion_fixed_action_screen_v3/results_01/DECISIONS.jsonl"
V2_SEAL = ROOT / "experiments/splitfusion_fixed_action_screen_v2/final_01/SEAL.json"
V3_SEAL = ROOT / "experiments/splitfusion_fixed_action_screen_v3/final_01/SEAL.json"
CELL = "capture/cells/a71__favorable_stable"
SOURCES = (  # source_id, campaign, arm directory, rows file, (mode, q)
    ("v2_a33_m5_q7000", "screen_v2", "arm0_a33", V2_ROWS, (5, 7000)),
    ("v2_a35_m5_q9800", "screen_v2", "arm6_a35", V2_ROWS, (5, 9800)),
    ("v2_a50_m8_q5000", "screen_v2", "arm3_a50", V2_ROWS, (8, 5000)),
    ("v2_a53_m8_q9800", "screen_v2", "arm5_a53", V2_ROWS, (8, 9800)),
    ("v2_a67_m11_q3000", "screen_v2", "arm4_a67", V2_ROWS, (11, 3000)),
    ("v3_a67_m11_q3000", "frontier_v3", "arm0_m11_q3000_a67", V3_ROWS, (11, 3000)),
    ("v3_a69_m11_q7000", "frontier_v3", "arm1_m11_q7000_a69", V3_ROWS, (11, 7000)),
    ("v3_offanchor_m11_q8000", "frontier_v3", "arm2_m11_q8000_offanchor", V3_ROWS, (11, 8000)),
)
FIELD_MAP = {"backlog_bytes": "pre_action_rlc_backlog_bytes", "camera_si": "camera_si", "cold": "cold",
             "family": "forced_action.family", "mode_id": "executed_action.mode_id",
             "network_profile": "network_profile_id", "payload_bytes": "payload_bytes", "prior_ul_mcs": "prior_ul_mcs",
             "q_e4": "executed_action.q_e4", "quantizer": "forced_action.quantizer", "radar_p40": "radar_p40"}
SCREEN_PACKAGES = ("rl_agent/splitfusion_fixed_action_screen_v2/", "rl_agent/splitfusion_fixed_action_screen_v3/")
BLOCK = 3


class BindingBuildError(ValueError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(NEU))


def arm_dir(campaign: str, arm: str) -> Path:
    return (V2_SCREEN if campaign == "screen_v2" else V3_SCREEN) / arm


def runtime_identity(directory: Path) -> dict[str, Any]:
    reg = json.loads((directory / "registration.json").read_text())
    sources = {p: h for p, h in reg["exact_gt_sources"].items() if not p.startswith(SCREEN_PACKAGES)}
    image = json.loads((directory / CELL / "edge_image_prelaunch.json").read_text()).get("id")
    return dict(runtime_pipeline_sources_sha256=canonical(sources), runtime_pipeline_source_count=len(sources),
                edge_image=image, cpu_partition=json.loads((directory / "arm.json").read_text()),
                exact_gt_limits=reg["exact_gt_limits"], deadline_ms_inclusive=reg["frozen_deadline_ms"],
                actor_weights_sha256=reg["policy"]["expected_weights_sha256"], network_profile=reg["network_profile"])


def rows_for(path: Path, arm: str) -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["decision_index"])


def split_of(position: int) -> str:
    return "EVAL" if (position // BLOCK) % 3 == 2 else "TRAIN"


def fallback_evidence() -> dict[str, Any]:
    """Causally ordered fallback evidence from each source's frame plan log (FORCED_ACTIONS.jsonl).

    Start-up fallbacks precede the first policy decision and are attributed to no action. After that, every
    opportunity is either a policy decision (with its hold frame) or a single fallback frame; a fallback at
    opportunity t+1 can depend only on actions executed up to t, so it is attributed to the fixed action of the
    source (the only action executed before it). p = fallback frames / opportunities after the first decision.
    """
    per_source, per_stratum = {}, {}
    for source_id, campaign, arm, _, (mode, q) in SOURCES:
        log = arm_dir(campaign, arm) / CELL / "forced_action/FORCED_ACTIONS.jsonl"
        kinds = [json.loads(l)["kind"] for l in log.read_text().splitlines() if l.strip()]
        first = kinds.index("POLICY_DECISION")
        after = [k for k in kinds[first + 1:] if k in ("POLICY_DECISION", "FALLBACK")]
        startup = sum(k == "FALLBACK" for k in kinds[:first])
        fallbacks = sum(k == "FALLBACK" for k in after)
        executed = [json.loads(l)["executed"] for l in log.read_text().splitlines() if l.strip()]
        fallback_actions = sorted({(e["mode_id"], e["q_e4"], e["action_id"]) for e, k in zip(executed, kinds) if k == "FALLBACK"})
        per_source[source_id] = dict(log={"path": rel(log), "sha256": sha(log)}, startup_fallbacks=startup,
                                     later_fallbacks=fallbacks, later_opportunities=len(after),
                                     fallback_executed=[list(a) for a in fallback_actions])
        entry = per_stratum.setdefault(f"{mode}:{q}", dict(later_fallbacks=0, later_opportunities=0, sources=[]))
        entry["later_fallbacks"] += fallbacks
        entry["later_opportunities"] += len(after)
        entry["sources"].append(source_id)
    for entry in per_stratum.values():
        entry["p"] = entry["later_fallbacks"] / entry["later_opportunities"]
    startups = {v["startup_fallbacks"] for v in per_source.values()}
    return dict(schema="splitfusion.run5_restricted_pilot.fallback_evidence.v1", per_source=per_source,
                per_stratum=per_stratum, startup_fallbacks_per_run=sorted(startups),
                estimator="maximum likelihood per measured action point; linear in q between measured points; "
                          "applied per opportunity after each executed decision (and after each fallback) using the "
                          "last executed policy action; start-up fallbacks at run genesis only, attributed to no action")


def build(out: Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    identities, sources, split, trace = {}, [], {}, []
    for source_id, campaign, arm, rows_path, (mode, q) in SOURCES:
        directory = arm_dir(campaign, arm)
        identities[source_id] = runtime_identity(directory)
        rows = rows_for(rows_path, arm)
        if not rows or any((r["executed_action"]["mode_id"], r["executed_action"]["q_e4"]) != (mode, q) for r in rows):
            raise BindingBuildError(f"{source_id}: rows absent or executed action differs from ({mode}, {q})")
        frames = directory / CELL / "numeric_gt/service/FRAMES.jsonl"
        reg_sha = sha(directory / "registration.json")
        warm = [r for r in rows if not r["cold"]]
        for position, r in enumerate(warm):
            part = split_of(position)
            split[f"{source_id}#{r['frame_id']}"] = part
            trace.append(dict(source_id=source_id, frame_id=r["frame_id"], decision_index=r["decision_index"],
                              split=part, prior_ul_mcs=int(r["prior_ul_mcs"]), snr_db=float(r["snr_db"]),
                              backlog_bytes=int(r["pre_action_rlc_backlog_bytes"])))
        sources.append(dict(source_id=source_id, kind="BOUNDARIES_WITH_INLINE_CONTEXT_V1", network_profile="FAVORABLE_STABLE",
                            row_filter={"arm": arm}, field_map=FIELD_MAP,
                            files={"boundaries": {"path": rel(rows_path), "sha256": sha(rows_path)},
                                   "service_frames": {"path": rel(frames), "sha256": sha(frames)}},
                            run_identity=dict(campaign=campaign, arm=arm, registration_file_sha256=reg_sha,
                                              final_seal={"path": rel(V2_SEAL if campaign == "screen_v2" else V3_SEAL),
                                                          "sha256": sha(V2_SEAL if campaign == "screen_v2" else V3_SEAL)})))
    distinct = {json.dumps(v, sort_keys=True) for v in identities.values()}
    compatibility = dict(schema="splitfusion.run5_restricted_pilot.runtime_compatibility.v1",
                         excluded_from_comparison=list(SCREEN_PACKAGES), per_source=identities,
                         compatible=len(distinct) == 1)
    (out / "COMPATIBILITY.json").write_text(json.dumps(compatibility, indent=1, sort_keys=True))
    if len(distinct) != 1:
        raise BindingBuildError("v2/v3 runtime identities differ; evidence cannot be combined")
    common = json.loads(distinct.pop())
    for s in sources:
        s["pipeline_identity"] = common
    document = dict(
        schema="splitfusion.corrected_latency_env.calibration_binding.v2", candidate_version=CANDIDATE_VERSION,
        composition_model="SERVICE_QUEUE_V2", deadline_ms=170, binding_id="RUN5_RESTRICTED_PILOT_V1",
        purpose="PILOT_EXPLORATORY_RUN5_RESTRICTED_FAVORABLE_ONLY: not a calibration",
        pipeline_identity=common, sources=sources,
        policy=dict(exclude_cold=True, action_match={"rule": "EXACT_MODE_Q", "q_tolerance_e4": 0},
                    min_chains_per_stratum=1,
                    scene_conditioning={"descriptors": ["camera_si", "radar_p40"], "k": 5, "support_margin_fraction": 0.0},
                    uplink={"mode": "CHAIN_ONLY", "chain_selection": "JOINT_KNN_STANDARDIZED_SI_P40_MCS",
                            "transport_artifact_sha256": "9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c"},
                    fallback={"status": "UNCALIBRATED"}),
        registered_chain_rejections=dict(REGISTERED_REJECTIONS))
    (out / "PILOT_BINDING.json").write_text(json.dumps(document, indent=1, sort_keys=True))
    (out / "FALLBACK.json").write_text(json.dumps(fallback_evidence(), indent=1, sort_keys=True))
    split_doc = dict(schema="splitfusion.run5_restricted_pilot.split.v1", rule=f"per source, warm decisions in time order, "
                     f"blocks of {BLOCK}; block % 3 == 2 -> EVAL", chains=split, trace=trace,
                     counts={p: sum(v == p for v in split.values()) for p in ("TRAIN", "EVAL")})
    (out / "SPLIT.json").write_text(json.dumps(split_doc, indent=1, sort_keys=True))
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return dict(compatible=True, sources=len(sources), split=split_doc["counts"], binding_sha256=canonical(document))


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1]))))
