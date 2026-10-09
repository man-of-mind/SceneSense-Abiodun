"""Bounded read-only source/binding audit; CPU preprocessing only, no model load.

Usage: python3 -m rl_agent.splitfusion_run5_diagnosis_v1.audit_inference --output PATH
The output is create-only. Inputs and all original evidence remain untouched.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
WT = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1"
CAP = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261002T224700Z_run5_live_avo_capture_1000_v1"
PILOT = "rl_agent/splitfusion_live_dispatch_v1/live_pilot_runtime.py"
BASE = "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
DATA = BASE + "splitfusion_fcos_r50_fpn_p2_p7_v1/data.py"
AE = BASE + "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_phase11b_gpu_qualification.py"
LOCK = BASE + "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/PERCEPTION_FORWARD_LOCK_P025_V1.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree(path):
    return ast.parse(Path(path).read_text())


def constant(path, name):
    for node in tree(path).body:
        names = [x.id for x in node.targets if isinstance(x, ast.Name)] if isinstance(node, ast.Assign) else [node.target.id] if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else []
        if name in names:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def extracted(path, name, namespace, cls=None):
    nodes = tree(path).body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(code, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), namespace)
    return namespace[name]


def preprocessing_probe():
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    # Spatial gradient plus nonzero radar makes padding distinguishable from resize.
    y, x = np.indices((720, 1280))
    bgr = np.stack(((x % 251), (y % 251), ((x + y) % 251)), axis=2).astype(np.uint8)
    radar_live = np.broadcast_to(np.arange(1, 5, dtype=np.float32)[:, None, None], (4, 448, 768)).copy()
    ns = {"cv2": cv2, "np": np, "torch": torch}
    live_fn = extracted(WT / PILOT, "_prepare_live_input", ns)
    live = live_fn(bgr, radar_live, torch.device("cpu"))
    original_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    image_proxy = SimpleNamespace(open=lambda _: original_image, Resampling=Image.Resampling)
    env = {"torch": torch, "np": np, "F": F, "Image": image_proxy,
           "CONTENT_W": 768, "CONTENT_H": 432, "PAD_BOTTOM": 16}
    rgb = extracted(ROOT / DATA, "_rgb", env, "InferenceDataset")
    getitem = extracted(ROOT / DATA, "__getitem__", env, "InferenceDataset")
    env["model_intrinsic"] = lambda _: None
    env["camera_extrinsic"] = lambda _: None
    obj = SimpleNamespace(dataset=Path("/unused"), rows=[{"rgb_path": "synthetic"}],
        mean=torch.tensor([.485, .456, .406]).view(3, 1, 1),
        std=torch.tensor([.229, .224, .225]).view(3, 1, 1))
    obj._rgb = lambda row: rgb(obj, row)
    obj._radar = lambda _: torch.from_numpy(radar_live[:, :432].copy())
    reference = getitem(obj, 0)[0].unsqueeze(0)
    delta = (reference - live).abs()
    result = {"synthetic_only": True, "no_model_inference": True,
        "original_functions_executed": [f"{DATA}:InferenceDataset._rgb/__getitem__", f"{PILOT}:_prepare_live_input"],
        "reference_shape": list(reference.shape), "live_shape": list(live.shape),
        "reference_bottom16_nonzero": int(torch.count_nonzero(reference[..., -16:, :])),
        "live_bottom16_nonzero": int(torch.count_nonzero(live[..., -16:, :])),
        "tensor_equal": torch.equal(reference, live), "max_abs_difference": float(delta.max()),
        "reference_content_height": 432, "live_content_height": 448,
        "vertical_scale_ratio": 448 / 432, "vertical_scale_difference_percent": 100 * (448 / 432 - 1),
        "reference_cy": 216.0, "live_resized_content_cy": 224.0,
        "edge_intrinsic_cy": 216.0,
        "interpretation": "Confirmed preprocessing/calibration-grid mismatch, not a measurement of its effect on real XY error.",
        "secondary_difference": "PIL bilinear offline versus OpenCV INTER_LINEAR live; separate from padding/grid mismatch.",
        "cuda_initialized": torch.cuda.is_initialized()}
    assert result["reference_shape"] == result["live_shape"] == [1, 7, 448, 768]
    assert result["reference_bottom16_nonzero"] == 0 and result["live_bottom16_nonzero"] > 0
    assert not result["tensor_equal"] and not result["cuda_initialized"]
    return result


def run():
    reg = json.loads((CAP / "registration.json").read_text())
    manifest = json.loads((CAP / "RUN5_POSTFLIGHT_INPUTS.json").read_text())
    files = manifest["files"]
    selected = ["capture/ACTOR_AUDIT_BEFORE.json", "capture/ACTOR_AUDIT_AFTER.json", "capture/PREFLIGHT.json",
                "capture/cells/a71__favorable_stable/run4_phase6/edge_report.json", "GT_POSTFLIGHT.json", "SNR_POSTFLIGHT.json"]
    checks = {p: {"recorded": files[p], "actual": sha(CAP / p)} for p in selected}
    assert all(v["recorded"] == v["actual"] for v in checks.values())
    sources = [PILOT, DATA, AE, LOCK,
        "rl_agent/splitfusion_live_dispatch_v1/frame_context.py",
        "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_ue_runtime_v2.py",
        "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_live_wire_v2.py",
        "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v3.py",
        "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v2.py",
        "rl_agent/splitfusion_run5_live_avo_v1/child_entry.py",
        "rl_agent/splitfusion_run5_live_avo_v1/quality_adapter.py",
        "rl_agent/splitfusion_run5_live_avo_v1/actor_binding.py",
        "rl_agent/splitfusion_run5_live_avo_v1/edge_quality.py",
        "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_threadcap.py",
        "rl_agent/splitfusion_single_gpu_profiler_v1/seams_ue_v1.py",
        BASE + "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/phase6_validation.py",
        BASE + "splitfusion_fcos_r50_fpn_p2_p7_v1/model.py",
        "carla_collect_parked_ego_fusion_training_data.py"]
    source_rows = {}
    for p in sources:
        q = WT / p if (WT / p).exists() else ROOT / p
        source_rows[p] = {"path": str(q), "sha256": sha(q)}
    frozen = constant(ROOT / AE, "FROZEN_INPUTS")
    assets = {}
    for name, row in frozen.items():
        q = WT / row["path"]
        digest = sha(q)
        assets[name] = {"path": str(q), "expected": row["sha256"], "actual": digest, "verified": digest == row["sha256"]}
    assert all(x["verified"] for x in assets.values())
    relevant_binds = {}
    for name, value in reg["experiment_binds"].items():
        if name.startswith("run5:") and name.split(":")[-1] in {"actor_binding.py", "child_entry.py", "quality_adapter.py", "edge_quality.py", "live_gt.py", "run5_state_contract.py", "run5_snr_v2.py", "actor_state_dict.pt"}:
            digest = sha(value["path"])
            relevant_binds[name] = {**value, "actual_sha256": digest, "verified": digest == value["sha256"]}
    assert all(x["verified"] for x in relevant_binds.values())
    assert reg["arms"]["C2"]["bundle"] == "production"
    findings = [
        {"id": "F1_INPUT_SPATIAL_CONTRACT", "status": "CONFIRMED_DEFECT", "references": [DATA + ":18", DATA + ":258", DATA + ":276", PILOT + ":682", "rl_agent/splitfusion_live_dispatch_v1/frame_context.py:257"], "statement": "Offline image/radar content is768x432 with16zero bottom rows; live image/radar content is768x448 withoutpadding; edgeintrinsics remain432-content grid. Cause magnitude unknown."},
        {"id": "F2_VEHICLE_THRESHOLD", "status": "CONFIRMED_SCORING_NONPARITY", "references": [BASE + "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/phase6_validation.py:424", "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/executor.py:597", "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_live_wire_v2.py:314", BASE + "splitfusion_fcos_r50_fpn_p2_p7_v1/model.py:204", "rl_agent/splitfusion_edge_optimization_v1/optimized_tail_v3.py:339"], "statement": "Offlinevehicle scorer appliescalibratedscore>=.20; livevehicle matcherdoesnot. Detector candidateflooris.02 beforecalibration; p025filters personsonly. Exactimpactcannotbereconstructedwithoutsavedrecords. Rewarddoesnotdirectlypenalizeprecision; anyQimpactthroughrecall/matched-errorisunknown."},
        {"id": "F3_SEGMENTATION_REFERENCE", "status": "CONFIRMED_SCORING_NONPARITY", "references": ["rl_agent/splitfusion_run5_live_avo_v1/quality_adapter.py:40", "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_live_wire_v2.py:295"], "statement": "LiveusesrawCARLA3classmask andresizesGTtoprediction; historicalv010usesdepthsupportedpersonmask, reconciledvehicles,IGNORE255andresizespredictiontotarget. Thisisnotapersonlocalizationexplanation."},
        {"id": "F4_PERSON_TARGET_MATCHING", "status": "CONFIRMED_SHARED_SEMANTICS", "references": ["carla_collect_parked_ego_fusion_training_data.py:295", "rl_agent/splitfusion_run5_live_avo_v1/live_gt.py:79", "rl_agent/splitfusion_run5_live_avo_v1/quality_adapter.py:251"], "statement": "BothusephysicalbboxcenterworldXY; personscore>=.25,AVO>=.65,greedymatching<=3m andignoredGTorderagree. Populationchangesandcompleteinputparityremainseparate."},
        {"id": "F5_VEHICLE_GT", "status": "CONFIRMED_SCORING_NONPARITY", "references": ["rl_agent/ue_route_b_split_cell_adapter_v1.py:1433", BASE + "route_b_v3_1_clean_base_v1/build_contract_v1.py:370"], "statement": "LivereusesactorGT/localizationfilter,notfullv010staticparkedvehiclereconciliationandignorepolygons. Vehicleprecisionnotcomparabletobaseline."},
        {"id": "F6_ACTOR_FEATURES", "status": "VERIFIED_RETAINED_EVIDENCE", "references": ["rl_agent/splitfusion_run5_live_avo_v1/actor_binding.py:127"], "statement": "OriginalRun5seed43update10000/22features; frozenRun5builderassertsexactRun4prefixandappendscausalSNR(5.5,24.5). Sealedbefore/afterauditspass. Noactorreplayperformedinthisaudit."},
        {"id": "F7_MODEL_IDENTITY", "status": "VERIFIED_ARTIFACT_HASHES", "references": [PILOT + ":608", "rl_agent/splitfusion_edge_optimization_v1/detached_edge_preload_v3.py:37"], "statement": "Bothsidesusetheoriginalepoch26perceptioncheckpoint;rankerepoch4andAE128/64/32matchfrozenartifacthashes. Noevidenceofwrongweights;thisdoesnotprovepreprocessingparity."}]
    # Expanded, reviewer-facing summaries; source references above remain exact.
    statements = {
        "F1_INPUT_SPATIAL_CONTRACT": "Offline image/radar content is 768x432 with 16 zero bottom rows; live image/radar content is 768x448 without padding. Edge intrinsics remain on the 432-content grid. The magnitude of its effect on localization is unknown.",
        "F2_VEHICLE_THRESHOLD": "The offline vehicle scorer applies calibrated score >= 0.20; the live vehicle matcher does not. The detector candidate floor is 0.02 before calibration, and p025 filters persons only. Exact impact cannot be reconstructed without saved records. Reward does not directly penalize precision; any effect through recall or matched error is unknown.",
        "F3_SEGMENTATION_REFERENCE": "Live evaluation uses raw CARLA 3-class masks and resizes GT to predictions. Historical v010 uses depth-supported person masks, reconciled vehicles, IGNORE 255, and resizes predictions to targets. This does not explain person localization errors directly.",
        "F4_PERSON_TARGET_MATCHING": "Both use physical bounding-box center world XY; person score >= 0.25, AVO >= 0.65, greedy matching <= 3 m, and ignored-GT ordering agree. Population changes and complete input parity remain separate questions.",
        "F5_VEHICLE_GT": "Live evaluation reuses actor GT and the localization filter, not the full v010 static parked-vehicle reconciliation and ignore polygons. Vehicle precision is not comparable with the frozen baseline.",
        "F6_ACTOR_FEATURES": "The original Run-5 seed-43/update-10000 actor has 22 features. The frozen Run-5 builder asserts the exact Run-4 prefix and appends causal SNR on [5.5,24.5]. Sealed before/after audits pass. No actor replay was performed in this audit.",
        "F7_MODEL_IDENTITY": "Both sides use the original epoch-26 perception checkpoint; ranker epoch 4 and AE128/64/32 match frozen artifact hashes. There is no evidence of wrong weights; this does not prove preprocessing parity.",
    }
    for finding in findings:
        finding["statement"] = statements[finding["id"]]
    return {"schema": "splitfusion.run5.diagnosis.inference_audit.v1", "capture": str(CAP),
        "capture_manifest_sha256": sha(CAP / "RUN5_POSTFLIGHT_INPUTS.json"),
        "read_only_inputs": True, "no_live_services": True, "no_model_load_or_inference": True,
        "selected_capture_hash_checks": checks, "source_hashes": source_rows,
        "relevant_registration_binds": relevant_binds, "frozen_assets": assets,
        "actor_audit": json.loads((CAP / selected[0]).read_text()),
        "actual_path_proof": {"registered_child": reg["child_module"], "arm": reg["arms"]["C2"],
            "call_chain": ["child_entry.py:main -> PART.main -> child_v1_threadcap.main", "child_v1_threadcap.py:47 optimized-onlypackingoverride skipped for production", "seams_ue_v1.py:366 wraps original _prepare_live_input, does not replace arithmetic", "phase6_ue_runtime_v2.py:622 calls base._prepare_live_input" ]},
        "preprocessing_probe": preprocessing_probe(), "findings": findings,
        "scene_feature_source_note": {"status": "SEPARATE_CONTRACT", "reference": "rl_agent/splitfusion_hybrid_sac_v1/scene_descriptors.py:186", "statement": "SI explicitly uses 768x448 luma. Do not change the SI grid automatically when correcting perception input packing. Raw source/catalogue parity of P40 was not fully audited."},
        "limitations": ["No causal localization improvement estimate", "No model inference", "No per-object score/prediction archive to rescore the old capture", "No changes to reward, policy, or existing source files"]}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "findings": len(result["findings"]), "probe": result["preprocessing_probe"]}, indent=2))


if __name__ == "__main__":
    main()
