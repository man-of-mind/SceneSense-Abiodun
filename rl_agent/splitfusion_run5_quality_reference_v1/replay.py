"""Offline three-branch same-input replay on the retention-v2 archive of the full Run-5 agent (quality only).

Branches, all on the identical retained prepared CPU tensor, the same frozen weights, the same frame context and the
same archived exact numeric GT (scored by the unchanged numeric scorer, cross-checked by the retained evaluator):
  FULL_LOCAL            frozen-front FP32 C2 straight into the tail (sealed ``replay_pair`` DENSE_DEFINITION)
  NOAE_UINT8_Q0         contract mode 0 SPLIT/noAE/UINT8, q_e4=0 (keep 21504/21504 = zero spatial dropping, ranker
                        bypassed) through the production UE encoder and edge decoder; its own branch identity
  AGENT_SELECTED        the agent's recorded action, decoded from the saved transmitted SFD4 bytes

Per-sample admission (registered before the capture, ``contract.RETENTION_ADMISSION``): the record must be
RETAINED_COMPLETE, its GT reference RESOLVED (exact rescoring + identity joins), every referenced GT file re-hashes,
and the selected branch must reproduce the archived inner payload bytes, native labels and complete records
(``replay_pair`` gates) AND the archived numerical measurement / Q_perc exactly. Otherwise the sample is reported as
missing with its reason; nothing is repaired, substituted or re-sampled. No live-latency claim for any branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_paired_quality_replay_v1 import replay as RP
from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import (CHECKPOINT_SHA256, load_adapter,
                                                                        source_binding_hashes)
from rl_agent.splitfusion_paired_retention_v2 import gt_adapter as G, validate as V, writer as WR

SCHEMA = "splitfusion.run5_quality_reference.replay.v1"
BRANCHES = ("full_local", "noae_uint8_q0", "agent_selected")
BASELINE_IDENTITY = dict(branch="NOAE_UINT8_Q0", mode_id=0, q_e4=0, anchor_action_id=0,
                         profile_id="split_noae_uint8_q0000", family="noAE", quantizer="UINT8", keep_count=21504,
                         drop_count=0, ranker_bypassed=True, external_comparator=True)
MAIN = Path(__file__).resolve().parents[2]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def baseline_c2(adapter, dense_c2):
    """noAE/UINT8/q0 through the production codec (same construction as splitfusion_paired_baselines_v1.run)."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_live_dispatch_v1.timing import StageRecorder, UE_STAGES, EDGE_STAGES
    from rl_agent.splitfusion_live_dispatch_v1.transport import require_inner_agreement
    profile = adapter.contract.resolve_q_e4(0, 0)
    view = X.codec_view(profile)
    if (profile.family, view.quantizer, view.keep_count, view.drop_count, profile.ranker_bypassed,
            profile.action_id) != ("noAE", "UINT8", 21504, 0, True, 0):
        raise RP.ReplayParityError(f"baseline profile differs from the registered identity: {view}")
    with torch.inference_mode():
        payload = adapter._ue._codec.encode(view, dense_c2.to(device=adapter.device), ranker=None, ae_encoder=None,
                                            timing=StageRecorder(UE_STAGES))
        codec = adapter._edge.runtime._codec
        timing = StageRecorder(EDGE_STAGES)
        inspected = codec.inspect(payload, timing=timing)
        require_inner_agreement(view, inspected.identity)
        decoded = codec.decode(inspected, decoder=None, tail_device=adapter.device, timing=timing)
    if not bool(decoded.finite):
        raise RP.ReplayParityError("baseline decoder produced nonfinite C2")
    return decoded.c2, payload, profile.execution_bundle_sha256


def branch_score(frame, prediction) -> dict:
    scored = G.score_prediction(frame, prediction.labels, list(prediction.records))
    if not scored["evaluate_prepared_measurement_equal"]:
        raise RP.ReplayParityError("numeric scorer and retained evaluator disagree on a branch measurement")
    return dict(measurement=scored["measurement"], q_perc=scored["q_perc"], quality_defined=scored["quality_defined"],
                labels_sha256=sha(prediction.labels.tobytes()), records_sha256=sha(canonical(list(prediction.records))),
                records=len(prediction.records))


def admitted_rows(retention_dir, *, manifest_sha256, plan_sha256, references_path, references_sha256):
    """All planned ordinals with their admission status; never raises for a single sample."""
    manifest = V.verify_archive_v2(retention_dir, expected_manifest_sha256=manifest_sha256,
                                   expected_plan_sha256=plan_sha256, strict=False)
    raw = Path(references_path).read_bytes()
    if sha(raw) != references_sha256:
        raise V.RetentionAdmissionError("unbound or modified GT references")
    refs = json.loads(raw)
    if (refs.get("schema") != G.REFERENCES_SCHEMA or refs.get("retention_manifest_sha256") != manifest_sha256
            or refs.get("plan_sha256") != plan_sha256 or refs.get("target_hash_domain") != G.TARGET_HASH_DOMAIN):
        raise V.RetentionAdmissionError("GT references bound to another manifest/plan/domain")
    by_ordinal = {r["ordinal"]: r for r in refs["rows"]}
    records = {e["ordinal"]: e for e in manifest["records"]}
    rows = []
    for ordinal in manifest["plan"]["ordinals"]:
        status = manifest["ordinal_status"][str(ordinal)]
        row = dict(ordinal=ordinal, retention_status=status)
        if status != "RETAINED_COMPLETE":
            row["admission"] = "MISSING_" + status
        else:
            ref = by_ordinal.get(ordinal)
            row.update(entry=records[ordinal], reference=ref)
            if ref is None:
                row["admission"] = "MISSING_NO_GT_REFERENCE"
            elif ref.get("status") != "RESOLVED" or not ref.get("rescore", {}).get("exact") or not all(ref.get("joins", {}).values()):
                row["admission"] = "MISSING_GT_" + str(ref.get("status"))
            elif ref.get("edge_target_bundle_sha256_consulted") is not False:
                row["admission"] = "MISSING_SILENT_TARGET_HASH_SUBSTITUTION"
            else:
                row["admission"] = "CANDIDATE"
        rows.append(row)
    return manifest, refs, rows


def replay_rows(adapter, rows, refs, retention_dir, *, baseline=baseline_c2):
    """Per-sample replay + admission (mutates ``rows``). ``adapter``/``baseline`` are injectable for CPU tests."""
    retention_dir = Path(retention_dir)
    for row in rows:
        if row["admission"] != "CANDIDATE":
            print(row["ordinal"], row["admission"], flush=True)
            continue
        entry, ref = row.pop("entry"), row.pop("reference")
        row.update(frame_id=entry["join_key"]["frame_id"], session_uuid=entry["join_key"]["session_uuid"],
                   tensor_seq=entry["join_key"]["tensor_seq"], action=entry["action"], frame_dir=ref["frame_dir"])
        try:
            frame_dir = Path(refs["archive_root"]) / ref["frame_dir"]
            for name, digest in ref["files"].items():
                if WR.sha256_file(frame_dir / name) != digest:
                    raise RP.ReplayParityError(f"referenced GT file modified: {name}")
            frame = G.load_frame(frame_dir)
            npy, wire_path = retention_dir / entry["input_file"], retention_dir / entry["wire_file"]
            if WR.sha256_file(npy) != entry["input_sha256"] or WR.sha256_file(wire_path) != entry["wire_sha256"]:
                raise RP.ReplayParityError("retained input/wire digest differs")
            array = np.load(npy, allow_pickle=False)
            wire = wire_path.read_bytes()
            archived = RP.Prediction(np.frombuffer(frame.raw["scorer/predicted_labels.u8"], np.uint8).reshape(720, 1280),
                                     tuple(frame.score["prediction_records"]))
            callbacks = adapter.replay_callbacks()
            captured, front = {}, callbacks["front"]

            def capture_front(x, _f=front):
                c2 = _f(x)
                captured["c2"] = c2.detach().clone()
                return c2
            callbacks["front"] = capture_front
            pair = RP.replay_pair(prepared_input=torch.from_numpy(array), selected_wire=wire,
                                  expected_identity=entry["identity"], archived_selected=archived, **callbacks)
            dense_c2 = RP._c2(captured["c2"], "captured front")
            c2b, payload, bundle = baseline(adapter, dense_c2)
            base = RP._prediction(adapter.tail(RP._c2(c2b, "baseline decoder").clone(), pair.metadata),
                                  pair.metadata, "noAE UINT8 q0 baseline")
            preds = dict(full_local=pair.dense, noae_uint8_q0=base, agent_selected=pair.selected)
            for name in BRANCHES:
                row[name] = branch_score(frame, preds[name])
            selected = row["agent_selected"]
            if (canonical(selected["measurement"]) != canonical(frame.score["measurement"])
                    or selected["q_perc"] != frame.score["q_perc"]):
                raise RP.ReplayParityError("selected branch score differs from the archived numerical score")
            row["parity"] = dict(pair.evidence, archived_measurement_identical=True, archived_q_perc_identical=True)
            row["baseline"] = dict(BASELINE_IDENTITY, execution_bundle_sha256=bundle, payload_bytes=len(payload),
                                   payload_sha256=sha(payload))
            row["agent_payload_bytes"] = len(wire)
            row["admission"] = "ADMITTED"
        except Exception as exc:   # classified per sample; never a silent pass
            for name in BRANCHES:
                row.pop(name, None)
            row["admission"] = "MISSING_REPLAY_PARITY_OR_FAULT"
            row["fault"] = f"{type(exc).__name__}: {exc}"[:500]
        print(row["ordinal"], row.get("frame_id"), row["admission"], row.get("fault", ""), flush=True)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retention-dir", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--references-sha256", required=True)
    p.add_argument("--worktree-root", type=Path, required=True, help="checkout that produced the capture")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    out = a.output.absolute()
    out.mkdir(parents=False, exist_ok=False)
    manifest, refs, rows = admitted_rows(a.retention_dir, manifest_sha256=a.manifest_sha256, plan_sha256=a.plan_sha256,
                                         references_path=a.references, references_sha256=a.references_sha256)
    # model identity: the main-checkout sources must equal the producing worktree's (adapter verifies checkpoints)
    expected = source_binding_hashes(MAIN)
    worktree = source_binding_hashes(a.worktree_root)
    if expected != worktree:
        raise SystemExit("model source binding differs between this checkout and the producing worktree")
    scorer_main, scorer_wt = G.scorer_source_digests(MAIN), G.scorer_source_digests(a.worktree_root)
    if scorer_main != scorer_wt or scorer_main != refs["scorer_sources"]:
        raise SystemExit("scorer sources differ from the producing worktree or from the GT references")
    adapter = load_adapter(device=a.device, expected_binding_hashes=expected, allow_cold_load=True)
    replay_rows(adapter, rows, refs, a.retention_dir)
    for row in rows:
        row.pop("entry", None)
        row.pop("reference", None)
    report = dict(schema=SCHEMA, claim_scope="OFFLINE_SAME_INPUT_QUALITY_ONLY", live_latency_claim=False,
                  branches=dict(full_local=RP.DENSE_DEFINITION, noae_uint8_q0=BASELINE_IDENTITY,
                                agent_selected="RECORDED_ACTION_OF_THE_FULL_RUN5_ACTOR_FROM_THE_SAVED_SFD4_BYTES"),
                  retention_manifest_sha256=a.manifest_sha256, plan_sha256=a.plan_sha256,
                  references_sha256=a.references_sha256, model_binding_hashes=dict(adapter.binding_hashes),
                  checkpoint_sha256=dict(CHECKPOINT_SHA256), scorer_sources=scorer_main, device=a.device,
                  code_sha256={n: sha((Path(__file__).parent / n).read_bytes()) for n in ("replay.py",)},
                  admission_counts={s: sum(r["admission"] == s for r in rows) for s in sorted({r["admission"] for r in rows})},
                  rows=rows)
    raw = json.dumps(report, sort_keys=True, indent=1, allow_nan=False).encode()
    with (out / "REPLAY.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(report["admission_counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
