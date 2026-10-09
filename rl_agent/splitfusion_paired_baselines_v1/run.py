"""Offline three-branch paired perception-quality replay (quality only, no latency claim).

Branches on the identical retained prepared input, frozen weights, frame context and archived exact GT:
  FULL_LOCAL           frozen-front FP32 C2 straight into the tail (sealed v1 DENSE_DEFINITION)
  FIXED_NOAE_UINT8_Q0  external comparator: contract mode 0 SPLIT/noAE/UINT8 at q_e4=0 (keep 21504/21504,
                       ranker bypassed) through the production UE encoder and edge decoder
  HISTORICAL_SELECTED  that capture's recorded action, replayed from its saved SFD4 bytes; the
                       sealed replay_pair byte/label/record parity gates must pass first
The baseline is computed with the selected frame's tail metadata exactly as the sealed dense branch is;
its own branch identity is recorded separately and never assigned the historical action identity.
Reuses splitfusion_paired_quality_replay_v2 joins/adapter and the retained complete-target evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_paired_quality_replay_v2 import analyze as A
from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import load_adapter
from rl_agent.splitfusion_paired_quality_replay_v1 import capture as C
from rl_agent.splitfusion_paired_quality_replay_v1 import replay as RP
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
from rl_agent.splitfusion_run5_restricted_pilot_v1.quality import candidate_from_row

# v11 handoff pipeline: the edge timing field target_bundle_sha256 carries the scoring-packet digest
# (== timing.scoring_packet_sha256), not the bundle-manifest digest analyze.join_frame expects. The edge
# rows are withheld from join_frame and the cross-check is re-done here in the v11 domain.
JOIN_SUBSTITUTION = "V11_SCORING_PACKET_HASH_DOMAIN: edge measurement==archived, avo sha==, enqueue==, prepared_content_sha256=="


def edge_crosscheck(cell, frame, joined):
    key = A._key(frame["identity"])
    edge = cell["evaluations"].get(key)
    if edge is None:
        return "NO_EDGE_ROW"
    saved = cell["archived"][key]
    orig = joined["original_evaluation"]
    tm = edge.get("timing", {})
    assert edge["enqueued_wall_ns"] == saved["enqueued_wall_ns"], "edge/prediction enqueue differs"
    assert orig is not None and edge.get("measurement") == orig["measurement"], "edge/archived measurement differs"
    assert tm.get("avo_sidecar_sha256") == joined["evidence"]["avo_sha256"], "edge AVO digest differs"
    assert tm.get("prepared_content_sha256") == joined["prepared"].content_sha256, "edge prepared target differs"
    assert tm.get("target_bundle_sha256") == tm.get("scoring_packet_sha256"), "unexpected target hash domain"
    return "PASS_V11_DOMAIN"


BRANCHES = ("full_local", "fixed_uint8", "historical_selected")
BASELINE_IDENTITY = dict(branch="FIXED_NOAE_UINT8_Q0", mode_id=0, q_e4=0, anchor_action_id=0,
                         profile_id="split_noae_uint8_q0000", external_comparator=True)


def baseline_c2(adapter, dense_c2):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_live_dispatch_v1.timing import StageRecorder, UE_STAGES, EDGE_STAGES
    from rl_agent.splitfusion_live_dispatch_v1.transport import require_inner_agreement
    profile = adapter.contract.resolve_q_e4(0, 0)
    view = X.codec_view(profile)
    assert (profile.family, view.quantizer, view.keep_count, view.drop_count, profile.ranker_bypassed,
            profile.action_id) == ("noAE", "UINT8", 21504, 0, True, 0), view
    with torch.inference_mode():
        payload = adapter._ue._codec.encode(view, dense_c2.to(device=adapter.device), ranker=None,
                                            ae_encoder=None, timing=StageRecorder(UE_STAGES))
        codec = adapter._edge.runtime._codec
        timing = StageRecorder(EDGE_STAGES)
        inspected = codec.inspect(payload, timing=timing)
        require_inner_agreement(view, inspected.identity)
        decoded = codec.decode(inspected, decoder=None, tail_device=adapter.device, timing=timing)
    assert bool(decoded.finite)
    return decoded.c2, payload, profile.execution_bundle_sha256


def score(frame, joined, prediction, spec):
    ev = E.evaluate_prepared(identity=frame["identity"], prepared=joined["prepared"],
                             predicted_labels=prediction.labels, predictions=prediction.records, avo=joined["avo"])
    out = A.summarize_measurement(ev.measurement, spec)
    m = dict(ev.measurement)
    qd = out["quality"]
    out["run5_agreed_value_if_timely"] = candidate_from_row(dict(m, q_perc=qd["q_perc"], q_seg=qd["q_seg"]))
    out["measurement"] = m
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-archive", type=Path, required=True)
    p.add_argument("--input-manifest-sha256", required=True)
    p.add_argument("--cell", type=Path, required=True)
    p.add_argument("--cell-pins", type=Path, required=True)
    p.add_argument("--reward-spec", type=Path, required=True)
    p.add_argument("--reward-spec-sha256", required=True)
    p.add_argument("--model-options", type=Path, required=True)
    p.add_argument("--decisions-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    out = a.output.absolute()
    out.mkdir(parents=False, exist_ok=False)
    manifest = C.verify_archive(a.input_archive, expected_manifest_sha256=a.input_manifest_sha256)
    spec = Q.load_reward_spec(a.reward_spec, a.reward_spec_sha256)
    E.verify_sources()
    pins = json.loads(a.cell_pins.read_text())
    cell = A.load_cell(a.cell, pins)
    join_cell = dict(cell, evaluations={})
    options = json.loads(a.model_options.read_text())
    adapter = load_adapter(**options)
    with a.decisions_csv.open() as s:
        decisions = {int(r["frame_id"]): r for r in csv.DictReader(s)}
    rows = []
    for frame in manifest["frames"]:
        ident = frame["identity"]
        row = dict(index=frame["index"], frame_id=ident["frame_id"], historical_mode_id=ident["mode_id"],
                   historical_q_e4=ident["q_e4"], status="FAULT")
        d = decisions.get(ident["frame_id"], {})
        row["feedback_class"] = d.get("feedback_class")
        row["timely_feedback"] = d.get("timely_feedback")
        rows.append(row)
        try:
            joined = A.join_frame(frame, join_cell)
            row["edge_crosscheck"] = edge_crosscheck(cell, frame, joined)
            wire = (a.input_archive / frame["wire_file"]).read_bytes()
            A.verify_wire_source_pose(wire, joined)
            array = np.load(a.input_archive / frame["input_file"], allow_pickle=False)
            cb = adapter.replay_callbacks()
            captured = {}
            front = cb["front"]
            def capture_front(x, _f=front):
                c2 = _f(x)
                captured["c2"] = c2.detach().clone()
                return c2
            cb["front"] = capture_front
            pair = RP.replay_pair(prepared_input=torch.from_numpy(array), selected_wire=wire,
                                  expected_identity=ident, archived_selected=joined["archived_selected"], **cb)
            dense_c2 = RP._c2(captured["c2"], "captured front")
            c2b, payload, bundle = baseline_c2(adapter, dense_c2)
            base = RP._prediction(adapter.tail(RP._c2(c2b, "baseline decoder").clone(), pair.metadata),
                                  pair.metadata, "fixed baseline")
            preds = dict(full_local=pair.dense, fixed_uint8=base, historical_selected=pair.selected)
            for name in BRANCHES:
                row[name] = score(frame, joined, preds[name], spec)
            orig = joined["original_evaluation"]
            row["selected_matches_original_evaluator"] = (None if orig is None else
                row["historical_selected"]["measurement"] == orig["measurement"])
            if orig is not None:
                assert row["selected_matches_original_evaluator"], "selected replay/original evaluator mismatch"
            row["parity"] = dict(pair.evidence)
            row["baseline"] = dict(BASELINE_IDENTITY, execution_bundle_sha256=bundle, payload_bytes=len(payload),
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                labels_sha256=hashlib.sha256(base.labels.tobytes()).hexdigest())
            row["historical_payload_bytes"] = len(wire)
            row["status"] = "PAIRED"
        except Exception as exc:
            row["fault"] = f"{type(exc).__name__}: {exc}"
        print(row["index"], row["frame_id"], row["status"], row.get("fault", ""), flush=True)
    for name, digest in pins.items():
        A._json(a.cell / name, digest)
    C.verify_archive(a.input_archive, expected_manifest_sha256=a.input_manifest_sha256)
    report = dict(schema="splitfusion.paired_baselines.v1", claim_scope="OFFLINE_SAME_INPUT_QUALITY_ONLY",
                  live_latency_claim=False, join_substitution=JOIN_SUBSTITUTION, branches=dict(full_local=RP.DENSE_DEFINITION,
                  fixed_uint8=BASELINE_IDENTITY, historical_selected="RECORDED_ACTION_OF_THIS_CAPTURE_HISTORICAL_POLICY"),
                  input_manifest_sha256=a.input_manifest_sha256, cell_pins=pins,
                  reward_spec_sha256=a.reward_spec_sha256, model_binding_hashes=dict(adapter.binding_hashes),
                  code_sha256=C.sha(Path(__file__)), rows=rows)
    C.write_json(out / "REPLAY.json", report)
    print(json.dumps(dict(paired=sum(r["status"] == "PAIRED" for r in rows), total=len(rows))))


if __name__ == "__main__":
    raise SystemExit(main())
