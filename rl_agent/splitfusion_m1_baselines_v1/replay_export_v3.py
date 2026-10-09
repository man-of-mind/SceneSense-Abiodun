"""Same-input full-local replay and durable detection-record export for M1.

This is offline quality/size analysis only.  The selected branch must reproduce
the archived live prediction exactly before the full-local result is admitted.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_paired_quality_replay_v1 import replay as RP
from rl_agent.splitfusion_paired_quality_replay_v2.model_adapter import (
    CHECKPOINT_SHA256, load_adapter, source_binding_hashes)
from rl_agent.splitfusion_paired_retention_v2 import gt_adapter as G, writer as WR
from rl_agent.splitfusion_run5_quality_reference_v1.replay import (
    admitted_rows, branch_score)


MAIN = Path(__file__).resolve().parents[2]
SCHEMA = "splitfusion.m1_baselines_v1.replay_export.v3"
SERIALIZER = (
    "rl_agent.splitfusion_edge_optimization_v1.detached_tail_v3."
    "DetachedOptimizedTailAdapterV3.serialize_product")
SERIALIZER_SOURCE = MAIN / "rl_agent/splitfusion_edge_optimization_v1/detached_tail_v3.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def replay_rows(adapter, rows, refs, retention_dir, records_dir, prior):
    retention_dir = Path(retention_dir)
    prior_by_ordinal = ({int(row["ordinal"]): row for row in prior.get("rows", [])}
                        if prior else {})
    for row in rows:
        if row["admission"] != "CANDIDATE":
            print(row["ordinal"], row["admission"], flush=True)
            continue
        entry, ref = row.pop("entry"), row.pop("reference")
        row.update(frame_id=entry["join_key"]["frame_id"],
                   session_uuid=entry["join_key"]["session_uuid"],
                   tensor_seq=entry["join_key"]["tensor_seq"],
                   action=entry["action"], frame_dir=ref["frame_dir"])
        try:
            frame_dir = Path(refs["archive_root"]) / ref["frame_dir"]
            for name, digest in ref["files"].items():
                if WR.sha256_file(frame_dir / name) != digest:
                    raise RP.ReplayParityError(f"referenced GT file modified: {name}")
            frame = G.load_frame(frame_dir)
            npy = retention_dir / entry["input_file"]
            wire_path = retention_dir / entry["wire_file"]
            if (WR.sha256_file(npy) != entry["input_sha256"]
                    or WR.sha256_file(wire_path) != entry["wire_sha256"]):
                raise RP.ReplayParityError("retained input/wire digest differs")
            array = np.load(npy, allow_pickle=False)
            wire = wire_path.read_bytes()
            archived = RP.Prediction(
                np.frombuffer(frame.raw["scorer/predicted_labels.u8"], np.uint8)
                  .reshape(720, 1280),
                tuple(frame.score["prediction_records"]))
            pair = RP.replay_pair(
                prepared_input=torch.from_numpy(array), selected_wire=wire,
                expected_identity=entry["identity"], archived_selected=archived,
                **adapter.replay_callbacks())
            row["full_local"] = branch_score(frame, pair.dense)
            row["live_selected"] = branch_score(frame, pair.selected)
            selected = row["live_selected"]
            if (canonical(selected["measurement"]) != canonical(frame.score["measurement"])
                    or selected["q_perc"] != frame.score["q_perc"]):
                raise RP.ReplayParityError(
                    "selected branch score differs from archived live numerical score")
            raw = canonical(list(pair.dense.records))
            digest = sha(raw)
            if digest != pair.evidence["dense_records_sha256"]:
                raise RP.ReplayParityError("full-local exported bytes/hash drift")
            prior_row = prior_by_ordinal.get(int(row["ordinal"]))
            prior_digest = ((prior_row or {}).get("full_local") or {}).get("records_sha256")
            if prior is not None and prior_digest != digest:
                raise RP.ReplayParityError("full-local hash differs from prior sealed replay")
            relative = (f"ordinal_{int(row['ordinal']):04d}__"
                        f"frame_{int(row['frame_id'])}.records.json")
            with (records_dir / relative).open("xb") as stream:
                stream.write(raw)
            keys = sorted({key for record in pair.dense.records for key in record})
            row["full_local_export"] = dict(
                file="full_local_records/" + relative,
                bytes=len(raw), sha256=digest, records=len(pair.dense.records),
                record_fields=keys,
                prior_replay_hash_match=(None if prior is None else True))
            row["selected_live_parity"] = dict(pair.evidence)
            row["selected_wire_bytes"] = len(wire)
            row["admission"] = "ADMITTED"
        except Exception as exc:
            for name in ("full_local", "live_selected", "full_local_export",
                         "selected_live_parity"):
                row.pop(name, None)
            row["admission"] = "MISSING_REPLAY_PARITY_OR_FAULT"
            row["fault"] = f"{type(exc).__name__}: {exc}"[:800]
        print(row["ordinal"], row.get("frame_id"), row["admission"],
              row.get("fault", ""), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--session-label", required=True)
    parser.add_argument("--prior-replay", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    output.mkdir(parents=False, exist_ok=False)
    records_dir = output / "full_local_records"
    records_dir.mkdir()
    manifest, refs, rows = admitted_rows(
        args.retention_dir, manifest_sha256=args.manifest_sha256,
        plan_sha256=args.plan_sha256, references_path=args.references,
        references_sha256=args.references_sha256)
    expected = source_binding_hashes(MAIN)
    worktree = source_binding_hashes(args.worktree_root)
    if expected != worktree:
        raise SystemExit("model source binding differs from producing worktree")
    scorer_main = G.scorer_source_digests(MAIN)
    if (scorer_main != G.scorer_source_digests(args.worktree_root)
            or scorer_main != refs["scorer_sources"]):
        raise SystemExit("scorer sources differ from capture evidence")
    prior = (json.loads(args.prior_replay.read_bytes())
             if args.prior_replay else None)
    adapter = load_adapter(device=args.device,
                           expected_binding_hashes=expected,
                           allow_cold_load=True)
    replay_rows(adapter, rows, refs, args.retention_dir, records_dir, prior)
    for row in rows:
        row.pop("entry", None)
        row.pop("reference", None)
    admitted = [row for row in rows if row["admission"] == "ADMITTED"]
    fields = sorted({field for row in admitted
                     for field in row["full_local_export"]["record_fields"]})
    bytes_values = [row["full_local_export"]["bytes"] for row in admitted]
    export_manifest = dict(
        schema="splitfusion.m1_baselines_v1.full_local_detection_export.v1",
        session_label=args.session_label,
        serializer=SERIALIZER,
        serializer_source_sha256=WR.sha256_file(SERIALIZER_SOURCE),
        encoding="UTF-8 canonical JSON array; sorted keys; compact separators; finite values",
        compression="NONE", byte_layer="SERIALIZED_DETECTION_RECORDS",
        included="all emitted detections including false positives and record identity metadata",
        excluded="GT, debug logs, label raster and dense segmentation",
        record_fields=fields, samples=len(admitted),
        files=[row["full_local_export"] for row in admitted],
        total_bytes=sum(bytes_values))
    manifest_raw = json.dumps(export_manifest, indent=1, sort_keys=True,
                              allow_nan=False).encode()
    with (output / "FULL_LOCAL_EXPORT_MANIFEST.json").open("xb") as stream:
        stream.write(manifest_raw)
    report = dict(
        schema=SCHEMA, session_label=args.session_label,
        claim_scope="OFFLINE_SAME_INPUT_QUALITY_AND_SERIALIZED_OUTPUT_SIZE_ONLY",
        live_latency_claim=False, network_transmission_claim=False,
        branches=dict(full_local=RP.DENSE_DEFINITION,
                      live_selected="ARCHIVED_LIVE_SELECTED_OUTPUT_EXACTLY_REPRODUCED"),
        retention_manifest_sha256=args.manifest_sha256,
        plan_sha256=args.plan_sha256,
        references_sha256=args.references_sha256,
        prior_replay_sha256=(WR.sha256_file(args.prior_replay)
                             if args.prior_replay else None),
        model_binding_hashes=dict(adapter.binding_hashes),
        checkpoint_sha256=dict(CHECKPOINT_SHA256), scorer_sources=scorer_main,
        device=args.device,
        code_sha256={"replay_export_v3.py": WR.sha256_file(__file__)},
        full_local_export_manifest_sha256=sha(manifest_raw),
        admission_counts=dict(collections.Counter(row["admission"] for row in rows)),
        rows=rows)
    raw = json.dumps(report, indent=1, sort_keys=True, allow_nan=False).encode()
    with (output / "REPLAY_EXPORT.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(admission_counts=report["admission_counts"],
                          full_local_samples=len(admitted),
                          total_detection_bytes=sum(bytes_values))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
