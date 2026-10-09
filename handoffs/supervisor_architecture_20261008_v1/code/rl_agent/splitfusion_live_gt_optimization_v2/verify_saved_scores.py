"""Exact candidate replay through original scoring on all45 native predictions."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import candidate as C
from . import profile_saved as P
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference, cases = P.load_cases()
    archive = P.CELL / "profiler/edge/prediction_archive"
    manifest = json.loads((archive / "MANIFEST.json").read_text())
    assert all(P.sha(archive / name) == digest for name, digest in manifest["sha256"].items())
    identities = {}
    for path in archive.glob("*.identity.json"):
        value = json.loads(path.read_text())
        frame = value["gt_identity"]["frame_id"]
        assert frame not in identities
        identities[frame] = (path.name.removesuffix(".identity.json"), value)
    assert len(identities) == 45
    ue = json.loads((P.CELL / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_text())
    received = {row["frame_id"]: row for row in ue["feedback_rows"]}
    spec = W.load_run4_quality_spec(P.ROOT)
    rows = []
    with P.V1.install_cpu_gt_optimizations(expected_sources=P.V1.source_inventory()), C.install():
        producer = reference.producer(*cases[0][:2])
        for bundle, evidence, kwargs, record, avo in cases:
            frame = kwargs["identity"]["frame_id"]
            population = evidence["dynamic_snapshot_evidence"]
            if producer.population != frozenset(population["registered_population_ids"]):
                producer.register_population(population["registered_population_ids"],
                    population["population_sha256"], before_frame_id=frame)
            with patch("rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync.time.clock_gettime_ns",
                       return_value=evidence["source_synchronization"]["observed_raw_ns"]):
                inputs = producer.build(**kwargs)
            assert inputs.provenance == bundle.inputs.provenance
            prepared = P.E.prepare_target(inputs)
            assert prepared.content_sha256 == bundle.prepared.content_sha256
            stem, identity = identities[frame]
            assert identity["gt_identity"] == kwargs["identity"]
            labels = np.load(archive / (stem + ".labels.npy"), allow_pickle=False)
            records = json.loads((archive / (stem + ".records.json")).read_text())
            saved = json.loads((archive / (stem + ".evaluation.json")).read_text())
            result = P.E.evaluate_prepared(identity=identity["action_identity"], prepared=prepared,
                predicted_labels=labels, predictions=records, avo=avo["avo"])
            assert P.G.canonical(result.evidence) == P.G.canonical(saved), f"scoring evidence changed frame{frame}"
            quality = Q.evaluate_exact_quality(spec, result.measurement)
            actual = None if quality is None else float(quality.q_perc)
            assert actual == received[frame]["q_perc"], f"Q_perc differs from actual live feedback frame{frame}"
            rows.append({"frame_id": frame, "measurement_and_evidence_exact": True,
                "q_perc": actual, "received_q_perc_exact": True,
                "evidence_sha256": P.sha(archive / (stem + ".evaluation.json"))})
    assert not torch.cuda.is_initialized()
    result = {"status": "ALL45_TARGET_SCORING_AND_QPERC_EXACT", "rows": rows,
        "frames": len(rows), "defined_quality": sum(row["q_perc"] is not None for row in rows),
        "undefined_quality": sum(row["q_perc"] is None for row in rows),
        "live_started": False, "training_started": False, "cuda_initialized": False,
        "source_sha256": {"candidate.py": P.sha(C.__file__), "verify_saved_scores.py": P.sha(__file__)},
        "prediction_manifest_sha256": P.sha(archive / "MANIFEST.json"),
        "quality_scope": "Actual native predictions and saved AVO, newly rebuilt complete candidate targets; original scorer and formula.",
        "limits": "Does not change or erase the45 observed late TIMEOUTs; not a new live evaluation."}
    P.write(args.output / "RESULT.json", result)
    P.write(args.output / "OUTPUT_SHA256.json", P.R.tree(args.output))
    print(json.dumps({key: result[key] for key in ("status", "frames", "defined_quality", "undefined_quality")}))


if __name__ == "__main__":
    main()
