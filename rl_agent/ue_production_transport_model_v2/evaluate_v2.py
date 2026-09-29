#!/usr/bin/env python3
"""Run Phase C/D selection, freeze the model, export the v2 artifact.

Order matters and is enforced:

1. grouped leave-one-whole-FIT-cell-out CV on FIT cells only;
2. evaluate the frozen acceptance gates against the pooled CV metrics;
3. refit once on all FIT rows and freeze every modelling decision;
4. export the v2 artifact;
5. **only then** touch the already-twice-inspected v1 validation population,
   once, as a descriptive engineering audit that is explicitly not a
   confirmation and does not make the v1 Gate 5 pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from rl_agent.ue_production_queue_capture_v1 import contract as V1

from . import artifact_v2 as A2
from . import contract_v2 as C2
from . import model_v2 as M


def run(join_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError("evaluation output is create-only")
    C2.verify_preserved()
    C2.verify_oai_ceiling()
    join_report = json.loads(
        (join_dir / "CAUSAL_JOIN_REPORT.json").read_text(encoding="utf-8"))
    rows = M.load_decisions(join_dir / "decisions.csv")
    fit_rows = [row for row in rows if row["partition"] == V1.FIT]
    validation_rows = [row for row in rows if row["partition"] == V1.VALIDATION]
    output_dir.mkdir(parents=True, exist_ok=False)

    # 1. selection
    cv = M.grouped_cross_validation(fit_rows)

    # 2. frozen acceptance gates, evaluated on the pooled CV metrics
    pooled = cv["pooled"]

    # 3. refit once on all FIT rows; every modelling decision is now frozen
    model = M.fit_model(fit_rows)
    monotone = M.monotonicity_violations(model)
    ranking = M.action_ranking_sensitivity(model, fit_rows)

    metrics_for_gates = {
        **pooled, "monotonicity_violations": monotone["violations"],
        "causal_coverage": join_report["coverage"],
        "feature_provenance_passed":
            join_report["feature_provenance_audit"]["passed"],
        "confusion": {},
    }
    verdicts = A2.gate_verdicts(metrics_for_gates)

    # 4. export
    document = A2.export_artifact(
        model=model, cv_metrics=metrics_for_gates,
        causal_coverage=join_report["coverage"],
        feature_provenance_passed=(
            join_report["feature_provenance_audit"]["passed"]),
        monotonicity_violations=monotone["violations"],
        action_ranking=ranking,
        raw_backlog_support=join_report["raw_backlog_support"])
    artifact_path = output_dir / "transport_model_v2.json"
    with artifact_path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    # 5. descriptive audit only, once, on the no-longer-pristine population
    audit: dict[str, Any] | None = None
    if document["all_gates_passed"]:
        audit = {
            "status": "DESCRIPTIVE_ENGINEERING_AUDIT_ONLY",
            "not_a_confirmation": True,
            "v1_gate5_status": C2.V1_GATE5_STATUS,
            "population_status": C2.VALIDATION_POPULATION_STATUS,
            **M.score(model, validation_rows),
        }

    report = {
        "schema": "scenesense.production_transport_model_v2_evaluation.v1",
        "contract_v2_sha256": C2.CONTRACT_V2_SHA256,
        "evidence_class": C2.EVIDENCE_CLASS,
        "causal_join": {
            "coverage": join_report["coverage"],
            "backlog_joins_changed_vs_v1":
                join_report["backlog_joins_changed_vs_v1"],
            "mcs_joins_changed_vs_v1":
                join_report["mcs_joins_changed_vs_v1"],
            "cutoff_field": join_report["cutoff_field"],
            "provenance_passed":
                join_report["feature_provenance_audit"]["passed"],
        },
        "cross_validation": cv,
        "frozen_model": model.to_dict(),
        "monotonicity": monotone,
        "action_ranking_sensitivity": ranking,
        "gate_verdicts": verdicts,
        "all_gates_passed": all(verdicts.values()),
        "backlog_mapping_report": C2.backlog_mapping_report(),
        "artifact_sha256": C2.sha256_file(artifact_path),
        "descriptive_validation_audit": audit,
    }
    with (output_dir / "EVALUATION_V2.json").open("x", encoding="utf-8") as h:
        json.dump(report, h, indent=2, sort_keys=True, allow_nan=False)
        h.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = run(Path(args.join_dir), Path(args.output_dir))
    json.dump({"gate_verdicts": report["gate_verdicts"],
               "all_gates_passed": report["all_gates_passed"]},
              sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
