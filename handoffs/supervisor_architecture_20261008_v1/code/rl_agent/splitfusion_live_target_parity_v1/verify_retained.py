"""Reproducible, create-only offline parity evidence; no prediction evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .adapter import (ROOT, build_complete_v010, required_collection_inputs,
                      verify_sources, sha256)
from .fixtures import FIXTURES, load_retained_fixture


def verify() -> dict:
    sources = verify_sources()
    rows = []
    for sample in FIXTURES:
        inputs, references = load_retained_fixture(sample)
        result = build_complete_v010(inputs)
        checks = {"segmentation_png_byte_identical": result["segmentation_png"] == references["segmentation"].read_bytes(),
                  "object_ignore_png_byte_identical": result["object_ignore_png"] == references["object_ignore"].read_bytes()}
        for name in ("object_boxes.csv", "object_ignore_regions.csv"):
            checks[name + "_rows_identical"] = result["tables"][name] == references[name]
        if not all(checks.values()):
            raise RuntimeError(f"retained target parity failed: {sample}")
        labels, counts = np.unique(result["segmentation"], return_counts=True)
        rows.append({"sample_id": sample, "checks": checks, "references": FIXTURES[sample],
                     "input_provenance": result["provenance"],
                     "label_pixels": dict(zip(map(str, labels), map(int, counts))),
                     "positive_rows": len(result["tables"]["object_boxes.csv"]),
                     "ignore_reasons": sorted(set(r["reason"] for r in result["tables"]["object_ignore_regions.csv"]))})
    if verify_sources() != sources:
        raise RuntimeError("source identity changed during parity verification")
    package = Path(__file__).parent
    return {"status": "OFFLINE_RETAINED_V010_TARGET_PARITY_PASS", "live_qualified": False,
            "policy_performance_claim": False, "sources": sources, "fixtures": rows,
            "package_sources": {p.name: sha256(p) for p in sorted(package.glob("*.py"))},
            "collection_interface": required_collection_inputs(),
            "limits": ["Two prediction-blind retained validation fixtures, not all corpus frames.",
                       "Fixtures cover all five recorded IGNORE reasons and all three positive source/class combinations.",
                       "No live collector or real-time cost was tested.",
                       "Existing Run5 AVO-only evidence is incomplete for v010, and dense predictions were not retained.",
                       "Historical sample-keyed quarantines do not apply to unseen live frames."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
        print(json.dumps({"status": result["status"], "output": str(args.output),
                          "sha256": sha256(args.output), "live_qualified": False}))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
