#!/usr/bin/env python3
"""Seal compact, Git-trackable provenance of the completed Run-5 deep campaign.

Read-only over the campaign directory; writes one create-only JSON at the
package root.  ``--verify`` recomputes every field from the untouched campaign
and compares it with the committed document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import run5_bundle as B
from . import run5_preregistration as PR

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
CAMPAIGN = PACKAGE / "campaign_runs" / "run5_three_seed_10000_v1"
OUTPUT = PACKAGE / "RUN5_DEEP_CAMPAIGN_EVIDENCE.json"
PREREGISTRATION_SHA256 = "2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf"
TRACKED_OFF_GIT = ("*.pt", "event.json", "decisions.jsonl", "metrics.jsonl",
                   "channel_state.json", "logs/", "campaign directory")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict:
    prereg = PR.load_sealed()
    assert prereg["sha256"] == PREREGISTRATION_SHA256, "preregistration differs"
    campaign_bytes = (CAMPAIGN / "CAMPAIGN_COMPLETE.json").read_bytes()
    authorization = PACKAGE / "DEEP_TRAINING_AUTHORIZATION.json"
    seeds = {}
    for seed in PR.CONFIG.seed_order:
        seed_dir = CAMPAIGN / f"seed_{seed}"
        complete_bytes = (seed_dir / "SEED_COMPLETE.json").read_bytes()
        checkpoints = {}
        for update in PR.CONFIG.deep_checkpoints:
            bundle = B.verify_bundle(seed_dir / "checkpoints" / B.bundle_name("checkpoint", update))
            checkpoints[str(update)] = {
                "name": bundle.name, "manifest": bundle.manifest,
                "manifest_sha256": bundle.manifest_sha256,
                "committed_marker": (bundle.path / B.COMMITTED).read_text().strip()}
        final = B.verify_bundle(seed_dir / B.bundle_name("final_actor", PR.CONFIG.deep_target_update))
        completion = json.loads(complete_bytes)
        seeds[str(seed)] = {
            "seed_complete": completion, "seed_complete_sha256": B.sha256_bytes(complete_bytes),
            "checkpoints": checkpoints,
            "final_actor": {"name": final.name, "manifest": final.manifest,
                            "manifest_sha256": final.manifest_sha256,
                            "committed_marker": (final.path / B.COMMITTED).read_text().strip(),
                            "tensor_tree_sha256": final.manifest["actor_tree_sha256"],
                            "file_sha256": final.manifest["files"]["actor_state_dict.pt"]["sha256"],
                            "fresh_process_verification": completion["final_actor"]},
            "latest_pointer": json.loads((seed_dir / "checkpoints" / B.LATEST).read_text()),
            "runs": [json.loads(l) for l in (seed_dir / "runs.jsonl").read_text().splitlines()]}
    artifacts = []
    for path in sorted(p for p in CAMPAIGN.rglob("*") if p.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path),
                          "in_git": False})
    return {
        "schema": "splitfusion.run5.deep_campaign_evidence.v1",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "campaign_directory": CAMPAIGN.relative_to(ROOT).as_posix(),
        "authorization": {"path": authorization.relative_to(ROOT).as_posix(),
                          "sha256": _sha(authorization),
                          "content": json.loads(authorization.read_text())},
        "campaign_complete": json.loads(campaign_bytes),
        "campaign_complete_sha256": B.sha256_bytes(campaign_bytes),
        "seeds": seeds,
        "artifacts": {"off_git": True, "not_committed": list(TRACKED_OFF_GIT),
                      "file_count": len(artifacts),
                      "total_bytes": sum(a["bytes"] for a in artifacts), "files": artifacts},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    document = build()
    if args.verify:
        committed = json.loads(OUTPUT.read_text())
        ok = committed == json.loads(json.dumps(document))
        print(json.dumps({"verified_against_untouched_campaign": ok}))
        return 0 if ok else 1
    with OUTPUT.open("x") as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"written": str(OUTPUT), "files": document["artifacts"]["file_count"],
                      "bytes": document["artifacts"]["total_bytes"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
