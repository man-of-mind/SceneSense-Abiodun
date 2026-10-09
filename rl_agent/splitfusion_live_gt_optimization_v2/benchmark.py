"""Two counter-ordered full45-frame replays of control and three ablations."""
from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from . import profile_saved as P
from . import candidate as C

ARMS = ("control", "metadata_only", "labels_only", "combined")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = {str(path.relative_to(P.ROOT)): P.sha(path) for path in
                    sorted(Path(__file__).parent.glob("*.py")) if not path.name.startswith("test_")}
    P.write(args.output / "DESIGN.json", {"arms": ARMS, "rounds": 2,
        "order": [list(ARMS), list(reversed(ARMS))], "sources": source_files,
        "selection": "Report all ablations; no live authority or170ms claim.",
        "primary": "paired full45-frame sequences; through-ready and through-archive wall and stageCPU",
        "acceptance": "Every AVO result/native input/target PNG/table/hash/archive byte exact; no altered validation/IO semantics."})
    original_scope = P.V1.install_cpu_gt_optimizations
    before = P.R.tree(P.CELL / "parity_complete_targets")
    reference, cases = P.load_cases()
    rows = []
    for repeat in range(2):
        for arm in (ARMS if repeat == 0 else tuple(reversed(ARMS))):
            @contextmanager
            def scope(*, expected_sources):
                with original_scope(expected_sources=expected_sources) as state:
                    if arm == "control":
                        yield state
                    else:
                        with C.install(fast_labels=arm in ("labels_only", "combined"),
                                       fast_metadata=arm in ("metadata_only", "combined")) as addition:
                            yield {"v1": state, "v2": addition}
            with patch.object(P.V1, "install_cpu_gt_optimizations", scope):
                with tempfile.TemporaryDirectory(prefix="gt-v2-paired-") as temporary:
                    measured, _ = P.run_pass(reference, cases, Path(temporary))
            P.write(args.output / f"round_{repeat}_{arm}.json", measured)
            rows.extend(dict(row, repeat=repeat, arm=arm) for row in measured)
            print(json.dumps({"round": repeat, "arm": arm, "frames": len(measured), "all_exact": True}), flush=True)
    assert not torch.cuda.is_initialized()
    assert before == P.R.tree(P.CELL / "parity_complete_targets")
    assert all(P.sha(P.ROOT / path) == digest for path, digest in source_files.items())
    result = {"status": "ALL_ABLATIONS_EXACT", "unique_frames": 45, "rounds": 2,
        "frames_per_arm": 90, "total_replays": len(rows), "rows": rows,
        "input_evidence_unchanged": True, "cuda_initialized": False, "live_started": False,
        "source_sha256": source_files, "summary": {}, "paired_savings": {},
        "limits": ["Not a live performance prediction or deadline qualification.",
                   "Repeated scenes are not independent live observations.",
                   "AVO+source+target+durable publish+archive replay excludes base-object collector, queue and edge/radio.",
                   "No original or sealed source/evidence changed; all durability writes remain synchronous."]}
    keyed = {(r["repeat"], r["arm"], r["frame_id"]): r for r in rows}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        result["summary"][arm] = {key: P.quantiles([row[key] for row in selected])
            for key in ("through_ready_ms", "through_archive_ms")}
        result["summary"][arm]["stages"] = {stage: {key: P.quantiles([r["timings"][stage][key] for r in selected])
            for key in ("wall_ms", "thread_cpu_ms")} for stage in P.STAGES}
        if arm != "control":
            result["paired_savings"][arm] = {key: P.quantiles([
                keyed[(r["repeat"], "control", r["frame_id"])][key] - r[key] for r in selected])
                for key in ("through_ready_ms", "through_archive_ms")}
    P.write(args.output / "RESULT.json", result)
    P.write(args.output / "OUTPUT_SHA256.json", P.R.tree(args.output))
    print(json.dumps({"status": result["status"], "paired_savings": result["paired_savings"]}, indent=2))


if __name__ == "__main__":
    main()
