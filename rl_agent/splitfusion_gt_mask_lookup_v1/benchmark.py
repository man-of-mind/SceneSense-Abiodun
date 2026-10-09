"""Paired timing of the original vs lookup-table mask label check on the retained valid_02 masks (one pinned core).

Run: env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 5 python3 -m rl_agent.splitfusion_gt_mask_lookup_v1.benchmark <out.json>
"""
import json
import sys
import time

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C

from .test_equivalence import ARCHIVE, original_mask


def main(out):
    pairs = [((p.parent / "segmentation.u8").read_bytes(), (p.parent / "object_ignore.u8").read_bytes())
             for p in sorted(ARCHIVE.glob("frame_*/scorer/segmentation.u8"))]
    result = {}
    for repeat in range(3):                                   # alternate order to control drift
        for name, function in ((("original", original_mask), ("lookup", C._mask)) if repeat % 2 == 0
                               else (("lookup", C._mask), ("original", original_mask))):
            times = []
            for seg, ignore in pairs:
                start = time.perf_counter_ns()
                function(seg, {0, 1, 2, 255}); function(ignore, {0, 255})
                times.append((time.perf_counter_ns() - start) / 1e6)
            result.setdefault(name, []).extend(times)
    summary = {name: dict(n=len(v), p50=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(np.max(v)))
               for name, v in result.items()}
    paired = np.array(result["original"]) - np.array(result["lookup"])
    summary["paired_saving_per_admission_ms"] = dict(p50=float(np.median(paired)), p10=float(np.percentile(paired, 10)),
                                                     min=float(paired.min()))
    summary.update(frames=len(pairs), masks_per_admission=2, mask_bytes=C.MASK_BYTES, note="per target admission (2 masks)")
    open(out, "w").write(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
