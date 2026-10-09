"""POST-HOC descriptive supplement (not part of the registered rule): the training objective and segmentation views
that the registered all-scored retention guardrail does not show.

  realized_C0_reward     mean over included decisions of D * value_C0 (the actors' own training objective)
  seg retention on-time  pooled seg IoU over on-time decisions / the same decisions' reference IoU
  delivered seg          sum(timely * inter) / sum(union), and the reference's sum(inter) / sum(union) (late = nothing delivered)

    python3 -m rl_agent.splitfusion_run5_lowmcs_retrain_v1.supplement --root <ABS experiments/splitfusion_run5_lowmcs_retrain_v1>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

import numpy as np

from . import evaluate as EV

STRATA = dict(all=lambda r: True, **EV.STRATA)


def block(rows):
    inc = [r for r in rows if not r["excluded"]]
    out = dict(n=len(rows), realized_C0_reward=float(np.mean([r["value"]["C0"] if r["timely"] else 0.0 for r in inc])))
    for c in ("v", "p"):
        ti = [r for r in rows if r["timely"]]
        iou = lambda rs, pre="": (sum(r[f"{pre}{c}_inter"] for r in rs) / max(1e-12, sum(r[f"{pre}{c}_union"] for r in rs)))
        out[f"seg_ret_all_{c}"] = iou(rows) / iou(rows, "ref_")
        out[f"seg_ret_on_time_{c}"] = iou(ti) / iou(ti, "ref_")
        out[f"delivered_seg_{c}"] = sum(r["timely"] * r[f"{c}_inter"] for r in rows) / sum(r[f"{c}_union"] for r in rows)
    return out


def run(root: Path) -> dict:
    groups = collections.defaultdict(list)
    with gzip.open(root / "eval_01/EVAL_DECISIONS.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["env"] == "E1":
                groups[(r["policy"], r["variant"])].append(r)
    doc = {f"{p}|{v}": {s: block([r for r in rows if f(r)]) for s, f in STRATA.items()} for (p, v), rows in sorted(groups.items())}
    out = root / "supplement_01"
    out.mkdir(exist_ok=False)
    (out / "SUPPLEMENT.json").write_text(json.dumps(dict(status="POST_HOC_DESCRIPTIVE_NOT_REGISTERED", env="E1",
                                                         results=doc), indent=1, sort_keys=True))
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    d = run(p.parse_args().root)
    for k, v in d.items():
        print(k, {s: (round(x["realized_C0_reward"], 4), round(x["seg_ret_all_v"], 3), round(x["seg_ret_on_time_v"], 3),
                      round(x["delivered_seg_v"], 3), round(x["delivered_seg_p"], 3)) for s, x in v.items() if s in ("all", "mcs_9_10", "mcs_ge16")})
