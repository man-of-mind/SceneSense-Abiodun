"""Part 3 extras: direct paired B-C and B-D intervals; scene-identity check across controllers; what A's non-mode-11
decisions buy on the SAME scene (A vs B rows at the same decision index, where the reward scene is identical; quality is the
grid expectation, timeliness is each controller's own closed-loop outcome).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m rl_agent.splitfusion_run5_mode_value_audit_v1.extras --out <ABS dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_run5_constrained_headroom_v1 import headroom as H
from rl_agent.splitfusion_run5_constrained_headroom_v1 import metrics as MT

from . import compare as CP
from . import evalb as EB


def main(root: Path):
    rng = np.random.default_rng(7)
    out = {}
    for env, var in (("E1", "CORRECTED"), ("E2", "CORRECTED")):
        A, B = CP.load_mine(root, "A", env, var), CP.load_mine(root, "B", env, var)
        C = H.load(CP.HROOT, "BEST_FIXED_S2", env, var, H.EVAL_SEEDS)
        D = H.load(CP.HROOT, "RULE_S2", env, var, H.EVAL_SEEDS)
        for a in (C, D):
            a["eval_seed"], a["profile"] = A["eval_seed"], A["profile"]
        ref_cols = [f"ref_{c}_{f}" for c in "vp" for f in ("eligible", "hits", "inter", "union", "tp", "fp")]
        same_scene = {k: float(np.mean(np.all([np.isclose(A[c], x[c]) for c in ref_cols], 0))) for k, x in (("B", B), ("C", C), ("D", D))}
        res = dict(same_scene_fraction_vs_A=same_scene, paired={})
        for pname in CP.PROFILES + ("ALL",):
            mask = (np.ones(len(A["mcs"]), bool) if pname == "ALL" else A["profile"] == CP.PROFILES.index(pname)).astype(float)
            res["paired"][pname] = {"B_minus_C": CP.paired_ci(C, B, mask, rng), "B_minus_D": CP.paired_ci(D, B, mask, rng)}
        # A's non-11 decisions on identical scenes
        same = np.all([np.isclose(A[c], B[c]) for c in ref_cols], 0)
        non11 = (A["mode"] != 11) & same
        rows = {}
        for pname in CP.PROFILES + ("ALL",):
            pm = non11 & ((A["profile"] == CP.PROFILES.index(pname)) if pname != "ALL" else True)
            n = int(pm.sum())
            if n == 0:
                continue
            modes, cnt = np.unique(A["mode"][pm], return_counts=True)
            s = lambda a, f: float(a[f][pm].sum())
            rows[pname] = dict(n=n, share_of_profile=float(n / ((A["profile"] == CP.PROFILES.index(pname)).sum() if pname != "ALL" else len(A["mode"]))),
                               modes={str(int(m)): int(c) for m, c in zip(modes, cnt)},
                               A_q_p50=float(np.median(A["q"][pm])), B_q11_p50=float(np.median(B["q"][pm])),
                               A_wire_p50_kb=float(np.median(A["wire"][pm]) / 1e3), B_wire_p50_kb=float(np.median(B["wire"][pm]) / 1e3),
                               A_timely=float((A["timely"][pm] > .5).mean()), B_timely=float((B["timely"][pm] > .5).mean()),
                               hits_v=(s(A, "v_hits"), s(B, "v_hits")), hits_p=(s(A, "p_hits"), s(B, "p_hits")),
                               seg_v_iou=(s(A, "v_inter") / s(A, "v_union"), s(B, "v_inter") / s(B, "v_union")),
                               prec_p=(s(A, "p_tp") / (s(A, "p_tp") + s(A, "p_fp")), s(B, "p_tp") / (s(B, "p_tp") + s(B, "p_fp"))),
                               fp_p=(s(A, "p_fp"), s(B, "p_fp")), mcs_p50=float(np.median(A["mcs"][pm])),
                               snr_p50=float(np.median(A["snr"][pm])))
        res["A_non11_vs_B_same_scene"] = rows
        res["A_non11_total"] = int((A["mode"] != 11).sum())
        res["A_non11_with_identical_scene_in_B"] = int(non11.sum())
        out[f"{env}_{var}"] = res
    with open(root / "PART3_EXTRAS.json", "x") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    main(p.parse_args().out)
