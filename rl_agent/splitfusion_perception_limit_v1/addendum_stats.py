"""Addendum-01 statistics: tie-aware Spearman (average ranks) for quoted latency correlations, and the radial share
of squared matched error on the original validation (complements the 'radial-dominant proportion')."""
import csv, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(".").resolve()
D = ROOT / "experiments/splitfusion_perception_latency_retention_v1/20261006_v1"
rows = list(csv.DictReader((D / "latency/DECISION_TIMELINE.csv").open()))
f = lambda k: np.array([float(r[k]) for r in rows])
pay = f("payload_bytes")
out = {"timeline_sha256": hashlib.sha256((D / "latency/DECISION_TIMELINE.csv").read_bytes()).hexdigest(), "spearman_tie_aware": {}}
for k in ("receipt", "EP", "EG", "post_join", "P3_ue_encode", "P5_uplink_to_reassembly", "G7_numeric_gt_compute"):
    rho, p = spearmanr(pay, f(k))
    out["spearman_tie_aware"][f"payload_vs_{k}"] = {"rho": float(rho), "p": float(p), "n": len(rows)}
rho, p = spearmanr(f("G7_numeric_gt_compute"), f("actor_boxes"))
out["spearman_tie_aware"]["G7_numeric_gt_compute_vs_actor_boxes"] = {"rho": float(rho), "p": float(p), "n": len(rows)}
out["ties"] = {"payload_unique": int(len(np.unique(pay))), "actor_boxes_unique": int(len(np.unique(f("actor_boxes"))))}
# radial share of squared error (original validation), reusing the original_radial_check matching
spec = importlib.util.spec_from_file_location("orc", ROOT / "rl_agent/splitfusion_perception_limit_v1/original_radial_check.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
captured = {}
orig_dumps = json.dumps
def grab(main_rows):
    pass
src = (ROOT / "rl_agent/splitfusion_perception_limit_v1/original_radial_check.py").read_text()
ns = {"__name__": "x", "__file__": str(ROOT / "rl_agent/splitfusion_perception_limit_v1/original_radial_check.py")}
src = src.replace('Path(out_path).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))', 'return rows')
exec(compile(src, "orc", "exec"), ns)
res = ns["main"](None)
tol = {"vehicle": 1.0, "person": 1.2}
out["radial_share_of_squared_error"] = {}
for cls, rs in res.items():
    a = np.array(rs)
    rad2 = a[:, 2] ** 2; lat2 = a[:, 3] ** 2
    b = a[:, 1] > tol[cls]
    out["radial_share_of_squared_error"][cls] = {
        "all_matched": float(rad2.sum() / (rad2.sum() + lat2.sum())),
        "beyond_tol_matched": float(rad2[b].sum() / (rad2[b].sum() + lat2[b].sum())),
        "n_all": int(len(a)), "n_beyond": int(b.sum())}
out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
Path(sys.argv[1]).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
