"""Apply the pre-registered selection rules (experiments/splitfusion_fixed_action_screen_v1/selection_01/
SELECTION_RULES.md) to existing evidence only. Writes SELECTION.json create-only.

Run from the live worktree root so the actor support is read from the actor that will run live.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

MAIN = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
RUN5_WT = MAIN.parent / "abiodun_run5_wt"
PAYLOAD_CSV = MAIN / "experiments/splitfusion_288_offline_rl_dataset_v1/20260909_offline_consolidation_v1/action_72_summary.csv"
CATALOG = WT / "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json"
FIT = RUN5_WT / "rl_agent/splitfusion_hybrid_sac_run5_v1/FIXED_ACTION_SELECTION.json"
CAPTURE_1000 = WT / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261002T224700Z_run5_live_avo_capture_1000_v1/"
                     "capture/cells/a71__favorable_stable/run4_phase6/PHASE6_UE_EVIDENCE.json")
RULES = MAIN / "experiments/splitfusion_fixed_action_screen_v1/selection_01/SELECTION_RULES.md"
ANCHORS = (0, 3000, 5000, 7000, 9000, 9800)
FAMILIES = ("noAE", "AE128", "AE64", "AE32")
SEED = 20261005


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select():
    from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as AC
    from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import ALL_PROFILE_TOTAL_TRANSMITTED_BYTES_SUPPORT as WIRE
    from rl_agent.splitfusion_run5_live_avo_v1.audit_actor import load_registered_actor
    if sha(CATALOG) != AC.CATALOG_SHA256:
        raise ValueError("catalogue differs from the pinned action contract")
    catalog = {p["action_id"]: p for p in json.loads(CATALOG.read_bytes())["profiles"]}
    actor = load_registered_actor()
    lower, upper = actor._lower, actor._upper                 # the bounds act_on_vector enforces live
    modes = {}
    for action_id, p in catalog.items():
        mode = action_id // 6
        modes.setdefault(mode, (p["family"], p["quantizer"]))
    payload = {}
    for row in csv.DictReader(open(PAYLOAD_CSV)):
        values = [float(row[f"live_estimated_wire_bytes_median__{c}"]) for c in
                  ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY") if row.get(f"live_estimated_wire_bytes_median__{c}")]
        payload[int(row["action_id"])] = dict(max_profile_median_wire_bytes=max(values) if values else None,
            favorable_median_wire_bytes=float(row["live_estimated_wire_bytes_median__FAVORABLE_STABLE"]) if row.get("live_estimated_wire_bytes_median__FAVORABLE_STABLE") else None,
            profile_id=row["profile_id"], family=row["family"], quantizer=row["quantizer"], q_e4=int(float(row["q_e4"])))
    table = {}
    for action_id, p in sorted(catalog.items()):
        mode = action_id // 6
        pay = payload.get(action_id, {})
        q_ok = lower[mode] <= p["q_e4"] <= upper[mode]
        w = pay.get("max_profile_median_wire_bytes")
        wire_ok = w is not None and WIRE[0] <= w <= WIRE[1]
        if pay and (pay["profile_id"] != p["profile_id"] or pay["q_e4"] != p["q_e4"]):
            raise ValueError(f"payload table identity differs for action {action_id}")
        table[action_id] = dict(action_id=action_id, profile_id=p["profile_id"], family=p["family"], quantizer=p["quantizer"],
            mode_id=mode, q_e4=p["q_e4"], keep_count=p["keep_count"], wire_bytes=w, favorable_wire_bytes=pay.get("favorable_median_wire_bytes"),
            actor_q_support=[lower[mode], upper[mode]], q_supported=q_ok, wire_supported=wire_ok, supported=q_ok and wire_ok)
    roles = {}
    # rule 1
    rule1 = {}
    for family in FAMILIES:
        candidates = [a for a in table.values() if a["family"] == family and a["supported"]]
        if not candidates:
            rule1[family] = dict(gap="no supported action")
            continue
        best = min(candidates, key=lambda a: (a["wire_bytes"], a["action_id"]))
        rule1[family] = best["action_id"]
        roles.setdefault(best["action_id"], []).append(f"min_payload_{family}_diagnostic_control")
    # rule 2
    frames = json.loads(CAPTURE_1000.read_bytes())["frames"]
    rule2 = {}
    for mode in (5, 8):
        qs = [int(f["action"]["q_e4"]) for f in frames if f.get("action") and int(f["action"]["mode_id"]) == mode
              and f.get("reward_requested") is True]
        entry = dict(decisions=len(qs))
        if len(qs) < 20:
            entry["gap"] = "fewer than 20 decisions"
        else:
            median = float(np.percentile(qs, 50, method="linear"))
            supported = [a for a in table.values() if a["mode_id"] == mode and a["supported"]]
            best = min(supported, key=lambda a: (abs(a["q_e4"] - median), -a["q_e4"]))
            entry.update(median_q_e4=median, selected=best["action_id"])
            roles.setdefault(best["action_id"], []).append(f"mode{mode}_representative")
        rule2[mode] = entry
    # rule 3
    fit = json.loads(FIT.read_bytes())["selected"]
    fid = int(fit["action_id"])
    entry = dict(artifact=str(FIT), artifact_sha256=sha(FIT), recorded=fit)
    cat = table[fid]
    entry["identity_verified"] = (fit.get("profile_id") == cat["profile_id"] and int(fit.get("mode_id", cat["mode_id"])) == cat["mode_id"]
                                  and int(fit.get("q_e4", cat["q_e4"])) == cat["q_e4"])
    if not entry["identity_verified"]:
        raise ValueError("FIT-selected identity does not match the catalogue")
    entry["supported"] = cat["supported"]
    if cat["supported"]:
        roles.setdefault(fid, []).append("fit_selected_baseline")
    rule3 = entry
    selected = sorted(roles)
    if len(selected) > 7:
        raise ValueError("more than seven actions")
    order = list(selected)
    random.Random(SEED).shuffle(order)
    return dict(schema="splitfusion.fixed_action_screen.selection.v1", rules_sha256=sha(RULES),
        sources=dict(catalog=[str(CATALOG), sha(CATALOG)], payload=[str(PAYLOAD_CSV), sha(PAYLOAD_CSV)],
                     fit=[str(FIT), sha(FIT)], capture_1000=[str(CAPTURE_1000), sha(CAPTURE_1000)]),
        actor_boundary_sha256=actor.boundary_sha256, wire_support=list(WIRE), rule1=rule1, rule2=rule2, rule3=rule3,
        roles={str(k): v for k, v in roles.items()}, selected=[table[a] for a in selected], arm_order=order,
        arm_order_seed=SEED, catalogue=list(table.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    doc = select()
    out = Path(args.out)
    with open(out, "x") as stream:
        json.dump(doc, stream, indent=1, sort_keys=True)
    print(json.dumps({k: doc[k] for k in ("rule1", "rule2", "roles", "arm_order")}, indent=1, default=str))
    for a in doc["selected"]:
        print(a)
