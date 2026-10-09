"""Offline quality/reward review and bounded-pilot evidence for the seven measured split actions.

Read-only. Inputs are hash-recorded; nothing is trained, launched, approved or modified. Two evidence sources:

* the seven-arm live screen (``results_combined_01/DECISIONS.jsonl``): per-object matched errors, eligibility,
  TP/FN/FP, segmentation counts and receipt time for every decision;
* the exact offline quality grid (``quality_rows.sqlite3``): the same per-object statistics for every FIT and
  held-scene frame under every (mode, q) anchor, without latency.

Quantities (reference tolerances vehicle 1.0 m, person 1.2 m; class weights person 0.6 / vehicle 0.4, as in the
approved Q_perc spec):

* ``R_tol,c``  = eligible class-c objects matched within tolerance / eligible class-c objects;
* ``Y``        = class-weighted arithmetic mean of R_tol,c over classes with eligible objects (undefined if none);
* ``OTR``      = 1[receipt <= 170 ms] * Y  -- on-time within-tolerance localisation, the proposed objective;
* ``CUR``      = the unchanged Run-4 reward: Q_perc - 0.25 L/170 on success, -1 on timeout.

Run from ``abiodun/``:  env -u PYTHONPATH python3 -m rl_agent.splitfusion_quality_pilot_design_v1.analysis OUT.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DECISIONS = "experiments/splitfusion_fixed_action_screen_v2/results_combined_01/DECISIONS.jsonl"
GRID = "experiments/splitfusion_hybrid_sac_quality_grid_v1/20260918_exact_continuous_q_grid_a1b_full/quality_rows.sqlite3"
P40 = "experiments/splitfusion_hybrid_sac_p40_sidecar_v1/20260921_exact_grid_corrected_p40_v1/corrected_p40_sidecar.json"
SEVEN_ARM_SEAL = "experiments/splitfusion_corrected_latency_env_v1/20261005_screen_v2_seven_arms_assessment/SEAL.json"

ACTIONS = {65: (10, 9800), 53: (8, 9800), 35: (5, 9800), 17: (2, 9800), 67: (11, 3000), 50: (8, 5000), 33: (5, 7000)}
ORDER = (65, 53, 35, 17, 67, 50, 33)                       # by payload
TOL = {"vehicle": 1.0, "person": 1.2}
WEIGHT = {"person": 0.6, "vehicle": 0.4}
DEADLINE_MS = 170.0
GATE_M = 3.0
BOOT = 4000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ metric definitions
def tol_hits(errors, cls):
    return sum(1 for e in errors if e <= TOL[cls])


def y_value(eligible: dict, errors: dict):
    """Class-weighted arithmetic within-tolerance recall; None when no class has eligible objects."""
    num = den = 0.0
    for cls in TOL:
        if eligible[cls] > 0:
            num += WEIGHT[cls] * tol_hits(errors[cls], cls) / eligible[cls]
            den += WEIGHT[cls]
    return None if den == 0 else num / den


def screen_y(row):
    return y_value({c: row[f"{c}_eligible_gt"] for c in TOL}, {c: row[f"{c}_matched_xy_errors_m"] for c in TOL})


def grid_y(row):
    return y_value({c: row[f"loc_{c}_eligible_gt"] for c in TOL}, {c: row[f"loc_{c}_matched_xy_errors_m"] for c in TOL})


def otr(row, shift_ms=0.0):
    y = screen_y(row)
    return None if y is None else (y if row["ue_receipt"] + shift_ms <= DEADLINE_MS else 0.0)


def current_reward(row, shift_ms=0.0):
    if not row["ue_receipt"] + shift_ms <= DEADLINE_MS:
        return -1.0
    if row["q_perc"] is None:
        return None                                       # timely undefined -> EVALUATOR_FAULT, excluded
    return row["q_perc"] - 0.25 * (row["ue_receipt"] + shift_ms) / DEADLINE_MS


def boot_ci(values, seed=0, n=BOOT):
    v = np.asarray(values, float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n, v.size))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def pct(values, q):
    return None if not values else float(np.percentile(values, q))


# ------------------------------------------------------------------ screen (live) analyses
def load_screen():
    rows = [json.loads(line) for line in (ROOT / DECISIONS).read_text(encoding="utf-8").splitlines()]
    for x in rows:                                         # bind action -> (mode, q) from the per-decision record
        ea = x["executed_action"]
        assert ACTIONS[ea["action_id"]] == (ea["mode_id"], ea["q_e4"]), ea
        assert (x["ue_receipt"] <= DEADLINE_MS) == x["timely"], x["frame_id"]
    return rows


def recompute_q_perc(rows):
    """Recompute every recorded Q_perc with the pinned approved spec and formula; must be exact."""
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import scientific_basis as SB
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
    spec = Q.load_reward_spec(ROOT / SB.QUALITY_SOURCE_RELATIVE_PATH, SB.verify_quality_source(ROOT))
    worst, compared, definedness = 0.0, 0, 0
    for x in rows:
        m = {"frame_id": x["frame_id"]}
        for c in TOL:
            for k in ("eligible_gt", "tp", "fn", "matched_xy_errors_m"):
                m[f"loc_{c}_{k}"] = x[f"{c}_{k}"]
            for k in ("gt", "pred", "intersection", "union"):
                m[f"seg_{c}_{k}_pixels"] = x[f"seg_{c}_{k}_pixels"]
        v = Q.evaluate_exact_quality(spec, m)
        q = None if v is None else v.q_perc
        if (q is None) != (x["q_perc"] is None):
            definedness += 1
        elif q is not None:
            compared += 1
            worst = max(worst, abs(q - x["q_perc"]))
    return dict(spec=spec.to_canonical_dict(), compared=compared, definedness_mismatches=definedness,
                max_abs_difference=worst)


def class_table(rows):
    out = {}
    for a in ORDER:
        arm = [x for x in rows if x["executed_action"]["action_id"] == a]
        out[a] = {}
        for label, sub in (("all", arm), ("timely", [x for x in arm if x["timely"]])):
            out[a][label] = {"decisions": len(sub)}
            for c in TOL:
                el = sum(x[f"{c}_eligible_gt"] for x in sub)
                errs = [e for x in sub for e in x[f"{c}_matched_xy_errors_m"]]
                hit = tol_hits(errs, c)
                out[a][label][c] = dict(
                    frames_with_eligible=sum(1 for x in sub if x[f"{c}_eligible_gt"] > 0),
                    eligible=el, matched_within_gate=len(errs), matched_within_tol=hit,
                    recall_gate=None if el == 0 else len(errs) / el,
                    recall_tol=None if el == 0 else hit / el,
                    matched_beyond_tol_fraction=None if not errs else (len(errs) - hit) / len(errs),
                    error_p50=pct(errs, 50), error_p90=pct(errs, 90), error_max=pct(errs, 100),
                    fp_per_decision=None if not sub else sum(x[f"{c}_fp"] for x in sub) / len(sub),
                    frames_with_eligible_and_zero_tp=sum(1 for x in sub if x[f"{c}_eligible_gt"] > 0 and x[f"{c}_tp"] == 0))
    return out


def utility_compression(table):
    """Pooled per-class U = sqrt(R_gate * exp(-median/tau)) next to R_tol, showing how the form compresses gaps."""
    out = {}
    for a in ORDER:
        out[a] = {}
        for c in TOL:
            t = table[a]["all"][c]
            u = None if not t["recall_gate"] else math.sqrt(t["recall_gate"] * math.exp(-t["error_p50"] / TOL[c]))
            out[a][c] = dict(pooled_utility=u, recall_tol=t["recall_tol"])
    return out


def concealment(rows):
    out = {}
    for c in TOL:
        n = med_hides_tail = med_ok_low_rtol = med_ok_far_tail = 0
        for x in rows:
            el, errs = x[f"{c}_eligible_gt"], x[f"{c}_matched_xy_errors_m"]
            if el == 0 or not errs:
                continue
            n += 1
            med = statistics.median(errs)
            if med <= TOL[c] and max(errs) > TOL[c]:
                med_hides_tail += 1
            if med <= TOL[c] and tol_hits(errs, c) / el < 0.5:
                med_ok_low_rtol += 1
            if med <= TOL[c] and max(errs) > 2 * TOL[c]:
                med_ok_far_tail += 1
        out[c] = dict(class_frames_with_a_match=n, median_within_tol_but_a_match_beyond_tol=med_hides_tail,
                      median_within_tol_but_recall_tol_below_half=med_ok_low_rtol,
                      median_within_tol_but_a_match_beyond_twice_tol=med_ok_far_tail)
    defined = [x for x in rows if x["q_perc"] is not None]
    zero = [x for x in defined if x["q_perc"] == 0.0]
    person_zero = [x for x in zero if x["person_eligible_gt"] > 0 and x["person_tp"] == 0]
    out["geometric_zero_forcing"] = dict(
        defined=len(defined), q_perc_zero=len(zero), zero_because_person_tp_zero=len(person_zero),
        of_which_vehicle_matched=sum(1 for x in person_zero if x["vehicle_eligible_gt"] > 0 and x["vehicle_tp"] > 0),
        per_action={a: dict(defined=sum(1 for x in defined if x["executed_action"]["action_id"] == a),
                            zero=sum(1 for x in zero if x["executed_action"]["action_id"] == a))
                    for a in ORDER})
    qs, rt, rg = [], [], []
    for x in defined:
        el = sum(x[f"{c}_eligible_gt"] for c in TOL)
        if el:
            qs.append(x["q_perc"])
            rt.append(sum(tol_hits(x[f"{c}_matched_xy_errors_m"], c) for c in TOL) / el)
            rg.append(sum(x[f"{c}_tp"] for c in TOL) / el)
    from scipy.stats import spearmanr
    out["spearman_q_perc_vs_object_recall_tol"] = float(spearmanr(qs, rt)[0])
    out["spearman_q_perc_vs_object_recall_gate"] = float(spearmanr(qs, rg)[0])
    out["spearman_n"] = len(qs)
    return out


def incentive(rows):
    out = {}
    for a in ORDER:
        arm = [x for x in rows if x["executed_action"]["action_id"] == a]
        cur = [v for v in map(current_reward, arm) if v is not None]
        ot = [v for v in map(otr, arm) if v is not None]
        succ = [x for x in arm if x["timely"] and x["q_perc"] is not None]
        k = sum(x["timely"] for x in arm)
        out[a] = dict(decisions=len(arm), timely=k, timely_wilson=wilson(k, len(arm)),
                      current_reward_mean=float(np.mean(cur)), current_reward_ci=boot_ci(cur, a), current_reward_n=len(cur),
                      otr_mean=float(np.mean(ot)), otr_ci=boot_ci(ot, a + 1), otr_n=len(ot),
                      q_perc_mean_timely=float(np.mean([x["q_perc"] for x in succ])),
                      receipt_mean_timely_ms=float(np.mean([x["ue_receipt"] for x in succ])))
    return out


def objective_variants(rows):
    """Ranking robustness: recall vs F1 at tolerance; arithmetic vs geometric combiner; vehicle only."""
    def per(x, kind):
        d = {}
        for c in TOL:
            el = x[f"{c}_eligible_gt"]
            if el:
                hit = tol_hits(x[f"{c}_matched_xy_errors_m"], c)
                d[c] = hit / el if kind == "recall" else 2 * hit / (el + x[f"{c}_tp"] + x[f"{c}_fp"])
        return d

    def combine(d, how):
        if not d:
            return None
        if how == "arithmetic":
            return sum(WEIGHT[c] * v for c, v in d.items()) / sum(WEIGHT[c] for c in d)
        if how == "geometric":
            if any(v == 0 for v in d.values()):
                return 0.0
            return math.exp(sum(WEIGHT[c] * math.log(v) for c, v in d.items()) / sum(WEIGHT[c] for c in d))
        return d.get("vehicle")

    out = {}
    for kind in ("recall", "f1"):
        for how in ("arithmetic", "geometric", "vehicle_only"):
            vals = {}
            for a in ORDER:
                v = []
                for x in rows:
                    if x["executed_action"]["action_id"] != a:
                        continue
                    y = combine(per(x, kind), how)
                    if y is not None:
                        v.append(y if x["timely"] else 0.0)
                vals[a] = float(np.mean(v))
            out[f"{kind}_{how}"] = dict(values=vals, ranking=sorted(vals, key=vals.get, reverse=True))
    return out


def paired_by_route_tick(rows):
    """Ego-position pairing: decisions at the same route tick share the identical ego pose in every arm; other
    actors are not guaranteed identical, so the pairing is partial (reported), not an exact counterfactual."""
    by = collections.defaultdict(dict)
    for x in rows:
        by[int(x["route_tick"])][x["executed_action"]["action_id"]] = x
    shared = [arms for arms in by.values() if len(arms) > 1]
    ego_identical = sum(1 for arms in shared
                        if max(math.hypot(v["ego_world"][0] - w["ego_world"][0], v["ego_world"][1] - w["ego_world"][1])
                               for v in arms.values() for w in arms.values()) <= 0.01)
    eligible_identical = sum(1 for arms in shared
                             if len({(v["vehicle_eligible_gt"], v["person_eligible_gt"]) for v in arms.values()}) == 1)
    out = dict(shared_ticks=len(shared), ego_pose_identical_ticks=ego_identical,
               eligible_counts_identical_ticks=eligible_identical,
               note="iid bootstrap over paired ticks; adjacent ticks are ~4 route ticks apart and not independent",
               comparisons={})
    for b in ORDER:
        if b == 67:
            continue
        res = {}
        for name, f in (("otr", otr), ("current_reward", current_reward)):
            d = [f(by[t][67]) - f(by[t][b]) for t in sorted(by)
                 if 67 in by[t] and b in by[t] and f(by[t][67]) is not None and f(by[t][b]) is not None]
            res[name] = dict(n=len(d), mean_difference=float(np.mean(d)), ci=boot_ci(d, b))
        out["comparisons"][f"67_minus_{b}"] = res
    return out


def deadline_shift(rows):
    out = {}
    for a in ORDER:
        arm = [x for x in rows if x["executed_action"]["action_id"] == a]
        lat = np.array([x["ue_receipt"] for x in arm])
        out[a] = dict(receipts_within_5ms_of_deadline=int(np.sum(np.abs(lat - DEADLINE_MS) <= 5)), n=len(arm),
                      by_shift_ms={s: dict(timely_rate=float(np.mean(lat + s <= DEADLINE_MS)),
                                           otr_mean=float(np.mean([v for v in (otr(x, s) for x in arm) if v is not None])))
                                   for s in (-10, -5, 0, 5, 10)})
    return out


def within_arm_drift_and_persistence(rows):
    out = {}
    for a in ORDER:
        s = sorted((x for x in rows if x["executed_action"]["action_id"] == a), key=lambda x: x["decision_index"])
        n = int(round(len(s) * 2 / 3))
        pairs = [(p["timely"], q["timely"]) for p, q in zip(s, s[1:]) if q["decision_index"] == p["decision_index"] + 1]
        after_t = [q for p, q in pairs if p]
        after_l = [q for p, q in pairs if not p]
        out[a] = dict(first_two_thirds=dict(timely=sum(x["timely"] for x in s[:n]), n=n,
                                            receipt_p50=float(np.median([x["ue_receipt"] for x in s[:n]]))),
                      last_third=dict(timely=sum(x["timely"] for x in s[n:]), n=len(s) - n,
                                      receipt_p50=float(np.median([x["ue_receipt"] for x in s[n:]]))),
                      timely_after_timely=None if not after_t else [sum(after_t), len(after_t)],
                      timely_after_late=None if not after_l else [sum(after_l), len(after_l)])
    cad = {}
    for a in ORDER:
        ts = sorted(x["capture_timestamp_ns"] for x in rows if x["executed_action"]["action_id"] == a)
        d = np.diff(ts) / 1e6
        cad[a] = [float(v) for v in np.percentile(d, [5, 50, 95])]
    return dict(per_action=out, inter_decision_ms_p5_p50_p95=cad)


# ------------------------------------------------------------------ grid (offline, no latency) analyses
def load_grid():
    p40 = {f"{r['episode_id']}|{r['sample_id']}|{r['frame_id']}": float(r["corrected_p40"])
           for r in json.loads((ROOT / P40).read_text(encoding="utf-8"))["records"]}
    con = sqlite3.connect(f"file:{ROOT / GRID}?mode=ro&immutable=1", uri=True)
    scenes = collections.defaultdict(dict)
    try:
        for split, mode, q, blob in con.execute("SELECT grid_split, mode_id, q_e4, row_json FROM quality_rows"):
            row = json.loads(blob)
            scenes[(split, f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}")][(mode, q)] = row
    finally:
        con.close()
    return scenes, p40


def descriptors(scenes, p40, key):
    any_row = next(iter(scenes[key].values()))
    return any_row.get("camera_si"), p40.get(key[1])


def coverage(scenes, p40, support):
    def inside(si, p, a):
        s = support[a]
        S = [x["camera_si"] for x in s]
        P = [x["radar_p40"] for x in s]
        return min(S) <= si <= max(S) and min(P) <= p <= max(P)    # binding support_margin_fraction = 0.0
    covered = {}
    for key, grid in scenes.items():
        si, p = descriptors(scenes, p40, key)
        if si is None or p is None or len(grid) != 132:
            continue
        covered[key] = all(inside(si, p, a) for a in ACTIONS)
    return covered


def grid_action_table(scenes, keys):
    out = {}
    for a in ORDER:
        rows = [scenes[k][ACTIONS[a]] for k in keys]
        qd = [r["q_perc"] for r in rows if r["q_perc"] is not None]
        ys = [v for v in map(grid_y, rows) if v is not None]
        entry = dict(scenes=len(rows), q_perc_mean=float(np.mean(qd)), q_perc_zero_fraction=float(np.mean([v == 0 for v in qd])),
                     y_mean=float(np.mean(ys)), payload_kb_mean=float(np.mean([r["total_transmitted_bytes"] for r in rows]) / 1e3))
        for c in TOL:
            el = sum(r[f"loc_{c}_eligible_gt"] for r in rows)
            errs = [e for r in rows for e in r[f"loc_{c}_matched_xy_errors_m"]]
            hit = tol_hits(errs, c)
            entry[c] = dict(eligible=el, recall_gate=len(errs) / el, recall_tol=hit / el,
                            matched_beyond_tol_fraction=(len(errs) - hit) / max(1, len(errs)), error_p90=pct(errs, 90),
                            fp_per_scene=sum(r[f"loc_{c}_fp"] for r in rows) / len(rows))
        out[a] = entry
    return out


def grid_reference_and_frontier(scenes, keys):
    anchors = sorted(next(iter(scenes[keys[0]].values())) and scenes[keys[0]].keys())
    table = []
    for mq in anchors:
        rows = [scenes[k][mq] for k in keys]
        ys = [v for v in map(grid_y, rows) if v is not None]
        rt = {}
        for c in TOL:
            el = sum(r[f"loc_{c}_eligible_gt"] for r in rows)
            rt[c] = sum(tol_hits(r[f"loc_{c}_matched_xy_errors_m"], c) for r in rows) / el
        table.append(dict(mode_id=mq[0], q_e4=mq[1], payload_kb_mean=float(np.mean([r["total_transmitted_bytes"] for r in rows]) / 1e3),
                          y_mean=float(np.mean(ys)), q_perc_mean=float(np.mean([r["q_perc"] for r in rows if r["q_perc"] is not None])),
                          vehicle_recall_tol=rt["vehicle"], person_recall_tol=rt["person"], measured_action=mq in ACTIONS.values()))
    table.sort(key=lambda t: t["payload_kb_mean"])
    frontier, best = [], -1.0
    for t in table:
        if t["y_mean"] > best + 1e-12:
            best = t["y_mean"]
            frontier.append(t)
    return dict(reference_max_vehicle_recall_tol=max(t["vehicle_recall_tol"] for t in table),
                reference_max_person_recall_tol=max(t["person_recall_tol"] for t in table),
                reference_anchor_mode2_q0=[t for t in table if t["mode_id"] == 2 and t["q_e4"] == 0][0],
                pareto_frontier_payload_vs_y=frontier)


def knn_timeliness(chains, si, p, k=5):
    """Same standardized SI/P40 distance and k as the binding's scene conditioning."""
    S = np.array([x["camera_si"] for x in chains])
    P = np.array([x["radar_p40"] for x in chains])
    sd = lambda v: v.std() if v.std() > 0 else 1.0
    nb = [chains[i] for i in np.argsort(np.hypot((S - si) / sd(S), (P - p) / sd(P)), kind="stable")[:k]]
    timely = [x["ue_receipt"] for x in nb if x["timely"]] or [x["ue_receipt"] for x in chains if x["timely"]] or [DEADLINE_MS]
    return float(np.mean([x["timely"] for x in nb])), float(np.mean(timely)) / DEADLINE_MS


def scene_values(scenes, p40, keys, chains, objective, conditioning):
    pooled = {a: (float(np.mean([x["timely"] for x in chains[a]])),
                  float(np.mean([x["ue_receipt"] for x in chains[a] if x["timely"]] or [DEADLINE_MS])) / DEADLINE_MS)
              for a in ACTIONS}
    table = {}
    for k in keys:
        si, p = descriptors(scenes, p40, k)
        v = {}
        for a, mq in ACTIONS.items():
            pt, ell = pooled[a] if conditioning == "pooled" else knn_timeliness(chains[a], si, p)
            row = scenes[k][mq]
            if objective == "otr":
                y = grid_y(row)
                v[a] = None if y is None else pt * y
            else:
                v[a] = None if row["q_perc"] is None else pt * (row["q_perc"] - 0.25 * ell) - (1 - pt)
        if None not in v.values():
            table[k] = v
    return table


def headroom(table):
    means = {a: float(np.mean([v[a] for v in table.values()])) for a in ORDER}
    best = max(means, key=means.get)
    oracle = float(np.mean([max(v.values()) for v in table.values()]))
    picks = collections.Counter(max(v, key=v.get) for v in table.values())
    return dict(n=len(table), fixed=means, best_fixed=best, hindsight_oracle=oracle, oracle_minus_best_fixed=oracle - means[best],
                oracle_choices={a: picks.get(a, 0) for a in ORDER})


def observable_rule(scenes, p40, fit_keys, held_keys, train_chains, eval_chains, objective, conditioning):
    """SI/P40 tertile cells (from FIT), best action per cell chosen on FIT with train chains, evaluated on held scenes
    with eval chains. The environment draws scenes i.i.d., so SI/P40 is the policy's only scene information."""
    F = scene_values(scenes, p40, fit_keys, train_chains, objective, conditioning)
    H = scene_values(scenes, p40, held_keys, eval_chains, objective, conditioning)
    si = np.array([descriptors(scenes, p40, k)[0] for k in fit_keys])
    pp = np.array([descriptors(scenes, p40, k)[1] for k in fit_keys])
    qs, qp = np.quantile(si, [1 / 3, 2 / 3]), np.quantile(pp, [1 / 3, 2 / 3])
    cell = lambda k: (int(np.searchsorted(qs, descriptors(scenes, p40, k)[0])), int(np.searchsorted(qp, descriptors(scenes, p40, k)[1])))
    groups = collections.defaultdict(list)
    for k, v in F.items():
        groups[cell(k)].append(v)
    rule = {c: max(ORDER, key=lambda a: np.mean([v[a] for v in L])) for c, L in groups.items()}
    best = max(ORDER, key=lambda a: np.mean([v[a] for v in F.values()]))
    diff = [v[rule.get(cell(k), best)] - v[best] for k, v in H.items()]
    return dict(train_best_fixed=best, rule_actions=sorted(set(rule.values())), held_n=len(H),
                held_best_fixed_value=float(np.mean([v[best] for v in H.values()])),
                held_rule_minus_fixed=float(np.mean(diff)), ci=boot_ci(diff, 7),
                held_fixed_values={a: float(np.mean([v[a] for v in H.values()])) for a in ORDER})


def chain_splits(support):
    contiguous, interleaved = ({}, {}), ({}, {})
    for a in ACTIONS:
        s = sorted(support[a], key=lambda x: x["decision_index"])
        n = int(round(len(s) * 2 / 3))
        contiguous[0][a], contiguous[1][a] = s[:n], s[n:]
        interleaved[0][a] = [x for i, x in enumerate(s) if (i // 3) % 3 != 2]
        interleaved[1][a] = [x for i, x in enumerate(s) if (i // 3) % 3 == 2]
    return dict(contiguous_2_3=contiguous, interleaved_blocks_of_3=interleaved)


REFERENCE_ANCHOR = (2, 0)                                  # noAE, q 0: least-compressed anchor on the grid
OTHER_FALLBACKS = {33: 1, 65: 4, 17: 0, 50: 0, 67: 1, 53: 5, 35: 3}   # final_01 TABLE.md fallbacks minus 2 start-up


def acceptance_snapshot(rows):
    """Per-arm values of the proposed acceptance quantities (live screen, warm decisions)."""
    out = {}
    for a in ORDER:
        arm = [x for x in rows if x["executed_action"]["action_id"] == a]
        timely = [x for x in arm if x["timely"]]
        entry = dict(timely=len(timely), decisions=len(arm), timely_wilson=wilson(len(timely), len(arm)),
                     other_fallbacks=OTHER_FALLBACKS[a])
        for c in TOL:
            per = [(x[f"{c}_eligible_gt"], tol_hits(x[f"{c}_matched_xy_errors_m"], c), len(x[f"{c}_matched_xy_errors_m"]),
                    x[f"{c}_fp"]) for x in timely]
            el = sum(p[0] for p in per)
            ci = None
            if el:
                rng = np.random.default_rng(11)
                arr = np.array(per, float)
                draws = arr[rng.integers(0, len(arr), (BOOT, len(arr)))]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = draws[:, :, 1].sum(1) / draws[:, :, 0].sum(1)
                ci = [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))]
            matched = sum(p[2] for p in per)
            entry[c] = dict(eligible_timely=el, recall_tol_timely=None if not el else sum(p[1] for p in per) / el,
                            recall_tol_timely_decision_cluster_ci=ci,
                            beyond_tol_share_of_matches_timely=None if not matched else (matched - sum(p[1] for p in per)) / matched,
                            fp_per_timely_decision=None if not timely else sum(p[3] for p in per) / len(timely))
        ot = [v for v in map(otr, arm) if v is not None]
        entry["otr_mean_admitted"] = float(np.mean(ot))
        entry["otr_per_opportunity_approx"] = float(np.mean(ot)) * len(arm) / (len(arm) + OTHER_FALLBACKS[a])
        out[a] = entry
    return out


def reference_on(scenes, keys):
    rows = [scenes[k][REFERENCE_ANCHOR] for k in keys]
    out = dict(anchor=dict(mode_id=REFERENCE_ANCHOR[0], q_e4=REFERENCE_ANCHOR[1]),
               y_mean=float(np.mean([v for v in map(grid_y, rows) if v is not None])))
    for c in TOL:
        el = sum(r[f"loc_{c}_eligible_gt"] for r in rows)
        errs = [e for r in rows for e in r[f"loc_{c}_matched_xy_errors_m"]]
        out[c] = dict(recall_tol=tol_hits(errs, c) / el, eligible=el,
                      beyond_tol_share_of_matches=(len(errs) - tol_hits(errs, c)) / max(1, len(errs)),
                      fp_per_scene=sum(r[f"loc_{c}_fp"] for r in rows) / len(rows))
    return out


def main(out_path: Path):
    screen = load_screen()
    warm = [x for x in screen if not x["cold"]]
    support = {a: [x for x in warm if x["executed_action"]["action_id"] == a] for a in ACTIONS}
    scenes, p40 = load_grid()
    cov = coverage(scenes, p40, support)
    fit_keys = sorted(k for k, v in cov.items() if v and k[0] == "fit")
    held_keys = sorted(k for k, v in cov.items() if v and k[0] == "held_scene")
    table = class_table(warm)
    result = dict(
        schema="splitfusion.quality_pilot_design.analysis.v1",
        inputs={p: sha256(ROOT / p) for p in (DECISIONS, GRID, P40, SEVEN_ARM_SEAL)},
        action_map={a: dict(mode_id=m, q_e4=q) for a, (m, q) in ACTIONS.items()},
        definitions=dict(tolerance_m=TOL, class_weights=WEIGHT, deadline_ms=DEADLINE_MS, match_gate_m=GATE_M,
                         y="class-weighted arithmetic mean of within-tolerance recall over classes with eligible GT",
                         otr="1[receipt<=170 ms] * Y", current_reward="Q_perc - 0.25 L/170 on success, -1 timeout"),
        q_perc_recomputation=recompute_q_perc(screen),
        screen=dict(warm_decisions=len(warm), class_table=table, utility_compression=utility_compression(table),
                    concealment=concealment(warm), incentive=incentive(warm), objective_variants=objective_variants(warm),
                    paired_by_route_tick=paired_by_route_tick(warm), deadline_shift=deadline_shift(warm),
                    drift_persistence_cadence=within_arm_drift_and_persistence(warm),
                    acceptance_snapshot=acceptance_snapshot(warm)),
        grid=dict(covered_fit=len(fit_keys), covered_held=len(held_keys),
                  complete_fit=sum(1 for k in cov if k[0] == "fit"), complete_held=sum(1 for k in cov if k[0] == "held_scene"),
                  fit_actions=grid_action_table(scenes, fit_keys), held_actions=grid_action_table(scenes, held_keys),
                  reference_and_frontier=grid_reference_and_frontier(scenes, fit_keys),
                  reference_anchor_fit=reference_on(scenes, fit_keys), reference_anchor_held=reference_on(scenes, held_keys)),
    )
    hr = {}
    for split, keys in (("fit", fit_keys), ("held", held_keys)):
        for obj in ("otr", "current_reward"):
            for cond in ("pooled", "knn5"):
                hr[f"{split}|{obj}|{cond}"] = headroom(scene_values(scenes, p40, keys, support, obj, cond))
    rules = {}
    for obj in ("otr", "current_reward"):
        for cond in ("pooled", "knn5"):
            rules[f"same_chains|{obj}|{cond}"] = observable_rule(scenes, p40, fit_keys, held_keys, support, support, obj, cond)
        for name, (train, ev) in chain_splits(support).items():
            rules[f"{name}|{obj}|knn5"] = observable_rule(scenes, p40, fit_keys, held_keys, train, ev, obj, "knn5")
    result["static_headroom"] = hr
    result["observable_context_rule"] = rules
    text = json.dumps(result, indent=1, sort_keys=True, default=lambda o: str(o))
    out_path.write_text(text + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main(Path(sys.argv[1]))
