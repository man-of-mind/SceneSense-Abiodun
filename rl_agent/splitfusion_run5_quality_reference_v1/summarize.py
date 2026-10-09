"""Paired three-branch quality tables, uncertainty and figures from REPLAY.json (no inference; create-only).

Implements the analysis plan registered in ``experiments/splitfusion_run5_full_live_retention_v1/ACCEPTANCE.json``
(``quality_analysis_plan``) before any frame was captured:
  * per class, pooled over admitted samples: recall within tolerance INCLUDING misses (vehicle 1.0 m, person 1.2 m),
    recall at the 3 m gate, precision, false positives, matched-error P50/P90/P95/max, pooled seg IoU, Q_perc;
  * paired differences agent-full, uint8-full, agent-uint8 with a moving-block bootstrap (block 5 samples in route
    order, 10,000 resamples, seed 20261006, percentile 95% CI) and an iid frame bootstrap alongside;
  * strata: all admitted samples, and samples whose LIVE agent reply was timely (<= 170 ms inclusive). Baselines are
    only stratified by the agent's live outcome; no baseline timeliness is claimed;
  * decision rules: compression-controllable when the CI of (full - agent) excludes 0 in the loss direction;
    otherwise 'not resolved at this sample size'; FULL_LOCAL's own shortfall is model/eligibility-limited;
  * the registered retention-ratio target candidates translated to absolute values (target = ratio x reference).
Figures (PNG): conservative localization CDF, three-branch comparison, paired-difference intervals, live latency CDF,
route coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BR = ("full_local", "noae_uint8_q0", "agent_selected")
LABEL = {"full_local": "Uncompressed (full-local)", "noae_uint8_q0": "noAE UINT8, no drop",
         "agent_selected": "Run-5 agent selection"}
COLOR = {"full_local": "#2a78d6", "noae_uint8_q0": "#eb6834", "agent_selected": "#1baf7a"}   # categorical slots 1-3
STYLE = {"full_local": "-", "noae_uint8_q0": "--", "agent_selected": ":"}
MARK = {"full_local": "o", "noae_uint8_q0": "s", "agent_selected": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOL = {"vehicle": 1.0, "person": 1.2}
PAIRS = (("agent_selected", "full_local"), ("noae_uint8_q0", "full_local"), ("agent_selected", "noae_uint8_q0"))
METRICS = ("recall_within_tol", "recall_3m", "precision", "fp_per_decision", "seg_iou_pooled")
LOSS_SIGN = {"recall_within_tol": 1, "recall_3m": 1, "precision": 1, "seg_iou_pooled": 1, "fp_per_decision": -1}
B, BLOCK, SEED = 10_000, 5, 20261006


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_sample(row, branch, cls):
    m = row[branch]["measurement"]
    errs = [float(e) for e in m[f"loc_{cls}_matched_xy_errors_m"]]
    return dict(eligible=int(m[f"loc_{cls}_eligible_gt"]), tp=int(m[f"loc_{cls}_tp"]), fp=int(m[f"loc_{cls}_fp"]),
                fn=int(m[f"loc_{cls}_fn"]), within=sum(e <= TOL[cls] for e in errs), errors=errs,
                inter=int(m[f"seg_{cls}_intersection_pixels"]), union=int(m[f"seg_{cls}_union_pixels"]),
                gt_px=int(m[f"seg_{cls}_gt_pixels"]))


def arrays(rows, branch, cls):
    s = [per_sample(r, branch, cls) for r in rows]
    return {k: np.array([x[k] for x in s], float) for k in ("eligible", "tp", "fp", "within", "inter", "union")}, s


def ratio(num, den):
    return float(num / den) if den > 0 else float("nan")


def metric_values(a, idx=None):
    """Pooled metrics for a (resampled) index set."""
    t = {k: (v if idx is None else v[idx]).sum(axis=-1) for k, v in a.items()}
    n = a["eligible"].shape[-1] if idx is None else idx.shape[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(recall_within_tol=t["within"] / t["eligible"], recall_3m=t["tp"] / t["eligible"],
                    precision=t["tp"] / (t["tp"] + t["fp"]), fp_per_decision=t["fp"] / n,
                    seg_iou_pooled=t["inter"] / t["union"])


def block_indices(n, rng):
    starts = rng.integers(0, max(1, n - BLOCK + 1), size=(B, int(np.ceil(n / BLOCK))))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(B, -1)[:, :n]
    return np.minimum(idx, n - 1)


def summarize_class(rows, cls):
    out, arr = {}, {}
    for b in BR:
        a, s = arrays(rows, b, cls)
        arr[b] = a
        errs = sorted(e for x in s for e in x["errors"])
        tot = {k: int(v.sum()) for k, v in a.items()}
        mv = {k: float(v) for k, v in metric_values(a).items()}
        defined_iou = [x["inter"] / x["union"] for x in s if x["union"] > 0]
        out[b] = dict(samples=len(s), eligible=tot["eligible"], matched_3m=tot["tp"], misses=tot["eligible"] - tot["tp"],
                      false_positives=tot["fp"], within_tol=tot["within"], **mv,
                      err_p50=float(np.percentile(errs, 50)) if errs else None,
                      err_p90=float(np.percentile(errs, 90)) if errs else None,
                      err_p95=float(np.percentile(errs, 95)) if errs else None, err_max=max(errs) if errs else None,
                      matched_beyond_tol=sum(e > TOL[cls] for e in errs),
                      seg_iou_mean_defined=float(np.mean(defined_iou)) if defined_iou else None,
                      seg_iou_defined_samples=len(defined_iou), samples_with_eligible=int(sum(x["eligible"] > 0 for x in s)),
                      errors=errs)
    n = len(rows)
    diffs = {}
    if n >= 2:
        rng = np.random.default_rng(SEED)
        bidx = block_indices(n, rng)
        iidx = np.random.default_rng(SEED + 1).integers(0, n, size=(B, n))
        for x, y in PAIRS:
            key = f"{x}-{y}"
            diffs[key] = {}
            for metric in METRICS:
                point = out[x][metric] - out[y][metric]
                vals = {}
                for name, idx in (("block", bidx), ("iid", iidx)):
                    d = metric_values(arr[x], idx)[metric] - metric_values(arr[y], idx)[metric]
                    d = d[np.isfinite(d)]
                    vals[name] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d.size else [None, None]
                lo, hi = vals["block"]
                if lo is not None and lo == hi == point == 0:
                    verdict = "identical (zero difference in every resample)"
                elif lo is not None:
                    s_ = LOSS_SIGN[metric]
                    if (s_ > 0 and hi < 0) or (s_ < 0 and lo > 0):
                        verdict = f"{x} WORSE than {y} (CI excludes 0)"
                    elif (s_ > 0 and lo > 0) or (s_ < 0 and hi < 0):
                        verdict = f"{x} BETTER than {y} (CI excludes 0)"
                    else:
                        verdict = "not resolved at this sample size"
                else:
                    verdict = "undefined"
                diffs[key][metric] = dict(point=point, ci_block=vals["block"], ci_iid=vals["iid"], verdict=verdict)
    return out, diffs


def q_summary(rows):
    out = {}
    for b in BR:
        q = [r[b]["q_perc"] for r in rows]
        d = [v for v in q if v is not None]
        out[b] = dict(defined=len(d), undefined=len(q) - len(d), mean=float(np.mean(d)) if d else None)
    return out


def analyze(rows):
    res = {cls: dict(zip(("branches", "paired"), summarize_class(rows, cls))) for cls in TOL}
    res["q_perc"] = q_summary(rows)
    res["samples"] = len(rows)
    return res


def targets(reference, acceptance):
    cand = acceptance["quality_analysis_plan"]["reward_trial_quality_retention_target_candidates"]
    keymap = dict(vehicle_recall_1p0m=("vehicle", "recall_within_tol"), person_recall_1p2m=("person", "recall_within_tol"),
                  vehicle_seg_iou=("vehicle", "seg_iou_pooled"), person_seg_iou=("person", "seg_iou_pooled"),
                  vehicle_precision=("vehicle", "precision"), person_precision=("person", "precision"))
    out = {}
    for trial in ("trial_1_moderate", "trial_2_strict"):
        out[trial] = {}
        for key, ratio_ in cand[trial].items():
            cls, metric = keymap[key]
            ref = reference[cls]["branches"]["full_local"][metric]
            agent = reference[cls]["branches"]["agent_selected"][metric]
            out[trial][key] = dict(ratio=ratio_, reference=ref, target_abs=min(ratio_ * ref, ref),
                                   current_agent=agent, current_agent_retention=agent / ref if ref else None)
    return out


# ------------------------------------------------------------------ figures
def _style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)


def fig_cdf(res, path, title_suffix):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, cls in zip(axes, TOL):
        for b in BR:
            r = res[cls]["branches"][b]
            errs = np.array(r["errors"])
            x = np.linspace(0, 3.0, 601)
            y = [(errs <= v).sum() / r["eligible"] for v in x] if r["eligible"] else [0] * len(x)
            ax.plot(x, y, STYLE[b], color=COLOR[b], lw=2,
                    label=f"{LABEL[b]}: {r['within_tol']}/{r['eligible']} = {r['recall_within_tol']:.3f} within {TOL[cls]} m")
        ax.axvline(TOL[cls], color=INK2, lw=1, ls="-.")
        ax.text(TOL[cls] + 0.03, 0.03, f"{TOL[cls]} m tolerance", color=INK2, fontsize=9)
        ax.set_xlim(0, 3.0); ax.set_ylim(0, 1.0)
        ax.set_xlabel("localization error (m)", color=INK)
        ax.set_ylabel("fraction of ALL eligible GT localized within x", color=INK)
        ax.set_title(f"{cls}: conservative CDF (misses never reach 1)", color=INK, fontsize=11)
        _style(ax)
        ax.legend(fontsize=8.5, loc="lower right", frameon=True, facecolor="white", edgecolor=GRID, framealpha=1, labelcolor=INK)
    fig.suptitle(f"Conservative localization CDF, same retained inputs and GT {title_suffix}", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_compare(res_all, res_timely, path):
    items = [("vehicle", "recall_within_tol", "vehicle recall ≤1.0 m"), ("person", "recall_within_tol", "person recall ≤1.2 m"),
             ("vehicle", "precision", "vehicle precision"), ("person", "precision", "person precision"),
             ("vehicle", "seg_iou_pooled", "vehicle seg IoU"), ("person", "seg_iou_pooled", "person seg IoU")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, (res, name) in zip(axes, ((res_all, "all admitted samples"), (res_timely, "agent reply timely (live)"))):
        y = np.arange(len(items))
        for k, b in enumerate(BR):
            vals = [res[c]["branches"][b][m] for c, m, _ in items]
            ax.scatter(vals, y + (k - 1) * 0.22, s=60, marker=MARK[b], color=COLOR[b], edgecolor="white", linewidth=1.5,
                       label=LABEL[b], zorder=3)
            for v, yy in zip(vals, y + (k - 1) * 0.22):
                ax.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([t for *_, t in items], color=INK)
        ax.set_xlim(0, 1.12); ax.invert_yaxis()
        ax.set_title(f"{name} (n={res['samples']})", color=INK, fontsize=11)
        ax.set_xlabel("pooled value", color=INK)
        _style(ax)
    axes[0].legend(fontsize=9, loc="lower left", frameon=False, labelcolor=INK)
    fig.suptitle("Three-branch quality on identical retained inputs (offline replay; no baseline latency implied)",
                 color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_paired(res, path):
    items = [(c, m) for c in TOL for m in ("recall_within_tol", "precision", "seg_iou_pooled")]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(items))
    for k, (pair, color, marker) in enumerate(((("agent_selected", "full_local"), COLOR["agent_selected"], "^"),
                                              (("noae_uint8_q0", "full_local"), COLOR["noae_uint8_q0"], "s"))):
        key = f"{pair[0]}-{pair[1]}"
        for i, (c, m) in enumerate(items):
            d = res[c]["paired"][key][m]
            yy = i + (k - 0.5) * 0.3
            ax.plot(d["ci_block"], [yy, yy], color=color, lw=2)
            ax.scatter([d["point"]], [yy], color=color, marker=marker, s=55, edgecolor="white", linewidth=1.5, zorder=3,
                       label=(f"{LABEL[pair[0]]} − uncompressed" if i == 0 else None))
            ax.text(max(d["ci_block"][1], d["point"]) + 0.01, yy, f"{d['point']:+.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{c} {m.replace('_', ' ')}" for c, m in items], color=INK)
    ax.invert_yaxis(); ax.set_xlabel("paired difference (95% moving-block bootstrap CI)", color=INK)
    ax.set_title(f"Paired quality change vs uncompressed (n={res['samples']} retained decisions)", color=INK, fontsize=11)
    _style(ax); ax.legend(fontsize=9, frameon=False, loc="lower left", labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_latency(decisions, retained_frames, path):
    v = np.sort([d["ue_receipt"] for d in decisions if d.get("ue_receipt") is not None])
    r = np.sort([d["ue_receipt"] for d in decisions if d.get("ue_receipt") is not None and d["frame_id"] in retained_frames])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLOR["agent_selected"], lw=2,
            label=f"all policy decisions (n={len(v)}, ≤170 ms: {np.mean(v <= 170):.3f})")
    if len(r):
        ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=COLOR["full_local"], lw=2, ls="--",
                label=f"retained sample (n={len(r)}, ≤170 ms: {np.mean(r <= 170):.3f})")
    ax.axvline(170, color=INK2, lw=1, ls="-.")
    ax.text(171, 0.05, "170 ms (inclusive)", color=INK2, fontsize=9)
    ax.set_xlabel("UE quality-feedback receipt after action-open (ms)", color=INK)
    ax.set_ylabel("cumulative fraction", color=INK)
    ax.set_title("Live feedback latency, Run-5 agent (this capture only; raw, unsmoothed)", color=INK, fontsize=11)
    _style(ax); ax.legend(fontsize=9, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_route(decisions, rows, path):
    xy = np.array([d["ego_world"][:2] for d in decisions if d.get("ego_world")])
    by = {d["frame_id"]: d for d in decisions}
    adm = np.array([by[r["frame_id"]]["ego_world"][:2] for r in rows if r["frame_id"] in by])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(xy[:, 0], xy[:, 1], color=GRID, lw=3, label=f"route, all {len(xy)} decisions")
    if len(adm):
        ax.scatter(adm[:, 0], adm[:, 1], s=26, color=COLOR["full_local"], edgecolor="white", linewidth=1, zorder=3,
                   label=f"admitted retained samples ({len(adm)})")
    ax.set_aspect("equal"); ax.set_xlabel("world x (m)", color=INK); ax.set_ylabel("world y (m)", color=INK)
    ax.set_title("Route coverage of the retained sample (every 5th decision)", color=INK, fontsize=11)
    _style(ax); ax.legend(fontsize=9, frameon=False, labelcolor=INK, loc="center")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def strip(res):
    return {k: ({c: dict(branches={b: {kk: vv for kk, vv in v.items() if kk != "errors"} for b, v in r["branches"].items()},
                         paired=r["paired"]) for c, r in res.items() if c in TOL} if k == "classes" else v)
            for k, v in dict(classes=res, q_perc=res["q_perc"], samples=res["samples"]).items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True, help="live DECISIONS jsonl of this capture")
    p.add_argument("--acceptance", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    out = a.out
    out.mkdir(parents=False, exist_ok=False)
    rep = json.loads(a.replay.read_bytes())
    acceptance = json.loads(a.acceptance.read_bytes())
    decisions = [json.loads(l) for l in a.decisions.read_text().splitlines() if l]
    by = {d["frame_id"]: d for d in decisions}
    rows = [r for r in rep["rows"] if r["admission"] == "ADMITTED"]
    assert all(r["parity"]["selected_inner_payload_byte_identical"] and r["parity"]["selected_native_labels_identical"]
               and r["parity"]["selected_complete_records_identical"] and r["parity"]["archived_measurement_identical"]
               for r in rows)
    for r in rows:
        d = by[r["frame_id"]]
        r["timely"] = bool(d["timely"])
        r["receipt_ms"] = d.get("ue_receipt")
        assert (d["executed_action"]["mode_id"], d["executed_action"]["q_e4"]) == (r["action"]["mode_id"], r["action"]["q_e4"])
    timely = [r for r in rows if r["timely"]]
    res_all, res_t = analyze(rows), analyze(timely)
    summary = dict(schema="splitfusion.run5_quality_reference.summary.v1", replay_sha256=sha(a.replay),
                   decisions_sha256=sha(a.decisions), acceptance_sha256=sha(a.acceptance),
                   admission_counts=rep["admission_counts"], missing=[dict(ordinal=r["ordinal"], admission=r["admission"],
                   fault=r.get("fault")) for r in rep["rows"] if r["admission"] != "ADMITTED"],
                   all_admitted=strip(res_all), timely_admitted=strip(res_t),
                   targets_from_registered_ratios=targets(res_all, acceptance),
                   agent_actions=dict(mode=sorted({r["action"]["mode_id"] for r in rows}),
                                      q_e4_p50=float(np.median([r["action"]["q_e4"] for r in rows])),
                                      payload_p50=float(np.median([r["agent_payload_bytes"] for r in rows])),
                                      uint8_payload_p50=float(np.median([r["baseline"]["payload_bytes"] for r in rows]))),
                   bootstrap=dict(kind="moving block", block=BLOCK, resamples=B, seed=SEED),
                   live_latency_claim_for_baselines=False, code_sha256=sha(__file__))
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=True))
    fig_cdf(res_all, out / "fig1_conservative_localization_cdf_all.png", f"(n={len(rows)})")
    fig_cdf(res_t, out / "fig1b_conservative_localization_cdf_timely.png", f"(agent timely, n={len(timely)})")
    fig_compare(res_all, res_t, out / "fig2_three_branch_comparison.png")
    fig_paired(res_all, out / "fig3_paired_differences.png")
    fig_latency(decisions, {r["frame_id"] for r in rows}, out / "fig4_live_latency_cdf.png")
    fig_route(decisions, rows, out / "fig5_route_coverage.png")
    with (out / "PAIRED_SAMPLES.csv").open("x", newline="") as s:
        w = csv.writer(s)
        w.writerow(["ordinal", "frame_id", "session_uuid", "mode_id", "q_e4", "agent_payload_bytes", "timely", "receipt_ms",
                    *[f"{b}_{c}_{k}" for b in BR for c in TOL for k in ("eligible", "tp", "fp", "within", "inter", "union")],
                    *[f"{b}_q_perc" for b in BR]])
        for r in rows:
            vals = []
            for b in BR:
                for c in TOL:
                    x = per_sample(r, b, c)
                    vals += [x["eligible"], x["tp"], x["fp"], x["within"], x["inter"], x["union"]]
            w.writerow([r["ordinal"], r["frame_id"], r["session_uuid"], r["action"]["mode_id"], r["action"]["q_e4"],
                        r["agent_payload_bytes"], r["timely"], r["receipt_ms"], *vals, *[r[b]["q_perc"] for b in BR]])
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({p.name: sha(p) for p in sorted(out.iterdir())
                                                        if p.name != "OUTPUT_SHA256.json"}, indent=1))
    print(json.dumps(dict(samples=len(rows), timely=len(timely)), indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
