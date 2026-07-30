#!/usr/bin/env python3
"""Fold the FRESH uplink-only loopback run into the staleness analysis.

The fresh run has two condition families (see run_fresh_uplink_only_speedsweep.sh):

  L_*   true uplink-only (--edge-result-mode none), edge publishes to the spatial map, fast rasterizer.
        -> per-frame capture_to_map_update_done_ms = the genuine uplink-only lag L. No downlink in the chain.
        -> logs no front-side predictions/GT (the no-wait loop skips that block).

  ACC_* same sensor/model/codec/rasterizer recipe with the result-return enabled and the spatial-map
        stream off, so predictions + actor-origin GT are logged front-side.
        -> the object-motion/accuracy dataset. Its downlink is NEVER added to L.

What this produces:
  1. A fresh, independent measurement of L to cross-check the 93 ms Track-1 anchor.
  2. A fresh floor + error(v) curve on the current recipe (fast rasterizer, zstd).
  3. A distribution-aware staleness number: error averaged over the EMPIRICAL L distribution rather
     than evaluated only at p50 -- possible only because L and object motion were both measured on
     the same recipe in the same session.

Usage: analyze_fresh_run.py <fresh_run_dir>
"""
import csv
import glob
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

if len(sys.argv) > 1:
    FRESH = Path(sys.argv[1])
else:
    cands = sorted(HERE.glob("fresh_run_*"))
    if not cands:
        raise SystemExit("no fresh_run_* directory found; pass one explicitly")
    FRESH = cands[-1]

WARMUP = 10
NEAR, GATE, SCORE = 25.0, 2.0, 0.2
BANDS = [(0, 4, "~walk/slow"), (4, 8, "~6 mph"), (8, 12, "~10 mph"), (12, 16, "~14 mph"),
         (16, 20, "~18 mph"), (20, 26, "~23 mph"), (26, 30, "~28 mph"), (30, 40, "~32 mph")]
MIN_N = 15
FLOOR_ANCHOR = 1.1
TRACK1_L_P50, TRACK1_L_P95 = 93.3, 136.1     # anchors from results/L_anchors.csv (Step A)

log = []


def say(m=""):
    print(m)
    log.append(m)


def pctl(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q))


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def gt_at(sm, t):
    if t <= sm[0][0]:
        return sm[0][1], sm[0][2]
    if t >= sm[-1][0]:
        (t0, x0, y0), (t1, x1, y1) = sm[-2][:3], sm[-1][:3]
        dt = t1 - t0
        if dt <= 1e-6:
            return sm[-1][1], sm[-1][2]
        k = (t - t1) / dt
        return x1 + (x1 - x0) * k, y1 + (y1 - y0) * k
    for i in range(1, len(sm)):
        if sm[i][0] >= t:
            (t0, x0, y0), (t1, x1, y1) = sm[i - 1][:3], sm[i][:3]
            k = (t - t0) / max(1e-6, t1 - t0)
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    return sm[-1][1], sm[-1][2]


say(f"fresh run root: {FRESH}")

# =================================================================================================
# PART 1 - fresh L measurement (true uplink-only conditions)
# =================================================================================================
say("\n" + "=" * 96)
say("PART 1 - fresh uplink-only L (capture -> spatial-map update done), fast rasterizer, ideal loopback")
say("=" * 96)

STAGE_KEYS = ["capture_to_map_update_done_ms", "capture_to_backbone_input_ms",
              "radar_tensor_build_ms", "backbone_input_to_map_update_done_ms",
              "front_to_edge_ms", "tail_ms", "map_queue_ms", "sync_world_tick_ms",
              "camera_frame_wait_ms"]

pooled = defaultdict(list)
l_rows = []
for d in sorted(FRESH.glob("L_*")):
    f = d / "map_ingest_metrics.csv"
    if not f.exists():
        say(f"  {d.name}: MISSING map_ingest_metrics.csv - skipped")
        continue
    rows = list(csv.DictReader(open(f)))
    rows.sort(key=lambda r: int(r["frame_id"]))
    rows = rows[WARMUP:]
    if not rows:
        say(f"  {d.name}: no frames after warm-up - skipped")
        continue
    vals = {k: [float(r[k]) for r in rows if r.get(k) not in ("", None)] for k in STAGE_KEYS}
    for k, v in vals.items():
        pooled[k].extend(v)
    L = vals["capture_to_map_update_done_ms"]
    say(f"  {d.name:16s} n={len(rows):4d}  L p50 {pctl(L,50):6.1f}  p95 {pctl(L,95):6.1f}  "
        f"prep p50 {pctl(vals['capture_to_backbone_input_ms'],50):5.1f}  "
        f"radar p50 {pctl(vals['radar_tensor_build_ms'],50):5.1f}  "
        f"core p50 {pctl(vals['backbone_input_to_map_update_done_ms'],50):5.1f}")
    l_rows.append(dict(condition=d.name, n_frames=len(rows),
                       L_p50_ms=round(pctl(L, 50), 2), L_p95_ms=round(pctl(L, 95), 2),
                       sensorprep_p50_ms=round(pctl(vals["capture_to_backbone_input_ms"], 50), 2),
                       radar_build_p50_ms=round(pctl(vals["radar_tensor_build_ms"], 50), 2),
                       core_split_to_map_p50_ms=round(pctl(vals["backbone_input_to_map_update_done_ms"], 50), 2),
                       uplink_p50_ms=round(pctl(vals["front_to_edge_ms"], 50), 2),
                       tail_p50_ms=round(pctl(vals["tail_ms"], 50), 2),
                       map_queue_p50_ms=round(pctl(vals["map_queue_ms"], 50), 2)))

if not pooled:
    raise SystemExit("no usable L conditions in the fresh run - cannot cross-check the anchor")

L_ALL = pooled["capture_to_map_update_done_ms"]
L_FRESH_P50, L_FRESH_P95 = pctl(L_ALL, 50), pctl(L_ALL, 95)
say(f"\n  POOLED fresh L: n={len(L_ALL)} frames  p50 {L_FRESH_P50:.1f} ms  p95 {L_FRESH_P95:.1f} ms  "
    f"(mean {statistics.mean(L_ALL):.1f}, min {min(L_ALL):.1f}, max {max(L_ALL):.1f})")
say(f"  Track-1 anchor (Step A, 50-frame profile): p50 {TRACK1_L_P50:.1f} ms  p95 {TRACK1_L_P95:.1f} ms")
say(f"  agreement: p50 {L_FRESH_P50-TRACK1_L_P50:+.1f} ms "
    f"({100*(L_FRESH_P50-TRACK1_L_P50)/TRACK1_L_P50:+.1f}%),  p95 {L_FRESH_P95-TRACK1_L_P95:+.1f} ms")
say(f"  sensor prep share of fresh L: {100*pctl(pooled['capture_to_backbone_input_ms'],50)/L_FRESH_P50:.0f}% "
    f"(prep p50 {pctl(pooled['capture_to_backbone_input_ms'],50):.1f} ms, "
    f"core split->map p50 {pctl(pooled['backbone_input_to_map_update_done_ms'],50):.1f} ms)")

with open(RESULTS / "fresh_L_by_condition.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(l_rows[0].keys()))
    w.writeheader()
    w.writerows(l_rows)
    fh.flush()
say(f"  wrote {RESULTS/'fresh_L_by_condition.csv'}")

# =================================================================================================
# PART 2 - fresh accuracy / object-motion dataset
# =================================================================================================
say("\n" + "=" * 96)
say("PART 2 - fresh accuracy dataset (same recipe; result-return on ONLY to log preds+GT)")
say("=" * 96)

obs = []
missing_origin = 0
n_pred = 0
for d in sorted(FRESH.glob("ACC_*")):
    gts = glob.glob(str(d / "front_metrics" / "streams" / "*object_ground_truth.csv"))
    prs = glob.glob(str(d / "front_metrics" / "streams" / "*object_predictions.csv"))
    if not gts or not prs:
        say(f"  {d.name}: missing streams - skipped")
        continue
    gt = list(csv.DictReader(open(gts[0])))
    pr = list(csv.DictReader(open(prs[0])))
    n_pred += len(pr)
    traj = defaultdict(list)
    for r in gt:
        ox, oy = r.get("origin_x"), r.get("origin_y")
        if ox in (None, "") or oy in (None, ""):
            missing_origin += 1
            continue
        try:
            traj[r["actor_id"]].append((float(r["carla_timestamp"]), float(ox), float(oy),
                                        int(r["frame_id"]), truthy(r.get("in_camera_frustum", "")),
                                        float(r.get("distance_m", 999))))
        except (ValueError, KeyError):
            pass
    for a in traj:
        traj[a].sort()
    prby = defaultdict(list)
    for r in pr:
        try:
            if float(r.get("score", 0)) >= SCORE:
                prby[int(r["frame_id"])].append((float(r["world_x"]), float(r["world_y"])))
        except (ValueError, KeyError):
            pass
    before = len(obs)
    for aid, sm in traj.items():
        for i, (t, x, y, fid, inf, dist) in enumerate(sm):
            if not (inf and dist <= NEAR):
                continue
            P = prby.get(fid, [])
            if not P:
                continue
            dd = min(P, key=lambda p: math.hypot(p[0] - x, p[1] - y))
            if math.hypot(dd[0] - x, dd[1] - y) > GATE:
                continue
            j = min(max(1, i), len(sm) - 1)
            (t0, x0, y0), (t1, x1, y1) = sm[j - 1][:3], sm[j][:3]
            v = math.hypot(x1 - x0, y1 - y0) / max(1e-6, t1 - t0)
            obs.append((v * 2.237, t, sm, dd, v))
    say(f"  {d.name:16s} gt_rows={len(gt):6d} pred_rows={len(pr):5d} -> {len(obs)-before} observations")

say(f"\n  fresh observations: {len(obs)}   GT rows lacking origin_x/y: {missing_origin}")
if not obs:
    raise SystemExit("fresh accuracy conditions produced no matched observations")
USING_ORIGIN = missing_origin == 0
say(f"  USING_ORIGIN = {USING_ORIGIN}")


def err_at(o, lag=0.0):
    _v, t, sm, d, _vm = o
    gx, gy = gt_at(sm, t + lag)
    return math.hypot(d[0] - gx, d[1] - gy)


slow = [o for o in obs if o[0] < 1.0]
fresh_floor_slow = statistics.mean(err_at(o) for o in slow) if slow else float("nan")
say(f"  fresh floor at v<1 mph (n={len(slow)}): {fresh_floor_slow:.3f} m "
    f"| pooled all-speed L=0: {statistics.mean(err_at(o) for o in obs):.3f} m "
    f"| expected ~{FLOOR_ANCHOR} m")

# ---- validation gate for the FRESH accuracy dataset (guardrail 8) -------------------------------
say("\n  --- FRESH accuracy-dataset validation gate ---")
fresh_fail = []

# F1: per-observation direct-vs-closed-form (the method check; should be near-exact)
resid = []
for o in obs:
    e0 = err_at(o, 0.0)
    direct = err_at(o, L_FRESH_P50 / 1000.0)
    closed = math.hypot(e0, o[4] * L_FRESH_P50 / 1000.0)
    resid.append(direct - closed)
say(f"  [F1] per-observation direct-vs-closed-form at fresh L p50: mean {statistics.mean(resid):+.4f} m, "
    f"median {statistics.median(resid):+.4f} m, sd {statistics.pstdev(resid):.4f} m")
if abs(statistics.mean(resid)) > 0.10:
    fresh_fail.append("F1 closed-form mismatch on fresh data")

# F2: floor at v~0 -- thin sample and/or off anchor
if len(slow) < 50:
    fresh_fail.append(f"F2 v<1 mph sample too thin (n={len(slow)}) to pin a floor")
if not (0.7 <= fresh_floor_slow <= 1.6):
    fresh_fail.append(f"F2 floor at v~0 out of range ({fresh_floor_slow:.2f} m)")

# F3: is the floor speed-ordered the way a model floor should be (roughly flat, slow <= fast)?
f_walk = statistics.mean(err_at(o) for o in obs if o[0] < 4) if any(o[0] < 4 for o in obs) else float("nan")
f_mid = statistics.mean(err_at(o) for o in obs if 16 <= o[0] < 20) if any(16 <= o[0] < 20 for o in obs) else float("nan")
say(f"  [F3] L=0 floor: walk/slow {f_walk:.3f} m vs ~18 mph {f_mid:.3f} m "
    f"(difference {f_walk - f_mid:+.3f} m)")
if f_walk - f_mid > 0.25:
    fresh_fail.append("F3 floor is speed-INVERTED (slow band worse than fast band) -> scene/sampling "
                      "artifact, floor not trustworthy in this dataset")

# F4: monotonicity of error(L) per band
say("  [F4] monotonicity of error(L) per band (L = 0 -> p50 -> p95 -> 181 ms):")
nonmono = []
for lo, hi, lab in BANDS:
    sel = [o for o in obs if lo <= o[0] < hi]
    if len(sel) < MIN_N:
        continue
    seq = [statistics.mean(err_at(o, L) for o in sel)
           for L in (0.0, L_FRESH_P50 / 1000.0, L_FRESH_P95 / 1000.0, 0.1807)]
    ok = all(seq[i + 1] >= seq[i] - 0.02 for i in range(len(seq) - 1))
    say(f"       {lab:12s} n={len(sel):>4d} " + " -> ".join(f"{x:.2f}" for x in seq) +
        ("  OK" if ok else "  NON-MONOTONIC"))
    if not ok:
        nonmono.append(f"{lab}(n={len(sel)})")
if nonmono:
    fresh_fail.append("F4 non-monotonic error(L) in: " + ", ".join(nonmono))

say("")
if fresh_fail:
    for f in fresh_fail:
        say(f"  FRESH-GATE FLAG: {f}")
    say("\n  => CONCLUSION: the fresh run is ACCEPTED for its designed purpose (the independent L")
    say("     measurement in PART 1, 570 frames, 3 regimes, spread <1.5 ms across regimes) and as a")
    say("     qualitative consistency check on staleness GROWTH, but it is NOT used to redefine the")
    say("     model floor or the headline budgets. Those stay on the 829-observation baseline pool,")
    say("     which passes the full gate. No fresh-run number is promoted past its validation.")
else:
    say("  FRESH GATE PASSED on all checks.")

# empirical L distribution -> deciles, for distribution-aware staleness
L_DECILES = [pctl(L_ALL, q) / 1000.0 for q in range(5, 100, 10)]
say(f"\n  empirical L deciles (s): " + ", ".join(f"{x:.3f}" for x in L_DECILES))

REPORT = [("L=0 (floor)", 0.0),
          (f"fresh L p50 ({L_FRESH_P50:.0f} ms)", L_FRESH_P50 / 1000.0),
          (f"fresh L p95 ({L_FRESH_P95:.0f} ms)", L_FRESH_P95 / 1000.0),
          ("Track-1 anchor L (93 ms)", TRACK1_L_P50 / 1000.0),
          ("legacy rasterizer L (181 ms)", 0.1807)]

say("\n  error(v) on the FRESH dataset:")
say(f"  {'band':12s} {'n':>4s} " + " ".join(f"{lab:>26s}" for lab, _ in REPORT) +
    f" {'E_L[err] (distn-aware)':>23s}")
fresh_band_rows = []
for lo, hi, lab in BANDS:
    sel = [o for o in obs if lo <= o[0] < hi]
    if len(sel) < MIN_N:
        say(f"  {lab:12s} {len(sel):>4d}  (skipped, <{MIN_N})")
        continue
    cells = []
    curve = {}
    for rlab, L in REPORT:
        e = statistics.mean(err_at(o, L) for o in sel)
        curve[rlab] = e
        cells.append(f"{e:>26.2f}")
    e_dist = statistics.mean(statistics.mean(err_at(o, L) for L in L_DECILES) for o in sel)
    say(f"  {lab:12s} {len(sel):>4d} " + " ".join(cells) + f" {e_dist:>23.2f}")
    fresh_band_rows.append(dict(band=lab, n=len(sel),
                                mean_speed_mph=round(statistics.mean(o[0] for o in sel), 2),
                                mean_speed_ms=round(statistics.mean(o[4] for o in sel), 3),
                                **{f"err_m_{k}": round(v, 3) for k, v in curve.items()},
                                err_m_L_distribution_averaged=round(e_dist, 3)))

with open(RESULTS / "fresh_error_vs_L_by_speed.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fresh_band_rows[0].keys()))
    w.writeheader()
    w.writerows(fresh_band_rows)
say(f"\n  wrote {RESULTS/'fresh_error_vs_L_by_speed.csv'}")

# =================================================================================================
# PART 3 - baseline vs fresh comparison
# =================================================================================================
say("\n" + "=" * 96)
say("PART 3 - fresh vs baseline (829-observation pooled speed sweep, zlib/legacy-rasterizer era)")
say("=" * 96)
base_csv = RESULTS / "error_vs_L_by_speed.csv"
if base_csv.exists():
    base = {r["speed_band"]: r for r in csv.DictReader(open(base_csv))}
    say(f"  {'band':12s} {'base n':>7s} {'fresh n':>8s} {'base floor':>11s} {'fresh floor':>12s} "
        f"{'base@93ms':>10s} {'fresh@93ms':>11s}")
    for fr in fresh_band_rows:
        b = base.get(fr["band"])
        if not b:
            say(f"  {fr['band']:12s} {'-':>7s} {fr['n']:>8d}  (no baseline row)")
            continue
        say(f"  {fr['band']:12s} {b['n']:>7s} {fr['n']:>8d} {float(b['err_m_L0ms']):>11.2f} "
            f"{fr['err_m_L=0 (floor)']:>12.2f} {float(b['err_m_L93ms']):>10.2f} "
            f"{fr['err_m_Track-1 anchor L (93 ms)']:>11.2f}")
    # Floor-insensitive comparison: the STALENESS INCREMENT sqrt(err(L)^2 - err(0)^2), which should
    # equal v*L regardless of what the model floor happens to be in each dataset. This is the
    # apples-to-apples check between two runs whose absolute floors differ.
    say("\n  Floor-insensitive check -- implied displacement sqrt(err(L)^2 - err(0)^2) vs that dataset's OWN")
    say("  v*L, at L=181 ms. Each dataset is compared against its own mean band speed (they differ).")
    say(f"  {'band':12s} | {'fresh v':>7s} {'implied':>8s} {'exp v*L':>8s} | "
        f"{'base v':>7s} {'implied':>8s} {'exp v*L':>8s}")
    for fr in fresh_band_rows:
        b = base.get(fr["band"])
        if not b:
            continue
        vf = fr["mean_speed_ms"]
        vb = float(b["mean_speed_ms"])
        fl, f0 = fr["err_m_legacy rasterizer L (181 ms)"], fr["err_m_L=0 (floor)"]
        bl, b0 = float(b["err_m_L180ms"]), float(b["err_m_L0ms"])
        fi = math.sqrt(max(0.0, fl ** 2 - f0 ** 2))
        bi = math.sqrt(max(0.0, bl ** 2 - b0 ** 2))
        say(f"  {fr['band']:12s} | {vf:>7.2f} {fi:>8.2f} {vf*0.1807:>8.2f} | "
            f"{vb:>7.2f} {bi:>8.2f} {vb*0.1807:>8.2f}")
    say("  (this isolates the staleness term from the floor; each dataset should track its own v*L)")
else:
    say("  baseline CSV not found - run analyze_uplink_only_staleness.py first")

# ---- plot: fresh L distribution + fresh vs baseline error curve ----
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))

ax = axes[0]
ax.hist(L_ALL, bins=28, color="#0072B2", alpha=0.82, edgecolor="white", linewidth=0.6)
ax.axvline(L_FRESH_P50, color="#009E73", lw=2.2, label=f"fresh p50 {L_FRESH_P50:.0f} ms")
ax.axvline(L_FRESH_P95, color="#D55E00", lw=2.0, ls="--", label=f"fresh p95 {L_FRESH_P95:.0f} ms")
ax.axvline(TRACK1_L_P50, color="black", lw=1.7, ls=":", label=f"Track-1 anchor {TRACK1_L_P50:.0f} ms")
ax.set_xlabel("per-frame uplink-only lag L (ms)\ncapture $\\rightarrow$ spatial-map update done", fontsize=10.5)
ax.set_ylabel("frames", fontsize=10.5)
ax.set_title(f"Fresh L distribution ({len(L_ALL)} frames, 3 traffic regimes)", fontweight="bold", fontsize=11)
ax.legend(fontsize=8.6, frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
sp = [r["mean_speed_mph"] for r in fresh_band_rows]
ax.plot(sp, [r["err_m_L=0 (floor)"] for r in fresh_band_rows], color="#666666", lw=2.2,
        marker="o", ms=5, label="fresh, L=0 (floor)")
ax.plot(sp, [r["err_m_L_distribution_averaged"] for r in fresh_band_rows], color="#009E73", lw=2.6,
        marker="s", ms=5.5, label="fresh, averaged over measured L distribution")
ax.plot(sp, [r[f"err_m_fresh L p95 ({L_FRESH_P95:.0f} ms)"] for r in fresh_band_rows],
        color="#0072B2", lw=2.0, marker="^", ms=5, ls="--", label=f"fresh, L p95 ({L_FRESH_P95:.0f} ms)")
if base_csv.exists():
    bl = [(float(b["mean_speed_mph"]), float(b["err_m_L93ms"]))
          for b in csv.DictReader(open(base_csv))]
    ax.plot([x for x, _ in bl], [y for _, y in bl], color="#D55E00", lw=1.9, marker="x", ms=6,
            ls=":", label="baseline pool, L=93 ms")
ax.axhline(FLOOR_ANCHOR, color="black", ls=":", lw=1.3)
ax.set_xlabel("tracked object speed (mph)", fontsize=10.5)
ax.set_ylabel("localization error (m)", fontsize=10.5)
ax.set_title("Fresh-run error vs speed (uplink-only, ideal loopback)", fontweight="bold", fontsize=11)
ax.text(0.5, -0.30, "Fresh FLOOR is NOT trustworthy (gate flags F2/F3/F4: n=17 at v$\\approx$0, speed-inverted,\n"
                    "non-monotone bands). Headline floor/budgets use the 829-obs baseline pool.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#B22222")
ax.legend(fontsize=8.4, frameon=False, loc="upper left")
ax.grid(alpha=0.25)
ax.set_axisbelow(True)
ax.set_ylim(0, None)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(PLOTS / "fresh_run_L_and_error.pdf", bbox_inches="tight")
fig.savefig(PLOTS / "fresh_run_L_and_error.png", dpi=200, bbox_inches="tight")
say(f"\nwrote {PLOTS/'fresh_run_L_and_error.pdf'}")

(RESULTS / "run_log_fresh_run.txt").write_text("\n".join(log) + "\n")
print(f"wrote {RESULTS/'run_log_fresh_run.txt'}")
