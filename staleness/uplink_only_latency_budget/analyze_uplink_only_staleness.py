#!/usr/bin/env python3
"""Steps B + C — uplink-only staleness: error(v) at the capture->map lag L, and the recomputed budgets.

Post-hoc on the EXISTING speed-sweep opportunity-window captures (no new CARLA runs). The staleness
physics is object kinematics, so the method is the original one (make_speed_error_report.py /
make_fps_speed_report.py) re-parameterized with the uplink-only lag L = capture -> map-update-done.

GUARDRAILS enforced here:
  * L = full capture_to_map_update_done (fast rasterizer, 93 ms p50). NOT the 38 ms core split->map.
  * NO downlink term. L has no Y_down.
  * GT = actor ORIGIN (origin_x/origin_y). Hard-fails if the origin columns are absent -- it does NOT
    silently fall back to bbox-centre world_x/world_y (that fallback caused the old ~1 m offset bug).
  * Ideal loopback only. No OAI transport mixed in.
  * Validation gate runs BEFORE any budget is written: sample size, origin convention, floor at v~0,
    and direct-GT(t+L) vs closed-form sqrt(floor^2+(v*L)^2) agreement.

Usage: analyze_uplink_only_staleness.py [NEAR_m=25] [gate_m=2.0] [score=0.2]
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

NEAR = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
GATE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
SCORE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
RUNS_GLOB = str(HERE.parent / "metrics_logs" / "scenesense_runs" / "*")

# ---- uplink-only lag anchors (from Step A, results/L_anchors.csv) --------------------------------
L_FAST_P50 = 0.0933      # capture -> map update done, fast rasterizer, Track-1 50-frame profile
L_FAST_P95 = 0.1361      #   <-- CONSERVATIVE DESIGN ANCHOR (the PLAN's operating lag)
L_FRESH_P50 = 0.0675     # same path/recipe, fresh 570-frame 3-regime run (2026-07-30)
L_FRESH_P95 = 0.1018     #   <-- current BEST ESTIMATE (larger sample)
L_LEGACY_P50 = 0.1807    # same path, pre-optimization rasterizer
L_LEGACY_P95 = 0.2475
L_CORE_FAST = 0.0377     # backbone_input -> map update done. Shown ONLY to prove it understates.

L_SWEEP = [0.0, 0.025, 0.0377, 0.050, L_FRESH_P50, 0.075, L_FAST_P50, L_FRESH_P95, 0.115,
           L_FAST_P95, 0.160, L_LEGACY_P50, 0.215, L_LEGACY_P95, 0.30]
L_SWEEP = sorted(set(L_SWEEP))

L_ANCHORS = [
    ("L=0 (model floor)", 0.0, "#666666"),
    ("core split->map 38 ms (understates)", L_CORE_FAST, "#999999"),
    ("uplink-only L=68 ms (fresh p50, best est.)", L_FRESH_P50, "#56B4E9"),
    ("uplink-only L=93 ms (design anchor, p50)", L_FAST_P50, "#009E73"),
    ("uplink-only L=136 ms (fast, p95)", L_FAST_P95, "#0072B2"),
    ("legacy L=181 ms (p50)", L_LEGACY_P50, "#D55E00"),
]
REPORT_L = [0.0, L_CORE_FAST, L_FRESH_P50, L_FAST_P50, L_FAST_P95, L_LEGACY_P50, L_LEGACY_P95]
# the two anchors budgets are reported at: (label, L)
BUDGET_ANCHORS = [("fresh best-estimate", L_FRESH_P50), ("conservative design anchor", L_FAST_P50)]

FPS_SWEEP = [1, 5, 10, 15, 20, 25, 30]
FPS_COLORS = {1: "#e41a1c", 5: "#377eb8", 10: "#4daf4a", 15: "#984ea3",
              20: "#ff7f00", 25: "#a65628", 30: "#f781bf"}
BANDS = [(0, 4, "~walk/slow"), (4, 8, "~6 mph"), (8, 12, "~10 mph"), (12, 16, "~14 mph"),
         (16, 20, "~18 mph"), (20, 26, "~23 mph"), (26, 30, "~28 mph"), (30, 40, "~32 mph")]
MIN_N = 15
EPS = [1.5, 2.0, 2.5, 3.0]
FLOOR_ANCHOR = 1.1   # live in-domain floor; offline knob-matrix no-AE u8 anchor is 0.95 m

log_lines = []


def say(msg=""):
    print(msg)
    log_lines.append(msg)


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def gt_at(sm, t):
    """GT origin position at absolute carla time t (linear interp, linear extrapolation past the end)."""
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


def _is_sweep(run):
    for m in glob.glob(run + "/streams/*metrics.csv"):
        try:
            rows = list(csv.DictReader(open(m)))
            if rows and str(rows[0].get("run_group", "")).startswith("speedsweep_"):
                return True
        except Exception:
            pass
    return False


runs = [r for r in sorted(glob.glob(RUNS_GLOB)) if _is_sweep(r)]
say(f"speed-sweep runs found: {len(runs)} -> " + ", ".join(Path(r).name[-8:] for r in runs))
if not runs:
    raise SystemExit("no speed-sweep runs found - refusing to produce numbers")

# ---- collect opportunity-window observations ----------------------------------------------------
obs = []                 # (v_mph, t, sm, pred_xy)
origin_offsets = []      # |origin - bbox centre| per GT row, to prove the conventions differ
n_gt_rows = n_pred_rows = 0
missing_origin_rows = 0

for run in runs:
    gt = list(csv.DictReader(open(glob.glob(run + "/streams/*ground_truth.csv")[0])))
    pr = list(csv.DictReader(open(glob.glob(run + "/streams/*predictions.csv")[0])))
    n_gt_rows += len(gt)
    n_pred_rows += len(pr)
    traj = defaultdict(list)
    for r in gt:
        ox, oy = r.get("origin_x"), r.get("origin_y")
        if ox in (None, "") or oy in (None, ""):
            missing_origin_rows += 1
            continue                                  # HARD: never fall back to world_x/world_y
        try:
            ox, oy = float(ox), float(oy)
            wx, wy = float(r["world_x"]), float(r["world_y"])
            origin_offsets.append(math.hypot(ox - wx, oy - wy))
            traj[r["actor_id"]].append((float(r["carla_timestamp"]), ox, oy, int(r["frame_id"]),
                                        truthy(r.get("in_camera_frustum", "")),
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
    for aid, sm in traj.items():
        for i, (t, x, y, fid, inf, dist) in enumerate(sm):
            if not (inf and dist <= NEAR):
                continue
            P = prby.get(fid, [])
            if not P:
                continue
            d = min(P, key=lambda p: math.hypot(p[0] - x, p[1] - y))
            if math.hypot(d[0] - x, d[1] - y) > GATE:
                continue
            j = min(max(1, i), len(sm) - 1)
            (t0, x0, y0), (t1, x1, y1) = sm[j - 1][:3], sm[j][:3]
            v = math.hypot(x1 - x0, y1 - y0) / max(1e-6, t1 - t0)
            obs.append((v * 2.237, t, sm, d, v))

say(f"GT rows {n_gt_rows}, prediction rows {n_pred_rows}, "
    f"opportunity-window observations (<={NEAR:.0f} m, gate {GATE} m, score>={SCORE}): {len(obs)}")

# =================================================================================================
# VALIDATION GATE - must pass before any budget number is written
# =================================================================================================
say("\n" + "=" * 96)
say("VALIDATION GATE")
say("=" * 96)
gate_fail = []

# V1 - sample size / non-empty predictions
say(f"[V1] sample: {len(obs)} matched observations from {len(runs)} runs; "
    f"{n_pred_rows} prediction rows present.")
if len(obs) < 200:
    gate_fail.append(f"V1 too few observations ({len(obs)})")
if n_pred_rows == 0:
    gate_fail.append("V1 predictions file empty")

# V2 - GT origin convention
USING_ORIGIN = missing_origin_rows == 0 and len(origin_offsets) > 0
off_p50 = statistics.median(origin_offsets) if origin_offsets else float("nan")
off_max = max(origin_offsets) if origin_offsets else float("nan")
say(f"[V2] USING_ORIGIN = {USING_ORIGIN}  (GT rows lacking origin_x/y: {missing_origin_rows}); "
    f"|origin - bbox_centre| p50 {off_p50:.3f} m, max {off_max:.3f} m")
say("     -> the two conventions genuinely differ, so this confirms we are on origin, "
    "not silently on bbox-centre.")
if not USING_ORIGIN:
    gate_fail.append("V2 origin convention not confirmed")

# V3 - model floor at v ~ 0 (L=0)
def err_at(o, lag=0.0):
    """|pred - GT_origin(t + lag)| for one observation."""
    _v_mph, t, sm, d, _v_ms = o
    gx, gy = gt_at(sm, t + lag)
    return math.hypot(d[0] - gx, d[1] - gy)


slow = [o for o in obs if o[0] < 1.0]
floor_slow = statistics.mean(err_at(o) for o in slow) if slow else float("nan")
allband_floor = statistics.mean(err_at(o) for o in obs)
say(f"[V3] floor at v<1 mph (n={len(slow)}): {floor_slow:.3f} m ; pooled all-speed L=0: {allband_floor:.3f} m ; "
    f"expected ~{FLOOR_ANCHOR} m (offline no-AE u8 anchor 0.95 m)")
if not (0.7 <= floor_slow <= 1.6):
    gate_fail.append(f"V3 floor at v~0 out of range ({floor_slow:.2f} m)")

# ---- per-band curves (needed for V4 and everything after) ---------------------------------------
band_rows = []
for lo, hi, lab in BANDS:
    sel = [o for o in obs if lo <= o[0] < hi]
    if len(sel) < MIN_N:
        say(f"     band {lab:12s} n={len(sel):<4d} skipped (<{MIN_N})")
        continue
    curve = {}
    for L in L_SWEEP:
        curve[L] = statistics.mean(
            math.hypot(o[3][0] - gt_at(o[2], o[1] + L)[0], o[3][1] - gt_at(o[2], o[1] + L)[1])
            for o in sel)
    v_ms = statistics.mean(o[4] for o in sel)
    band_rows.append(dict(label=lab, n=len(sel), v_mph=statistics.mean(o[0] for o in sel),
                          v_ms=v_ms, curve=curve, sel=sel))

# V4 - direct GT(t+L) vs closed form sqrt(floor^2 + (v L)^2)
say("\n[V4] cross-check: direct GT(t+L) vs closed form sqrt(err0^2 + (v*L)^2)")
say("     (a) PER-OBSERVATION, at the operating lag L=93 ms - each obs uses its own err0 and its own v:")
per_obs_resid = []
for v_mph, t, sm, d, v_ms in obs:
    e0 = math.hypot(d[0] - gt_at(sm, t)[0], d[1] - gt_at(sm, t)[1])
    direct = math.hypot(d[0] - gt_at(sm, t + L_FAST_P50)[0], d[1] - gt_at(sm, t + L_FAST_P50)[1])
    closed = math.hypot(e0, v_ms * L_FAST_P50)
    per_obs_resid.append(direct - closed)
say(f"         residual (direct - closed): mean {statistics.mean(per_obs_resid):+.4f} m, "
    f"median {statistics.median(per_obs_resid):+.4f} m, "
    f"sd {statistics.pstdev(per_obs_resid):.4f} m, n={len(per_obs_resid)}")
if abs(statistics.mean(per_obs_resid)) > 0.10:
    gate_fail.append(f"V4a per-observation closed-form mismatch "
                     f"({statistics.mean(per_obs_resid):+.3f} m)")

say("     (b) PER-BAND at each reported L (band floor + band mean speed):")
say(f"         {'band':12s} {'n':>4s} {'v m/s':>6s} " +
    " ".join(f"{int(L*1000):>3d}ms:dir/closed" for L in REPORT_L[2:]))
v4_band_resid = []
for br in band_rows:
    cells = []
    for L in REPORT_L[2:]:
        direct = br["curve"][L]
        closed = math.hypot(br["curve"][0.0], br["v_ms"] * L)
        v4_band_resid.append(direct - closed)
        cells.append(f"{direct:5.2f}/{closed:5.2f}")
    say(f"         {br['label']:12s} {br['n']:>4d} {br['v_ms']:>6.2f} " + "  ".join(cells))
say(f"         per-band residual: mean {statistics.mean(v4_band_resid):+.3f} m, "
    f"median {statistics.median(v4_band_resid):+.3f} m, "
    f"max |{max(abs(x) for x in v4_band_resid):.3f}| m")
say("         (per-band residual is looser than per-observation by construction: a band pools a spread of\n"
    "          speeds, and real targets accelerate/turn, which the constant-velocity closed form omits.)")

say("")
if gate_fail:
    for f in gate_fail:
        say(f"GATE FAIL: {f}")
    raise SystemExit("VALIDATION GATE FAILED - stopping before writing any finding (per PLAN guardrail 8).")
say("GATE PASSED - proceeding to budgets.")

# =================================================================================================
# RESULT 1 - error vs speed / vs L
# =================================================================================================
say("\n" + "=" * 96)
say("RESULT 1 - localization error vs uplink-only lag L (ideal loopback, capture->map, no downlink)")
say("=" * 96)
hdr = f"{'band':12s} {'n':>4s} " + " ".join(f"{int(L*1000):>6d}ms" for L in REPORT_L)
say(hdr)
for br in band_rows:
    say(f"{br['label']:12s} {br['n']:>4d} " + " ".join(f"{br['curve'][L]:>8.2f}" for L in REPORT_L))
say("                  " + "  ".join(f"{'':>6s}" for _ in REPORT_L))
say("legend: 0=floor  38=core split->map (UNDERSTATES: omits sensor prep)  67=fresh measured p50 (best est.)  "
    "93=conservative design anchor p50  136=fast p95  181=legacy p50  248=legacy p95")

with open(RESULTS / "error_vs_L_by_speed.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["speed_band", "n", "mean_speed_mph", "mean_speed_ms"] +
               [f"err_m_L{int(L*1000)}ms" for L in L_SWEEP])
    for br in band_rows:
        w.writerow([br["label"], br["n"], round(br["v_mph"], 2), round(br["v_ms"], 3)] +
                   [round(br["curve"][L], 3) for L in L_SWEEP])
say(f"\nwrote {RESULTS/'error_vs_L_by_speed.csv'}")

# staleness cost of the legacy rasterizer, per band
say("\nFast-rasterizer staleness benefit (L 181 -> 93 ms, i.e. -87 ms of age):")
say(f"{'band':12s} {'v (m/s)':>8s} {'err@181':>8s} {'err@93':>8s} {'measured d':>11s} {'v*0.087 m':>10s}")
fast_gain = []
for br in band_rows:
    dm = br["curve"][L_LEGACY_P50] - br["curve"][L_FAST_P50]
    say(f"{br['label']:12s} {br['v_ms']:>8.2f} {br['curve'][L_LEGACY_P50]:>8.2f} "
        f"{br['curve'][L_FAST_P50]:>8.2f} {dm:>11.2f} {br['v_ms']*0.0874:>10.2f}")
    fast_gain.append(dict(band=br["label"], v_ms=round(br["v_ms"], 3),
                          err_legacy_m=round(br["curve"][L_LEGACY_P50], 3),
                          err_fast_m=round(br["curve"][L_FAST_P50], 3),
                          measured_gain_m=round(dm, 3),
                          predicted_gain_m=round(br["v_ms"] * 0.0874, 3)))
with open(RESULTS / "fast_rasterizer_staleness_gain.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fast_gain[0].keys()))
    w.writeheader()
    w.writerows(fast_gain)

# =================================================================================================
# RESULT 2 - map-hold staleness (FPS) on top of L
# =================================================================================================
say("\n" + "=" * 96)
say("RESULT 2 - FPS as map-hold staleness, ON TOP of the uplink-only L")
say("map holds the last detection between updates -> extra age s/FPS (s=0.5 average query, s=1 worst case)")
say("=" * 96)

fps_tables = {}
for s_tag, s in (("avg(s=0.5)", 0.5), ("worst(s=1)", 1.0)):
    for L_tag, L in (("L=0", 0.0), ("L=93ms", L_FAST_P50)):
        key = (s_tag, L_tag)
        say(f"\n-- {s_tag}, {L_tag} --")
        say(f"{'band':12s} {'n':>4s} " + " ".join(f"{f:>6d}fps" for f in FPS_SWEEP))
        rowsout = []
        for br in band_rows:
            vals = []
            for f in FPS_SWEEP:
                lag = L + s / f
                vals.append(statistics.mean(
                    math.hypot(o[3][0] - gt_at(o[2], o[1] + lag)[0],
                               o[3][1] - gt_at(o[2], o[1] + lag)[1]) for o in br["sel"]))
            say(f"{br['label']:12s} {br['n']:>4d} " + " ".join(f"{v:>9.2f}" for v in vals))
            rowsout.append((br["label"], br["n"], br["v_ms"], vals))
        fps_tables[key] = rowsout

with open(RESULTS / "error_vs_fps.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["map_hold_term", "L_condition", "speed_band", "n", "mean_speed_ms"] +
               [f"err_m_{f}fps" for f in FPS_SWEEP])
    for (s_tag, L_tag), rowsout in fps_tables.items():
        for lab, n, v_ms, vals in rowsout:
            w.writerow([s_tag, L_tag, lab, n, round(v_ms, 3)] + [round(v, 3) for v in vals])
say(f"\nwrote {RESULTS/'error_vs_fps.csv'}")

# =================================================================================================
# RESULT 3 - budgets
# =================================================================================================
say("\n" + "=" * 96)
say("RESULT 3 - uplink-only budgets.  MASTER: v*(L + s/FPS) <= B(eps) = sqrt(eps^2 - floor^2), NO Y_down")
say("=" * 96)


def max_L_measured(curve, eps):
    """Interpolate the measured error(L) curve for the largest L holding error <= eps."""
    ls = L_SWEEP
    if curve[ls[0]] > eps:
        return "—"                                    # model-limited, not latency-limited
    if curve[ls[-1]] <= eps:
        return f">{int(ls[-1]*1000)}"
    for k in range(1, len(ls)):
        if curve[ls[k]] > eps:
            l0, l1 = ls[k - 1] * 1000, ls[k] * 1000
            e0, e1 = curve[ls[k - 1]], curve[ls[k]]
            return f"{l0 + (l1-l0)*(eps-e0)/(e1-e0):.0f}"
    return f">{int(ls[-1]*1000)}"


say("\n(3a) Latency UPPER bound: max uplink-only L (ms, capture->map) to hold error <= eps.")
say("     measured = interpolated from the direct GT(t+L) curve; closed = B(eps)/v with floor 1.1 m.")
say(f"     '—' = model floor already exceeds eps.   Measured operating range today: "
    f"L = {L_FRESH_P50*1000:.0f}-{L_FAST_P50*1000:.0f} ms p50.")
say(f"{'band':12s} {'v m/s':>6s} " + " ".join(f"{'eps<='+str(e)+'m':>18s}" for e in EPS))
budget_rows = []
for br in band_rows:
    cells = []
    for e in EPS:
        meas = max_L_measured(br["curve"], e)
        B = math.sqrt(max(0.0, e ** 2 - FLOOR_ANCHOR ** 2))
        closed = "—" if e <= FLOOR_ANCHOR else f"{1000*B/br['v_ms']:.0f}" if br["v_ms"] > 1e-3 else "inf"
        cells.append(f"{meas:>8s} /{closed:>8s}")
        budget_rows.append(dict(band=br["label"], v_ms=round(br["v_ms"], 3), eps_m=e,
                                max_L_ms_measured=meas, max_L_ms_closedform=closed))
    say(f"{br['label']:12s} {br['v_ms']:>6.2f} " + " ".join(cells))
say("     (cells are  measured / closed-form)")

say("\n(3b) FPS LOWER bound (uplink-only), at BOTH measured L anchors:")
say("     FPS_min(v,eps) = v / (B(eps) - v*L)   [worst case s=1; halve the term for s=0.5]")
fps_rows = []
for a_lab, a_L in BUDGET_ANCHORS:
    say(f"\n     -- L = {a_L*1000:.0f} ms ({a_lab}) --")
    say(f"     {'band':12s} {'v m/s':>6s} " + " ".join(f"{'eps<='+str(e)+'m':>12s}" for e in EPS))
    for br in band_rows:
        cells = []
        for e in EPS:
            B = math.sqrt(max(0.0, e ** 2 - FLOOR_ANCHOR ** 2))
            v = br["v_ms"]
            slackww = B - v * a_L
            if e <= FLOOR_ANCHOR:
                cell = "—"
            elif v < 1e-3:
                cell = "any"
            elif slackww <= 0:
                cell = "INFEAS"
            else:
                cell = f"{v/slackww:.1f}"
            cells.append(f"{cell:>12s}")
            fps_rows.append(dict(band=br["label"], v_ms=round(v, 3), eps_m=e,
                                 L_ms=round(a_L * 1000, 1), L_anchor=a_lab,
                                 fps_min_worstcase=cell))
        say(f"     {br['label']:12s} {br['v_ms']:>6.2f} " + " ".join(cells))
say("\n     'INFEAS' = L alone already spends the whole budget: no FPS fixes it, must cut L.")
say("     '—'      = model floor exceeds eps: model problem, not a latency/FPS problem.")

with open(RESULTS / "budget_latency_upper.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(budget_rows[0].keys()))
    w.writeheader()
    w.writerows(budget_rows)
with open(RESULTS / "budget_fps_lower.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fps_rows[0].keys()))
    w.writeheader()
    w.writerows(fps_rows)
say(f"\nwrote {RESULTS/'budget_latency_upper.csv'} and {RESULTS/'budget_fps_lower.csv'}")

# headroom at the operating point
say("\n(3c) Headroom at each measured operating point (uplink-only):")
head_rows = []
for a_lab, a_L in BUDGET_ANCHORS:
    say(f"\n     -- L = {a_L*1000:.0f} ms ({a_lab}) --")
    say(f"     {'band':12s} {'v m/s':>6s} {'v*L (m)':>8s} " +
        " ".join(f"{'B('+str(e)+')':>9s}" for e in EPS))
    for br in band_rows:
        vL = br["v_ms"] * a_L
        cells = []
        for e in EPS:
            B = math.sqrt(max(0.0, e ** 2 - FLOOR_ANCHOR ** 2))
            cells.append(f"{B - vL:>+9.2f}")
            head_rows.append(dict(band=br["label"], v_ms=round(br["v_ms"], 3), L_ms=round(a_L*1000, 1),
                                  L_anchor=a_lab, eps_m=e, v_times_L_m=round(vL, 3),
                                  remaining_budget_m=round(B - vL, 3)))
        say(f"     {br['label']:12s} {br['v_ms']:>6.2f} {vL:>8.2f} " + " ".join(cells))
say("\n     value = remaining staleness budget after L is paid; negative = already over budget at any FPS.")
with open(RESULTS / "budget_headroom.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(head_rows[0].keys()))
    w.writeheader()
    w.writerows(head_rows)
say(f"     wrote {RESULTS/'budget_headroom.csv'}")

# =================================================================================================
# PLOTS
# =================================================================================================
# P1 - error vs speed at the three L values
fig, ax = plt.subplots(figsize=(8.4, 5.4))
speeds = [br["v_mph"] for br in band_rows]
for lab, L, c in L_ANCHORS:
    ax.plot(speeds, [br["curve"][L] for br in band_rows], color=c, lw=2.4, marker="o", ms=5,
            label=lab, ls="--" if "understates" in lab else "-",
            alpha=0.75 if "understates" in lab else 1.0)
ax.axhline(FLOOR_ANCHOR, color="black", ls=":", lw=1.4)
ax.text(34.4, FLOOR_ANCHOR - 0.12, f"model floor ~{FLOOR_ANCHOR} m (model-limited)",
        fontsize=8.5, ha="right", va="top")
ax.set_xlabel("tracked object speed (mph)", fontsize=11)
ax.set_ylabel("localization error (m)", fontsize=11)
ax.set_title("Uplink-only staleness: error vs object speed at each capture$\\rightarrow$map lag L\n"
             "(ideal loopback, no downlink return)", fontweight="bold", fontsize=12)
ax.legend(fontsize=8.8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)
ax.set_axisbelow(True)
ax.set_ylim(0, None)
ax.set_xlim(0, 35)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS / "error_vs_speed_by_L.pdf", bbox_inches="tight")
fig.savefig(PLOTS / "error_vs_speed_by_L.png", dpi=200, bbox_inches="tight")
say(f"\nwrote {PLOTS/'error_vs_speed_by_L.pdf'}")

# P2 - error vs L per speed band, with the operating anchors
fig, ax = plt.subplots(figsize=(8.4, 5.4))
cmap = plt.cm.viridis(np.linspace(0, 0.92, len(band_rows)))
for i, br in enumerate(band_rows):
    ax.plot([L * 1000 for L in L_SWEEP], [br["curve"][L] for L in L_SWEEP],
            color=cmap[i], lw=2.2, marker="o", ms=3.5, label=br["label"])
ytop = ax.get_ylim()[1]
for lab, L, c in L_ANCHORS[1:4]:
    ax.axvline(L * 1000, color=c, ls=":", lw=1.7, alpha=0.9)
    ax.text(L * 1000 + 3, ytop * 0.62, lab, rotation=90, va="top", ha="left", fontsize=7.6, color=c)
ax.set_xlabel("uplink-only lag L (ms) = capture $\\rightarrow$ spatial-map update", fontsize=11)
ax.set_ylabel("localization error (m)", fontsize=11)
ax.set_title("Error vs uplink-only freshness age, per object speed\n(no downlink term; ideal loopback)",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=8.6, frameon=False, loc="upper left")
ax.grid(alpha=0.25)
ax.set_axisbelow(True)
ax.set_ylim(0, None)
ax.set_xlim(0, L_SWEEP[-1] * 1000)
ax.margins(x=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS / "error_vs_L_by_speed.pdf", bbox_inches="tight")
fig.savefig(PLOTS / "error_vs_L_by_speed.png", dpi=200, bbox_inches="tight")
say(f"wrote {PLOTS/'error_vs_L_by_speed.pdf'}")

# P3 - FPS x L budget heat/line: error vs FPS at L=0 and L=93 ms
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
for axi, (L_tag, L) in zip(axes, (("L = 0 (ideal, map-hold only)", 0.0),
                                  (f"L = {L_FAST_P50*1000:.0f} ms (deployed uplink-only)", L_FAST_P50))):
    for f in FPS_SWEEP:
        ys = []
        for br in band_rows:
            lag = L + 1.0 / f
            ys.append(statistics.mean(
                math.hypot(o[3][0] - gt_at(o[2], o[1] + lag)[0],
                           o[3][1] - gt_at(o[2], o[1] + lag)[1]) for o in br["sel"]))
        axi.plot(speeds, ys, color=FPS_COLORS[f], lw=2.3, marker="o", ms=4.5, label=f"{f} FPS")
    axi.axhline(FLOOR_ANCHOR, color="black", ls=":", lw=1.3)
    axi.set_title(L_tag, fontweight="bold", fontsize=11)
    axi.set_xlabel("tracked object speed (mph)", fontsize=10.5)
    axi.grid(alpha=0.25)
    axi.set_axisbelow(True)
    axi.set_xlim(0, 35)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
axes[0].set_ylabel("localization error (m)", fontsize=11)
axes[0].set_ylim(0, None)
axes[0].legend(fontsize=9, frameon=False, loc="upper left", ncol=2)
fig.suptitle("Map-hold staleness (worst case 1/FPS) with and without the uplink-only lag",
             fontweight="bold", fontsize=12.5)
fig.tight_layout()
fig.savefig(PLOTS / "fps_x_L_budget.pdf", bbox_inches="tight")
fig.savefig(PLOTS / "fps_x_L_budget.png", dpi=200, bbox_inches="tight")
say(f"wrote {PLOTS/'fps_x_L_budget.pdf'}")

# P4 - feasibility map: which (L, FPS) satisfy eps=2 m at each speed
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharey=True)
Ls = np.linspace(0, 300, 160)
Fs = np.linspace(2, 30, 160)
LL, FF = np.meshgrid(Ls, Fs)
for axi, (v_lab, v_ms) in zip(axes, [("pedestrian (1.4 m/s)", 1.4),
                                     ("~18 mph (8.0 m/s)", 8.0),
                                     ("~32 mph (14.3 m/s)", 14.3)]):
    B = math.sqrt(2.0 ** 2 - FLOOR_ANCHOR ** 2)
    err = np.sqrt(FLOOR_ANCHOR ** 2 + (v_ms * (LL / 1000.0 + 1.0 / FF)) ** 2)
    cs = axi.contourf(LL, FF, err, levels=np.linspace(1.0, 4.0, 25), cmap="magma_r", extend="max")
    axi.contour(LL, FF, err, levels=[2.0], colors="#00E5FF", linewidths=2.6)
    axi.axvline(L_FAST_P50 * 1000, color="white", ls="--", lw=1.6, alpha=0.9)
    axi.text(L_FAST_P50 * 1000 + 5, 28, "deployed\nL=93 ms", color="white", fontsize=8, va="top")
    axi.set_title(f"{v_lab}   (B(2 m)={B:.2f} m)", fontsize=10.5, fontweight="bold")
    axi.set_xlabel("uplink-only L (ms)", fontsize=10)
axes[0].set_ylabel("map update rate (FPS)", fontsize=10.5)
cb = fig.colorbar(cs, ax=axes, fraction=0.028, pad=0.02)
cb.set_label("predicted error (m)", fontsize=10)
fig.suptitle("Uplink-only feasibility: cyan line = $\\epsilon$ = 2 m boundary (feasible region is to its LEFT),\n"
             "error $=\\sqrt{1.1^2+(v(L+1/\\mathrm{FPS}))^2}$",
             fontweight="bold", fontsize=11.5)
fig.subplots_adjust(top=0.78)
fig.savefig(PLOTS / "feasibility_L_fps.pdf", bbox_inches="tight")
fig.savefig(PLOTS / "feasibility_L_fps.png", dpi=200, bbox_inches="tight")
say(f"wrote {PLOTS/'feasibility_L_fps.pdf'}")

(RESULTS / "run_log_staleness.txt").write_text("\n".join(log_lines) + "\n")
print(f"\nwrote {RESULTS/'run_log_staleness.txt'}")
