#!/usr/bin/env python3
"""Step A — the uplink-only freshness-age decomposition L = capture -> map-update-done.

Re-derives the Track-1 loopback latency decomposition directly from the per-frame
map_ingest_metrics.csv of the legacy and fast radar-rasterizer profile runs, instead of
quoting TRACK1_IDEAL_LOOPBACK_RESULTS.md. Same convention as that doc: first 10 frames
excluded as warm-up.

UPLINK-ONLY: there is no downlink/result-return term. L ends when the spatial map finishes
applying the update.

Writes results/L_decomposition.csv and results/plots/freshness_age_breakdown.{pdf,png}.
"""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

RUNS = HERE.parent.parent / "uplink_only_spatial_map_pipeline" / "runs"
PROFILES = {
    "legacy": RUNS / "live_front_prep_profile_50f" / "map_ingest_metrics.csv",
    "fast": RUNS / "live_front_prep_profile_fast_50f" / "map_ingest_metrics.csv",
}
WARMUP = 10

# stage -> (label, bucket in the uplink-only lag)
STAGES = [
    ("sync_world_tick_ms",        "CARLA sync tick",            "sensorprep"),
    ("camera_frame_wait_ms",      "camera frame wait",          "sensorprep"),
    ("radar_wait_ms",             "radar packet wait",          "sensorprep"),
    ("rgb_convert_ms",            "RGB convert",                "sensorprep"),
    ("radar_tensor_build_ms",     "radar tensor build",         "sensorprep"),
    ("model_preprocess_ms",       "model preprocess",           "sensorprep"),
    ("capture_to_backbone_input_ms", "Y_sensorprep (total)",     "rollup"),
    ("front_backbone_ms",         "front backbone (split enc)", "front"),
    ("feature_serialize_ms",      "feature serialize (zstd)",   "front"),
    ("front_to_edge_ms",          "Y_uplink (front->edge)",     "uplink"),
    ("tail_ms",                   "Y_tail (edge tail)",         "tail"),
    ("map_queue_ms",              "map UDP ingest/queue",       "mapinsert"),
    ("map_service_ms",            "map service (update apply)", "mapinsert"),
    ("backbone_input_to_map_update_done_ms", "core split->map (NOT L)", "rollup"),
    ("capture_to_map_update_done_ms", "L = capture->map update", "rollup"),
]


def pct(vals, q):
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def load(path):
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: int(r["frame_id"]))
    kept = rows[WARMUP:]
    return kept


def col(rows, name):
    out = []
    for r in rows:
        v = r.get(name, "")
        if v in ("", None):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


data = {}
for tag, path in PROFILES.items():
    if not path.exists():
        raise SystemExit(f"missing per-frame profile: {path}")
    rows = load(path)
    data[tag] = rows
    print(f"{tag:7s}: {len(rows)} frames after excluding first {WARMUP} (file has {len(rows)+WARMUP})")

# ---- table ----
out_rows = []
print(f"\n{'stage':32s} {'bucket':11s} {'legacy p50':>10s} {'p95':>8s} {'fast p50':>9s} {'p95':>8s} {'d p50':>8s}")
for key, label, bucket in STAGES:
    lg, fs = col(data["legacy"], key), col(data["fast"], key)
    if not lg or not fs:
        print(f"{label:32s} {bucket:11s}   (missing column {key})")
        continue
    lg50, lg95, fs50, fs95 = pct(lg, 50), pct(lg, 95), pct(fs, 50), pct(fs, 95)
    print(f"{label:32s} {bucket:11s} {lg50:10.1f} {lg95:8.1f} {fs50:9.1f} {fs95:8.1f} {fs50-lg50:+8.1f}")
    out_rows.append(dict(stage_key=key, stage=label, bucket=bucket,
                         legacy_p50_ms=round(lg50, 2), legacy_p95_ms=round(lg95, 2),
                         fast_p50_ms=round(fs50, 2), fast_p95_ms=round(fs95, 2),
                         delta_p50_ms=round(fs50 - lg50, 2)))

with open(RESULTS / "L_decomposition.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"\nwrote {RESULTS/'L_decomposition.csv'}")

# ---- consistency check: do the buckets sum to L? ----
by = {r["stage_key"]: r for r in out_rows}
L_fast = by["capture_to_map_update_done_ms"]["fast_p50_ms"]
L_legacy = by["capture_to_map_update_done_ms"]["legacy_p50_ms"]
core_fast = by["backbone_input_to_map_update_done_ms"]["fast_p50_ms"]
prep_fast = by["capture_to_backbone_input_ms"]["fast_p50_ms"]
print("\n--- additivity check (p50s are not exactly additive; medians of sums != sum of medians) ---")
for tag, L, prep, core in (("fast", L_fast, prep_fast, core_fast),
                           ("legacy", L_legacy, by["capture_to_backbone_input_ms"]["legacy_p50_ms"],
                            by["backbone_input_to_map_update_done_ms"]["legacy_p50_ms"])):
    print(f"{tag:7s}: Y_sensorprep {prep:6.1f} + core split->map {core:6.1f} = {prep+core:6.1f}  vs L {L:6.1f} "
          f"({100*(prep+core-L)/L:+.1f}%)")

# per-frame additivity (exact, same frame)
for tag in ("legacy", "fast"):
    rows = data[tag]
    resid = []
    for r in rows:
        try:
            resid.append(float(r["capture_to_backbone_input_ms"]) +
                         float(r["backbone_input_to_map_update_done_ms"]) -
                         float(r["capture_to_map_update_done_ms"]))
        except (ValueError, KeyError):
            pass
    print(f"{tag:7s}: per-frame residual (prep+core-L) p50 {pct(resid,50):+.3f} ms, max |{max(abs(x) for x in resid):.3f}| ms")

# sensor-prep share of L
print(f"\nsensor prep share of L: fast {100*prep_fast/L_fast:.0f}%  legacy "
      f"{100*by['capture_to_backbone_input_ms']['legacy_p50_ms']/L_legacy:.0f}%")
print(f"fast-rasterizer staleness saving: L {L_legacy:.1f} -> {L_fast:.1f} ms  "
      f"(-{L_legacy-L_fast:.1f} ms p50)")

# ---- plot: stacked freshness-age breakdown, legacy vs fast ----
BUCKETS = [
    ("radar tensor build",    "radar_tensor_build_ms",   "#D55E00"),
    ("other sensor prep",     None,                      "#E69F00"),
    ("front compute+serialize", None,                    "#009E73"),
    ("uplink (loopback)",     "front_to_edge_ms",        "#0072B2"),
    ("edge tail",             "tail_ms",                 "#56B4E9"),
    ("map ingest+update",     None,                      "#7A5195"),
]


def bucket_vals(tag):
    r = {k: (by[k]["fast_p50_ms"] if tag == "fast" else by[k]["legacy_p50_ms"]) for k in by}
    radar = r["radar_tensor_build_ms"]
    prep_total = r["capture_to_backbone_input_ms"]
    other_prep = max(0.0, prep_total - radar)
    front = r["front_backbone_ms"] + r["feature_serialize_ms"]
    up = r["front_to_edge_ms"]
    tail = r["tail_ms"]
    core = r["backbone_input_to_map_update_done_ms"]
    mapins = max(0.0, core - front - up - tail)
    return [radar, other_prep, front, up, tail, mapins]


fig, ax = plt.subplots(figsize=(8.6, 5.0))
tags = ["legacy", "fast"]
xs = np.arange(len(tags))
bottoms = np.zeros(len(tags))
vals_by_tag = {t: bucket_vals(t) for t in tags}
for i, (label, _key, color) in enumerate(BUCKETS):
    heights = np.array([vals_by_tag[t][i] for t in tags])
    ax.bar(xs, heights, bottom=bottoms, width=0.5, color=color, label=label,
           edgecolor="white", linewidth=0.8)
    for x, h, b in zip(xs, heights, bottoms):
        if h > 6:
            ax.text(x, b + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
    bottoms += heights

for x, t in zip(xs, tags):
    L = by["capture_to_map_update_done_ms"][f"{t}_p50_ms"]
    ax.text(x, bottoms[x] + 4, f"L = {L:.0f} ms", ha="center", fontsize=10.5, fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels(["legacy rasterizer", "fast rasterizer\n(current)"], fontsize=10.5)
ax.set_ylabel("freshness age p50 (ms)", fontsize=11)
ax.set_title("Uplink-only freshness age: capture $\\rightarrow$ spatial-map update\n"
             "(ideal loopback, no downlink return)", fontweight="bold", fontsize=12)
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, bottoms.max() * 1.18)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(PLOTS / "freshness_age_breakdown.pdf", bbox_inches="tight")
fig.savefig(PLOTS / "freshness_age_breakdown.png", dpi=200, bbox_inches="tight")
print(f"wrote {PLOTS/'freshness_age_breakdown.pdf'}")

with open(RESULTS / "L_anchors.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["anchor", "L_p50_ms", "L_p95_ms", "note"])
    w.writerow(["fast rasterizer (conservative design anchor; 40-frame profile)", f"{L_fast:.1f}",
                f"{by['capture_to_map_update_done_ms']['fast_p95_ms']:.1f}",
                "ideal loopback, uplink-only, no-AE u8, zstd, 200k PPS"])
    w.writerow(["legacy rasterizer (pre-optimization)", f"{L_legacy:.1f}",
                f"{by['capture_to_map_update_done_ms']['legacy_p95_ms']:.1f}",
                "same recipe, python per-point radar raster"])
    w.writerow(["core split->map (NOT the staleness lag)", f"{core_fast:.1f}",
                f"{by['backbone_input_to_map_update_done_ms']['fast_p95_ms']:.1f}",
                "backbone_input->map_update_done; excludes sensor prep"])
print(f"wrote {RESULTS/'L_anchors.csv'}")
