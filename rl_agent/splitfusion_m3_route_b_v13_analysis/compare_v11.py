"""Descriptive V11 GOOD_ALL vs V13 BAD_ALL comparison (offline, CPU only, create-only output).

Both captures are read through the SAME registered analyzer outputs (``DECISION_TIMELINE.csv``,
``FRAME_MAP_PAYLOAD.csv``, ``MAP_AOI_TIMELINE.csv``, ``METRICS.json``) and the same calibrated T-tracer readers over the
same route-window definition (first through last decision capture wall time).  The two runs are different sessions
and scenes on the same route; nothing here is a controlled A/B or a causal attribution to the channel.

    python3 -B -m rl_agent.splitfusion_m3_route_b_v13_analysis.compare_v11 \
        --v11-analysis <dir> --v11-capture <root> --v12-analysis <dir> --v12-capture <root> --out <new dir>

Outputs: COMPARISON.json / COMPARISON.csv (side-by-side metrics with denominators) and 01_v11_v13_comparison.png
(six small multiples, one y-axis each: PUSCH SNR, selected MCS, prediction completion, feedback receipt, L2 payload,
map AoI).
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import telemetry as T

SERIES = (("V11", "V11 GOOD_ALL 23.5 dB", "#2a78d6", "-"), ("V12", "V12 BAD_ALL 8.5 dB", "#eb6834", "--"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dc"


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def floats(values):
    out = []
    for value in values:
        try:
            if value not in ("", None):
                out.append(float(value))
        except ValueError:
            pass
    return out


def q(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return dict(n=0)
    return dict(n=int(a.size), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), mean=float(a.mean()))


def load(analysis, capture):
    analysis, capture = Path(analysis), Path(capture)
    timeline = rows(analysis / "DECISION_TIMELINE.csv")
    frames = rows(analysis / "FRAME_MAP_PAYLOAD.csv")
    aoi = rows(analysis / "MAP_AOI_TIMELINE.csv")
    metrics = json.loads((analysis / "METRICS.json").read_bytes())
    start = int(min(float(r["capture_wall_s"]) for r in timeline) * 1e9)
    end = int(max(float(r["capture_wall_s"]) for r in timeline) * 1e9)
    streams = T.load(capture / "capture/m3_radio/ttracer")
    pusch = [r["pusch_snr_db"] for r in streams["pusch"]["rows"] if start <= r["wall_ns"] <= end]
    mcs = [r["selected_mcs"] for r in streams["mcs"]["rows"] if start <= r["wall_ns"] <= end]
    timely = sum(r["timely_170"] == "True" for r in timeline)
    return dict(
        timeline=timeline, metrics=metrics,
        prediction=floats(r["prediction_completion_ms"] for r in timeline),
        feedback=floats(r["feedback_receipt_ms"] for r in timeline),
        payload=floats(r["payload_l2_bytes"] for r in frames if r["prepare_status"] == "SENT"),
        aoi=floats(r["aoi_ms"] for r in aoi), pusch=pusch, mcs=mcs,
        modes=collections.Counter(r["mode_id"] for r in timeline),
        summary=dict(decisions=len(timeline), timely_170=timely, timely_fraction=timely / len(timeline),
                     prediction_completion_ms=q(floats(r["prediction_completion_ms"] for r in timeline)),
                     feedback_receipt_ms=q(floats(r["feedback_receipt_ms"] for r in timeline)),
                     payload_l2_bytes=q(floats(r["payload_l2_bytes"] for r in frames if r["prepare_status"] == "SENT")),
                     payload_l2_total_bytes=metrics["payload"]["l2_total_bytes"],
                     sent_frames=metrics["payload"]["sent_frames"],
                     map_aoi_ms=q(floats(r["aoi_ms"] for r in aoi)),
                     map_age_at_install_ms=metrics["map"]["age_at_install_ms"],
                     pusch_snr_db=dict(q(pusch), p05=float(np.percentile(pusch, 5)) if pusch else None),
                     selected_mcs=dict(q(mcs), histogram={str(k): v for k, v in sorted(collections.Counter(mcs).items())}),
                     modes={k: v for k, v in sorted(collections.Counter(r["mode_id"] for r in timeline).items())},
                     q_e4_mode11=metrics["actions"]["q_e4_mode11"],
                     quality={c: {k: metrics["quality"][c][k] for k in ("eligible_gt", "within_tolerance",
                                                                          "recall_within_tolerance",
                                                                          "precision_3m_gate", "seg_iou_pooled")}
                              for c in ("vehicle", "person")},
                     quality_defined=metrics["quality"]["defined"], quality_undefined=metrics["quality"]["undefined_scored"],
                     route=dict(completed=metrics["route"]["completed"],
                                coverage=metrics["route"]["ordered_progress_coverage"]["coverage"],
                                interventions=metrics["route"]["summary"]["intervention_count_total"],
                                capture_span_s=metrics["route"]["capture_span_s"])))


def cdf(ax, values, label, color, style):
    a = np.sort(np.asarray(values, dtype=float))
    ax.plot(a, np.arange(1, a.size + 1) / a.size, color=color, ls=style, lw=2, label=label)


def figure(path, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#fcfcfb")
    panels = [
        ("Calibrated PUSCH SNR per transmission (route window)", "dB", "pusch", "hist"),
        ("Scheduler selected UL MCS (route window)", "MCS index", "mcs", "bar"),
        ("Prediction completion (all decisions)", "ms", "prediction", "cdf"),
        ("Feedback receipt at UE (all decisions)", "ms", "feedback", "cdf"),
        ("L2 application payload per sent frame", "kB", "payload", "cdf"),
        ("Map age of information", "ms", "aoi", "cdf")]
    for ax, (title, unit, key, kind) in zip(axes.flat, panels):
        ax.set_facecolor("#fcfcfb")
        for index, (label, name, color, style) in enumerate(SERIES):
            values = data[label][key]
            if key == "payload":
                values = [v / 1000.0 for v in values]
            if kind == "cdf":
                cdf(ax, values, name, color, style)
            elif kind == "hist":
                bins = np.arange(min(min(data[s[0]][key]) for s in SERIES) - 0.25,
                                 max(max(data[s[0]][key]) for s in SERIES) + 0.5, 0.5)
                counts, edges = np.histogram(values, bins=bins)
                ax.bar(edges[:-1] + 0.25 + (-0.12 if index == 0 else 0.12), counts / max(1, len(values)),
                       width=0.22, color=color, label=name, edgecolor="#fcfcfb", linewidth=1)
            else:
                counter = collections.Counter(values)
                keys = sorted(counter)
                offset = -0.2 if index == 0 else 0.2
                ax.bar([k + offset for k in keys], [counter[k] / max(1, len(values)) for k in keys], width=0.38,
                       color=color, label=name, edgecolor="#fcfcfb", linewidth=2)
        if key in ("prediction", "feedback"):
            ax.axvline(170, color=MUTED, lw=1, ls=":")
            ax.text(171, 0.05, "170 ms deadline", color=MUTED, fontsize=9)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel(unit, color=MUTED)
        ax.set_ylabel("cumulative fraction" if kind == "cdf" else "fraction of rows", color=MUTED)
        ax.grid(True, color=GRID, lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=11)
    fig.suptitle("M3 Route B: V11 GOOD_ALL vs V13 BAD_ALL (different sessions/scenes; descriptive, not a controlled A/B)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v11-analysis", "v11-capture", "v12-analysis", "v12-capture", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    data = dict(V11=load(args.v11_analysis, args.v11_capture), V13=load(args.v12_analysis, args.v12_capture))
    document = dict(schema="splitfusion.m3_route_b_v13.v11_comparison.v1",
                    caveat="different sessions and scenes on the same route; descriptive only",
                    v11=data["V11"]["summary"], v13=data["V12"]["summary"],
                    inputs={label: {name: hashlib.sha256((Path(getattr(args, f"{label.lower()}_analysis")) / name)
                                                         .read_bytes()).hexdigest()
                                    for name in ("DECISION_TIMELINE.csv", "FRAME_MAP_PAYLOAD.csv",
                                                 "MAP_AOI_TIMELINE.csv", "METRICS.json")} for label in ("V11", "V12")})
    (args.out / "COMPARISON.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    flat = []
    for metric, getter in (
            ("decisions", lambda s: s["decisions"]), ("timely_170", lambda s: s["timely_170"]),
            ("timely_fraction", lambda s: s["timely_fraction"]),
            ("prediction_completion_p50_ms", lambda s: s["prediction_completion_ms"]["p50"]),
            ("prediction_completion_p95_ms", lambda s: s["prediction_completion_ms"]["p95"]),
            ("prediction_completion_p99_ms", lambda s: s["prediction_completion_ms"]["p99"]),
            ("feedback_receipt_p50_ms", lambda s: s["feedback_receipt_ms"]["p50"]),
            ("feedback_receipt_p95_ms", lambda s: s["feedback_receipt_ms"]["p95"]),
            ("feedback_receipt_p99_ms", lambda s: s["feedback_receipt_ms"]["p99"]),
            ("payload_l2_p50_bytes", lambda s: s["payload_l2_bytes"]["p50"]),
            ("payload_l2_p95_bytes", lambda s: s["payload_l2_bytes"]["p95"]),
            ("payload_l2_p99_bytes", lambda s: s["payload_l2_bytes"]["p99"]),
            ("payload_l2_total_bytes", lambda s: s["payload_l2_total_bytes"]),
            ("sent_frames", lambda s: s["sent_frames"]),
            ("map_aoi_p50_ms", lambda s: s["map_aoi_ms"]["p50"]), ("map_aoi_p95_ms", lambda s: s["map_aoi_ms"]["p95"]),
            ("pusch_snr_p05_db", lambda s: s["pusch_snr_db"]["p05"]),
            ("pusch_snr_p50_db", lambda s: s["pusch_snr_db"]["p50"]),
            ("pusch_snr_p95_db", lambda s: s["pusch_snr_db"]["p95"]),
            ("selected_mcs_p50", lambda s: s["selected_mcs"]["p50"]),
            ("q_e4_mode11_p50", lambda s: s["q_e4_mode11"].get("p50")),
            ("vehicle_recall_1p0", lambda s: s["quality"]["vehicle"]["recall_within_tolerance"]),
            ("person_recall_1p2", lambda s: s["quality"]["person"]["recall_within_tolerance"]),
            ("vehicle_precision_3m", lambda s: s["quality"]["vehicle"]["precision_3m_gate"]),
            ("person_precision_3m", lambda s: s["quality"]["person"]["precision_3m_gate"]),
            ("vehicle_seg_iou", lambda s: s["quality"]["vehicle"]["seg_iou_pooled"]),
            ("person_seg_iou", lambda s: s["quality"]["person"]["seg_iou_pooled"])):
        flat.append(dict(metric=metric, v11=getter(data["V11"]["summary"]), v13=getter(data["V12"]["summary"])))
    with (args.out / "COMPARISON.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "v11", "v12"])
        writer.writeheader()
        writer.writerows(flat)
    figure(args.out / "01_v11_v13_comparison.png", data)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir()) if p.is_file()}
    (args.out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({row["metric"]: [row["v11"], row["v12"]] for row in flat}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
