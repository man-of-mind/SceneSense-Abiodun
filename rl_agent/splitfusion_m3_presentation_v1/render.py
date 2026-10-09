"""Read-only M3 presentation renderer. No runtime/model imports or inference.

Reads compact, sealed analysis exports. All figure data and definitions are
exported beside the PNGs. Existing capture, report and analysis files are never
written. Run with one CPU thread and low scheduling priority during live work.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INK, MUTED, GRID = "#132C45", "#52677A", "#E5EBF0"
BLUE, TEAL, ORANGE, PURPLE = "#147ABB", "#078778", "#D86624", "#8058A5"
RED, GOLD = "#BE354B", "#B78213"
MODE_COLORS = {8: PURPLE, 9: GOLD, 10: BLUE, 11: TEAL}
DPI = 240


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    return float(value) if value not in (None, "", "None") else float("nan")


def array(rows, key):
    return np.asarray([number(r.get(key)) for r in rows])


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(n=len(values), **{f"p{q}": float(np.percentile(values, q)) for q in (50, 95, 99)},
                max=float(np.max(values))) if len(values) else dict(n=0)


def ecdf(values, denominator=None):
    """Incomplete records remain in the denominator; no latency is imputed."""
    values = np.asarray(values, dtype=float)
    n = len(values) if denominator is None else denominator
    x = np.sort(values[np.isfinite(values)])
    return np.r_[0.0, x], np.r_[0.0, np.arange(1, len(x) + 1) / n]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
        "font.weight": "medium", "axes.titlesize": 20, "axes.titleweight": "bold",
        "axes.labelsize": 17, "axes.labelweight": "bold", "xtick.labelsize": 14,
        "ytick.labelsize": 14, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#C4CFD8",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.8,
        "legend.frameon": False, "legend.fontsize": 14, "savefig.facecolor": "white"})


class Deck:
    def __init__(self, args):
        self.a = args
        self.out = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.analysis, self.layers = args.analysis, args.layers
        self.paths = [self.analysis / n for n in ("METRICS.json", "DECISION_TIMELINE.csv",
            "FRAME_MAP_PAYLOAD.csv", "LOCALIZATION_CDF.csv", "LATENCY_CDF.csv", "MAP_AOI_TIMELINE.csv",
            "OUTPUT_SHA256.json")]
        self.paths += [self.layers / "ROUTE_LAYERS_JOINED.csv", self.layers / "PROVENANCE.json",
            ROOT / "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.progress.csv",
            Path(__file__).resolve()]
        self.hashes = {str(p.resolve()): sha(p) for p in self.paths}
        manifest = json.loads((self.analysis / "OUTPUT_SHA256.json").read_text())
        for p in self.paths:
            if p.parent == self.analysis and p.name in manifest:
                assert sha(p) == manifest[p.name], f"Sealed source changed: {p}"
        provenance = json.loads((self.layers / "PROVENANCE.json").read_text())
        assert sha(self.layers / "ROUTE_LAYERS_JOINED.csv") == provenance["outputs_sha256"]["ROUTE_LAYERS_JOINED.csv"]
        self.m = json.loads((self.analysis / "METRICS.json").read_text())
        self.dec = read_csv(self.analysis / "DECISION_TIMELINE.csv")
        self.frames = read_csv(self.analysis / "FRAME_MAP_PAYLOAD.csv")
        self.sent = [r for r in self.frames if r["prepare_status"] == "SENT"]
        self.loc = read_csv(self.analysis / "LOCALIZATION_CDF.csv")
        self.aoi_rows = read_csv(self.analysis / "MAP_AOI_TIMELINE.csv")
        self.layer = read_csv(self.layers / "ROUTE_LAYERS_JOINED.csv")
        by_id = {int(r["frame_id"]): r for r in self.layer}
        assert len(by_id) == len(self.layer) == len(self.dec)
        self.layer = [by_id[int(r["frame_id"])] for r in self.dec]
        self.t = array(self.dec, "t_since_first_capture_s")
        self.t0 = number(self.dec[0]["capture_wall_s"])
        ids = [int(r["frame_id"]) for r in self.sent]
        assert len(set(ids)) == len(ids) and ids == sorted(ids)
        ordinal = {fid: i + 1 for i, fid in enumerate(ids)}
        self.frame_x = np.asarray([ordinal[int(r["frame_id"])] for r in self.dec])
        self.xy = np.column_stack([array(self.dec, "ego_x"), array(self.dec, "ego_y")])
        route = read_csv(self.paths[-2])
        self.route = np.column_stack([array(route, "ego_x"), array(route, "ego_y")])
        self.pred = array(self.dec, "prediction_completion_ms")
        self.feedback = array(self.dec, "feedback_receipt_ms")
        self.timely = np.isfinite(self.feedback) & (self.feedback <= 170)
        self.late = np.isfinite(self.feedback) & (self.feedback > 170)
        self.missing = ~np.isfinite(self.feedback)
        self.age = array(self.sent, "map_age_at_install_ms")
        self.aoi = array(self.aoi_rows, "aoi_ms")
        self.install_t = array(self.sent, "install_wall_s") - self.t0
        order = np.argsort(self.install_t)
        self.install_t, self.age = self.install_t[order], self.age[order]
        self.gaps = np.diff(self.install_t) * 1000
        self.payload = array(self.sent, "payload_l2_bytes") / 1000
        self.mode = array(self.dec, "mode_id").astype(int)
        self.q = array(self.dec, "q_e4") / 10000
        self.snr = array(self.layer, "measured_pusch_snr_db")
        self.mcs = array(self.layer, "measured_final_mcs")
        self.si, self.p40 = array(self.layer, "camera_si"), array(self.layer, "radar_p40")
        self.highlight = None
        if args.highlight_window is not None:
            start, end = args.highlight_window
            assert 0 <= start < end <= self.t[-1], "Highlight must lie inside observed route"
            self.highlight = (self.t >= start) & (self.t < end)
            assert self.highlight.any()
        self.gc_stop_t = None
        self.gc_trace = None
        if args.gc_trace is not None:
            assert args.radio_metadata is not None, "GC raw clock requires the recorded wall/raw anchor"
            for path in (args.gc_trace, args.radio_metadata):
                self.paths.append(path)
                self.hashes[str(path.resolve())] = sha(path)
            self.gc_trace = json.loads(args.gc_trace.read_text())
            stop = self.gc_trace.get("safety_stop")
            if stop:
                anchor = json.loads(args.radio_metadata.read_text())["clock_anchor"]
                self.gc_stop_t = (int(anchor["wall_ns"]) + int(stop["raw_ns"]) - int(anchor["raw_ns"])) / 1e9 - self.t0
                assert 0 < self.gc_stop_t < self.t[-1]
        self.figs = []
        self.summary = {}
        self.validate()

    def validate(self):
        m = self.m
        assert len(self.dec) == m["decisions"]["policy_decisions"]
        assert len(self.sent) == m["counts"]["transmitted"]
        assert np.all(np.diff(self.t) > 0) and np.all(np.diff(self.frame_x) > 0)
        assert int(self.timely.sum()) == m["latency_ms"]["timely_170"]
        assert int((self.pred <= 170).sum()) == m["latency_ms"]["prediction_completed_by_170"]
        for d, l in zip(self.dec, self.layer):
            for key in ("frame_id", "decision_index", "mode_id", "q_e4", "ego_x", "ego_y", "capture_wall_s"):
                assert number(d[key]) == number(l[key]), key
            assert number(d["payload_l2_bytes"]) == number(l["application_payload_bytes"])
        for cls in ("vehicle", "person"):
            rows = [r for r in self.loc if r["class"] == cls]
            q = m["quality"][cls]
            assert int(rows[0]["eligible_gt"]) == q["eligible_gt"]
            assert np.all(np.diff(array(rows, "fraction_of_all_eligible_gt")) >= 0)
            hit = [r for r in rows if number(r["threshold_m"]) == q["tolerance_m"]]
            assert len(hit) == 1 and abs(number(hit[0]["fraction_of_all_eligible_gt"]) - q["recall_within_tolerance"]) < 1e-12
            assert q["matched_3m"] + q["misses"] == q["eligible_gt"]
            assert sum(int(r[cls + "_eligible"] or 0) for r in self.dec) == q["eligible_gt"]
        for vals, expected in ((self.pred, m["latency_ms"]["prediction_completion"]),
            (self.feedback, m["latency_ms"]["feedback_receipt"]), (self.age, m["map"]["age_at_install_ms"]),
            (self.aoi, m["map"]["aoi_ms"])):
            for key in ("p50", "p95", "p99"):
                assert abs(quantiles(vals)[key] - expected[key]) < 1e-8, key
        assert abs(quantiles(self.payload)["p99"] * 1000 - m["payload"]["l2_bytes"]["p99"]) < 1e-8
        assert np.isfinite(self.snr).all() and np.isfinite(self.si).all()
        assert np.all((self.p40 >= 0) & (self.p40 <= 1))
        assert np.all((self.q >= 0) & (self.q <= 1))
        self.summary = dict(decisions=len(self.dec), sent_frames=len(self.sent),
            prediction_by_170=int((self.pred <= 170).sum()), timely_feedback=int(self.timely.sum()),
            late_feedback=int(self.late.sum()), missing_feedback=int(self.missing.sum()),
            prediction_ms=quantiles(self.pred), feedback_ms=quantiles(self.feedback),
            map_install_age_ms=quantiles(self.age), map_aoi_ms=quantiles(self.aoi),
            map_update_gaps_ms=quantiles(self.gaps), payload_kB=quantiles(self.payload),
            sampled_aoi_over_500=int((self.aoi > 500).sum()),
            map_install_age_over_500=int((self.age > 500).sum()),
            route_quality={c: self.m["quality"][c] for c in ("vehicle", "person")},
            policy_modes={str(k): int((self.mode == k).sum()) for k in sorted(set(self.mode))},
            measured_snr_at_decisions=quantiles(self.snr), checks="PASS")
        if self.highlight is not None:
            self.summary["exploratory_highlight"] = dict(
                window_s=self.a.highlight_window, label="A", selected=self.subset_stats(self.highlight),
                whole_route=self.subset_stats(np.ones(len(self.dec), dtype=bool)),
                interpretation="Post-hoc illustrative window, not a causal test or independently selected validation window")
        if self.gc_stop_t is not None:
            before = self.t < self.gc_stop_t
            self.summary["gc_safety_stop_context"] = dict(
                elapsed_wall_s=self.gc_stop_t, source_stop=self.gc_trace["safety_stop"],
                partition="Decision frame captured before vs at/after recorded safety-stop time; not a randomized comparison",
                before=self.subset_stats(before), after=self.subset_stats(~before),
                caveat="A frame captured before the stop may still be processing when it occurs. Route/scene/time differ between subsets.",
                in_request_gen2=sum(e.get("op") is not None for e in self.gc_trace["gen2"]))

    def subset_stats(self, mask):
        n = int(mask.sum())
        return dict(decisions=n, timely=int(self.timely[mask].sum()), late=int(self.late[mask].sum()),
                    timely_fraction=float(self.timely[mask].mean()),
                    prediction_ms=quantiles(self.pred[mask]), feedback_ms=quantiles(self.feedback[mask]),
                    gt_publication_ms=quantiles(array(self.dec, "gt_published_ms")[mask]),
                    camera_si=quantiles(self.si[mask]), radar_p40=quantiles(self.p40[mask]),
                    first_frame=int(self.frame_x[mask][0]), last_frame=int(self.frame_x[mask][-1]))

    def context_band(self, ax, use_frames=False, label=False):
        """Presentation markers only; do not filter any metric or observation."""
        if self.a.hide_context_markers:
            return
        if self.highlight is not None:
            bounds = self.frame_x[self.highlight][[0, -1]] if use_frames else self.a.highlight_window
            ax.axvspan(*bounds, color=GOLD, alpha=.12, lw=0, zorder=0)
            for bound in bounds:
                ax.axvline(bound, color=GOLD, lw=.8, ls=":", alpha=.8)
            if label:
                start, end = self.a.highlight_window
                ax.text(float(np.mean(bounds)), 1.04, f"A · {start:g}–{end:g} s", transform=ax.get_xaxis_transform(),
                        ha="center", color=GOLD, weight="bold", size=11)
        if self.gc_stop_t is not None:
            pos = float(np.interp(self.gc_stop_t, self.t, self.frame_x)) if use_frames else self.gc_stop_t
            ax.axvline(pos, color=RED, ls="--", lw=1.2, zorder=4)
            if label:
                ax.text(pos, 1.04, f"GC safety stop · {self.gc_stop_t:.0f} s", transform=ax.get_xaxis_transform(),
                        ha="right", color=RED, weight="bold", size=11)

    def route_highlight(self, ax):
        if self.a.hide_context_markers or self.highlight is None:
            return
        coords = self.xy[self.highlight]
        low, high = coords.min(axis=0) - 4, coords.max(axis=0) + 4
        ax.add_patch(Rectangle(low, *(high-low), fill=False, edgecolor=GOLD, lw=2, ls="--", zorder=7))
        ax.annotate("A", (high[0], high[1]), xytext=(10, 10), textcoords="offset points",
                    color=GOLD, weight="bold", size=16, zorder=9,
                    bbox=dict(facecolor="white", edgecolor=GOLD, boxstyle="round,pad=.2"))

    def context_note(self):
        if self.a.hide_context_markers:
            return ""
        if self.highlight is not None:
            s = self.summary["exploratory_highlight"]["selected"]
            start, end = self.a.highlight_window
            return (f"A ({start:g}–{end:g} s): {s['late']}/{s['decisions']} late; median P40 {s['radar_p40']['p50']:.2f}, "
                    f"SI {s['camera_si']['p50']:.1f}. Exploratory association, not causal attribution.")
        if self.gc_stop_t is not None:
            return f"Dashed red line: GC safety stop at {self.gc_stop_t:.1f} s; standard GC restored. All subsequent observations remain included."
        return ""

    def new(self, title, subtitle=""):
        fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor="white")
        fig.text(.055, .957, title, size=29, weight="bold", va="top")
        fig.text(.055, .902, subtitle, size=15, color=MUTED, va="top")
        fig.text(.945, .97, self.a.label, size=11, ha="right", color=MUTED)
        fig.add_artist(Line2D([.055, .945], [.878, .878], transform=fig.transFigure, color=GRID, lw=1.1))
        return fig

    def finish(self, fig, name, note):
        fig.text(.055, .047, note, size=11.5, color=MUTED, va="top", linespacing=1.35)
        fig.canvas.draw()
        # Main text must fit the canvas; data annotations are inspected visually too.
        renderer = fig.canvas.get_renderer()
        text_items = list(fig.texts)
        for ax in fig.axes:
            text_items.extend([ax.xaxis.label, ax.yaxis.label, ax.title])
        for text in text_items:
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, text.get_text()
        fig.savefig(self.out / name, dpi=DPI, metadata={"Title": fig.texts[0].get_text(),
            "Description": note, "Source": str(self.analysis)})
        self.figs.append(name)
        plt.close(fig)
        print("Rendered " + name, flush=True)

    @staticmethod
    def axis(ax, percent=False):
        ax.grid(axis="both", color=GRID, lw=.8, zorder=0)
        ax.set_axisbelow(True)
        if percent:
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.yaxis.set_major_locator(MultipleLocator(.2))

    def completion(self):
        n = len(self.dec)
        fig = self.new("Prediction completion and feedback delivery", f"Full Route B  ·  {n:,} policy decisions  ·  170 ms feedback deadline")
        ax = fig.add_axes([.08, .18, .57, .61])
        for values, color, label in ((self.pred, BLUE, "Prediction ready at edge"), (self.feedback, ORANGE, "Feedback received at UE")):
            x, y = ecdf(values)
            ax.step(x, y, where="post", color=color, label=label, lw=3.3)
            ax.scatter([170], [np.sum(values <= 170) / n], color=color, s=65, zorder=4)
        ax.axvline(170, color=INK, ls="--", lw=1.7)
        ax.set(xlim=(70, max(430, np.nanmax(self.feedback) + 10)), xlabel="Time since action opened (ms)", ylabel="Fraction of all policy decisions")
        self.axis(ax, True)
        ax.legend(loc="lower right", fontsize=13.5)
        for top, color, title, value in ((.76, BLUE, "PREDICTION BY 170 ms", (self.pred <= 170).sum()),
                                       (.56, ORANGE, "FEEDBACK BY 170 ms", self.timely.sum())):
            fig.text(.70, top, title, size=13, weight="bold", color=color)
            fig.text(.70, top - .073, f"{value / n:.1%}", size=35, weight="bold", color=color)
            fig.text(.70, top - .12, f"{value:,} / {n:,} decisions", size=14, color=MUTED)
        fig.text(.70, .32, "Median / P99", size=14, weight="bold")
        fig.text(.70, .26, f"Prediction   {np.nanmedian(self.pred):.1f} / {np.nanpercentile(self.pred,99):.1f} ms\nFeedback    {np.nanmedian(self.feedback):.1f} / {np.nanpercentile(self.feedback,99):.1f} ms", size=13.5, linespacing=1.7)
        self.finish(fig, "01_completion_cdf.png", "Prediction ready = edge.compute completion with CPU outputs materialized; feedback additionally includes evaluation and return.\nAll decisions remain in the denominator. The 170 ms line is the feedback deadline, not a separately qualified prediction SLA.")

    def freshness_cdf(self):
        fig = self.new("Map freshness", "Age when an update arrives, and age of the latest map between updates")
        ax = fig.add_axes([.08, .18, .58, .62])
        for values, color, name in ((self.age, TEAL, f"Age at installation · {len(self.age):,} updates"),
                                   (self.aoi, PURPLE, f"Map age of information · {len(self.aoi):,} time samples")):
            x, y = ecdf(values)
            ax.step(x, y, where="post", color=color, label=name, lw=3.3)
        self.axis(ax, True)
        ax.set(xlim=(0, max(570, np.nanmax(self.aoi) + 15)), xlabel="Age since frame capture (ms)", ylabel="Cumulative fraction")
        ax.axvline(500, color=MUTED, ls="--", lw=1.5)
        ax.text(492, .12, "500 ms installation-age limit", rotation=90, ha="right", va="bottom", size=11.5, color=MUTED)
        ax.legend(loc="lower right", fontsize=12.5)
        for top, values, color, title in ((.74, self.age, TEAL, "AGE AT INSTALLATION"), (.51, self.aoi, PURPLE, "MAP AGE OF INFORMATION")):
            fig.text(.71, top, title, size=12.5, weight="bold", color=color)
            fig.text(.71, top - .075, f"{np.nanmedian(values):.1f} ms", size=30, weight="bold", color=color)
            fig.text(.71, top - .125, f"P99  {np.nanpercentile(values,99):.1f} ms", size=16, color=MUTED)
        fig.text(.71, .25, f"{self.m['map']['gaps_over_500ms']} update gaps > 500 ms", size=18, weight="bold")
        self.finish(fig, "02_map_freshness_cdf.png", "Installation age is capture-to-install delay. AoI is sampled every 100 ms and grows between updates; it can exceed installation age.\nThe 500 ms limit governs admission at installation, not an AoI guarantee. Prediction/feedback latencies use a different start boundary.")

    def quality_cdf(self):
        fig = self.new("Localisation accuracy, including missed objects", "Conservative CDFs  ·  All scored decision frames, including late feedback")
        for left, cls, title, color in ((.08, "vehicle", "Vehicles", BLUE), (.57, "person", "Pedestrians", PURPLE)):
            ax = fig.add_axes([left, .20, .37, .56])
            q = self.m["quality"][cls]
            rows = [r for r in self.loc if r["class"] == cls]
            x, y = array(rows, "threshold_m"), array(rows, "fraction_of_all_eligible_gt")
            ax.step(x, y, where="post", lw=3.5, color=color)
            ax.axvline(q["tolerance_m"], ls="--", color=INK, lw=1.4)
            ax.scatter([q["tolerance_m"]], [q["recall_within_tolerance"]], s=75, color=color, zorder=4)
            ax.annotate(f"{q['recall_within_tolerance']:.1%} within {q['tolerance_m']:.1f} m", xy=(q["tolerance_m"], q["recall_within_tolerance"]),
                        xytext=(12, -35), textcoords="offset points", color=color, weight="bold", size=15,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.95, pad=3))
            ax.set(xlim=(0, 3), xlabel="Localisation error threshold (m)", ylabel="Fraction of eligible GT objects")
            ax.set_title(title + f"  ·  {q['eligible_gt']:,} GT instances", pad=20)
            ax.text(.98, .04, f"{q['misses']:,} unmatched at the 3 m gate", transform=ax.transAxes, ha="right", size=12.5, color=MUTED)
            self.axis(ax, True)
        self.finish(fig, "03_localisation_cdf.png", "Missed eligible objects stay in the denominator; curves need not reach 100%. Repeated objects across frames are object-frame instances.\nNo full-local baseline is inferred for this route. Frames with no eligible localisation GT contribute no objects to these denominators.")

    def quality_bars(self):
        fig = self.new("Segmentation overlap and detection precision", "All scored decision frames  ·  Vehicle and pedestrian results use their own pooled denominators")
        specs = ((.08, "seg_iou_pooled", "Segmentation IoU", "Intersection / union"),
                 (.57, "precision_3m_gate", "Object-detection precision", "Matched / (matched + false positives)"))
        for left, key, title, subtitle in specs:
            ax = fig.add_axes([left, .22, .36, .52])
            vals = [self.m["quality"][c][key] for c in ("vehicle", "person")]
            bars = ax.bar([0, 1], vals, color=[BLUE, PURPLE], width=.52, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + .025, f"{v:.3f}", ha="center", weight="bold", size=24)
            ax.set(xticks=[0, 1], xticklabels=["Vehicle", "Pedestrian"], ylabel="Score", xlim=(-.65, 1.65))
            ax.set_title(title, pad=34)
            ax.text(.5, 1.025, subtitle, transform=ax.transAxes, ha="center", size=12, color=MUTED)
            self.axis(ax, True)
            ax.set_ylim(0, 1.12)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
        self.finish(fig, "04_segmentation_and_precision.png", "Precision is object-detection precision using the registered 3 m matching gate, NOT pixel segmentation precision.\nThese are all-scored quality results, not deadline-weighted quality; late predictions are included and remain map-eligible under freshness rules.")

    def payload_bars(self):
        p50, p99 = np.percentile(self.payload, [50, 99])
        fig = self.new("Transmitted payload size", f"Deployed agent  ·  All {len(self.sent):,} sent frames, including holds and fallbacks")
        ax = fig.add_axes([.10, .20, .51, .59])
        bars = ax.bar([0, 1], [p50, p99], width=.51, color=[TEAL, BLUE], zorder=3)
        for bar, v in zip(bars, [p50, p99]):
            ax.text(bar.get_x()+bar.get_width()/2, v+6, f"{v:.1f} kB", ha="center", size=27, weight="bold")
        ax.set(xticks=[0, 1], xticklabels=["P50 · median", "P99"], ylabel="Application-frame payload (kB)", xlim=(-.65, 1.65), ylim=(0, p99*1.26))
        self.axis(ax)
        ax.grid(axis="x", visible=False)
        fig.text(.70, .69, f"{np.sum(self.payload)/1000:.1f} MB", size=35, weight="bold", color=TEAL)
        fig.text(.70, .62, "Total application traffic", size=17, color=MUTED)
        fig.text(.70, .49, f"{len(self.dec):,} decisions\n{self.m['decisions']['holds']:,} held-action frames\n{self.m['decisions']['fallbacks']:,} fallback frames", size=19, linespacing=1.7)
        self.finish(fig, "05_payload_p50_p99.png", "Byte layer: complete SFD4 application frame (the archived L2 payload field). Radio and network encapsulation overhead are excluded.\nDecimal units: 1 kB = 1,000 bytes; 1 MB = 1,000,000 bytes. These are measured agent sizes, not a baseline comparison.")

    def route_base(self, ax):
        ax.plot(*self.route.T, color="#CCD6DE", lw=7, zorder=1, solid_capstyle="round")
        ax.scatter(*self.xy[0], s=140, marker="*", color=INK, edgecolor="white", lw=.9, zorder=8)
        low = np.minimum(self.route.min(axis=0), self.xy.min(axis=0)) - 18
        high = np.maximum(self.route.max(axis=0), self.xy.max(axis=0)) + 18
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="CARLA world x (m)", ylabel="CARLA world y (m)")
        ax.set_aspect("equal")
        self.axis(ax)

    def route_outcomes(self):
        fig = self.new("Where feedback met the deadline", f"Route B  ·  {self.m['route']['ordered_progress_coverage']['matched']}/301 ordered points covered  ·  50 vehicles / 50 pedestrians")
        ax = fig.add_axes([.05, .34, .55, .49])
        self.route_base(ax)
        for mask, color, size in ((self.timely, TEAL, 15), (self.late, ORANGE, 27), (self.missing, RED, 32)):
            ax.scatter(*self.xy[mask].T, color=color, s=size, lw=.2, edgecolor="white", zorder=3)
        self.route_highlight(ax)
        for frac in (.25, .5, .75):
            i = int(frac * (len(self.dec)-1))
            ax.annotate(f"f{self.frame_x[i]:,}", self.xy[i], xytext=(9, 11), textcoords="offset points",
                        size=10, color=INK, weight="bold", bbox=dict(facecolor="white", edgecolor=GRID, alpha=.95, pad=2),
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7))
        for top, mask, color, title in ((.75, self.timely, TEAL, "Feedback ≤ 170 ms"), (.59, self.late, ORANGE, "Late feedback / controller timeout"),
                                      (.43, self.missing, RED, "No feedback observed")):
            fig.text(.60, top, title, color=color, weight="bold", size=17)
            fig.text(.60, top-.06, f"{int(mask.sum()):,} / {len(mask):,}  ({mask.mean():.1%})", size=25, weight="bold", color=color)
        raster = fig.add_axes([.095, .155, .84, .07])
        for mask, color, y in ((self.timely, TEAL, 1), (self.late, ORANGE, 0), (self.missing, RED, -1)):
            raster.scatter(self.frame_x[mask], np.full(mask.sum(), y), marker="|", s=70, color=color, lw=.8)
        raster.set(ylim=(-.6, 1.6), yticks=[0, 1], yticklabels=["Late", "Timely"],
                   xlim=(1, len(self.sent)), xlabel="Transmitted frame index (1-based)")
        raster.grid(axis="x", color=GRID)
        self.context_band(raster, use_frames=True)
        fig.text(.095, .25, "Grey line: planned route   ★ start / return area   f = transmitted-frame index", size=11.5, color=MUTED)
        route = self.m["route"]["summary"]
        context = self.context_note() or "Holds/fallbacks are not feedback decisions; the bottom strip preserves frame order."
        self.finish(fig, "06_route_feedback_outcomes.png", f"All late-feedback predictions were map-installed. Overlapping positions can hide repeated visits. Roadblock removals: {route['npc_roadblocks_cleared_total']}; collision incidents: {route['collision_incident_count_total']}.\n" + context)

    def route_pair(self, title, subtitle, fields, filename, note):
        fig = self.new(title, subtitle)
        for left, (values, heading, color_label, cmap, limits, ticks) in zip((.08, .57), fields):
            ax = fig.add_axes([left, .27, .36, .48])
            self.route_base(ax)
            points = ax.scatter(*self.xy.T, c=values, s=22, cmap=cmap, norm=Normalize(*limits), edgecolors="none", zorder=3)
            self.route_highlight(ax)
            ax.set_title(heading, pad=18)
            box = ax.get_position()
            cax = fig.add_axes([box.x0, .15, box.width, .021])
            bar = fig.colorbar(points, cax=cax, orientation="horizontal", ticks=ticks)
            bar.outline.set_visible(False)
            bar.set_label(color_label, size=13)
        if self.highlight is not None and not self.a.hide_context_markers:
            note = note.split("\n")[0] + "\n" + self.context_note()
        self.finish(fig, filename, note)

    def route_layers(self):
        self.route_pair("Measured channel along Route B", "Recorded values at policy-decision positions  ·  Grey: planned route  ·  Star: start / return area", [
            (self.snr, "Measured PUSCH SNR", "PUSCH snrx10 / 10 (dB)", "viridis", (5.5, 24.5), [5.5, 10, 15, 20, 24.5]),
            (self.mcs, "Scheduler uplink MCS", "Final UL MCS index", "plasma", (0, 28), [0, 7, 14, 21, 28])],
            "07_route_snr_mcs.png", "PUSCH = per-transmission gNB measurement, joined from the latest preceding transmission at each decision. Not the MAC power-control statistic.\nMeasured SNR is diagnostic only; the actor receives the commanded proxy and prior UE MCS. Fixed colour scales support comparison across M3 conditions.")
        self.route_pair("Scene descriptors along Route B", "Decoded from the agent's recorded states  ·  Independent units and colour scales", [
            (self.si, "Camera spatial information (SI)", "SI (luma-gradient units)", "viridis", (60, max(180, float(np.ceil(self.si.max()/20)*20))), [60, 100, 140, 180]),
            (self.p40, "Radar proximity (P40)", "P40 (larger = nearer returns)", "magma", (0, 1), [0, .25, .5, .75, 1])],
            "08_route_si_p40.png", "SI: standard deviation of Sobel gradient magnitude on 768 × 448 luma. P40: mean clip(1 − range / 40 m, 0, 1) over valid radar returns.\nThey are scene descriptors, not object counts or causal explanations of individual decisions. Colours appear only at recorded decision positions.")

    def timeline(self, use_frames=False):
        x = self.frame_x if use_frames else self.t
        label = "Transmitted frame index (1-based)" if use_frames else "Elapsed wall time from first policy-frame capture (s)"
        fig = self.new("Actions, channel and scene — aligned", f"{len(self.dec):,} decisions  ·  Mode 11 (AE32 / UINT4): {(self.mode == 11).mean():.1%}  ·  Shared horizontal axis; no smoothing")
        axes = fig.subplots(6, 1, sharex=True, gridspec_kw=dict(left=.145, right=.947, bottom=.125, top=.83, hspace=.23))
        panels = (("Mode", self.mode, TEAL), ("Drop fraction\nq", self.q, TEAL),
                  ("Payload\n(kB)", array(self.dec, "payload_l2_bytes")/1000, BLUE),
                  ("PUSCH SNR\n(dB)", self.snr, PURPLE), ("Radar P40", self.p40, ORANGE), ("Camera SI", self.si, BLUE))
        for i, (ax, (ylabel, values, color)) in enumerate(zip(axes, panels)):
            ax.set_ylabel(ylabel, size=13, rotation=0, ha="right", va="center", labelpad=16)
            ax.tick_params(labelsize=11)
            ax.grid(color=GRID, lw=.7)
            if i == 0:
                for mode in sorted(set(self.mode)):
                    selected = self.mode == mode
                    ax.scatter(x[selected], values[selected], s=8, color=MODE_COLORS.get(mode, MUTED), zorder=3)
                ax.set_ylim(min(self.mode)-.45, max(self.mode)+.7)
                ax.set_yticks(sorted(set(self.mode)))
            elif i == 3:
                ax.scatter(x, values, s=4, color=color, alpha=.55, label="Measured PUSCH")
                ax.plot(x, array(self.dec, "snr_proxy_db"), color=INK, lw=1.1, ls="--", label="Commanded / actor proxy")
                ax.set_ylim(5.5, 25.5)
                ax.set_yticks([8.5, 15, 23.5])
                ax.legend(loc="lower left" if np.median(self.snr) >= 15 else "upper left",
                          ncol=2, fontsize=10, borderaxespad=.2)
            else:
                ax.plot(x, values, color=color, lw=.85, alpha=.85)
                ax.scatter(x, values, color=color, s=2, alpha=.35)
                if i in (1, 4):
                    ax.set_ylim(0, 1)
                    ax.set_yticks([0, .5, 1])
                if i == 2:
                    ax.set_ylim(0, max(300, np.max(values)*1.08))
                    ax.set_yticks([0, 150, 300])
            ax.tick_params(axis="x", labelbottom=(i == 5))
            self.context_band(ax, use_frames=use_frames, label=(i == 0))
        axes[-1].set_xlabel(label, size=15)
        axes[-1].set_xlim((1, len(self.sent)) if use_frames else (0, float(self.t[-1]) + 1))
        self.finish(fig, "10_action_scene_channel_frames.png" if use_frames else "09_action_scene_channel_time.png",
            "q = q_e4 / 10,000; larger q means more spatial dropping. All 12 modes remain available. Measured gNB SNR is not an actor input.\n" + (self.context_note() or "Only policy decisions are shown; holds/fallbacks retain their positions on the frame axis. Associations are descriptive."))

    def freshness_timeline(self):
        fig = self.new("Latency and map freshness over the full route", "One shared time axis  ·  Raw observations, without smoothing")
        axes = fig.subplots(3, 1, sharex=True, gridspec_kw=dict(left=.12, right=.945, bottom=.13, top=.80, hspace=.40))
        ax = axes[0]
        ax.plot(self.t, self.pred, color=BLUE, lw=.8, alpha=.9, label="Prediction ready at edge")
        ax.plot(self.t, self.feedback, color=ORANGE, lw=.8, alpha=.85, label="Feedback received at UE")
        ax.axhline(170, color=INK, lw=1.3, ls="--", label="170 ms feedback deadline")
        ax.set_ylabel("Since action\nopened (ms)", size=14)
        ax.set_ylim(70, max(430, np.nanmax(self.feedback)+10))
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.27), ncol=3, fontsize=12)
        ax = axes[1]
        ax.plot(self.install_t, self.age, color=TEAL, lw=.85, label="Age at installation")
        ax.plot(array(self.aoi_rows, "wall_s")-self.t0, self.aoi, color=PURPLE, lw=.7, alpha=.7, label="Map age of information (AoI)")
        ax.axhline(500, color=MUTED, lw=1.2, ls="--", label="500 ms installation-age limit")
        ax.set_ylabel("Since frame\ncapture (ms)", size=14)
        ax.set_ylim(80, max(570, np.nanmax(self.aoi)+10))
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.27), ncol=3, fontsize=12)
        ax = axes[2]
        ax.plot(self.install_t[1:], self.gaps, color=TEAL, lw=.85)
        ax.axhline(500, color=MUTED, ls="--", lw=1.2)
        ax.set(ylabel="Map update\ngap (ms)", ylim=(0, max(530, np.max(self.gaps)+20)),
               xlabel="Elapsed wall time from first policy-frame capture (s)")
        ax.text(.015, .86, f"{self.m['map']['gaps_over_500ms']} gaps > 500 ms · max {np.max(self.gaps):.1f} ms", transform=ax.transAxes,
                size=13, weight="bold", bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
        for ax in axes:
            ax.grid(color=GRID, lw=.7)
            ax.tick_params(labelsize=12)
            self.context_band(ax)
        if self.context_note():
            fig.text(.945, .854, self.context_note(), size=10.5, color=GOLD if self.highlight is not None else RED, ha="right")
        axes[-1].set_xlim(0, max(float(self.t[-1]), float(self.install_t[-1])) + 1)
        self.finish(fig, "11_latency_map_freshness_time.png", "Top: policy decisions only. Middle/bottom: all map updates; AoI sampled every 100 ms. AoI may exceed the installation-age limit between updates.\nLate feedback does not block map installation. 170 ms is the feedback deadline; 500 ms is a separate capture-age admission limit.")

    def export(self):
        rows = []
        for i, (d, l) in enumerate(zip(self.dec, self.layer)):
            rows.append(dict(d, transmitted_frame_index=int(self.frame_x[i]), q_drop_fraction=float(self.q[i]),
                camera_si=float(self.si[i]), radar_p40=float(self.p40[i]), measured_pusch_snr_db=float(self.snr[i]),
                measured_final_mcs=float(self.mcs[i])))
        with (self.out / "PRESENTATION_DECISIONS.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (self.out / "MAP_FRESHNESS_CDF.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["series", "age_ms", "cumulative_fraction", "denominator"])
            for name, values in (("installation_age", self.age), ("map_aoi", self.aoi)):
                x, y = ecdf(values)
                writer.writerows((name, float(a), float(b), len(values)) for a, b in zip(x, y))
        after = {str(p.resolve()): sha(p) for p in self.paths}
        assert after == self.hashes, "An input changed during rendering"
        self.summary.update(scope="Presentation only; no replay/inference/runtime change", figures=self.figs,
            context_markers_visible=not self.a.hide_context_markers,
            input_hashes=self.hashes, inputs_unchanged=True,
            frame_axis="1-based ordinal in the actual ordered SENT frame list; not twice decision index or raw CARLA frame ID",
            measured_channel="Existing latest-preceding-transmission join; gNB diagnostic, not an actor input",
            precision="Object detection, 3 m matching gate; NOT segmentation pixel precision",
            localization="All eligible GT denominator, including misses; late feedback included",
            freshness="Installation age per update; AoI uniform 100 ms time samples",
            pixels=[3840, 2160], format="PNG", dpi=DPI,
            reproduce="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -B " + " ".join(shlex.quote(s) for s in sys.argv))
        (self.out / "PLOT_DATA_AND_CHECKS.json").write_text(json.dumps(self.summary, indent=2, sort_keys=True)+"\n")
        outputs = {p.name: sha(p) for p in sorted(self.out.iterdir()) if p.is_file() and p.name != "OUTPUT_SHA256.json"}
        (self.out / "OUTPUT_SHA256.json").write_text(json.dumps(outputs, indent=2, sort_keys=True)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--layers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="M3.1  ·  GOOD_ALL  ·  V11")
    parser.add_argument("--highlight-window", type=float, nargs=2, metavar=("START_S", "END_S"))
    parser.add_argument("--gc-trace", type=Path)
    parser.add_argument("--radio-metadata", type=Path)
    parser.add_argument("--hide-context-markers", action="store_true",
                        help="Hide exploratory highlights and GC-stop annotations without dropping their analysis or observations")
    args = parser.parse_args()
    style()
    deck = Deck(args)
    deck.completion()
    deck.freshness_cdf()
    deck.quality_cdf()
    deck.quality_bars()
    deck.payload_bars()
    deck.route_outcomes()
    deck.route_layers()
    deck.timeline()
    deck.timeline(use_frames=True)
    deck.freshness_timeline()
    deck.export()
    print(json.dumps(dict(status="PASS", figures=deck.figs, out=str(args.out))), flush=True)


if __name__ == "__main__":
    main()
