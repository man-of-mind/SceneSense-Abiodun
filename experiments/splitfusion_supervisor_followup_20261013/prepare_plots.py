"""PNG-only comparison of three existing static-mode campaign sessions.

No inference, live services, GT rescoring or modifications of recorded evidence.
Run from the main workspace: python3 experiments/splitfusion_supervisor_followup_20261013/prepare_plots.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/splitfusion_three_arm_mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "three_arm_plots_01"
CAMPAIGN = ROOT / "experiments/splitfusion_static_modes_v13"
CAPTURES = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_static_modes_v13"
ARMS = ["agent", "mode03", "mode09"]
LABEL = {"agent": "Agent", "mode03": "AE128 / 8-bit", "mode09": "AE32 / 8-bit"}
COLOR = {"agent": "#087F8C", "mode03": "#C36828", "mode09": "#7356A6"}
INK, MUTED, GRID = "#162B3B", "#536574", "#DEE6EC"
COMMON = "FAVORABLE_STABLE • 1,000 transmitted frames per session • Static configurations: q = 0 (no dropping)"
SI_MEAN, SI_STD = 117.08553307797729, 13.336440703846854
INPUTS, VALIDATION, CAPTIONS = {}, {}, {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def input_path(path):
    path = Path(path).resolve()
    INPUTS[str(path)] = sha(path)
    return path


def read_json(path):
    return json.loads(input_path(path).read_text())


def read_csv(path):
    with input_path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def quantiles(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    assert np.all(np.isfinite(a))
    return {"n": len(a), **{f"p{q}": float(np.percentile(a, q)) if len(a) else None for q in [50, 95, 99]}}


def unique(rows, key):
    result = {int(r[key]): r for r in rows}
    assert len(result) == len(rows), ("duplicate", key)
    return result


def aoi_cdf(segments, x):
    lengths = np.array([s["duration_s"] for s in segments])
    ages = np.array([s["age_start_s"] for s in segments])
    return float(np.clip(x - ages, 0, lengths).sum() / lengths.sum())


def aoi_quantile(segments, p):
    lo, hi = 0.0, max(s["age_start_s"] + s["duration_s"] for s in segments)
    for _ in range(70):
        m = (lo + hi) / 2
        if aoi_cdf(segments, m) < p:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def extract():
    reference = read_json(CAMPAIGN / "campaign_report_05/CAMPAIGN_REPORT.json")["arms"]
    completion = read_csv(CAMPAIGN / "presentation_01/PREDICTION_COMPLETION_DECISIONS.csv")
    input_path(ROOT / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/live_state_v2.py")
    all_decisions, all_frames, all_radio, all_installs, all_segments, summary = [], [], [], [], [], {}
    data = {}
    for arm in ARMS:
        cell = CAPTURES / arm / "capture/cells/a71__favorable_stable"
        sealed = reference[arm]
        frames_raw = [r for r in read_csv(cell / "per_frame_metrics.csv") if r["prepare_status"] == "SENT"]
        frames_by_id = unique(frames_raw, "frame_id")
        assert len(frames_raw) == 1000
        stream = {r["stream_id"] for r in frames_raw}
        assert len(stream) == 1
        start = min(float(r["capture_wall_s"]) for r in frames_raw)
        end = max(float(r["capture_wall_s"]) for r in frames_raw)
        duration = end - start
        frames = [{"arm": arm, "frame_id": int(r["frame_id"]), "elapsed_s": float(r["capture_wall_s"]) - start,
                   "capture_wall_s": float(r["capture_wall_s"]), "payload_bytes": int(r["payload_bytes"]),
                   "q": int(r["q_e4"]) / 10000} for r in frames_raw]
        payload = quantiles(r["payload_bytes"] for r in frames)
        assert np.isclose(payload["p50"], sealed["payload"]["p50"])
        assert np.isclose(payload["p99"], sealed["payload"]["p99"])
        phase = read_json(cell / "run4_phase6/PHASE6_UE_EVIDENCE.json")
        policy = [r for r in phase["decisions"] if r["kind"] == "POLICY_DECISION"]
        unique(policy, "frame_id")
        pred = unique([r for r in completion if r["arm"] == arm], "frame_id")
        assert set(pred) == {int(r["frame_id"]) for r in policy}
        feedback = unique(phase["feedback_rows"], "frame_id")
        service = {}
        with input_path(cell / "numeric_gt/service/FRAMES.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("decision"):
                    fid = int(r["frame_id"])
                    assert fid not in service
                    service[fid] = r
        decisions = []
        for r in policy:
            fid = int(r["frame_id"])
            assert fid in frames_by_id
            fr = frames_by_id[fid]
            assert r["action_open"]["domain"] == "CLOCK_MONOTONIC_RAW"
            fb = feedback.get(fid)
            fb_ms = None
            if fb:
                assert fb["receipt"]["domain"] == "CLOCK_MONOTONIC_RAW"
                fb_ms = (fb["receipt"]["ns"] - r["action_open"]["ns"]) / 1e6
                assert fb_ms >= 0
            features = r.get("features")
            result = service.get(fid, {}).get("result") or {}
            measurement = result.get("measurement") or {}
            row = {"arm": arm, "frame_id": fid, "elapsed_s": float(fr["capture_wall_s"]) - start,
                   "mode": r["mode_id"], "q": r["q_e4"] / 10000,
                   "payload_bytes": int(fr["payload_bytes"]),
                   "SI": features[0] * SI_STD + SI_MEAN if features else None,
                   "P40": features[1] if features else None,
                   "prior_UE_MCS": round(features[2] * 28) if features else None,
                   "prediction_state": pred[fid]["state"],
                   "prediction_ms": float(pred[fid]["latency_ms"]) if pred[fid]["latency_ms"] else None,
                   "feedback_ms": fb_ms, "feedback_kind": fb["kind"] if fb else "NO_RECORDED_RECEIPT",
                   "quality_defined": bool(result.get("quality_defined")),
                   "Q_perc": result.get("q_perc") if result.get("quality_defined") else None,
                   "gt_outcome": service.get(fid, {}).get("outcome")}
            for cls, tol in [("vehicle", 1.0), ("person", 1.2)]:
                d = measurement.get(f"loc_{cls}_eligible_gt")
                errs = measurement.get(f"loc_{cls}_matched_xy_errors_m", [])
                row[f"{cls}_recall_completed"] = sum(e <= tol for e in errs) / d if d else None
            decisions.append(row)
        decisions.sort(key=lambda r: r["elapsed_s"])
        n = len(decisions)
        pc = Counter(r["prediction_state"] for r in decisions)
        assert n == sealed["completion"]["decisions"]
        assert pc["TIMELY"] == sealed["completion"]["completed_by_170"]
        assert pc["LATE"] == sealed["completion"]["completed_later"]
        assert pc["NONE"] == sealed["completion"]["no_completed_prediction"]
        fbc = sum(r["feedback_ms"] is not None and r["feedback_ms"] <= 170 for r in decisions)
        assert fbc == sealed["feedback_by_170"], (arm, fbc, sealed["feedback_by_170"])
        prediction_q = quantiles(r["prediction_ms"] for r in decisions)
        feedback_q = quantiles(r["feedback_ms"] for r in decisions)
        for q in ["p50", "p99"]:
            assert np.isclose(prediction_q[q], sealed["completion"]["latency_ms"][q])

        radio = []
        with input_path(CAPTURES / arm / "capture/m3_radio/ttracer/live_pusch_with_ingest.csv").open() as f:
            reader = csv.reader(f)
            header = next(reader)
            names = ["wall_ns", "raw_ns"] + header[2:]
            for values in reader:
                if len(values) != len(names) or not values[0].isdigit():
                    raise ValueError("Malformed radio row")
                r = dict(zip(names, values))
                t = int(r["wall_ns"]) / 1e9
                if start <= t <= end:
                    radio.append({"arm": arm, "elapsed_s": t - start, "PUSCH_SNR_dB": int(r["snrx10"]) / 10,
                                  "PUSCH_MCS": int(r["mcs"]), "rnti": r["rnti"]})
        assert radio
        commands = [{"elapsed_s": int(r["command_ack_wall_ns"]) / 1e9 - start, "target_snr_db": float(r["target_snr_db"])}
                    for r in read_csv(cell / "radio_trace.csv") if r.get("command_ack_wall_ns")]
        # Keep the latest pre-window command for a causal step curve, if available.
        before = [r for r in commands if r["elapsed_s"] <= 0]
        commands = (before[-1:] if before else []) + [r for r in commands if 0 < r["elapsed_s"] <= duration]

        installed_raw = [r for r in read_csv(cell / "direct_edge_map/direct_map_ingest.csv") if r["outcome"] == "RESULT_INSTALLED"]
        unique(installed_raw, "frame_id")
        installs = []
        for r in installed_raw:
            assert r["stream_id"] in stream
            fid = int(r["frame_id"])
            assert fid in frames_by_id
            cap, ts = float(r["capture_timestamp"]), float(r["map_install_at"])
            assert abs(cap - float(frames_by_id[fid]["capture_wall_s"])) < 1e-6
            age = (ts - cap) * 1000
            assert abs(age - float(r["map_age_at_install_ms"])) < .001
            installs.append({"arm": arm, "frame_id": fid, "install_elapsed_s": ts - start,
                             "capture_elapsed_s": cap - start, "install_age_ms": age})
        installs.sort(key=lambda r: r["install_elapsed_s"])
        assert all(b["capture_elapsed_s"] >= a["capture_elapsed_s"] for a, b in zip(installs, installs[1:]))
        segments = []
        for i, r in enumerate(installs):
            a = max(0., r["install_elapsed_s"])
            b = min(duration, installs[i + 1]["install_elapsed_s"] if i + 1 < len(installs) else duration)
            if b > a:
                segments.append({"arm": arm, "start_s": a, "end_s": b, "duration_s": b - a,
                                 "age_start_s": a - r["capture_elapsed_s"]})
        valid_installs = [r for r in installs if r["install_elapsed_s"] <= duration]
        gaps = np.diff([r["install_elapsed_s"] for r in valid_installs]).tolist()
        tail_gap = duration - valid_installs[-1]["install_elapsed_s"]
        assert segments and abs(sum(s["duration_s"] for s in segments) + installs[0]["install_elapsed_s"] - duration) < 1e-6
        s = {"label": LABEL[arm], "decisions": n, "transmitted_frames": len(frames), "duration_s": duration,
             "capture_origin_wall_s": start, "scene_observations": sum(r["SI"] is not None for r in decisions),
             "prediction_counts": dict(pc), "prediction_ms": prediction_q, "feedback_ms": feedback_q,
             "feedback_timely": fbc, "feedback_kinds": dict(Counter(r["feedback_kind"] for r in decisions)),
             "feedback_missing": n - feedback_q["n"], "payload_bytes": payload,
             "quality": sealed["quality"], "gt_coverage": sealed["gt_coverage"],
             "radio_rows": len(radio), "radio_snr_db": quantiles(r["PUSCH_SNR_dB"] for r in radio),
             "radio_mcs": quantiles(r["PUSCH_MCS"] for r in radio),
             "map_installs": len(installs), "map_install_age_ms": quantiles(r["install_age_ms"] for r in installs),
             "map_aoi_s": {"p50": aoi_quantile(segments, .50), "p99": aoi_quantile(segments, .99)},
             "map_aoi_fraction_above_500ms": 1 - aoi_cdf(segments, .5),
             "initial_no_map_s": installs[0]["install_elapsed_s"],
             "inter_install_gaps_over_500ms": sum(g > .5 for g in gaps),
             "trailing_no_install_s": tail_gap, "max_inter_install_gap_s": max(gaps) if gaps else None}
        summary[arm] = s
        data[arm] = {"summary": s, "frames": frames, "decisions": decisions, "radio": radio,
                     "commands": commands, "installs": installs, "segments": segments}
        all_decisions.extend(decisions); all_frames.extend(frames); all_radio.extend(radio)
        all_installs.extend(installs); all_segments.extend(segments)
    write_csv("DECISIONS.csv", all_decisions)
    write_csv("FRAMES.csv", all_frames)
    write_csv("PUSCH.csv", all_radio)
    write_csv("MAP_INSTALLS.csv", all_installs)
    write_csv("AOI_SEGMENTS.csv", all_segments)
    write_csv("CHANNEL_COMMANDS.csv", [{"arm": a, **r} for a in ARMS for r in data[a]["commands"]])
    write_json("METRICS.json", summary)
    return data


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": INK,
                         "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.labelsize": 16,
                         "axes.titlesize": 19, "axes.titleweight": "bold", "axes.labelweight": "medium",
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                         "legend.frameon": False, "legend.fontsize": 15, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "lines.linewidth": 2.2})


def canvas(title, subtitle=COMMON, notes=()):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=200)
    fig.text(.045, .961, title, fontsize=29, weight="bold", va="top")
    fig.text(.045, .908, subtitle, fontsize=15, color=MUTED, va="top")
    for i, line in enumerate(notes):
        fig.text(.045, .082 - i * .023, line, fontsize=11.5, color=MUTED, va="top")
    return fig


def legend(fig):
    fig.legend(handles=[Line2D([], [], color=COLOR[a], lw=4, label=LABEL[a]) for a in ARMS],
               loc="center", bbox_to_anchor=(.5, .850), ncol=3, columnspacing=3)


def axis(ax, ylabel=None):
    ax.grid(axis="y", color=GRID, lw=.8)
    if ylabel:
        ax.set_ylabel(ylabel)


def time_axis(ax, ylabel=None):
    axis(ax, ylabel)
    ax.set_xlim(0, 131)
    ax.set_xticks([0, 30, 60, 90, 120])


def save(fig, name, caption):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    assert (w, h) == (3840, 2160)
    renderer = fig.canvas.get_renderer()
    outside = []
    # Axis tick objects can exist outside their view interval; only inspect drawn tick labels.
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        for ticks, lim in [(ax.xaxis.get_major_ticks(), ax.get_xlim()), (ax.yaxis.get_major_ticks(), ax.get_ylim())]:
            texts += [t.label1 for t in ticks if min(lim) <= t.get_loc() <= max(lim)]
    for text in texts:
        if text.get_visible() and text.get_text():
            b = text.get_window_extent(renderer)
            if b.x0 < -2 or b.x1 > w + 2 or b.y0 < -2 or b.y1 > h + 2:
                outside.append(text.get_text())
    if outside:
        raise ValueError((name, "text outside canvas", outside))
    fig.savefig(OUT / name, dpi=200, metadata={"Software": "SplitFusion three-arm retained-evidence renderer"})
    VALIDATION[name] = {"pixels": [w, h], "text_inside_canvas": True}
    CAPTIONS[name] = caption
    plt.close(fig)


def cdf(ax, values, n, color, maximum=550):
    a = np.sort([v for v in values if v is not None])
    assert len(a) and max(a) < maximum
    ax.step(np.r_[0, a, maximum], np.r_[0, np.arange(1, len(a) + 1) / n, len(a) / n], where="post", color=color, lw=3)


def completion_plots(data):
    fig = canvas("When does each result become available?", notes=[
        "All policy decisions stay in the denominator: Agent 493; each static configuration 500. Curves do not renormalize away missing results.",
        "Prediction readiness and UE feedback receipt are separate milestones. Sessions are not paired; no packet-loss cause is inferred."])
    legend(fig)
    for i, (key, title) in enumerate([("prediction_ms", "Prediction ready at the edge"), ("feedback_ms", "Policy feedback received at the UE")]):
        ax = fig.add_axes([.08 + i * .47, .21, .39, .54])
        for a in ARMS:
            rows, s = data[a]["decisions"], data[a]["summary"]
            cdf(ax, [r[key] for r in rows], s["decisions"], COLOR[a])
            v = sum(r[key] is not None and r[key] <= 170 for r in rows) / s["decisions"]
            ax.plot(170, v, "o", color=COLOR[a], ms=7)
        ax.axvline(170, color=MUTED, ls=":", lw=1.8)
        ax.text(177, .46, "170 ms", color=MUTED, fontsize=13)
        ax.set(xlim=(0, 550), ylim=(-.025, 1.035), title=title, xlabel="Time from action start (ms)")
        ax.set_yticks(np.arange(0, 1.01, .2)); ax.yaxis.set_major_formatter(PercentFormatter(1))
        axis(ax, "Share of all decisions")
    save(fig, "01_completion_curves.png", "Cumulative incidence, not a completed-only CDF. AE128 finishes only 4/500 predictions; Agent 493/493; AE32 476/500. The vertical line is 170 ms from action start.")

    fig = canvas("Completion outcomes and observed latency tails", notes=[
        "P50/P99 use recorded completions or receipts only; missing decisions have no imputed latency. AE128 quantiles use just 4 observations.",
        "Feedback includes explicit evaluator-fault replies as well as successful replies; its timely count is not a scored-quality count."])
    legend(fig)
    for col, (key, title) in enumerate([("prediction_ms", "Prediction readiness"), ("feedback_ms", "UE feedback receipt")]):
        ax = fig.add_axes([.13 + col * .46, .54, .32, .22])
        for j, a in enumerate(ARMS):
            s = data[a]["summary"]; vals = [r[key] for r in data[a]["decisions"]]
            counts = [sum(v is not None and v <= 170 for v in vals), sum(v is not None and v > 170 for v in vals), sum(v is None for v in vals)]
            left = 0
            for k, (c, color) in enumerate(zip(counts, [COLOR[a], "#E7B863", "#E0E5EB"])):
                width = c / len(vals)
                ax.barh(j, width, left=left, height=.55, color=color)
                if width > .10:
                    ax.text(left + width / 2, j, f"{c}/{len(vals)}", ha="center", va="center", fontsize=12,
                            color="white" if k == 0 else INK, weight="bold")
                left += width
        ax.set(yticks=range(3), yticklabels=[LABEL[a] for a in ARMS], xlim=(0, 1), title=title)
        ax.invert_yaxis(); ax.set_xticks([0, .5, 1]); ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis="y", length=0)
        ax.text(.5, -.34, "Arm colour: ≤170 ms     Gold: later     Grey: missing", transform=ax.transAxes, ha="center", fontsize=11.5)
        bottom = fig.add_axes([.13 + col * .46, .19, .32, .22])
        for j, a in enumerate(ARMS):
            q = data[a]["summary"][key]
            for off, k, alpha in [(-.16, "p50", 1), (.16, "p99", .5)]:
                bottom.bar(j + off, q[k], width=.28, color=COLOR[a], alpha=alpha)
                bottom.text(j + off, q[k] + 10, f"{q[k]:.0f}", ha="center", fontsize=13, weight="bold")
        bottom.set_xticks(range(3), [f"{LABEL[a]}\nn={data[a]['summary'][key]['n']}" for a in ARMS])
        bottom.set_ylim(0, 580); bottom.set_yticks([0, 170, 300, 500]); axis(bottom, "Latency (ms)")
        bottom.axhline(170, color=MUTED, ls=":", lw=1)
        bottom.text(.5, 1.08, "P50: solid bar     P99: pale bar", transform=bottom.transAxes, ha="center", fontsize=12)
    save(fig, "02_outcomes_and_latency_quantiles.png", "Top: on-time, late and missing outcomes over every decision. Bottom: conditional P50/P99; four AE128 observations cannot characterize typical session latency or a reliable tail.")


def payload_plot(data):
    fig = canvas("Application payload: P50 and P99", notes=[
        "SFD4 application-frame bytes; decimal kB (1 kB = 1,000 B). All 1,000 sent frames per session, including holds/fallbacks.",
        "One shared linear scale. Separate live sessions; recorded payloads and statistics are unchanged. Timelines are now in plot 13."])
    legend(fig)
    ax = fig.add_axes([.105, .235, .80, .53])
    for j, a in enumerate(ARMS):
        q = data[a]["summary"]["payload_bytes"]
        for off, key, alpha in [(-.18, "p50", 1), (.18, "p99", .5)]:
            val = q[key] / 1000
            ax.bar(j + off, val, width=.31, color=COLOR[a], alpha=alpha)
            ax.text(j + off, val + 55, f"{val:,.1f}", ha="center", fontsize=19, weight="bold")
    ax.set_xticks(range(3), [LABEL[a] for a in ARMS]); ax.set_ylim(0, 2800)
    axis(ax, "Application payload (kB)")
    ax.tick_params(axis="x", labelsize=19)
    ax.text(.5, -.17, "P50: solid bar     P99: pale bar", transform=ax.transAxes, ha="center", fontsize=16)
    save(fig, "03_payload_statistics_and_time.png", "Plot 03 now contains only P50/P99 payload bars on one shared linear scale. Median application payload: Agent 117.3 kB, AE128/8-bit 2,339.1 kB, AE32/8-bit 600.2 kB. The original filename is kept to preserve links; timelines have moved to 13_payload_time_and_relative_variation.png. Data and statistics are unchanged.")


def payload_time_plot(data):
    fig = canvas("Application payload over time", notes=[
        "SFD4 application-frame bytes; decimal kB (1 kB = 1,000 B). All 1,000 sent frames per session, including holds/fallbacks.",
        "One shared linear kB scale; no smoothing. Time origins and scenes differ by session."])
    legend(fig)
    absolute = fig.add_axes([.10, .20, .84, .56])
    for a in ARMS:
        rows = data[a]["frames"]
        times = np.array([r["elapsed_s"] for r in rows])
        values = np.array([r["payload_bytes"] for r in rows], dtype=float)
        assert 0 <= min(values / 1000) <= max(values / 1000) <= 2650
        absolute.plot(times, values / 1000, color=COLOR[a], lw=2.1)
    time_axis(absolute, "Application payload (kB)")
    absolute.set_ylim(0, 2650); absolute.set_yticks([0, 500, 1000, 1500, 2000, 2500])
    absolute.set_xlabel("Elapsed time in each session (s)")
    save(fig, "13_payload_time_and_relative_variation.png", "Absolute payload time series only, on one shared linear kB scale with no smoothing. All 1,000 sent frames per configuration, including holds and fallbacks, remain included. Equal elapsed time is not a paired scene. The relative-variation panel has been removed; the original filename is retained to preserve links.")


def radio_plot(data):
    fig = canvas("Channel context: measured PUSCH SNR and MCS", notes=[
        "Solid SNR/MCS: 1-second medians of per-transmission values; shaded SNR band: observed minimum–maximum in each second.",
        "Aligned by telemetry-ingest wall time, not a packet-matched radio clock. The MAC-log power-control statistic is not used. Separate sessions."])
    for j, a in enumerate(ARMS):
        rows = data[a]["radio"]
        bins = {}
        for r in rows:
            bins.setdefault(int(r["elapsed_s"]), []).append(r)
        t, snr, lower, upper, mcs = [], [], [], [], []
        for k, rr in sorted(bins.items()):
            t.append(k + .5)
            vals = [r["PUSCH_SNR_dB"] for r in rr]
            snr.append(float(np.median(vals))); lower.append(float(min(vals))); upper.append(float(max(vals)))
            mcs.append(float(np.median([r["PUSCH_MCS"] for r in rr])))
        ax = fig.add_axes([.075 + j * .31, .52, .255, .26])
        ax.fill_between(t, lower, upper, color=COLOR[a], alpha=.18)
        ax.plot(t, snr, color=COLOR[a])
        time_axis(ax, "SNR (dB)" if j == 0 else None)
        ax.set_title(LABEL[a], color=COLOR[a], pad=20); ax.set_ylim(3, 55); ax.set_yticks([10, 20, 30, 40, 50])
        stats = data[a]['summary']['radio_snr_db']
        ax.text(.97, .90, f"P50/P99: {stats['p50']:.1f}/{stats['p99']:.1f} dB", ha="right", transform=ax.transAxes, fontsize=12)
        ax = fig.add_axes([.075 + j * .31, .19, .255, .24])
        ax.step(t, mcs, color=COLOR[a], where="mid")
        time_axis(ax, "PUSCH MCS index" if j == 0 else None)
        ax.set_ylim(7, 31); ax.set_yticks([9, 15, 21, 27]); ax.set_xlabel("Elapsed time (s)")
        stats = data[a]['summary']['radio_mcs']
        ax.text(.97, .90, f"P50/P99: {stats['p50']:.0f}/{stats['p99']:.0f}", ha="right", transform=ax.transAxes, fontsize=12)
    save(fig, "04_snr_mcs_time.png", "Per-transmission PUSCH measurements, not MAC-log SNR. One-second medians and the full within-bin SNR range make dense radio records legible; start-up/high-SNR values are retained when inside the transmitted-frame window. P50/P99 use all raw events. The same FAVORABLE_STABLE trace is used, but wall-clock and scene timing differ across sessions. Commanded targets remain in CHANNEL_COMMANDS.csv.")


def latency_time_plot(data):
    fig = canvas("Latency over time: readiness is not feedback receipt", notes=[
        "Each dot is one policy decision with a recorded milestone. Grey bottom ticks mark missing milestones, not zero latency.",
        "Dotted line: original 170-ms feedback boundary (also applied to prediction readiness for comparison). Same scales; no smoothing."])
    for j, a in enumerate(ARMS):
        rows = data[a]["decisions"]
        for i, (key, label) in enumerate([("prediction_ms", "Prediction ready (ms)"), ("feedback_ms", "UE feedback receipt (ms)")]):
            ax = fig.add_axes([.075 + j * .31, .52 - i * .33, .255, .26])
            present = [r for r in rows if r[key] is not None]; missing = [r for r in rows if r[key] is None]
            ax.scatter([r["elapsed_s"] for r in present], [r[key] for r in present], s=12, color=COLOR[a], alpha=.85)
            if missing:
                ax.plot([r["elapsed_s"] for r in missing], np.full(len(missing), 15), "|", color="#A8B2BA", markersize=4)
            ax.axhline(170, color=MUTED, ls=":", lw=1.4)
            time_axis(ax, label if j == 0 else None); ax.set_ylim(0, 550); ax.set_yticks([0, 170, 300, 500])
            ax.text(.97, .89, f"Observed {len(present)}/{len(rows)}", ha="right", transform=ax.transAxes, fontsize=12)
            if i == 0:
                ax.set_title(LABEL[a], color=COLOR[a], pad=20)
            else:
                ax.set_xlabel("Elapsed time (s)")
    save(fig, "05_prediction_feedback_time.png", "Raw per-decision prediction and feedback latencies. The AE128 overload appears as missing milestones across most of the session, not artificially low latency. No causal downlink or packet-loss attribution is made.")


def quality_plot(data):
    fig = canvas("Perception available within 170 ms", "Prediction-ready boundary • Recall and IoU include the cost of late/missing predictions • Static q = 0", notes=[
        "≤ marks an upper bound where failed decisions lack GT. Recall counts eligible objects; precision conditions on timely predictions only.",
        "AE128: no timely prediction; vehicle IoU is zero by construction. Pedestrian IoU is N/A (GT union unmeasured). Separate scenes; no causal ranking."])
    legend(fig)
    for row, (cls, cls_label, tol) in enumerate([("vehicle", "Vehicle", "1.0 m"), ("person", "Pedestrian", "1.2 m")]):
        for col, kind in enumerate(["recall", "iou", "precision"]):
            ax = fig.add_axes([.075 + col * .31, .53 - row * .34, .255, .235])
            for j, a in enumerate(ARMS):
                q = data[a]["summary"]["quality"][cls]
                if kind == "recall":
                    val = q["recall_deadline"]["value"]; bound = q["recall_deadline"]["bound"].startswith("UPPER")
                elif kind == "iou":
                    val = q["seg_iou_deadline"]; bound = a == "mode09"
                else:
                    val = q["precision_timely"]; bound = False
                if val is None:
                    ax.text(j, .10, "N/A", ha="center", fontsize=16, color=MUTED, weight="bold")
                else:
                    ax.bar(j, val, .54, color=COLOR[a], hatch="//" if bound else None, edgecolor="white")
                    ax.text(j, val + .045, ("≤" if bound else "") + f"{val * 100:.1f}%", ha="center", fontsize=15, weight="bold")
            ax.set_xticks(range(3), ["Agent", "AE128/8", "AE32/8"]); ax.set_ylim(0, 1.16)
            ax.set_yticks([0, .5, 1]); ax.yaxis.set_major_formatter(PercentFormatter(1)); axis(ax)
            if col == 0:
                ax.set_ylabel(cls_label, weight="bold", fontsize=19)
            if row == 0:
                titles = {"recall": "Recall by 170 ms", "iou": "Segmentation IoU by 170 ms", "precision": "Timely-prediction precision"}
                ax.set_title(titles[kind], pad=21, fontsize=17)
            if kind == "recall":
                ax.text(.97, .94, f"Tolerance: {tol}", transform=ax.transAxes, ha="right", fontsize=11.5, color=MUTED)
    save(fig, "06_deadline_quality.png", "All values come unchanged from the sealed campaign report. Recall uses 1.0 m vehicle / 1.2 m pedestrian tolerance; precision uses the existing 3 m matching gate and only timely predictions. It can be high even when very few predictions are timely. AE32 recall/IoU are upper bounds; AE128 precision is undefined, not zero. The boundary here is prediction readiness, not UE-feedback receipt.")


def scene_plot(data):
    rr = data["agent"]["decisions"]
    fig = canvas("Agent: scene, compression and channel over time",
                 f"FAVORABLE_STABLE • {len(rr):,} policy decisions • Five aligned time series", notes=[
        "SI: camera-image spatial information (Sobel variability). P40: mean clipped radar proximity within 40 m. Higher q means more spatial dropping.",
        "Payload: SFD4 application bytes (1 kB = 1,000 B), matched to each decision. Holds/fallbacks excluded from the first four panels; no smoothing.",
        "PUSCH SNR: measured 1-s median, shaded min–max; telemetry-ingest time, not frame-matched. This is not the actor's commanded-SNR input."])
    panels = [("SI", "Camera SI", (65, 160), [80, 110, 140]),
              ("P40", "Radar P40", (0, 1), [0, .5, 1]),
              ("q", "Dropping q", (-.06, 1.04), [0, .5, 1]),
              ("payload_bytes", "Payload (kB)", (0, 300), [0, 150, 300]),
              ("PUSCH_SNR_dB", "PUSCH SNR\n(dB)", (0, 55), [10, 30, 50])]
    shared = None
    for i, (key, label, ylim, ticks) in enumerate(panels):
        ax = fig.add_axes([.105, .735 - i * .145, .83, .12], sharex=shared)
        if shared is None:
            shared = ax
        if key == "PUSCH_SNR_dB":
            radio = data["agent"]["radio"]
            bins = {}
            for r in radio:
                bins.setdefault(int(r["elapsed_s"]), []).append(r[key])
            assert radio and min(r[key] for r in radio) >= ylim[0] and max(r[key] for r in radio) <= ylim[1]
            seconds = range(min(bins), max(bins) + 1)
            times = [s + .5 for s in seconds]
            medians = [np.median(bins[s]) if s in bins else np.nan for s in seconds]
            lower = [min(bins[s]) if s in bins else np.nan for s in seconds]
            upper = [max(bins[s]) if s in bins else np.nan for s in seconds]
            ax.fill_between(times, lower, upper, color=COLOR["agent"], alpha=.18, step="mid")
            ax.step(times, medians, color=COLOR["agent"], lw=1.8, where="mid")
        else:
            vals = np.array([r[key] if r[key] is not None else np.nan for r in rr], dtype=float)
            if key == "payload_bytes":
                vals = vals / 1000
            present = vals[np.isfinite(vals)]
            assert not len(present) or (min(present) >= ylim[0] and max(present) <= ylim[1]), key
            ax.plot([r["elapsed_s"] for r in rr], vals, color=COLOR["agent"], lw=1.8, marker=".", ms=2)
        time_axis(ax, label); ax.set_ylim(*ylim); ax.set_yticks(ticks)
        if i < len(panels) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Elapsed time from first transmitted capture (s)")
    save(fig, "07_scene_complexity_and_q_time.png", "Agent only: camera SI, radar P40, dropping control q, application payload and measured PUSCH SNR in five aligned panels. SI/P40 are reconstructed from the recorded 22-D state using the pinned scaling constants. Each payload is the SFD4 application-frame size for the same policy decision, in decimal kB. Holds and fallbacks are excluded from the first four panels, whose values are unsmoothed. The fifth panel uses all retained per-transmission PUSCH SNR events in the session, shown as one-second medians with a minimum–maximum band, including startup extremes. Empty bins stay absent. Radio is aligned by telemetry-ingest wall time, not by frame; measured PUSCH SNR is not the commanded SNR given to the actor or the MAC-log power-control statistic. Alignment shows association, not causal feature attribution. Static configurations remain excluded.")


def map_plot(data):
    fig = canvas("Map freshness: successful updates and time without updates", notes=[
        "AoI = age of the latest installed frame, time-weighted between first and last transmitted capture. Initial no-map time is reported separately.",
        "Age keeps increasing without an update; it does not mean stale objects stay visible. AE128 installs only 4 frames, all near session start."])
    legend(fig)
    ax = fig.add_axes([.08, .52, .36, .24])
    for a in ARMS:
        v = np.sort([r["install_age_ms"] for r in data[a]["installs"]])
        ax.step(np.r_[0, v, 550], np.r_[0, np.arange(1, len(v)+1)/len(v), 1], where="post", color=COLOR[a], lw=2.5)
    ax.set(xlim=(0, 550), ylim=(0, 1.03), title="Age at successful map installation", xlabel="Capture → installation (ms)")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); axis(ax, "Share of installed frames")
    ax = fig.add_axes([.56, .52, .37, .24])
    x = np.geomspace(.1, 150, 800)
    for a in ARMS:
        ax.plot(x, [aoi_cdf(data[a]["segments"], v) for v in x], color=COLOR[a], lw=2.5)
    ax.set_xscale("log"); ax.set_xlim(.1, 150); ax.set_xticks([.1, .5, 1, 10, 100]); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
    ax.set(ylim=(0, 1.03), title="Latest-installed-frame age over time", xlabel="Information age (s, log scale)")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); axis(ax, "Share of observed time")
    ax = fig.add_axes([.08, .18, .85, .22])
    for a in ARMS:
        xx, yy = [], []
        for s in data[a]["segments"]:
            xx.extend([s["start_s"], s["end_s"]]); yy.extend([s["age_start_s"], s["age_start_s"] + s["duration_s"]])
        ax.plot(xx, yy, color=COLOR[a], lw=1.25)
    ax.axhline(.5, color=MUTED, ls=":", lw=1)
    ax.text(131, .5, "  500 ms", va="center", fontsize=11, color=MUTED)
    time_axis(ax, "Latest-frame age (s)"); ax.set_yscale("log"); ax.set_ylim(.1, 160)
    ax.set_yticks([.1, .5, 1, 10, 100]); ax.yaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
    ax.set_xlabel("Elapsed time in each session (s)")
    save(fig, "08_map_freshness_cdf_and_time.png", "The successful-install CDF alone hides AE128's overload: only four updates occur. The time-weighted AoI CDF and timeline expose the ensuing approximately 123-second no-update period. Before the first install AoI is undefined; no-map duration is listed in METRICS.json. Frame-level age is not object-track age or policy-feedback latency.")


def quality_time_plot(data):
    fig = canvas("Perception measurements over time", "Recorded scored predictions only • Missing or zero-denominator measurements remain absent • Separate scenes", notes=[
        "These are conditional per-frame measurements, not deadline-aware session scores. Use plot 06 for quality available within 170 ms.",
        "Dots are unsmoothed; a coloured bottom rug shows prediction-ready decisions ≤170 ms. AE128 has no timely result and only 4 scored frames."])
    for j, a in enumerate(ARMS):
        rr = data[a]["decisions"]
        for i, (key, label) in enumerate([("vehicle_recall_completed", "Vehicle recall"),
                                         ("person_recall_completed", "Pedestrian recall"),
                                         ("Q_perc", "Quality Q_perc")]):
            ax = fig.add_axes([.075 + j * .31, .62 - i * .205, .255, .155])
            obs = [r for r in rr if r[key] is not None]
            ax.scatter([r["elapsed_s"] for r in obs], [r[key] for r in obs], color=COLOR[a], s=9, alpha=.7)
            timely = [r for r in rr if r["prediction_state"] == "TIMELY"]
            ax.plot([r["elapsed_s"] for r in timely], [-.06] * len(timely), "|", color=COLOR[a], markersize=3, alpha=.7)
            time_axis(ax, label if j == 0 else None); ax.set_ylim(-.1, 1.1); ax.set_yticks([0, .5, 1])
            if i == 0:
                ax.set_title(LABEL[a], color=COLOR[a], pad=24)
            if i == 2:
                ax.set_xlabel("Elapsed time (s)")
            ax.text(.97, .83, f"n={len(obs)}", transform=ax.transAxes, ha="right", fontsize=11.5)
    save(fig, "09_perception_time.png", "Optional detail figure: per-frame localisation recall and Q_perc on scored, defined samples. Empty intervals are not zero quality. Do not average these displayed recalls to recover pooled object recall; denominators differ by frame. The campaign-level deadline-aware comparison is plot 06.")


def map_quantile_plot(data):
    fig = canvas("Map freshness P50 / P99: update delay versus information age", notes=[
        "Left conditions on successful installations, including drain results. Right is time-weighted over the session, after the first installation.",
        "AE128 has only 4 installations and then ~123 s without an update. Latest-frame AoI is not displayed-object age; stale objects may expire."])
    legend(fig)
    for col, title in enumerate(["Capture → successful installation", "Age of latest installed frame"]):
        ax = fig.add_axes([.08 + col * .47, .23, .36, .52])
        for j, a in enumerate(ARMS):
            s = data[a]["summary"]
            q = s["map_install_age_ms"] if col == 0 else s["map_aoi_s"]
            for off, key, alpha in [(-.16, "p50", 1), (.16, "p99", .5)]:
                val = q[key]
                ax.bar(j + off, val, width=.28, color=COLOR[a], alpha=alpha)
                text = f"{val:.1f}" if col == 0 else f"{val:.3f}" if val < 10 else f"{val:.1f}"
                ax.text(j + off, val + 12 if col == 0 else val * 1.16, text, ha="center", fontsize=13, weight="bold")
        ax.set_xticks(range(3), [f"{LABEL[a]}\nn={data[a]['summary']['map_installs']}" if col == 0 else LABEL[a] for a in ARMS])
        ax.set_title(title, pad=22)
        if col == 0:
            ax.set_ylim(0, 570); axis(ax, "Install age (ms)")
        else:
            ax.set_yscale("log"); ax.set_ylim(.1, 220); ax.set_yticks([.1, 1, 10, 100])
            ax.yaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
            axis(ax, "Information age (s, log scale)")
        ax.text(.5, -.21, "P50: solid bar     P99: pale bar", transform=ax.transAxes, ha="center", fontsize=13)
    save(fig, "10_map_freshness_quantiles.png", "Conditional successful-install latency versus time-weighted latest-installed-frame age. AE128 successful-install P50 is 470 ms, but its session AoI P50 is 61.7 seconds because updates stop. Log scale on the AoI panel, linear milliseconds on install latency. Numbers and initial no-map durations are in METRICS.json.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    input_path(__file__)
    data = extract()
    style()
    completion_plots(data)
    payload_plot(data)
    payload_time_plot(data)
    radio_plot(data)
    latency_time_plot(data)
    quality_plot(data)
    scene_plot(data)
    map_plot(data)
    quality_time_plot(data)
    map_quantile_plot(data)
    assert all(sha(p) == digest for p, digest in INPUTS.items()), "Input changed during analysis"
    write_json("INPUT_SHA256.json", INPUTS)
    write_json("VALIDATION.json", {"inputs_unchanged": True, "counts_match_sealed_campaign": True,
               "prediction_quantiles_match_sealed_campaign": True, "payload_quantiles_match_sealed_campaign": True,
               "feedback_counts_match_sealed_campaign": True, "source_frame_joins_checked": True, "figures": VALIDATION})
    write_json("CAPTIONS.json", CAPTIONS)
    (OUT / "READ_ME.txt").write_text(
        "THREE-ARM PRESENTATION PLOTS — PNG ONLY\n\n"
        "Compare Agent, mode03 AE128/UINT8/q0 and mode09 AE32/UINT8/q0 from the same 13-arm OAI campaign.\n"
        "These are separate 1,000-frame sessions, not full-route or matched-scene trials. No new experiment was run.\n"
        "Suggested core plots: 01 completion curves; 03 payload; 06 deadline-aware perception; 08 map freshness.\n"
        "Context/detail: 02 quantiles and missing outcomes; 04 channel; 05 latency traces; 07 scene and q; 09 observed quality traces; 10 map quantiles; 13 payload timelines.\n"
        "Plot 03 is bars only; plot 13 shows absolute payload time series on one shared linear kB scale. Original filenames retained.\n"
        "Plot 07 is agent-only: five aligned panels for SI, P40, q, decision-matched application payload and measured PUSCH SNR.\n"
        "All PNGs are 3840 x 2160. No PDF or slide deck is produced. Full captions are in CAPTIONS.json.\n\n"
        "Definitions:\n"
        "- Completion/feedback time: CLOCK_MONOTONIC_RAW from action-open, independently for each milestone.\n"
        "- Quality deadline: prediction ready <=170 ms, following the sealed static campaign, not UE receipt.\n"
        "- Timely precision conditions on timely predictions; unlike recall it cannot express missing-detection coverage.\n"
        "- Scene gaps and undefined quality are never filled with zero or interpolated.\n"
        "- PUSCH radio time: host telemetry ingest; no frame-to-radio causal mapping is asserted.\n"
        "- Map AoI CDF: exact time weighting of linear age intervals after the first successful install,\n"
        "  stopping at the last sent capture. Initial no-map seconds are separate. Successful install ages include drain results.\n"
        "- P50/P99: NumPy linear percentiles; latency quantiles condition on recorded events, payload uses all sent frames.\n"
        "- Bars with <= / hatching are upper bounds caused by missing GT, not confidence intervals.\n\n"
        "Reproduce from workspace root:\n"
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python3 experiments/splitfusion_supervisor_followup_20261013/prepare_plots.py\n"
        "Only this output directory is written. Historical figures, runtime, UI and M4 evidence are untouched.\n")
    write_json("OUTPUT_SHA256.json", {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "OUTPUT_SHA256.json"})
    print(json.dumps({"output": str(OUT), "png_count": len(VALIDATION), "metrics": {a: data[a]["summary"] for a in ARMS}}, indent=2))


if __name__ == "__main__":
    main()
