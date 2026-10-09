"""PNG-only, read-only Run-5 reporting; refuse unqualified GT/capture evidence."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import verify_live_gt_capture, verify_snr_capture


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0}
    if not np.isfinite(values).all():
        raise ValueError("non-finite measurement")
    return dict(n=len(values), mean=float(values.mean()),
                **{key: float(np.quantile(values, q, method="linear"))
                   for key, q in (("p50", .5), ("p95", .95), ("p99", .99))},
                max=float(values.max()))


def components(measurement, name):
    get = lambda suffix: int(measurement[f"seg_{name}_{suffix}_pixels"])
    gt, pred, inter, union = (get(s) for s in ("gt", "pred", "intersection", "union"))
    if min(gt, pred, inter, union) < 0 or union != gt + pred - inter or inter > min(gt, pred):
        raise ValueError("invalid segmentation counts")
    count = lambda suffix: int(measurement[f"loc_{name}_{suffix}"])
    eligible, tp, fp, fn = (count(s) for s in ("eligible_gt", "tp", "fp", "fn"))
    errors = list(measurement[f"loc_{name}_matched_xy_errors_m"])
    if eligible != tp + fn or tp != len(errors) or min(eligible, tp, fp, fn) < 0:
        raise ValueError("invalid localization counts")
    if any(not math.isfinite(v) or v < 0 for v in errors):
        raise ValueError("invalid matched-object error")
    return dict(iou=inter / union if union else None, intersection=inter, union=union,
                eligible_gt=eligible, tp=tp, fp=fp, fn=fn, errors_m=errors,
                xy_median_m=float(np.median(errors)) if errors else None,
                recall=tp / eligible if eligible else None,
                precision=tp / (tp + fp) if tp + fp else None)


def unique_index(values, key, name):
    indexed = {}
    for value in values:
        identity = key(value)
        if identity in indexed:
            raise ValueError(f"duplicate {name} identity: {identity}")
        indexed[identity] = value
    return indexed


def join_records(*, ue, edge, snr, frames, tickets):
    """Join exact transmitted identities, not merely similarly numbered rows."""
    frame_key = lambda row: int(row["frame_id"])
    frame_map = unique_index(frames, frame_key, "frame")
    decision_map = unique_index([r for r in ue["decisions"] if r["kind"] == "POLICY_DECISION"], frame_key, "UE decision")
    ticket_map = unique_index(tickets, frame_key, "ticket")
    transmitted = unique_index(ue["transmitted_identities"], frame_key, "transmission")
    observations = unique_index([r for r in snr["decision_snr"] if r["admitted"]], frame_key, "admitted SNR")
    feedback = unique_index(ue["feedback_rows"], frame_key, "feedback")
    evaluations = unique_index(edge["evaluations"], frame_key, "evaluation")
    resolutions = unique_index(ue["resolutions"], lambda row: (row["identity"]["session_uuid"], int(row["identity"]["decision_seq"])), "resolution")
    policy_frames = {fid for fid, row in frame_map.items() if row["frame_kind"] == "POLICY_DECISION"}
    if not policy_frames or set(ticket_map) != policy_frames or set(decision_map) != policy_frames or set(observations) != policy_frames:
        raise ValueError("Decision/ticket/frame/SNR population mismatch")
    if set(transmitted) != set(frame_map) or not set(feedback).issubset(policy_frames) or not set(evaluations).issubset(policy_frames):
        raise ValueError("Foreign or missing transmission/feedback/evaluation identity")
    sequence = sorted(int(frame["tensor_seq"]) for frame in frames)
    if sequence != list(range(len(frames))):
        raise ValueError("Transmitted-frame sequence is not contiguous")
    used_resolutions, joined = set(), []
    for fid, ticket in ticket_map.items():
        frame, decision, observed = frame_map[fid], decision_map[fid], observations[fid]
        wire = transmitted[fid]["run4_identity"]
        for key in ("frame_id", "tensor_seq", "session_uuid", "decision_seq", "ticket_seq", "mode_id", "q_e4", "frame_kind", "reward_requested"):
            if str(wire[key]) != frame[key]:
                raise ValueError(f"transmitted/frame identity mismatch: {fid}/{key}")
        if wire["reward_requested"] is not True:
            raise ValueError("Policy frame did not request a reward")
        for key in ("frame_id", "decision_seq", "mode_id", "q_e4", "payload_bytes"):
            if int(ticket[key]) != int(frame[key]):
                raise ValueError(f"ticket/frame identity mismatch: {fid}/{key}")
        for key in ("frame_id", "tensor_seq", "mode_id", "q_e4"):
            if int(decision[key]) != int(frame[key]):
                raise ValueError(f"UE decision/frame identity mismatch: {fid}/{key}")
        if decision["decision_session_uuid"] != wire["session_uuid"]:
            raise ValueError("UE decision session mismatch")
        resolution_key = (wire["session_uuid"], int(wire["decision_seq"]))
        resolution = resolutions[resolution_key]
        used_resolutions.add(resolution_key)
        identity = {key: resolution["identity"][key] for key in ("session_uuid", "decision_seq", "ue_id")}
        if observed["decision_identity"] != identity or observed["boundary"]["identity"] != identity:
            raise ValueError("SNR/actual decision identity mismatch")
        boundary = observed["boundary"]
        if (boundary["clock_domain"] != decision["state_commit"]["domain"]
                or boundary["clock_domain"] != decision["action_open"]["domain"]
                or boundary["clock_domain"] != "CLOCK_MONOTONIC_RAW"
                or int(boundary["state_commit_timestamp_ns"]) != int(decision["state_commit"]["ns"])
                or int(boundary["action_open_timestamp_ns"]) != int(decision["action_open"]["ns"])
                or int(boundary["action_open_timestamp_ns"]) != int(frame["action_open_raw_ns"])
                or int(boundary["action_open_timestamp_ns"]) != int(resolution["action_open_timestamp_ns"])):
            raise ValueError("SNR/actual decision boundary mismatch")
        if len(decision["features"]) != 22 or decision["features"][21] != observed["snr_scaled"]:
            raise ValueError("SNR/actual actor feature mismatch")
        if resolution["terminal"] != ticket["resolution_terminal"]:
            raise ValueError("Ticket/resolution terminal mismatch")
        reply, evaluation = feedback.get(fid), evaluations.get(fid)
        if evaluation is not None and evaluation["run4_identity"] != wire:
            raise ValueError("Evaluation/transmission full identity mismatch")
        latency = None
        if reply is not None:
            if (evaluation is None or reply["sha256"] != evaluation["feedback_sha256"]
                    or reply["q_perc"] != evaluation["q_perc"]
                    or reply["kind"] != evaluation["kind"] or reply["reason"] != evaluation["reason"]):
                raise ValueError("Received feedback/edge evaluation mismatch")
            # R4FB serializes Q_perc as binary64 ('d'), so JSON floats agree
            # exactly; no floating-point tolerance is silently introduced.
            if reply["receipt"]["domain"] != "CLOCK_MONOTONIC_RAW":
                raise ValueError("Feedback clock mismatch")
            latency = (int(reply["receipt"]["ns"]) - int(frame["action_open_raw_ns"])) / 1e6
            if latency < 0 or abs(latency - float(ticket["action_open_to_feedback_ms"])) > 1e-8:
                raise ValueError("latency join mismatch")
        elif ticket["action_open_to_feedback_ms"] not in ("", None):
            raise ValueError("Missing feedback has a populated latency")
        quality = evaluation["q_perc"] if evaluation is not None else None
        if quality is not None and (not math.isfinite(quality) or not 0 <= quality <= 1):
            raise ValueError("Invalid edge-measured quality")
        quality_status = ("NO_EVALUATION" if evaluation is None else "UNDEFINED_QUALITY" if quality is None
                          else "DEFINED_RECEIVED_FEEDBACK" if reply is not None else "DEFINED_NO_FEEDBACK")
        record = dict(frame_number=int(frame["tensor_seq"]) + 1, frame_id=fid,
            session_uuid=wire["session_uuid"], decision_seq=int(wire["decision_seq"]), ticket_seq=int(wire["ticket_seq"]),
            mode_id=int(ticket["mode_id"]), q_e4=int(ticket["q_e4"]), payload_bytes=int(ticket["payload_bytes"]),
            latency_ms=latency, timely=latency is not None and latency <= 170,
            q_perc=quality, quality_source="EDGE_EVALUATION" if evaluation is not None else None,
            quality_status=quality_status, feedback_available=reply is not None,
            terminal=ticket["resolution_terminal"], snr_db=observed["snr"]["value_db"],
            evaluation_available=evaluation is not None)
        if evaluation is not None:
            for name in ("vehicle", "person"):
                record[name] = components(evaluation["measurement"], name)
        joined.append(record)
    if used_resolutions != set(resolutions):
        raise ValueError("Unaccounted resolution identity")
    joined.sort(key=lambda row: row["frame_number"])
    return joined


def load(capture):
    capture = Path(capture).resolve()
    qualification = read(capture / "QUALIFICATION_RESULT.json")
    if qualification.get("passed") is not True or not all(qualification["gates"].values()):
        raise ValueError("Capture did not pass qualification; no performance plots")
    postflight = read(capture.parent / "RUN5_POSTFLIGHT.json")
    if postflight.get("passed") is not True:
        raise ValueError("Run-5 integration postflight failed; no performance plots")
    snr_check = verify_snr_capture.verify_capture(capture)
    if snr_check["passed"] is not True:
        raise ValueError("Run-5 actor/SNR verification failed")
    gt_check = verify_live_gt_capture.verify_capture(capture)
    export = capture / "export_v1"
    manifest = read(export / "MANIFEST.json")
    for name, digest in manifest["export_sha256"].items():
        if Path(name).name != name or sha(export / name) != digest:
            raise ValueError(f"export hash mismatch: {name}")
    cells = list((capture / "cells").iterdir())
    if len(cells) != 1:
        raise ValueError("Expected one registered network cell")
    cell = cells[0]
    ue = read(cell / "run4_phase6/PHASE6_UE_EVIDENCE.json")
    edge = read(cell / "run4_phase6/edge_report.json")
    snr = read(cell / "run5_policy/SNR_EVIDENCE.json")
    frames = rows(export / "frames.csv")
    tickets = rows(export / "tickets.csv")
    joined = join_records(ue=ue, edge=edge, snr=snr, frames=frames, tickets=tickets)
    verification = dict(capture=str(capture), export_manifest_sha256=sha(export / "MANIFEST.json"),
                        qualification_sha256=sha(capture / "QUALIFICATION_RESULT.json"),
                        postflight_sha256=sha(capture.parent / "RUN5_POSTFLIGHT.json"),
                        actor_snr=snr_check, live_gt=gt_check)
    return joined, frames, verification


def render(capture, output):
    records, frames, verification = load(capture)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "savefig.facecolor": "white"})
    x = np.asarray([r["frame_number"] for r in records])
    value = lambda key: np.asarray([np.nan if r.get(key) is None else r[key] for r in records])
    latency, quality, modes, q, snr = (value(k) for k in ("latency_ms", "q_perc", "mode_id", "q_e4", "snr_db"))
    timely = value("timely").astype(bool)
    context = f"Run-5 • Favorable stable • {len(frames):,} transmitted frames • {len(records)} policy decisions"
    note = "Single-run descriptive results; corrected live CARLA labels + pedestrian AVO ≥0.65, not full offline FCOS-mask parity."
    def finish(fig, name, title, footer=note):
        fig.suptitle(title, fontsize=17, y=.98)
        fig.text(.5, .925, context, ha="center", fontsize=10)
        fig.text(.04, .02, footer, fontsize=9, va="bottom")
        for text in fig.findobj(match=Text):
            text.set_fontweight("bold")
        fig.savefig(output / name, dpi=190)
        plt.close(fig)
    def axis(ax):
        ax.set_xlim(0, len(frames)); ax.grid(alpha=.2)
        ax.set_xlabel("Transmitted frame number")
    labels = [f"{i}: {family} / {quant}" for i, (family, quant) in enumerate(
        ( (f, qn) for f in ("noAE", "AE128", "AE64", "AE32") for qn in ("UINT8", "UINT6", "UINT4") ))]
    counts = np.zeros((12, 10), dtype=int)
    for r in records:
        counts[r["mode_id"], min(r["q_e4"] // 1000, 9)] += 1
    fig, ax = plt.subplots(figsize=(13, 7)); fig.subplots_adjust(left=.19, right=.92, bottom=.2, top=.83)
    im = ax.imshow(counts, aspect="auto", cmap="Blues", vmin=0)
    ax.set_yticks(range(12), labels)
    ax.set_xticks(range(10), [f"{i/10:.1f}–{min((i+1)/10,.98):.2f}" for i in range(10)], rotation=30)
    ax.set_xlabel("Drop fraction q (higher = more cells dropped)")
    for i, j in zip(*np.nonzero(counts)):
        ax.text(j, i, str(counts[i, j]), ha="center", va="center", color="white" if counts[i,j] > .55*counts.max() else "#222")
    fig.colorbar(im, ax=ax, label="New policy decisions")
    finish(fig, "01_action_heatmap.png", "Which actions did Run-5 select?", "Holds and fallbacks are excluded; continuous q is binned for display only. " + note)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.82, hspace=.28)
    axes[0].plot(x, latency, lw=.9, color="#286a98", label="Measured feedback latency")
    axes[0].scatter(x[~timely], latency[~timely], color="#ad3b2e", s=13)
    axes[0].axhline(170, ls="--", color="#ad3b2e", label="170-ms deadline")
    axes[0].set_ylabel("Action-open → feedback (ms)"); axes[0].set_ylim(bottom=0)
    axes[0].legend(ncol=2, loc="upper right")
    axes[1].plot(x, snr, color="#2a8071", lw=1)
    axes[1].set_ylabel("Causal SNR proxy (dB)"); axes[1].set_ylim(0, 28)
    for ax in axes: axis(ax)
    finish(fig, "02_feedback_latency_and_snr.png", f"Run-5 feedback latency — {timely.sum()}/{len(records)} timely ({timely.mean():.1%})",
           "Raw decisions, no smoothing. Missing replies remain missing and count as deadline misses; SNR is the external simulator proxy.")

    fig, ax = plt.subplots(figsize=(14, 6)); fig.subplots_adjust(left=.08, right=.98, bottom=.17, top=.82)
    defined = np.isfinite(quality)
    received_quality = np.isfinite(latency)
    ax.scatter(x[defined & timely], quality[defined & timely], s=15, color="#247a74", label="Timely, defined quality")
    ax.scatter(x[defined & ~timely & received_quality], quality[defined & ~timely & received_quality], s=22, facecolors="none", edgecolors="#ad3b2e", label="Late measured quality")
    ax.scatter(x[defined & ~received_quality], quality[defined & ~received_quality], s=30, marker="x", color="#a36c18", label="Measured at edge; no feedback")
    ax.scatter(x[~defined], np.full((~defined).sum(), -.03), s=35, marker="|", color="#777", label="Undefined / not evaluated")
    ax.set_ylabel("Composite perception quality Q_perc"); ax.set_ylim(-.06, 1.02); axis(ax); ax.legend(ncol=3)
    finish(fig, "03_quality_over_frames.png", "Perception quality of Run-5-selected actions",
           "Quality comes from completed edge evaluations. Late/missing-feedback quality is descriptive only; ticket outcomes are unchanged.\n" + note)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.15, top=.83, hspace=.32, wspace=.24)
    for col, cls in enumerate(("vehicle", "person")):
        extract = lambda key: np.asarray([r.get(cls, {}).get(key) if r.get(cls, {}).get(key) is not None else np.nan for r in records])
        axes[0,col].plot(x, extract("iou"), lw=.8); axes[0,col].set(title=f"{cls.title()}: segmentation IoU", ylim=(0,1), ylabel="Intersection / union")
        axes[1,col].plot(x, extract("xy_median_m"), lw=.8, marker=".", ms=2)
        axes[1,col].set(title="Median matched-object XY error", ylabel="Error (m)", ylim=(0,None))
        axes[2,col].plot(x, extract("recall"), lw=.8, label="Recall")
        axes[2,col].plot(x, extract("precision"), lw=.8, label="Precision")
        axes[2,col].set(ylabel="Detection coverage", ylim=(-.02,1.02)); axes[2,col].legend(ncol=2)
        for ax in axes[:,col]: axis(ax)
    finish(fig, "04_segmentation_localization_and_coverage.png", "Separate perception metrics, with detection coverage",
           "XY error includes matched objects only; no match remains missing. Person localization uses AVO; vehicle matching is unchanged.\n" + note)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8)); fig.subplots_adjust(left=.08, right=.98, bottom=.18, top=.82, wspace=.25)
    received = np.sort(latency[np.isfinite(latency)])
    axes[0].step(received, np.arange(1,len(received)+1)/len(records), where="post")
    axes[0].axvline(170, ls="--", color="#ad3b2e"); axes[0].set(xlabel="Feedback latency (ms)", ylabel="Fraction of all decisions", ylim=(0,1.01), xlim=(0,None))
    axes[1].hist(received, bins=30, color="#286a98"); axes[1].axvline(170, ls="--", color="#ad3b2e")
    axes[1].set(xlabel="Feedback latency (ms)", ylabel="Received-feedback count")
    for ax in axes: ax.grid(alpha=.2)
    finish(fig, "05_latency_cdf_and_frequency.png", "Feedback latency distribution", "CDF denominator includes missing replies; histogram contains received replies only. No latency clipping.")

    fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.96, bottom=.13, top=.85, hspace=.43, wspace=.15)
    for mode, ax in enumerate(axes.flat):
        selected = modes == mode
        ax.scatter(q[selected] / 10000, latency[selected], s=16, color="#286a98", alpha=.7)
        ax.axhline(170, color="#ad3b2e", ls="--", lw=1)
        ax.set_title(f"{labels[mode]} · n={selected.sum()}", fontsize=10)
        ax.set_xlim(0,.98); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
        ax.set_xticks([0,.5,.98])
    fig.supxlabel("Selected drop fraction q", y=.07)
    fig.supylabel("Action-open → feedback (ms)", x=.015)
    finish(fig, "06_latency_by_selected_action.png", "Latency of Run-5-selected (mode, q) actions",
           "Each dot is a received reply; all late replies are retained. Observational associations, not causal action comparisons.")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(left=.17, right=.90, bottom=.14, top=.83, hspace=.2)
    marks = axes[0].scatter(x, modes, c=q/10000, vmin=0, vmax=.98, cmap="viridis", marker="s", s=15)
    axes[0].set_yticks(range(12), labels); axes[0].set_ylim(11.5, -.5)
    axes[1].plot(x, q/10000, lw=.7, marker=".", ms=2); axes[1].set(ylabel="Drop fraction q", ylim=(0,1))
    for ax in axes: axis(ax)
    fig.colorbar(marks, ax=axes, fraction=.02, pad=.025, label="Drop fraction q")
    finish(fig, "07_action_timeline.png", "Run-5 action choices across transmitted frames",
           "Only actual policy decisions are shown; held/fallback frames do not count as new selections. No smoothing or action snapping.")
    summary = dict(frames=len(frames), frame_kinds=dict(collections.Counter(r["frame_kind"] for r in frames)),
        decisions=len(records), timely=int(timely.sum()), missing_replies=int((~np.isfinite(latency)).sum()),
        outcomes=dict(collections.Counter(r["terminal"] for r in records)), latency_ms=stats([r["latency_ms"] for r in records]),
        quality=stats([r["q_perc"] for r in records]), percentile_method="NumPy linear", claim_scope="SINGLE_RUN_DESCRIPTIVE",
        quality_population="DEFINED_EDGE_EVALUATIONS_INCLUDING_LATE_OR_MISSING_FEEDBACK",
        quality_statuses=dict(collections.Counter(r.get("quality_status", "TEST_FIXTURE_UNSPECIFIED") for r in records)),
        sensor_sync_and_gt_contract="CURRENT_LIVE_AVO_VERIFIED; not like-for-like quality comparison against old Run4 GT")
    save(output / "SUMMARY.json", summary); save(output / "POLICY_ROWS.json", records)
    save(output / "INPUT_VERIFICATION.json", verification)
    save(output / "OUTPUT_SHA256.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.capture, args.output), indent=2))
