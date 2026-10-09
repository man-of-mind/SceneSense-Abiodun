"""Join retained Run-5 measurements and render descriptive, CPU-only PNGs.

No inference, simulator, radio, container, training or live service is started.
Only a new create-only output directory is written. Existing inputs are verified
against their sealed manifests, joined by identity, and hashed again at the end.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_run5_live_avo_v1 import render_results as canonical


# Frozen live_state_v2.TRAINING_SCALING, digest cb1a3e4d9c9478a9106287fc3cf8b1bac5f92cfe5294fbf32543da0679155671.
SI_CENTER = 117.08553307797729
SI_SCALE = 13.336440703846854
BACKLOG_LOG_SCALE = math.log1p(50_000_000)
RAW = "CLOCK_MONOTONIC_RAW"
WALL = "PHYSICAL_CAPTURE_WALL_CLOCK_REALTIME"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_bound_file(root, name, digest, inputs):
    path = Path(root) / name
    require(not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Manifest path escapes source root")
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked input: {name}")
    require(sha(path) == digest, f"Input hash mismatch: {name}")
    inputs[str(path.resolve())] = digest
    return path


def verify_inputs_unchanged(inputs):
    require(all(sha(path) == digest for path, digest in inputs.items()), "Retained source changed during reporting")


def finite(value, label):
    value = float(value)
    require(math.isfinite(value), f"Nonfinite {label}")
    return value


def stats(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    require(bool(np.isfinite(a).all()), "Nonfinite statistic input")
    return {"n": len(a), **({"mean": float(a.mean()), "p50": float(np.quantile(a, .5)),
        "p95": float(np.quantile(a, .95)), "p99": float(np.quantile(a, .99)),
        "min": float(a.min()), "max": float(a.max())} if len(a) else {})}


def frame_aoi(events):
    """Exact piecewise-linear latest-installed-FRAME age between installations.

Both timestamps are from the same host wall-clock domain. No value is imputed
before the first installation or after the last. This is not object/track AoI.
"""
    require(bool(events), "No map-install events")
    ordered = sorted(events, key=lambda r: r["map_install_wall_s"])
    canonical.unique_index(ordered, lambda r: r["frame_id"], "AoI frame")
    origin = min(e["capture_wall_s"] for e in ordered)
    points, previous = [], None
    for event in ordered:
        require(event["clock_domain"] == WALL, "AoI clock mismatch")
        install = finite(event["map_install_wall_s"], "map install timestamp")
        capture = finite(event["capture_wall_s"], "map capture timestamp")
        require(install >= capture, "Map install precedes source capture")
        if previous is not None:
            require(install > previous["map_install_wall_s"], "Duplicate or regressing map install time")
            require(capture > previous["capture_wall_s"], "Regressing map capture time")
            points.append(dict(elapsed_wall_s=install-origin, wall_s=install,
                age_ms=(install-previous["capture_wall_s"])*1000, phase="BEFORE_INSTALL",
                latest_frame_id=previous["frame_id"], event_frame_id=event["frame_id"], clock_domain=WALL))
        points.append(dict(elapsed_wall_s=install-origin, wall_s=install,
            age_ms=(install-capture)*1000, phase="AFTER_INSTALL",
            latest_frame_id=event["frame_id"], event_frame_id=event["frame_id"], clock_domain=WALL))
        previous = event
    return points


def join_story(*, records, ue, edge, snr, frames, tickets, map_rows, map_identities):
    rebuilt = canonical.join_records(ue=ue, edge=edge, snr=snr, frames=frames, tickets=tickets)
    require(rebuilt == records, "Canonical policy result differs from raw-source identity join")
    key = lambda r: int(r["frame_id"])
    fidx = canonical.unique_index(frames, key, "export frame")
    d = canonical.unique_index([r for r in ue["decisions"] if r["kind"] == "POLICY_DECISION"], key, "decision")
    transmitted = canonical.unique_index(ue["transmitted_identities"], key, "transmission")
    midx = canonical.unique_index(map_rows, key, "map install")
    identities = canonical.unique_index(map_identities, key, "map identity")
    observations = canonical.unique_index([r for r in snr["decision_snr"] if r["admitted"]], key, "SNR")
    require(set(midx) == set(fidx) == set(identities), "Missing/foreign map-frame identity")
    require(len({r["stream_id"] for r in map_rows}) == 1, "Multiple UE streams require separate AoI timelines")
    origin = min(int(r["capture_wall_ns"]) / 1e9 for r in frames)
    events = []
    for fid, frame in fidx.items():
        m, identity = midx[fid], identities[fid]
        require(json.loads(identity["run4_identity_json"]) == transmitted[fid]["run4_identity"],
                "Map/transmission full identity mismatch")
        require(m["outcome"] == identity["outcome"] == "RESULT_INSTALLED" and str(m["accepted"]) in ("1", "True"),
                "This existing-capture report requires explicitly installed frames")
        capture = finite(m["capture_timestamp"], "capture wall timestamp")
        require(capture == int(frame["capture_wall_ns"])/1e9, "Map/frame capture timestamp mismatch")
        install = finite(m["map_install_at"], "install wall timestamp")
        age = (install-capture)*1000
        require(age >= 0 and abs(age-finite(m["map_age_at_install_ms"], "map age")) <= 1e-6,
                "Map age arithmetic mismatch")
        events.append(dict(frame_id=fid, frame_number=int(frame["tensor_seq"])+1,
            session_uuid=frame["session_uuid"], decision_seq=int(frame["decision_seq"]),
            tensor_seq=int(frame["tensor_seq"]), frame_kind=frame["frame_kind"], stream_id=m["stream_id"],
            mode_id=int(frame["mode_id"]), q_e4=int(frame["q_e4"]),
            capture_wall_s=capture, map_install_wall_s=install, map_age_at_install_ms=age,
            map_install_elapsed_wall_s=install-origin, capture_elapsed_wall_s=capture-origin,
            record_count=int(m["record_count"]), clock_domain=WALL))
    events.sort(key=lambda r: r["frame_number"])
    eidx = {r["frame_id"]: r for r in events}
    decisions = []
    raw_origin = min(int(row["action_open"]["ns"]) for row in d.values())
    for record in records:
        fid = record["frame_id"]
        decision, observation, event = d[fid], observations[fid], eidx[fid]
        require(decision["capture"]["domain"] == WALL and
                int(decision["capture"]["ns"]) == int(fidx[fid]["capture_wall_ns"]), "Source capture clock/identity mismatch")
        require(decision["state_commit"]["domain"] == decision["action_open"]["domain"] == RAW,
                "Decision clock mismatch")
        feature, stages = decision["features"], decision["stages"]
        commit, opened = int(decision["state_commit"]["ns"]), int(decision["action_open"]["ns"])
        start, end = int(stages["si_p40_start_raw_ns"]), int(stages["si_p40_end_raw_ns"])
        rgb = int(stages["rgb_receipt_raw_ns"])
        require(rgb <= start <= end <= commit <= opened, "Scene measurement/source ordering mismatch")
        require(observation["snr"]["clock_domain"] == RAW, "SNR clock mismatch")
        effective, heartbeat = (int(observation["snr"][k]) for k in ("effective_since_ns", "heartbeat_ns"))
        require(effective <= commit and heartbeat <= commit, "Future SNR source at state commit")
        mcs = round(float(feature[2])*28)
        backlog_f = math.expm1(float(feature[3])*BACKLOG_LOG_SCALE)
        backlog = round(backlog_f)
        require(0 <= mcs <= 28 and abs(feature[2]-mcs/28) < 1e-12 and backlog >= 0
                and abs(backlog_f-backlog) < 1e-5, "Network feature inversion mismatch")
        mcs_age, backlog_age = (finite(decision[k], k)/1e6 for k in ("mcs_age_ns", "backlog_age_ns"))
        require(min(mcs_age, backlog_age) >= 0, "Negative measurement age")
        row = dict(record, camera_si=feature[0]*SI_SCALE+SI_CENTER, radar_p40=float(feature[1]),
            si_p40_compute_ms=(end-start)/1e6, scene_rgb_receipt_age_ms=(commit-rgb)/1e6,
            scene_compute_complete_age_ms=(commit-end)/1e6,
            ul_mcs=mcs, rlc_backlog_bytes=backlog, mcs_age_ms=mcs_age, rlc_backlog_age_ms=backlog_age,
            snr_effective_age_ms=(commit-effective)/1e6, snr_heartbeat_age_ms=(commit-heartbeat)/1e6,
            action_elapsed_raw_s=(opened-raw_origin)/1e9, action_open_raw_ns=opened,
            state_commit_raw_ns=commit, capture_wall_ns=int(decision["capture"]["ns"]),
            capture_elapsed_wall_s=event["capture_elapsed_wall_s"],
            map_install_elapsed_wall_s=event["map_install_elapsed_wall_s"],
            map_age_at_install_ms=event["map_age_at_install_ms"], map_record_count=event["record_count"],
            class_confidence_status="NOT_RETAINED", classification_confidence=None,
            quality_interpretation="PROVISIONAL_RAW_SEMANTIC_REFERENCE_LIMITATION")
        require(math.isfinite(row["camera_si"]) and row["camera_si"] >= 0 and 0 <= row["radar_p40"] <= 1,
                "Invalid scene feature")
        decisions.append(row)
    return decisions, events, frame_aoi(events)


def load_inputs(capture_root, results):
    root, results = Path(capture_root).resolve(), Path(results).resolve()
    inputs = {}
    result_manifest = read(results / "OUTPUT_SHA256.json")
    for name, digest in result_manifest.items():
        verify_bound_file(results, name, digest, inputs)
    inputs[str(results / "OUTPUT_SHA256.json")] = sha(results / "OUTPUT_SHA256.json")
    original_verification = read(results / "INPUT_VERIFICATION.json")
    require(Path(original_verification["capture"]).resolve() == root / "capture", "Capture/results mismatch")
    postpath = verify_bound_file(root, "RUN5_POSTFLIGHT.json", original_verification["postflight_sha256"], inputs)
    post = read(postpath)
    require(post["passed"] is True and len(post["checks"]) == 11 and all(post["checks"].values()), "Run5 postflight not qualified")
    manifest = read(verify_bound_file(root, "RUN5_POSTFLIGHT_INPUTS.json", post["input_manifest_sha256"], inputs))["files"]
    capture = root / "capture"
    cells = list((capture / "cells").iterdir())
    require(len(cells) == 1, "Expected one cell")
    cell = cells[0]
    def bound(path):
        name = str(Path(path).relative_to(root))
        require(name in manifest, f"Required source absent from postflight manifest: {name}")
        return verify_bound_file(root, name, manifest[name], inputs)
    names = {
        "ue": cell / "run4_phase6/PHASE6_UE_EVIDENCE.json",
        "edge": cell / "run4_phase6/edge_report.json",
        "snr": cell / "run5_policy/SNR_EVIDENCE.json",
        "frames": capture / "export_v1/frames.csv", "tickets": capture / "export_v1/tickets.csv",
        "map_rows": cell / "direct_edge_map/direct_map_ingest.csv",
        "map_identities": cell / "direct_edge_map/run4_map_identity.csv"}
    loaded = {name: (csv_rows(bound(path)) if path.suffix == ".csv" else read(bound(path))) for name, path in names.items()}
    radio = csv_rows(bound(cell / "radio_trace.csv"))
    map_report = read(bound(cell / "direct_edge_map/direct_map_report.json"))
    records = read(results / "POLICY_ROWS.json")
    summary = read(results / "SUMMARY.json")
    decisions, events, aoi = join_story(records=records, **loaded)
    require(len(events) == summary["frames"] and len(decisions) == summary["decisions"], "Summary population mismatch")
    profiles = dict(collections.Counter(r["profile_id"] for r in radio))
    require(set(profiles) == {"FAVORABLE_STABLE"}, "Presentation is registered for existing favorable-stable capture only")
    counts = {cls: {key: sum(r["measurement"][f"seg_{cls}_{key}_pixels"] for r in loaded["edge"]["evaluations"])
                       for key in ("gt", "pred", "intersection", "union")} for cls in ("person", "vehicle")}
    provenance = dict(source_inputs_sha256=inputs, raw_sources=str(root), canonical_results=str(results),
        postflight_checks=post["checks"], frame_count=len(events), decision_count=len(decisions),
        radio_profiles=profiles, map_render_thread_started=map_report["render_thread_started"],
        segmentation_counts=counts, percentile_method="NumPy linear",
        SI_scaling=dict(center=SI_CENTER, scale=SI_SCALE, inverse_of="registered feature0"),
        backlog_scaling=dict(log1p_scale=BACKLOG_LOG_SCALE, inverse_of="registered feature3"),
        policy_features="Original frozen Run5,22D,previous Q_perc retained; no retraining",
        causal_join="Exact frame/tensor/session/decision/ticket identities; no timestamp-nearest joins",
        confidence_status="NOT_RETAINED; no score distributions or confidence bins can be reconstructed",
        channel_step_hold_status="NOT_PRESENT; natural fluctuations in one favorable-stable profile only",
        clock_rules={"decision_measurements": RAW, "map_and_capture": WALL,
                     "clock_domains_never_subtracted": True, "common_frame_axes": "transmitted tensor_seq+1"},
        limitations=["Single-run descriptive; no convergence, causal action benefit, SNR benefit or adaptation claim.",
            "P40 is mean clipped proximity weight within40m over current-sweep valid returns, not object count or percentile.",
            "Scene/network values are available at policy decisions; holds/fallbacks are not imputed new measurements.",
            "Frame AoI is latest installed frame-source age between first and last install; not object/track freshness.",
            "All edge-evaluated policy frames included for localization, including late feedback; repeated observations, not unique actors.",
            "Matched XY error is conditional on matches; precision/recall accompany it.",
            "Person localization uses AVO; vehicle matching unchanged. Q_perc/person segmentation remain provisional due to empty GT-person semantic coverage.",
            "Map ingestion was active but rendering disabled; no two-UE or occlusion-alert trial was captured."])
    return decisions, events, aoi, summary, provenance


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_csv(path, rows):
    keys = list(rows[0])
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":"), allow_nan=False) if isinstance(v, (dict,list)) else v for k,v in row.items()})


def statistical_series(decisions, events):
    """Exact empirical populations, with absent values excluded only explicitly."""
    received = [r for r in decisions if r["latency_ms"] is not None]
    require(bool(decisions) and bool(events), "Empty statistical population")
    result = dict(all_decisions=len(decisions), received=len(received),
        missing=len(decisions)-len(received), timely=sum(r["timely"] for r in decisions),
        timely_undefined=sum(r["timely"] and r["q_perc"] is None for r in decisions),
        terminal_counts=dict(collections.Counter(r["terminal"] for r in decisions)),
        feedback_ms=sorted(r["latency_ms"] for r in received),
        defined_feedback_ms=[r["latency_ms"] for r in received if r["q_perc"] is not None],
        undefined_feedback_ms=[r["latency_ms"] for r in received if r["q_perc"] is None and r["evaluation_available"]],
        unevaluated_feedback_ms=[r["latency_ms"] for r in received if r["q_perc"] is None and not r["evaluation_available"]],
        si_p40_ms=sorted(r["si_p40_compute_ms"] for r in decisions),
        xy_errors_m={cls:sorted(error for r in decisions for error in r.get(cls,{}).get("errors_m",[]))
                     for cls in ("vehicle","person")},
        installed_frame_age_ms=sorted(r["map_age_at_install_ms"] for r in events),
        selected_mode_counts=[sum(r["mode_id"]==mode for r in decisions) for mode in range(12)])
    for values in (result["feedback_ms"],result["si_p40_ms"],result["installed_frame_age_ms"],*result["xy_errors_m"].values()):
        require(all(math.isfinite(v) and v >= 0 for v in values), "Invalid empirical measurement")
    require(sum(len(result[k]) for k in ("defined_feedback_ms","undefined_feedback_ms","unevaluated_feedback_ms")) == len(received),
            "Feedback histogram population mismatch")
    return result


def plot_statistics(decisions, events, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    data=statistical_series(decisions,events)
    fig,axes=plt.subplots(3,2,figsize=(16,13))
    fig.subplots_adjust(left=.09,right=.97,bottom=.205,top=.85,hspace=.44,wspace=.23)
    fig.suptitle("Latency, scene computation, localization and freshness — distributions",y=.97,fontsize=17)
    fig.text(.5,.927,f"Original Run-5 · favorable-stable · {len(events)} installed frames · {len(decisions)} policy decisions",ha="center",fontsize=11)
    def ecdf(ax,values,denominator,*,color,label):
        if values:
            ax.step([0,*values],[0,*[i/denominator for i in range(1,len(values)+1)]],where="post",color=color,label=label,lw=1.3)
        ax.set_ylim(0,1.02); ax.set_xlim(left=0)
    ax=axes[0,0]
    ecdf(ax,data["feedback_ms"],data["all_decisions"],color="#286a98",label=f"Received {data['received']} / all {data['all_decisions']} decisions")
    ax.axvline(170,color="#b04436",ls="--",lw=1,label="170-ms deadline")
    ax.scatter([170],[data["timely"]/data["all_decisions"]],color="#b04436",s=22,zorder=3)
    ax.text(.98,.18,f"Timely: {data['timely']}/{data['all_decisions']} ({data['timely']/data['all_decisions']:.1%})\nMissing replies: {data['missing']}",transform=ax.transAxes,ha="right",fontsize=10)
    ax.set(title="Feedback latency: empirical CDF",xlabel="Action-open → feedback (ms)",ylabel="Fraction of all decisions"); ax.legend(fontsize=9,loc="upper left")
    ax=axes[0,1]
    received=data["feedback_ms"]
    bins=np.linspace(0,max(170,max(received,default=170))*1.02,26)
    series=[data["defined_feedback_ms"],data["undefined_feedback_ms"]]
    labels=[f"Defined quality (n={len(series[0])})",f"Undefined quality (n={len(series[1])})"]
    colors=["#286a98","#bf8846"]
    if data["unevaluated_feedback_ms"]:
        series.append(data["unevaluated_feedback_ms"]); labels.append(f"Not evaluated (n={len(series[-1])})"); colors.append("#777777")
    ax.hist(series,bins=bins,stacked=True,color=colors,label=labels)
    ax.axvline(170,color="#b04436",ls="--",lw=1)
    ax.set(title="Feedback latency: observed frequency",xlabel="Action-open → feedback (ms)",ylabel="Received-reply count",xlim=(0,None)); ax.legend(fontsize=9)
    ax=axes[1,0]
    ecdf(ax,data["si_p40_ms"],len(data["si_p40_ms"]),color="#27836d",label=f"Policy-decision frames (n={len(data['si_p40_ms'])})")
    ax.set(title="Combined camera SI + radar P40 computation",xlabel="Combined compute duration (ms)",ylabel="Fraction of policy frames"); ax.legend(fontsize=9,loc="lower right")
    ax=axes[1,1]
    for cls,color in (("vehicle","#286a98"),("person","#b46541")):
        values=data["xy_errors_m"][cls]
        ecdf(ax,values,len(values),color=color,label=f"{cls.title()}: {len(values)} matched observations")
    ax.set(title="Matched-object localization error",xlabel="Matched XY error (m)",ylabel="Fraction of class matches"); ax.legend(fontsize=9,loc="lower right")
    ax=axes[2,0]
    ecdf(ax,data["installed_frame_age_ms"],len(events),color="#27836d",label=f"All installed frames (n={len(events)})")
    ax.set(title="Age of installed frame updates",xlabel="Capture → map install (ms)",ylabel="Fraction of installed frames"); ax.legend(fontsize=9,loc="lower right")
    ax=axes[2,1]
    bars=ax.bar(range(12),data["selected_mode_counts"],color="#286a98")
    for bar,count in zip(bars,data["selected_mode_counts"]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(data["selected_mode_counts"])*.025,str(count),ha="center",fontsize=10)
    ax.set(title="Frequency of new selected modes",xlabel="Mode ID (0–2 noAE; 3–5 AE128; 6–8 AE64; 9–11 AE32)",ylabel="Policy-decision count",xticks=list(range(12)),ylim=(0,max(data["selected_mode_counts"])*1.16))
    for ax in axes.flat: ax.grid(alpha=.2); ax.set_axisbelow(True)
    fig.text(.045,.13,
        f"All available late feedback retained. Timely includes {data['timely_undefined']} undefined-quality exclusions; SUCCESS={data['terminal_counts'].get('SUCCESS',0)} is a separate count.",fontsize=10)
    fig.text(.045,.098,"Feedback CDF denominator includes missing replies; histogram contains received replies only. Curves are empirical, without smoothing.",fontsize=10)
    fig.text(.045,.066,"XY errors are matched observations only (3-m matching radius); misses/false positives require coverage plots. Repeated objects are not unique actors.",fontsize=10)
    fig.text(.045,.034,"Freshness CDF uses install-age samples, NOT irregular AoI points. Confidence absent; person segmentation and Q_perc remain provisional.",fontsize=10)
    for text in fig.findobj(match=Text): text.set_fontweight("bold")
    with (output/"03_statistical_distributions.png").open("xb") as stream: fig.savefig(stream,format="png",dpi=180)
    plt.close(fig)
    return data


def plot_story(decisions, events, aoi, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10.5,
        "axes.spines.top":False, "axes.titleweight":"bold", "axes.labelweight":"bold"})
    x = np.asarray([r["frame_number"] for r in decisions])
    vals = lambda k: np.asarray([np.nan if r.get(k) is None else r[k] for r in decisions])
    context = f"Original Run-5 · favorable-stable profile · {len(events)} transmitted frames · {len(decisions)} policy decisions"
    def frameaxis(ax):
        ax.set_xlim(0, 1000); ax.set_xlabel("Transmitted frame number"); ax.grid(alpha=.2)
    def finish(fig, name, title, notes):
        fig.suptitle(title, y=.97, fontsize=18)
        fig.text(.5,.927,context,ha="center",fontsize=11)
        for i, note in enumerate(notes): fig.text(.055,.10-i*.027,note,fontsize=9.5)
        for text in fig.findobj(match=Text): text.set_fontweight("bold")
        with (output/name).open("xb") as stream: fig.savefig(stream,format="png",dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3,2,figsize=(16,13)); fig.subplots_adjust(left=.09,right=.90,bottom=.19,top=.85,hspace=.43,wspace=.37)
    ax=axes[0,0]; ax.plot(x,vals("camera_si"),color="#286a98",lw=.8); ax.set(title="Scene texture and nearby radar support",ylabel="Camera SI")
    twin=ax.twinx(); twin.plot(x,vals("radar_p40"),color="#27836d",lw=.8); twin.set(ylabel="Radar P40",ylim=(0,1))
    ax.tick_params(axis="y",labelcolor="#286a98"); twin.tick_params(axis="y",labelcolor="#27836d")
    ax=axes[0,1]; ax.plot(x,vals("snr_db"),color="#27836d",lw=.85); ax.set(title="Channel state actually admitted by the actor",ylabel="External SNR proxy (dB)",ylim=(0,28))
    twin=ax.twinx(); twin.plot(x,vals("ul_mcs"),color="#b46541",lw=.8); twin.set(ylabel="Prior uplink MCS",ylim=(0,29))
    ax.tick_params(axis="y",labelcolor="#27836d"); twin.tick_params(axis="y",labelcolor="#b46541")
    ax=axes[1,0]; points=ax.scatter(x,vals("mode_id"),c=vals("q_e4")/10000,cmap="viridis",vmin=0,vmax=.98,s=11)
    ax.set(title="New action selections",ylabel="Mode ID",ylim=(-.5,11.5),yticks=list(range(12)))
    bar=fig.colorbar(points,ax=ax,pad=.02,fraction=.025); bar.set_label("Drop fraction q")
    ax=axes[1,1]; ax.plot(x,vals("payload_bytes")/1000,color="#286a98",lw=.85); ax.set(title="Payload and pre-action queue state",ylabel="Payload (kB)",ylim=(0,None))
    twin=ax.twinx(); twin.plot(x,vals("rlc_backlog_bytes")/1000,color="#b46541",lw=.8); twin.set(ylabel="RLC backlog (kB)",ylim=(0,max(1,float(vals("rlc_backlog_bytes").max()/1000)*1.1)))
    ax.tick_params(axis="y",labelcolor="#286a98"); twin.tick_params(axis="y",labelcolor="#b46541")
    ax=axes[2,0]; ax.plot(x,vals("latency_ms"),color="#286a98",lw=.8); ax.axhline(170,color="#b04436",ls="--",label="170-ms deadline")
    ax.set(title="Action-open to received feedback",ylabel="Latency (ms)",ylim=(0,None)); ax.legend(fontsize=9)
    ax=axes[2,1]; ax.plot(x,vals("si_p40_compute_ms"),color="#286a98",lw=.8,label="SI + P40 computation")
    ax.plot(x,vals("scene_rgb_receipt_age_ms"),color="#b46541",lw=.8,label="RGB receipt → state commit")
    ax.set(title="Scene computation and availability delay",ylabel="Elapsed time (ms)",ylim=(0,None)); ax.legend(fontsize=9)
    for ax in axes.flat: frameaxis(ax)
    finish(fig,"01_scene_network_actions_latency.png","Scene, network state, action and latency — one joined timeline",[
        "Raw policy-decision samples, no smoothing. Holds/fallbacks are not new decisions; missing values are never filled with zero.",
        "SNR is the causal external simulator proxy. One fluctuating favorable-stable profile, not a controlled bad-channel step/hold.",
        "P40 is range-weighted radar support, not a count. These associations do not establish action benefit or online adaptation."])

    fig, axes=plt.subplots(3,2,figsize=(16,13)); fig.subplots_adjust(left=.09,right=.96,bottom=.19,top=.85,hspace=.43,wspace=.25)
    for col, cls in enumerate(("vehicle","person")):
        cv=lambda k: np.asarray([np.nan if r.get(cls,{}).get(k) is None else r[cls][k] for r in decisions])
        ax=axes[0,col]; ax.plot(x,cv("xy_median_m"),lw=.8,marker=".",ms=2,color="#286a98")
        ax.set(title=f"{cls.title()}: matched-object localization",ylabel="Within-frame median XY error (m)",ylim=(0,None))
        ax=axes[1,col]; ax.plot(x,cv("precision"),color="#b46541",lw=.8,label="Precision"); ax.plot(x,cv("recall"),color="#27836d",lw=.8,label="Recall")
        ax.set(title=f"{cls.title()}: localization coverage",ylabel="Fraction",ylim=(-.02,1.02)); ax.legend(fontsize=9,ncol=2)
    ax=axes[2,0]; ax.plot([e["frame_number"] for e in events],[e["map_age_at_install_ms"] for e in events],lw=.8,color="#286a98")
    ax.set(title="Freshness of every installed frame update",ylabel="Capture → map install (ms)",ylim=(0,None))
    for ax in list(axes[:2,:].flat)+[axes[2,0]]: frameaxis(ax)
    ax=axes[2,1]; ax.plot([r["elapsed_wall_s"] for r in aoi],[r["age_ms"] for r in aoi],lw=.65,color="#27836d")
    ax.set(title="Latest-installed-frame age between updates",xlabel="Wall time since first capture (s)",ylabel="Frame AoI (ms)",ylim=(0,None),xlim=(0,None)); ax.grid(alpha=.2)
    finish(fig,"02_localization_coverage_map_freshness.png","Localization, detection coverage and map-update freshness",[
        "All evaluated policy frames, including late feedback; XY error is conditional on matched objects. Read precision/recall alongside it.",
        "Map plots include all installed decisions/holds/fallbacks. AoI uses only wall-clock capture/install times; it is NOT object/track age.",
        "Class confidence was not retained. Person segmentation and Q_perc remain provisional; person localization alone uses AVO ≥0.65."])


def run(capture_root, results, output):
    decisions, events, aoi, old_summary, provenance = load_inputs(capture_root, results)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    save_json(output/"JOINED_POLICY_ROWS.json",decisions); save_csv(output/"JOINED_POLICY_ROWS.csv",decisions)
    save_json(output/"MAP_FRAME_EVENTS.json",events); save_csv(output/"MAP_FRAME_EVENTS.csv",events)
    save_json(output/"FRAME_AOI_TIMELINE.json",aoi); save_csv(output/"FRAME_AOI_TIMELINE.csv",aoi)
    summary=dict(frames=len(events),policy_decisions=len(decisions),frame_kinds=old_summary["frame_kinds"],
        timing=old_summary["latency_ms"],timely=old_summary["timely"],missing=old_summary["missing_replies"],
        measurements={k:stats([r[k] for r in decisions]) for k in ("camera_si","radar_p40","si_p40_compute_ms",
            "scene_rgb_receipt_age_ms","snr_db","ul_mcs","rlc_backlog_bytes","mcs_age_ms","rlc_backlog_age_ms")},
        map_age_at_install_ms=stats([r["map_age_at_install_ms"] for r in events]),
        frame_AoI=dict(points=len(aoi),start_elapsed_wall_s=aoi[0]["elapsed_wall_s"],
            end_elapsed_wall_s=aoi[-1]["elapsed_wall_s"],max_ms=max(r["age_ms"] for r in aoi),
            definition="Latest installed frame source age, not object AoI; not sampled-uniformly, no point-weighted mean"),
        classification_confidence="NOT_RETAINED",quality="PROVISIONAL; unchanged raw semantic-reference limitation",
        claim_scope="SINGLE_CAPTURE_DESCRIPTIVE_NO_ADAPTATION_OR_CAUSAL_POLICY_CLAIM")
    plot_story(decisions,events,aoi,output)
    distributions=plot_statistics(decisions,events,output)
    save_json(output/"STATISTICAL_POPULATIONS.json",distributions)
    verify_inputs_unchanged(provenance["source_inputs_sha256"])
    provenance["source_integrity_after_render"]="ALL_USED_INPUT_HASHES_UNCHANGED"
    provenance["reporting_sources_sha256"]={str(Path(__file__).resolve()):sha(__file__),str(Path(canonical.__file__).resolve()):sha(canonical.__file__)}
    save_json(output/"SUMMARY.json",summary); save_json(output/"PROVENANCE.json",provenance)
    save_json(output/"OUTPUT_SHA256.json",{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.capture_root,args.results,args.output),indent=2,allow_nan=False))
