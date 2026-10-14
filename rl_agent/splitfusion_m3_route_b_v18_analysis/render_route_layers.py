"""Create-only, CPU-only route presentation overlays from a sealed M3 capture (v18 copy of v11; the schedule name and
target come from the v18 contract, i.e. OPERATING_12DB 12.0 dB).

V11 changes, all presentation-only: the commanded GOOD_ALL target is read from the registered v11 schedule instead of
the V8 literal 21.5 dB, the SNR layers are labelled by statistic, and a fourth figure shows the MEASURED calibrated
channel -- per-transmission PUSCH SNR and the scheduler's final MCS -- joined to each decision by host wall clock
from the run's own T-tracer telemetry.  Joins, features and the existing three figures are otherwise unchanged.

V8 note: V8 reuse of experiments/splitfusion_m3_route_b_v6/presentation_route_layers_02_INCOMPLETE_ROUTE/render_route_layers.py.
Only change: the route-status label, footnote wording and provenance status/scope are derived from METRICS
route.completed (V6 hard-coded INCOMPLETE ROUTE / partial run). Joins, features and rendering are unchanged.

V6 note: V6 reuse of experiments/splitfusion_m3_route_b_v1/presentation_route_layers_01_INCOMPLETE_ROUTE/render_route_layers.py.
Only change: the v1-specific literal decision count, coverage assertion and subtitle text are derived from the
analysis inputs (DECISION_TIMELINE.csv rows must equal METRICS decisions.decision_rows; coverage from METRICS).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

from rl_agent.splitfusion_fixed_action_screen_analysis_v2.report import decode_state
from rl_agent.splitfusion_m3_route_b_v1_analysis import route_geometry as RG
from rl_agent.splitfusion_m3_route_b_v18 import contract as C
from rl_agent.splitfusion_m3_route_b_v18_analysis import telemetry as TEL
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.live_state_v2 import TRAINING_SCALING


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    target_db = float(C.SCHEDULE["target_snr_db"])
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = ["01_route_channel.png", "02_route_scene_complexity.png", "03_route_payload_freshness.png",
               "04_route_measured_channel.png", "ROUTE_LAYERS_JOINED.csv", "PROVENANCE.json"]
    for name in outputs:
        if (args.out / name).exists():
            raise FileExistsError(f"Create-only output already exists: {args.out / name}")
    cells = list((args.capture / "capture/cells").iterdir())
    assert len(cells) == 1
    cell = cells[0]
    phase_path = cell / "run4_phase6/PHASE6_UE_EVIDENCE.json"
    timeline_path = args.analysis / "DECISION_TIMELINE.csv"
    metrics_path = args.analysis / "METRICS.json"
    pose_path = cell / "pose_archive/pose_evidence.jsonl"
    snr_path = cell / "run5_policy/SNR_EVIDENCE.json"
    sources = [phase_path, timeline_path, metrics_path, pose_path, snr_path, ROOT / C.ROUTE_JSON,
               ROOT / C.PROGRESS_CSV, Path(__file__).resolve(),
               ROOT / "rl_agent/splitfusion_fixed_action_screen_analysis_v2/report.py",
               ROOT / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/live_state_v2.py",
               ROOT / "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py",
               ROOT / "rl_agent/splitfusion_hybrid_sac_v1/scene_descriptors.py",
               ROOT / "rl_agent/splitfusion_m3_route_b_v1_analysis/route_geometry.py"]
    hashes_before = {str(p.resolve()): sha(p) for p in sources}
    with timeline_path.open(newline="") as f:
        timeline = list(csv.DictReader(f))
    expected_rows = json.loads(metrics_path.read_text())["decisions"]["decision_rows"]
    assert len(timeline) == expected_rows
    phase = json.loads(phase_path.read_text())["decisions"]
    by_frame = {int(d["frame_id"]): d for d in phase}
    assert len(by_frame) == len(phase)
    snr = json.loads(snr_path.read_text())["decision_snr"]
    snr_by_frame = {int(d["frame_id"]): d for d in snr}
    assert len(snr_by_frame) == len(snr)
    joined = []
    for t in timeline:
        fid = int(t["frame_id"])
        d = by_frame[fid]
        assert d["kind"] == t["kind"] == "POLICY_DECISION"
        assert len(d["features"]) == 22
        scene = decode_state(d["features"])
        causal = snr_by_frame[fid]["snr"]
        assert causal["valid"] and causal["kind"] == "SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB"
        assert float(t["snr_proxy_db"]) == causal["value_db"] == target_db
        assert int(t["prior_ul_mcs"]) == scene["prior_ul_mcs"]
        assert int(t["mode_id"]) == d["mode_id"] and int(t["q_e4"]) == d["q_e4"]
        row = {k: t[k] for k in ["frame_id", "decision_index", "kind", "capture_wall_s", "t_since_first_capture_s", "progress_index", "ego_x", "ego_y", "snr_proxy_db", "prior_ul_mcs", "mode_id", "q_e4", "map_age_at_install_ms"]}
        row.update(camera_si=scene["camera_si"], radar_p40=scene["radar_p40"],
                   application_payload_bytes=int(t["payload_l2_bytes"]),
                   snr_semantics=causal["kind"])
        joined.append(row)
    assert len({r["frame_id"] for r in joined}) == len(joined)
    measured = TEL.load(args.capture / "capture/m3_radio/ttracer")
    pusch = sorted((r["wall_ns"] / 1e9, r["pusch_snr_db"]) for r in measured["pusch"]["rows"])
    mcs_rows = sorted((r["wall_ns"] / 1e9, r["final_mcs"]) for r in measured["mcs"]["rows"])
    def preceding(series, when):
        index = np.searchsorted([t for t, _ in series], when, side="right") - 1
        return series[index][1] if index >= 0 else float("nan")
    for row in joined:
        when = float(row["capture_wall_s"])
        row["measured_pusch_snr_db"] = preceding(pusch, when)
        row["measured_final_mcs"] = preceding(mcs_rows, when)
    measured_ok = sum(1 for r in joined if np.isfinite(r["measured_pusch_snr_db"]))
    numeric = ["ego_x", "ego_y", "snr_proxy_db", "prior_ul_mcs", "camera_si", "radar_p40", "application_payload_bytes", "map_age_at_install_ms", "measured_pusch_snr_db", "measured_final_mcs"]
    values = {k: np.array([float(r[k]) for r in joined]) for k in numeric}
    assert all(np.isfinite(v).all() for k, v in values.items() if not k.startswith("measured_"))
    assert ((values["radar_p40"] >= 0) & (values["radar_p40"] <= 1)).all()
    with (args.out / "ROUTE_LAYERS_JOINED.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    route = RG.load_route(ROOT)
    planned = np.asarray(route["progress"])
    poses = RG.poses_from_archive(pose_path)
    observed = np.asarray([(x, y) for _, x, y, _ in poses])
    metrics = json.loads(metrics_path.read_text())
    coverage = metrics["route"]["ordered_progress_coverage"]
    assert coverage["points"] == 301 and 0 <= coverage["matched"] <= coverage["points"]
    n_dec = len(joined)
    status_label = "ROUTE COMPLETED" if metrics["route"].get("completed") is True else "INCOMPLETE ROUTE"
    cov_text = f"{100 * coverage['coverage']:.1f}% ordered route coverage ({coverage['matched']} / {coverage['points']} points)"
    low = np.minimum(planned.min(axis=0), observed.min(axis=0)) - 20
    high = np.maximum(planned.max(axis=0), observed.max(axis=0)) + 20
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 19,
                         "axes.titleweight": "bold", "axes.labelsize": 15, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "text.color": "#182637", "axes.labelcolor": "#344357",
                         "axes.edgecolor": "#c4cdd6", "xtick.color": "#536479", "ytick.color": "#536479"})
    legend = [Line2D([], [], color="#cbd2db", lw=5, label="Planned full route"),
              Line2D([], [], color="#566779", lw=1.7, label="Recorded ego path"),
              Line2D([], [], marker="o", color="#183d5a", markeredgecolor="white", markersize=10, ls="", label="Observed start"),
              Line2D([], [], marker="X", color="#c33836", markeredgecolor="white", markersize=11, ls="", label="Observed end")]

    def new_figure(title, subtitle):
        fig = plt.figure(figsize=(16, 9), dpi=240, facecolor="#ffffff")
        fig.text(.045, .954, title, fontsize=27, weight="bold", va="top")
        fig.text(.045, .9, subtitle, fontsize=15.5, color="#4d5f72", va="top")
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.042, .86), ncol=4,
                   frameon=False, fontsize=12.5, columnspacing=2.6)
        fig.text(.955, .954, "M3.7 · " + C.SCHEDULE["schedule_id"], fontsize=15, weight="bold", ha="right", va="top")
        axes = [fig.add_axes([x, .24, .395, .54]) for x in (.065, .555)]
        for ax in axes:
            ax.set_facecolor("#fbfcfd")
            ax.plot(*planned.T, color="#cbd2db", lw=5.5, solid_capstyle="round", zorder=1)
            ax.plot(*observed.T, color="#566779", lw=1.4, zorder=2)
            ax.scatter(*observed[0], marker="o", s=125, color="#183d5a", edgecolor="white", linewidth=1.4, zorder=5)
            ax.scatter(*observed[-1], marker="X", s=145, color="#c33836", edgecolor="white", linewidth=1.2, zorder=5)
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="CARLA world x (m)", ylabel="CARLA world y (m)")
            ax.set_aspect("equal")
            ax.grid(color="#e7ebf0", linewidth=.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        return fig, axes

    def layer(fig, ax, key, title, label, cmap, norm=None, ticks=None):
        v = values[key]
        artist = ax.scatter(values["ego_x"], values["ego_y"], c=v, s=28, cmap=cmap,
                            norm=norm, edgecolors="white", linewidths=.16, zorder=3)
        ax.set_title(title, pad=12)
        box = ax.get_position()
        cax = fig.add_axes([box.x0, .145, box.width, .02])
        bar = fig.colorbar(artist, cax=cax, orientation="horizontal", ticks=ticks)
        bar.set_label(label, fontsize=13, labelpad=7)
        bar.outline.set_visible(False)
        return {"field": key, "min": float(v.min()), "max": float(v.max()), "colormap": cmap,
                "color_min": float(artist.norm.vmin), "color_max": float(artist.norm.vmax)}

    configs = []
    fig, axes = new_figure("Channel conditions along Route B", f"{n_dec} policy decisions  •  {cov_text}  •  {status_label}")
    configs.append(layer(fig, axes[0], "snr_proxy_db", f"Commanded target (actor proxy) · constant {target_db:g} dB", "Commanded UL target SNR / actor-visible proxy (dB)", "viridis", Normalize(target_db - .5, target_db + .5), [target_db]))
    configs.append(layer(fig, axes[1], "prior_ul_mcs", "Prior observed uplink MCS", "Prior new-data UL MCS index", "plasma", ticks=sorted(set(values["prior_ul_mcs"]))))
    fig.text(.045, .044, f"{target_db:g} dB is the actor-visible simulator proxy / commanded target, not a measurement. The measured per-transmission\nPUSCH SNR is a separate signal and is shown in figure 04. Color is shown only at recorded decision positions.", fontsize=12, color="#4d5f72", linespacing=1.55)
    fig.savefig(args.out / outputs[0], dpi=240)
    plt.close(fig)

    fig, axes = new_figure("Scene descriptors along Route B", f"{n_dec} matched policy states  •  {cov_text}  •  {status_label}")
    configs.append(layer(fig, axes[0], "camera_si", "Camera spatial information (SI)", "SI (luma-code gradient units)", "viridis"))
    configs.append(layer(fig, axes[1], "radar_p40", "Radar proximity (P40)", "P40 (dimensionless; larger = nearer returns)", "magma", Normalize(0, 1), [0, .25, .5, .75, 1]))
    fig.text(.045, .044, "Camera SI: standard deviation of 3 × 3 Sobel gradient magnitude on 768 × 448 luma.  Radar P40: mean clip(1 − range / 40 m, 0, 1).\nSeparate color scales preserve the descriptors’ units. Values are decoded from recorded policy states; no route-wide interpolation.", fontsize=12, color="#4d5f72", linespacing=1.55)
    fig.savefig(args.out / outputs[1], dpi=240)
    plt.close(fig)

    values["application_payload_kb"] = values["application_payload_bytes"] / 1000
    fig, axes = new_figure("Payload and map freshness along Route B", f"{n_dec} policy decisions  •  {cov_text}  •  {status_label}")
    configs.append(layer(fig, axes[0], "application_payload_kb", "Transmitted application payload", "Compressed application payload (kB; 1 kB = 1,000 bytes)", "cividis"))
    configs.append(layer(fig, axes[1], "map_age_at_install_ms", "Map age at installation", "Capture-to-map-install age (ms)", "inferno"))
    fig.text(.045, .044, "Payload is the recorded compressed application size; wire overhead is not included.  Map age runs from frame capture to its map installation.\nThese spatial associations describe this run; they do not establish that scene or radio conditions caused an outcome.", fontsize=12, color="#4d5f72", linespacing=1.55)
    fig.savefig(args.out / outputs[2], dpi=240)
    plt.close(fig)
    fig, axes = new_figure("Measured channel along Route B", f"{n_dec} policy decisions  •  {cov_text}  •  {status_label}")
    configs.append(layer(fig, axes[0], "measured_pusch_snr_db", "Measured PUSCH SNR (per transmission)", "gNB per-transmission PUSCH SNR, snrx10/10 (dB)", "viridis"))
    configs.append(layer(fig, axes[1], "measured_final_mcs", "Scheduler final UL MCS", "Final UL MCS index", "plasma", ticks=sorted(set(values["measured_final_mcs"]))))
    fig.text(.045, .044, "Per-transmission PUSCH SNR (GNB_MAC_PUSCH_POWER_CONTROL.snrx10) is the Phase-14A calibrated channel SNR; each decision takes the\nlast transmission at or before its capture. The gNB MAC-log 'SNR' is a different, TPC-pinned power-control statistic and is not plotted.", fontsize=12, color="#4d5f72", linespacing=1.55)
    fig.savefig(args.out / outputs[3], dpi=240)
    plt.close(fig)

    hashes_after = {str(p.resolve()): sha(p) for p in sources}
    assert hashes_before == hashes_after, "Input changed during rendering"
    provenance = {
        "status": status_label.replace(" ", "_"), "scope": "CPU-only presentation overlays of an existing capture; no live run or model inference",
        "join": {"key": "frame_id", "timeline_rows": len(timeline), "joined_rows": len(joined), "missing_keys": 0,
                 "duplicate_timeline_keys": 0, "missing_plotted_values": 0, "raw_phase6_rows": len(phase),
                 "selection": f"Exact {len(joined)} POLICY_DECISION rows from existing DECISION_TIMELINE.csv; holds/fallbacks not included",
                 "checks": ["22-D feature count", "action identity", "prior MCS matches existing timeline", "SNR value and valid simulator-proxy semantics", "finite raw feature values", "P40 in [0,1]"]},
        "features": {"camera_si": {"definition": "std of 3x3 Sobel gradient magnitude on 768x448 uint8 luma", "units": "luma-code gradient units", "decode": "features[0] * camera_si_scale + camera_si_center", "scale": TRAINING_SCALING.camera_si_scale, "center": TRAINING_SCALING.camera_si_center},
                     "radar_p40": {"definition": "mean(clip(1-range_m/40,0,1)) over valid returns in current non-overlapping 100-ms sweep; >40m contributes zero and remains in denominator", "units": "dimensionless [0,1]", "decode": "features[1]"},
                     "prior_ul_mcs": {"decode": "round(features[2] * 28)", "semantics": "strictly prior new-data uplink grant MCS index"},
                     "snr_proxy_db": {"kind": "SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB", "value": target_db, "warning": "Causal simulator proxy / commanded target, not achieved gNB SNR"},
                     "measured_pusch_snr_db": {"kind": "GNB_MAC_PUSCH_POWER_CONTROL.snrx10/10", "semantics": "per-transmission calibrated channel SNR (Phase-14A); nearest preceding transmission on the host wall clock", "decisions_with_a_preceding_transmission": measured_ok, "rows": len(pusch), "not_the_mac_log_statistic": True},
                     "measured_final_mcs": {"kind": "GNB_MAC_UL_MCS_DECISION.final_mcs", "rows": len(mcs_rows)},
                     "application_payload_bytes": {"source_column": "DECISION_TIMELINE.payload_l2_bytes", "semantics": "Compressed application payload from per_frame_metrics; excludes wire overhead"}},
        "geometry": {"planned_route_points": len(planned), "measured_pose_samples": len(poses), "ordered_coverage": coverage,
                     "decision_positions": "Existing timeline ego_x/ego_y, same pose association as original 05_route_timeliness", "markers": "First and final recorded ego poses", "common_xlim": [float(low[0]), float(high[0])], "common_ylim": [float(low[1]), float(high[1])], "unobserved": "Planned route remains gray; no interpolation into unobserved route"},
        "render": {"pixels": [3840, 2160], "dpi": 240, "format": "PNG only", "layers": configs},
        "reproduce": "env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -B -m rl_agent.splitfusion_m3_route_b_v18_analysis.render_route_layers --capture <v11 root> --analysis <v11 analysis dir> --out <new dir>",
        "inputs_sha256": hashes_before, "inputs_unchanged_during_render": True,
        "outputs_sha256": {name: sha(args.out / name) for name in outputs if name != "PROVENANCE.json"}}
    with (args.out / "PROVENANCE.json").open("x") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")
    print(json.dumps({"outputs": outputs, "joined_rows": len(joined), "missing": 0, "ranges": configs}, indent=2))


if __name__ == "__main__":
    main()
