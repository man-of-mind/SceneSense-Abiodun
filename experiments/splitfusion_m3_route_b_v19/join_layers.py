"""V17 route-layer join reused for spatial targets; rendering uses the existing presentation Deck.
Only changes: no constant-target assertion; emit the existing joined CSV/provenance without duplicate plots.
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
from rl_agent.splitfusion_m3_route_b_v19 import contract as C
from rl_agent.splitfusion_m3_route_b_v17_analysis import telemetry as TEL
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
    C.approved_design(ROOT)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = ["ROUTE_LAYERS_JOINED.csv", "PROVENANCE.json"]
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
        assert float(t["snr_proxy_db"]) == causal["value_db"]
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

    # Preserve exact reader joins and final-input identities; no auxiliary gallery.
    for stream in measured.values():
        sources.append(Path(stream["file"]))
    hashes_after = {str(p.resolve()): sha(p) for p in sources}
    assert all(hashes_after[k] == v for k, v in hashes_before.items())
    provenance = dict(status="JOINED", join=dict(key="frame_id", rows=len(joined),
        measured_preceding_transmission_coverage=measured_ok), inputs_sha256=hashes_after,
        outputs_sha256={"ROUTE_LAYERS_JOINED.csv":sha(args.out/"ROUTE_LAYERS_JOINED.csv")},
        scope="Existing V17 joins, with dynamic commanded targets; no new inference/plot framework")
    with (args.out/"PROVENANCE.json").open("x") as f:
        json.dump(provenance,f,indent=1,sort_keys=True)
    print(json.dumps(dict(joined_rows=len(joined),measured_coverage=measured_ok)))


if __name__ == "__main__":
    main()
