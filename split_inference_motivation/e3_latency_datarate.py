"""E3 (part 2) - per-hop end-to-end latency + uplink data rate for architectures A/B/C.

Combines:
  - measured compute from E1 (this study)
  - measured payloads from E3_payloads.json (this study)
  - measured OAI per-hop latency from PRIOR work (cited inline, not re-measured)

Emits results/E3_latency_datarate.json.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "results"
TARGET_FPS = 10.0

# ---------------------------------------------------------------------------
# PRIOR MEASURED OAI DATA - not measured in this study. Sources:
#  [1] abiodun/downlink_latency_fps/OAI_TRANSPORT_BOTTLENECK_DISCUSSION.md
#      "Latency and payload breakdown" table; live CARLA frontend, 10 FPS target,
#      1300 frames, corrected drivable route, no-AE per-channel-u8 unless noted.
#  [2] abiodun/oai_layer_latency/README.md, Phase-2b instrumented CARLA run
#      (918,409 matched SDUs): per-packet UE PDCP-ingress -> gNB PDCP-deliver.
#  [3] abiodun/rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md (ideal loopback compute anchors).
# ---------------------------------------------------------------------------
OAI_ROWS = {
    "C_noae_106PRB_zstd": {  # [1] row: "Corrected drivable OAI, 106PRB, zstd"
        "source": "[1] Corrected drivable OAI, 106PRB, zstd",
        "uplink_payload_KB": 1055.2, "chunks": 19, "delivery_pct": 83.6,
        "front_ms": 25.2, "uplink_ms": 151.1, "edge_ms": 6.9, "downlink_ms": 3.0,
        "rtt_p50_ms": 162.2, "rtt_p95_ms": 175.3, "capture_to_result_ms": 188.0,
        "downlink_payload_KB": 2.2,
    },
    "C_ae128_u6_roi05_106PRB_zstd": {  # [1] row: AE-128 uint6 ROI0.5
        "source": "[1] Corrected default OAI, 106PRB, AE-128 uint6 ROI0.5, zstd",
        "uplink_payload_KB": 152.7, "chunks": 3, "delivery_pct": 99.8,
        "front_ms": 21.9, "uplink_ms": 52.6, "edge_ms": 7.5, "downlink_ms": 3.2,
        "rtt_p50_ms": 64.2, "rtt_p95_ms": 76.8, "capture_to_result_ms": 86.5,
        "downlink_payload_KB": 2.3,
    },
    "C_ideal_loopback_zstd": {  # [1] row: ideal loopback (no radio)
        "source": "[1] Corrected ideal loopback, zstd (NO radio - upper bound)",
        "uplink_payload_KB": 1053.9, "chunks": 18, "delivery_pct": 100.0,
        "front_ms": 26.7, "uplink_ms": 7.6, "edge_ms": 8.3, "downlink_ms": 1.6,
        "rtt_p50_ms": 18.3, "rtt_p95_ms": 39.2, "capture_to_result_ms": 46.1,
        "downlink_payload_KB": 2.4,
    },
}

# [2] Same RAN, two traffic regimes - the natural experiment that bounds A vs C.
OAI_PER_SDU = {
    "smooth_small_packets_iperf": {
        "source": "[2] Phase-2 iperf-validated: UE PDCP-ingress -> gNB PDCP-deliver",
        "mean_ms": 4.6, "p50_ms": 3.1, "p95_ms": 8.4, "max_ms": 83.6,
        "note": "smooth small-packet traffic - the regime architecture A operates in",
    },
    "carla_1MB_feature_bursts": {
        "source": "[2] Phase-2b instrumented CARLA run, 918409 matched SDUs",
        "mean_ms": 105.2, "p50_ms": 112.0, "p95_ms": 163.0, "p99_ms": 173.0, "max_ms": 225.0,
        "note": "1 MB feature bursts - the regime architecture C operates in; "
                "dominated by UE RLC queue-wait at UL MCS ~4 (~10.9 Mbps drain)",
    },
}
OAI_UL_CAPACITY_MBPS = 10.9  # [2] measured drain at MCS 4 under bursty CARLA load
K2_GRANT_MS = 3.0            # [2] fixed DCI->PUSCH grant delay, both regimes


def mbps(kb_per_frame, fps=TARGET_FPS):
    return kb_per_frame * fps * 8.0 / 1024.0


def main():
    pay = json.loads((OUT / "E3_payloads.json").read_text())
    e1 = json.loads((OUT / "E1_raw.json").read_text())

    def p50(v):
        s = sorted(v)
        return s[len(s) // 2]

    # On-car compute from E1. Use the 8-thread CPU figure as the vehicle-plausible
    # operating point (GPU p50 also reported); flagged as host-CPU, not a vehicle SoC.
    cpu8 = {k: p50(v) for k, v in e1["cpu_thread_sweep_ms"]["8"].items()}
    gpu = {k: p50(v) for k, v in e1["gpu_lat_all_ms"].items()}

    A_kb = pay["A_det_json_raw"]["mean_KB"]
    B_kb = pay["B_jpeg92_plus_radar"]["mean_KB"]
    B_kb_q75 = pay["B_jpeg75"]["mean_KB"] + pay["B_radar_npy_zstd"]["mean_KB"]
    C_kb = pay["C_feat_u8_zstd"]["mean_KB"]
    C_ae_kb = OAI_ROWS["C_ae128_u6_roi05_106PRB_zstd"]["uplink_payload_KB"]

    arch = {
        "A_full_local": {
            "name": "A. Full-local + share detections (late fusion)",
            "car_compute_ms_cpu8": round(cpu8["FULL"], 2),
            "car_compute_ms_gpu": round(gpu["FULL"], 3),
            "uplink_payload_KB": A_kb,
            "uplink_Mbps_at_10fps": round(mbps(A_kb), 3),
            "uplink_latency_ms": OAI_PER_SDU["smooth_small_packets_iperf"]["mean_ms"],
            "uplink_latency_basis": "measured, small-packet regime [2]",
            "edge_compute_ms": 0.0,
            "edge_note": "no edge inference; edge/peer only fuses detection lists",
            "downlink_ms": OAI_PER_SDU["smooth_small_packets_iperf"]["mean_ms"],
            "fits_UL_capacity": mbps(A_kb) < OAI_UL_CAPACITY_MBPS,
        },
        "B_full_offload": {
            "name": "B. Full-offload raw (send RGB+radar, whole model on edge)",
            "car_compute_ms_cpu8": 0.0,
            "car_compute_ms_gpu": 0.0,
            "car_note": "encode-only on car (JPEG); no inference",
            "uplink_payload_KB": round(B_kb, 2),
            "uplink_payload_KB_q75": round(B_kb_q75, 2),
            "uplink_Mbps_at_10fps": round(mbps(B_kb), 2),
            "uplink_Mbps_at_10fps_q75": round(mbps(B_kb_q75), 2),
            "uplink_latency_ms": None,
            "uplink_latency_basis": "NOT MEASURED - no OAI run for this payload; see md",
            "edge_compute_ms": round(gpu["FULL"], 3),
            "downlink_ms": OAI_PER_SDU["smooth_small_packets_iperf"]["mean_ms"],
            "fits_UL_capacity": mbps(B_kb) < OAI_UL_CAPACITY_MBPS,
            "fits_UL_capacity_q75": mbps(B_kb_q75) < OAI_UL_CAPACITY_MBPS,
        },
        "C_split_noae": {
            "name": "C. Split + feature fusion, no-AE u8+zstd (OURS, as first deployed)",
            "car_compute_ms_cpu8": round(cpu8["FRONT"], 2),
            "car_compute_ms_gpu": round(gpu["FRONT"], 3),
            "uplink_payload_KB": round(C_kb, 2),
            "uplink_Mbps_at_10fps": round(mbps(C_kb), 2),
            "uplink_latency_ms": OAI_ROWS["C_noae_106PRB_zstd"]["uplink_ms"],
            "uplink_latency_basis": "measured live over OAI [1]",
            "edge_compute_ms": OAI_ROWS["C_noae_106PRB_zstd"]["edge_ms"],
            "downlink_ms": OAI_ROWS["C_noae_106PRB_zstd"]["downlink_ms"],
            "rtt_p50_ms": OAI_ROWS["C_noae_106PRB_zstd"]["rtt_p50_ms"],
            "capture_to_result_ms": OAI_ROWS["C_noae_106PRB_zstd"]["capture_to_result_ms"],
            "delivery_pct": OAI_ROWS["C_noae_106PRB_zstd"]["delivery_pct"],
            "fits_UL_capacity": mbps(C_kb) < OAI_UL_CAPACITY_MBPS,
        },
        "C_split_ae128": {
            "name": "C'. Split + feature fusion, AE-128 uint6 ROI0.5 (OURS, compressed)",
            "car_compute_ms_cpu8": round(cpu8["FRONT"], 2),
            "car_compute_ms_gpu": round(gpu["FRONT"], 3),
            "uplink_payload_KB": C_ae_kb,
            "uplink_Mbps_at_10fps": round(mbps(C_ae_kb), 2),
            "uplink_latency_ms": OAI_ROWS["C_ae128_u6_roi05_106PRB_zstd"]["uplink_ms"],
            "uplink_latency_basis": "measured live over OAI [1]",
            "edge_compute_ms": OAI_ROWS["C_ae128_u6_roi05_106PRB_zstd"]["edge_ms"],
            "downlink_ms": OAI_ROWS["C_ae128_u6_roi05_106PRB_zstd"]["downlink_ms"],
            "rtt_p50_ms": OAI_ROWS["C_ae128_u6_roi05_106PRB_zstd"]["rtt_p50_ms"],
            "capture_to_result_ms": OAI_ROWS["C_ae128_u6_roi05_106PRB_zstd"]["capture_to_result_ms"],
            "delivery_pct": OAI_ROWS["C_ae128_u6_roi05_106PRB_zstd"]["delivery_pct"],
            "fits_UL_capacity": mbps(C_ae_kb) < OAI_UL_CAPACITY_MBPS,
        },
    }

    out = {
        "target_fps": TARGET_FPS,
        "oai_ul_capacity_Mbps_measured": OAI_UL_CAPACITY_MBPS,
        "k2_grant_ms": K2_GRANT_MS,
        "architectures": arch,
        "oai_rows_cited": OAI_ROWS,
        "oai_per_sdu_cited": OAI_PER_SDU,
        "payload_source": "E3_payloads.json (measured this study, 25 real test frames)",
        "compute_source": "E1_raw.json (measured this study)",
    }
    (OUT / "E3_latency_datarate.json").write_text(json.dumps(out, indent=2))

    print("== uplink data rate @ 10 FPS (measured payloads) ==")
    print(f"  measured OAI UL capacity under bursty load: {OAI_UL_CAPACITY_MBPS} Mbps (MCS ~4)\n")
    for k, a in arch.items():
        fit = a.get("fits_UL_capacity")
        print(f"  {a['name'][:52]:54s} {a['uplink_payload_KB']:9.2f} KB  "
              f"{a['uplink_Mbps_at_10fps']:8.2f} Mbps  {'FITS' if fit else 'EXCEEDS CAPACITY'}")
    print("\n== per-hop latency ==")
    for k, a in arch.items():
        ul = a["uplink_latency_ms"]
        ul_s = f"{ul:7.1f}" if ul is not None else "    n/a"
        print(f"  {k:16s} car {a['car_compute_ms_cpu8']:7.2f} | UL {ul_s} | "
              f"edge {a['edge_compute_ms']:6.2f} | DL {a['downlink_ms']:5.1f} ms")
    print(f"\nwrote {OUT/'E3_latency_datarate.json'}")


if __name__ == "__main__":
    main()
