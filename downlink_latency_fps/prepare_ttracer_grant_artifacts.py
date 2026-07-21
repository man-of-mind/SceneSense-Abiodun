#!/usr/bin/env python3
"""Build compact CARLA/OAI T-tracer artifacts for presentation plots.

This joins the live frontend metrics, optional UE tunnel sampler, and UE
T-tracer grant windows into the same lightweight artifact layout used by the
default OAI bottleneck plots.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ABIODUN = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--network-dir", default="")
    parser.add_argument("--ttracer-root", default=str(ABIODUN / "metrics_logs" / "scenesense_ttracer"))
    parser.add_argument("--out-root", default=str(ABIODUN / "metrics_logs" / "carla_oai_ttracer"))
    parser.add_argument("--window-s", type=float, default=1.0)
    return parser.parse_args()


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def q(series: pd.Series, percent: float) -> float:
    vals = num(series).dropna()
    if vals.empty:
        return float("nan")
    return float(np.percentile(vals, percent))


def finite_or_blank(value: float) -> float | str:
    return value if math.isfinite(value) else ""


def find_metrics_csv(run_group: str, explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = sorted((ROOT / "runs").glob(f"**/streams/{run_group}_metrics.csv"))
    if not matches:
        raise FileNotFoundError(f"could not find frontend metrics CSV for run_group={run_group}")
    return matches[-1]


def frontend_rates_1s(metrics: pd.DataFrame) -> pd.DataFrame:
    app = metrics.copy()
    app["wall_time"] = pd.to_datetime(app["wall_time_iso"], errors="coerce")
    app = app[app["wall_time"].notna()].copy()
    app["feature_payload_bytes"] = num(app["feature_payload_bytes"]).fillna(0.0)
    t0 = app["wall_time"].min()
    app["sec"] = np.floor((app["wall_time"] - t0).dt.total_seconds()).astype(int)
    rates = (
        app.groupby("sec", as_index=False)
        .agg(
            feature_payload_bytes_1s=("feature_payload_bytes", "sum"),
            frames=("frame_id", "count"),
            received=("result_received", "sum"),
        )
        .sort_values("sec")
    )
    rates["app_offered_mbps"] = rates["feature_payload_bytes_1s"] * 8.0 / 1e6
    return rates


def load_network_summary(run_group: str, network_dir: str) -> dict[str, float]:
    base = Path(network_dir).expanduser().resolve() if network_dir else ABIODUN / "metrics_logs" / "scenesense_network" / run_group
    summary_path = base / "network_summary.csv"
    if not summary_path.exists():
        return {"tunnel_tx_mbps_p50": float("nan"), "tunnel_tx_mbps_p95": float("nan")}
    summary = pd.read_csv(summary_path).iloc[0]
    if "tx_bitrate_mbps_p50" in summary:
        return {
            "tunnel_tx_mbps_p50": float(summary["tx_bitrate_mbps_p50"]),
            "tunnel_tx_mbps_p95": float(summary["tx_bitrate_mbps_p95"]),
        }
    timeseries_path = base / "network_timeseries.csv"
    if timeseries_path.exists():
        ts = pd.read_csv(timeseries_path)
        return {
            "tunnel_tx_mbps_p50": q(ts["tx_bitrate_mbps"], 50),
            "tunnel_tx_mbps_p95": q(ts["tx_bitrate_mbps"], 95),
        }
    return {
        "tunnel_tx_mbps_p50": float(summary.get("avg_tx_mbps", float("nan"))),
        "tunnel_tx_mbps_p95": float("nan"),
    }


def write_compact_grants(run_group: str, ttracer_root: Path, out_dir: Path) -> pd.DataFrame:
    windows_path = ttracer_root / run_group / "ue" / "analysis" / "nrue_grant_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(windows_path)
    windows = pd.read_csv(windows_path)
    ul = windows[windows["direction_label"].astype(str).str.lower() == "ul"].copy()
    if ul.empty:
        raise ValueError(f"no UL rows in {windows_path}")
    ul["window_start_s"] = num(ul["window_start_s"])
    ul = ul.sort_values("window_start_s")
    ul["t_norm"] = ul["window_start_s"] - float(ul["window_start_s"].min())
    columns = [
        "window_index",
        "window_start_s",
        "t_norm",
        "grants",
        "grant_rate_hz",
        "scheduled_mbps",
        "avg_rb_size",
        "p50_rb_size",
        "p95_rb_size",
        "avg_mcs",
        "p95_mcs",
        "avg_tbs_bytes",
        "p95_tbs_bytes",
        "retx_rate",
    ]
    compact = ul[[c for c in columns if c in ul.columns]].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    compact.to_csv(out_dir / "nrue_ul_grant_windows_compact.csv", index=False)
    return compact


def main() -> int:
    args = parse_args()
    run_group = args.run_group
    out_dir = Path(args.out_root).expanduser().resolve() / run_group
    ttracer_root = Path(args.ttracer_root).expanduser().resolve()

    metrics_path = find_metrics_csv(run_group, args.metrics_csv)
    metrics = pd.read_csv(metrics_path)
    received = metrics[num(metrics["result_received"]).fillna(0).astype(bool)].copy()
    app_rates = frontend_rates_1s(metrics)
    compact = write_compact_grants(run_group, ttracer_root, out_dir)
    network = load_network_summary(run_group, args.network_dir)

    if not received.empty:
        upload_ms = (num(received["t_edge_recv_wall_s"]) - num(received["t_front_send_wall_s"])) * 1000.0
    else:
        upload_ms = pd.Series(dtype=float)

    summary = {
        "run_group": run_group,
        "frames": int(len(metrics)),
        "received": int(num(metrics["result_received"]).fillna(0).sum()),
        "delivery": float(num(metrics["result_received"]).fillna(0).mean()),
        "front_ms_p50": q(metrics["front_ms"], 50),
        "rtt_recv_ms_p50": q(received["round_trip_result_recv_ms"], 50) if not received.empty else float("nan"),
        "rtt_recv_ms_p95": q(received["round_trip_result_recv_ms"], 95) if not received.empty else float("nan"),
        "back_ms_p50": q(received["back_ms"], 50) if not received.empty else float("nan"),
        "downlink_ms_p50": q(received["result_send_to_recv_ms_wall"], 50) if not received.empty else float("nan"),
        "feature_upload_payload_handling_ms_p50": q(upload_ms, 50),
        "feature_kb_p50": q(metrics["feature_payload_bytes"], 50) / 1024.0,
        "feature_chunks_p50": q(metrics["feature_payload_chunks"], 50),
        "app_offered_mbps_1s_p50": q(app_rates["app_offered_mbps"], 50),
        "app_offered_mbps_1s_p95": q(app_rates["app_offered_mbps"], 95),
        "tunnel_tx_mbps_p50": network["tunnel_tx_mbps_p50"],
        "tunnel_tx_mbps_p95": network["tunnel_tx_mbps_p95"],
        "ul_sched_mbps_p50": q(compact["scheduled_mbps"], 50),
        "ul_sched_mbps_p95": q(compact["scheduled_mbps"], 95),
        "ul_prb_p50_window": q(compact["avg_rb_size"], 50),
        "ul_prb_p95_window": q(compact["avg_rb_size"], 95),
        "ul_avg_mcs_p50_window": q(compact["avg_mcs"], 50),
        "ul_p95_mcs_p50_window": q(compact["p95_mcs"], 50),
        "ul_retx_rate_mean": float(num(compact["retx_rate"]).mean()),
        "ego_speed_mean_mps": float(num(metrics["ego_speed_mps"]).mean()),
        "moving_gt0p5_frac": float((num(metrics["ego_speed_mps"]).fillna(0.0) > 0.5).mean()),
    }
    clean_summary = {k: finite_or_blank(v) if isinstance(v, float) else v for k, v in summary.items()}
    pd.DataFrame([clean_summary]).to_csv(out_dir / "CARLA10_OAI_TTRACER_SUMMARY.csv", index=False)

    manifest = out_dir / "ARTIFACTS.md"
    manifest.write_text(
        "\n".join(
            [
                f"# T-tracer artifacts: {run_group}",
                "",
                f"- Frontend metrics: `{metrics_path}`",
                f"- Compact UL grants: `{out_dir / 'nrue_ul_grant_windows_compact.csv'}`",
                f"- Summary: `{out_dir / 'CARLA10_OAI_TTRACER_SUMMARY.csv'}`",
                "",
                "These artifacts are generated from the live CARLA frontend plus UE-side `NRUE_MAC_DCI_GRANT` traces.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote compact T-tracer artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
