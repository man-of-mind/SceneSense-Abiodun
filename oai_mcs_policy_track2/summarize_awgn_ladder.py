#!/usr/bin/env python3
"""Summarize 106PRB AWGN ladder policy runs.

This script is deliberately profile/policy aware so we do not accidentally
compare a 273PRB diagnostic run or a clear-channel run against the AWGN ladder.

Example:

  python3 abiodun/oai_mcs_policy_track2/summarize_awgn_ladder.py \
    --base-batch track2_awgn_ladder_20260801_120000 \
    --profiles "mild medium strong" \
    --policies "vanilla hold aimd_cap"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "oai_mcs_policy_track2" / "results"

PROFILE_NOISE_DB = {
    "mild": -10,
    "medium": -5,
    "strong": -4,
    "harsh": 0,
    "edge": 5,
}


def q(series: pd.Series, p: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(p))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def find_metrics_csv(run_group: str) -> Path:
    runs_root = ROOT / "downlink_latency_fps" / "runs"
    candidates = sorted(runs_root.glob(f"*/fps_10_*/streams/{run_group}_metrics.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"metrics CSV not found for run_group={run_group}")


def grant_summary(run_group: str) -> pd.Series:
    path = (
        ROOT
        / "metrics_logs"
        / "scenesense_ttracer"
        / run_group
        / "ue"
        / "analysis"
        / "nrue_grant_summary.csv"
    )
    df = safe_read_csv(path)
    if df.empty or "direction_label" not in df:
        return pd.Series(dtype=float)
    ul = df[df["direction_label"].eq("ul")]
    return ul.iloc[0] if not ul.empty else pd.Series(dtype=float)


def bler_decision_summary(run_group: str) -> Dict[str, float]:
    path = (
        ROOT
        / "metrics_logs"
        / "scenesense_ttracer"
        / run_group
        / "gnb"
        / "csv"
        / "GNB_MAC_BLER_MCS_DECISION.csv"
    )
    df = safe_read_csv(path)
    if df.empty:
        return {}

    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ul = df[df.get("direction").eq(1)] if "direction" in df else df
    updated = ul[ul.get("updated").eq(1)] if "updated" in ul else ul
    if updated.empty:
        return {}

    num_sched = pd.to_numeric(updated.get("num_sched"), errors="coerce")
    num_retx = pd.to_numeric(updated.get("num_retx"), errors="coerce")
    bler_window_pct = pd.to_numeric(updated.get("bler_window_ppm"), errors="coerce") / 10000.0
    bler_after_pct = pd.to_numeric(updated.get("bler_after_ppm"), errors="coerce") / 10000.0
    upper_pct = pd.to_numeric(updated.get("upper_ppm"), errors="coerce") / 10000.0
    old_mcs = pd.to_numeric(updated.get("old_mcs"), errors="coerce")
    new_mcs = pd.to_numeric(updated.get("new_mcs"), errors="coerce")
    branch = pd.to_numeric(updated.get("branch"), errors="coerce")

    valid_upper = upper_pct.dropna()
    upper_value = float(valid_upper.iloc[0]) if not valid_upper.empty else 15.0
    above_upper = bler_after_pct > upper_value
    retx_window = num_retx.fillna(0) > 0
    total_sched = float(num_sched.clip(lower=0).sum())
    total_retx = float(num_retx.clip(lower=0).sum())

    out = {
        "bler_windows": float(len(updated)),
        "num_sched_p50": q(num_sched, 0.50),
        "num_sched_p95": q(num_sched, 0.95),
        "window_bler_p50_pct": q(bler_window_pct, 0.50),
        "window_bler_p95_pct": q(bler_window_pct, 0.95),
        "window_bler_max_pct": float(bler_window_pct.max()) if not bler_window_pct.dropna().empty else float("nan"),
        "filtered_bler_p50_pct": q(bler_after_pct, 0.50),
        "filtered_bler_p95_pct": q(bler_after_pct, 0.95),
        "filtered_bler_max_pct": float(bler_after_pct.max()) if not bler_after_pct.dropna().empty else float("nan"),
        "filtered_above_upper_pct": 100.0 * float(above_upper.mean()) if len(above_upper) else float("nan"),
        "bad_retx_window_pct": 100.0 * float(retx_window.mean()) if len(retx_window) else float("nan"),
        "sum_window_retx_pct": 100.0 * total_retx / total_sched if total_sched > 0 else float("nan"),
        "branch1_inc": float((branch == 1).sum()),
        "branch2_dec": float((branch == 2).sum()),
        "branch3_sparse": float((branch == 3).sum()),
        "branch4_hold": float((branch == 4).sum()),
        "branch3_pct": 100.0 * float((branch == 3).mean()) if len(branch) else float("nan"),
    }
    dec = updated[branch == 2]
    if not dec.empty:
        out.update(
            {
                "branch2_delta_med": q(
                    pd.to_numeric(dec["new_mcs"], errors="coerce")
                    - pd.to_numeric(dec["old_mcs"], errors="coerce"),
                    0.50,
                ),
                "branch2_old_mcs_med": q(pd.to_numeric(dec["old_mcs"], errors="coerce"), 0.50),
                "branch2_new_mcs_med": q(pd.to_numeric(dec["new_mcs"], errors="coerce"), 0.50),
            }
        )
    return out


def snr_summary(run_group: str) -> Dict[str, float]:
    path = (
        ROOT
        / "metrics_logs"
        / "scenesense_ttracer"
        / run_group
        / "gnb"
        / "csv"
        / "GNB_MAC_UL_MCS_DECISION.csv"
    )
    df = safe_read_csv(path)
    if df.empty or "avg_snr_x10" not in df:
        return {}
    snr_db = pd.to_numeric(df["avg_snr_x10"], errors="coerce") / 10.0
    return {
        "snr_p50_db": q(snr_db, 0.50),
        "snr_p05_db": q(snr_db, 0.05),
        "snr_p95_db": q(snr_db, 0.95),
    }


def layer_latency_summary(run_group: str) -> Dict[str, float]:
    path = (
        ROOT
        / "metrics_logs"
        / "scenesense_ttracer"
        / run_group
        / "layer_latency"
        / "uplink_layer_latency.csv"
    )
    df = safe_read_csv(path)
    if df.empty:
        return {}
    out: Dict[str, float] = {}
    for name in ["rlc_queue_wait_ms", "pdcp_to_gnb_pdcp_ms"]:
        if name in df:
            out[f"{name}_p50"] = q(df[name], 0.50)
            out[f"{name}_p95"] = q(df[name], 0.95)
    return out


def summarize_run(profile: str, policy: str, run_group: str) -> Dict[str, object]:
    metrics_path = find_metrics_csv(run_group)
    df = pd.read_csv(metrics_path)
    received = pd.to_numeric(df.get("result_received"), errors="coerce").fillna(0.0)
    row: Dict[str, object] = {
        "profile": profile,
        "policy": policy,
        "run_group": run_group,
        "noise_power_dB": PROFILE_NOISE_DB.get(profile, float("nan")),
        "frames": int(len(df)),
        "returned": int(received.sum()),
        "delivery_pct": 100.0 * float(received.mean()) if len(df) else float("nan"),
        "payload_p50_kib": q(df["feature_payload_bytes"], 0.50) / 1024.0
        if "feature_payload_bytes" in df
        else float("nan"),
    }

    if {
        "capture_to_backbone_input_ms",
        "front_backbone_ms",
        "feature_serialize_ms",
        "capture_to_front_send_ms",
        "round_trip_result_recv_ms",
        "t_edge_recv_wall_s",
        "t_front_send_wall_s",
    }.issubset(df.columns):
        feature_build = (
            pd.to_numeric(df["capture_to_backbone_input_ms"], errors="coerce")
            + pd.to_numeric(df["front_backbone_ms"], errors="coerce")
            + pd.to_numeric(df["feature_serialize_ms"], errors="coerce")
        )
        uplink = (
            pd.to_numeric(df["t_edge_recv_wall_s"], errors="coerce")
            - pd.to_numeric(df["t_front_send_wall_s"], errors="coerce")
        ) * 1000.0
        uplink = uplink.where(uplink >= 0.0)
        capture_to_result = (
            pd.to_numeric(df["capture_to_front_send_ms"], errors="coerce")
            + pd.to_numeric(df["round_trip_result_recv_ms"], errors="coerce")
        )
        row.update(
            {
                "front_build_p50_ms": q(feature_build, 0.50),
                "uplink_p50_ms": q(uplink, 0.50),
                "uplink_p95_ms": q(uplink, 0.95),
                "capture_result_p50_ms": q(capture_to_result, 0.50),
                "capture_result_p95_ms": q(capture_to_result, 0.95),
            }
        )

    row.update(
        {
            "edge_tail_p50_ms": q(df["back_ms"], 0.50) if "back_ms" in df else float("nan"),
            "downlink_p50_ms": q(df["result_send_to_recv_ms_perf"], 0.50)
            if "result_send_to_recv_ms_perf" in df
            else float("nan"),
        }
    )

    ul = grant_summary(run_group)
    row.update(
        {
            "ul_sched_mbps": float(ul.get("scheduled_mbps", float("nan"))),
            "mcs_avg": float(ul.get("avg_mcs", float("nan"))),
            "mcs_p50": float(ul.get("p50_mcs", float("nan"))),
            "mcs_p95": float(ul.get("p95_mcs", float("nan"))),
            "prb_p50": float(ul.get("p50_rb_size", float("nan"))),
            "retx_rate_pct": 100.0 * float(ul.get("retx_rate", float("nan"))),
            "avg_tbs_bytes": float(ul.get("avg_tbs_bytes", float("nan"))),
            "p95_tbs_bytes": float(ul.get("p95_tbs_bytes", float("nan"))),
        }
    )
    row.update(bler_decision_summary(run_group))
    row.update(snr_summary(run_group))
    row.update(layer_latency_summary(run_group))
    row["hypothesis_read"] = hypothesis_read(row)
    return row


def hypothesis_read(row: Dict[str, object]) -> str:
    profile = str(row.get("profile", ""))
    policy = str(row.get("policy", ""))
    filtered_above = float(row.get("filtered_above_upper_pct", float("nan")))
    filtered_p95 = float(row.get("filtered_bler_p95_pct", float("nan")))
    mcs_p50 = float(row.get("mcs_p50", float("nan")))
    delivery = float(row.get("delivery_pct", float("nan")))

    if math.isfinite(delivery) and delivery < 50.0:
        return "boundary/failure regime; do not use for fair policy ranking"
    if not math.isfinite(filtered_above) or not math.isfinite(mcs_p50):
        return "missing BLER/MCS evidence"
    if profile != "mild" and policy in {"aimd", "aimd_cap"} and filtered_above < 5.0 and filtered_p95 < 18.0:
        return "AIMD keeps filtered BLER mostly below threshold; compare latency/retransmission tradeoff"
    if filtered_above < 15.0 and filtered_p95 < 18.0:
        return "BLER not persistent; channel may be too mild for decisive bad-channel test"
    if mcs_p50 >= 20.0:
        return "BLER appears sustained but MCS remains high; weak/slow backoff evidence"
    return "MCS lowers under sustained BLER; compare latency/retransmission tradeoff"


def parse_run(value: str) -> Tuple[str, str, str]:
    # PROFILE:POLICY=RUN_GROUP
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("use PROFILE:POLICY=RUN_GROUP")
    left, run_group = value.split("=", 1)
    profile, policy = left.split(":", 1)
    return profile.strip(), policy.strip(), run_group.strip()


def constructed_runs(base_batch: str, profiles: Iterable[str], policies: Iterable[str]) -> List[Tuple[str, str, str]]:
    runs = []
    for profile in profiles:
        for policy in policies:
            condition = f"oai_default106_awgn_{profile}_track2_{policy}"
            batch = f"{base_batch}_{profile}_{policy}"
            run_group = f"downlink_{condition}_fps10_{batch}"
            runs.append((profile, policy, run_group))
    return runs


def format_float(v: object) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return ""
    return f"{f:.3f}"


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        vals = [format_float(row[col]) if pd.api.types.is_numeric_dtype(df[col]) else str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-batch", help="Base batch ID used by run_awgn_106prb_ladder.sh")
    parser.add_argument("--profiles", default="mild medium strong")
    parser.add_argument("--policies", default="vanilla hold aimd_cap")
    parser.add_argument("--run", action="append", type=parse_run, help="Manual PROFILE:POLICY=RUN_GROUP entry")
    parser.add_argument("--out-prefix", default="")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    run_specs: List[Tuple[str, str, str]] = []
    if args.base_batch:
        run_specs.extend(constructed_runs(args.base_batch, args.profiles.split(), args.policies.split()))
    if args.run:
        run_specs.extend(args.run)
    if not run_specs:
        raise SystemExit("provide --base-batch or at least one --run PROFILE:POLICY=RUN_GROUP")

    rows = []
    missing = []
    for profile, policy, run_group in run_specs:
        try:
            rows.append(summarize_run(profile, policy, run_group))
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue

    if missing and not args.allow_missing:
        msg = [
            "Missing expected AWGN ladder run artifacts; refusing to create a complete summary.",
            "",
            "Missing:",
        ]
        msg.extend(f"- {item}" for item in missing)
        msg.extend(
            [
                "",
                "If the ladder intentionally stopped early or you want a partial summary, rerun with:",
                "  --allow-missing",
            ]
        )
        raise SystemExit("\n".join(msg))

    if not rows:
        raise SystemExit("no runs could be summarized")

    df = pd.DataFrame(rows)
    preferred_cols = [
        "profile",
        "policy",
        "noise_power_dB",
        "frames",
        "returned",
        "delivery_pct",
        "payload_p50_kib",
        "snr_p50_db",
        "mcs_avg",
        "mcs_p50",
        "mcs_p95",
        "ul_sched_mbps",
        "retx_rate_pct",
        "filtered_bler_p50_pct",
        "filtered_bler_p95_pct",
        "filtered_above_upper_pct",
        "window_bler_p95_pct",
        "bad_retx_window_pct",
        "branch1_inc",
        "branch2_dec",
        "branch3_sparse",
        "branch4_hold",
        "branch2_delta_med",
        "uplink_p50_ms",
        "uplink_p95_ms",
        "capture_result_p50_ms",
        "capture_result_p95_ms",
        "rlc_queue_wait_ms_p50",
        "pdcp_to_gnb_pdcp_ms_p50",
        "hypothesis_read",
        "run_group",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.out_prefix or (f"awgn106_ladder_{args.base_batch}" if args.base_batch else "awgn106_ladder_manual")
    csv_path = OUT_DIR / f"{suffix}.csv"
    md_path = OUT_DIR / f"{suffix}.md"
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# 106PRB AWGN ladder policy summary\n\n")
        if args.base_batch:
            f.write(f"Base batch: `{args.base_batch}`\n\n")
        f.write(to_markdown(df))
        f.write("\n")
        if missing:
            f.write("\n## Missing runs\n\n")
            for item in missing:
                f.write(f"- {item}\n")

    print(csv_path)
    print(md_path)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
