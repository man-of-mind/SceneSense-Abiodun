#!/usr/bin/env python3
"""Aggregate the loopback latency/reliability sweep (CARLA transport) into a table + JSON.

Reads experiments/.../sweeps_loopback/<variant>/**/*_metrics.csv (fusion loopback client output)
and computes, per quant x entropy profile: payload KB, front (UE) latency, transport RTT, and
delivery rate (fraction of frames that returned a result). Emits:
  - LOOPBACK_LATENCY.md  (human table + the payload->latency/reliability curve)
  - loopback_latency.json (keyed by "quant|entropy" for the knob-matrix join)

Usage: agg_loopback.py <sweeps_loopback_dir> <out_md> <out_json>
"""
import csv, glob, json, math, re, sys, statistics as st
from pathlib import Path

QMAP = {"u8": "per_channel_uint8", "u6": "per_channel_uint6", "u4": "per_channel_uint4",
        "ptensor": "per_tensor_uint8"}

def parse_variant(name):
    """Extract the full action profile (quant, entropy, roi, ae) from the variant dir name, e.g.
    'q_u8_zstd', 'roi0.3_u8_zstd', 'ae_b64_u8_zstd'. Keyed on (quant,roi,ae) downstream so ROI/AE
    variants (which share quant|entropy) don't collide."""
    quant = next((full for tag, full in QMAP.items() if f"_{tag}_" in name or name.endswith(f"_{tag}")), "")
    entropy = next((e for e in ("zlib", "zstd", "none") if name.endswith(e) or f"_{e}" in name), "")
    mroi = re.search(r"roi([0-9.]+)", name)
    roi = float(mroi.group(1)) if mroi else 0.0
    mae = re.search(r"ae_b([0-9]+)", name)
    ae = int(mae.group(1)) if mae else 0
    return quant, entropy, roi, ae

def col(rows, key):
    """Finite numeric values only. float('nan') PARSES ok, so a 'nan'/'' cell (= no result / timeout)
    must be excluded explicitly, else timed-out frames get counted as delivered."""
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out

def main():
    sweeps, out_md, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    rows_out, jmap = [], {}
    for vdir in sorted(Path(sweeps).glob("*")):
        if not vdir.is_dir():
            continue
        csvs = glob.glob(str(vdir / "**" / "*_metrics.csv"), recursive=True)
        if not csvs:
            continue
        R = list(csv.DictReader(open(csvs[0])))
        if not R:
            continue
        quant, entropy, roi, ae = parse_variant(vdir.name)
        pay = col(R, "feature_payload_bytes")
        front = col(R, "front_ms")
        back = col(R, "back_ms")
        transport = col(R, "transport_round_trip_ms_estimate")
        rtt = col(R, "round_trip_ms")
        n_total = len(R)
        n_result = len(rtt)  # rows with a numeric round_trip = a result came back
        rec = {
            "variant": vdir.name, "quant": quant, "entropy": entropy, "roi": roi, "ae": ae,
            "payload_kb": round(st.mean(pay) / 1024, 1) if pay else None,
            "front_ms": round(st.mean(front), 1) if front else None,
            "back_ms": round(st.mean(back), 1) if back else None,
            "transport_ms": round(st.mean(transport), 1) if transport else None,
            "rtt_ms": round(st.mean(rtt), 1) if rtt else None,
            "delivery_rate": round(n_result / n_total, 3) if n_total else None,
            "frames": n_total,
        }
        rows_out.append(rec)
        if quant:
            jmap[f"{quant}|{roi}|{ae}"] = rec  # profile key: quant x ROI x AE (entropy is a minor latency knob)
    rows_out.sort(key=lambda r: (r["payload_kb"] is None, r["payload_kb"] or 0))
    L = ["# Loopback latency sweep (M') — IDEAL transport",
         "",
         "Split-inference latency per profile under an **ideal local transport** (8 MB socket buffers via "
         "net.core.rmem_max/wmem_max; NO bandwidth cap / no Linux tc shaping). Columns: front (UE compute), "
         "back (edge compute), transport (localhost round-trip), RTT (total). delivery ~1.0 by design here — "
         "REAL reliability under bandwidth/RF-loss is the OAI + Sionna phase (Month 3).",
         "",
         "| profile | quant | entropy | payload KB | front ms | back ms | transport ms | RTT ms | delivery | frames |",
         "|---|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows_out:
        L.append(f"| {r['variant']} | {r['quant']} | {r['entropy']} | {r['payload_kb']} | "
                 f"{r['front_ms']} | {r['back_ms']} | {r['transport_ms']} | {r['rtt_ms']} | "
                 f"{r['delivery_rate']} | {r['frames']} |")
    Path(out_md).write_text("\n".join(L) + "\n")
    Path(out_json).write_text(json.dumps(jmap, indent=1))
    print(f"[agg_loopback] {len(rows_out)} profiles -> {out_md}")

if __name__ == "__main__":
    main()
