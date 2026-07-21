#!/usr/bin/env python3
"""Regenerate CODEC_LATENCY_AB.md from loopback_latency_zstd.json + loopback_latency_zlib.json.
Compares transport/front/RTT per overlapping profile. Usage: make_codec_ab.py"""
import json
from pathlib import Path

base = Path(__file__).resolve().parent
Z = json.load(open(base / "loopback_latency_zstd.json"))
L = json.load(open(base / "loopback_latency_zlib.json"))


def norm(d):
    o = {}
    for k, v in d.items():
        p = k.split("|")
        o[(p[0], p[1], p[2])] = v
    return o


Z, L = norm(Z), norm(L)
keys = sorted(set(Z) & set(L), key=lambda k: L[k]["payload_kb"])
o = ["# Codec latency A/B — zstd vs zlib (ideal 8 MB loopback, 100% delivery, ALL profiles measured)", "",
     "**Accuracy codec-invariant** (lossless). **Payload ~±5% codec-dependent** (compression ratio). Both codecs now",
     "measured on the full profile set (`sweeps_loopback_ideal_{zstd,zlib}_full`). `transport` incl. reassembly+",
     "**decompress**; `front` incl. **compress**.", "",
     "| profile | zstd KB | zlib KB | front z→z | transport zstd→zlib | ×penalty | RTT zstd→zlib |",
     "|---|--:|--:|--:|--:|--:|--:|"]
for k in keys:
    z, l = Z[k], L[k]
    q, roi, ae = k
    nm = f"{q.replace('per_channel_', '')} roi{roi}" + (f" ae{ae}" if ae != "0" else "")
    pen = l["transport_ms"] / z["transport_ms"] if z["transport_ms"] else 0
    o.append(f"| {nm} | {z['payload_kb']} | {l['payload_kb']} | {z['front_ms']}→{l['front_ms']} | "
             f"{z['transport_ms']}→{l['transport_ms']} | {pen:.1f}× | {z['rtt_ms']}→{l['rtt_ms']} |")
nk = ("per_channel_uint8", "0.0", "0")
if nk in Z and nk in L:
    z, l = Z[nk], L[nk]
    cz, cl = round(z["front_ms"] + z["rtt_ms"]), round(l["front_ms"] + l["rtt_ms"])
    o += ["", "## Headline — no-AE u8 (~1 MB) capture→result floor",
          f"- **zstd:** {cz} ms   **zlib (deployed):** {cl} ms   → zlib→zstd cuts the floor ~{100*(cl-cz)/cl:.0f}%, same accuracy."]
o += ["", "## Takeaways",
      "- Penalty grows with payload: ~1–1.5× at small AE payloads, ~4× at the ~1 MB no-AE payload.",
      "- It is **compute** (codec (de)compress), channel-independent — inflates OAI anchors identically.",
      "- zstd is a **~free** latency lever (lossless, payload ~±5%). Deployed = zlib → train on `PERMODEL_KNOB_MATRIX_ZLIB.md`."]
(base / "CODEC_LATENCY_AB.md").write_text("\n".join(o) + "\n")
print(f"[make_codec_ab] {len(keys)} overlapping profiles -> CODEC_LATENCY_AB.md")
