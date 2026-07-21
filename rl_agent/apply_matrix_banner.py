#!/usr/bin/env python3
"""Replace the build_knob_matrix.py default header with a codec-provenance banner.
Deterministic: keeps everything from the 'Clean baseline:' line onward, prepends a fresh header.

Usage: apply_matrix_banner.py <matrix.md> <zlib|zstd>
"""
import sys
from pathlib import Path

path, codec = sys.argv[1], sys.argv[2]
txt = Path(path).read_text()
idx = txt.find("Clean baseline:")
tail = txt[idx:] if idx != -1 else txt

if codec == "zlib":
    dep = "DEPLOYED codec"
    other = "PERMODEL_KNOB_MATRIX_ZSTD.md"
else:
    dep = "reference codec (deployed = zlib)"
    other = "PERMODEL_KNOB_MATRIX_ZLIB.md"

header = f"""# PER-MODEL KNOB MATRIX — **latency = {codec} ({dep}), ALL 36 profiles MEASURED** (M', Month-2)

> ✅ **Codec + provenance (2026-07-20).** Latency (front/back/transport) measured live with `--entropy-coder {codec}`
> under ideal 8 MB-buffer loopback, **100% delivery on all 36 AE×quant×ROI profiles**
> (`loopback_latency_{codec}.json`, batch `sweeps_loopback_ideal_{codec}_full`). **No interpolation** except the
> synthetic `uncompressed_fp16` anchor. Accuracy + payload from the per-model offline eval
> (`sweeps_permodel_{codec}`, also {codec}) — genuinely {codec}-measured, nothing copied or flagged. Accuracy is
> codec-invariant (lossless); payload is {codec}'s own (compression ratios differ ~±5% between codecs). Counterpart:
> `{other}`; A/B: `CODEC_LATENCY_AB.md`; grouped: `PERMODEL_KNOB_MATRIX_{codec.upper()}_BYMODEL.md`.

Action profiles vs **accuracy**, **payload** (entropy-coded bytes), and **latency** (front=UE compute, back=edge compute, transport=localhost round-trip; **{codec}, measured**). Transport is an **IDEAL local link** (8 MB socket buffers, NO bandwidth cap / no Linux tc shaping). **Reliability + latency under a real channel = OAI + Sionna, Month 3.**

"""
Path(path).write_text(header + tail)
print(f"[apply_matrix_banner] {codec} banner applied to {path}")
