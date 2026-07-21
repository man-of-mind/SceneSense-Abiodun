#!/usr/bin/env python3
"""Regroup a PERMODEL_KNOB_MATRIX_*.md into (quant . ROI) blocks with the 4 AE models adjacent,
keeping ALL columns. Lets you compare AE-128/64/32/no-AE at a fixed compression level at a glance.

Usage: make_bymodel_grouped.py <src_matrix.md> <out.md>
"""
import sys
from pathlib import Path

src, out = sys.argv[1], sys.argv[2]
rows = []
for ln in Path(src).read_text().splitlines():
    if not ln.startswith("| ") or "profile" in ln or "---" in ln:
        continue
    c = [x.strip() for x in ln.strip().strip("|").split("|")]
    if len(c) < 17:
        continue
    prof = c[0]
    if any(t in prof for t in ("clean", "fp16", "uncompressed")):
        continue
    quant, ent, roi, ae, paykb, paypct, miou, veh, pedr, objr, loc, pedloc, front, back, tr, acc = c[1:17]
    model = f"AE-{ae}" if ae not in ("-", "") else "no-AE"
    rows.append(dict(quant=quant.replace("per_channel_", ""), ent=ent, roi=roi, ae=ae, model=model,
                     paykb=paykb, paypct=paypct, miou=miou, veh=veh, pedr=pedr, objr=objr, loc=loc,
                     pedloc=pedloc, front=front, back=back, tr=tr, acc=acc))
qo = {"uint4": 0, "uint6": 1, "uint8": 2}
mo = {"AE-128": 0, "AE-64": 1, "AE-32": 2, "no-AE": 3}
rows.sort(key=lambda r: (qo.get(r["quant"], 9), float(r["roi"]), mo.get(r["model"], 9)))
codec = rows[0]["ent"] if rows else "?"
L = [f"# PER-MODEL KNOB MATRIX ({codec}) — grouped by (quant · ROI), 4 AE models per block",
     "",
     f"> Latency = **{codec}** (deployed codec if {codec}=zlib), ideal 8 MB loopback, 100% delivery, **all 36 measured**",
     "> (no interpolation). Each block fixes quant + ROI-drop and lists AE-128 / AE-64 / AE-32 / no-AE so the",
     "> **AE-bottleneck** effect at a fixed compression level reads at a glance (payload, accuracy, latency). All",
     f"> columns from the source matrix. Accuracy codec-invariant; payload/latency are {codec}.",
     "",
     "| quant · ROI | model | entropy | payload KB | pay % | mIoU | veh IoU | ped rec | obj rec | loc m | ped-loc m | front ms | back ms | transport ms | accept |",
     "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|"]
last = None
for r in rows:
    g = f"{r['quant']} · ROI {r['roi']}"
    lead = g if g != last else ""
    if g != last and last is not None:
        L.append("| | | | | | | | | | | | | | | |")
    last = g
    L.append(f"| {lead} | {r['model']} | {r['ent']} | {r['paykb']} | {r['paypct']} | {r['miou']} | {r['veh']} | "
             f"{r['pedr']} | {r['objr']} | {r['loc']} | {r['pedloc']} | {r['front']} | {r['back']} | {r['tr']} | {r['acc']} |")
Path(out).write_text("\n".join(L) + "\n")
print(f"[make_bymodel_grouped] {len(rows)} rows ({codec}) -> {out}")
