#!/usr/bin/env python3
"""Re-group PERMODEL_KNOB_MATRIX.md so related rows are adjacent: grouped by
(quant, ROI), with the 4 models (AE-128/64/32/no-AE) listed together in each group.
Presentation-friendly: keeps the accuracy + payload columns, drops interpolated latency.
Writes PERMODEL_KNOB_MATRIX_GROUPED.md (leaves the payload-sorted original intact).
"""
import sys, re
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rl_agent/PERMODEL_KNOB_MATRIX.md")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("rl_agent/PERMODEL_KNOB_MATRIX_GROUPED.md")

QORD = {"per_channel_uint4": 0, "per_channel_uint6": 1, "per_channel_uint8": 2}
QLBL = {"per_channel_uint4": "uint4", "per_channel_uint6": "uint6", "per_channel_uint8": "uint8"}
MORD = {"128": 0, "64": 1, "32": 2, "-": 3}
MLBL = {"128": "AE-128", "64": "AE-64", "32": "AE-32", "-": "no-AE"}

def model_from_profile(p):  # clean rows carry '-' in the AE column; model is in the profile name
    p = p.lower()
    for k in ("128", "64", "32"):
        if p.startswith("ae" + k):
            return k
    return "-"

def parse_rows(text):
    rows = []
    for ln in text.splitlines():
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 12 or cells[0].lower() == "profile" or set(cells[0]) <= set("-: "):
            continue
        rows.append(cells)  # 0 profile,1 quant,2 entropy,3 ROI,4 AE,5 payKB,6 pay%,7 miou,8 veh,9 ped,10 obj,11 loc,12 pedloc,...
    return rows

def main():
    rows = parse_rows(SRC.read_text())
    comp, clean, raw = [], [], []
    for r in rows:
        quant, roi, ae, pay = r[1], r[3], r[4], r[5]
        if quant in QORD:
            comp.append(r)
        elif pay.lower() in ("nan", "") or "clean" in r[0].lower():
            clean.append(r)
        else:
            raw.append(r)

    def keyfn(r):
        return (QORD.get(r[1], 9), float(r[3]) if re.match(r"[-\d.]+$", r[3]) else 9, MORD.get(r[4], 9))
    comp.sort(key=keyfn)

    hdr = ("| group (quant · ROI) | model | payload KB | payload % | mIoU | veh IoU | ped recall | "
           "obj recall | loc m | ped-loc m |")
    sep = "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"
    lines = ["# KNOB MATRIX — grouped by (quant, ROI), models adjacent",
             "",
             "Each block compares AE-128 / AE-64 / AE-32 / no-AE at the **same** quantization + ROI-drop "
             "setting, so you can read the model effect at a fixed compression level. Payload = entropy-coded "
             "(zlib) bytes; loc = localization MAE. Sorted quant(4→8) · ROI(0→0.5) · model(128→none).",
             "", hdr, sep]

    prev = None
    for r in comp:
        grp = f"{QLBL[r[1]]} · ROI {float(r[3]):.1f}"
        gcell = grp if grp != prev else ""
        if grp != prev and prev is not None:
            lines.append("| | | | | | | | | | |")  # visual blank divider between groups
        prev = grp
        lines.append(f"| {gcell} | {MLBL.get(r[4], r[4])} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | "
                     f"{r[9]} | {r[10]} | {r[11]} | {r[12]} |")

    # clean references + raw, in model order
    lines += ["", "## Clean (no compression) references", "", hdr, sep]
    for r in sorted(clean, key=lambda r: MORD.get(model_from_profile(r[0]), 9)):
        lines.append(f"| clean | {MLBL.get(model_from_profile(r[0]))} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | "
                     f"{r[9]} | {r[10]} | {r[11]} | {r[12]} |")
    if raw:
        lines += ["", "## Raw transmit (reference)", "", hdr, sep]
        for r in raw:
            lines.append(f"| raw | {r[0]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | {r[9]} | {r[10]} | {r[11]} | {r[12]} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}  ({len(comp)} compressed + {len(clean)} clean + {len(raw)} raw rows)")

if __name__ == "__main__":
    main()
