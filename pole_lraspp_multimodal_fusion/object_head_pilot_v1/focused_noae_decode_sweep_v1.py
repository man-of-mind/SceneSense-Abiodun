#!/usr/bin/env python3
"""Decode focused_noae_v1 checkpoints at registered q anchors, create-only.

Wraps evaluate_route_b_checkpoint_v1.py (fixed decoder: score 0.20, top-k 120,
NMS 2 px, 3.0 m match, 40 m GT eligibility) and adds the anchor q via the
evaluator's --feature-drop-fraction passthrough. q=0.00 is a structural no-op,
so the clean pass is the unmodified production decode path.

Existing eval tags are skipped, never overwritten. Split is always 'val';
the locked test split is never touched.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The six registered training anchors.
ANCHORS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.98]
# Interval midpoints: q values the hybrid sampler covers only through its continuous
# 40% branch, never as an exact draw. Scoring these on the shortlist is what shows
# whether the model interpolates between anchors or only memorises them.
MIDPOINTS = [0.15, 0.40, 0.60, 0.80, 0.94]



def tag_for(epoch: int, q: float) -> str:
    return f"focused_ep{epoch:03d}_q{int(round(q * 100)):03d}"


def is_anchor(q: float) -> bool:
    return any(abs(q - a) < 1e-9 for a in ANCHORS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-dir", required=True, type=Path)
    ap.add_argument("--epochs", required=True,
                    help="comma list, or 'A-B' inclusive range")
    ap.add_argument("--anchors", default="0.0",
                    help="comma list of q values; 'all' = the six registered anchors; "
                         "'full' = the six anchors plus the five interval midpoints")
    ap.add_argument("--config", default=str(HERE / "configs" / "route_b_noae_precision_pilot_v1.yaml"))
    ap.add_argument("--trial", default="focused_noae_v1")
    args = ap.parse_args()

    if "-" in args.epochs and "," not in args.epochs:
        a, b = args.epochs.split("-")
        epochs = list(range(int(a), int(b) + 1))
    else:
        epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    if args.anchors == "all":
        anchors = list(ANCHORS)
    elif args.anchors == "full":
        anchors = sorted(ANCHORS + MIDPOINTS)
    else:
        anchors = [float(x) for x in args.anchors.split(",")]

    exp = args.experiment_dir.resolve()
    ck_dir = exp / "checkpoints" / args.trial
    done, skipped, failed = 0, 0, []
    for ep in epochs:
        ckpt = ck_dir / f"epoch_{ep:03d}.pt"
        if not ckpt.is_file():
            print(f"missing checkpoint, skipping: {ckpt}", flush=True)
            continue
        for q in anchors:
            tag = tag_for(ep, q)
            if (exp / "eval" / tag).exists():
                skipped += 1
                continue
            t0 = time.monotonic()
            cmd = [
                sys.executable, str(HERE / "evaluate_route_b_checkpoint_v1.py"),
                "--experiment-dir", str(exp), "--checkpoint", str(ckpt),
                "--tag", tag, "--config", args.config, "--split", "val",
            ]
            env_extra = ["--feature-drop-fraction", str(q)]
            r = subprocess.run(cmd + env_extra, capture_output=True, text=True)
            dt = time.monotonic() - t0
            if r.returncode != 0:
                failed.append((tag, r.returncode, r.stderr[-600:]))
                print(f"FAIL {tag} rc={r.returncode} ({dt:.0f}s)\n{r.stderr[-600:]}", flush=True)
            else:
                done += 1
                print(f"ok   {tag}  {dt:.0f}s", flush=True)
    print(f"\ndecoded={done} skipped={skipped} failed={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
