#!/usr/bin/env python3
"""Generate the AE-from-phase-1 (from-scratch) trial JSONs for a given bottleneck.

Faithful clones of the ORIGINAL M' build stages (stage1 seg, stage2 obj) + the joint
phase-3 recipe, with the ONLY change being the AE integrated (ae_bottleneck>0) from
stage 1 onward. This is the clean ablation: AE present from phase 1 vs phase 3.

The AE is carried across the three phases via an extracted ae_state file
(extract_ae.py) pointed to by each later phase's ae_init_checkpoint. Phase-1 AE
starts random (no ae_init_checkpoint) so it is learned together with the backbone
from the very first epoch.
"""
import json, sys, os
from pathlib import Path

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
MP = AB / "rl_agent/m_prime"
AEI = AB / "rl_agent/ae_integrated"

def load(p): return json.load(open(p))

def main():
    bn = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    parent = Path(sys.argv[2]) if len(sys.argv) > 2 else AB / "experiments/ae_integrated_fs_20260713"
    base = parent / f"ae{bn}"
    s1_dir, s2_dir, p3_dir = base / "stage1", base / "stage2", base / "phase3"
    s1_ck = s1_dir / "checkpoints" / f"fs_stage1_ae{bn}" / "best.pt"
    s2_ck = s2_dir / "checkpoints" / f"fs_stage2_ae{bn}" / "best.pt"
    s1_ae = s1_dir / "ae_extracted.pt"   # written by extract_ae.py after stage1
    s2_ae = s2_dir / "ae_extracted.pt"   # written by extract_ae.py after stage2

    # --- stage 1: seg, object head frozen oracle, AE integrated (random init) ---
    t1 = load(MP / "stage1_seg_drop.json")
    t1["name"] = f"fs_stage1_ae{bn}"
    t1["ae_bottleneck"] = bn
    t1["ae_arch"] = "v2"
    t1.pop("ae_init_checkpoint", None)   # phase-1 AE learned from scratch

    # --- stage 2: obj head, backbone+seg frozen, AE carried from stage1 ---
    t2 = load(MP / "stage2_obj_drop.json")
    t2["name"] = f"fs_stage2_ae{bn}"
    t2["ae_bottleneck"] = bn
    t2["ae_arch"] = "v2"
    t2["init_rgb_checkpoint"] = str(s1_ck)          # backbone+seg from stage1 (AE keys ignored on load)
    t2["ae_init_checkpoint"] = str(s1_ae)           # AE weights carried from stage1

    # --- phase 3: joint, all unfrozen, AE carried from stage2 ---
    t3 = load(AEI / "mprime_joint_noae.json")       # the joint recipe (all trainable, seg+obj 1:1)
    t3["name"] = f"fs_phase3_ae{bn}"
    t3["ae_bottleneck"] = bn
    t3["ae_arch"] = "v2"
    t3["init_rgb_checkpoint"] = str(s2_ck)
    t3["init_object_checkpoint"] = str(s2_ck)
    t3["ae_init_checkpoint"] = str(s2_ae)

    for name, t in [(f"fs_stage1_ae{bn}", t1), (f"fs_stage2_ae{bn}", t2), (f"fs_phase3_ae{bn}", t3)]:
        out = AEI / f"{name}.json"
        json.dump(t, open(out, "w"), indent=2)
        print(f"wrote {out}")

if __name__ == "__main__":
    main()
