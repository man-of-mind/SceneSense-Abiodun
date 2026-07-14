#!/usr/bin/env python3
"""Extract the integrated AE weights (feature_ae.*) from a training checkpoint's
'model' state_dict and write them as {'ae_state': ...} so the next phase's
ae_init_checkpoint can load them (train_fusion loads ae_init_checkpoint['ae_state']).

Usage: extract_ae.py <in_checkpoint.pt> <out_ae_state.pt>
Exit 0 on success, 2 if no feature_ae.* keys were found (AE was not attached).
"""
import sys, torch

def main():
    in_ck, out_p = sys.argv[1], sys.argv[2]
    ck = torch.load(in_ck, map_location="cpu")
    msd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    pref = "feature_ae."
    ae_state = {k[len(pref):]: v for k, v in msd.items() if k.startswith(pref)}
    if not ae_state:
        print(f"ERROR: no '{pref}*' keys in {in_ck} -- AE was not integrated?", file=sys.stderr)
        sys.exit(2)
    torch.save({"ae_state": ae_state}, out_p)
    print(f"extracted {len(ae_state)} AE tensors -> {out_p}")

if __name__ == "__main__":
    main()
