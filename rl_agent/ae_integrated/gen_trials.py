#!/usr/bin/env python3
"""Emit trial-JSONs for the 3 integrated-AE models (bottleneck 64/32/128), warm-started from M',
AE integrated end-to-end, ALL weights trainable, both seg+object losses, ROI drop-aware. entropy=zlib."""
import json, os
AB = "/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
MP = f"{AB}/experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt"
OBJH = {"heatmap_radius_px": 4, "fuse_low_feature": True, "head_arch": "shared", "use_coordconv": False,
        "head_depth": 3, "use_groundplane_prior": False, "predict_bbox2d": True,
        "adaptive_heatmap_radius": True, "max_gt_distance_m": 40}
OBJLW = {"center": 4.0, "location": 1.5, "dimensions": 0.6, "yaw": 0.3, "parked": 0.2,
         "radar_support": 0.1, "bbox2d": 1.0}
here = os.path.dirname(os.path.abspath(__file__))
for bn in (64, 32, 128):
    trial = {
        "name": f"ae{bn}_integrated",
        "optimizer": "adamw", "lr": 1.5e-4, "weight_decay": 1e-4,
        "augment_strength": "strong", "geometric_augment": False,   # geo-aug not allowed with object loss
        "input_size": [768, 432], "batch_size": 16, "num_workers": 4,
        "prefetch_factor": 2, "persistent_workers": False,
        "epochs": 40, "early_stop_patience": 12,
        "init_rgb_checkpoint": MP,            # warm-start backbone + seg from M'
        "init_object_checkpoint": MP,         # warm-start object head from M'
        "freeze_backbone": False, "freeze_classifier": False, "freeze_bn": True, "freeze_object_head": False,
        "lr_scheduler": "cosine", "lr_warmup_epochs": 3, "min_lr_ratio": 0.01, "poly_power": 0.9,
        "selection_score_mode": "loc_dim_loss",     # loc is the make-or-break; GATE re-checks seg+recall
        "class_loss_weights": [0.5, 1.0, 4.0], "lovasz_weight": 0.5,
        "loss_weights": {"object_total": 1.0, "segmentation": 1.0, "object": OBJLW},
        "object_heads": OBJH,
        "feature_drop_max": 0.8,               # ROI-robust (train q~U(0,0.8))
        "ae_bottleneck": bn, "ae_arch": "v2",
        "ae_init_checkpoint": f"{AB}/rl_agent/feature_ae/checkpoints/ae_b{bn}_v2clean.pt",  # warm-start AE
    }
    fn = os.path.join(here, f"ae{bn}_integrated.json")
    json.dump(trial, open(fn, "w"), indent=2)
    print("wrote", fn)
