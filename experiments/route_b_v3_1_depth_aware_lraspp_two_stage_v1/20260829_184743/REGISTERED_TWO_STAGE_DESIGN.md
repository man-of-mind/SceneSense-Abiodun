# Registered Route B v3.1 two-stage LR-ASPP design

Registered: 2026-08-29T18:47:49.560536+00:00
Source commit: `a27b19f32e9b5807100d6abdeb7e927e8689d78e`
Config SHA-256: `a1907078565b70d6ebf9fdf08692c55a880f705029f80190cb25724366c96698`

This is the final LR-ASPP experiment. Stage 1 trains only the representation allowlist for exactly 20 epochs with segmentation, dense-depth and radar-consistency losses. Stage 2 is forbidden unless the earliest passing Stage-1 epoch (10 then 20) satisfies all frozen semantic and train-baseline-relative depth gates. If authorized, both private object branches are deterministically reset and trained alone for exactly 30 epochs. There is no joint fine-tuning, Stage 3, threshold sweep or LR-ASPP follow-up.

The exact parameter allowlists, seeds, optimizer schedules, baseline values, gates, evaluation checkpoints, selection order, expected artifact counts, source hashes and six exclusive terminals are recorded in `REGISTERED_TWO_STAGE_DESIGN.json`. Validation is not opened during optimization. Deployable inference accepts only RGB-radar input and never a depth label.

Constant depth is 3.220703125 m (`log1p` 1.440001731467378). Frozen train episode-macro log-MAE values are {"20_30": 1.7883755037653106, "30_40": 2.1318111606309573, "overall": 0.7163404399368064}.
