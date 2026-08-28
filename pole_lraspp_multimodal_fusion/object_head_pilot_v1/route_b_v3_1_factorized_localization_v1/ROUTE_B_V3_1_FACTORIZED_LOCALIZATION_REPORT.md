# Route B v3.1 factorized-localization report

Terminal: `LRASPP_FACTORIZED_LOCALIZATION_CONTRACT_INVALID`

Phase A reconstructed all 13,597 primary-v0.10 validation GT centres using the
stored CARLA camera frame, the recorded per-frame calibration, deterministic inverse
projection, and the existing camera-to-world transform. All reconstructions were
finite. World-XY round-trip error was 0.000003077 m median, 0.000011785 m p99, and
0.000019878 m maximum, comfortably inside the 0.01 m / 0.05 m numerical gates.

The semantic factorization gate failed. The requested `depth = exp(log_depth)` target
must be strictly positive, but 34 eligible vehicle rows have non-positive stored
camera-forward centre depth (range -1.477135 m to -0.030206 m). They comprise 26
dynamic-actor rows and 8 environment-static rows across 11 source identities. In all
34 cases a visible bounding-box corner is in front of the camera, while the physical
3D centre is behind the camera and its projection lies outside the image. Consequently,
the frozen eligible GT set cannot supply the requested log-depth/projected-centre
target for every object.

No target was dropped, reclassified, or assigned proxy depth. Phase B was not
unlocked: no localization head was implemented, no model was instantiated on GPU, no
launch batch or training ran, no checkpoints or inference payloads were created, and
epochs 4/8/12 were not evaluated. The selected checkpoint is `none`.

The native epoch-15 checkpoint remains the unchanged read-only baseline. At v0.10 its
vehicle/person F1 values are 0.7565/0.4791 and XY MAE values are 0.9847/1.3961 m. Its
vehicle FP taxonomy contains 1,705 `TWO_D_CORRECT_WORLD_WRONG` and 991 duplicate FP;
its person FN taxonomy at score 0.02 contains 854 `CENTER_PRESENT_WORLD_WRONG` and 685
`HEATMAP_CENTER_MISS`. At v0.25, vehicle/person F1 values are 0.7943/0.5023 and XY MAE
values are 0.9433/1.3947 m.

The transported split bundle remains the pre-existing `{low, high}` bundle because no
model or runtime source was changed. Canonical v3/v3.1, locked test payloads, CARLA,
OAI, q/AE, prior checkpoints/experiments, and the 288-measurement campaign were not
modified or launched.

# LRASPP_FACTORIZED_LOCALIZATION_CONTRACT_INVALID
