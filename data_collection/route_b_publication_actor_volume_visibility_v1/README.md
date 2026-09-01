# route_b_publication_actor_volume_visibility_v1

Geometry-derived pedestrian visible-box ratio, and the bounded feasibility audit
of it against the 100 human-annotated validation panels.

## Why

The deployed `route_b_depth_visibility_interval_v1` occupancy compares each ROI
depth pixel against a *global* per-actor near/far depth interval, so road and
ground pixels at a compatible range can be counted as actor support. This package
replaces that test with a per-actor **oriented-bounding-box containment** test:
a depth pixel supports an actor only when the 3D point it back-projects to lies
inside that specific actor's oriented volume and sits above the box floor.

CARLA pedestrian semantic and instance masks are deliberately **not** used as
inputs — this build does not reliably render pedestrian silhouettes in those
sensors.

## Modules

| File | Role |
|---|---|
| `core.py` | I/O-free geometry: oriented box, projection, back-projection, actor-local containment, ground rejection, competing-pedestrian assignment, band edges |
| `scoring.py` | `visibility = area(B_visible) / area(B_full_clipped)` for one actor-frame, plus the required diagnostics |
| `agreement.py` | Confusion matrix, exact agreement, linear weighted Cohen's kappa, Spearman, balanced accuracy |
| `run_audit.py` | Create-only audit driver: qualification gates, scoring, agreement, decision, provenance |
| `training_reference.py` | Training-only expected-clear-support reference: view-angle/height binning, 95th-percentile estimator, fallback hierarchy |
| `build_training_reference.py` | Part 1 driver: builds and hashes the immutable training reference (never opens the human pilot directory) |
| `run_normalized_audit.py` | Part 2 driver: rescores the 100 panels with the normalised denominator against the frozen reference |
| `contact_sheet.py` | Per-sample overlay panels, the tiled contact sheet, and the disagreement sheet |
| `tests/` | Synthetic-array checks; no dataset, model, CARLA or CUDA involvement |

## Locked constants

`0.05 m` containment tolerance · `0.03 m` ground-rejection margin above the box
bottom · bands `[0, 0.20)` not-observable, `[0.20, 0.65)` heavy,
`[0.65, 0.90)` partial, `[0.90, 1.00]` bare · decision bars kappa >= 0.60 and
balanced accuracy >= 0.80.

`B_visible` is the tight pixel-extent box of the retained points **intersected
with** `B_full_clipped`, so the ratio is a genuine sub-area fraction; see
`scoring.score_actor_frame` for why the outward ROI rasterisation makes this
necessary.

## Run

```
CUDA_VISIBLE_DEVICES="" python3 -m \
  data_collection.route_b_publication_actor_volume_visibility_v1.run_audit
CUDA_VISIBLE_DEVICES="" python3 -m unittest discover \
  -s data_collection/route_b_publication_actor_volume_visibility_v1/tests -t .
```

The driver refuses to start unless `CUDA_VISIBLE_DEVICES` is empty and `torch`
is unimported, and the run directory is create-only.

## Follow-up: training-normalized denominator

`training_reference.py` + `build_training_reference.py` + `run_normalized_audit.py`
replace the projected-cuboid denominator with the expected unoccluded surface
support of comparable training pedestrians:

```
support_density        = retained_actor_volume_pixels / clipped_projected_box_area
normalized_visibility  = clamp(support_density / expected_clear_support_density, 0, 1)
```

`expected_clear_support_density` is the 95th percentile (`method="higher"`) of
support density over a training-only group conditioned on actor type, folded
relative view angle and projected box height, with a fixed count-based fallback
hierarchy (50 / 100 / 100). The actor-volume extraction is byte-identical to the
pilot; only the statistic built on it changes.

## Result

`20260901_191239` — 18/18 qualification checks pass; weighted kappa 0.4581
(bar 0.60), balanced accuracy 0.8523 (bar 0.80), medians monotonic, and better
than the old depth-only metric on both kappa (0.2043) and balanced accuracy
(0.5568). Terminal
`ACTOR_VOLUME_VISIBILITY_PILOT_NOT_FEASIBLE_RETAIN_HUMAN_BANDS`. See
`data_collection/experiments/route_b_publication_actor_volume_visibility_v1/20260901_191239/AUDIT_REPORT.md`.

`normalized/20260901_213040` — 22/22 qualification checks pass and the training
reference is leakage-free, but weighted kappa is 0.5175 (bar 0.60) and balanced
accuracy 0.7955 (bar 0.80, and below the unnormalized 0.8523). Normalisation
improves every ordinal statistic (kappa 0.4581 -> 0.5175, Spearman
0.7169 -> 0.7953) yet loses the >= 0.65 decision, because pixel-count density
conflates external occlusion with silhouette sparsity. Terminal
`TRAIN_NORMALIZED_ACTOR_VOLUME_VISIBILITY_NOT_FEASIBLE_RETAIN_HUMAN_BANDS`;
the conditional independent-audit package was not generated. See
`NORMALIZED_AUDIT_REPORT_20260901_213040.md`.
