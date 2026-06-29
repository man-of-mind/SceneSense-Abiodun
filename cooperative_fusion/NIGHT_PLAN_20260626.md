# Autonomous overnight plan — 2026-06-26 (night)

Goal tonight: (A) push near-object detection of the SEG-fusion model toward >0.9 recall/precision
within the operating range, WITHOUT a detector swap; (B) strengthen the two-view cooperative-fusion
demo. Stay on this side of the SEG/OD phase line: no R-CNN/YOLO, no large data collection.

## Phase A — detection quality (SEG-fusion model)
- A1 [RUNNING] Gated retrain `det_rangegated40_archK` (40 m targets, gated eval). Record gated
  F1/recall/precision (and at <=30 m), compare vs: ungated baseline (0.357) and eval-only gating
  (0.465 @40m, 0.481 @30m). Per-class (vehicle/person) recall is the number that matters.
- A2  Radar-gated decoding + val-tuned threshold on the A1 model (OFFLINE, no retrain, no CARLA):
  reject decoded centers with no radar support; sweep threshold on val for best F1. Attacks PRECISION
  + lets us drop the threshold for RECALL. Record P/R near.
- A3  Detection-via-seg+radar (OFFLINE eval on test split): connected components of the model's
  vehicle/person seg mask = detections; split merged instances with radar clusters; read the
  localization regression at each CC center. Measure near-object recall/precision vs the heatmap.
  This exploits seg=0.95 and is the most likely route to >0.9 near.
- GATE G1: pick the approach with best near (<=30 m) recall+precision. RECORD a recommendation only —
  do NOT re-architect or commit to retraining strategy beyond what's measured here.

## Phase B — cooperative-fusion demo (deliverable)
- B0  DIMENSION FUSION (new — the front-view/side-view insight). The full deliverable is the fused
  world-frame 3D box = centroid (triangulation) + DIMENSIONS + yaw. A single view can't observe the
  extent along its own line of sight (front view sees W+H, not L; side view sees L+H, not W). Add
  `fuse_dimensions(views)` to `fusion.py`: per box-axis, weight each view's size prediction by how
  perpendicular that view's ray is to the axis (best-observed wins), combine -> full W x L x H.
  Validate the fused 3D box (center + size + yaw) against CARLA GT size_x/y/z (offline self-test
  first w/ synthetic two-view box, then live in the two-ego scene). Report dimension MAE: per-view
  vs fused.
- B1  Add the PEDESTRIAN to `phase2_two_view_fusion.py` (radar-cluster association for person, since
  seg-person is weak). Report XY error vs GT for car AND person, all estimators. (CARLA)
- B2  Baseline sweep: ego B at ~3 / 8 / 15 m, log triangulation error vs baseline -> validates the
  bearing-limited regime live (great slide). (CARLA)
- B3  Multi-frame averaging on the static scene to cut bearing pixel noise; re-measure. (CARLA)
- Record all in `RESULTS_phase2_two_view.md`.

## Phase C — consolidate
- Update RESULTS_* docs + memory. Write a morning summary: numbers, what each gate decided, and the
  single recommended next lever for detection + for fusion.

## Execution mechanics & guardrails
- GPU is 32 GB: training (~6 GB) + CARLA (~7 GB) run concurrently fine (verified). Tear down CARLA
  when idle (`pkill -f CarlaUnreal`; exit 144 from the trap is harmless).
- Order: A1->A2->A3 (offline, sequential), then B1->B2->B3 (CARLA). Offline A2/A3 can overlap a CARLA B-step.
- Decision gates choose the next SUB-step; I will NOT (without you): swap to R-CNN/YOLO, start a large
  data collection, or change the project phase plan. Editing convention: work only in /abiodun and
  /cooperative_fusion; never edit top-level scripts.
- Each step writes its result to disk immediately so nothing is lost if interrupted.
- I resume automatically when the A1 retrain notifies completion; then chain through A2->...->C.
