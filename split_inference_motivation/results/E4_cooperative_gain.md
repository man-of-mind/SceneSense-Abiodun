# E4 — Cooperative gain: localization + map coverage (single ego vs two egos)

**Date:** 2026-07-27 · **Raw:** `E4_raw.json` · **Figure:** `E4_coverage_localization.png`
**Scripts:** `../e4_cooperative_gain.py`, `../e4_plot.py`

Run under the **reframed** E4 (PLAN.md, 2026-07-22): localization error + map coverage, **not** detection recall.

## What is real and what is synthesized (read this first)

| Element | Status |
|---|---|
| Object world positions, dimensions, class | **Real CARLA GT** (`object_boxes.csv`, test split) |
| Ego A camera pose, yaw, FOV (120°) | **Real** (manifest, test split) |
| Scene composition / object density | **Real** — 300 frames, 1096 in-range objects (≤40 m gate) |
| **Ego B camera pose** | **SYNTHESIZED** at a controlled offset from ego A (0.7·baseline lateral + 0.3·baseline forward, yawed at the scene centroid) |
| Occlusion | **Geometric model**: objects as ground-plane discs of radius ½·max(size_x, size_y); line-of-sight blocked if a nearer disc intersects the ray |
| Sensor noise | **Model**: bearing σ=0.3°, monocular depth σ swept 1.2 / 2.5 / 3.5 m |

**No new live CARLA run was performed** — CARLA was not running and the plan explicitly permits scoping to a
controlled scene. The live two-ego measurements this is checked against are prior work
(`cooperative_fusion/RESULTS_phase2_two_view.md`).

## Part 1 — Map coverage (the clean cooperative win)

Fraction of in-range GT objects with unoccluded line-of-sight, 1096 objects over 300 real frames:

| baseline | ego A alone | ego B alone | **cooperative (either)** | seen by both (triangulable) | seen by neither |
|---|---|---|---|---|---|
| 4 m | 73.9 % | 72.9 % | **81.7 %** | 65.2 % | 18.3 % |
| 8 m | 73.9 % | 66.7 % | **82.6 %** | 58.0 % | 17.4 % |
| 14 m | 73.9 % | 66.6 % | **85.0 %** | 55.6 % | 15.0 % |
| 20 m | 73.9 % | 65.6 % | **87.3 %** | 52.2 % | 12.7 % |

> **A second ego raises map coverage from 73.9 % to 81.7–87.3 %** — recovering **7.8 to 13.4 percentage points**
> of objects that a single ego physically cannot see. At a 20 m baseline it cuts the completely-unseen fraction
> from 26.1 % to 12.7 %, i.e. **roughly halves the blind spot**.

This is the part a single local ego *cannot* fix at any compute budget: the objects are occluded or out of FOV,
so no better model on ego A recovers them. It requires another viewpoint.

**The baseline trade-off is real and runs in both directions:** widening the baseline improves *coverage*
(more of the scene is seen by someone) while reducing *overlap* (fewer objects seen by both, 65.2 % → 52.2 %).
Since triangulation needs both views, coverage and triangulation quality pull against each other — a deployed
system would want peers at a **moderate baseline (~8–14 m)**, not maximally spread.

## Part 2 — Localization error

Objects visible to both egos; monocular depth noise swept because it is the dominant single-view error term.
Triangulation is bearing-only, so it is **independent of depth noise** — the same column repeats across blocks.

| depth σ | baseline | single ego A | single ego B | two-view mean | **triangulation** | gain vs best single |
|---|---|---|---|---|---|---|
| 1.2 m | 4 m | 1.016 | 0.955 | 0.719 | 2.222 | **0.43× (worse)** |
| 1.2 m | 8 m | 1.010 | 0.966 | 0.733 | 1.059 | 0.91× (worse) |
| 1.2 m | 14 m | 1.020 | 0.961 | 0.734 | **0.546** | 1.76× |
| 1.2 m | 20 m | 1.028 | 0.962 | 0.742 | **0.363** | 2.65× |
| 2.5 m | 8 m | 2.075 | 1.983 | 1.510 | **1.059** | 1.87× |
| 2.5 m | 20 m | 2.113 | 1.979 | 1.533 | **0.363** | 5.46× |
| **3.5 m** | 4 m | 2.915 | 2.736 | 2.064 | 2.222 | 1.23× |
| **3.5 m** | **8 m** | 2.897 | 2.768 | 2.109 | **1.059** | **2.61×** |
| **3.5 m** | 14 m | 2.924 | 2.759 | 2.117 | **0.546** | 5.05× |
| **3.5 m** | 20 m | 2.951 | 2.765 | 2.143 | **0.363** | 7.62× |

The σ=3.5 m block is the deployment-relevant one: it is calibrated to the **measured** live monocular error of
3.56 m.

### Independent agreement with the live measurement

| | this controlled study (σ=3.5 m, 8 m baseline) | prior live 2-ego run (8.6 m baseline) |
|---|---|---|
| single-view monocular | 2.90 / 2.77 m | 3.56 / 5.35 m |
| **triangulation** | **1.06 m** | **1.40 m** |
| **gain vs best single view** | **2.61×** | **2.54×** |

Two independent routes to the cooperative gain — a geometric study over 300 real scenes, and a live two-ego
CARLA capture — land at **2.6× vs 2.5×**, with triangulation absolute error in the same 1–1.4 m band. Neither
was tuned to match the other.

### The ill-conditioning result reproduces too

Triangulation is **worse than a single view at a 4 m baseline** (2.22 m vs 0.96 m at σ=1.2). Near-parallel
bearing rays make the least-squares intersection ill-conditioned. The prior live run found exactly this
("4 m is geometrically ill-conditioned"), and it reappears here from independent geometry. **Cooperative fusion
is not unconditionally better — it needs adequate baseline**, and a deployed system must gate on it.

Note also that at short baselines the naive **two-view mean beats triangulation** (0.72 m vs 2.22 m at 4 m).
A real system should switch estimator on baseline, not always triangulate.

## What E4 does and does not establish

**Establishes:**
- Two viewpoints recover **7.8–13.4 pp of map coverage** a single ego cannot obtain at any compute budget, and
  roughly halve the unseen fraction. This is the cleanest cooperative-perception result here.
- Two-view triangulation cuts localization error **2.6–7.6×** vs monocular at realistic noise and adequate
  baseline, reproducing the live measurement independently.
- Both gains require sharing perception with a peer/edge — which is what motivates offloading.

**Does NOT establish (per the plan's honest-scope note — do not claim these):**
- **That intermediate feature fusion beats late detection fusion.** Triangulation is *geometric 2-view fusion of
  bearings*, which both architecture A (share detections) and architecture C (share features) could in principle
  feed. E4 justifies **cooperation**, not the **feature-level split point**. No side-by-side
  share-features vs share-detections experiment was run, so that claim stays unmade.
- Anything about detection recall — the head is a known dead-end and was deliberately not used as a metric.

**Where the split-point defence actually rests:** compute/throughput (E1/E2), privacy (E5), and the
architectural argument that the edge can run heavier fusion than the car can. E4 supplies the *cooperative*
premise those build on.

## Caveats
- Ego B's pose is synthesized; a real peer would be constrained to the road graph and would not always achieve
  the ideal offset. Coverage numbers are therefore a plausible-geometry estimate, not a deployment measurement.
- The disc-based occlusion model ignores object height and building/terrain occlusion; buildings would *lower*
  single-ego coverage and so likely **understate** the cooperative gain.
- Association is assumed solved (as in the prior live work, which used oracle association). Real cross-ego data
  association is the open problem flagged in `spatial_map_coop/README.md` stage 4.
