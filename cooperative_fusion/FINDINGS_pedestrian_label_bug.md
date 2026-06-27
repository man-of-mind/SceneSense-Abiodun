# Pedestrian SEG label bug (found 2026-06-26, Phase 0b sanity)

## Summary
The "person" segmentation class in all moving-ego data is **road-lines + ground, with zero
real pedestrians**. Two independent bugs:

1. **Mapping bug (fixed):** `PERSON_TAGS` was `{4,12,13,24,25}` — wrongly included
   24=RoadLine, 25=Ground, 4=Wall. Corrected to `{12,13}` in
   `pole_lraspp_multimodal_fusion/common.py`.

2. **CARLA 0.10 does not render pedestrian semantic labels (build limitation).**
   - A walker actor *declares* `semantic_tags = [12]` (correct).
   - But BOTH the `sensor.camera.semantic_segmentation` AND
     `sensor.camera.instance_segmentation` cameras render the walker's pixels as the
     background behind it (e.g. sidewalk tag 2). Zero tag-12 pixels are ever produced.
   - Confirmed two ways: live in-view walker test, and 0 tag-12 px across 300 training frames.
   - Vehicles are unaffected (they stencil correctly, tags 14-19).

## Evidence
- Training data (300 frames): tag-12=0, tag-13=0, tag-24(RoadLine)=7.3M px,
  tag-25(Ground)=5.4M px. "person" class = 100% roadline+ground.
- Live: walker.semantic_tags=[12]; semantic & instance cameras both render the walker
  region as tag 2 (sidewalk), no tag 12 anywhere.

## Impact
- **Every prior "person IoU" is invalid** — it measured roadline+ground segmentation, not
  people. (Explains road-lines being masked at live deployment, and chronically poor
  pedestrian detection alongside "good" person SEG.)
- **Vehicle SEG/detection/boxes remain valid** (vehicle tags render correctly).
- **Pedestrian detection/localization used real actor boxes** (object_boxes.csv) -> that GT
  is real; only pedestrian SEG was contaminated.

## Options for pedestrian masks going forward
A. **Rasterize pedestrian masks from actor 3D boxes** (project walker bbox -> image). Gives a
   usable person class, but box-shaped (not pixel-accurate silhouette) -> capped IoU.
B. **Refocus pedestrians on detection + world position** (real actor GT, already works) and
   treat pixel-accurate pedestrian SEG as out-of-scope in this CARLA build. Vehicle SEG stays.
C. Investigate a CARLA build/setting where walkers stencil correctly (uncertain; 300-frame
   evidence says the current build does not).

## Done
- Fixed PERSON_TAGS -> {12,13} (stops lane-line/ground contamination going forward).
