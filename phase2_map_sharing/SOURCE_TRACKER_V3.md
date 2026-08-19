# Source-local causal tracker v3

Status: **implemented for bounded offline comparison; not integrated into the
capture runtime and not yet an accepted operating point**.

The immutable corpus retains `source_local_nearest_cv.v1` provenance. The v3
implementation is a sibling module, `source_tracker_v3.py`, so replaying v3
cannot rewrite or silently relabel capture-time tracks.

## Fixed candidate behavior

`SourceLocalCausalTrackerV3.update(frame_id, timestamp_s, detections)` accepts
only the current detection set. Its internal state contains observations from
earlier calls only. Frame IDs must increase strictly and timestamps cannot move
backwards.

The default candidate applies these operations in order:

1. Discard detections without finite world `x/y` coordinates.
2. Within each exact class, suppress detections no farther than **0.75 m** in
   world space, retaining highest score and then lowest input index on a tie.
3. Associate greedily by class-consistent distance to the causal constant-
   velocity prediction, within the existing **5 m** association gate. Also
   require observed displacement to be no more than
   `class_max_speed * dt + 0.75 m`.
4. Bound the finite-difference planar speed, bound vertical speed to **8 m/s**,
   and apply causal exponential smoothing with `alpha = 0.5`.
5. Publish a new track only after **two consecutive hits**. A tentative track
   dies on its first miss. Once confirmed, a track retains the existing
   **three missed-frame** grace.

Default planar speed limits are 12 m/s for person/pedestrian/walker, 25 m/s for
cyclist/bicycle, 60 m/s for vehicle/car/truck/bus, and 40 m/s for an unknown
object. Every value is constructor-visible; none is inferred from ground truth.

Outputs keep the v1 track fields and add confirmation state and whether the
latest velocity was limited. Association diagnostics distinguish tentative
birth/match/death, confirmation, confirmed match/miss/death, and duplicate
suppression.

## Claim boundary and risks

- This is a causal engineering candidate, not a calibrated or safety-certified
  tracker. No ground truth, actor ID, future frame, or future trajectory enters
  an update.
- Confirmation delays earliest publication by one 10 Hz frame and can suppress
  a genuinely intermittent hazard. Target recall and warning lead must remain
  blocking checks in the bounded comparison.
- A 0.75 m same-class radius can merge two genuinely distinct close actors,
  especially pedestrians in a crowd. Report suppression counts by class and
  inspect close-pair cases before adopting it.
- The speed limits and 0.75 m localization slack are plausibility guards, not
  learned uncertainty. They can split a badly localized but real track.
- Greedy nearest association remains order-stable but is not a global assignment
  or covariance-aware tracker. V3 does not by itself solve equal-time
  multi-source fusion or warning hysteresis.

The next authorized use is a small, create-only offline v1-versus-v3 comparison
on the accepted audit batch. Freeze this one candidate before comparing
false-warning burden, fragmentation, registered-target recall, and warning
lead; do not tune it through the 72-point warning surface.
