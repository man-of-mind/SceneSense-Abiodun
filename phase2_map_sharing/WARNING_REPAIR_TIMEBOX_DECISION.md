# Phase-2 warning repair time-box decision

Status: **offline repair implemented; current development fixture remains
FAIL-HOLD; no collection or OAI authorized**.

This note closes the bounded repair started after the three-trajectory warning
surface failed its human gate. It does not reopen parameter tuning and does not
replace the binding calibration/test design.

## What was wrong

The actor-origin truth stream is incomplete for Town10HD. In the designed
benign trajectory it contains the controlled occluder and the other ego, while
the retained RGB frames contain many parked taxis, cars, and vans baked into
the map. Eight high-frequency warning tracks project onto those visible static
vehicles. They account for 56 of 67 unmatched warning rows in the retained
overlap. Examples include a parked taxi assigned `(10.7, -15.1) m/s` and two
simultaneous tracks on the same parked car.

This is not merely a labeling error. A generous static-position proxy still
leaves most covered warnings physically clear of the recipient over the next
five seconds. Static truth would relabel them as matched non-hazards, so the
warnings remain nuisance. A separate naturalistic one-frame `person` detection
on a vehicle also persists through TTL. The root problem is the combination of
incomplete static truth, fragmented tracks, and fictitious finite-difference
motion; detector retraining is not justified by this evidence.

## Bounded v3 repair

The capture-time v1 tracker and v2 map engine remain unchanged for provenance.
The versioned offline candidate adds:

- two-hit causal confirmation before publication;
- same-class 0.75 m world-space duplicate suppression;
- displacement plausibility, bounded velocity, and causal velocity smoothing;
- no republication of missed source observations;
- source-separated, deterministic equal-time quality/moment fusion; and
- explicit lifecycle and fusion diagnostics.

The single preregistered smoke is
`c20_a30_t05_u00`: confidence 0.20, association 3 m, TTL 0.5 s, and uncertainty
multiplier 0. It is the already least-nuisance v2 setting, not a newly selected
point. The create-only result is
`data_collection/experiments/phase2_warning_repair_screen_v3/20260819_010500_screen`.

| Arm | v2 benign false-active | v3 benign false-active | Change |
|---|---:|---:|---:|
| ego-only | 18.57% | 10.00% | -8.57 pp |
| send-everything | 45.71% | 25.71% | -20.00 pp |
| hazard-only | 55.71% | 42.86% | -12.86 pp |

Naturalistic nuisance also falls to 1.43%, 2.86%, and 8.57%, respectively.
The registered pedestrian is not missed by any arm. However, both cooperative
arms lose the apparent 2.6 s lead and tie ego-only. Replaying the v3 source
tracks through the old v2 engine produces the same zero-lead result, so the
loss is not caused by the new fusion rule. The earlier lead depended on noisy
one-step motion estimates and is not a benefit to preserve.

The v3 screen therefore passes technically but remains **scientific
FAIL-HOLD**: cooperative nuisance exceeds both the 10% absolute and +2 pp
non-inferiority gates, and causal cooperative lead is zero. The seven-second
eligible exposure is still too short for the one-episode/minute gate.

## Minimum viable scenario contract for the agent environment

Do not turn tracking into a new research project. Freeze the v3 candidate and
make the next pilot test whether the environment contains a genuine causal
decision opportunity. Use exactly three scenario roles:

1. **Designed positive opportunity.** A helper observes the moving pedestrian
   for at least five consecutive 10 Hz frames while the recipient remains
   occluded and actively approaching. The pedestrian begins hazard-directed
   motion during this helper-visible interval, and the registered future
   trajectory enters the recipient safety envelope within the five-second
   horizon. The helper-only observation margin must precede the recipient's
   first confirmed track by at least 1.0 s; the causal
   send-everything/hazard-only truth-positive warning must precede ego-only by
   at least 0.5 s.
2. **Matched benign negative.** Identical route, traffic, rendering, static
   context, and seeds, with the registered hazard removed. It prevents a
   curated positive from flattering the controller and retains the 10% absolute
   nuisance and +2 pp cooperative non-inferiority gates.
3. **Naturalistic operation.** Mixed unforced traffic and no registered target.
   It remains the honest denominator and reports warning burden, action use,
   freshness, and bytes without being discarded when the designed suite is
   favourable.

The positive opportunity is a designed stratum, not the claimed deployment
distribution. Scenario groups and their train/calibration/validation/test split
must be registered before controller work. Geometry, target timing, and helper
visibility vary across groups; the controller never observes the scenario
label or future truth.

Before any new full collection, run only one positive/benign pair plus one
naturalistic trajectory. It must prove all of the following from causal logs:

- static and dynamic truth are complete enough to adjudicate every warning;
- both source roles produce stable confirmed tracks with no GT identity input;
- the registered target is not missed;
- cooperative truth-positive warning lead is at least 0.5 s;
- matched-benign false-warning active frames are at most 10%, with cooperative
  excess at most 2 percentage points; and
- the naturalistic result is reported beside the designed result.

If this three-trajectory pilot fails, stop. Do not sweep tracker constants,
collect the full suite, or insert OAI.

## Truth and later network overlays

The next capture contract must snapshot a hashed Town/map-specific static
environment catalog at setup: environment-object ID, semantic class, transform,
oriented bounding-box centre/extents, enabled state, map name/hash, and capture
time. Dynamic actors remain a separate per-frame stream. Adjudication matches
dynamic actors first, then static objects, and evaluates physical future
clearance; static matches are not automatically safe or hazardous.

Only after the local three-scenario warning pilot passes should identical
captured messages be replayed through good, near-knee, fade/recovery, and later
route-indexed SNR treatments. Network state remains causal, and a route-ahead
radio map is a separate ablation. The later agent can then compare
`SPLIT_FEATURE`, `LOCAL_INFER`, and `SKIP_INFERENCE`, followed by
`PUBLISH_ALL`, `PUBLISH_HAZARD_SUBSET`, or `SKIP_PUBLICATION`, without
confounding scene validity, perception stability, and transport effects.
