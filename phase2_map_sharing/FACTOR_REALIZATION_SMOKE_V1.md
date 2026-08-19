# Phase-2 factor-realization smoke v1

Status: **offline contract drafted; collection blocked until the per-row factor
adapter, recipient-install event, and exact policy-feature projection are
implemented and reviewed.** This document does not authorize CARLA, OAI,
warning selection, controller evaluation, or RL.

## Question and cap

Before scaling collection, answer one narrow question: do the declared speed
and hazard-horizon cells produce different physical trajectories and a causal
recipient-installed-track endpoint, or are they only manifest labels?

The cap is 16 world trajectories:

```text
2 geometries
  x 2 closing-speed bands
  x 2 pre-intervention hazard-proximity-horizon bands
  x {positive, matched benign}
= 16 trajectories / 8 paired groups
```

The selected geometries are one pedestrian case
(`curbside_bus_occluded_pedestrian`) and one vehicle case
(`occluded_cross_traffic_vehicle`). This is defensible as an integration
smoke—not a power calculation—because it exercises every corner once in both
supported hazard classes while retaining the matched negative needed to prove
that only the hazard treatment changed. More geometries or repeats cannot fix
a broken factor adapter and are forbidden at this gate.

These are exact replicate-0 calibration rows, not throwaway scenarios. If and
only if the complete atomic batch passes, its 16 immutable captures count
toward the planned 66-trajectory calibration tranche. If any row or gate
fails, the whole batch remains an excluded diagnostic fixture and is never
double-counted as calibration evidence. A PASS stops for human review; it does
not chain into the superseded 15-row audit or any larger collection.

## Physical-factor contract

The v2 design uses `requested_hazard_onset_s` as a provisional, authored scene
control. That is legitimate evaluation metadata; it is not evidence that the
requested physical cell occurred, and it is never a policy input. Admission is
decided only from the geometry-specific kinematics at the first realized onset
sample, before recipient intervention. A later manual-driving/MWC holdout must
replace the authored clock trigger with route-progress or recipient-ETA
triggering so different human approaches encounter the conflict naturally.

The runtime must record both the complete row request and realized values for:

- typed pre-intervention closing speed; and
- typed `pre_intervention_hazard_proximity_horizon_s`.

Each value carries the exact geometry, closing-speed, and proximity-horizon
basis identifiers pinned in the v2 manifest. The proximity horizon is an
instantaneous relative-linear-motion diagnostic, not collision TTC, braking
TTC, or a safety guarantee. The matching benign twin repeats the positive's
complete non-treatment motion plan and removes only the target; its immutable
non-treatment-plan hash must equal the positive's.

Realized positive values must be inside their manifest-declared closing-speed
and proximity-horizon bands. Error from the nominal target is reported, not
made into an invented tolerance before calibration. A positive must also
realize the geometry-specific pre-intervention proximity condition. Missing,
mis-binned, or wrong-basis measurements fail; labels are never accepted as
proof. The historical column name `time_to_hazard_band` remains a
stratification label only; the typed measured quantity is
`pre_intervention_hazard_proximity_horizon_s`.

## Recipient-knowledge endpoint

Helper-local confirmation is an explanatory sensing upper bound. It does not
prove that the recipient knew the object. The smoke endpoint is:

```text
recipient_available_confirmed_track_margin_s
  = recipient_own_target_confirmation_at_s
    - helper_target_track_recipient_available_at_s
```

The helper chain must identify source confirmation, contribution publication,
recipient install, recipient-map track, and availability on one aligned clock.
The required order is:

```text
helper confirmation <= publish <= install <= recipient availability
```

No event is replaced with an arbitrary time. Each positive is typed as
`numeric`, `ego_right_censored`, `cooperative_miss`, or `both_miss`, with an
explicit horizon and hashed evidence chain. Thus a detector miss remains a
scientific outcome while an absent install log remains a contract failure.
This smoke is local-loopback only: it makes no OAI transport claim and runs no
warning-parameter selection.

## No policy shortcut

Audit metadata may retain frame IDs, scenario time, trajectory IDs, route IDs,
world coordinates, seeds, and truth. The learned policy projection may not.
Placement and publication each have an exact, stage-specific feature list in
`data_collection/configs/phase2_factor_realization_smoke_v1.yaml`. The eventual
consumer must prove it uses that exact projection; merely writing an allowlist
beside a wider dictionary is insufficient.

In particular, factor bands, requested/realized hazard timing, geometry IDs,
authored onset seconds, and absolute episode clocks are evaluation metadata.
The controller sees
causal relative kinematics, derived ages/uncertainty/deadline summaries,
lagged network state, compute state, and prior outcomes—not the answer key.

## What a PASS does not prove

This smoke proves only that the corpus generator can realize all four physical
factor corners in one pedestrian and one vehicle geometry, and that the
recipient-knowledge endpoint is computable. It does not establish population
performance, warning thresholds, OAI latency, human-driver behaviour, or
generalization to a fixed demo route. Those claims require the pre-registered
calibration/validation/test suites, with route/seed separation and a separate
manual-driving holdout. The agent controls inference placement and publication;
it does not learn CARLA autopilot steering or braking quirks.

## Gates and sequence

All gates are conjunctive:

1. the exact 16 pinned replicate-0 rows and their source hashes match;
2. every positive realizes its requested closing/horizon cell under the typed
   geometry-specific basis;
3. every benign twin has no registered target and matches its positive's
   non-treatment-plan hash;
4. every positive has a typed, recomputable recipient-installed-track endpoint;
5. the exact policy feature projection contains no forbidden clock, identity,
   route, geometry, factor-label, or truth field;
6. warnings were record-only, OAI did not run, and artifacts are immutable; and
7. no downstream stage was chained.

The offline validator is:

```bash
python -m data_collection.validate_phase2_factor_realization_smoke
```

It currently returns `PASS_OFFLINE_DESIGN_COLLECTION_BLOCKED` because the
factor adapter, install event, policy-feature projection, and separate launch
wrapper do not yet exist. That is intentional. `--require-runtime-ready`
returns nonzero, so the design check cannot be mistaken for launch authority.
A later capture result is checked with `--validate-results`; only its atomic
PASS verdict admits these exact rows into calibration.
