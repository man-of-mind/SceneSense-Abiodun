# Phase-2 factor-realization smoke v1

Status: **the four runtime adapters are implemented and offline-verified;
collection remains blocked on the hash-bound eight-corner physical review and
a separate detached-launch decision.** This document by itself does not
authorize CARLA, OAI, warning selection, controller evaluation, or RL.

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

This whole-batch rule supersedes any earlier wording that could be read as
admitting individually passing rows. Partial admission, post-hoc relabelling,
or selecting successful cells after inspecting outcomes is forbidden.

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

The surface-clearance gate is a counterfactual relative-motion prediction used
to establish conflict proximity. A value of zero may mean predicted OBB
overlap; it does not claim a realized collision or a non-collision. Actual
collision count is an independent zero-collision structural gate.

The exact raw window is 40 samples at 10 Hz (a measured span of 3.9 s).
For factor rows its configured start is authored onset minus 0.9 s, bounded to
the trajectory, and rounded to the first sensor sample on or after that offset.
Authored onset remains evaluation-only and is never a policy feature. Before
the exact-16 launch, every positive corner review must show that its measured
physical onset leaves at least 2.9 s through the exact expected last retained
sample. This is a prelaunch one-tick margin over the immutable 2.8 s postflight
minimum; an onset one tick after authored timing passes the unbounded cells,
while a two-tick slip rejects them rather than risking a long unusable capture.

## Recipient-knowledge endpoint

Helper-local confirmation is an explanatory sensing upper bound. It does not
prove that the recipient knew the object. The smoke endpoint is:

```text
recipient_available_confirmed_track_margin_s
  = recipient_self_target_track_recipient_available_at_s
    - helper_target_track_recipient_available_at_s
```

The helper chain must identify source confirmation, contribution publication,
recipient install, recipient-map track, and availability on one aligned clock.
The recipient-self chain must separately identify source confirmation, local
map install, and consumer availability. The required orders are:

```text
helper confirmation <= publish <= install <= recipient availability
recipient-self confirmation <= local install <= recipient-self availability
```

No event is replaced with an arbitrary time. Each positive is typed as
`numeric`, `ego_right_censored`, `cooperative_miss`, or `both_miss`, with an
explicit horizon and hashed evidence chain. Thus a detector miss remains a
scientific outcome while an absent install log remains a contract failure.
This smoke is local-loopback only: it makes no OAI transport claim and runs no
warning-parameter selection.

The endpoint replay is byte-pinned to the frozen
`source_local_confirmed_cv.v3` tracker, `RecipientMapEngineV3`, the v3 warning
repair config, and the calibration-replay config named in the smoke YAML. The
complete postflight dependency set (tracker/map engines, causal schema,
replayers, truth adjudicator, postflight, and validator) is byte-pinned there
as well. The local install and following consumer snapshot share one simulated
timestamp,
so the explicit consumer hand-off delay is 0.0 s. That is a same-tick software
ordering assumption, not a claim of zero physical or OAI latency.

The transport-inclusive margin is a signed measurement, not a success
threshold. A near-zero or negative value is a legitimate immutable result: it
means helper-track consumer availability did not beat recipient-self-track
consumer availability after their respective causal install paths.
It is not permission to retune confirmation, transport, association, or the
already-failed v3 warning rule after seeing the result. No minimum positive
margin is registered for admission.

## No policy shortcut

Audit metadata may retain frame IDs, scenario time, trajectory IDs, route IDs,
world coordinates, seeds, and truth. The learned policy projection may not.
Placement and publication each have an exact, stage-specific feature list in
`data_collection/configs/phase2_factor_realization_smoke_v1.yaml`. The eventual
consumer must prove it uses that exact projection; merely writing an allowlist
beside a wider dictionary is insufficient.

Every real trajectory in this tranche must also audit every consumed field at
the loader boundary and establish `available_at_s <= decision_at_s`. Both
placement and publication must execute at least one real decision per row with
zero temporal violations. Once per trajectory, an isolated evaluation-only/GT
canary is offered to the loader and must be rejected; the canary is never
passed into the policy. Thus the real 16 exercise both the positive causal path
and the fail-closed leakage path rather than relying only on unit tests.

This is a loader-plumbing exercise, not a policy rollout. Placement is fixed to
`SPLIT_FEATURE`, publication to `PUBLISH_ALL`, and neither action is selected
from the projected values. Because OAI and a compute monitor are absent, the
network/scheduler/prior-delivery/compute fields use the exact finite sentinel
values registered in the YAML with provenance
`preregistered_disabled_channel_fixture`. Zero capacity and zero uncertainty
are a disabled/no-OAI sentinel, **not** a neutral or observed channel. Live
track fields remain causal. Recipient state, relative kinematics, and installed
map feedback at the helper decision locus are explicitly tagged
`local_loopback_transport_abstraction`: they use same-simulated-timestamp
post-capture handoff and are not observed live transport or network state. The
result can prove exact-schema loading, availability ordering, and forbidden-
field rejection; it cannot prove complete observed state, policy readiness,
performance, or action quality. A future live controller must reject every
fixture-backed field until a causal measurement or registered missingness
contract replaces it, and must replace each local-loopback transport
abstraction with a measured causal transport or missingness path.

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
   geometry-specific basis, and its reviewed physical onset leaves the
   registered 2.9 s exact-sample retention margin;
3. every benign twin has no registered target and matches its positive's
   non-treatment-plan hash;
4. every positive has a typed, recomputable recipient-installed-track endpoint;
5. the exact policy feature projection contains no forbidden clock, identity,
   route, geometry, factor-label, or truth field;
6. every consumed feature on every real row satisfies
   `available_at_s <= decision_at_s`, both decision stages execute, and each
   row's isolated forbidden-field canary is rejected by the actual loader;
7. protocol-false-install, duplicate-install, source-to-recipient
   fragmentation, and recipient-map-pollution denominators are typed and the
   frozen structural integrity gates pass; an install is protocol-false only
   when valid contribution/object/track provenance is absent, while
   truth-unmatched installs are a separate report-only evaluation diagnostic
   (the 16 estimate distributions but do not validate numeric usability
   thresholds);
8. the resolved run captures the frozen Car/Truck/Bus static-environment truth
   registry after each fresh world reload and before any dynamic actor spawn;
9. both role manifests on every row record a model-checkpoint SHA-256 that is
   recomputed from the checkpoint used at capture time, and one identical
   frozen checkpoint digest is used by both roles across all 16 rows; and
10. every row passes the radar/sensor and traffic sanity checks with zero
    relevant collision incidents, and each completed positive/benign pair
    passes the initial, scenario-owned, static-environment, and full-trajectory
    equality gates from their hash-bound source artifacts;
11. warnings were record-only, OAI did not run, and artifacts are immutable;
    and
12. no downstream stage was chained.

The offline validator is:

```bash
python -m data_collection.validate_phase2_factor_realization_smoke
```

While any runtime-readiness field is unverified, it returns
`PASS_OFFLINE_DESIGN_COLLECTION_BLOCKED`; `--require-runtime-ready` then
returns nonzero, so an offline design check cannot be mistaken for launch
authority. The current implementation has verified the factor adapter,
recipient-availability postflight, exact causal loader audit, atomic
result-bundle validation, and detached wrapper. That readiness still does not
start CARLA: the eight physical corners must pass and be accepted separately.
During capture, postflight runs immediately after each complete matched pair,
while final admission remains all-or-none. A later capture result is checked
with `--validate-results`; only the registered atomic PASS verdict
`PASS_ATOMIC_EXACT_16_ADMITTED` admits these exact rows into calibration.
