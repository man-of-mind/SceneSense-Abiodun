# Phase-2 factor-realization contract

Status: **offline design v2; live collection remains blocked.** This contract
does not authorize CARLA, OAI, warning selection, controller evaluation, or
RL training.

## Why v2 exists

Suite-A v1 crossed `closing_speed_band` with `time_to_hazard_band`, but those
columns were labels: the calibration runtime still requested the same ego
speeds and hazard onset for every row. A model trained on that corpus could
therefore appear insensitive to the factors simply because the environment
never realized them.

Suite-A v2 keeps the historical factor labels for design stratification and
adds deterministic requested controls to every manifest row:

- helper and recipient speed;
- registered hazard-actor speed;
- onset-driver speed (different from the hazard actor for queue reveal);
- requested hazard onset;
- target and bounds for realized radial closing speed; and
- target and bounds for a typed pre-intervention proximity horizon.

The positive and matched-benign rows in a group carry exactly the same
requested controls. The benign row removes only the registered hazard, so its
realized hazard metrics are `not_applicable`; it must never be assigned a
fabricated target measurement. Naturalistic Suite B also has no fabricated
hazard request.

## Scientific claim boundary

`time_to_hazard_band` is **not scientifically realized yet**. The intended
realized diagnostic is named
`pre_intervention_hazard_proximity_horizon_s`. It must include a
geometry-specific `geometry_measurement_basis`, plus explicit closing-speed
and proximity-horizon basis identifiers. It is an instantaneous kinematic
proximity diagnostic—not collision TTC, braking TTC, warning lead, or a
safety guarantee.

For pedestrian, cross-traffic, pullout, and queue-reveal geometries, the
registered hazard and the actor that initiates reveal/onset are typed
separately. In queue reveal, for example, the lead vehicle is the stationary
hazard while the moving occluder is the onset driver. This distinction must be
preserved by the runtime and verifier.

## Required live gate before factor freeze

The next implementation must consume the v2 row controls, log requested and
realized kinematics, and write a positive-only result with all of the
following:

1. a realized onset sample before recipient intervention;
2. realized recipient, hazard, and onset-driver speeds;
3. realized radial closing speed and typed proximity horizon;
4. exact basis IDs copied from the design row; and
5. an in-band verdict for both realized factors.

A missing sample, missing basis, non-finite value, or out-of-band positive
fails closed. Benign twins remain `not_applicable` for the hazard metric while
retaining identical requested controls. Only a bounded factor smoke may
calibrate or freeze the provisional control-to-bin mapping.

## Frozen offline artifacts

- Config: `phase2_map_sharing/configs/phase2_suite_ab_design_v2.yaml`
- Manifest: `phase2_map_sharing/design/phase2_suite_ab_v2/trajectory_group_manifest.csv`
- Manifest schema: `scenesense.phase2_suite_design_manifest.v2`

The v1 files remain unchanged as historical provenance for completed pilots.

The v2 primary endpoint is named
`recipient_available_confirmed_track_margin_s`, but its recipient-install
event chain and calibration distribution do not yet exist. Consequently v2
registers no effect size and grants no power authorization. The generated
power table preserves the old 0.5 s warning-lead sensitivity only as an
explicitly non-authoritative historical reference; it cannot justify v2
sample size or confirmatory collection.

The earlier absolute warning-nuisance gate is likewise retained as
`historical_failed_secondary_not_blocking_C2`. It still blocks any future
warning-performance claim, but it cannot block or validate C2 installed-track
evidence.

Installed-track quality uses a non-circular two-stage contract. Before the
exact 16-row calibration tranche, freeze metric definitions, denominators,
and structural integrity gates for false installs, duplicates,
source-to-recipient fragmentation, and recipient-map pollution. The 16 rows
then estimate those distributions. Numeric research-usability thresholds are
registered from that estimate before any additional calibration or
validation; they are never claimed as validated on the same 16 rows that set
them.
