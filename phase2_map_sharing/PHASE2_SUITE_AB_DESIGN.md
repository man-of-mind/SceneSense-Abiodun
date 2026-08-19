# Phase-2 powered Suite A/B design

Status: **deterministic design-freeze candidate; collection remains
unauthorized**, 2026-08-17.

Authoritative machine-readable inputs and outputs:

- config: `configs/phase2_suite_ab_design_v1.yaml`;
- generator/validator: `design_suite_manifest.py`;
- manifest: `design/phase2_suite_ab_v1/trajectory_group_manifest.csv`;
- sensitivity table: `design/phase2_suite_ab_v1/power_sensitivity.csv`; and
- hashed summary/provenance: `design/phase2_suite_ab_v1/`.

## 1. Suite identity and claim boundary

The names are fixed:

- **Suite A — designed decision opportunities.** This is the powered,
  regime-bounded C2 test. It intentionally produces helper-visible hazards
  and matched benign negatives.
- **Suite B — naturalistic operation.** This is the honest denominator. It
  never forces a hazard and is not pooled with Suite A for the headline.

The accepted two-trajectory pilot is excluded. Policy arms are replayed from
one immutable capture with isolated state. A Suite A independent group contains
two world trajectories—the controlled positive and its benign twin. A Suite B
group contains one naturalistic world trajectory; its ego-only,
send-everything, and hazard-only arms still use isolated replay state.

## 2. Frozen candidate inventory

| Suite | Calibration groups | Validation groups | Test groups | World trajectories |
|---|---:|---:|---:|---:|
| A designed | 24 | 24 | 72 | 240 |
| B naturalistic | 18 | 18 | 54 | 90 |
| **Total** | **42** | **42** | **126** | **330** |

Suite A contains 24 factor cells and five independent route/seed replicates per
cell. Each cell is split exactly 1 calibration / 1 validation / 3 test:

- six geometry families: three pedestrian and three vehicle;
- low/high closing-speed bands; and
- short/long time-to-hazard bands.

Traffic density is an orthogonally balanced nuisance factor: each split is
exactly balanced across sparse, typical, and dense groups. Cyclists are not
silently included because the current perception model has no validated
cyclist contract.

Suite B has 30 independent seeds on each of three paired route families. Its
declared quotas are 25% sparse / 50% typical / 25% dense approximately, and
70% clear / 20% cloudy / 10% wet exactly across each route. Hazards arise at
natural prevalence; a run with no positive hazard remains in the denominator.

The signalized-demo and safe-perimeter loops share their first 44 route points
(165.887 m), which is longer than any registered 12 s trajectory can traverse.
They therefore cannot all start at row zero. Each loop has six
geometry-only, non-junction start-anchor strata distributed around the route.
Every anchor appears once in calibration, once in validation, and three times
in test. The helper starts 10--20 m ahead on the same native lane and direction;
the exact source-route hashes and recipient/helper indices are manifest fields.
Anchor selection never uses detections, warnings, or outcomes. Results remain
conditional on these Town10HD_Opt route families and are not a town-wide
generalization claim.

Every independent group has a deterministic unique CARLA, traffic, and sensor
seed. Positive/benign twins share seeds and cannot cross splits. Confirmatory
rows are flagged before any outcome exists. Tests fail if Suite labels are
inverted, a group leaks across splits, a seed is reused, or a factor cell loses
its 1/1/3 assignment.

### Renderer contract

Every one of the 330 primary design rows declares CARLA `Epic` and the exact
server flag `-quality-level=Epic`. CARLA cannot expose this setting over RPC, so
the launch and run manifests must repeat the operator declaration and fail closed
if it is missing or differs. Existing Low captures remain a labelled
perception-domain stress diagnostic and are not multiplied into this powered
corpus; no future Low collection is authorized by the design. This is an
operational primary-setting freeze, not a claim that Epic statistically dominated
every class: the pre-registered <=12 m dense pedestrian component had zero support.

## 3. Power position—what is and is not claimed

The excluded one-pair pilot cannot estimate between-group variance. Its 3.6 s
descriptive lead is therefore not used to power the study.

The candidate freezes:

- smallest effect of interest: 0.5 s;
- two-sided alpha: 0.05;
- 72 untouched Suite A positive test groups;
- planned 10% nonnumeric/censored loss, leaving 64 numeric pairs; and
- minimum required power: 0.80.

The 0.5 s value is a **cross-cell research floor**, not a braking-safety
threshold. At the 10 Hz sensor/evaluation cadence it is five frames; at the
separate 20 Hz surrogate policy clock it is ten decision opportunities. Under
the registered closing-speed bands it corresponds to 1--2 m of closing travel
in the low band and 3--5 m in the high band. Warning lead by speed band, those
distance equivalents, and causal deadline slack are mandatory reports. A
braking-derived actionable deadline is deferred until every arm has the same
fixed warning-actuation adapter and the advisor freezes reaction, deceleration,
and clearance parameters.

The checked-in sensitivity table shows approximate paired-t power of 0.883 at
0.5 s effect, 1.25 s paired standard deviation, and 10% censoring. This is a
planning sensitivity—not the registered censored/clustered analysis and not a
claim that the true standard deviation is 1.25 s.

After calibration, simulate the **actual registered estimator** using observed
event yield, censoring, paired variance, false-warning exposure, and
missed-hazard discordance. Validation may start only if the planned test counts
provide at least 0.80 power for the 0.5 s lead endpoint and adequate precision
for both non-inferiority endpoints. If not, stop and revise counts before any
validation/test data are collected. Do not weaken the endpoint or margins.

Power alone is insufficient. Every setting retained for validation must also
pass an absolute research-usability gate on Suite-A matched benign negatives:
adjudicated false-warning-active frames at most 10%, false-warning episodes at
most 1/min, and the existing cooperative-versus-ego margin of no more than
+2 percentage points. Failure stops before validation. These are C2 research
gates, not certified automotive limits. First-warning timing remains
registered-target-specific; unrelated warnings cannot move that endpoint.
The two absolute rates pool their numerators and eligible benign exposure over
all calibration trajectories for an arm/candidate, with trajectory-cluster
intervals reported. In particular, the episode-rate gate is not applied to
each short trajectory separately.

## 4. Retention and runtime

The pilot measured about 2.75 MB of aligned inputs and 19.95 MB of logits per
role/frame. Continuous full retention for 330 trajectories would be roughly
1.8 TB and is forbidden.

The candidate uses:

- causal lightweight records, unfiltered detections, final detections, tracks,
  truth, ego state, actions, queue/network timestamps, and manifests for every
  trajectory;
- a 4 s aligned-input window for every calibration and validation trajectory;
- logits plus inputs for one calibration audit group per designed geometry and
  naturalistic route (15 world trajectories total); and
- no heavy raw window in the confirmatory test.

Using pilot-measured bytes, the total estimate is **54.61 GB**, below the
hard **80 GB** design cap while preserving the existing 500 GB free-space
floor. The collector must still reserve space and enforce permits before every
write; this estimate is not permission to exceed a quota.

Pilot wall time was about 2.9 minutes per world trajectory. The candidate
therefore estimates:

| Stage | World trajectories | Capture time estimate |
|---|---:|---:|
| Calibration | 66 | 3.2 h |
| Validation | 66 | 3.2 h |
| Confirmatory test | 198 | 9.6 h |
| **Total** | **330** | **16.0 h** |

Each stage uses a detached, self-logging runner and stops at its human gate.
No stage is chained into the next.

### Calibration-audit execution contract

The first runnable stage is frozen in
`data_collection/configs/phase2_calibration_audit_v1.yaml`. It selects exactly
nine calibration groups from the immutable manifest: six designed matched
groups (positive plus benign) and three naturalistic groups, for 15 world
trajectories. Each trajectory runs for 120 frames at the native 10 Hz CARLA
clock under Epic rendering. Lightweight causal records span the full 12 s;
each role retains exactly 40 aligned input/logit pairs inside its reviewed 4 s
window. The expected heavy-data total is 27.24 GB, protected by a 3 GB
per-trajectory cap, an 80 GB stage cap, a 500 GB post-write free-space floor,
and a 580 GB preflight requirement. Automatic deletion is forbidden.

The runner fails at the first trajectory that violates actor cleanup, zero
unintended-collision traffic sanity, the exact M-prime
sensor contract, the approximately 18,591.5 projected-radar-points/frame
reference (plus or minus 10%), causal-field completeness, the 40-pair raw
retention contract, truth recoverability, or artifact hashes. Matched designed
positive/benign trajectories must also realize identical frozen ambient
signatures and full-frame positions. Their generic background context contains
**no ambient vehicles or walkers**: the reviewed scenario already owns the
helper, recipient, occluder, and, where applicable, registered pedestrian or
target vehicle. The traffic generator and its READY/RELEASE handshake are not
launched for Suite A. Thus designed twins exactly reproduce the manually
validated scenario and have no dependency on CARLA's sparse vehicle spawn
catalog, random navigation mesh, or background traffic controller. A
stopped-network/gridlock test is inapplicable to this declared static context,
but owned-actor initial-state equality, collision monitoring, and cleanup remain
hard. Suite-A manifest rows are explicitly labelled
`traffic_density=not_applicable`, `ambient_population_mode=scenario_owned_only`,
and `ambient_population_process_required=0`; density evidence comes from Suite
B only. Explicit distractor clutter may be added later only through reviewed,
scenario-owned transforms if calibration shows it is needed.

Naturalistic trajectories are unpaired and use ordinary safe Traffic Manager
motion plus walker AI. Sparse/typical/dense request 6/4, 10/8, and 15/12
vehicles/walkers, respectively. Their spawn corridor is wider (0--30 m) and
may use the reviewed-route fallback; collision and persistent-gridlock gates
both remain hard. The output manifest records `ambient_evidence_layer`, counts,
and both motion modes so controlled causal contrasts cannot be pooled silently
with the naturalistic denominator.

A successful CARLA stage is labelled
`audit_capture_and_per_trajectory_verification_complete`. That status proves
capture and replay *sufficiency* only: it checks that the retained artifacts
support the registered 4 x 3 x 2 x 3 = 72-setting map-engine grid, but
deliberately does not claim those offline replays have run. The four search
axes are recipient warning-emission confidence floor
`{0.05, 0.10, 0.15, 0.20}`, map association base gate `{2, 3, 4}` m, map-track
TTL `{0.5, 1.0}` s, and warning uncertainty multiplier `{0, 1, 2}`. The
confidence floor is applied only after map installation and association; it is
not a map-admission filter. Source perception is not a fifth calibration
surface: the detector candidate floor remains fixed at 0.05, all decoded
observations remain eligible for the recipient map, and the captured
source-local tracker remains fixed at its 5 m / three-missed-frame contract.
Exact OAI enqueue/on-wire/reassembly/install fields also remain blocking and
unmeasured in this CARLA-only stage.
Both items require their own post-capture human gate; no remaining calibration,
validation, test, controller, or RL work is chained automatically.

The accepted three-trajectory regression batch `20260818_230028_audit`
retains its create-only `resolved_config.yaml`, which records the superseded
96-point declaration used at capture time. That historical file is immutable
provenance and must not be rewritten. No replay was executed under it. The
retained source artifacts remain sufficient for the narrower binding 72-point
map-engine grid, which governs every new replay/configuration.

## 5. Scenario-authoring status

Already reviewed:

- curbside bus-occluded pedestrian geometry;
- curbside legal opposing helper/recipient route;
- signalized-corner stopped-van pedestrian geometry; and
- its distinct-lane recipient-turn/helper-straight route pair;
- parked-van midblock pedestrian geometry; and
- its non-junction legal opposing-lane route pair;
- occluded cross-traffic vehicle geometry; and
- its accepted signalized ego routes plus frozen northbound target route;
- parked-vehicle pullout geometry; and
- its accepted midblock ego routes plus frozen curb-to-lane target route.
- queue-reveal stopped-lead vehicle geometry; and
- its accepted midblock ego routes plus frozen queue-member curb-exit route.
- signalized-demo naturalistic route with all six same-lane start-anchor
  strata automatically gated and visually accepted.
- safe-perimeter naturalistic route with all six same-lane start-anchor strata
  automatically gated and visually accepted.

No scenario geometry remains pending. Both naturalistic families are frozen
under `town10hd_opt_same_lane_helper_ahead_v1`; this removes the geometry gate
but does not authorize collection past the staged calibration gates.

Every new geometry needs the same manual review used for the pilot: legal lane
IDs/headings, no pose overlap, visible helper advantage for positive hazards,
matched benign equivalence, realistic motion, collision-free ambient traffic,
camera visibility, and clean actor teardown.

## 6. Staged gate sequence

1. **Complete:** both paired naturalistic routes are visually accepted and the
   shared pair-contract ID plus both route hashes are frozen.
2. Run only the nine preselected calibration audit groups—six designed groups
   (positive + benign) and three naturalistic groups, 15 world trajectories,
   about 44 minutes and about 27 GB of heavy-window data.
3. After the CARLA stage stops, run the bounded 72-setting offline map-engine
   replay from the retained artifacts and review it separately; then verify
   the exact OAI timestamp and byte fields in the later OAI measurement.
   Neither result is implied by a CARLA capture-complete sentinel.
4. Complete the remaining calibration trajectories and run the registered
   simulation-power gate. Stop on insufficient event yield, excessive
   censoring, warning burden, or power.
5. Only after review, collect validation and freeze one operating point.
6. Only after a second review, collect the untouched test and apply the C2
   decision rule.

LOCAL/OAI calibration and the causal controller ladder remain downstream.
Nothing in this design authorizes RL.
