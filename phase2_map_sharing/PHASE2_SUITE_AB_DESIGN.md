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

Still blocking collection:

- paired helper/recipient versions of the signalized-demo and safe-perimeter
  naturalistic routes.

Every new geometry needs the same manual review used for the pilot: legal lane
IDs/headings, no pose overlap, visible helper advantage for positive hazards,
matched benign equivalence, realistic motion, collision-free ambient traffic,
camera visibility, and clean actor teardown.

## 6. Staged gate sequence

1. Author and visually accept the two paired naturalistic routes. Freeze their
   IDs and route hashes.
2. Run only the nine preselected calibration audit groups—six designed groups
   (positive + benign) and three naturalistic groups, 15 world trajectories,
   about 44 minutes and about 27 GB of heavy-window data.
3. Prove every bounded confidence/association/TTL/uncertainty setting replays
   from retained artifacts; verify exact local/OAI timestamp and byte fields.
4. Complete the remaining calibration trajectories and run the registered
   simulation-power gate. Stop on insufficient event yield, excessive
   censoring, warning burden, or power.
5. Only after review, collect validation and freeze one operating point.
6. Only after a second review, collect the untouched test and apply the C2
   decision rule.

LOCAL/OAI calibration and the causal controller ladder remain downstream.
Nothing in this design authorizes RL.
