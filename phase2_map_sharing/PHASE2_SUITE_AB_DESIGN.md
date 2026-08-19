# Phase-2 powered Suite A/B design

Status: **2026-08-19 endpoint-reconciliation proposal pending joint review;
collection remains unauthorized**. The checked-in v1 machine-readable artifacts
preserve the warning-lead-era design and must not be overwritten. The offline v2
factor manifest now exists, but it has no runtime or power authority; its
adapters, track-quality guardrails, and bounded smoke remain prerequisites.

Historical v1 machine-readable inputs and outputs:

- config: `configs/phase2_suite_ab_design_v1.yaml`;
- generator/validator: `design_suite_manifest.py`;
- manifest: `design/phase2_suite_ab_v1/trajectory_group_manifest.csv`;
- sensitivity table: `design/phase2_suite_ab_v1/power_sensitivity.csv`; and
- hashed summary/provenance: `design/phase2_suite_ab_v1/`.

Proposed offline v2 inputs and outputs:

- config: `configs/phase2_suite_ab_design_v2.yaml`;
- manifest and sealed artifacts: `design/phase2_suite_ab_v2/`;
- factor contract: `PHASE2_FACTOR_REALIZATION_CONTRACT.md`; and
- bounded smoke contract: `FACTOR_REALIZATION_SMOKE_V1.md` plus
  `../data_collection/configs/phase2_factor_realization_smoke_v1.yaml`.

## 1. Suite identity and claim boundary

The names are fixed:

- **Suite A — designed decision opportunities.** This is the future powered,
  regime-bounded C2 test. It intentionally produces helper-visible hazards
  and matched benign negatives. Its primary timing endpoint is a causally
  delivered, accepted usable helper track available to the recipient consumer
  before a recipient-self track is available there; actionability slack is mandatory
  stratification of potential timeliness, not proof of a valid warning or safe
  response.
- **Suite B — naturalistic operation.** This is the honest denominator. It
  never forces a hazard and is not pooled with Suite A for the headline.

Both completed pilots, including the 2026-08-19 decision-opportunity
positive/benign/naturalistic batch, are excluded. Policy arms are replayed from
one immutable capture with isolated state. A Suite A independent group contains
two world trajectories—the controlled positive and its benign twin. A Suite B
group contains one naturalistic world trajectory; its ego-only,
send-everything, and hazard-only arms still use isolated replay state.

## 2. Preserved v1 candidate inventory; realization gap blocks launch

| Suite | Calibration groups | Validation groups | Test groups | World trajectories |
|---|---:|---:|---:|---:|
| A designed | 24 | 24 | 72 | 240 |
| B naturalistic | 18 | 18 | 54 | 90 |
| **Total** | **42** | **42** | **126** | **330** |

The v1 candidate contains 24 intended factor cells and five independent
route/seed replicates per cell. Each cell is split exactly 1 calibration / 1
validation / 3 test:

- six geometry families: three pedestrian and three vehicle;
- low/high closing-speed bands; and
- short/long time-to-hazard bands.

Those speed and time-to-hazard values are **labels only today**. The v1
manifest does not bind them to per-geometry actor controls or fail a row when
its realized kinematics land outside its label. Consequently none of the old
15-trajectory audit, 66-trajectory calibration plan, or 330-trajectory full
plan is launch authorization. Visual geometry acceptance does not close this
scientific gap.

The versioned successor must use a typed urgency contract per geometry rather
than one universal TTC: define the conflict event/surface, evaluation-only truth
fields, prediction model and horizon, intended band, numeric realized bounds,
and tolerance. Persist realized recipient speed, closing/range or clearance to
that surface, conflict-time/censoring, helper-visible interval,
recipient-self-availability time, usable-helper install and consumer-availability
times, and actionability slack. A row counts only if those realized quantities satisfy its
preassigned cell.

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

The successor manifest additionally balances/jitters hazard onset, route start,
and sensor/scheduler phase independently; varies preregistered recipient-motion
profiles; and reserves an unseen onset range, one complete scripted
driver-motion profile, and a geometry/route combination as explicit
extrapolation stresses. Split assignment occurs before feature construction.
The policy receives causal relative kinematics, AoI/uncertainty, lagged network
state, and protocol state—not scenario time/frame, ID, factor label, seed,
planned target phase, future truth, absolute hazard location, or driver-profile
identity. Absolute route position is isolated to a registered radio-map
ablation with route-family holdout.

Manual audience driving for MWC 2027 is a post-freeze human-in-the-loop holdout,
never part of training, calibration, model selection, or confirmatory C2. The
frozen system observes live ego kinematics and recomputes urgency; manual runs
report system availability, driver response, clearance/collision, and comfort
separately and include an explicit safety override. Scripted research outcomes
must not be represented as measured human response.

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

Neither excluded pilot estimates between-group variance. In particular, the
2026-08-19 pilot's 2.4 s helper-local-versus-recipient-self confirmation gap is
a one-trajectory, zero-transport upper bound, not recipient-available C2 gain.
Its 3.3 s warning lead also remains secondary and failed the unchanged benign
specificity gates. Neither number powers the future study.

The checked-in v1 sensitivity table and its 0.5 s warning-lead effect, 1.25 s
paired standard deviation, 10% censoring, 72 positive test groups, and 0.883
illustrative power are preserved as **historical planning assumptions**. They
must not be relabelled as power for `recipient_available_confirmed_track_margin_s`. Two-sided alpha
0.05 and minimum required power 0.80 remain design principles; the effect size,
censoring model, estimator, and counts are re-frozen from the versioned
endpoint plan and passing calibration evidence before validation.

The future analysis has one primary information endpoint and one required
safety-relevance interpretation:

1. `recipient_available_confirmed_track_margin_s = recipient_self_available_s -
   recipient_usable_helper_available_s`, where helper availability follows
   accepted install and both sides use the same recipient-consumer boundary,
   with explicit censoring and target-track validity/quality requirements; and
2. mandatory actionable-success and continuous actionability-slack strata under typed
   per-geometry conflict definitions and preregistered response profiles. If
   reaction/deceleration/clearance profiles are not frozen, no actionable claim
   is made. Even when frozen, these are potential-timeliness interpretations;
   they do not show that a valid warning or actuation occurred. C3 remains
   failed/unresolved on the present warning stack.

The proposed 16-trajectory factor-realization tranche is assigned to
calibration before launch. Every passing trajectory counts toward calibration;
it is not discarded as another pilot. Use it to estimate realization yield,
censoring, recipient-consumer track availability, cluster variance, and benign map
pollution. Validation remains blocked until the registered estimator has at
least 0.80 planned power and track/map-quality guardrails have adequate
precision. Failed/out-of-cell rows are never relabelled, and no margin is
weakened after seeing data.

The old warning-rule usability gates—at most 10% false-warning-active frames,
at most 1/min false-warning episodes, and no more than +2 percentage points
cooperative excess—remain unchanged for any warning claim. The current v3
warning stack failed them and stays disqualified; track timing does not rescue
it. The new primary endpoint separately requires preregistered false/duplicate/
fragmented installed-track and benign map-pollution guardrails so a permissive
tracker cannot manufacture early knowledge.

## 4. Retention and runtime

The pilot measured about 2.75 MB of aligned inputs and 19.95 MB of logits per
role/frame. Continuous full retention for 330 trajectories would be roughly
1.8 TB and is forbidden.

The preserved v1 candidate planned:

- causal lightweight records, unfiltered detections, final detections, tracks,
  truth, ego state, actions, queue/network timestamps, and manifests for every
  trajectory;
- a 4 s aligned-input window for every calibration and validation trajectory;
- logits plus inputs for one calibration audit group per designed geometry and
  naturalistic route (15 world trajectories total); and
- no heavy raw window in the confirmatory test.

Using pilot-measured bytes, the v1 total estimate is **54.61 GB**, below the
hard **80 GB** design cap while preserving the existing 500 GB free-space
floor. The collector must still reserve space and enforce permits before every
write; this estimate is not permission to exceed a quota.

Pilot wall time was about 2.9 minutes per world trajectory. The v1 candidate
therefore estimates:

| Stage | World trajectories | Capture time estimate |
|---|---:|---:|
| Calibration | 66 | 3.2 h |
| Validation | 66 | 3.2 h |
| Confirmatory test | 198 | 9.6 h |
| **Total** | **330** | **16.0 h** |

Each stage uses a detached, self-logging runner and stops at its human gate.
No stage is chained into the next. These storage/runtime figures remain useful
planning bounds, but the 66/66/198 schedule is not launch authorization and
must be regenerated for the successor manifest.

### Historical v1 planned calibration-audit contract

The warning-era audit **plan** was defined in
`data_collection/configs/phase2_calibration_audit_v1.yaml`. It selects exactly
nine calibration groups from the immutable manifest: six designed matched
groups (positive plus benign) and three naturalistic groups, for 15 world
trajectories. Each trajectory runs for 120 frames at the native 10 Hz CARLA
clock under Epic rendering. Lightweight causal records span the full 12 s;
each role retains exactly 40 aligned input/logit pairs inside its reviewed 4 s
window. The expected heavy-data total is 27.24 GB, protected by a 3 GB
per-trajectory cap, an 80 GB stage cap, a 500 GB post-write free-space floor,
and a 580 GB preflight requirement. Automatic deletion is forbidden.

The full 15-trajectory plan did not complete. The accepted
`20260818_230028_audit` execution contains only three trajectories / two groups;
it is the completed structural subset described below. Do not infer
15-trajectory evidence from the historical plan or its storage estimate.

That v1 config is now provenance, not a runnable next-stage contract. The next
proposed unit is a separately versioned 16-trajectory factor-realization
tranche preassigned to calibration. Its rows must bind typed per-geometry
urgency/horizon parameters and realized gates before launch. If a row passes
capture, integrity, and realization gates, it counts toward calibration; this
avoids another disposable pilot. A failed/out-of-cell row is retained and does
not count or change labels. No launch is authorized until the successor config,
manifest, hashes, and analysis plan receive joint review.

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
retained source artifacts remain sufficient for the narrower historical
72-point warning-engine grid. That grid preserves the failed warning study; it
does not govern or select the future installed-track endpoint.

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
but does not realize the labelled speed/urgency cells and does not authorize
collection. Typed per-geometry factor controls and gates remain blocking.

Every new geometry needs the same manual review used for the pilot: legal lane
IDs/headings, no pose overlap, visible helper advantage for positive hazards,
matched benign equivalence, realistic motion, collision-free ambient traffic,
camera visibility, and clean actor teardown.

## 6. Staged gate sequence

1. **Complete:** both paired naturalistic routes are visually accepted and the
   shared pair-contract ID plus both route hashes are frozen.
2. **Complete, historical:** the warning-era audit/replay and the bounded
   decision-opportunity pilot established capture integrity and a local sensing
   opportunity, but the warning rule failed its unchanged specificity gates.
3. **Current design/runtime gate:** jointly review the checked-in v2 factor
   manifest and exact 16 calibration rows, then implement and verify the
   per-row runtime adapter, typed realized urgency/horizon gates,
   recipient-available installed-track event, exact anti-memorization feature
   projection, track-quality guardrails, and create-only launcher. The v1
   15/66/330 launch paths remain false, and v2 has no registered power claim.
4. If separately authorized, collect only the 16 preassigned calibration
   trajectories. Count every passing row toward calibration; stop on contract,
   out-of-cell realization, censoring, track/map-quality, or integrity failure.
5. Analyze local installed-track gain and mandatory actionability strata. Then,
   only after a separate human gate, verify accepted install and consumer
   `available_at` over two-UE OAI using identical contributions. Do not promote
   helper-local confirmation or raw install time to recipient knowledge.
6. Regenerate clustered power/counts for the new endpoint. Complete remaining
   calibration only if the result is supportable; validation and untouched test
   each require separate approval. Warning lead remains secondary/failed unless
   a genuinely new warning design is later preregistered; its old gates are not
   weakened.

LOCAL/OAI calibration and the causal controller ladder remain downstream.
Nothing in this proposal authorizes CARLA, OAI, a controller run, or RL.
