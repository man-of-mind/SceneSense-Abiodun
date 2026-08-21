# UE split-only baseline experiment plan

> **SUPERSEDED FOR CURRENT EXECUTION (2026-08-20).** This document preserves
> the earlier reuse/candidate-filtering design and its audit contracts. The
> supervisor discussion changed the experiment to a full 72-action,
> time-varying-network characterization. Use
> [`UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md`](UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md)
> as the current plan. Do not use the `N_total`, static-four-regime, replay-only,
> or no-install-ACK instructions below to launch new work.

**Status:** Reuse-only Stage-A audit and offline candidate proposal complete;
Abiodun--Codex candidate review and final action-catalog selection remain
pending. Supervisor discussion is welcome but is not an approval dependency.
This does not authorize a new CARLA/OAI run, ROI gap fill, controller training,
or runtime implementation.

**Decision date:** 2026-08-20

**Reading guide:** Sections 1--7 are the supervisor-facing plan. Sections
8--15 are the reproducibility and data-contract appendix for implementation.

**Concise supervisor handoff:**
[`UE_SPLIT_ONLY_SUPERVISOR_DELIVERABLE.md`](UE_SPLIT_ONLY_SUPERVISOR_DELIVERABLE.md)
summarizes the agreed scope, proposed 3-normal-plus-1-rescue shortlist, four
network regimes, measurements, and conditional two-cell starting plan. Its
companion
[`UE_SPLIT_ONLY_SUPERVISOR_COMBINATIONS_V1.csv`](UE_SPLIT_ONLY_SUPERVISOR_COMBINATIONS_V1.csv)
lists all 16 logical combinations without claiming that they have been
measured or authorizing a run.

## 1. Purpose and immediate deliverable

Build the smallest empirical data sheet needed to answer:

> Under a measured network condition, which registered split-model/profile
> bundle gives the best edge-map result without unstable queueing, excessive
> latency, or missed map updates?

This is a measurement baseline, not yet an agent-training experiment. We will
use the completed table to derive the first deterministic profile-selection
logic before considering `SKIP`, `LOCAL`, MPC, or a learned policy.

The deliverable has three explicit decision states rather than silently
inventing a quality floor or final action set:

1. this approved written experiment contract;
2. one audited 72-row existing SPLIT-profile evidence pool;
3. one `OBJECT_MAP_V1` quality-floor sensitivity table;
4. one four-regime network catalog and small transport-anchor evidence table;
5. one explicit unresolved-measurement list for owner/supervisor discussion;
6. after one absolute quality floor is selected, one immutable
   `CANDIDATE_REVIEW_REQUIRED` proposal separating normal candidates from any
   degraded-rescue candidate; then, only after the remaining evidence and
   catalog-equivalence decisions, one frozen `N_total`-row eligible action
   catalog and one composed `N_total`-profile x 4-regime logical state-action
   surface, with evidence provenance on every row; and
7. a short interpretation of feasible profiles and remaining evidence gaps.

The initial reuse assembly ends in `REVIEW_REQUIRED`: it has no final
`N_total`, no final action catalog or `N_total x 4` surface, and no
`COMPLETED.json`. The approved
quality floor is recorded in an immutable decision input. Its reuse-only
successor ends in `CANDIDATE_REVIEW_REQUIRED`, with 26 normal aggregate
candidates and one separately typed provisional rescue, all
`final_eligible=false`. Only a later owner-approved sibling may enter `FROZEN`
and write the final catalog/surface. None of these artifacts authorizes a
Cartesian live OAI sweep.

The completed Stage-A review is
`rl_agent/experiments/ue_split_stage_a_v1/20260820_024055_review`. The approved
floor is recorded in
`rl_agent/decisions/ue_split_object_map_v1_floor_v1.yaml`. Its validated
successor is
`rl_agent/experiments/ue_split_catalog_proposal_v1/20260820_042414_candidate`.
It is a candidate proposal, not a final catalog: `eligible_action_count`
remains null and every measurement remains unauthorized. Its manifest SHA-256
is `d88cbabeee74bf862d3b7743c0bf447fd08f42f9fcf13e119af7e2bb58191d42`.

## 2. Scope lock

### Included now

- One vehicle UE and one edge spatial-map endpoint.
- The deployed RGB-plus-radar fusion task and output schema.
- Registered end-to-end split-model/profile bundles as the only action axis.
- Fixed UE/edge hardware and processing allocation.
- OAI network regimes with measured radio and queue outcomes.
- The real bundle-specific decoder/tail and fixed publish-all map path.
- Application, transport, processing, accepted map-update, freshness, and
  perception-quality measurements.

### Explicitly deferred

- `SKIP_INFERENCE`, `LOCAL_INFER`, remote full-frame inference, or a learned
  action policy.
- Radar-risk action selection, semantic urgency, reward design, and RL.
- Helper/recipient cooperation, selective map sharing, occlusion reasoning,
  warnings, braking, and controlled-NPC choreography.
- Sionna or position-indexed channel maps.
- Variable compute power or multi-UE contention.

The UE's responsibility here is only to deliver its own split-inference
evidence to the edge map efficiently. Reasoning over the completed map belongs
to a later stage.

### Approved output-service decision

The current service contract is `OBJECT_MAP_V1`. The object head
already produces vehicle/person class, confidence, local/world position,
dimensions, yaw, parked-state score, radar-support score, and an optional 2-D
box. The current map server associates, fuses, and tracks those object records.

The semantic head independently produces a dense three-class
`{background, vehicle, person}` mask. The current publisher carries only a
per-class pixel-count summary to the map server; the dense mask is not consumed
by the current object-map fusion path. Segmentation therefore remains an
offline quality diagnostic and auxiliary model output in this baseline, not a
hard map-service requirement. It is not removed from the model or the logs.

The required v1 object record is deliberately narrow: source/capture identity,
a valid-empty-versus-missing distinction, confidence, vehicle-or-pedestrian
class, and predicted actor-reference world XY. The latter is the model's
learned object reference location, not a segmentation-mask centroid or an
independently guaranteed geometric-box center. The map may name it `location`;
the service contract calls it `world_location_xy`. The selected `profile_id`
is mandatory wire/provenance metadata, but it is not a semantic object-service
field.

Segmentation quality remains visible through macro mIoU plus vehicle and person
IoU on CARLA/offline ground truth. IoU is not a live model output because it
requires ground truth. A segmentation prediction or class summary may be
logged live, while its IoU is computed only in evaluation.

For a later task-utility formulation, pedestrian/vehicle detection and
source-time/map-time localization receive the primary service role.
Segmentation may receive a smaller secondary utility weight, but no numerical
weight is frozen before metric scales and trade-offs are reviewed. A future
consumer that genuinely requires dense semantic masks must declare a separate
service contract and action-quality mask before the UE decision.

Hard physical, freshness, and object-quality gates are applied before utility
ranking. Among profiles that pass those gates, the eventual descriptive
profile-ranking utility may take the form

```text
U_task = w_det * Q_det + w_loc * Q_loc + w_seg * Q_seg,
with w_seg lower than the object terms.
```

This is neither an RL reward nor a change to the model-training loss. The
metric definitions, normalisation, weights, and sensitivity remain a later
Abiodun--Codex decision after the data sheet is assembled, informed by the
supervisor discussion.

### Approved experimental quality floor

The following inclusive floor is approved for forming **normal aggregate
candidates**. It is a quality-preservation screen under the frozen evaluator,
not an autonomous-driving or deployment-safety certificate:

| Metric | Normal floor |
|---|---:|
| Vehicle recall | >= 0.90 |
| Pedestrian recall | >= 0.85 |
| Vehicle precision | >= 0.49 |
| Pedestrian precision | >= 0.61 |
| Vehicle world-XY MAE | <= 0.90 m |
| Pedestrian world-XY MAE | <= 1.20 m |
| False positives/frame | <= 1.45 |

Vehicle recall `0.91` is a preferred, non-gating target. Every normal candidate
must also pass the existing `prior_reference_exploratory` same-model,
same-quantizer, `q=0` incremental screen. Segmentation never participates in
this veto. Point estimates near a boundary remain visible for later paired
uncertainty review rather than being described as physically exact.

One separately typed `DEGRADED_RESCUE` candidate may be retained. It is
available only if no normal action is physically/network feasible, must pass
its own registered minimums and network mask, emits service debt, and never
counts as normal-quality success. The current proposal is `ae32/u4/q0.9`, with
a bounded pedestrian-recall minimum of `0.84`; it remains provisional pending
difficult-object review.

These floors preserve quality under the frozen offline evaluator. Because the
same 2,162-frame set informed candidate selection, it is catalog-development
evidence rather than an independent final test. Similar live performance is
not assumed: the final catalog requires a later bounded, independently
reported runtime parity/held-out validation before deployment claims.

## 3. Frozen baseline flow

```text
one retained aligned RGB+radar sequence
        -> selected registered bundle front + SPLIT(profile)
        -> actual serialized feature tensors
        -> new wall-clock 10 Hz monotonic replay release
        -> OAI uplink
        -> matching registered bundle decoder + tail
        -> fixed publish-all path
        -> edge validation + authoritative map update-done event
```

Every replay sample is offered once. There is no controller skip, local branch,
application retry, or learned publication decision. Lower-layer RLC/HARQ
queueing and retransmissions remain measured network behavior.

A UE-facing installation ACK is not needed for this baseline because the map
server can log accepted updates directly. A front-side publish or enqueue event
is **not** proof of installation. Each accepted result must be joined by
`stream_id` and `frame_id` to the map server's update-done record.

## 4. Sensor and motion provenance

- A source sample is the existing aligned three-channel RGB plus four-channel
  radar tensor.
- Current CARLA `ego_speed_mps` is calculated from the magnitude of the
  vehicle actor's world velocity returned by `actor.get_velocity()`. It is not
  an IMU measurement.
- CARLA target/object speeds and ground truth are evaluation-only. They are not
  profile-selection inputs in this baseline.
- CARLA radar supplies individual, unlabelled surface returns containing
  altitude, azimuth, depth, and radial velocity. Several returns can come from
  one object, and a return has no semantic object identity.
- The current fusion input aggregates these returns spatially into radar
  channels, while the raw per-return arrays can also be retained. This
  baseline does not reduce them into a TTC/risk policy signal.
- A real vehicle would later replace CARLA actor velocity with an available
  source such as CAN/wheel odometry or validated localization/IMU fusion. That
  deployment substitution is outside this experiment.

If radar later enters a `SKIP` decision or an aggressive-ROI guard, it will
require clustering and deduplication first. One registered `radar_risk` summary
would be based on the **minimum valid positive time-to-collision** among
approaching clusters, not the highest TTC and not one extreme raw return. That
later reducer must validate velocity sign, ego-motion effects, noise, and
processing cost. High or invalid radar risk may conservatively remove
aggressive-ROI profiles from the eligible set. Low radar activity cannot prove
that the current frame is empty or safe and cannot by itself authorize a
profile that failed its offline object-quality gate. This guard is a future
controller rule, not an experimental factor in the split-only baseline. Before
using it, the offline audit must explicitly include GT-positive/radar-quiet,
pedestrian, small/far-object, and active-radar strata so quiet or weak returns
do not become a false safety certificate.

## 5. Fixed controls

The following remain fixed across comparison cells:

- input modalities, class definitions, and output/map semantics;
- camera/radar calibration, preprocessing, input tensors, and retained frame
  ordering;
- a new wall-clock replay target of 10.00 Hz;
- UE and edge hardware allocation and background-load policy;
- fixed front/transport/tail interfaces and runtime placement, plus fixed map
  schema, TTL/expiry policy, and publish-all behavior;
- lossless `zstd` settings;
- OAI 106-PRB configuration and SINR-driven MCS policy;
- packetization/chunking contract;
- experiment duration, warm-up, timeout, and drop definitions; and
- software, configuration, model, input, and output hashes.

The integrated AE-32, AE-64, AE-128, and no-AE choices do not share one
identical trained checkpoint. Each action is therefore a registered end-to-end
split-model/profile bundle. `profile_id` binds all model weights, the split
point, encoder/decoder or AE bottleneck, quantization, codec, and ROI setting.
The input, class/output schema, hardware allocation, and map endpoint stay
fixed. Each complete bundle is byte-identical across all network regimes and
replicates. Results are attributed to the bundle, not to compression alone.

For an AE profile, use the integrated checkpoint that produced its registered
offline quality row. A legacy external AE checkpoint with a similar payload is
not interchangeable.

All four checkpoint families exist, but the current live runtime loads one
checkpoint and one fixed profile per process. This does not block the evidence
sheet or a later fixed-profile-per-launch boundary measurement. Simultaneous
UE/edge residency, model and codec warm-up, profile-tagged wire messages,
matching edge decoder/tail selection, GPU memory, and switch overhead are a
later controller-implementation gate before per-frame selection is claimed.

## 6. Experimental factors

### 6.1 Split-profile evidence pool and action catalog

The Stage-A evidence pool contains all 72 already measured bundles:

```text
4 model families {no-AE, AE-32, AE-64, AE-128}
x 3 quantizers {uint8, uint6, uint4}
x 6 measured ROI drop fractions {0.00, 0.30, 0.50, 0.70, 0.90, 0.98}
= 72 evidence profiles
```

The original 36-profile matrix is the subset with `q={0.00,0.30,0.50}`; the
later density study measured the additional 36 high-ROI profiles on the same
2,162-frame test split. Evidence-pool membership does not automatically make a
profile a registered policy action. Stage A verifies provenance and reports
`OBJECT_MAP_V1` ROI-incremental sensitivity. The approved absolute floor
produces `M_normal` aggregate normal candidates and, separately, up to
`R_rescue` provisional rescue candidates. Required difficult-object evidence
is resolved only for useful survivors. The later owner decision freezes
`N_normal <= M_normal` and `N_rescue <= R_rescue`, with
`N_total = N_normal + N_rescue`.
The exact enumerator evaluates all `N_total` before any smaller deployed
catalog is accepted. Convex-hull membership alone is not a sufficient pruning
rule.

The approved v1 floor produces `M_normal=26` aggregate normal candidates from
the 72-profile pool: 28 pass the absolute floor and 26 also pass the selected
same-`q=0` screen. One provisional rescue is tracked separately as
`R_rescue=1`; it is not added to `M_normal`, cannot be reported as `N=27`, and
cannot satisfy normal service. Exact eight-metric dominance removes only three
normal candidates, leaving 23 non-dominated point estimates. Consequently, a
small final `N_total` must not be invented without an approved equivalence or
catalog-budget rule.

The 72 profiles exhaust only this measured factorial design. They do not
exhaust possible AE widths, quantizers, ROI fractions, split points, codecs, or
radio configurations.

A smaller set is used only as **transport-calibration anchors**. The following
are provisional candidates, not the policy action space:

| Transport anchor | Approx. payload | Nominal payload rate at 10 Hz | Baseline role |
|---|---:|---:|---|
| one validated sub-90-KiB bundle | about 49--65 KiB | about 4.0--5.3 Mbps | severe-link fallback region |
| `ae32__uint4__roi0.0` | 90.0 KiB | 7.37 Mbps | compact zero-drop transport anchor |
| `ae128__uint4__roi0.0` | 129.2 KiB | 10.58 Mbps | compact higher-recall zero-drop anchor |
| `noae__uint4__roi0.0` | 392.0 KiB | 32.11 Mbps | large zero-drop quality reference |

One additional non-selectable stress/reference profile is retained:

| Profile | Approx. payload | Nominal payload rate at 10 Hz | Use |
|---|---:|---:|---|
| `noae__uint8__roi0.0` | 1050.3 KiB | 86.04 Mbps | existing non-selectable stress reference |

The stress profile is not rerun unless Abiodun--Codex explicitly authorize a
saturation demonstration; supervisor input is optional and nonblocking. Its
nominal rate already exceeds the observed clear capacity by a wide margin.
After `N_total` is frozen, choose the smallest anchor set that brackets useful
payload-capacity knees. An eligible exact profile is
preferred; a shaped or legacy substitute is labelled `payload_proxy`, never an
exact action measurement. Freeze that set before any new network measurement.
`zstd` remains fixed at level 3.

These payload rates exclude transport overhead and are not feasibility verdicts.
They are used only to identify which measured capacity boundaries need direct
10-Hz validation.

`zstd` is lossless. The AE bottleneck and integer quantization are the lossy
profile transformations and are included in the offline quality comparison.

#### ROI meaning and bounded expansion

The parameter historically named `roi_threshold` is a **drop fraction**, not a
semantic box threshold: `q=0.30` rank-drops the lowest-objectness 30% of feature
cells. The runtime accepts arbitrary `q` in `[0,1)`, and all four integrated
families were trained with random `q` in `[0,0.8]`. That training provenance
makes `q=0.10` or `q=0.15` plausible, but it does not make them measured actions.

Existing integrated evidence already covers the higher fractions
`{0.70, 0.90, 0.98}`. They are part of the 72-row evidence pool, not
automatically selectable actions. High ROI drop can preserve object detection
while severely damaging dense segmentation. Under `OBJECT_MAP_V1`, segmentation
is reported as a secondary diagnostic rather than used as a hard veto, while
an aggressive-ROI profile may enter `N_normal` only after it passes held-out,
class-specific pedestrian/vehicle recall, precision/false-positive, and
localization gates, including difficult non-empty strata. A profile that works
only when a post-hoc label says the frame was empty is not a causal v1 action.

Do not run the proposed `q={0.10,0.15}` gap fill in Stage A. First audit the
existing 72 profiles. If the completed object-quality/network sheet identifies
a specific useful payload-quality interval around a measured capacity knee, a
bounded offline same-sample gap fill may be proposed. It still requires a
separate explicit Abiodun--Codex authorization. Any accepted value creates a
versioned successor evidence pool, repeats the integrity/quality audit, and
regenerates `N_total` and the logical surface; it is never silently appended to
v1.

ROI remains categorical in v1. A continuous or hybrid ROI action is deferred
until a measured curve demonstrates reliable interpolation at held-out `q`
values. Finite rank-drop, entropy coding, and perception quality make the
response stepwise and potentially nonlinear, so an arbitrary continuous action
must not receive interpolated reward without validation. Any accepted ROI
expansion creates a versioned successor catalog and regenerates the logical
profile-by-network surface.

### 6.2 Network regimes

Use the four existing calibrated, static OAI AWGN regimes. They are balanced
experimental conditions, not a deployment channel distribution. The
per-sample latency, delivery, retransmission, and queue outcomes within each
regime form the empirical outcome distribution.

| Presentation regime | Historical config ID | Existing achieved PUSCH SNR | Existing served ceiling | Interpretation |
|---|---|---:|---:|---|
| clear | `clear` | about 50.3 dB | about 36.7 Mbps | high-capacity reference |
| mild | `mild` | about 19.5 dB | about 27.8 Mbps | moderate degradation |
| mid | `mid15` | about 15.6 dB | about 19.7 Mbps | constrained but usable |
| **poor** | `strong` | about 8.2--8.5 dB | about 9.2--10.4 Mbps | poor/worst tested link |

The old name `strong` meant strong AWGN impairment, not a strong link. We use
`poor` in reports and retain `strong` only as the immutable historical config
identifier. The commanded regime is configuration provenance. Achieved gNB PUSCH SNR,
MCS, grants, PRBs, BSR, and retransmissions are the measurement truth.

Retain these four regimes as broad calibration anchors, then add conditions
only on either side of unresolved MCS or payload-feasibility knees. The initial
priority thresholds are the achieved-SNR regions around 9.8, 11.8--12.4, 16.2,
20.2--21.2, and 24.5 dB. Do not run a dense evenly spaced SNR sweep: above
24.5 dB this radio is already capped at MCS 28, so more SNR alone does not
increase its fixed-resource service ceiling. Capacity/queue behavior, not SNR
or a regime label alone, is the eventual controller-relevant quantity.

### 6.3 Four-artifact combination contract

The frozen analysis contains four different artifacts:

```text
A. 72-row audited SPLIT-profile evidence pool
B. sensitivity -> absolute floor -> M candidates -> difficult-evidence gate -> N eligible actions
C. K transport anchors x 4 broad network regimes
D. N registered actions x 4 regimes = 4N logical policy rows
```

The initial review assembly produces A, the sensitivity table, the network and
transport evidence, and unresolved rows. It does not emit B or D until the
quality-floor decision is recorded. The later logical rows do **not** mean a
Cartesian live experiment. Populate them from existing profile, channel, and
staleness evidence and label each as one of:

```text
direct_new | existing_reused | payload_proxy | composed |
monotonic_inference | unresolved
```

The current capacity ordering predicts broad feasible sets, while the complete
catalog preserves intermediate payload/quality choices inside those sets:

- poor link: 90 KiB should fit; 129 KiB is on the boundary; 392 KiB should not
  fit;
- mid/mild links: 90 and 129 KiB should fit; 392 KiB should not fit, with mild
  closest to its boundary; and
- clear link: bundles through approximately 392 KiB may fit, with the largest
  actions closest to the boundary.

Only `unresolved` or decision-boundary rows receive new measurements. A result
may fill another row by monotonic inference only when the already observed
capacity ordering supports it, and the derived row must retain its evidence
source rather than being labelled measured.

For a directly measured cell used to establish a final breakpoint, use three
independent replicate blocks with at least 200 post-warm-up releases per block:
at least 600 samples. Report each replicate separately. For a pooled descriptive
95% interval, use a moving-block bootstrap with contiguous 10-sample (1-s)
blocks nested within replicates, plus a 20-sample block-length sensitivity.
Do not claim that three whole-run values alone define a precise 95% interval.

### 6.4 Environment and vehicle movement

- Reuse one representative, single-UE Town10HD sequence with natural scene and
  vehicle movement. Do not run CARLA independently for every network cell.
- Replay the identical retained frame sequence and ordering in every paired
  comparison.
- Do not add controlled interacting NPCs, helper/recipient actors, or an
  occlusion scenario.
- Record source ego/object motion and ground truth only for evaluation strata.
  They do not select the split profile in this baseline.
- If no suitable retained sequence exists, record that as an evidence gap. A
  short replacement collection requires separate explicit Abiodun--Codex
  authorization; plan approval alone is not run authority and no scenario
  corpus is implied.

An optional later validation may replay a few representative choices on one
simple live route. It is not part of the minimum baseline acceptance gate.

## 7. Execution stages

### Stage A — reuse and integrity audit

1. Verify the existing 72-profile same-sample payload/quality evidence pool,
   including its original 36-profile subset.
2. Produce the `OBJECT_MAP_V1` class-specific quality-floor sensitivity table;
   leave final eligibility pending owner review and do not run a new ROI gap
   fill.
3. Resolve the four model-family checkpoint hashes and every registered profile
   binding.
4. Record one-fixed-profile-per-launch as sufficient for a later measurement;
   defer simultaneous residency and switching implementation.
5. Verify the existing four-regime OAI surface and achieved radio metrics.
6. Freeze and validate the retained train/validation/test identifier manifests.
   Keep the ordered transport replay sequence explicitly unselected until
   `N_total` is known; when selected later, give it a separate role ID/hash and
   record any overlap rather than conflating it with the quality set.
7. Produce the evidence pool, quality sensitivity, four-regime catalog,
   transport evidence, and unresolved-measurement sheet, including nominal
   10-Hz load, evidence status, provenance, and uncertainty.
8. Validate the Stage-A assembler fail-closed and publish its manifest with
   `decision_state=REVIEW_REQUIRED`, `quality_floor_id=null`, and
   `eligible_action_count=null`. Do not write a final action catalog, logical
   surface, or `COMPLETED.json`.
9. Stop for Abiodun--Codex review before any new run; collect supervisor input
   when available, but it is nonblocking. Every run requires a separate
   explicit Abiodun--Codex authorization.

After one absolute quality floor is approved, create a new immutable sibling
candidate output that references the review-manifest hash, records `M_normal`
and `R_rescue` separately, and remains `CANDIDATE_REVIEW_REQUIRED`.
Resolve required difficult-object evidence only for decision-relevant
survivors; then freeze `N_normal <= M_normal` and
`N_rescue <= R_rescue`, write the `N_total`-row eligible catalog and exact
`4N_total`-row logical surface, and recompute the unresolved list. A rescue is
always labelled separately and never counted as normal-quality service. If
`N_total=0`, stop without inventing an action or surface.

Current evidence supports exact frame/density strata for every candidate and
horizontal-range recall/localization for the pinned `q<=0.5` diagnostic
profiles. Exact small-object recall remains unavailable: FN rows lack GT box
size and the frozen source `object_boxes.csv` is absent. This limitation is
reported rather than repaired with a broad rerun. High-ROI `q=0.7/q=0.9`
profiles additionally lack retained per-object match rows and remain
provisional where that evidence is required.

The existing approximately 90-KiB OAI cell used a legacy external AE-32
checkpoint. Reuse it as payload/network evidence only, not as an exact
end-to-end validation of the integrated AE-32 profile.

### Stage B — paired profile characterization

Reuse existing same-input characterization for every registered profile. Rerun
only a profile with a checkpoint, ROI, compute, or provenance gap. Generate and
preserve the **actual serialized tensors** for the transport anchors that Stage
C will replay. Measure or verify:

- payload and on-wire size before network effects;
- pedestrian/vehicle precision, recall, false positives, and localization
  quality;
- secondary segmentation mIoU;
- UE front, profile transform, quantization, and serialization time; and
- matching decode and edge-tail time.

Remeasure AE compute with the registered integrated checkpoint where historical
loopback evidence used the external checkpoint.

### Stage C — adaptive fixed-rate OAI boundary checks

Release the actual Stage-B serialized tensors on a new monotonic schedule
targeting 10.00 Hz. Send them through OAI, the matching profile decoder, the
matching registered bundle tail, and the map server.

- Never run the full registered-profile x SNR Cartesian product.
- Begin with one diagnostic block for each transport-anchor boundary cell that
  remains unresolved after Stage A, including one validated sub-90-KiB
  fallback, the 90- and 129-KiB poor-link boundary, and the 341--392-KiB
  mild/clear boundaries where applicable.
- Add `mid x 129 KiB` only if the poor-link 129-KiB result fails or remains
  borderline and the next feasible transition is not established.
- Replicate only the directly measured cells needed for a final breakpoint.
  Expand to an adjacent cell only if a measured result contradicts the
  registered monotonic capacity ordering.
- Do not rerun the 1050-KiB stress reference unless separately requested.
- A directly measured cell is invalid if the producer cannot sustain 9.9--10.1
  releases/s without growing producer-side release lag.
- Network-side queue growth after valid release is an experimental outcome.
- A continuously growing or saturated BSR queue is classified as
  overload/infeasible, not averaged into a stable latency point.
- Give every cell/replicate a unique `stream_id` and every offered sample a
  strictly increasing `replay_sequence_id`. Before warm-up, restart or isolate
  the map-server namespace and clear all prior streams, sequence registries,
  fused tracks, counters, and queues. An empty-state gate must pass before the
  first warm-up release so another cell cannot contaminate the result.
- Accept a map update only when its replay sequence is newer than the newest
  accepted sequence for that stream. Reject duplicate or older late arrivals
  without replacing map state, and log their status and reason.

Stage C directly measures replay-release-to-map-update latency. The complete
service estimate is a clearly labelled composition of paired Stage-B front and
serialization timing plus Stage-C release-to-update timing. It is not
misrepresented as one live end-to-end timestamp path.

A synthetic shaped-byte sender may be used only for an optional transport-only
knee diagnostic. It cannot produce decoder/tail, map-update, or perception
quality evidence.

### Stage D — data-sheet composition and decision

Join profile quality, radio outcomes, queue behavior, and authoritative map
update-done events. Publish the registered action catalog, transport-anchor
table, and complete logical profile-by-regime surface with evidence status and
source; never present a composed or monotonic-inference row as directly
measured.

Only if that table leaves a material sequential queue-recovery question may an
optional `clear -> constrained -> clear` trace be proposed for profiles near
the measured knee. It is not part of the initial deliverable and still
requires separate explicit Abiodun--Codex authorization.

### Stage E — optional bounded live validation

After the data sheet is reviewed, a few table predictions may be proposed for
validation on one simple live CARLA route. Execution requires separate explicit
Abiodun--Codex authorization. It is not a new training corpus and cannot
silently expand into complex NPC or cooperation work.

## 8. Timestamp and map-update contract

The replay has two deliberately separate time domains:

- `source_capture_sim_time_s` and original `source_frame_id` are immutable
  CARLA/evaluation provenance; and
- `replay_release_monotonic_ns` starts a new capture-equivalent service clock
  for each replayed sample.

Never subtract CARLA simulation time from a host monotonic clock. All latency
and AoI arithmetic uses one verified monotonic domain, or a measured clock
offset when components do not share that domain.

Record these events where applicable:

- `source_capture_sim_time_s`, `source_frame_id`;
- Stage-B `front_start`, `front_done`, and `serialize_done`;
- Stage-C `replay_release`, `application_send`, `edge_receive`, `tail_done`;
- map-server `map_update_done`; and
- timeout/drop classification time.

The authoritative accepted-update event is the map server's update-done record
joined by stream and frame identity. Front publish/enqueue is only an upstream
stage timestamp.

The offline evaluator reconstructs the map at a fixed 10-Hz query cadence. At
each scheduled replay-release instant, immediately before the new sample is
released, it selects the newest previously accepted update for that stream and
records `query_monotonic_ns`. Time above a freshness threshold is also
integrated over the exact intervals between accepted updates, with expiry at
the frozen map TTL. AoI results therefore do not depend on analyzer polling
speed.

For an accepted replay result, service AoI at query time is:

```text
map_aoi_s(t)
  = query_monotonic_time
    - replay_release_time_of_newest_accepted_map_update
```

Before the first accepted map update, AoI is `unavailable`, not zero. Following
a drop or delay, the previous accepted result remains newest and its AoI grows
until the fixed, logged map TTL expires it. A later live run uses actual live
capture monotonic time in place of replay release time.

## 9. Measurements

### Radio, transport, and delivery

- application payload and on-wire bytes;
- target and achieved release rate, offered Mbps, and producer release lag;
- map-update accepted/drop/timeout counts and rates;
- achieved PUSCH SNR, MCS, PRBs, grants, scheduled UL rate, BSR/backlog, BLER,
  and HARQ/RLC behavior; and
- overload onset and queue-recovery time when applicable.

### Processing, latency, and freshness

- UE front/profile/serialization, application transport, decoder/tail, and map
  update stage timing;
- measured replay-release-to-map latency;
- explicitly composed full service latency;
- n, p50, p90, p95, every replicate's p95, the registered nested moving-block
  bootstrap interval and block-length sensitivity, and maximum;
- deadline-miss and delivery count/rate, every replicate's rate, and the same
  nested moving-block bootstrap interval used for correlated latency samples;
- accepted map-update rate, AoI distribution, and time above each provisional
  freshness threshold.

P95 is not a standalone pass/fail statistic, and one maximum is not renamed
p95. The primary operational evidence is stable delivery/queue behavior and
deadline-miss rate under adequate samples.

### Perception and map quality

- pedestrian and vehicle recall, precision/false positives, and class-aware
  failure counts as the primary `OBJECT_MAP_V1` quality evidence;
- object localization error at source time;
- time-aligned localization error at map query time;
- consequences of holding the prior map state after a dropped/delayed update;
  and
- segmentation mIoU as a secondary diagnostic, not a hard baseline service
  gate.

Ground truth is joined after measured outcomes for evaluation only.

## 10. AoI analysis policy

The experiment measures the age/error relationship; it does not silently
declare one universal deployment `AoI_max`.

Use the registered sensitivity model:

```text
error(v, age, profile)
  ~= sqrt(base_localization_error(profile)^2 + (v * age)^2)
```

Report provisional localization tolerances of 1.5, 2.0, 2.5, and 3.0 m.
Abiodun--Codex select the acceptable service tolerance after reviewing the
trade-off with the supervisor. The corresponding age budget is then derived by
movement regime.
The earlier 2.0 m value remains a reference, not a deployment guarantee.

## 11. Data-sheet contract

Write a new immutable timestamped experiment directory containing the resolved
config, source/model/config hashes, processed tables, and artifact manifest.
Do not combine logical cells and direct-run replicates in one ambiguous matrix.

### `ue_split_evidence_pool.csv`

Exactly 72 rows, one per measured offline profile:

```text
evidence_pool_version, profile_id, display_profile_id, model_family,
checkpoint_sha256, quantization_mode, roi_drop_fraction,
entropy_coder, entropy_level,
quality_set_id, quality_frame_count,
payload_bytes_mean, payload_bytes_p95,
recall_vehicle, recall_pedestrian,
precision_vehicle, precision_pedestrian, fp_per_frame,
xy_mae_m, xy_mae_vehicle_m, xy_mae_pedestrian_m,
miou, iou_background, iou_vehicle, iou_person,
provenance_status, evidence_source_id, exclusion_reason
```

### `ue_split_quality_floor_sensitivity.csv`

One row per evidence profile and exploratory ROI-incremental tolerance. The
initial table compares `q>0` with the same-model/same-quantizer `q=0` row. It
reports class-specific deltas, but it is not an absolute service-quality floor
and cannot make a profile eligible. The approved freeze must add absolute
vehicle/pedestrian precision, recall, localization, and difficult-object gates:

```text
profile_id, sensitivity_floor_id, quality_floor_config_sha256,
screen_basis, roi_incremental_screen_pass,
vehicle_recall_drop, pedestrian_recall_drop,
vehicle_precision_drop, pedestrian_precision_drop,
vehicle_localization_increase_m, pedestrian_localization_increase_m,
absolute_object_quality_gate_status, difficult_object_gate_status,
quality_gate_status, quality_gate_reason, final_eligible
```

`ue_split_quality_strata.csv` is the corresponding long-form all-72 table for
density and vehicle/pedestrian-positive frame strata. These are offline
evaluation strata, never current-frame policy inputs.

### Candidate-proposal artifacts

After the floor decision, a separate offline assembler writes a new immutable
`CANDIDATE_REVIEW_REQUIRED` bundle. Its principal tables are:

- `ue_split_absolute_quality_gate.csv`: all 72 profiles and every inclusive
  absolute/incremental gate result;
- `ue_split_candidate_catalog.csv`: `M_normal` aggregate candidates plus any
  separately typed provisional rescue, all with `final_eligible=false`;
- `ue_split_audit_priority_shortlist.csv`: a nonbinding evidence-review
  shortlist, not the final action space;
- `ue_split_candidate_quality_strata.csv`: retained exact density/class strata;
- `ue_split_range_audit.csv` and `ue_split_range_comparison.csv`: horizontal
  camera-to-GT range diagnostics for pinned per-object evidence; and
- `ue_split_candidate_profile_regime_screen.csv`: exactly four provisional,
  unauthorized regime rows per candidate.

The proposal must omit `ue_split_action_catalog.csv`,
`ue_split_profile_network_surface.csv`, `FROZEN.json`, and `COMPLETED.json`.
Its `CANDIDATE_REVIEW_REQUIRED.json` and manifest record 26 normal aggregate
candidates, one separately typed provisional rescue,
`eligible_action_count=null`, `measurement_authorized=false`, and no-run
authority. Every candidate has `final_eligible=false`. The rescue count is
never folded into the normal-candidate count.

### `ue_split_action_catalog.csv`

This file is absent from both the initial `REVIEW_REQUIRED` assembly and the
`CANDIDATE_REVIEW_REQUIRED` proposal. A later immutable freeze output writes
exactly `N_total` rows after the owner-approved difficult-evidence and
catalog-budget decisions:

```text
action_index, profile_id, action_tier, evidence_pool_version, service_contract_id,
selected_quality_floor_id, checkpoint_sha256,
quantizer, roi_drop_fraction, codec, codec_level,
object_quality_gate_pass, profile_eligible, evidence_source_id
```

### `ue_split_network_regimes.csv`

Exactly four rows, one per presentation regime, with the historical config ID,
registered achieved-radio evidence IDs, and whether the evidence is direct,
reused, or unresolved.

### `ue_split_transport_evidence.csv`

One row per transport cell and replicate. This table owns replicate IDs,
release-rate validity, achieved radio/queue measurements, and direct versus
proxy provenance; it is not multiplied into the logical surface.

### `ue_split_profile_network_surface.csv`

This file is absent from the initial `REVIEW_REQUIRED` assembly and the
`CANDIDATE_REVIEW_REQUIRED` proposal. The freeze output contains exactly one
row for every eligible profile and broad regime: exactly `4N_total` unique
`(profile_id, network_regime)` keys.

```text
cell_id, profile_id, network_regime, target_offer_hz,
service_contract_id, selected_quality_floor_id,
object_quality_gate_pass, network_feasibility_status,
profile_evidence_source_id, network_evidence_source_id,
staleness_evidence_source_id, derivation_rule_id,
evidence_status, directly_measured, uncertainty_status, exclusion_reason
```

One row-level source string is not sufficient for a composed cell. A
`directly_measured=true` row requires the exact registered checkpoint/profile,
a valid 10-Hz producer, achieved-radio evidence, matching edge decoder/tail,
and authoritative map update in the same run. The legacy approximately
90-KiB external-AE cell remains `payload_proxy`.

The initial review may also contain
`ue_split_profile_regime_screen.csv` and
`ue_split_boundary_candidates.csv`. They compare profile P95 feature payloads
plus an explicit custom-header/UDP/IPv4 estimate with the historical capacity
projection. Lower-layer GTP/PDCP/RLC/MAC overhead remains unknown. Every row is
`UNRESOLVED`, every candidate is unauthorized, and neither file is the final
`4N_total` logical surface.

### `ue_split_unresolved_measurements.csv`

One row per named evidence gap, including why it affects a decision boundary,
the smallest proposed measurement, and its approval state. This is the only
sheet from which a new run may be proposed.

The reuse audit also writes:

- `ue_split_latency_tolerance_proxy.csv`, the historical fixed-floor
  localization/latency sensitivity. It is not direct AoI evidence and is not
  profile-specific;
- `ue_split_staleness_latency_anchors.csv`, direct or registered historical
  capture-to-`map_update_done` loopback anchors; and
- `ue_split_staleness_error_sensitivity.csv`, the retained localization-error
  response to imposed information lag.

None selects `AoI_max`. A later transport run must derive AoI from authoritative
accepted-update events.

### Later direct-measurement artifacts

`ue_split_frame_metrics.parquet` contains one row per offered Stage-C replay
sample. `ue_split_cell_summary.csv` contains one row per directly measured
cell/replicate summary. Stage-B compute joined into a Stage-C replay row is
marked `composed`, not falsely labelled as simultaneous live measurement.

### Manifest and decision-state contract

`manifest.json` records at least:

```text
schema, experiment_id, created_utc, audit_state, decision_state,
verdict,
service_contract{service_contract_id, segmentation_role, quality_floor_id,
                 quality_floor_config_sha256, approval_reference},
factor_contract{evidence_pool_version, expected_profile_count,
                actual_profile_count, network_regimes,
                eligible_action_count, expected_logical_surface_rows},
datasets{quality_set{quality_set_id, split, sample_count,
                     ordered_sample_ids_sha256,
                     training_disjointness_status},
         transport_replay{sequence_id, ordered_sample_ids_sha256, status}},
repository{git_commit, dirty, repository_diff_sha256,
           assembler_path, assembler_sha256, resolved_config_sha256},
inputs[{artifact_id, role, path, sha256, bytes, rows, schema_sha256,
        provenance_status}],
outputs[{artifact_id, path, sha256, bytes, rows, schema_id}],
evidence_counts, unresolved_count,
audit{verdict, tests, fatal_errors, warnings}
```

The initial assembly must say `decision_state=REVIEW_REQUIRED`, use a null
quality floor and null eligible count, omit the final action catalog and
logical surface, and omit `COMPLETED.json`. The candidate sibling must say
`decision_state=CANDIDATE_REVIEW_REQUIRED`, bind the parent manifest and floor
decision, keep `eligible_action_count=null`, record normal and rescue counts
separately, make all measurements unauthorized, and omit the final catalog,
surface, `FROZEN.json`, and `COMPLETED.json`. A later freeze output may say
`FROZEN` only when its recorded floor and catalog-budget decisions are
approved, its catalog has exactly `N_total` rows, its surface has exactly
`4N_total` rows, and every artifact hash validates.

### Fail-closed assembler acceptance

- Require four primary per-frame files with 38,916 data rows each and 155,664
  profile-frame rows in total: 4 models x 3 quantizers x 6 ROI fractions x
  2,162 identical sample IDs. Missing, duplicate, or extra factor/sample keys
  are fatal integrity failures.
- Require the same sample/frame and GT counts across profiles, positive payloads,
  nonnegative count/confusion fields, and internally consistent TP/FP/FN
  arithmetic. Recompute aggregates from the per-frame rows.
- Require exact evaluation settings and checkpoint hashes. Missing or uncertain
  provenance can remain visible as `unresolved` but can never yield
  `profile_eligible=true`.
- Require finite primary metrics and adequate class/stratum support. An unknown
  or failed required object-quality gate cannot pass. Segmentation is logged
  but does not veto under `OBJECT_MAP_V1`.
- Without an approved `quality_floor_id`, never infer an eligible action count,
  publish a final catalog/surface, or emit a completion marker.
- In a freeze output, require the action catalog to contain exactly the eligible
  evidence-pool IDs and the logical surface to equal their exact Cartesian
  product with the four regimes. Orphan, ineligible, missing, or duplicate keys
  are fatal.
- Require a registered evidence status and component source IDs on every
  logical row. Commanded regime, nominal payload rate, or a proxy alone cannot
  be reported as a direct feasibility measurement.
- Write to a fresh temporary directory and finalize atomically only after all
  declared output row counts and SHA-256 hashes validate. A failed assembly may
  write a diagnostic report but never presents partial tables as final.

## 12. Questions the baseline must answer

1. Which registered profiles are physically feasible in each achieved network
   regime at 10 Hz?
2. Which measured profiles bracket each regime's payload-capacity knee?
3. Which physically feasible profile maximizes `OBJECT_MAP_V1` utility after
   satisfying hard object-quality and freshness gates?
4. How do delay and missed updates change map AoI and time-aligned localization
   error?
5. Does a simple lagged-capacity-to-payload lookup explain the choices, or is
   additional state-conditioned policy logic likely to add value?
6. Under the proposed object-map service, which aggressive-ROI profiles retain
   acceptable pedestrian/vehicle detection and localization, and which must be
   masked despite their payload saving?

## 13. Existing evidence and smallest missing measurement

Reuse first:

- `rl_agent/density_knob/raw/perframe_{noae,ae32,ae64,ae128}.csv` as the
  primary 72-profile per-frame payload/object/segmentation evidence;
- `rl_agent/density_knob/raw/eval_settings.json`,
  `rl_agent/density_knob/raw/analysis_settings.json`, and
  `rl_agent/density_knob/raw/gate_report.txt` for the registered evaluation
  settings, analysis assumptions, and prior integrity results;
- `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md` only as an independent reproduction
  check for the overlapping 36 profiles;
- `channel_condition_sweep/combined_surface.csv` and
  `channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` for the existing three
  payload anchors across four OAI regimes;
- `staleness/uplink_only_latency_budget/results/` for the age/error
  sensitivity; and
- one retained single-UE frame sequence selected during Stage A.

The sources are complementary rather than interchangeable:

| Evidence source | Already answers | Does not yet answer |
|---|---|---|
| 72-profile per-frame pool | bundle -> payload and held-out object/segmentation outcomes | exact 10-Hz network feasibility and accepted map update |
| 36-profile knob matrix | independent overlap reproduction and ideal-loopback compute | high-ROI action eligibility or exact OAI feasibility |
| Existing channel sweep | payload x radio regime -> delivery, latency, queue behavior at about 5.8--8.0 fps | exact registered-bundle outcome at wall-clock 10 Hz |
| Staleness study | speed x information age -> localization-error sensitivity | which bundle/network action caused that age |
| Joined split baseline | network context x bundle -> delivery, accepted update, AoI, installed quality | later SKIP/LOCAL outcomes or sequential policy value |

The joined sheet is useful to the agent work because it creates the first
auditable state-action-outcome surface:

```text
evaluation context: measured network regime/capacity
action:             SPLIT(registered bundle)
outcome:            queue, delivery, latency, accepted update, AoI, quality
derived label:      feasible/infeasible and best feasible bundle
```

It first supports an action mask, deterministic lookup, and table-driven
environment. It is not yet an RL replay buffer. A deployed policy later sees a
causally available lagged capacity/queue estimate, not the commanded regime or
same-sample achieved SNR used here as experimental truth.

The first gap is not a run: it is the audited 72-profile evidence pool,
quality-floor sensitivity, four-regime/transport evidence, and explicit
unresolved list. The initial assembly stops in `REVIEW_REQUIRED`. After the
absolute floor and difficult-evidence gates freeze `N_total`, its `4N_total`
logical surface determines whether any correctly paced 10-Hz OAI boundary
check should be proposed. Any such check still requires separate explicit
Abiodun--Codex authorization and uses exact registered split tensors, a
frame-aware edge path, timing/queue diagnostics, and authoritative accepted
map updates.

The existing OAI surface ran at approximately 5.8--8.0 offered frames/s despite
a nominal 10-Hz schedule, used one run per cell, and lacks usable semantic/map
quality rows. Its 10-Hz budgets are therefore projections. Its approximately
90-KiB anchor also used the legacy external AE-32 checkpoint, so it is
payload/network evidence rather than exact integrated-profile evidence.

## 14. Review and acceptance checklist

- [x] Abiodun and Codex approve this reuse-only scope, `OBJECT_MAP_V1`, and the
  absolute floor. Evidence:
  `rl_agent/decisions/ue_split_object_map_v1_floor_v1.yaml`, 2026-08-20.
- [ ] Abiodun--Codex complete the required difficult-object/catalog-budget
  decisions before an `N_total`-action catalog is frozen. Supervisor discussion
  is useful but nonblocking.
- [x] The initial manifest is `REVIEW_REQUIRED`, has no selected quality floor
  or eligible count, and publishes no final action catalog, logical surface, or
  `COMPLETED.json`. Evidence: `20260820_024055_review/manifest.json`.
- [x] The primary evidence pool contains exactly 72 unique profiles and 155,664
  complete profile-frame keys on the identical 2,162-sample quality set.
- [x] The four model-family checkpoint hashes and all registered profile
  bindings are resolved before a profile is eligible.
- [x] The live one-profile-per-launch limitation is recorded and does not block
  the evidence sheet; simultaneous residency and switching remain a later
  controller-implementation gate.
- [x] The existing 72-profile evidence pool is audited before any new ROI
  value is evaluated; `q={0.10,0.15}` remains conditional on a named gap.
- [x] `OBJECT_MAP_V1` is the current owner-approved service contract: object
  detection/localization are primary and segmentation is a logged secondary
  diagnostic rather than a hard profile gate. Supervisor feedback may refine
  the later final catalog without silently replacing this decision.
- [ ] Any aggressive-ROI action passes held-out, class-specific object-quality
  gates without relying on current-frame density or other post-action labels.
- [x] The approved aggregate floor is hash-bound in
  `rl_agent/decisions/ue_split_object_map_v1_floor_v1.yaml`; segmentation is
  non-vetoing, `M_normal` and rescue are separate, and no measurement or final
  freeze authority is granted.
- [x] The validated candidate proposal records 26 normal candidates and one
  separate provisional rescue, all `final_eligible=false`, with no measurement
  authority or final artifacts. Evidence:
  `20260820_042414_candidate/manifest.json`.
- [ ] The candidate proposal is reviewed; incomplete small-object/high-ROI
  evidence and an equivalence/catalog-budget rule are resolved or explicitly
  bounded before final `N_total` is selected.
- [ ] The held-out quality set and ordered transport replay sequence have
  separate role IDs and hashes; any overlap is recorded rather than conflated.
- [ ] Radar-risk remains outside the split-only experiment factors; its later
  use is limited to a conservative aggressive-ROI veto, with invalid/unknown
  radar defaulting to the stricter mask.
- [ ] Identical retained inputs are used for every paired comparison.
- [ ] All non-profile/non-network factors are fixed and logged.
- [ ] Achieved radio measurements are recorded; commanded labels alone do not
  qualify a cell.
- [ ] After the floor decision, the frozen catalog contains exactly `N_total`
  eligible evidence-pool IDs and the logical surface contains exactly
  `4N_total` unique profile/regime keys, with normal and rescue counts separate.
- [ ] Every logical profile/regime combination has an explicit evidence status
  and component source IDs; no composed/inferred row is presented as directly
  measured.
- [ ] Every directly measured cell used for a final breakpoint has three
  replicate blocks and at least 600 valid post-warm-up releases.
- [ ] Directly measured cells sustain 9.9--10.1 producer releases/s without
  producer-side lag growth.
- [ ] Queue-growing cells are labelled overload/infeasible, not averaged into
  stable latency.
- [ ] Every accepted result joins to a map-server update-done record.
- [ ] Unique stream IDs, strict replay-sequence acceptance, a full isolated
  empty-map reset, and duplicate/late rejection prevent cross-cell or
  out-of-order map regression.
- [ ] AoI is evaluated on the registered 10-Hz query grid and exact
  accepted-update intervals, with TTL expiry.
- [ ] Static profile quality is not mistaken for installed quality after
  delay/drop.
- [ ] No SKIP, LOCAL, radar policy, reward, RL, helper/recipient logic, or
  complex NPC scenario is introduced.
- [ ] Every new run closes a named evidence gap.
- [ ] No dominated/interior cell is rerun unless a boundary result invalidates
  the registered monotonic inference.
- [ ] The complete combination sheet and provenance manifest exist before any
  profile-selection policy is implemented.

## 15. Work unlocked after the table

Only after review of the completed data sheet:

1. run an exact feasibility-masked enumerator over the complete registered
   catalog and compare it against any proposed reduced catalog;
2. derive a deterministic network-to-profile lookup and exact greedy baseline;
3. implement and measure simultaneous UE/edge model residency, profile-tagged
   wire selection, matching decoder/tail dispatch, warm-up, memory, and switch
   overhead before claiming per-frame profile selection;
4. construct spatially and temporally correlated SNR/MCS/capacity traces, with
   complete route/fade regions and intermediate capacity bands held out for
   evaluation;
5. add `SKIP` as a separately measured extension if freshness evidence shows a
   credible safe-abstention opportunity;
6. measure and add `LOCAL` only after its compute, compact-result transport, and
   map-update behavior are supported; and
7. consider bandit/MPC/RL only if simpler baselines leave a meaningful
   held-out sequential advantage.
