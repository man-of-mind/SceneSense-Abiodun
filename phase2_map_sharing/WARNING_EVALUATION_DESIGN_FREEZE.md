# Phase-2 warning evaluation design freeze

Status: **frozen for the post-pilot design stage; no full collection is authorized by this document**  
Date: 2026-08-17

## 1. What the accepted pilot establishes

The immutable pilot batch `20260817_181354_pilot` and its versioned
`evaluation_v4` / `verification_v4` outputs establish that the paired causal
capture is structurally usable. All nine gates pass. In particular, the
registered controlled pedestrian can be followed from helper frame 156300,
through retained input and logits, causal tracking, hazard-only publication,
recipient-map installation, warning, and the separate CARLA truth stream.

The evaluation-only `hazard_adjudication_v2` adds one-to-one future-hazard
labels and stopping diagnostics without changing any runtime artifact. Its
seven integrity gates pass. The earlier `hazard_adjudication_v1` is superseded
because it used the already-yielding positive recipient trajectory as hazard
truth.

The two trajectories are a plumbing and computability gate, not a calibration
set and not confirmatory C2 evidence. The provisional 3.6 s target-warning lead
and byte counts are descriptive only. Shared-GPU capture-to-install timing is
non-citable.

## 2. Correct warning labels

The following terms are fixed and must not be conflated:

- **Registered-target warning:** the warning is evaluation-matched to the
  scenario's registered hazard actor. This is the source of first-warning time
  and target-warning lead.
- **Truth-hazard warning:** evaluation-only future trajectory truth shows that
  the warned actor enters the class-specific safety radius within the declared
  warning horizon. The actor need not be the registered target.
- **Matched non-target warning:** the warning matches a real actor other than
  the registered target. This is a diagnostic, not automatically a false alarm.
- **Unmatched warning:** the warning has no actor-origin truth match at the
  warning time. It is a false-positive diagnostic.
- **False warning:** the warning is unmatched, or it matches an actor that does
  not satisfy the truth-hazard rule. This label is created only in the separate
  evaluation namespace; it can never enter tracking, selection, or policy state.

Evaluation matching is class-constrained, one-to-one center/origin matching at
the warning frame with the registered 5 m gate. For a matched actor, compute the
minimum actor-to-recipient origin separation over `[warning_time,
warning_time + 5 s]`; the warning is truth-hazard-positive when that distance
reaches the frozen class safety radius.

The recipient trajectory used for that label must be **pre-intervention**. In a
controlled positive/benign pair, align the matched benign recipient trajectory
by elapsed simulation time and use it as the positive trajectory's no-target,
no-yield counterfactual. Using the realized positive trajectory would create an
intervention paradox: a correct yield makes the future collision disappear and
would relabel the warning as false. The realized positive trajectory remains
the source for stopping/collision outcomes, kept in a separate table with its
actuation attribution. A trajectory that ends or loses required truth before
the full horizon is censored and reported, never silently labelled false.

The inherited `false_warning = any(non-target warning)` field remains only as a
named legacy proxy. It cannot support a paper claim.

## 3. Exposure and fragmentation units

The primary nuisance metric is **false-warning-active-frame rate**: the
fraction of 10 Hz evaluation frames containing at least one false warning. It
is insensitive to the number of simultaneous warning records and less
sensitive to tracker fragmentation than raw event count.

Also report:

- false-warning episodes per minute, where a new episode begins after at least
  1.0 s with no false warning;
- unmatched-warning-active-frame rate;
- warning events per active frame;
- canonical warning-track count per truth actor;
- source-track and map-track births, expirations, and ID switches per minute;
- target-warning frames, first-warning time, misses, and continuous lead; and
- every result separately for pedestrian, cyclist, and vehicle where supported.

Frames are not treated as independent samples. Confidence intervals and tests
use paired trajectory/scenario clusters. The positive and benign member of a
matched pair always remain in the same data split.

## 4. Frozen split and calibration procedure

The existing two-trajectory pilot is excluded from all tuning and test claims.
The full plan must assign trajectory groups before looking at outcome metrics:

- 20% calibration/design;
- 20% validation/freeze;
- 60% untouched confirmatory test.

Routes, encounter geometry, traffic seed, and repeated actor trajectory are
grouping keys. Exact trajectory counts require a cluster-level power calculation
before collection; percentages do not authorize guessing a small sample size.
Naturalistic and designed-opportunity suites are stratified inside each split
and reported separately.

Only the calibration split may search this bounded grid:

- detector/map confidence floor: `{0.05, 0.10, 0.15, 0.20}`;
- map association base gate: `{2, 3, 4}` m;
- track TTL: `{0.5, 1.0}` s;
- warning uncertainty multiplier: `{0, 1, 2}`.

The CV model, 5 s warning horizon, class safety radii (2.5 m pedestrian, 3 m
cyclist/vehicle), source-local tracker, truth-matching gate, contribution schema,
and causal state are not tuned. Any change to those items creates a new design
version and requires validation again.

Candidates are filtered on calibration data, then exactly one setting is chosen
on validation data by the following order:

1. reject a hazard-only setting whose missed-hazard rate exceeds
   send-everything by more than 5 percentage points;
2. reject a cooperative setting whose false-warning-active-frame rate exceeds
   ego-only by more than 2 percentage points;
3. maximize median registered-target warning lead;
4. break ties by lower unmatched-warning-active-frame rate, then lower payload,
   then lexicographic parameter order.

The 5 pp miss and 2 pp nuisance margins are **C2 research non-inferiority
margins**, not a certified automotive safety requirement. Absolute deployment
limits belong to the later C3 safety case and require advisor/application input.
The selected setting and its hash are frozen before opening the test outcomes.
The complete calibration/validation frontier is retained so a favourable single
operating point cannot hide a poor trade-off.

## 5. C2 decision rule

On untouched paired test trajectories, hazard-only cooperation is useful only
if all of the following hold:

1. the designed-opportunity suite has a cluster-paired 95% confidence interval
   for warning-lead improvement over ego-only whose lower bound exceeds 0 s;
2. the point estimate is at least 0.5 s, the pre-registered minimum practically
   meaningful lead for this study;
3. missed-hazard and false-warning non-inferiority margins from Section 4 hold;
4. application and on-wire bytes are lower than send-everything with a paired
   confidence interval excluding zero; and
5. the same contribution bytes remain semantically identical when the later
   two-UE OAI path is inserted, with transport latency reported separately.

The naturalistic suite is always the honest denominator and reports the same
metrics. A gain only in the designed suite supports a regime-bounded C2 claim,
not a broad traffic claim. A null naturalistic result is not discarded.

## 6. Gates before full collection

Before a full run, review and freeze:

1. the evaluation-only future-trajectory hazard adjudicator, including tests
   proving that it cannot populate runtime state (**complete in v2**);
2. the trajectory-group inventory, suite strata, power-based counts, and split
   manifest (**deterministic candidate generated as `phase2_suite_ab_v1`;
   acceptance remains conditional on scenario review and the calibration
   simulation-power gate**);
3. per-suite scenario distributions and the designed/naturalistic headline
   interpretation;
4. raw-retention windows and quotas—continuous heavy retention remains banned;
5. a small calibration capture that proves the selected files are sufficient
   to replay every grid point without recollection; and
6. exact local/OAI timestamp and byte-accounting fields.

Only then may the staged collection start: calibration/design first, stop and
freeze; validation second, stop and freeze the operating point; confirmatory
test last. Controller/RL work remains downstream of C2 and the LOCAL
measurement table.

The candidate inventory is documented in `PHASE2_SUITE_AB_DESIGN.md`: Suite A
is designed, Suite B is naturalistic, the pilot is excluded, and the exact
20/20/60 assignments are hashed before collection. It is not runtime
authorization.

The complete Phase-2 constraint ranking, stopping-outcome boundary, and SKIP
semantics are frozen in `PHASE2_CONSTRAINT_CATALOG.md`. They do not turn this
pilot into a reward-calibration or braking-actuation experiment.

## 7. Pilot diagnostic carried forward

The provisional warning rule is intentionally not frozen. In the benign pilot,
warning-active-frame rate is 89.2% for ego-only and 93.3% for both cooperative
arms. Unmatched warnings alone are active on 84.2%, 90.8%, and 90.8% of benign
frames respectively. The controlled pedestrian is represented by 4–5 canonical
warning tracks in each arm rather than one stable track.

Hazard-only emits more warning events than send-everything even though it sends
fewer objects. This is not silently called a controller win or a code failure:
pre-publication hazard filtering changes which helper observations reach map
association, so it can change track persistence and the fraction of installed
tracks already near the warning boundary. The frozen diagnostics must expose
that coupling; calibration must not optimize lead while ignoring it.

## 8. Adjudicator v2 pilot result and limits

Under the matched benign/no-yield recipient counterfactual, the controlled
pedestrian reaches a minimum center separation of **0.254 m** and is
future-hazard-positive in all three arm outputs that register it. The earliest
registered-target warning remains 3.6 s earlier for both cooperative arms than
ego-only. These are computability diagnostics from one excluded pilot pair,
not C2 estimates.

On the realized positive trajectory, the scenario orchestrator yields with
zero collision, minimum surface clearance about **2.88 m**, and sustained-stop
surface clearance about **2.89 m**. Warnings were not actuated, so these values
cannot be credited to ego-only, send-everything, hazard-only, or any policy.
The ego dimensions are a same-blueprint truth proxy; direct ego bounding-box
logging is required before override claims.
