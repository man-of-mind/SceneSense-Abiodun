# Frozen offline study plan — quality-unavailable history, seed 17

One control and one candidate; exactly 10,000 SAC updates each after the inherited
288-decision warm-up, four decisions/update, CPU float32/four training threads.
Only final actors are evaluated. No live work, deployment, tuning, new seeds or
checkpoint selection. This plan is frozen before training; source/input hashes
are sealed in `FREEZE.json` after targeted checks and before either arm starts.

## Control and initialization

Reuse the sealed backlog-pair CONTROL: C0 reward, FIT scenes, TRAIN_GCV2,
binding_02, frozen low-MCS correction, **no backlog transition**. Reuse the
original collector, orchestrator, SAC numerical update, random streams, replay,
support and timing. No transport fitting/rebinding. Inherited authorization:
`experiments/splitfusion_run5_lowmcs_retrain_v1/registration_01/REGISTRATION.json`,
SHA256 `1a15ab019f55d479c4d3d403f77b5a7d5994437f7b1c05e4820fb39ce10e1aeb`.
Expected control actor SHA256:
`cbdb705cc502a0eb8db65a211e1181ae6b9528b87aa873d16157c416a75db7c1`.
Expected initial/final bundle SHA256:
`4df115d85773ad30f68db8d146b97c68d857ba71fb348403c818e5ff44f9088d` /
`86843b6eb9b73aaf08b7fee7e0ba9ecd8fc9fba0f8e98c8adb0a8b034681d104`.
Reference: `experiments/splitfusion_run5_backlog_pair_v1/20261007_v1/runs_01/CONTROL_seed17`.
Deployed reference actor remains
`bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce`.
Candidate copies every initial actor/critic/target weight; added input columns
are zero, with critic action inputs shifted intact. No extra gradient steps.
Environment and private RNG seeds/schedules are identical; policy-dependent
outcomes and replay eligibility may subsequently differ and are not forced equal.

## Candidate-only observation semantics (24 inputs)

Keep the original 22 feature meanings, including success-only quality/latency
and `prev_success` meaning a defined, successful learning outcome. Add
`prev_quality_available` and `prev_unscored_receipt_latency_normalized` (receipt
milliseconds / 170). The second added field avoids overloading the legacy
success-only latency field. All original current scene/channel/backlog fields
are untouched. This is a default-off offline adapter, not a runtime patch.

* Defined timely success: legacy history unchanged; quality_available=1;
  unscored latency=0.
* Timely valid undefined: retain previous one-hot mode and q, prev_present=1;
  legacy quality/latency/success remain their zero **sentinels**, not measured
  zero quality. quality_available=0; unscored latency holds the actual receipt.
  Require excluded EVALUATOR_FAULT, finite receipt in [0,170] ms, absent scalar
  quality/reward, and every contributing pinned quality-grid anchor to have
  absent quality AND zero eligible vehicle/person GT. Repeated events replace
  history with the immediately preceding event, not an older successful one.
* Timeout: legacy action/present history unchanged; both new fields zero.
* Genuine evaluator/infrastructure failure: retain legacy exclusion/reset;
  clear candidate observation history. Unknown reasons are never benign.
* Genesis or genuine episode/environment boundary: clear all history.

The underlying collector's exclusion/session bookkeeping remains unchanged;
the overlay bridges only its identified no-eligible-GT bookkeeping reset, not
an environment boundary. History is applied only after the resolved event and
checked against next-state commit time/session identity. Replay and recorded-
event checks use the same encoder.

## Missing-quality learning and boundaries

Undefined timely transitions remain excluded: no replay row, reward, critic
target, or actor/entropy minibatch contribution from that transition. The
collector's ordinary transitions retain original rewards, duration=2, gamma²,
successor and bootstrap flags. Do not splice across an excluded transition or
fabricate a zero reward/terminal. A preceding included transition still
bootstraps its actually observed next decision state; a later included row may
start with valid unscored history. There is no return computed across the gap.
Timeout reward/handling remain unchanged. Genuine boundaries clear the overlay
and keep the existing termination/bootstrap contract.

## Evaluation and fixed decision rule

Compare final matched CONTROL, CANDIDATE, unchanged DEPLOYED using inherited
deterministic categorical argmax and that mode's conditional mean q. Evaluation
seeds 9017/9029/9043, 6,000 decisions each, all four registered channel profiles.
E1-CORRECTED: held_scene + HELDOUT_GCV2 primary; E2-CORRECTED and E1-OLD
sensitivity retain prior Q5. These are **previously inspected offline evidence**,
not untouched validation or live measurements. Assert identical scene/channel
schedules and eligibility across policies. No recorded M3 counterfactual latency
or quality claims. Report all-profile and per-profile results, mechanism exposure
and next actions. Poor channel means ADVERSE_STABLE, plus low-MCS <=10 diagnostic.

P = .4 * timely vehicle within-1.0-m hits / all eligible vehicles + .6 * timely
person within-1.2-m hits / all eligible persons. Misses/late predictions remain
in eligible denominators. Quality retentions use same-input noAE/UINT8/q0
reference totals; report precision/segmentation on all scored cells separately.
All decisions enter timeliness and payload P50/P99; receipt is feedback, not
prediction-completion latency.

All gates must pass (undefined gate => insufficient evidence):

1. E1 P gain >= +.010 against **both** control and deployed; inherited paired
   block-bootstrap (50-decision blocks) 95% lower bound >0 against each.
2. Absolute retention floors: vehicle/person localisation .95/.90;
   segmentation .85/.80; vehicle precision .95. No shared-miss waiver.
   Person precision retention >= each baseline minus .005.
3. FAVORABLE_STABLE: versus each baseline, dP >= -.005 with CI lower >= -.015,
   timeliness >= baseline-.010, all six retentions >= baseline-.020; same five
   absolute floors apply within this profile.
4. E2-CORRECTED dP>0 and E1-OLD dP>=-.010 against each baseline.
5. ADVERSE_STABLE: P >= each baseline, timeliness >= each baseline-.010;
   report poor-channel payload changes explicitly, plus MCS<=10 outcomes.

A pass is single-seed screening, only “qualifies for further review,” never
deployment approval. Otherwise “no qualifying benefit”; close this candidate,
retain deployed actor, no next variation. Missing evidence yields “insufficient
evidence.” At most three figures; no unrelated qualification suites.
