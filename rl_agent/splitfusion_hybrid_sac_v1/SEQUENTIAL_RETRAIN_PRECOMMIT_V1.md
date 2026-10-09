# SplitFusion Hybrid-SAC sequential retrain precommit v1

Status: **NO-GO for a full retrain until every P0 gate below passes.**

This document prevents a second long training run from silently retaining the
one-step assumptions of Run-3. Run-3 remains immutable baseline evidence. The
corrected run must use a new versioned environment, transition, replay, runner,
and evidence namespace.

## Decisions retained

1. Frames continue through the split-inference path at nominal 10 FPS.
2. Policy invocation is feedback-gated, not a fixed 5-Hz timer:
   - the first frame of a hold requests reward;
   - later frames reuse the exact action without requesting another reward;
   - a new action is allowed only after the prior ticket is resolved/timed out
     **and** at least two frames used the held action;
   - therefore the nominal decision cadence is about 5 Hz, while delayed
     feedback or timeout can extend the hold beyond two frames.
3. Every held frame enqueues its real payload and affects UE backlog even
   though only the reward-requested frame receives reward.
4. The realized reward remains unchanged for the controlled retrain:

   ```text
   valid success by 200 ms: Q_perc - 0.25 * (L / 200 ms)
   verified reassembly/admission/service failure or simulator timeout: -1
   infrastructure/evaluator fault: excluded from learning
   ```

   Admission probability belongs to the simulator transition kernel, not the
   scalar reward. There is no direct `q`, payload, or action-switch penalty in
   this controlled correction.
5. `q` keeps its registered meaning: larger `q` drops more selectively ranked
   spatial support. Unnecessary compression should lose quality; sending too
   much should lose latency, timeout, and future-backlog return.
6. LOCAL remains outside this retrain. The purpose is to test whether a
   corrected SPLIT controller learns the measured trade-off.

## Causal state requirements

The preferred state remains the registered 31-feature layout only if every
field has a causal source and train support:

- current camera SI and radar P40;
- UE-policy-visible SNR, MCS, and pre-action BSR;
- measurement ages derived from timestamps, never supplied as guessed values;
- previous executed mode and `q`;
- previous terminal class;
- previous exact quality and latency when available, with validity masks.

`BSR = 0` means no older reportable UE backlog at the sampled layer and
instant. It does not prove zero radio scheduling delay, no HARQ activity, no
packets in flight, or zero gNB/core/edge delay.

The BSR sample must precede current-action payload enqueue:

```text
measure older UE backlog
        -> build state
        -> choose (mode, q)
        -> construct/enqueue current payload
```

If source timing or availability is not proven, BSR stays outside the learned
state and may only drive a separately registered guard. Missing or stale radio
values are never replaced with zero or forward-filled.

The current `clip(log1p(BSR), 0, 1)` normalization is prohibited: every
integer value of two bytes or more saturates. A replacement may be frozen only
from the training partition after the causal source and payload support are
qualified.

## Existing implementation that must be reused

Do not recreate the following mature contracts:

- `RewardTicketController` and `ActionHoldManifest` for one outstanding
  ticket, exact action reuse, `reward_requested`, `k_min=2`, timeout,
  duplicate feedback, and late-orphan handling;
- `CausalStateV1`, `PreviousOutcomeV1`, and `build_policy_features` for causal
  state construction and previous-outcome masks;
- semi-Markov duration/discount mechanics in the generic replay/trainer.

Run-3-specific terminal replay and trainer cannot simply be toggled:

- they hard-code `next_state=None`, `terminated=True`, `duration=1`, and
  `target=reward`;
- the generic contract censors controller `FEEDBACK_TIMEOUT`, while the
  registered simulator reward scores a verified simulator service timeout as
  `-1`.

The corrected path therefore needs a versioned sequential simulator adapter.
It must preserve the controller's immutable closure-time timeout/PENDING
snapshot for the next policy state while carrying a separately attested,
simulator-only outcome for a verified `-1` replay reward. A timeout must never
be relabelled as `ACTION_PATH_FAILURE` merely to pass the generic replay gate.

## Temporal evidence finding

The current quality-grid and Run-3 partition are not temporal episodes:

- scenes and radio rows were independently sampled with replacement;
- the existing hash split has no purge gap;
- normalization used all fit covariates before the split;
- radio validation uses blocks adjacent to training blocks;
- the Route-B source saved at 5 Hz, although inputs were prepared at 10 Hz;
- the quality grid is a further sparse selection with irregular gaps.

Consequently current evidence cannot identify the payload produced by every
intervening 10-FPS held-action frame. Duplicating a 5-Hz scene or its payload
would be a new model assumption, not a measurement.

Candidate scene partition for auditing (not frozen until autocorrelation and
near-duplicate gates pass):

| Partition | Fit episode frames | Eligible rows |
|---|---:|---:|
| train | `<= 5099` | 363 |
| purge | `5100..5299` | 23 |
| validation | `>= 5300` | 90 |

The last train and first validation rows are separated by about 10.6 seconds.
The entire second episode remains the episode-level final set. It is an
independent route run, but not an untouched quality-surface set because it was
already used for surface qualification.

The four radio profiles currently have one 100-step trace each. A same-trace
tail split can diagnose interpolation but cannot establish run-to-run radio
generalization. Independent replicate traces are required for that claim.

## P0 gates before any 10,000-update run

### UE telemetry and queue

- [ ] Exact pre-action BSR source and event order proved from logger/runtime
      code and evidence.
- [ ] SNR/MCS/BSR policy visibility and observation ages measured.
- [ ] BSR/action/payload/radio-service/next-BSR joined in one causal transition.
- [ ] Queue accounting includes every reward-requested and held-action frame.
- [ ] SplitFusion small/medium/large payload support is covered under at least
      contrasting radio conditions.
- [ ] BSR scaler is fit on train only; zero preservation, clipping, and
      saturation rates are reported.

### Sequential environment

- [ ] One controller lineage persists across decisions in an episode.
- [ ] The actor is invoked only when the gate is available.
- [ ] Held frames reuse the bit-exact executed action and request no reward.
- [ ] Hold duration equals the number of governed transmitted tensors.
- [ ] Previous outcome in `s_(t+1)` binds exactly to transition `t`.
- [ ] Nonterminal transitions carry real next states and bootstrap; trace/end
      boundaries terminate or truncate explicitly.
- [ ] Simulator-only timeout adjudication and controller PENDING snapshot stay
      distinct and hash-bound.
- [ ] Hidden network-profile labels and future telemetry never enter policy
      features.

### Data partition and normalization

- [ ] Temporal train/purge/validation identities are disjoint.
- [ ] No replay sample crosses a purge, trace, or episode boundary.
- [ ] Scene and radio timestamps are monotonic and used without replacement
      inside episodes.
- [ ] Autocorrelation/near-duplicate analysis justifies the purge width.
- [ ] Every fitted normalization constant uses training rows only.
- [ ] Checkpoint selection is frozen before validation; the final episode is
      evaluated once after selection.

### Coverage and scientific sanity

- [ ] All 12 discrete modes and registered continuous-`q` support receive
      deliberate warm-up coverage across radio/BSR strata.
- [ ] All retained state features vary or have a documented invariant reason;
      no random-initialization-only feature reaches the actor.
- [ ] With scene held matched, the empirical oracle does not compress more in
      aggregate under good/low-BSR conditions than bad/high-BSR conditions.
- [ ] Increasing payload/backlog cannot spuriously improve queue delay or
      delivery in the calibrated queue backend.
- [ ] High `q` remains viable where it preserves localization and prevents a
      timeout; no added penalty suppresses it by construction.
- [ ] Random, fixed-action, train-oracle, and exact-enumeration comparators are
      generated before learned-policy evaluation.

## Training launch sequence

1. Complete and review the UE-state evidence audit.
2. If required, run the smallest action-conditioned queue calibration—not a
   new 288-cell campaign.
3. Bind the queue backend, normalization, temporal partition, environment,
   transition/replay schema, reward, and endpoint selection in one manifest.
4. Run deterministic unit/integration tests and a no-training preflight.
5. Run one seed for 500 updates; inspect state sensitivity, action/q coverage,
   target/critic finiteness, and validation mechanics.
6. Continue the same seed to 1,500 updates only if the smoke gates pass.
7. Launch three pre-registered 10,000-update seeds only after both checkpoints
   pass. Preserve every earlier run as comparison evidence.

Training duration begins only at step 7. Engineering, calibration, and smoke
qualification are separate from the approximately 3.5–4 hour sequential
three-seed run.
