# Run-4 exploration and gradient-start gate

## Outcome

Run 4 should be decision-cycle driven, not trained against an artificial 5 Hz
or 10 Hz clock.  One training transition is:

1. observe the current causal scene/network state;
2. select one policy action;
3. wait for that action's success/failure feedback, bounded by the registered
   170 ms deadline;
4. expose the completed previous action and outcome to the next decision; and
5. only then create the next policy transition.

The live deployment may run near 5 Hz, but that cadence is not a training-state
feature or a reason to discard intermediate reward feedback.  Held
transmissions may evolve the RLC queue between decisions; they are not policy
decisions and receive neither an action identity nor a reward transition.

`exploration.py` implements only the essential pre-gradient boundary and
diagnostics.  It does not implement a trainer, load evidence, launch a runtime,
or choose scientific thresholds. Production coverage enters only as an
attested `SemiMarkovTransitionV2`; caller-constructed numeric observations
cannot authorize gradient start.

## Run-3 audit

Run 3 already got two important policy paths right:

- `_actor_action` in `empirical_contextual_run3_runner.py` samples the
  categorical mode with `torch.multinomial` and samples conditional q through
  `sample_all_modes`.  Training collection after warm-up is stochastic SAC.
- `ConditionalHybridActor.deterministic_execution` uses the categorical argmax
  and the selected conditional mean.  That is a separate evaluation path.

The gap is `_warmup_action`: it draws both mode and q uniformly at random.
With 1,024 samples, broad coverage is likely, but no individual mode/q stratum
is guaranteed before the first update.  There was also no hard gate proving
that SI, P40, radio state, backlog, or previous outcomes varied before learning.

Run 4 therefore keeps stochastic actor sampling and deterministic evaluation,
but replaces the probabilistic warm-up assumption with an exact schedule and a
fail-closed ledger.  `require_selection_path` rejects deterministic actor use
during training and stochastic actor use during evaluation.

## Policy observation audited by this gate

The compact observation audited here contains:

- current scene SI;
- current radar P40;
- latest strictly prior UE-decoded round-0 UL MCS;
- current UE RLC backlog;
- previous mode and executed q;
- previous success/failure;
- previous quality and latency when the action succeeded.

The previous outcome contract is exact:

- success requires finite quality in `[0,1]` and finite latency in
  `[0,170]` ms;
- failure/timeout requires both quality and latency to be absent.

Raw previous reward is deliberately not duplicated in the next observation.
The transition still carries reward as the critic target, while the next state
gets the causal outcome components that explain it.  This avoids making the
same outcome enter twice under different scalings.

Identifiers and privileged labels are rejected by the mapping boundary.
Network-profile identity, profile labels, frame/session IDs, future fields, and
reward fields cannot enter `CoverageObservation`.  Measurement timestamps and
validity remain runtime adapter guards; they are not actor features in this
minimal state. `CoverageObservation.from_mapping` and the bare-record helpers
exist only for isolated statistics tests. Their ledger is explicitly
`TEST_ONLY_UNATTESTED`, and its report always refuses gradient start regardless
of apparent variation.

## Stratified warm-up

`WarmupScheduleConfig` has no q-support default.  The caller supplies the exact
inclusive q bounds for all 12 modes and the support-contract identity.  The
optional `registered_modeled_support_config` helper explicitly binds the
existing hash-registered modeled support; it is not imported or selected at
module import.

For each mode, `partition_quality_support` divides the integer wire support
into caller-selected equal-width strata.  It does not use the six historical
catalog anchors.  The schedule then guarantees:

- each block of 12 decisions contains all 12 modes once;
- each mode visits every q bin the configured number of times;
- two or more samples per bin include both exact bin boundaries;
- action order is privately permuted from the configured seed;
- every action has a unique counter-derived identity; and
- Python's global RNG is never advanced.

The action schedule is independent of state.  The environment must provide the
next naturally evolving state before consuming the next action.  The schedule
never manufactures a radio condition, scene, success, failure, or network
profile to satisfy coverage.

## Gradient-start gate

The production `ExplorationCoverageLedger` accepts only the next exact attested
transition. It derives all coverage values from the guarded current state,
checks the scheduled mode/q, and requires one session, UE, and strictly
incrementing decision sequence beginning at sequence zero. For every decision
after the first, the current guarded-state digest must equal the preceding
transition's successor digest. This also binds the visible previous action and
outcome to the exact preceding reward resolution. The final scheduled
transition must be `CONTINUES`; its real successor is recorded automatically as
the final feedback state. Thus every scheduled action is observed once as a
causal predecessor before gradients can start.

The report and gate cover:

- current per-mode action counts;
- current per-mode/q-bin counts;
- previous-action per-mode and per-mode/q-bin counts;
- prior-outcome presence, successes, and failures;
- finite count, distinct count, span, and boundary saturation for SI, P40,
  prior UL MCS, and RLC backlog;
- the same state-variation checks within every mode, so global variation cannot
  conceal a mode seen in only one state regime; and
- finite count, distinct count, and span for successful previous quality and
  latency.

Any non-finite required state or outcome value fails closed.  Constant state,
boundary-clipped state, a missing mode/bin, an absent final feedback state, or
insufficient previous success/failure coverage refuses gradient start.

Every quantitative threshold is a required constructor argument in
`CoverageGateConfig`, `StateFeatureThreshold`, or `OutcomeMetricThreshold`.
There are no embedded claims that a particular sample count, span, or
saturation fraction is scientifically sufficient.  The campaign must bind
those provisional values before collection and retain `config_sha256` in the
run manifest.

The intended integration order is:

```text
state = environment.reset()
for scheduled_action in warmup_schedule:
    execute scheduled_action and wait for feedback/timeout
    transition = build the attested transition with its real successor
    ledger.record_transition(transition)

ledger.require_gradient_start()

training:   sample categorical mode and conditional q stochastically
evaluation: categorical argmax and conditional mean only
```

If an episode boundary or excluded fault occurs before this chain is complete,
start a new ledger and a fresh session UUID. Do not splice independent genesis
draws or states from another episode into the coverage record.

If the gate fails because naturally observed state or outcome variation was
insufficient, do not synthesize missing states.  Extend or repeat a
preregistered collection schedule in the real environment, or revise the
provisional threshold in a new preregistration.

## Ongoing actor diagnostics

`summarize_action_trace` is usable on later stochastic-training or
deterministic-evaluation decisions.  It exposes, without inventing pass/fail
thresholds:

- per-mode counts, normalized mode entropy, and dominant-mode fraction for
  discrete mode collapse;
- per-mode q-bin counts and distinct-q counts;
- separate low-q and high-q-bin fractions;
- exact lower/upper-boundary counts for q-boundary collapse; and
- for each current state feature, q-support-fraction Pearson association and
  categorical-mode eta-squared.

The associations are observational diagnostics, not proof that the actor uses
a feature.  A value near zero is a reason to run a controlled intervention,
not by itself a failure verdict.

## Required 500-update smoke diagnostic

Before Run 4 is allowed to continue from 500 to 1,500 updates, bind a small
fixed diagnostic grid and its thresholds in the smoke configuration.  This is
diagnostic/evaluation work; it does not change trainer updates.

The fixed grid must check all of the following:

1. **Critic rank agreement and exact-enumerator regret.**  At each fixed state,
   enumerate all 12 modes and the preregistered q diagnostic points.  Compare
   critic ranking with the available exact/realized diagnostic target, report
   rank agreement and regret of the critic-selected action relative to the
   exact enumerator, and break results out by mode.  Explicitly report mode 11
   to detect recurrence of the Run-2 critic-misranking failure.
2. **q support coverage.**  Report per-mode q-bin counts, distinct q values,
   and low/high-bin fractions from stochastic training decisions.  The upper q
   bin must be represented; an outer-bin total alone is insufficient because
   it can hide low-boundary collapse.
3. **Controlled MCS/backlog response.**  Starting from fixed observed states,
   perturb prior UL MCS and RLC backlog only within preregistered observed or
   calibrated support while holding SI, P40, and the previous outcome fixed.
   Record mode probabilities and every conditional q head, plus the
   deterministic evaluation action.  This reveals whether the actor ignores
   network state.  These diagnostic perturbations are never inserted into
   replay as real transitions.
4. **Scene response.**  Apply the same bounded paired-state method to SI and
   P40 so apparent network responsiveness does not conceal ignored scene
   inputs.

Grid identities, perturbation values, target source, rank/regret definitions,
minimum high-q coverage, response criteria, and all pass/fail thresholds must
be configuration-bound before the 500-update run.  Do not select them after
viewing the checkpoint.  If the gate fails, preserve the checkpoint and stop;
do not continue to 1,500 updates merely because losses are finite.

## Scope exclusions

This utility does not:

- choose q-bin counts, warm-up length, or scientific gate thresholds;
- add grant, TBS, network-profile identity, age, validity, or raw reward to the
  actor state beyond the registered prior-UL-MCS feature;
- turn held transmissions into policy decisions;
- claim that UL MCS is an unquantized or instantaneous SNR measurement;
- implement entropy/alpha tuning; or
- start training or any live service.
