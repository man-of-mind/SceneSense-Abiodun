# Hybrid-SAC Run 3 reward and preflight plan

**Status:** preregistered design for implementation and train-only preflight.
No Run-3 training result is claimed by this document.

## 1. Purpose

Run 3 tests the existing Hybrid-SAC architecture with a reward that has the
same information available in the simulator and in deployment.  Run 1 and
Run 2 remain immutable historical experiments.  Their probability-weighted
and P95-hinge utilities are not renamed or silently reused as the Run-3
runtime reward.

Run 3 changes the reward/transition semantics first.  It does **not** also
change the actor, critics, learning rates, entropy coefficients, seeds or
action support.  The supervisor-requested training horizon is extended to
10,000 gradient updates, but update 5,000 is retained as the matched-horizon
comparison with Run 2.  This separates the reward comparison at 5,000 from the
longer-horizon stability question.

## 2. Per-decision reward

For a successful exact feedback received by the inclusive deadline
\(B=200\,\mathrm{ms}\),

\[
r_t = Q^{perc}_t - 0.25\frac{L_t}{B}.
\]

For a simulated feature-reassembly failure, edge-admission failure or service
timeout, \(r_t=-1\).  An evaluator or infrastructure fault is excluded from
learning rather than converted into an action failure.

The reward API must not accept reassembly or admission probability.  Those
probabilities belong only to the simulator transition kernel, which samples a
realized outcome.  Offline analysis may report the mathematical expectation
over those outcomes, but that expectation is not the runtime reward and is
never a policy observation.

Mode- and compression-switch penalties are zero in the initial Run 3.  They
may be introduced only in a separately registered experiment if oscillation
is actually observed.

## 3. Perception quality retained for the initial run

The initial quality hypothesis remains

\[
Q^{perc}=Q^{loc}\left(0.7+0.3Q^{seg}\right).
\]

Thus, for defined \(Q^{seg}\in[0,1]\),

\[
0.7Q^{loc}\le Q^{perc}\le Q^{loc}.
\]

Segmentation can modulate preserved localization utility by at most 30%; it
cannot erase a localization-preserving high-compression action.  This is a
safety-oriented initial hypothesis, not a claim that \(\beta=0.30\) is a
universally optimal weight.

Every preflight and evaluation output must retain and report \(Q^{loc}\),
\(Q^{seg}\), \(Q^{perc}\), person/vehicle localization terms and
person/vehicle segmentation terms separately.  In particular, actions near
\(q=0.70,0.75,0.80\) must be reported explicitly so a localization-preserving
segmentation sacrifice is visible rather than hidden by the joint scalar.

## 4. Simulator/runtime separation

The offline network model may use measured, action/profile-conditioned
reassembly and admission rates to sample a terminal event.  The deployed agent
does not calculate or observe those probabilities.  It observes only the
eventual success, proven action-path failure or deadline timeout associated
with its identity-bound decision.

The current campaign retains only conditional survivor latency quantiles, not
an unconditional per-frame latency distribution.  Any Run-3 latency sampler
constructed from P50/P95/P99 must therefore be separately hash-bound and
labelled a quantile-based simulator proxy.  It is not new measured latency and
not a live SLA.  Because the base proxy caps the unknown upper 1% at P99, the
preflight must also report a simple registered upper-tail stress sensitivity;
this is uncertainty analysis, not a new measured tail or a reward-weight tune.

Every stochastic training visit uses a unique decision key derived from the
run/session seed, episode or visit number, and decision sequence.  The key must
not contain the selected action.  Reusing only scene/profile would freeze one
outcome forever, while including the action would create action-specific luck
that the policy could memorize.

The production contract currently censors an ambiguous feedback-only timeout.
The simulator may assign \(-1\) only because it knows the sampled underlying
service outcome.  Adopting the same rule live requires a later identity-bound
way to distinguish service failure from loss of the compact feedback packet.

## 5. Mandatory train-only preflight before training

Using only the registered training partition, enumerate supported mode/\(q\)
actions and report:

1. separate localization, segmentation and joint quality;
2. payload and datagram count;
3. modeled reassembly and conditional admission rates;
4. modeled timely-feedback rate and timeout/failure rate;
5. success-branch reward and expected simulator return;
6. winners and action frequencies by network profile;
7. summaries for \(q<0.70\), \(0.70\le q<0.80\), and \(q\ge0.80\);
8. fixed-action, fixed-mode/best-\(q\), and contextual-oracle comparators.
9. the registered unknown-upper-tail sensitivity alongside the base proxy.

The preflight is a design check, not checkpoint selection.  Initialization
re-verifies the hash-pinned full quality artifact and its pre-existing legacy
D1 qualification, including its registered legacy held-scene evidence, and
loads reward-blind partition metadata containing fit-validation identities.
Neither legacy held-scene nor fit-validation identities enter any Run-3
aggregate, action selection or reward evaluation: only the 391 registered
training identities do.  No Run-3 reward evaluation or checkpoint selection
is performed on fit-validation until the reward/kernel hashes, training
endpoint and three seeds are frozen.

## 6. Initial Run-3 decision rule

Proceed to the initial three-seed run only if all of the following hold:

- the reward function has no probability argument;
- simulator probabilities cannot enter the policy observation;
- exact reward and kernel hashes are recorded;
- simulator draws revalidate against unique, action-independent decision keys;
- high-\(q\) quality/latency trade-offs are present and not excluded by an
  implementation error;
- at least one supported SPLIT action has a non-degenerate timely-success
  region in each modeled profile;
- fixed and oracle comparators reproduce deterministically;
- repeated preflight runs are byte-identical.

## 7. Registered training horizon and checkpoint policy

Run three registered seeds for 10,000 gradient updates.  Update 10,000 is the
primary endpoint; update 5,000 is the preregistered matched-horizon comparison.
Intermediate results describe the trajectory and must not be used to replace
the primary endpoint after looking at the curves.

Record scalar training metrics continuously and write lightweight model-only
evaluation snapshots every 500 updates.  To avoid repeating Run 2's multi-GB
replay-buffer snapshots, write full resumable checkpoints only at updates 0,
5,000 and 10,000.  A `latest` reference must not duplicate the final checkpoint
bytes.  The run must pass a free-space/projection gate before launch.

At every 500-update evaluation point report reward components, separate
quality components, realized terminal outcomes, latency/deadline outcomes,
mode frequencies, q distributions, discrete and continuous entropy, and
critic-ranking/calibration diagnostics.  A degradation after an early peak is
a result to report, not a reason to cherry-pick the peak.

## 8. Scope limitation

The existing empirical environment is a one-step contextual problem over
sparse selected Route-B frames.  It can test whether the policy learns a
scene/radio-to-action mapping.  It cannot demonstrate 10-Hz action holding,
delayed feedback, timeout recovery or temporal adaptation.  Those claims
require a later causal rollout over contiguous Route-B frames and an ordered
radio trace.  Run 3 must be described accordingly.
