# Hybrid-SAC Run 2 preregistration

Run 2 is a paired follow-up to the retained Run 1 P50-reward baseline. It is
not an attempt to make the raw reward approach 1.0. Reward magnitude depends
on the chosen quality, latency and non-admission terms; it is not an accuracy
percentage. The Run 1 contextual oracle itself averaged only about `0.473`.

## Fixed intervention

Run 2 changes only the learning reward to the train-derived exact P95 penalty:

\[
R_2 = p_{\mathrm{adm}}
\left(
Q_{\mathrm{perc}} - 0.25\frac{L_{95}}{200}
- 0.5742957604173842\,\mathbf{1}[L_{95}>200]
\right)
+ (1-p_{\mathrm{adm}})(-1).
\]

The coefficient was derived analytically on the registered training support.
It was not chosen from the fit-validation panel. Run 2 keeps the same three
seeds, update budget, replay size, model, optimizer and sampling schedule as
Run 1. The original D1 reward and the Run 2 reward must remain separately
auditable.

## Why this is feasible

- Every registered training context has a P95-feasible SPLIT action.
- The fixed penalty makes the exhaustive shaped-reward winner equal the
  constrained winner in all `1,564` training contexts.
- That constrained solution retains `96.493%` of the unconstrained mean
  quality and does not reduce mean admission.
- SAC does not differentiate through the binary environment reward. Its
  critic can learn an approximation, although smoothing near 200 ms is a
  known risk that the evaluation explicitly measures.

Before training, the same frozen coefficient must be audited on the
fit-validation panel without retuning it. If the shaped oracle fails to match
the constrained oracle there, the result is a penalty-generalization failure;
it must not be repaired using that panel.

## How learning will be judged

The fixed primary endpoint is the deterministic policy at update 5000 for all
three seeds. Earlier checkpoints are learning-curve diagnostics, not candidates
from which to pick a flattering result.

Four result tiers are reported separately:

1. **Mechanical validity:** all three runs complete with exact provenance,
   finite updates, valid actions, reproducible resume and no train/validation
   leakage.
2. **Deadline-learning signal:** every seed improves over its initialization,
   and the pooled P95 miss reduction is at least 50% versus both initialization
   and Run 1 rescored under the Run 2 reward.
3. **Useful deadline behavior:** pooled P95 misses are at most 5%, no seed or
   profile exceeds 10%, at least 90% of constrained-oracle quality is retained,
   and admission is within 0.001 of the constrained oracle.
4. **Strict modeled compliance:** zero deterministic P95 misses in every seed
   and profile. This remains a modeled conditional-survivor result, not a live
   service-level guarantee.

Contextual adaptation is a separate claim. The learned policy must beat the
best exact fixed action selected using training contexts only. The paired
difference is evaluated with a scene-clustered bootstrap; the 95% confidence
interval must exclude zero and at least two of three seeds must improve. If
deadline control succeeds but this comparison fails, the result supports a
good static choice, not the need for RL.

## Convergence presentation

Raw reward is shown, but no arbitrary `0.7` or `1.0` target is used. The main
scale-free curve is

\[
P = \frac{J_{\mathrm{policy}}-J_{\mathrm{random}}}
         {J_{\mathrm{context\ oracle}}-J_{\mathrm{random}}}.
\]

We also report oracle regret, P95 miss rate, quality, admission, all three
seeds and action behavior. A stable last-five-checkpoint curve is only plateau
evidence; it is not success unless the deadline and adaptation tests pass.

The complete machine-readable preregistration is in `preregistration.json`.
