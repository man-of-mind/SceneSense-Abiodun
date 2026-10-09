# Addendum to the Run-5 held-scene diagnosis (clarifications only)

**Method.** Read-only, with nothing retrained or rerun. The numbers come from
the same exactness-gated reconstruction as `run5_diagnosis.py`: each row's
executed reward, oracle maxima and refusal count were reproduced exactly again.
`OUTPUT_MANIFEST` `cba1a71b…` was re-verified before and after. This addendum
supersedes the wording in `RUN5_HELDSCENE_DIAGNOSIS_REPORT.md` where the two
differ.

## 1. Timeout controllability

Definitions, per decision on the policy's own trajectory (3,060 decisions per
policy):

- **Unavoidable:** the policy timed out, and every admissible catalogue anchor
  also times out. An anchor is admissible when it is inside the transport
  support, and it times out when its realized reward is −1 on the same draws.
- **Avoidable:** the policy timed out, but at least one admissible anchor
  succeeds.

The classification is restricted to the catalogue: the policies' continuous-q
actions are compared only against the registered anchors.

| Policy | Timeouts | Unavoidable | Avoidable | Avoidable through latency > 170 ms | Avoidable through transport draw |
|---|---|---|---|---|---|
| Run-5 seed 17 | 464 | 428 | 36 | 34 | 2 |
| Run-5 seed 29 | 474 | 426 | 48 | 47 | 1 |
| Run-5 seed 43 | 467 | 426 | 41 | 39 | 2 |
| Run-4 seed 43 | 464 | 426 | 38 | 37 | 1 |
| Fixed (mode 11, q 3000) | 458 | 426 | 32 | 30 | 2 |

About 92% of every policy's timeouts are unavoidable for the whole catalogue.
Only 32–48 decisions per policy (1.0–1.6%) are avoidable, almost all through a
lower-payload action that would have finished within 170 ms. The 16-decision
spread in avoidable timeouts, from 32 to 48, accounts for the small success-rate
differences between policies. In no decision did a policy succeed while every
anchor timed out.

## 2. Reconciling the SNR disagreement figures

The two numbers measure different things, over the same denominator (3,060
decisions per seed, pooled over all 12 contexts; neither is a per-profile range):

| Seed | Same-state ablation | Closed-loop disagreement |
|---|---|---|
| 17 | 7.9% | 19.4% |
| 29 | 15.2% | 32.3% |
| 43 | 21.6% | 40.1% |

- **Same-state ablation (7.9% / 15.2% / 21.6%).** This is a controlled one-step
  SNR ablation. On each state of the true-SNR trajectory, the actor is re-queried
  with only feature 21 replaced by the shuffled value, and the mode is compared.
  It is `shuffled_snr.one_step_mode_disagreement_rate` in `SUMMARY.json`, and
  equals `one_step_probe_mode_disagreement` in the diagnosis. The earlier results
  message wrongly called this figure "closed-loop"; it is the same-state ablation.
- **Closed-loop disagreement (19.4% / 32.3% / 40.1%).** This compares mode choices
  decision by decision between two separate trajectories, true SNR and shuffled
  SNR. After their first differing action, backlog and previous-action/outcome
  features diverge, so the gap also counts differences in history. It is not a
  controlled same-state ablation.
- **The 19–40% range.** It is the spread of the closed-loop figure across the
  three seeds. The per-profile closed-loop values in the diagnosis JSON range
  from 9.0% to 56.0%.

Reward changes are the closed-loop trajectory differences reported before: at
most 0.0011 per seed overall.

## 3. Wording correction on the critics

The report said the critics "barely rank actions like realized value" and named
"critic action ranking and value resolution" as the primary cause. The precise
and only supported statement is:

> Across the admissible anchors, the final twin critics' action rankings show
> **weak agreement with the evaluator's one-step expected utility**: Spearman
> 0.135 / 0.164 / 0.186, with top-1 agreement of 5.3% / 2.5% / 5.3% for seeds
> 17 / 29 / 43.

This is a ranking comparison against a one-step, restricted-anchor utility. It is
**not** a direct Bellman-Q calibration and does not show that the critics are
incorrect. The critics estimate a discounted soft return, including entropy terms
and effects on later states, under the training scene distribution; the one-step
utility also relies on draws the policies cannot observe.

The supported conclusion is narrower than before: the learned policies do not
select the actions that score best under the evaluator's one-step utility. Weak
critic–utility ranking agreement and an entropy term (α_d·H ≈ 0.05–0.06) that is
large relative to the inter-action value gaps (≈ 0.01–0.03) are consistent with
that. Neither is shown to be the cause. Recommendation C is unchanged.
