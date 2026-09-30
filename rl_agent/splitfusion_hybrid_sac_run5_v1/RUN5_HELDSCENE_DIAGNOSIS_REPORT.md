# Run-5 held-scene diagnosis: why Run 5 ≈ Run 4 < the fixed action

**What this is.** A read-only analysis of the sealed evaluation outputs
(`OUTPUT_MANIFEST` `cba1a71b…7825`, re-verified before and after). Nothing was
retrained, the evaluator was not rerun, and no seed or checkpoint was selected.

**Scope.** Modeled held-scene evaluation only. These are not live-system
results.

**Reconstruction.** The evaluator stored each decision's oracle *maximum*, but not
the 72 individual anchor scores or the full 22-D states. Both were rebuilt offline
with the evaluator's own scoring function and the registered RNG streams, and
were accepted only because they matched the stored data exactly:

- 15,300 rows and 1,101,600 anchor scores;
- every executed reward, expected reward, oracle maximum and refusal count;
- 9,180 of 9,180 Run-5 actions and 3,060 of 3,060 Run-4 actions reproduced from
  the rebuilt states, using batch-1 inference as the evaluator did (batched
  matmul flips 2 quantized q values by 1 unit).

Artifacts: `RUN5_HELDSCENE_DIAGNOSIS.json` and `run5_diagnosis.py`, with tests in
`test_run5_diagnosis.py`.

## Phase 2 — exact reward decomposition

The mean reward splits into delivered quality, latency penalty and timeout
contributions. There is no action-switch or other registered term. The
components reproduce the recorded reward exactly on every row (3,060/3,060 per
policy), with a mean residual of at most 1e-15.

| Policy | Reward | Delivered quality | Latency penalty | Timeout |
|---|---|---|---|---|
| Fixed (mode 11, q 3000) | 0.2239 | 0.4846 | −0.1110 | −0.1497 |
| Run-4 | 0.2035 | 0.4689 | −0.1138 | −0.1516 |
| Run-5 seed 17 | 0.2054 | 0.4703 | −0.1132 | −0.1516 |
| Run-5 seed 29 | 0.1932 | 0.4683 | −0.1203 | −0.1549 |
| Run-5 seed 43 | 0.2071 | 0.4761 | −0.1164 | −0.1526 |

The fixed action's edge is mostly **quality** (+0.009 to +0.016), then latency
(+0.002 to +0.009), then timeouts (+0.002 to +0.005). The same ordering holds in
every profile; see the JSON.

**Why mode 11, q 3000 wins on quality, latency and success at once:**

- **Quality and payload.** On held scenes, mode 11's Q_perc is flat from q 0 to
  q 5000 (0.571 / 0.573 / 0.567) while payload falls from 231 KB to 130 KB. At
  q 3000 it sits at that Q_perc peak with a moderate 177 KB payload. The learned
  actors average 201–273 KB and still get lower Q_perc (0.557–0.566), so they
  sit on a worse point of the quality–payload trade-off.
- **Success.** It barely depends on the action. About 97% of timeouts come from
  the modeled end-to-end latency exceeding 170 ms, driven by the exogenous
  retained residual (mean 79 ms, tail up to about 710 ms), which is identical for
  every policy. The on-time probability is 0.996–0.997 for every policy.
- **Static optimum.** As a descriptive check with no selection, the fixed action
  is within 0.003 of the best static anchor on these states (0.2257 against
  0.2286 for mode 11, q 5000).

## Phase 3 — mode versus q (one-step restricted-anchor diagnostic)

This is a one-step diagnostic over the admissible catalogue anchors, not a
continuous or sequential oracle. Its per-decision optimum also relies on the
residual draw and per-action scene quality, which the policies cannot observe.

| Policy | Within-mode q opportunity | Discrete-mode opportunity | Total one-step regret | Fixed minus executed |
|---|---|---|---|---|
| Run-5 seed 17 | 0.054 | 0.067 | 0.122 | +0.018 |
| Run-5 seed 29 | 0.070 | 0.065 | 0.136 | +0.031 |
| Run-5 seed 43 | 0.061 | 0.060 | 0.121 | +0.017 |
| Run-4 | 0.067 | 0.058 | 0.125 | +0.021 |

The within-mode and mode columns add exactly to the total. The regret is split
about evenly between choosing q within a mode and choosing the mode.

**The over-compression hypothesis is not supported.** Executed q minus the best
registered q within the chosen mode has a mean of −517 to −686 (median −399 to
−727), and the best within-mode anchor has the *higher* q in 54–58% of decisions.
If anything, the actors under-compress within their chosen mode relative to the
one-step anchor optimum. Executed mean q is 0.47–0.49.

## Phase 4 — actor, critic and entropy (ranking diagnostic, not Bellman calibration)

| Seed | Actor follows critic mode | Q gap, best − actor mode | Spearman (critic vs one-step) | Top-1 agreement | Discrete entropy (nats) | α_d·H |
|---|---|---|---|---|---|---|
| 17 | 70.7% | 0.019 | 0.135 | 5.3% | 1.01 | 0.050 |
| 29 | 62.4% | 0.023 | 0.164 | 2.5% | 1.20 | 0.060 |
| 43 | 66.3% | 0.021 | 0.186 | 5.3% | 1.12 | 0.056 |

The Q gap is measured with the twin critics' minimum. The ranking columns compare
the critics' ordering of the admissible anchors with the one-step expected-reward
ordering.

- **Following the critic would not help.** The critic-preferred mode is slightly
  worse one-step (0.203 / 0.192 / 0.201) than the actor's choice (0.207 / 0.194 /
  0.209).
- **The critics barely rank actions like realized value.** A low correlation with
  one-step value is meaningful here: backlog is 0 in 99.93% of decisions and
  MCS/SNR are exogenous, so an action's soft return is essentially its immediate
  reward plus an action-independent term.
- **Entropy is large relative to the value gaps.** The fixed α_d·H of 0.05–0.06
  exceeds both the critics' inter-mode Q gap (about 0.02) and the gap to the fixed
  action (0.017–0.031). The policy stays diffuse: the mean argmax-mode probability
  is 0.54–0.64.
- **Primary cause: critic action ranking and value resolution.** The value gaps
  between actions (about 0.01–0.03) are below the entropy temperature and the
  critics' ranking error. Actor–critic disagreement is not the cause, since
  following the critic would not help, and entropy is a contributing factor, not a
  demonstrated cause.

## Phase 5 — SNR

| Seed | Mode changes | Mean abs q change | Reward change |
|---|---|---|---|
| 17 | 19.4% | 839 | +0.0008 |
| 29 | 32.3% | 907 | −0.0011 |
| 43 | 40.1% | 1,259 | −0.0005 |

These compare the true-SNR and shuffled-SNR trajectories decision by decision. At
a fixed state (the one-step probe), shuffling SNR changes the mode on 7.9%, 15.2%
and 21.6% of decisions.

- **Profiles and transitions.** Mid-variable and fade-recovery show the most
  disagreement, and SNR-transition decisions (|ΔSNR| ≥ 3 dB, 411 of 3,060) more
  than steady ones. The largest per-profile reward change is 0.008.
- **Redundancy with MCS.** corr(SNR, MCS) = 0.84, so a linear fit on MCS explains
  R² = 0.70 of SNR.

In this modeled evaluation SNR steers action choice but not outcomes. This is not
a claim that SNR is intrinsically useless or useful.

## Phase 6 — recommendation: **C**

Retain the present agent, and acknowledge that the fixed action is superior in
this modeled regime.

- **Demonstrated learning: yes.** Held-scene reward rises monotonically from
  0.162 at update 500 to 0.202 at update 10,000, and regret falls from 0.166 to
  0.126, across all three seeds.
- **Convergence: not established.** The last interval, from update 7,500 to
  10,000, still gained +0.0047.
- **Modeled held-scene performance.** Run-5 ≈ Run-4, both below a fixed action
  that sits within 0.003 of the static optimum. This must be reported as such.
- **Future live performance: unknown.** Nothing here measures it.

**Why not the alternatives:**

- **A (continue training).** There is no evidence that more updates would close a
  0.017–0.031 gap given critic ranking near ρ ≈ 0.15 and a fixed-α entropy floor.
- **B (entropy or optimizer correction).** It is not demonstrated: the critics'
  own preferences are not better one-step, so sharpening the policy toward them is
  not shown to help, and any such change would need its own prospective
  registration.
- **D (a different validation first).** It is partly supported: timeouts are
  exogenous, backlog is nearly always 0, and SNR has little leverage. But the
  agents fail even to match the static optimum that this regime makes available,
  so a lack of adaptive pressure does not explain the loss.
