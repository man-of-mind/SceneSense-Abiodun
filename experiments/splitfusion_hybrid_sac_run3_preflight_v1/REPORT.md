# Hybrid-SAC Run-3 train-only preflight

## Result

**PASS.** Two independent executions produced byte-identical copies of all
seven registered artifacts. The primary sweep evaluated exactly 81,703,360
train scene/profile/action combinations: 391 registered training scenes, four
network profiles and 52,240 executable `(mode_id, q_e4)` choices per scene.

This is an analytic simulator preflight, not policy training, validation-set
performance, a live-network experiment or an SLA result. Latency is the
registered admitted-survivor P50/P95/P99 proxy; the upper-tail alternatives
are counterfactual sensitivity checks.

## Why Run 3 may proceed

- The runtime reward remains a realized reward: timely success receives
  `Qperc - 0.25*(L/200)`; reassembly failure, edge-admission failure or service
  timeout receives `-1`. Network probabilities remain simulator-only fields.
- Every network profile has a positive-width timely region. The profile-wise
  gate identifies mode 11 with nonzero modeled timely probability across all
  9,792 executable q points in its support.
- The train-selected fixed action is mode 11 at `q_e4=6095`: expected return
  0.329791, modeled timely-feedback probability 0.980060 and P50 latency
  162.17 ms.
- Allowing q to depend on context while holding mode 11 raises expected return
  to 0.403031 and modeled timely-feedback probability to 0.985918. The
  unrestricted train-only oracle is 0.446271. These are comparators, not
  achieved policy results.
- Mode 11 remains the best fixed mode under both registered tail stresses.
  Reoptimizing the full contextual action changes only about 12--13% of exact
  winners, and improves expected return over the base winner by only
  0.00056--0.00057. The stress-optimal fixed q shifts only from 0.6095 to
  0.6000.

## High-q finding

The joint perception score is not excluding high compression. Among
train-only contextual oracle winners, the weighted fraction with `q>=0.80` is
48.2% in FAVORABLE_STABLE, 53.3% in FADE_RECOVERY, 58.9% in MID_VARIABLE and
64.0% in ADVERSE_STABLE. The corresponding weighted mean q values are 0.719,
0.755, 0.780 and 0.810.

For mode 11, moving from q=0.70 to q=0.80 changes average localization quality
from 0.6414 to 0.6210, segmentation quality from 0.4541 to 0.3773 and joint
quality from 0.5449 to 0.5096, while payload falls from 79.3 KiB to 53.7 KiB.
The preferred q therefore changes with network profile: among the three
explicit points, ADVERSE_STABLE prefers q=0.75 and MID_VARIABLE prefers
q=0.80, whereas the two favorable/recovery profiles prefer q=0.70. This is the
contextual trade-off the policy is intended to learn.

## Bindings and evidence

- Implementation commit: `808c1a0d0a59ee755387cc0318f6ac769e2ea617`
- Reward SHA-256: `f594b204c4ca9bb47ae94d73be20200881cd7771713b17dbe3a9450209ca9841`
- Kernel SHA-256: `1435958ff4df4b0aaf68af02e4113a9b9f3b0c7953b6f73aa5b089a7d2280c02`
- Preflight result SHA-256: `52801ce029eaac4dfde4834a2d8031429663a3efb383deed7ec1ed06d50ebab7`
- Focused reward/preflight tests: 56 passed

The committed evidence is the first execution under
`20260922_train_only_reward_v1_run1/`. The second byte-identical execution is
retained on disk under `20260922_train_only_reward_v1_run2/` and is summarized
by `REPRODUCIBILITY.json`; it is not force-added as duplicate evidence.
