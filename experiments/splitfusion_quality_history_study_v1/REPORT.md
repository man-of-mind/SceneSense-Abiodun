# Quality-unavailable history: no qualifying benefit

**Decision: NO_QUALIFYING_BENEFIT. Recommended next step: close this candidate and retain the deployed actor.** History retention works, but the one matched seed-17 pair does not meet the frozen performance/quality requirements. No further variation or deployment is justified by this result.

## Comparison

Exactly one CONTROL and one CANDIDATE were trained for 10,000 updates each, with 288 warm-up + 40,000 stochastic decisions each. CONTROL reproduces the sealed backlog-pair control's actor, initial/final bundles and complete final digest **exactly**. The frozen C0 reward, gc_v2/low-MCS configuration, transport, support, channel and original backlog transition were reused without fitting or tuning.

Primary evaluation: three common schedules (9017/9029/9043), 6,000 decisions each, held_scene + HELDOUT_GCV2 with the frozen correction. All four profiles are included. This is **previously inspected offline modeled evidence**, not untouched validation or live candidate measurements. Eligible denominators are identical: 55,222 vehicles and 21,049 persons per policy; misses and late replies remain in them.

| Policy | Joint P | Feedback ≤170 ms | Timely vehicle recall | Timely person recall | Payload P50 / P99, kB |
|---|---:|---:|---:|---:|---:|
| Matched control, 22D | .53055 | 90.850% | 65.376% | 44.841% | 88.37 / 156.21 |
| Candidate, 24D | .52989 | 90.872% | 65.268% | 44.804% | 83.39 / 135.24 |
| Deployed seed-43, 22D | .51169 | 85.056% | 62.837% | 43.391% | 105.78 / 222.83 |

P = .4 × timely vehicle within-1.0-m recall + .6 × timely person within-1.2-m recall. Timeliness and payload use all 18,000 decision frames; kB is decimal. The 170-ms boundary is the original **feedback** deadline, not prediction-completion latency. No new prediction-completion latency was measured.

Candidate − control P = **−.00065**, paired 95% block-bootstrap CI **[−.00239, +.00111]**: not the required +.010 gain. Candidate − deployed = +.01820 [+.01442, +.02204], but this cannot waive quality or control-comparison requirements.

All-scored retention ratios use the same-input noAE/UINT8/q0 reference totals:

| Retention | Control | Candidate | Deployed | Candidate requirement |
|---|---:|---:|---:|---|
| Vehicle localisation | .9321 | .9304 | .9531 | ≥.9500 — fail |
| Person localisation | .9141 | .9154 | .9234 | ≥.9000 — pass |
| Vehicle segmentation | .8201 | .8125 | .9039 | ≥.8500 — fail |
| Person segmentation | .8013 | .8114 | .8539 | ≥.8000 — pass |
| Vehicle precision | .9367 | .9335 | .9574 | ≥.9500 — fail |
| Person precision | .7739 | .7698 | .8072 | ≥each baseline−.005 — fails deployed parity |

Favourable-profile absolute vehicle localisation/segmentation/precision also fail (.9346/.8344/.9417), as do several relative retention checks. No shared-miss exception was used. E2-CORRECTED and E1-OLD sensitivity gates pass, but cannot rescue the primary failures. All 38 frozen checks, including 16 failures, are in [EVALUATION.json](eval_01/EVALUATION.json).

**Poor channel:** ADVERSE_STABLE has 4,500 decisions/policy. Candidate versus control: P .50193 vs .50230; feedback timeliness **85.889% vs 87.289%** (−1.400 percentage points); payload P50 **63.12 vs 54.72 kB** (+15.4%), P99 117.85 vs 133.85 kB. It fails both adverse-profile gates against control, despite outperforming deployed there. At MCS≤10 (n=4,206), candidate P is .47686 vs .48224 and timeliness 82.691% vs 85.473%. Aggregate timeliness does not hide this deterioration.

## Mechanism and limits

Training had 2,409 candidate timely undefined-quality events; all 2,409 retained the immediately preceding action and receipt history, including 168 events following another undefined event. None entered replay: candidate/control accepted rows are 37,879/37,897. Control cleared history after all 2,391 such events. Training post-event median q fell from .7678 to .7190 and payload rose from 74.73 to 80.47 kB; these are each policy's stochastic training observations, not a performance guarantee.

In primary evaluation, control and candidate share the same **58 post-undefined decision indices**. Candidate retains history 58/58; control 0/58. Their median selected q is **.6768 vs .6393**, payload 87.99 vs 96.98 kB, and feedback timeliness 52/58 vs 53/58. Thus retaining history does not mechanically lower q after retraining. Mode 11 is selected 56/58 by each; other mode identities remain categorical.

Within the 15 adverse post-undefined decisions, both always select mode 11. Candidate conditional-q median is .7871 vs .8387 and payload median 58.23 vs 44.17 kB, but both are timely on **12/15**. The overall adverse regression accompanies higher median payload; this small subset does not establish a payload-mediated causal explanation for it.

The FIT catalogue has 36/512 no-eligible-GT scenes; held_scene has only 1/256. Direct held-scene mechanism exposure is sparse. This is one training seed, and bootstrap intervals cover evaluation-schedule variability, not training-seed uncertainty. No M3 state swap was used to infer new latency, payload or quality. These limitations do not overturn the failed acceptance rule.

## Implementation and checks

New default-off package: `rl_agent/splitfusion_quality_history_study_v1/` (`adapter.py`, `checks.py`, `study.py`). The original 22 features keep their meanings. Two appended features are quality availability and unscored timely receipt/170; a separate receipt feature preserves the legacy success-only latency field. Defined success and timeout retain their original encodings. Only a timely excluded event with absent quality/reward **and zero eligible GT at every contributing pinned grid anchor** can retain history. Other faults and genuine boundaries clear it. Genesis is empty; repeated events replace, rather than recycle, history.

Undefined transitions remain excluded from replay/reward/critic targets/actor minibatches. There is no fabricated quality or reward and no splicing across gaps. Ordinary transitions retain duration, reward, real successor and bootstrap masks. The collector's exclusion bookkeeping stays intact; only identified no-GT resets are bridged by the observation adapter. See [STUDY_PLAN.md](STUDY_PLAN.md) for the frozen contract and gates.

[CHECKS.json](CHECKS.json) covers five outcome cases, repeated undefined, future-data refusal, genuine environment boundaries, and eight serialized real warm-up event replays through the shared encoder. Initial actor heads and all four critics were bit-identical on 64 test states after copying all weights and adding zero input columns; initial executed actions matched 64/64. Checks took **zero gradient steps**. All 40,288 training scene/channel schedules match; private action/target RNG final states match, while replay RNG naturally differs with policy-dependent exclusion counts. CUDA remained uninitialized.

One aggregation issue was resolved **without another evaluation**: the frozen driver saved all 162,000 rows, then exact equality rejected `3.0` versus `2.9999999999999996` eligible GT after interpolation. `summarize_saved.py` verifies each raw value exactly against the pinned catalogue, proves action-independent integer eligibility across every cell, and uses those authoritative integers for aggregation. Raw rows and the original driver are unchanged. Maximum per-row discrepancy is 1.78e−15; all primary P values are unchanged. The original error, proof and final analysis-source hash are retained in [AGGREGATION_NOTE.json](eval_01/AGGREGATION_NOTE.json). Gates and training sources were not changed.

## Artifacts, identities and scope

[Full comparison table](eval_01/RESULTS_TABLE.csv) includes all profiles, poor-MCS and post-undefined strata in all three settings. Exactly two presentation PNGs: [profile comparison](presentation_01/01_profile_comparison.png), [quality retention](presentation_01/02_quality_retention.png). [AUDIT.json](presentation_01/AUDIT.json) binds the plots, evaluation and matched training schedules. Reproduce using [COMMANDS.txt](COMMANDS.txt); outputs are create-only.

SHA256 identities:

```text
CONTROL actor     cbdb705cc502a0eb8db65a211e1181ae6b9528b87aa873d16157c416a75db7c1
CANDIDATE actor   2398b4e07b5edc9bd286f10802c312850d8856ffd09f01344226f3559211a7e1
DEPLOYED actor    bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce
FREEZE.json       1ec01c9007c9ba196147839bcbce9bb9d408c6601c889222a3ee2256bba2c63f
adapter.py        62508f34190d7deca0560c93d6381f5341c365f86a50cd4eb3681c4f56042567
summarize_saved   94a16c4798b48c0e7668ef03e1effb132ba7e124350e8473bf2eca05d87a6b21
```

The plan and 124 training/evaluation source/evidence files were frozen before training at 2026-10-12 03:45:12 UTC and reverified unchanged. Full inherited registration, dependency versions and source identities are in [FREEZE.json](FREEZE.json); checkpoint/binding identities are in each `runs_01/*/RUN_RESULT.json` and `RUN_BINDING.json`.

Training: control 390.2 s, candidate 404.0 s, concurrently on CPU with four threads each. Evaluation and row persistence: approximately 373.2 s; saved-row aggregation: 9.2 s. Study artifacts occupy approximately 50 MiB; remaining disk is **11.06 GiB**. No cleanup/deletion was needed. No live capture, CARLA/OAI startup, GPU inference, production/runtime change, live registration, extra training, commit or push occurred. Historical sources, weights, reports and verdicts remain unchanged. Stop here.
