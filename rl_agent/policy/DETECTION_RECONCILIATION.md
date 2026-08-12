# Detection coverage reconciliation

Date: 2026-08-11  
Analysis status: `ANALYSIS_COMPLETE`  
Corpus disposition: `HOLD_DETECTION_CONFIG_RECONCILIATION`

## Verdict

**This is not purely a metric-definition effect. Treat it as a genuine collection-pipeline/config regression,
not as a regression of the saved model weights.** The live coverage metric is stricter and more scene-dependent
than the curated offline recall metric, which explains a large part of the apparent gap. However, applying the
identical live metric to old staleness/policy traces gives 44.70–54.95% vehicle object-row coverage, versus
34.66% in the new corpus. An offline-visibility denominator proxy does not remove that new-versus-old deficit.

The checkpoint path and SHA-256 are unchanged. The clearest concrete drift is that all 24 new runs inherited
the runtime's 5,000 radar-points/s default, while the validation dataset and the strongest old controls used
200,000 points/s. The live decoder also used NMS radius 4/top-80, versus radius 2/top-120 in offline validation.
Those factors are confounded with the harder, denser new scenes, so the existing tables cannot assign a causal
share to each. They are sufficient to reject the “metric-only, model OK, proceed” disposition.

This finding **holds** the pedestrian-scope decision, fast-car supplement, and controller ladder. No CARLA or
OAI process was started for this analysis.

## 1. Decisive old-corpus test

The exact metric from the freshness re-score was reused without changing its headline constants:

- GT row is in `in_camera_frustum`, at most 25 m, and has finite actor-origin world XY;
- stored prediction score is at least 0.20;
- class-aware, one-to-one greedy world-XY association within 5 m;
- object-row coverage counts every eligible actor appearance; frame coverage asks whether at least one eligible
  object of the class was matched in the frame.

The old source is not one monolithic corpus. The three held-out replays named by the surrogate pilot manifest
are the decisive cohort. Six old speed-sweep traces and three fresh 200k-pps ACC traces were also scored as
sensitivity cohorts.

| Corpus/cohort | Runs | Class | Eligible rows | Object-row coverage | Frame coverage |
|---|---:|---|---:|---:|---:|
| New policy corpus v1 | 24 | pedestrian | 11,648 | 18.81% | 45.60% |
| New policy corpus v1 | 24 | vehicle | 19,228 | 34.66% | 63.24% |
| Old held-out policy replays | 3 | vehicle | 2,518 | 49.09% | 75.57% |
| Old speed sweeps | 6 | vehicle | 4,546 | 44.70% | 69.42% |
| Old fresh ACC, 200k pps | 3 | vehicle | 2,153 | 54.95% | 76.32% |

The old traces contain no pedestrian ground truth, so an exact old-versus-new pedestrian comparison is
impossible. That missing denominator is itself why pedestrian adequacy remains held.

The requested decision rule produced an intermediate but decisive result: old live coverage is far below
offline recall, proving the metrics are not interchangeable, yet it is materially above the new live corpus,
disproving a purely definitional explanation.

## 2. Why offline recall and live coverage differ

The quoted 0.883 pedestrian / 0.910 object recall is from the **AE128** model family, not from the checkpoint
used to collect this corpus. The exact `mprime_joint_noae/best.pt` profile is
`noae__uint8__roi0.0`, whose validated test recalls are:

| Offline metric | Recall |
|---|---:|
| Pedestrian | 0.8546 |
| Vehicle | 0.8930 |
| Overall object | 0.8789 |

That evaluation uses 2,162 curated test samples from
`moving_ego_pps200000_merged_8loops_stride2`; actor GT with a projected center inside the image and at least
24 px clipped box area; a 40 m range cap; score 0.20; NMS radius 2; top-120; and a 5 m class-aware match gate.
The live score instead counts every in-frustum actor appearance within 25 m, including tiny/edge appearances
and repeated frames in crowded scenes. It also grades already-decoded top-80/NMS-4 outputs. Therefore the
offline recall number must not be presented as an expected live per-frame coverage percentage.

As a sensitivity check, the live tables were filtered to an **offline-visibility proxy**: projected actor center
inside 854x480 and clipped box area at least 24 px. This is not a re-creation of the curated test set, but it
removes the two major GT-eligibility differences available in the recorded schema.

| Cohort | Class | Current denominator | Offline-visibility proxy |
|---|---|---:|---:|
| New policy corpus v1 | pedestrian | 18.81% | 19.20% |
| New policy corpus v1 | vehicle | 34.66% | 37.84% |
| Old held-out policy replays | vehicle | 49.09% | 54.43% |
| Old speed sweeps | vehicle | 44.70% | 47.92% |
| Old fresh ACC, 200k pps | vehicle | 54.95% | 59.36% |

The proxy raises both old and new values but does not reconcile them.

## 3. Coverage versus range

New-corpus direct object-row coverage under the current denominator is:

| Range | Pedestrian matched/eligible | Pedestrian coverage | Vehicle matched/eligible | Vehicle coverage |
|---|---:|---:|---:|---:|
| 0–5 m | 0/0 | n/a | 658/2,390 | 27.53% |
| 5–10 m | 10/19 | 52.63% | 3,930/8,367 | 46.97% |
| 10–15 m | 512/1,175 | 43.57% | 932/3,740 | 24.92% |
| 15–20 m | 1,126/4,982 | 22.60% | 628/2,586 | 24.28% |
| 20–25 m | 543/5,472 | 9.92% | 517/2,145 | 24.10% |

The pedestrian 5–10 m bin is only 19 rows and cannot support a “close-range matches validation” claim. Even the
10–15 m pedestrian bin reaches only 43.57% (49.19% under the visibility proxy), not approximately 85%. Vehicle
coverage is comparable to old traces at 5–10 m but falls well below the old held-out cohort from 10–25 m:

| Range | New vehicle | Old held-out | Old speed sweeps | Old ACC 200k |
|---|---:|---:|---:|---:|
| 0–5 m | 27.53% | 25.00% | 31.52% | 11.68% |
| 5–10 m | 46.97% | 49.73% | 38.49% | 45.54% |
| 10–15 m | 24.92% | 55.79% | 53.13% | 67.53% |
| 15–20 m | 24.28% | 57.19% | 48.12% | 69.55% |
| 20–25 m | 24.10% | 42.73% | 39.59% | 51.99% |

This is not merely a far-field-tail artifact.

## 4. Configuration and provenance audit

| Item | New collection | Validated/offline reference | Finding |
|---|---|---|---|
| Checkpoint | `mprime_joint_noae/best.pt` | same | Path and SHA-256 match |
| Checkpoint SHA-256 | `f319e2a5...e131d4fa` | same | Model weights intact |
| Radar points/s | 5,000 in all 24 runs | 200,000 dataset | **40x input-density drift** |
| Projected radar points/frame, median | 470.25 across run medians | ~18,584 in old 200k fast controls | Confirms the effective drift |
| Rasterizer | fast | legacy in offline eval | Prior same-frame 200k shadow test found equivalent tensors and zero unmatched decoded objects |
| Radar temporal window | 2 | recorded moving-ego recipe | No discrepancy found in the live cohort |
| Score threshold | live decode 0.05, re-score 0.20 | evaluation 0.20 | Same headline score, but live candidates were decoded first |
| NMS / maximum objects | radius 4 / top-80 | radius 2 / top-120 | Live decoder can suppress candidates that cannot be recovered from CSV |
| Back-half device / timeout | CPU / 2.0 s | CUDA offline | Audited separately; no timeout contamination |

The fast rasterizer is not the leading explanation: the same-frame validation at 200k pps compared 30 identical
measurements and reported radar tensors equal to numerical precision, equal object counts, and zero unmatched
decoded objects. By contrast, the 5,000-pps value was not explicitly locked in the corpus YAML and came from the
runtime default. The recorded corpus contains prediction/GT/metrics tables, not the dense logits and raw aligned
model inputs needed to post-hoc re-decode it under 200k pps or NMS-2/top-120.

Old 5k-pps speed sweeps still reach 44.70%, so pps alone is not proven causal. Scene density, actor mix, and
decoder capacity remain confounded. “Genuine regression” here means the **collected perception path is not the
validated recipe and performs worse under the same live score**; it does not mean the checkpoint learned worse
weights.

## 5. Timeout and empty-result accounting

All 12,000 new frames report `result_received=true`; there are **zero** 2 s result timeouts. The old cohorts also
have 100% result receipt. Therefore tail-on-CPU timeouts were not counted as detector misses.

There are 1,049 successful new frames whose decoded object count is zero. Only 45 of those frames contain any
eligible GT, totaling 222 rows: 98 pedestrian and 124 vehicle. Removing those rows as a deliberately generous
diagnostic changes coverage only from 18.81% to 18.97% for pedestrians and from 34.66% to 34.89% for vehicles.
Successful empty outputs are real detector outcomes, not transport timeouts, and remain in the headline metric.

## 6. Decision and next admissible action

1. Do not make the pedestrian-scope call from this corpus yet.
2. Do not collect the fast-car supplement yet; its perception outputs would inherit the unresolved recipe.
3. Do not start the rule/bandit/MPC/RL controller ladder yet.
4. Preserve the original `FAIL_QUARANTINED` verification and raw corpus unchanged.
5. After Abiodun and Claude review this finding, the smallest causal follow-up is a pre-registered matched
   detector A/B on identical scene conditions: first 5k versus 200k pps, then—if needed—NMS-4/top-80 versus
   NMS-2/top-120. This should be a small diagnostic, not a 24-run recollection. It requires a prospective run
   because the current corpus did not retain the raw inputs needed for faithful re-inference.

## Reproduction and artifacts

Run from the repository root with the CARLA virtual environment; this command reads tables only:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  data_collection/reconcile_detection_coverage.py
```

Final output:
`data_collection/experiments/detection_reconciliation/20260811_031710/`

The output contains pooled/per-run coverage, range tables, timeout tables, live config audit, input hashes,
offline reference, and a hashed `analysis_manifest.json`. The manifest status is `ANALYSIS_COMPLETE`; generated
experiment output remains ignored, while this report and the analyzer are tracked review artifacts.

## 7. 2026-08-12 retained on-contract diagnostic — Verdict B confirmed

This supersedes the prospective 5k/200k A/B proposed above. The later advisor close-crossing smoke showed that
distance alone did not explain the pedestrian deficit; its full sensor contract still differed from training.
The decisive run therefore retained the exact live inputs and logits while restoring the full training contract:

- 10 Hz synchronous capture, 1280×720 RGB, camera and radar HFOV 120°;
- 200,000 radar pps, legacy training rasterizer, radius 4, two-frame temporal maximum;
- score 0.20, NMS radius 2, top-120; actor-origin XY association at 5 m;
- 140/140 lossless RGB, exact radar tensor/raw projected points/calibration, and live logits retained.

On 134 close, in-frustum controlled-pedestrian opportunities, the live logits matched **111/134 = 82.84%**
(Wilson 95% CI [75.56%, 88.28%]). The old 0.855 reference lies inside that interval. A fresh replay of the
same tensors through the per-channel-u8 split path and a fresh monolithic replay both produced the identical
111/134 result, with zero per-frame decision disagreements across all three paths. Median matched confidence is
0.551 and median localization error is 0.666 m.

Radar density also reproduces training: median 18,592.5 raw returns/frame versus the ~18,584 reference. All
eligible frames have raw returns inside the target's projected GT box (median 1,686). The target remains
in-frustum at 3.07–8.42 m and reaches 1.076 m/s.

The evidence confirms **B: the prior 10% result was caused by the live sensor-contract mismatch, not broken M′
weights, bad pedestrian localization labels, the center/origin metric, or the split runtime. Do not retrain M′.**
Before any new corpus or controller result is admissible, align its collector to the training contract and review
the resolved manifest. The corpus, freshness, baseline, and RL jobs were not run as part of this diagnostic.

Reproducible artifacts are under
`data_collection/experiments/pedestrian_on_contract_diagnostic_v1/20260812_213148_smoke/`; the decisive files are
`retained_inputs/retention_manifest.json`, `on_contract_replay/summary.json`, and
`on_contract_replay/per_frame_replay.csv` within the single run directory.
