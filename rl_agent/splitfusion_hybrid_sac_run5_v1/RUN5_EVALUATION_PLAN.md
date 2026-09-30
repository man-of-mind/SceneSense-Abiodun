# Run-5 held-scene evaluation plan (sealed; sweep not run)

**Status: `EVALUATOR_BUILT_MANIFEST_SEALED__SWEEP_NOT_RUN`.** No held-scene policy
outcome has been produced or inspected. Tests use a synthetic fixture built from
FIT scenes only.

## Why the registered set changed

The previously named 85-scene partition `44dad342…e1061` is the `fit_validation`
half of the FIT grid. All 85 of its scenes are in the catalogue that both Run-5
and Run-4 v2 trained on, so it is not held out. The registered set is option A:
the grid's `held_scene` split, which training never loads.

## Bindings

| Item | Value |
|---|---|
| Preregistration | `2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf` |
| Campaign evidence | `RUN5_DEEP_CAMPAIGN_EVIDENCE.json` (commit `8c00a37`); `CAMPAIGN_COMPLETE` `22338fcc…4fa1` |
| Held-scene partition | `RUN5_HELD_SCENE_PARTITION.json`, partition SHA-256 `97628525…0cab`, eligible-keys SHA-256 `1bc50806…5e4d` |
| Candidates / eligible | 256 / **255**. Excluded: `…_000905_frame3847`, all 132 grid rows with undefined Q_perc (no eligible GT). Eligibility uses the training-catalogue GT rule only. Overlap with training is 0. |
| Order | route order (episode, frame) |
| Contexts | 4 profiles (FAVORABLE_STABLE, MID_VARIABLE, ADVERSE_STABLE, FADE_RECOVERY) × validation seeds 9017/9029/9043 = **12** |
| Run-5 actors | seeds 17/29/43 × checkpoints 500/1500/2500/5000/7500/10000. Update 10,000 is the only registered final actor; the others are learning-progress diagnostics. |
| Run-4 comparator | native 21-D seed-43/update-10,000 actor, weights `d064013d…8e29`, run through its registered `act_on_vector` rule on features 0–20 |
| Fixed-action baseline | FIT-only selection (`FIXED_ACTION_SELECTION.json`): the 47 anchors inside transport support on all FIT scenes were each rolled out for 1,000 FIT decisions with seed 7017. 3 queue-saturating candidates left the fitted support and were recorded ineligible. Selected: **action 67, `split_ae32_uint4_q3000` (mode 11, q 3000)**, mean FIT reward 0.2492. |
| Oracle | all 72 registered catalogue anchors, scored per decision with the kernel-exact expected (primary) and realized (secondary) immediate reward; out-of-support anchors are refused and counted |
| SNR control | the same 18 Run-5 actors with an audit-only shuffled SNR input: the context's own exogenous SNR tape permuted by `Random(derive(vseed, 'snr-shuffle:<profile>'))`, in support by construction. The environment keeps the true channel. |

## Semantics

- **Trajectory.** One trajectory per (profile, validation seed): the 255 scenes in
  order, one decision per scene. The state is reset only at the start of each
  trajectory. Previous action/outcome, backlog and channel evolve across it. The
  channel is the registered joint SNR/MCS process, fixed to the context's
  profile, with seed `derive(vseed, 'validation-channel')`.
- **Actor-independent tapes.** Every policy in a context gets the same:
  - scene order and held-tensor scene;
  - retained-residual draw;
  - transport-success draw;
  - SNR/MCS channel.

  A per-trajectory tape digest must be identical across all 38 policies, and the
  sweep asserts it. Backlog and previous-action/outcome history are separate per
  policy.
- **Action rule.** Deterministic evaluation (argmax mode, conditional-mean q).
  Feature scaling is the training scaling; no held-scene statistics are used.
- **Faults.** Evaluator/infrastructure exceptions become fault rows, reported
  separately and never dropped. Actions inside the actors' registered support
  never left transport support on held scenes (0 of 21,165 probes).
- **Diagnostics:**
  - unconditional reward, with timeouts and registered failures included;
  - success and timeout rates;
  - delivered-latency P50/P95/P99 with the censored count;
  - eligible-GT Q_perc (executed action and delivered);
  - mode counts and the q distribution (mean, SD, distinct values, histogram);
  - oracle regret (expected and realized) and refused-anchor counts;
  - paired Run-5 − Run-4 and Run-5 − fixed per context (mean/min/max);
  - true- vs shuffled-SNR one-step disagreement and closed-loop metric changes.

  Every seed is reported separately, plus the mean/min/max across seeds. No seed
  or checkpoint is selected. The results are diagnostic only and cannot change
  training or hyperparameters.

## Row counts

| Quantity | Count |
|---|---|
| Policies per context | 18 Run-5 true-SNR + 18 Run-5 shuffled-SNR + 1 Run-4 + 1 fixed = **38** |
| Decision rows | 12 contexts × 38 policies × 255 decisions = **116,280** (9,690 per context) |
| Oracle anchor scores | 116,280 × 72 = **8,372,160** |

The sweep marks itself complete only if all 116,280 rows exist with zero faults.

## Runtime and disk (measured on the FIT fixture)

- Throughput is about 14.5 ms per decision, including the 72-anchor oracle. That
  gives about 1,690 CPU-seconds, or **about 3–5 min wall** with 12 workers (one per
  context, one torch thread each, on 24 cores).
- Output is about **81 MB**: 116,280 JSONL rows at ~650 B each, plus the summary.
  This is well under the 18 GB free.

## Command (manifest-sealed; not executed)

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun_run5_wt
CUDA_VISIBLE_DEVICES='' python3 -m rl_agent.splitfusion_hybrid_sac_run5_v1.run5_evaluator \
  --evidence-root /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun \
  --run --workers 12 \
  --output-dir rl_agent/splitfusion_hybrid_sac_run5_v1/evaluation_runs/heldscene_eval_v1
```

Before running, the command recomputes the manifest from current sources, the
partition, the campaign evidence, the fixed-action selection and the actor
hashes, and refuses on any drift. It also refuses an existing output directory
or a different output path.

**Files:** `run5_evaluator.py`, `run5_held_scene_partition.py`,
`test_run5_evaluator.py` (11 tests), `RUN5_HELD_SCENE_PARTITION.json`,
`FIXED_ACTION_SELECTION.json`, `RUN5_EVALUATION_MANIFEST.json`.
