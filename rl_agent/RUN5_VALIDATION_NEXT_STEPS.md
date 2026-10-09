# Run 5 validation and internship action sequence

Direction agreed by Abiodun and the supervisor on 2 October 2026: complete the single-UE evidence story, measure channel response, confirm map operation, extend to two UEs, then add occlusion alerts. Run 5 is the main agent and Run 4 the comparator. Diagnose before deciding on retraining; finish one stage before starting the next.

This is the current validation sequence. It does not authorize an unattended training campaign, new radio configuration, multi-UE launch or changes to frozen evidence. No new live run or training was launched when this plan was recorded.

## Diagnosis hold requested by Abiodun

On 2 October 2026, Abiodun paused the supervisor's action-item sequence until the localization and agent-performance diagnosis is reviewed. Current work is an offline audit of inference/evaluator parity, recorded outcomes and action/state distributions, and training-versus-live environment assumptions. Do not change rewards, retrain, wire new live behavior or launch a validation as part of this audit. Preserve existing captures and report confirmed defects separately from hypotheses and missing evidence. The supervisor's stages below remain the intended sequence, but are on hold.

The [completed offline diagnosis](../experiments/splitfusion_run5_diagnosis_v1/20261003_v1/REPORT.md) confirms live 448-row stretching versus the trained 432-content-plus-16-padding contract, a missing live vehicle confidence cutoff, and an omitted edge-compute interval in the inherited training latency composition. Actor/checkpoint identity and previous-outcome/hold bookkeeping pass. Quantitative causal contributions remain unmeasured. Review a bounded perception/evaluator parity repair before resuming Stage 1 or deciding on retraining; do not change the separate SI descriptor contract with perception packing. No repair or live run was performed by the audit.

Subsequently authorized repair: the [three-pass repair record](../experiments/splitfusion_live_parity_repair_v1/20261003_v1/REPORT.md) documents the additive `splitfusion_live_parity_repair_v1` package. Abiodun selected a versioned conservative conflict detector for new live GT. Perception packing, complete target production/evaluation, native-source/prediction preservation and process-local composition are implemented and CPU-tested (342 passed, one historical-evidence test skipped). This is not deployment or live qualification: new registration/controller lineage, the edge trampoline, a short correctness/cost qualification and model-context pose reconciliation are still required. No actor, state, reward equation, deadline or inherited training-latency model was changed. Downstream stages and retraining remain on hold.

## Fixed scope

- Retain the single GPU, UE numerical-thread cap and disjoint radio/edge CPU partition. Run 4B and Run 5B remain historical, not active deployment variants.
- Keep the original quality-aware actors and state schemas. Quality remains in the reward and previous-outcome state. The inclusive 170 ms action-open-to-UE-feedback boundary includes quality evaluation; holds reuse an action without creating a policy decision.
- Preserve captures, weights, failures and hashes. Offline rescoring must not overwrite live rewards, states, terminals or trajectories.
- Additional SNR information is not proof of robustness; model confidence is not accuracy; an integration PASS is not a policy-performance endorsement.

## Current evidence and the missing evaluation component

Historical optimized Run 4 had 426/499 timely replies (85.37%). The new Run 5 capture had 355/497 (71.43%): 999 frames, 497 decisions, 497 holds and five fallbacks. All frames reached the map. There were no missing replies or GT expiries. Outcomes were 313 defined successes, 142 timeouts and 42 timely undefined-quality exclusions. Total undefined quality was 65; its 23 late replies are already counted as timeouts. Received-feedback P50/P95/P99 were 161.0/191.3/220.1 ms, using NumPy linear quantiles.

The decrease is real but not an actor-only comparison. Run 5 added depth acquisition, exact sensor synchronization, AVO, corrected raw semantic labels and causal SNR plumbing. Different quality measurements also change subsequent actor states. Mean payload rose from approximately 245.6 to 285.8 KB. The first 1,033 channel target/noise steps match, but Run 5 consumes 1,228 steps because its decision span takes approximately 117.16 instead of 99.60 wall-clock seconds. Thus the same scenes do not necessarily encounter the same channel steps.

Read-only stage analysis attributes the observed 9.77 ms increase in mean feedback latency arithmetically to UE send preparation +2.17 ms, uplink +2.98 ms, edge processing +4.04 ms, remaining downstream work +0.63 ms and edge queue −0.05 ms. This is decomposition, not causal attribution. The earlier severe tail stalls have not returned: observed frozen-tail maxima were approximately 53.3 and 58.5 ms.

The missing CARLA person silhouettes were already known. The original person segmentation reference uses **depth-consistent pixels inside projected 2D actor boxes**, not filled rectangles and not raw person tags. The final target builder also applies positive/IGNORE decisions and vehicle reconciliation. Q_perc uses pixel-mask intersection/union; pairwise detection-box IoU is a different metric. AVO separately determines observable-person localization eligibility. The recent live path restored AVO but omitted the established segmentation target builder.

Sources:

- [Projected-box depth mask](../data_collection/route_b_perception_v3/visibility_v1.py), `visibility_from_corners`, lines 108–160: no filled-box fallback.
- [Original collection](../data_collection/run_route_b_perception_collection_v3.py), lines 533–558: paint eligible depth-supported person pixels while preserving vehicles.
- [Final target contract](../pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1/build_contract_v1.py): positive/IGNORE construction and v010 mask binding.
- [Quality-grid measurements](splitfusion_hybrid_sac_v1/offline_quality_grid/executor.py), lines 580–595: pixel confusion counts; AVO localization is separate at line 620 onward.

## Stage 1 Complete the single UE evidence story

Status: the offline target adapter now reproduces the original segmentation and object-IGNORE PNGs byte-for-byte on two retained fixtures. Its 13 CPU tests cover person, dynamic-vehicle and static-vehicle positives, all five IGNORE reasons, source pins, incomplete inputs and the original scorer's resize/IGNORE semantics. This is **offline parity, not live qualification**. No retraining or live run was launched.

The existing-log analysis now joins 497 policy decisions and 999 map installs by their recorded identities. It includes scene SI/P40 and their combined computation time, channel state, actions and payload, feedback latency, AVO person localization with detection coverage, and frame-level map freshness. Classification confidence is explicitly unavailable. Segmentation and Q_perc remain provisional under the old live reference.

Retrospective correction is blocked by the recorded data, not by missing analysis code. This capture lacks predicted dense labels and full object-score records, as well as the original full vehicle/static visibility and quarantine inputs. Aggregate intersection/union counts cannot be rescored against a different target. The next capture must preserve these inputs; historical frame-specific quarantine decisions cannot simply be copied onto new live frames.

1. Restore the established segmentation target construction in a versioned evaluator, including relevant IGNORE and vehicle-reference semantics. Reuse source code and thresholds, not a new filled-box or AVO-pixel surrogate. Prove parity on retained fixtures: absent raw person tags, visible and occluded people, overlapping boxes, camera/depth geometry and resizing. Explicitly identify missing static-environment/source inputs rather than silently omitting them.
2. If retained inputs suffice, rescore saved outputs offline to diagnose the reference difference. This is post-hoc scoring, not a corrected closed-loop trajectory: changed previous Q_perc would change subsequent actions. Fresh validation under the corrected live evaluator is required for that claim.
3. Join exact frame/session/decision/tensor identities into one timeline: scene SI/P40; SNR/MCS/backlog; action mode/q/payload; latency and deadline outcome; localization error plus precision/recall; correctly defined segmentation; and map freshness. Include SI/P40 computation time in the stage diagram. Do not forward-fill absent state observations on holds/fallbacks as new measurements.
4. Add bounded prediction logging for the next capture: object class, score, box/position, timestamp and source identities. The current retained aggregate reports do not contain complete per-object confidence histories. Do not infer confidence from Q_perc or label a model re-execution as recorded live output.
5. Compare frozen Run 4 and Run 5 through the same corrected evaluator, sensor/barrier path, hardware settings and route/channel design. Check actual scene/channel alignment, payload/action mix and stage costs. Same-state offline actor queries can isolate action differences, not establish closed-loop performance. Use a controlled, counterbalanced live comparison before attributing the observed gap to the actor; avoid launching another broad campaign merely to repeat the historical comparison.

Metric definitions:

- SI is spatial variation of Sobel gradient magnitude, not object count. P40 is mean `clip(1-range/40, 0, 1)` over valid current-sweep radar returns, not a percentile. Current exports contain combined SI/P40 computation duration, not separately measured durations for each function.
- Map age at installation is distinct from frame-level age of information: `AoI(t) = t - capture_time_of_latest_installed_frame`. Neither is automatically per-object freshness. Preserve clock domains and include update gaps.
- Classification scores are not calibrated correctness. Show classwise score distributions and detection counts alongside error, recall and precision. Do not mix legacy perception CSV matching with the authoritative AVO measurements.
- Retain CDF/frequency plots, timeouts, exclusions and fallbacks. PNG only, legible text and raw unsmoothed traces; any smoothing must be separately labeled.

Exit: evaluator-parity evidence and a source-bound joined analysis, explicit missing metrics, and a fair policy-comparison/retraining decision. Passing integration gates alone is insufficient.

## Stage 2 Measure channel response and recovery

Run a planned good-to-bad channel step followed by a sustained bad interval, within the supported domain. The actor is frozen: this is **response and recovery time**, not online training convergence. Existing favorable-stable dips are descriptive observations, not this controlled test.

Before collection, specify levels, duration, route context, repetitions, baselines and recovery criterion. Record effective command/ACK time, first causal admitted changed state, first new decision and subsequent actions, payload, latency and quality. Separate observation delay, decision-availability delay, action response and KPI recovery. Report decisions elapsed and wall-clock time, holds/fallbacks separately, and no-recovery cases. Do not count one lucky reply as sustained recovery. Check fixed-action feasibility: a channel that cannot deliver any reasonable action is not an actor-recovery failure.

If recovery is quick without unacceptable quality loss, lower its priority. If slow, unstable or worse than a baseline, raise policy/environment diagnosis and possible retraining. Define practical quick/slow thresholds before seeing new outcomes.

## Stage 3 Confirm good network operation with the spatial map

The current capture already has map ingestion and 999 installed updates; map visualization was disabled. Reuse valid evidence where possible. Distinguish ingestion from a visible renderer: enabling visualization adds workload and requires measurement.

Establish the corrected-quality single-UE baseline with freshness, confidence, localization and identity logs. Resolve unexplained stalls, stale updates, missing/duplicate results and quality/coverage problems before expanding deployment.

## Stage 4 Extend to two UEs

After the single-UE baseline is satisfactory, use two independent agents with a shared edge/map. Preserve UE/session/frame/decision/tensor/action identifiers through tail output, map and feedback. Compare per-UE latency, quality, confidence, AoI, throughput and fairness with single-UE results. Check misrouting and contention before adding tail replicas or changing the architecture.

## Stage 5 Enable occlusion detection and alerts

Only after the two-UE checks pass, add occlusion logic and bounding-box alerts to the correct UE. Log source/recipient identities, capture and alert times, false/missed alerts, localization and warning freshness. Do not start this phase while single-UE evaluation remains unresolved.

## Retraining decision

Training speed is not a reason to repeat an unchanged environment. Before retraining, verify target-mask/reward parity and audit the full action-open-to-quality-feedback latency model against the optimized live pipeline. The historical [latency-boundary audit](splitfusion_hybrid_sac_run4b_v1/LATENCY_BOUNDARY_AUDIT.md) found outdated GT delay in the inherited residual and a possible missing edge-compute term; recheck applicability. GT remains inside the present deadline, so removing GT time is not the remedy. Account for concurrency and avoid double counting.

Retrain after a specific state, action-ranking, transition-model or optimization defect is supported. Retain atomic weight/optimizer/RNG checkpoints, smoke/resume-equivalence tests, independent held-out evaluation and fixed-action comparisons. More features or updates alone do not guarantee improvement.

## Evidence and next handoff

- Current [Run 5 results](../experiments/splitfusion_run5_live_avo_results_v1/20261002_run5_results.html) and [live status](splitfusion_run5_live_avo_v1/LIVE_QUALIFICATION_STATUS.json).
- Sibling profiler worktree capture: `rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261002T224700Z_run5_live_avo_capture_1000_v1/`.
- Historical Run 4 in that worktree: `rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261002T074000Z_disjoint_cpu_partition_v2/`.
- Current exact state: `capture/cells/a71__favorable_stable/run4_phase6/PHASE6_UE_EVIDENCE.json`; quality: `run4_phase6/edge_report.json`; SNR: `run5_policy/SNR_EVIDENCE.json`; map: `direct_edge_map/direct_map_ingest.csv` and `run4_map_identity.csv`.

The [offline adapter](splitfusion_live_target_parity_v1/adapter.py) exposes `required_collection_inputs()` for the complete GT and prediction recording contract. Its [retained parity evidence](splitfusion_live_target_parity_v1/RETAINED_PARITY_EVIDENCE.json) binds the original fixtures and source hashes. The [joined-analysis code](splitfusion_run5_story_v1/render_story.py) reproduces the existing-log story without changing live outcomes.

The joined figures are PNG only:

- [Scene, channel, action and latency timeline](../experiments/splitfusion_run5_story_v1/20261002_v2/01_scene_network_actions_latency.png).
- [Localization, detection coverage and map freshness](../experiments/splitfusion_run5_story_v1/20261002_v2/02_localization_coverage_map_freshness.png).
- [Empirical CDFs and frequency distributions](../experiments/splitfusion_run5_story_v1/20261002_v2/03_statistical_distributions.png).

Their [summary](../experiments/splitfusion_run5_story_v1/20261002_v2/SUMMARY.json), joined tables, provenance and output hashes are alongside the figures. SI/P40 combined compute P50/P95 is 2.11/5.98 ms; RGB receipt to state commit is 42.60/56.61 ms. Map age at installation P50/P95 is 194.56/233.04 ms across all 999 installed frames. Frame AoI peaks at 433.30 ms between updates; this is not the maximum installation age or object/track freshness. These are descriptive measurements from one capture, not causal evidence or an actor comparison.

Next handoff is **stage/register the CPU-tested parity repair and qualify correctness and cost**, before Stage 1 resumes. Reconcile model-context poses with owned sensors, verify complete target/source/prediction preservation, and do not report the offline repair as a qualified live evaluator. Subsequent stages remain on hold, not concurrently launched.
