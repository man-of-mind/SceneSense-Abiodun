# abiodun/ — project state for Claude (repo-tracked; survives ~/.claude cache wipes)

Abiodun Ganiyu (IDCC). Research: an **instrumented, network- and safety-aware multi-modal cooperative-perception
system** (RGB+radar split inference, CARLA 0.10 / Town10HD_Opt → OAI 5G edge over RFsim). This file is the durable
index — the authoritative detail lives in the docs below (read them before acting on their topic).

## Current Run 4 and Run 5 validation sequence 2026 10 02

For the current agent-validation work, follow `rl_agent/RUN5_VALIDATION_NEXT_STEPS.md`.
It supersedes older execution-order statements below for this scope, not their historical evidence.
Abiodun has paused the downstream sequence for offline diagnosis of localization, evaluation parity and training-versus-live assumptions. No reward changes, retraining or new live experiments during this audit.
The completed audit is `experiments/splitfusion_run5_diagnosis_v1/20261003_v1/REPORT.md`: confirmed perception packing/calibration mismatch, vehicle scoring non-parity and inherited training-latency boundary gap; correct actor identity and prior/hold bookkeeping.
Abiodun subsequently authorized repair with a versioned conservative live conflict detector. `experiments/splitfusion_live_parity_repair_v1/20261003_v1/REPORT.md` records the additive CPU-tested repair (342 passed, one historical-evidence test skipped). It is NOT deployed/live-qualified: stage and register the new composition/edge trampoline, reconcile model-context pose, then qualify cost and evidence preservation before a new capture. Actor/state/reward/deadline and old evidence remain unchanged; the training-latency gap is not repaired yet.
Numeric GT critical-path decision (2026-10-05, Claude, Codex to review): `experiments/splitfusion_numeric_gt_runtime_v1/critical_path_decision_01/REPORT.md`.
Verdict **B / NO-GO** for another exact-GT live cycle in cell `a71__favorable_stable` (71 is the cell anchor, not the selected action; 170 ms inclusive):
on the 48 live100 decisions the prediction itself is ready at 165 ms median (150 ms in the legacy Run-5 capture); under the GT-never-late
scenarios the reply would be 185.6 ms median (9/48 timely) with the scorer child or 177.0 ms (17/48) with legacy-cost evaluation, versus the
legacy 355/497. Correction addendum (cell naming, host-span vs GPU completion, per-frame statistics, scenario labelling, unresolved UE slowdown):
`experiments/splitfusion_numeric_gt_runtime_v1/critical_path_correction_01/CORRECTION_ADDENDUM_01.md`; it governs where the two differ. The gap decomposes into GT wait +63 ms, scorer-child
RPC chain +26 ms, UE main-thread slowdown +13 ms (every numeric attempt, no legacy capture; mechanism unresolved), edge +3 ms. Hashing/copying the 11.66 MB source is
not the handoff cost (2.2 ms per SHA pass); the 66 ms handoff is Python-level validation across three UE threads plus two IPC round trips.
No repair was implemented and no live run was launched. The supervisor's confidence-calibration alternative is prepared (not implemented) in
`ALTERNATIVE_PREPARATION.md` beside the report; the only complete paired prediction/GT set is the 48 live100 frames.
Confidence-proxy feasibility (2026-10-05, prepared only): `experiments/splitfusion_quality_proxy_feasibility_v1/20261005_v1/FEASIBILITY_ASSESSMENT.md`.
Bounded GO for component-wise *observation* estimators (pedestrian presence/recall, pedestrian segmentation plausibility, vehicle localisation
plausibility) with exact GT Q_perc kept as reward; NO-GO for a scalar Q_perc proxy: on 3,345 recorded validation frames held out by episode,
confidence features give Q_perc Spearman 0.23-0.26 (no MAE gain), vehicle recall 0.05-0.08, but pedestrian-miss AUC 0.70-0.86 and pedestrian
seg IoU 0.75-0.79. Gating step: offline re-execution of the deleted 72-action validations (recorded inputs) to get compression-covered paired
data; the 288 campaign kept no scores, the v5 corpus is missing from disk. Nothing trained, deployed, collected or re-executed.
Early exact-GT service live check (2026-10-05, CLOSED; corrected): `experiments/splitfusion_exact_gt_live_v1/closeout_01/CORRECTION_ADDENDUM_01.md`
(governs `live_result_01/REPORT.md`) and `BASELINE_SUMMARY.md`. GT preparation/handoff works (A2-A9, CAPTURE_VALID) but registered A1 is
literally unmet (99/100 frames, retained cycle-boundary stop). Complete feedback path NOT qualified: UE receipt 188.7 ms median, 5/48 within
170 ms. Prediction ready 165.3 ms median; post-prediction 21.4 ms (GT-first). Training composition omits reassembly->prediction (56.6 ms) and
uses a stale residual; live previous-success 8% vs 85% in training. Retraining not justified until per-action live floor, latency-model
reconciliation, runtime overhead choices and state shift are resolved (addendum section 3). No retrain/run authorized.
Training-vs-live diagnosis (2026-10-05, original Run-4/Run-5 actors): `experiments/splitfusion_training_live_mismatch_v1/20261005_v1/REPORT.md`
— decision **INSUFFICIENT EVIDENCE for retraining**; unchanged-environment retraining is not justified. Both actors' shared `collector_v1`
composition omits 7-channel input and reassembly->prediction (~79 ms/decision live; edge gap measured in the training probe itself, P50 57.5);
its stale GT tail caused 77% of training timeouts; it predicts 40.4/48 timely for the live100 decisions vs 5 measured. T_v2b validated at the
median (+0.9/+2.2 ms). Quality targets correct (live was wrong, now fixed). Missing measurement: per-action fixed-action A–G latency vectors
under the pipeline to be deployed. Live timing authority: `experiments/splitfusion_exact_gt_live_v1/closeout_01/`.
Fixed-action screen (2026-10-05, STOPPED for review): `experiments/splitfusion_fixed_action_screen_v1/stop_review_01/REPORT.md`.
Arm 0 (action 33) only; registered stop rule fired. Early-GT deferred GT binding refuses catalogue-anchor actions (capture identity
carries action_id/profile_id, not the action-neutral template) -> no GT for any anchor arm; service then crashed on unhandled
WAITING_CAPACITY. Forced-action seam worked. Decision needed: admit anchor captures, off-anchor q arms, or legacy GT path; plus service fix.
Corrected training-environment candidate (2026-10-05, CANDIDATE ONLY, launch blocked): `rl_agent/splitfusion_corrected_latency_env_v1/`
(README, CALIBRATION_GAPS.md; 43 CPU tests). Measured joint stage-vector chains (action-open..UE receipt, each stage once), single-worker
GT/prediction join (GT concurrent, not added), live outcome/state parity incl. fault->genesis, hash-bound bindings, UncalibratedTermError +
launch_blockers; no APPROVAL.json. Calibration evidence does NOT exist (screen produced none). Recommendation + 3-part answer:
`experiments/splitfusion_corrected_latency_env_v1/20261005_live100_fixture_assessment/RECOMMENDATION.md`. Pending (not done, no reward
change): class-specific within-tolerance recall / localisation-tail quality review (vehicle 1.0 m, person 1.2 m reference).
Screen v2 arm 0 (action 33, mode5 AE128/UINT4 q7000) bound unchanged as PARTIAL_SINGLE_ACTION_MEASUREMENT:
`experiments/splitfusion_corrected_latency_env_v1/20261005_screen_v2_arm0_partial_assessment/RECOMMENDATION.md` — 1/46 warm timely,
prediction at service <=170 ms on 16/46 (LATENCY_INFEASIBLE); no feasible action yet; arms 1-6 await Abiodun's I9 stop-rule decision.
Screen v2 arms 0-1 bound (PARTIAL_TWO_ACTION_MEASUREMENT): `experiments/splitfusion_corrected_latency_env_v1/20261005_screen_v2_arms0_1_partial_assessment/`.
Action 65 (min payload ~11 KB) 37/45 timely but timely Q_perc median 0.0 (3/33 >=0.4); action 33 1/46 timely. Frontier bracketed, not met;
GT gates small-payload decisions (GT-first 16/45); 2/95 chains dispatch GT second despite earlier publish -> compose anchors realized order.
Arms 2-6 await Abiodun's review of B's P0 coverage stop (6/53 fallbacks). Training blocked.
Seven-arm screen bound (composition SERVICE_QUEUE_V2: explicit exact-GT queue, separate GT prep/bind/publish/dispatch/scoring,
priority PREDICTION>TARGET serial score channel, carry-over; 334+48 chains reproduce exactly):
`experiments/splitfusion_corrected_latency_env_v1/20261005_screen_v2_seven_arms_assessment/RECOMMENDATION.md`. Frontier: min-payload
controls 51-82% timely but quality ~nil; only action 67 (m11 AE32/U4 q3000) near both: 27/46 timely, timely Q50 0.51, vehicle
R@1.0m 0.82, person R@1.2m 0.21. GT-path floor ~150 ms receipt (FREEZE 85-90 ms + BIND 16 ms) is runtime, not policy. Training blocked;
limited 7-action FAVORABLE experiment needs explicit decisions (action space, channel, 253 scenes, uplink scope, fallback, baseline).
Fixed-action screen v2 (2026-10-05, STOPPED after arm 0 for review): `experiments/splitfusion_fixed_action_screen_v2/stop_review_01/REPORT.md`.
Repair verified offline (v2 pkgs exact_gt_live_v2/service_proto_v2: anchor provenance mapping, refusal cleanup, lost-archive-completion fix;
53 tests, 4 exact parity incl. 49-frame anchor proof, composed rehearsal, container). Arm 0 (a33) integration-valid, 48/48 exact rescore,
1/48 timely; stop came from my I9 'launch exit 0' clause, unpassable because the retained runner exits 1 when G8 (actor_replay) is
false by design. Amendment 01 (narrow forced-screen verdict, amendment_01/) authorized; continuation stopped after arm 1 (a65, ~11 KB
min-payload control): integration clean, 38/47 timely but Q_perc mean 0.147, yet retained Phase-6 P0 policy coverage 89% < 95% (extra
MCS-missing fallbacks) -> cell INCONCLUSIVE_OR_FAILED -> STOP (`stop_review_02/REPORT.md`). Amendment 02 (P0 report-only, this
diagnostic screen only) -> arms 2-6 all PASS. COMPLETE seven-arm result `experiments/splitfusion_fixed_action_screen_v2/final_01/REPORT.md`:
registered decision = no convincing feasible operating point in this panel. Min-payload controls 72-81% timely but Q_perc ~0.15-0.19;
FIT baseline a67 27/48 timely (Wilson 0.42-0.69, inconclusive), vehicle R 0.99 / IoU 0.73, person weak; a33 1/48, a50 8/49 timely.
Quality acceptance still undecided. Favourable cell only; not retraining calibration.
Quality/reward review + bounded seven-action pilot design (2026-10-05, analysis only, decisions pending):
`experiments/splitfusion_quality_pilot_design_v1/20261005_v1/DECISION_MEMO.md` (+ ADDENDUM_01 correcting the seven-arm
denominators/caveats; code `rl_agent/splitfusion_quality_pilot_design_v1/`, 9 tests; Q_perc recomputed exactly 312/312).
Current reward cannot separate useless-timely from useful: live E[r] a35 -0.243, a65 -0.261, a67 -0.284 (CIs overlap);
cause = timeout -1 vs success floor -0.25. Q_perc overstates via 3 m gate recall + sqrt + geometric person zeroing
(65 vehicle-matched frames zeroed) + FP ignored; frame median is minor. Proposed objective OTR = 1[<=170 ms] x
class-weighted within-tolerance recall (v 1.0 m, p 1.2 m); a67 beats all six others on OTR, paired by route tick.
Forecast: SI/P40 rule gain on held scenes/chains = 0 (always 67); hindsight headroom +0.04-0.08 only. Panel misses
the grid knee (m11 q8000 55 KB / q7000 81 KB at 96-98% of a67's Y); a67 timeliness flips under +/-5 ms. Person R@1.2
is model-limited (best anchor 0.385 FIT / 0.616 held, a67 0.36 / 0.51) and lower live than offline. Decisions D1-D11 in the memo; nothing trained.
D10 three-arm m11 frontier LIVE measurement DONE (2026-10-05, stopped for retraining decision):
`experiments/splitfusion_fixed_action_screen_v3/final_01/REPORT.md` (v3 pkgs `rl_agent/splitfusion_fixed_action_screen_v3/` +
`_analysis_v3/`; off-anchor q8000 executed with the contract's own identity action_id=None; all 3 arms PASS amendment-02
individual conditions + strict I11/I12; 49/49/48 exact re-score). q7000 (a69, 82 KB): on time 37/49, OTR 0.444 vs new
a67 0.232 (+0.21, CI excl. 0, paired too), vehicle R@1.0 0.75 vs 0.63, BUT vehicle seg IoU 0.37 vs 0.72 and Q_perc not
higher. q8000 (55 KB): no improvement (vehicle FP 18, seg 0.30). Below ~82 KB GT path (~133 ms) gates receipt (~157-160).
New a67 vehicle R@1.0 is 0.19 lower than the seven-arm a67 at shared ticks (cause unresolved) -> reference drift.
Restricted Run-5 retraining pilot DONE (2026-10-06, offline modeled): `experiments/splitfusion_run5_restricted_pilot_v1/report_01/REPORT.md`
(+ ASSUMPTIONS.md; code `rl_agent/splitfusion_run5_restricted_pilot_v1/`, report pkg `_report_v1/`; 5 original Run-5 training
modules restored byte-exact from 55b5e56 into main). Modes 5/8/11 masked brackets, v2+v3 chains (compatible), CONTROL vs
CANDIDATE reward, seeds 17/29/43 x 10k updates. Held-out: all 6 actors WORSE than fixed m11 q7000/q8000 on both rewards and
timeliness (vs q7000 cand -0.09..-0.15); arms behave alike (m11, q~3600-4700). Cause: train/held-out chain halves rank m11
latency oppositely (q8000 60% vs 89% timely); evidence too thin -> no retraining on present latency evidence; need repeated
sessions per action. Runs stopped once at session 1000 (inherited UUID format bug) -> fixed, resumed bit-identically.
Pilot actor live inference (2026-10-06): `experiments/splitfusion_run5_pilot_live_v1/REPORT.md`. Adapter
`rl_agent/splitfusion_run5_pilot_live_v1/` (FrozenPilotRun5Actor, CANDIDATE s43 u10000 = live-test choice; versioned policy-seam/
child-entry copies; G8 replay uses pilot loader; no forced layer). 100-frame diagnostic PASS (CAPTURE_VALID, all G/P gates incl.
G8, 49/49 exact re-score/replay): m11 only, q P50 4865, 32/49 timely, receipt P50 162.5. 1,000-frame run STOPPED at service start:
pinned archive bound (live_gt_handoff_v1/archive.py <=1000) vs v2 service sizing 4 x max_frames -> v2 pipeline caps a run at
~250 decisions (~500 frames). Replay-reward plots in report_02. Option (c) taken: archive capacity repair v1
(`experiments/splitfusion_gt_archive_capacity_v1/20261006_v1/REPAIR.md`; in-place archive.py max_frames = finite cumulative
job bound 4000 + separate bookkeeping bound; module name must stay (process-scope observers); v2 pipeline proofs reused only
under the registered 2-file substitution). 1,000-frame validation `valid_02` PASS (REPORT_02.md): 999 frames (registered
cycle-boundary stop), 493 decisions all m11 (q P50 4150), 493/493 replay + exact re-score, 234/493 timely, receipt P50 170.9,
current reward -0.422 / candidate 0.231, vehicle R@1.0 0.70/0.68, person R@1.2 0.56/0.50 (all/timely). No improvement claim.
Meeting plots + late breakdown + mask-lookup candidate (2026-10-06): `experiments/splitfusion_run5_pilot_live_v1/REPORT_03.md`.
Late 259 = 25 pred-late / 46 GT-pub-late / 5 both / 183 both-ready (136 within 10 ms); target admission on/blocking path in
107/183. Candidate `integration_v1.contract._mask` np.unique -> exact uint8 lookup (in place, original kept): offline exact,
-1.60 ms/admission; registered 3-file substitution (`splitfusion_gt_mask_lookup_v1.contract.substitution_ok`). 100-frame
live check diag_02 PASS (integration-qualified) but no measurable live latency effect (session spread > 1.6 ms).
Single-buffer UE input v2 + 1,000-frame valid_03 (2026-10-06): `experiments/splitfusion_run5_pilot_live_v1/REPORT_04.md`.
`rl_agent/splitfusion_ue_input_prep_v2/` (nested seam inside v1 BGR->RGB; one fresh output tensor; bit-exact CPU+CUDA on 49
recorded tensors; -1.93 ms idle). Scorer admission unchanged (3 SHA domains + JSON ownership = distinct bindings). valid_03
PASS (495/495 replay+re-score): live input prep P50 21.5->13.5 ms (P99 unchanged), prediction at service -7.9 ms, timely
265+30/495 (0.596) vs 234/493; GT publication now binding (48 GT-late). Descriptive, different sessions; no causal claim.
FULL Run-5 agent (2026-10-06, Abiodun's change of direction; pilot now HISTORICAL_STATUS):
`experiments/splitfusion_run5_full_v1/REPORT.md` (+ report_01/ASSUMPTIONS.md; code `rl_agent/splitfusion_run5_full_v1/`,
live `rl_agent/splitfusion_run5_full_live_v1/` + `_analysis_v1/`). 12 modes, original MODELED_SMOKE_SUPPORT, no mask,
original Run-5 actor/trainer/warm-up/settings, original 4-profile SNR/MCS channel, agreed reward Y(0.7+0.3Q_seg)/timeout 0.
Latency: valid_03 base chains (A,B,H,GT,score,R) + 288-table C/D/F deltas x live transfer scales (1.026/1.200/0.587, 40
captures) + T_v2b uplink delta/deadline head; LOAO on 10 screen arms MAE 0.053 timely (compresses payload sensitivity).
Seeds 17/29/43 x 10k (~8 min each); policy ~all mode 11. Seed-43 live: diag PASS, valid_01 PASS (493 dec, 323 timely
0.655, receipt P50 164.4, all mode 11 q P50 5349, agreed reward 0.355). Model predicted 0.605 for the same live contexts.
Offline target-mask parity and the joined single-UE analysis are now implemented in
`rl_agent/splitfusion_live_target_parity_v1/` and `rl_agent/splitfusion_run5_story_v1/`.
Paired perception baselines (2026-10-06, offline, quality only): `experiments/splitfusion_run5_paired_baselines_v1/20261006_125848_v11_treatment/REPORT.md`
(code `rl_agent/splitfusion_paired_baselines_v1/`). valid_b1000 has no retained inputs -> used v11 treatment, 49 decisions (all late),
selected parity 49/49. Within-tol recall full-local = UINT8 q0: vehicle 0.81 (64/79), person 0.35 (8/23, model-limited);
historical v11 actions (mostly AE64/UINT4, NOT the full Run-5 actor) vehicle 0.77 / person 0.39 but vehicle seg IoU 0.92->0.57.
Retention spec for next validation (v1 observer raises past 100): `experiments/splitfusion_run5_paired_baselines_v1/RETENTION_FOR_NEXT_RUN5_VALIDATION.md`.
Perception limit + latency losses + retention v2 (2026-10-06, offline): `experiments/splitfusion_perception_latency_retention_v1/20261006_v1/REPORT.md`.
Full-local faithfully re-executes the original FCOS stack (V3 tail byte-identical to un-optimized tail 49/49; evaluator verbatim);
gap = metric definition + scene mix (23 v11 persons = 3 actors at 25-40 m). Original 3,345-frame within-tol recall is only vehicle 0.746 /
person 0.533 (30-40 m: 0.51/0.20); 97% of beyond-tol error is RADIAL (range) in both classes -> next perception step = range-only
correction scored on frozen validation (no evaluator repair). valid_01 latency: 170 late = 45 periodic GT-prep stall (every 12th decision
24-228, all late, cause unverified) / 43 within 5 ms / 34 other GT / 28 payload / 20 pred; run non-stationary (72.8%->58.3%), not fair for
agent judgement yet; stall fix counterfactual +20..+32. Retention v2 `rl_agent/splitfusion_paired_retention_v2/` (31 tests, 493/493 exact
rescore, disabled, not wired). Paired report ADDENDUM_01: UINT8 counts identical, errors/labels/Q not.
Two candidates (2026-10-06, offline): `experiments/splitfusion_improvement_candidates_v1/20261006_v1/HANDOFF.md`.
A RETAIN for live qualification: GT-prep spikes = CPython gen2 GC scanning the long-lived heap (25-31 ms, frees ~0);
`rl_agent/splitfusion_gt_worker_spike_fix_v1/` (gen2 threshold 200 + freeze after 2 decisions + idle full collect) -> 0 in-request
gen2 (vs 16), FREEZE+BIND P99 115.5->92.5 ms, byte-identical GT; needs versioned launcher (service.py hard-codes worker).
B REJECT: range correction gated to identity; detector trained on all train frames -> in-sample range errors unbiased; validation
unchanged (0.746/0.533). Reopen only with fresh non-train non-val GT episodes.
GC fix live qualification (2026-10-06, ONE pair, a71 favourable, seed-43): `experiments/splitfusion_run5_full_live_gc_v1/REPORT.md`.
Registered decision REJECT: only failing criterion idle-collection WALL P99 5.60 ms > 5.0 ms (CPU P99 2.84 ms; tail = core
preemption; 2/481 request collisions <=1.67 ms). Mechanism worked: in-request gen2 FREEZE/BIND 41 -> 0, spikes 48 -> 3,
FREEZE+BIND P99 192.9 -> 131.3 ms, GT-ready P95 210 -> 161; RSS growth 31.0 vs 26.2 MB. Both arms integration PASS, exact
rescore 495/496. Timely 0.479 vs 0.706 NOT attributable (sessions differ; control heavier). Package
`rl_agent/splitfusion_run5_full_live_gc_v1/` (trampolines, role check, safety stops). Re-registration of the idle bound = Abiodun.
GC policy: conditional ENGINEERING adoption (2026-10-06) with explicit exception for idle-wall-P99; experiment verdict stays
REJECT (`experiments/splitfusion_run5_full_live_gc_v1/ENGINEERING_ADOPTION_DECISION.md`). Exact producer V2 (offline):
`experiments/splitfusion_numeric_gt_producer_v2/REPORT.md` — `rl_agent/splitfusion_numeric_gt_producer_v2/` (lazy radar KD-tree,
ROI-limited visibility/static counts, per-frame bbox memo, qualified fast conflict digests). Parity 49/49 vs v1 AND retained
digests, commit/cancel + refusals identical, whole-worker outputs/Q_perc identical (392/392). Whole worker: FREEZE CPU P50
74.2->65.9 ms, paired median saving 6.05 ms (CI 5.4-7.7), prep CPU/decision 120.8->114.5. Live use needs registered switch
(engine NS proxy + exact owner type V2). Recommend at most one paired GT-path live check; no timeliness change expected.
Guarded launcher + producer v2 (2026-10-06): `experiments/splitfusion_run5_full_live_gc_v2/PREPARED.md` — `rl_agent/splitfusion_run5_full_live_gc_v2/`
(adopted GC + producer switch in prep only, 2 registered substitutions, constructed-class check, v1 rollback). Launcher rehearsal
PASS (v2 and v1 rollback identical incl. cancel/timeout/terminal). ONE 1,000-frame validation PREPARED, NOT RUN (awaits approval).
Validation gc_v2 (adopted GC + producer v2), 2026-10-06: `experiments/splitfusion_run5_full_live_gc_v2/REPORT.md` — integration PASS,
494/494 exact rescore, producer V2 constructed, no safety stop; timely 411/494 = 0.832 (Wilson 0.80-0.86), receipt P50 158.8, GT-ready
P50 127.1; prediction now the later input (354/494; 63/83 late). Session comparisons descriptive only (lighter scene, 23.4 actors).
Same-frame uncompressed quality reference (2026-10-06): `experiments/splitfusion_run5_quality_reference_v1/REPORT.md`.
Launcher `rl_agent/splitfusion_run5_full_live_retention_v1/` (= gc_v2 + retention v2 innermost, pre-launch stride-5 plan,
attempt-wide ordinal, POST_DRAIN after owner thread dead) ran ONE 1,000-frame capture: integration PASS, 394/494 timely
(0.798), 99/100 retained (1 NOT_REACHED), 99/99 GT RESOLVED, replay 99/99 agent-branch byte/label/record/score exact.
UINT8 no-drop == uncompressed. Agent (m11 ~120 KB) vs uncompressed: vehicle R@1.0 0.767 vs 0.808 (-4.1 pp resolved),
vehicle seg IoU 0.810 vs 0.913 (resolved, largest), person seg 0.500 vs 0.556 (resolved); person R@1.2 0.488 vs 0.512
unresolved; uncompressed person 0.512 = model-limited. Undefined Q (gc_v2 55): 42 beyond 40 m, 10 AVO<0.65, 3 builder
refusals, 0 faults. Registered retention-ratio target tiers in ACCEPTANCE.json (trial 1/2); campaign STOP was a
false positive from my own Monitor argv (ADDENDUM_01). Nothing trained.
Run-5 channel-adaptation validation (2026-10-06, ONE capture, deployed seed-43): `experiments/splitfusion_run5_channel_response_v1/REPORT.md`.
Pinned FADE_RECOVERY (seed 2026082104, unchanged) via versioned launcher `rl_agent/splitfusion_run5_channel_response_v1/` (only change:
phase6_live_runner_v2.PROFILE substituted in runner seam, cell a71__fade_recovery verified); integration PASS, 492/492 replay+rescore,
98/98 retained exact. Live: on-time good 0.976 / ADV core 0.662 / recovery 0.873, whole 0.776; mode 11 always, q/payload adapt
(101 -> 71.5 -> 132 KB) but inconsistently (first fade no cut for 17 decisions; T_DEG 6-decision wrong-direction payload rise).
Same-segment FAVORABLE comparator also ~0.73 on time -> degradation largely route/scene, not channel alone. Offline paired: vehicle R@1.0
retention 0.885 (resolved), person 0.948 (n.s.), seg 0.78/0.81, person FP 26->77; L2 105.7 KB vs noAE/UINT8 3,555 KB (-97.1%, SFD4
overhead 452 B). ADDENDUM_01 corrects first-fade/T_DEG wording (window start vs confirmation; T_DEG SLOW on confirmation).
ACTOR_SENSITIVITY.md (CPU, 42/42 exact): channel-pair sensitivity weak (dq ~-320 / +110), conflicts with history (late history
lowers q ~870) and scene (T_DEG ~+2400); eventual reductions = session-reset history (dec 138) and scene (dec 302). Stopped there.
TRAINING_CORRECTION_DECISION.md (+ADDENDUM_02): state/history construction exact (491/491, same encoder in training); actor
not missing env-supported actions; PRINCIPAL CAUSE = training env: T_v2b uplink barely prices payload at low MCS (MCS 9: 46->60 ms
for 28->192 KB, p_on_time 0.99) so P(timely) flat across payload while live MCS<=10 zero-backlog on-time 0.80/0.84/0.10 by payload
tercile. Next (needs approval, not started): recalibrate T_v2b payload x MCS term for MCS<=15 with held-out tercile acceptance.
M3.1 GOOD_ALL full Route-B capture (2026-10-07, ONE attempt): `experiments/splitfusion_m3_route_b_v1/REPORT.md`. INCOMPLETE_ROUTE at
1,204 frames (40 %, legs 1-9): retained owned-pose parity `pose.py` ATOL 1e-5 m is unreliable for float32 world translations beyond
128 m (delta 1.043e-5 at y=128.6; see v1 ERRATUM_01: unreliable, not 'every loop must fail'). Launcher
`rl_agent/splitfusion_m3_route_b_v1/` (GOOD_ALL constant 21.5 dB via controller seam, ROUTE_COMPLETED termination, post-budget TypeError
repaired at cause = EarlyGt finish assigns into bool, capacity substitutions C1-C5 to 3,900 frames, map journal, edge submit guard, gNB stats
observer); all M3 checks passed on the partial; map objects are class-less ('Unknown'), exact join to scorer records 22,561/22,561.
M3.1 v2 (2026-10-07, second attempt): `experiments/splitfusion_m3_route_b_v2/REPORT.md`. INCOMPLETE_ROUTE at 1,232 frames: R-POSE
(max(1e-5, 2 ULP32(max|t|)) within 512 m, 4 UE reconstructed checks) and R-CLASS (map type from model class_name) both verified live,
but a 5th reconstructed check in the GT prep/score workers (`numeric_gt_integration_v2/ready.py:_frame_geometry`) rejected 1.0009e-5 m.
v3 (`rl_agent/splitfusion_m3_route_b_v3/`, P3 in both GT workers) is PREPARED_NOT_RUN; a third attempt needs Abiodun's authorization.
M3.1 v5 live attempt (2026-10-07): PRE_FRAME_CHILD_VERIFIER_FAILURE (`experiments/splitfusion_m3_route_b_v5/LIVE_CAPTURE_REPORT.md`):
v4 child -> v3 `entries.child` -> `from .host import verify_prospective` resolved the v3 host/contract. v6 repair (2026-10-07,
offline only, NOT launched): `experiments/splitfusion_m3_route_b_v6/REPORT.md`, package `rl_agent/splitfusion_m3_route_b_v6/`.
One v6 contract + per-role `launcher_binding`; native child/edge/seam/retention validators; also fixes 3 latent same-class
defects (v3 retention plan id, v5 edge compose mounted the v4 root, mislabelled GT trampolines). Fresh-process startup
verification of host/runner/child/edge(slim image)/GT service+workers PASS, 22/22 negatives rejected; launch gated on that
proof. Prepared root `20261007_m3_route_b_v6/good_all_prospective_02` (deployment worktree).
M3.1 v6 live capture (2026-10-07, ONE run): INCOMPLETE_ROUTE, `experiments/splitfusion_m3_route_b_v6/LIVE_CAPTURE_REPORT.md`.
Startup binding worked; GT prep worker died in FREEZE (`BufferError: cannot close exported pointers exist`) after 530 GT
decisions -> service channel reset -> route runner BrokenPipe at 1,073 frames, 32.2% route coverage. 450/530 timely, 529/529
exact rescore, 1,073/1,073 maps installed (AoI P50 245 ms). Cause not investigated. Free disk now 15.4 GB < 16 GB launch gate.
M3.1 v7 (2026-10-08, offline only, PREPARED_NOT_RUN): `experiments/splitfusion_m3_route_b_v7/REPORT.md`. Frame-3557 initiating
error UNRECOVERABLE (reply carries only the final exception; logs empty; no sealed source). Repair
`splitfusion_gt_freeze_lifetime_v1` bound via native v7 service/worker trampolines (prep only; score untouched; no older global
rebound). Real-service proof PASS 48/48: V6 path reproduces the exact live 3-line error list from a known injected ProducerError;
V7 reports the original + complete snapshot after teardown. Prep GC trace is lost on any fatal prep failure (service SIGTERM ->
SystemExit(143) interrupts the worker finally), unrepaired. Startup verification 6/6 roles, 17/17 negatives. Next capture = diagnostic.
M3.1 v7 live capture (2026-10-08, ONE run): INCOMPLETE_ROUTE, `experiments/splitfusion_m3_route_b_v7/LIVE_CAPTURE_REPORT.md`.
V7 exposed the original error: prep FREEZE frame 3502 (decision 472) `LiveGtContractError: yaw-only AVO does not support tilted
actor 117: TILTED_SCOREABLE_TARGET` (walker at ~39 m, pitch -0.03 deg / roll -0.01 deg > 1e-6; `run5_live_avo_v1/quality_adapter.py:171`).
Snapshot complete + hash-verified; V6 trigger plausibly the same check but unconfirmed. 962 frames, 26.9% coverage, 399/472 timely,
471/471 exact rescore. Decision pending: per-frame refusal vs registered tilt tolerance vs tilt-aware AVO. No relaunch until decided.
Tilt-aware pedestrian AVO candidate (2026-10-08, offline, NOT registered): `experiments/splitfusion_avo_tilt_v1/REPORT.md`,
pkg `rl_agent/splitfusion_avo_tilt_v1/`. Upright (|p|,|r|<=1e-6) keeps frozen yaw_rotation (bit-identical); tilted uses the frozen
CARLA actor matrix (= target-path convention, validated vs CARLA Euler) for corners/structural/containment/ground/ownership;
ground-rule envelope |R20|ex+|R21|ey<=0.03 m else fatal. 3502 replay: old raises live error verbatim, new scores 117 AVO 0.6223.
Upright parity 49/49 (2,450 rows) score-identical; 16 tests; real service 19/19; +0.34 ms/decision. Next: V8 registration, then one capture.
M3.1 V8 (2026-10-08, offline, READY, NOT launched): `experiments/splitfusion_m3_route_b_v8/REPORT.md`, pkg `rl_agent/splitfusion_m3_route_b_v8/`
(v8 worker_entry = qualified candidate trampoline: prep installs avo_tilt switch, then unchanged v7 worker -> FREEZE repair).
Prospective `20261008_m3_route_b_v8/good_all_prospective_01` (worktree; registration `2f8ee892…`, differs from V7 only in
identity/binding; AVO sources at service_proof_02 digests). Binding regression 13/13, final-path service proof 19/19
(upright 1527/1535 measurement digests = yaw-only ref; tilted 1531 TILT_AWARE_GEOMETRY; rc 0), startup 6/6 roles + 19/19
negatives, host launch gate PASS. Disk 17.71 GB vs 16 GB gate; capture ~9.2 GB. Full-route reliability unproven.
M3.1 V8 live capture (2026-10-08, ONE run): FIRST COMPLETED GOOD_ALL ROUTE (`experiments/splitfusion_m3_route_b_v8/LIVE_CAPTURE_REPORT.md`):
301/301 coverage, 3,590 frames, 1,786 decisions all scored + exact rescore, tilt-aware path live (19 frames, structural ignores).
Registered verdict NOT PASS: only G1 one_terminal_per_sent_frame (final frame 8595 in flight at ROUTE_COMPLETED); unmodified
verdict tool crashes on offline rescore without P3 (>128 m), supplementary P3 run = STOP on that G1 alone. Timely only 0.395:
GT prep CPU grows (FREEZE 68->100, BIND 14->37 ms) -> ~0 timely after 240 s; cause unresolved. Disk left 9.14 GB < 16 GB gate.
M3.1 V9 live capture (2026-10-09, ONE run; one earlier launch aborted at bring-up, no live process started):
`experiments/splitfusion_m3_route_b_v9/LIVE_CAPTURE_REPORT.md`, pkg `rl_agent/splitfusion_m3_route_b_v9/` (+ `_analysis/`).
Four registered changes: exact GT storage accounting + bounded per-call timer (prep only), GOOD_ALL 23.5 dB -> RFsim -12.0,
calibrated T-tracer PUSCH/avg_snr/MCS telemetry (radio CPUs, evidence only), and two bookkeeping repairs (bounded
end-of-route terminal drain; registered offline P3). ROUTE_COMPLETED, retained runner CAPTURE_VALID (G1-G9 + P0-P8 all
true), 301/301 coverage, 3,029 terminals for 3,029 sent frames -> the V8 G1 final-in-flight failure is CLOSED (drain
waited 112 ms for frame 7450; its real ACK arrived, expired_on_real_clock 0). Replay 1,497 spans 0 mismatches; exact
rescore 1,497/1,497; retention 100/100 RETAINED_COMPLETE and 100/100 GT RESOLVED. Registered M3 verdict tool = **STOP**
on three SCHEDULE_* checks ONLY because it reads the V3 21.5 dB schedule; against the registered V9 schedule all three
pass (3,819 rows all ACK_ON_TIME, 23.5/-12.0, v9 controller, 1,497 admitted at feature 0.947368). Unmodified verdict
tool still crashes (PoseParityError 1.4018e-05) -> the offline-P3 registration is necessary; under it 1,497 calls,
0 rejected, 17 above the retained 1e-5 bound. MISSING MEASUREMENT NOW EXISTS: 5,988 per-call _account timings
(0 dropped/errors/foreign), early 4.07 -> late 8.77 ms per call (~2x the candidate's offline 4.32 ms, above the
diagnosis's fitted 1.37x); FREEZE 67.3->85.3, BIND 11.8->21.6, FREEZE+BIND rise +27.8 ms vs V8's +54 ms reference
(confounded: V9 ran 1,497 decisions / 3,029 frames vs 1,786 / 3,590). Timely 1,123/1,497 = 0.750 (Wilson 0.728-0.771);
prediction P50 130.9 flat all route, feedback P50 160.6, GT publication is the binding and growing input (110->153 ms);
no bin collapses to 0 (V8 went to 0.00 after 240 s). Measured channel 23.5 dB P50 / MCS 27 confirms the qualification AND
that the V8 "17.2 dB" was the TPC-pinned MAC-log power-control view (17.6 here at the same time). Memory: prep RSS
690->725 MiB, 0 in-request gen2, no safety stop. Map 3,029 installed, AoI P50 248.7 ms, 0 gaps >500 ms. Do NOT attribute
the timeliness change to the GT repair (accounting AND target AND session all differ) - assess the repair by the
per-call measurement. Disk 10.2 GB < 16 GB gate. Decisions open: register the verdict tool's schedule per revision,
confirm the offline-P3 standing rule, residual per-pair growth, use the measured (not fitted) live factor, free disk.
M3.2 NOT started and not yet unblocked (the registered verdict is STOP pending the tooling decision).
M3.1 V10 live capture (2026-10-09, ONE run, the corrected GT accounting): `experiments/splitfusion_m3_route_b_v10/LIVE_CAPTURE_REPORT.md`,
pkgs `rl_agent/splitfusion_m3_route_b_v10{,_offline_v1,_analysis}`. ONE registered change vs V9: GT prep storage accounting
v1 -> corrected `splitfusion_gt_storage_accounting_v2.bounded_pair_history_summary.v2` (one bounded AccountingSession +
the V9 timer); 43 registration fields verified byte-identical to the ACTUAL V9 registration. FULL ROUTE, every layer
PASS: ROUTE_COMPLETED, CAPTURE_VALID (G1-G9, P0-P8), coverage 301/301, 3,026 terminals for 3,026 sent, exact rescore
1,499/1,499, replay 1,499 spans 0 mismatches, retention 100/100 + GT 100/100. **Host exit status 0 CAPTURED at launch**
(launch.rc), not inferred -> the retained check set runs complete. Registered authority = the registration-bound
evaluator `..._v10_offline_v1.postflight` (named in launcher_binding BEFORE launch): **PASS_M3_FULL_ROUTE_INTEGRATION
24/24, 0 reasons**. The V3-pinned M3 verdict tool still STOPs on the same three SCHEDULE_* only (it reads V3's 21.5 dB)
and still crashes unmodified (PoseParityError 1.373e-05); both retained beside the authority, neither modified.
Accounting: 5,996 calls, 0 dropped/errors/foreign; mean 4.827 ms/call vs V9 6.292 (-23%), per decision 19.31 vs 25.17 ms,
total 28.9 s vs 37.7 s. BUT the cost MOVED: compute_unbound (FREEZE, on the GT-publication path) 4.08->17.02 ms P50
while bind_plan 0.19->3.03 and set_ready 0.17->1.66; accounting inside FREEZE 8.33->24.09/decision (V9 6.68->16.60),
inside BIND 0.36->4.69 (V9 7.43->17.40). FREEZE+BIND rise +33.3 ms vs V9 +27.8 -> growth reduced, NOT removed, and
FREEZE is now worse. Cache bounded+closed (910/4096 entries, 8/8 snapshots, 494k hits, 0 alias fallbacks, 0
invalidations); dict-mutation-tag fast path verified live (entry REFUSES to start without it). Timely 1,197/1,499 =
0.799 (Wilson 0.777-0.818) vs V9 0.750 - DESCRIPTIVE, different session/scene; prediction P50 132.0 (by 170 ms
1,487/1,499) vs feedback P50 159.5; GT publication 107->158 is the growing binding input. Prep RSS 704.9->782.7 MiB
(+77.8) vs V9 +35.8 - observed, NOT diagnosed. 0/1,489 gen2 in-request; map 3,026 installed, AoI P50 249.7, 0 gaps
>500 ms, 302 late-feedback frames still installed. Measured PUSCH P50 23.5/MCS 27 (46,300 rows, 0 dropped); MAC-log
17.6 kept separate. Traffic unchanged: traffic_50_50, 50/50 spawned. Quality vehicle R@1.0 0.776 / person R@1.2 0.478.
Open: first-call FREEZE cost, prep RSS, per-revision registration of the M3 verdict tool, offline-P3 standing rule,
disk 13.76 GB < 16 GB gate. M3.2 NOT started.
GT history design (2026-10-09, design only, nothing implemented): `experiments/splitfusion_m3_gt_history_design_v1/REPORT.md`.
Conflict history is necessary (exact full-history medians/count>=3 decide quarantine -> eligibility); frame independence is a
semantic change. GT derive is already incremental (+0.7 ms); the per-decision re-walk is V2 ACCOUNTING of the fresh rollback
snapshot: every compute_unbound is a snapshot miss (1,499/1,499) doing 2 tokens + descendants + full float/row re-validation +
pin rebuild + traversal (replay late 7.8 ms vs V1 walk 4.0). Exact V10 replay reproduces live cache counters exactly.
Recommended: accounting V3 = per-object reachable-set summaries with C-level id/version vectors (bound prototype 1,499/1,499
exact, 1.08 ms late); counterfactual 1,312-1,338/1,499 timely vs 1,197. Non-accounting FREEZE +12 ms trend unresolved.
GT storage-accounting V3 (2026-10-08, OFFLINE ONLY, READY_FOR_INDEPENDENT_OFFLINE_REVIEW; nothing registered/deployed):
`experiments/splitfusion_gt_storage_accounting_v3/REPORT.md` + HANDOFF.md, pkg `rl_agent/splitfusion_gt_storage_accounting_v3/`
(default-off prep seam; reachable(S)-seen per call, rows memo by CPython 3.10 dict tag, tuples by pinned id, sweep, bound
32,768 -> exact fallback; owner shortcut now behind a WHOLE-GRAPH witness: V2 checked only the snapshot although other
frames' ops run between FREEZE and BIND). Tests 75/75, fault injection 14/14 (+1 equivalent), replay V10 8,994 / V8 10,716
calls equal, complete+composite 2,184/2,184 each (incl. interleaved), output 49/49, service proof 15/15. Plan-sealed
criteria all PASS (V10 late composite CPU P50): first call 11.79->3.56 ms (0.30), four-call 20.37->8.81 (0.43), growth
0.19x, every stage lower; short-history BIND calls 0.09->0.16 ms (witness cost). Needs independent review before any V11.
V3 review_01 (HOLD): witness checked ndarray nbytes but not dtype.hasobject (ndarray.__setstate__ -> object dtype at
equal nbytes bypassed the original refusal). Correction_01 -> `.v2` (witness (hasobject, nbytes)); failing-before/passing-
after recorded, review evidence preserved; requalified on final sources: 77/77 tests, faults 15/15, replay/process/output/
service all PASS, criteria PASS (V10_03, host gate met). Authoritative: `experiments/splitfusion_gt_storage_accounting_v3/HANDOFF_02.md`.
V8 GT-prep growth CAUSE (2026-10-08, offline): `experiments/splitfusion_m3_gt_prep_growth_diag_v1/REPORT.md`. ownership.payload_storage_bytes
(4 calls/decision via _account: FREEZE x2, BIND x2) walks staged._restore conflict snapshot whose _PairHistory keeps every pair
observation (310 -> ~10k); exact conflict replay (quarantine 1786/1786; evidence digest NOT reproduced) walk 0.9 -> 9.2 ms/call;
BIND = 11.9 + 2.73*walk R2 0.99; ~45 of 54 ms FREEZE+BIND rise. Value only feeds 64 MiB gate. Proposed (not built): value-exact memo
of _PairHistory size; projection (not measured) 705 -> ~1523/1786 timely.
GT storage-accounting candidate QUALIFIED OFFLINE (2026-10-08, not deployed/live-qualified): `experiments/splitfusion_gt_storage_accounting_v1/REPORT.md`,
pkg `rl_agent/splitfusion_gt_storage_accounting_v1/` (cache-free exact closed forms for flat-scalar dicts/containers; prep-only
switch -> unchanged V8 worker). Parity: 1786/1786 partial V8 states, 182/182 complete real frames + 182 composite, 13 tests;
offline replay 49/49 byte-identical; service_proof_03 PASS (live semantic digests carry observed_raw_ns -> differ even V8 vs V8;
V8-repeat baseline). Late per call 9.29 -> 4.32 ms; residual growth per distinct pair (6.8 us), not per observation. Live needs V9
registration + disk (8.3 GB < 16 GB gate); not prepared.
GOOD_ALL SNR qualification (2026-10-08, radio only): `experiments/splitfusion_m3_good_all_snr_qual_v1/20261008_run01/REPORT.md`.
V8's "17.2 dB" is the MAC-log power-control view (avg_snr + tpc_in_flight, TPC-pinned near pusch_Target 15 + 2 dB at ANY
channel SNR; clean channel reads ~23), not channel SNR. Calibrated per-tx PUSCH snrx10 = scheduler EMA: 21.5 nominal
(-11) -> 21.5 med, P05 21.0 (19.9% at 21.0), MCS 26; 23.5 nominal (-12) -> 23.5 med/23.0 P05/24.0 P95, MCS 27, 0 errors,
8 fps x 112.8 KB. Recommended GOOD_ALL = 23.5 dB (noise -12); needs schedule+actor input change at next registration.
M3.2 V12 BAD_ALL full Route-B capture (2026-10-09, ONE run): `experiments/splitfusion_m3_route_b_v12/LIVE_CAPTURE_REPORT.md`,
pkgs `rl_agent/splitfusion_m3_route_b_v12{,_offline_v1,_analysis}`. V11 unchanged except schedule: BAD_ALL constant 8.5 dB ->
RFsim -4.0 (Phase-14A anchor 09, state 0, feature 0.157895; anchor row hash-bound; host refuses any V11 diff outside the
channel schedule, normalised-section identity). rc 0, CAPTURE_VALID, registered authority PASS 24/24, 301/301, 3,119 frames
all installed, replay/rescore 1,550/1,550, retention 100/100. Fidelity F1-F5 MATCH (PUSCH 7.5/8.5/8.5, MCS 9, 0 retx/errors).
Timely 642/1,550 = 0.414 (V11 0.873): prediction now binding (P50 150 ms vs 131), post-ready tail unchanged ~23 ms;
deadline-aware recall vehicle 0.276 / person 0.213 (V11 0.647/0.433). GC safety stop at decision 1,386 (55.6 ms wall / 12.5 CPU,
preemption) -> stock GC, 13 in-request FREEZE gen2 in last ~164 decisions. Disk 11.24 GB < 16-GiB gate; 82 dangling CN5G
mysql anon volumes (~16 GB, one leaked per bring-up) proposed for deletion, not done. Stopped; nothing further started.
M1 presentation extensions (2026-10-09, existing data only): `experiments/splitfusion_m1_baselines_v1/presentation_ext_01/README.md`
(code `rl_agent/splitfusion_m1_presentation_ext_v1/`). Deadline-aware (pred materialized <=170 ms, late = no detections/empty
mask) recall veh/ped: agent 73.2%/48.0% (exact), AE32/UINT8 <=22.4%/<=16.1% (upper bound: 30 failed decisions lack seg GT,
4 lack all GT), noAE 0/368, 0/254. Unscored-decision eligibility re-derived from archived target_metadata via the scorer's
frozen load_gt + AVO>=0.65 (1,465/1,465 exact). Fixed m11/q0.60 >= agent on timeliness, payload, vehicle recall/IoU; agent
higher on ped IoU; CIs overlap. Compute: full local 271.77 GMACs, UE front 6.47 (2.4%), edge tail 265.30; AE <=+0.55%.
M1 kept no PUSCH SNR (timeline uses commanded actor SNR proxy). Phase B 12-mode q=0 campaign PREPARED NOT LAUNCHED
(`PHASE_B_PROPOSAL.md`): only mode 11 has q=0 in actor support; fixed path needs contract/preflight/verdict/driver
generalization; ~25-39 GB vs ~10.3 GB headroom -> staged; decisions D1-D4 pending.
Static-mode v13 (2026-10-09, PREPARED NOT LAUNCHED): `experiments/splitfusion_static_modes_v13/HANDOFF.md`; pkg
`rl_agent/splitfusion_static_modes_v13{,_offline_v1}` = V12 runtime + FAVORABLE_STABLE retained controller, 1,000-frame
retained budget stop (+V3 finish repair, V12 drain), retention off, actor-free fixed engine (12 modes q=0). All checks PASS
(identities 12/12, regression, service proof 24/24, M1-reproducing assessment, 13/13 startup proofs). Stage 1 = agent,
mode08, mode11 (25.46 GB free vs 16e9 gate + 2.5 GB/arm); ~23 GB more needed for the other 10. Phase-A addendum:
`experiments/splitfusion_m1_baselines_v1/presentation_ext_01/revision_01/`.
Static-mode v13 BATCH 1 LIVE (2026-10-09, authorized; agent/mode08/mode11, 1 attempt each): all VALID_MEASUREMENT,
budget stop signature as registered. Completed<=170 ms: agent 479/493, mode08 (AE64/UINT4 q0) 210/500, mode11 (AE32/UINT4 q0)
453/500; payload P50 117/469/234 kB; deadline-aware vehicle R@1.0 0.632/0.320/0.675. ~2.3 GB per arm; 20.27 GB free.
`experiments/splitfusion_static_modes_v13/batch1_report_01/REPORT.md`. Other 10 arms NOT authorized.
Static-mode v13 BATCH 2 LIVE (2026-10-09, mode03/02/01, 1 attempt each): all VALID_MEASUREMENT (rc 1 = G8-only, as
registered); completed 4/153/1 of 500, 0 within 170 ms (overload; GT-producer refusals 319/197/304 reported, never zero).
Cumulative table: `experiments/splitfusion_static_modes_v13/campaign_report_03/REPORT.md`. 18.65 GB free -> 1 more arm max;
remaining 10,7,0,6,4,9,5 ready but need ~14.9 GB more. Not authorized.
Static-mode v13 BATCH 3 LIVE (2026-10-09, mode10/07/00/06/04/09/05, 1 attempt each) -> CAMPAIGN COMPLETE 13/13 VALID_MEASUREMENT
(mode00 ZERO_COMPLETION). Completed<=170 of 500: m10 292, m7 2, m0 0, m6 0, m4 0, m9 123, m5 0; only AE32 modes + m8 meaningfully
timely; agent best (479/493 at 117 kB) then m11 q0 (453 at 234 kB). Feedback<=170 separate (agent 390, m11 304, m10 47, m8 30, m9 8,
rest 0). m9/m7 deadline-aware = UPPER_BOUND (GT unknown on failed decisions). Sessions differ -> descriptive only. 26.48 GB free,
host clean. Final: `experiments/splitfusion_static_modes_v13/campaign_report_05/REPORT.md` (04 = same data, bad plot layout).
Live target collection and prediction/confidence preservation now have tested additive wiring; deployment and live qualification remain;
then controlled channel response, good-network map checks, two UEs, and finally occlusion alerts.
Diagnose the Run-4/Run-5 comparison before retraining. Keep quality in state/reward and inside the 170-ms clock.
The recent live AVO capture cannot be exactly rescored: it lacks dense predictions and complete vehicle/static target inputs. Offline parity is not a live qualification.
Do not start later stages or new training merely because this sequence is documented.

> **Current execution reset (2026-08-20): split-profile characterization first.**
> The sole current authority is `rl_agent/UE_AGENT_EXECUTION_CHECKLIST_V2.md`.
> The active experiment starts from all 72 measured split-profile anchors and
> characterizes them on one repeatable route under four saved time-varying OAI
> traces.
> SKIP, LOCAL, continuous-q promotion, and policy training follow only after
> the split table and simple baselines. Helper-recipient work is parked in
> `phase2_map_sharing/PARKED_STATUS_2026-08-19.md`. Learned control remains
> conditional on a residual gap after simple baselines. External reviews are
> advisory, not approval gates; scope changes require an explicit
> Abiodun–Codex decision.

## Conventions (do not violate)
- Work only in `abiodun/`; never edit top-level shared scripts (copy into an `abiodun/` subfolder).
- GT convention = **actor origin** offline, and offline knob-matrix numbers are the accuracy anchor (~0.95 m
  loc floor), never a loose-matcher live number.
- **Do NOT export `PYTHONPATH` for a CARLA client** (front/back/loopback) — it shadows `abiodun/` with the
  stale `neu_collab/` copy → `UDPMessageSocket … remote_host`. Analysis/eval scripts that only `import
  carla` are fine.
- T-tracer byte-compares `T_messages.txt` vs the compiled copy → rebuild BOTH softmodems after any edit.
- Don't kill other users' CARLA/OAI; reuse a running server; check `/proc/loadavg` + `docker ps` first.
- Deployed entropy codec = **zstd** (lossless; accuracy codec-invariant).
- Be systematic; validate the pipeline before writing findings; never rescue broken data with "relative
  patterns still hold"; don't lower score gates to force a pass.

## Authoritative docs (source of truth)
- **Current execution authority:** `rl_agent/UE_AGENT_EXECUTION_CHECKLIST_V2.md`.
- **Current split experiment plan:** `rl_agent/UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md`.
- **Current UE diagram:** `rl_agent/state_diagram.md`.
- **Historical measured RL design:** `rl_agent/AGENT_CONSTRAINTS.md` — §9 =
  frozen Phase-1 state/action/reward synthesis; §8 = the density+segmentation
  finding; §1–§6 = the staleness/latency/FPS bounds.
- **Knob matrix (accuracy↔knob↔payload, transport-invariant):** `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md`.
- **Density+seg study (DONE, 9/9 gates):** `rl_agent/density_knob/DENSITY_KNOB_RESULTS.md` — ROI drop
  destroys segmentation; seg-safe knob is `ae32/u4/ROI0 ≈ 90 KB`, density-invariant.
- **Channel sweep (DONE):** `channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` and `combined_surface.csv` —
  106PRB OAI channelmod AWGN surface used by the table-driven policy environment.
- **OAI MCS policy work:** `oai_mcs_policy_track2/` — SINR-driven UL MCS implemented at
  `OAI/…/gNB_scheduler_ulsch.c:2027`; fixes the clean-channel MCS-7 sparse-window artifact.
- **Uplink-only architecture + staleness redo:** `uplink_only_spatial_map_pipeline/`,
  `staleness/uplink_only_latency_budget/`.
- **Papers:** `SCAN_AI_03_13_26_2.pdf` (single-UE foundation), `V2X_for_AD.pdf` (CoDriving).

## Current status / next (2026-08-19)
- Density+seg study, RL design lock, measured loopback latency, and the **channel-condition sweep: DONE**
  (`channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md` + `combined_surface.csv` + `plots/`). Knee: 1 MB→clear
  only, 400 KB→to 15.6 dB, 90 KB seg-safe floor→every rung. `AGENT_CONSTRAINTS §9.1` holds the measured
  channel_state + `payload_budget=capacity/fps` rule.
- The native-10-Hz advisor-rich v5 corpus is accepted for perception QA, workload characterization, and legacy
  matched-support replay. It is **not** a paired helper-recipient causal-control corpus and cannot measure C2
  warning lead. Collection completed 24/24 runs; verification
  `data_collection/experiments/policy_corpus_advisor_rich_v5/20260813_045142_full/verification/20260813_061952`
  is `PASS` on structural controller-corpus gates after excluding only impact run `pcarv5_mixed_va01` (23/24
  retained). Recall is report-only: held-out pedestrian <=12 m is 67.87% and vehicle <=25 m is 67.20%, with
  trajectory-grouped CIs. Accepted-run radar density is 19,404.5/frame; traffic and cleanup are clean.
- Freshness re-score `freshness_rescore/20260813_062203` has no QC exclusions and confirms both slow pedestrian
  and sustained >=10 m/s vehicle regimes, with 54.43% GT-seeded mapped freshness pressure.
- The completed reward-v5 Phase-1 ladder artifact is
  `rl_agent/policy/experiments/controller_ladder/20260813_063514`. Greedy reward is 0.19655 and MPC is 0.19834
  with identical 91.13% matched-safe rate; they disagree on only 2.54% of finite frames. A 2026-08-14 audit found
  same-frame post-tail observation leakage and GT-assisted track construction. These numbers are therefore a
  **noncausal matched-support upper-bound study**, not deployable controller evidence. See
  `rl_agent/RL_JOURNEY_REPORT.md` and `data_collection/EVALUATION_CONTRACT_DECISION_V5.md`.

## RL decision (2026-08-14) — static NO-GO retained; dynamic decision reopened
1. **Single-UE ladder:** greedy ~= MPC (above), bootstrap interval covers zero, but the observation is noncausal.
   This does **not** close the deployable dynamic-controller question.
2. **Expanded action space** (`rl_agent/policy/experiments/expanded_action_gate/20260813_233947_pdt`):
   greedy 0.192625 vs oracle 0.195290 = **+1.383%**, below the registered +5%/+0.01 bar →
   `EXPANDED_SURROGATE_NO_GO_STOP`. **Scope:** the oracle is **one-step, on greedy-visited states, matched-support**
   (`policy/expanded_gate.py:581`) and uses the same noncausal state — it does **NOT** bound causal sequential
   policies. Claim only *"no useful one-step headroom within the noncausal, static-quality, queue-free,
   matched-support contract."*
3. **Multi-UE contention** (`rl_agent/multiue_oai/`): measured N=2 shows **no coordination gap** — the MAC scheduler
   already sits at the capacity ceiling (6.090 vs 6.077 Mbps). Corrected N=50/100 screen: 0/216 cells survive
   (`STOP_CHEAP_NO`). DG-B / N=4 / the campaign / the ladder / RL are all **stopped**.

**Still valid:** the static measured-table profile-selection result and Task C's full-36-profile scalar analysis.
**Untested, therefore NOT falsified:** a causal dynamic controller; scene-conditioned knob selection (`policy/shield.py:49 profile_quality()`
takes no observation → the Phase-1 hypothesis was unrepresentable, not rejected); queue-coupled surrogates;
calibrated LOCAL actions; Phase-2 object-selective map sharing. Reopening RL requires a *new* pre-registered gap on
an expanded contract, not a retune of these gates.

## Phase-15 real-time recovery (2026-09-06) — VALIDATED
- The retry4 deadline failure is repaired and validated on four fresh Route-B cells
  (`experiments/splitfusion_phase15_realtime_recovery_v1/20260906_actions20_71_favorable_fade_retry1`,
  verdict in `..._retry1_reevaluated`, `POST_REPAIR_REALTIME_QUALIFICATION`). The dense 720x1280
  label map is off the radio (result datagrams 102-103 -> 4-5 median) and persisted hash-verified on
  the edge mount; one authoritative capture-based 500 ms deadline is enforced at every stage; both
  FIFOs are bounded latest-frame-first. Median capture-to-install AoI 3,889/33,905/110,237 ms ->
  460/357/486/353 ms; timely installs 0/35,870 -> 176/1,968/3/1,749; the payload-inverse AoI ordering
  is **inverted** (a71 now below a20). All 12 validity gates pass; terminal accounting exact.
- **Preparation coverage is 0.889-0.920, still below the unchanged 0.95 target** (up from 0.67-0.76).
  The target was not weakened. Remaining bottleneck is CARLA sensor-window assembly + radar
  rasterisation (~90 ms of a ~110-114 ms worker period; model path is only 18-23 ms), so closing it
  needs the preparation path parallelised, not a threshold moved.
- **Action 20 is uplink-limited, not defective:** at ~97 datagrams/frame only 218/2,839 features
  arrive complete and fresh (FAVORABLE_STABLE) and 6/3,285 (FADE_RECOVERY, 1,156 incomplete
  reassemblies). That is the payload/deadline frontier, and it is the main input to any later
  split-profile characterization.
- The 16-cell retry4 pilot is retained as `STRUCTURAL_INTEGRATION_QUALIFICATION`; it was not rerun.
  The 288-cell campaign remains unauthorized.
- **Pin warning:** three independent SHA-256 pin layers guard this path (campaign config, the
  Phase-14A binding's `map_install.artifacts`, and the SFD1-v2 authority). The first launch attempt
  died at the OAI launcher because `ue_map_install_feedback_v1.py` is pinned in the Phase-14A
  binding. Verify all enforced pins before any live launch.

## Direct edge-to-map architecture correction (2026-09-14) — VALIDATED

The completed 288-cell campaign routed compact object results **edge -> UE `10.0.0.2:51004` over the radio
downlink -> UE-forwarded publication -> map on localhost**. The spatial map is an edge application, so that
detour was an architectural defect: it put the map update on the downlink and made installation depend on UE
reachability. Corrected path: **edge inference -> direct edge-to-map publication/install -> compact
feedback/agent-credit message to the UE**.

- **Implementation is additive** in `rl_agent/splitfusion_direct_edge_map_v1/`. Every SHA-pinned module
  (`live_pilot_runtime.py`, `ue_route_b_split_cell_adapter_v1.py`, the baseline map server,
  `ue_map_install_feedback_v1.py`, `edge_runtime.py`, `context_tail.py`) is **imported unchanged** — hashes
  still match their config pins. The architecture audit is `splitfusion_direct_edge_map_v1/PHASE_A_ARCHITECTURE_AUDIT.md`.
- **Direct endpoint:** the host's address on the CN5G bridge, resolved from
  `docker network inspect oai-cn5g-public-net` -> `192.168.70.129:39320`. Link-scope container-to-host; it never
  enters the UPF, GTP or RFsim. The publisher refuses the UE tunnel, loopback, `0.0.0.0` and the UE result ports.
- **Live validation DONE:** `experiments/splitfusion_direct_edge_map_v1/20260914_live_validation_retry1`,
  4/4 cells PASSED, **16/16 gates PASS**, actions 15/30/50/71 under FAVORABLE_STABLE. Terminals exactly equal
  frames sent (2967/2923/2969/2939). Live direct tail->install p50 = 9.0-14.5 ms; ACK arrival p50 = 0.75-2.84 ms,
  measured separately and never folded into map AoI.
- **Corrected 288-cell counterfactual DONE:**
  `experiments/splitfusion_direct_edge_map_v1/20260914_direct_map_288_counterfactual`. Only
  `post_publication_install_ns` changed; transport/admission/edge-service counts are asserted identical between
  the two replays. Pre-registered Kruskal-Wallis retained **PER_FAMILY** pooling (p=8.9e-27).
  **Headline:** the detour imported the downlink's channel dependence into map freshness. Post-edge install path
  71.5 ms -> ~11-13 ms under ADVERSE_STABLE; median cell map AoI 343.6 -> 295.6 ms there, 307.2 -> 297.4 ms
  overall. Corrected map AoI is nearly channel-independent (290.3/298.7/295.6/299.8 ms across the four profiles).
- **Figures/tables:** `experiments/splitfusion_direct_edge_map_v1/20260914_presentation`.
- **Scope:** the counterfactual is a model, not a remeasurement. `10.0.0.2` is local to this host
  (`oaitun_ue1`), so map->UE feedback is host-local kernel delivery and the ACK latency is a **lower bound** on an
  over-the-air control delay. **No further 288-cell live campaign is authorized.**

## Sensor-preparation optimization (2026-09-14) — VALIDATED

`SENSOR_PREPARATION_OPTIMIZATION_VALIDATED`. Report:
`rl_agent/splitfusion_sensor_optimization_v1/SENSOR_PREPARATION_OPTIMIZATION_V2_REPORT.md`;
evidence `experiments/splitfusion_sensor_preparation_live_v1/20260914_v2_*`.

- **Diagnosis:** the cost sat in two *already vectorized* stages. `np.maximum.at` (5 calls,
  3.52 ms) dominated rasterization, and `FastStationaryTrackAccumulator` sorted the same data
  twice (`np.unique(return_inverse)` + `argsort(inverse)` = 2.95 ms) because `inverse` is only a
  rank-preserving relabelling of `packed`.
- **Integrated, all bit-exact:** CUDA `scatter_reduce(amax)` + `max_pool2d` rasterizer
  (7.03→1.02 ms incl. H2D/D2H/sync); single-sort tracker (4.62→2.94 ms); identity radar-resize
  short circuit (`build_radar_sample` already rasterizes at 768x448).
- **Live (full sent population):** sensor compute P50 45.844→32.024, P95 85.984→64.519,
  P99 125.440→104.492 ms; radar chain P50 31.705→21.525 ms; coverage 0.956825→0.986347;
  14/14 gates on both cells; 7-channel tensor and radar evidence exactly equal live.
- **Targets NOT met** (P95<50, P99<100, chain 12–15 ms): they were anchored to a 37.662 ms
  historical baseline, and today's contemporaneous baseline is 45.844 ms on a busier host.
  In the pre-registered 500-frame window P99 *regresses* (107.3→117.4) while P50/P95 improve;
  over the full population P99 improves. Both are reported — do not quote only one.
- **Method note worth keeping:** the *previous* attempt was confounded because every stage,
  touched or not, regressed together between its cells. The fix is a same-frame paired
  measurement plus the baseline cell as an ordering control (production-vs-production still
  shows 2.325 ms, so the raw 11.436 ms paired reduction corrects to ~9.1 ms).
- **Largest remaining tail:** `radar_window_ms` P99 30.663 ms (P50 5.338) then
  `spherical_to_world`. These are the world→spherical→world round trip: removing it saves
  1.107 ms but shifts world XYZ by up to 6.49e-06 m, so it is **not bit-exact** and is left as
  an explicit Abiodun decision, not taken.
- **Do not repeat:** the normalization-constant cache (real but tiny), a shared flat-index
  sort feeding `maximum.reduceat` (slower), and the `transform_points` rewrites (0.02 ms).

## Optimized 288-cell policy analysis (2026-09-14) — COMPLETE

`experiments/splitfusion_supervisor_analysis_v1/20260914_optimized_pipeline_policy_analysis_v2`
replays the immutable 288-cell sent population with measured radio
reassembly/admission outcomes fixed, equal-percentile live sensor optimization,
newest repaired-v3 family edge medians, renderer-off direct map service, and the
qualified one-slot latest-only scheduler. It is an offline counterfactual, not a
second live campaign. Inventory: 896,856 sent; 712,732 complete reassemblies;
602,315 measured edge admissions; 586,853 simulated map installations. Figures
01a–04f keep semantic, vehicle/person overlap and centroid error separate and
restore joint segmentation-localization quality as view `f`; Figure 05 is the
P50/P95/P99 causal-stage breakdown, Figure 06 is the live sensor before/after
breakdown with presentation-local labels P01–P18, and Figure 07 reports complete feature delivery
by profile. Figures 01 and 04 begin at seven-channel concatenation rather than
RGB capture. The joint score is a presentation coordinate, not a
calibrated probability or the PPO reward. One send-finished/reassembly timestamp
boundary inversion was excluded rather than clamped; all imputed scheduler
arrivals remain excluded from the feature-uplink plots.

## Current work (scope-reset 2026-08-20)
- **Binding path:** follow
  `rl_agent/UE_AGENT_EXECUTION_CHECKLIST_V2.md`. Qualify the repeatable route,
  calibrate the 100-ms OAI SNR-trace actuator, validate authoritative map
  feedback, then run the 4x4 pilot before separately authorizing the 72x4
  split-profile characterization. SKIP/LOCAL and policy training remain later.
- **Phase 2 parked:** recipient-specific map sharing, warning, factor
  realization, exact-16, and paired CARLA/OAI work are preserved but inactive.
  They resume only under the conditions in
  `phase2_map_sharing/PARKED_STATUS_2026-08-19.md`.
- **Positive result:** load-shaping deadline-feasibility frontier — 7/200 feasible cells at 400 KiB → 58/200 at
  90 KB → 89/200 at 49.4 KB (even the smallest payload leaves ~55% infeasible).
- **Banked later-stage gap: C2, the transport-conditioned cooperation gain, does not exist yet.** The 1.40 m two-view
  triangulation result is groundwork only — static egos, oracle association, **no OAI transport**.
- **Superseded ordering:** the earlier 7-10 week helper-recipient critical path
  remains a paper-stage estimate, not the current implementation sequence.
- **Task A complete:** the exact 36-profile/1,683-frame screen includes the already-available per-frame
  segmentation metrics and finds no practical class/range reversal under the registered gate. True occlusion and
  cyclists remain outside scope. Artifact: `rl_agent/contextual_knob/experiments/20260814_214749`.
- **Task B complete:** hard observed-vulnerable no-skip + low-confidence ROI0 clamp, with an explicit C1-conflict
  flag. Paired replay improves matched-safe rate +1.63 pp at a finite-reward cost of -0.0477 and +1.10 Mbps;
  detector misses are not protected. The rule/contract is valid, but those empirical deltas inherit the noncausal
  replay caveat. Artifact: `rl_agent/policy/experiments/vulnerable_guardrail/20260814_215337`.
- **Task C complete:** full-36 lambda-RDO agrees with exact enumeration at 80.56% of payload breakpoints (max
  utility gap 0.011686; max duality gap 0.017359). The retained-catalog runtime ladder agrees 100% with zero reward
  gap only inside the noncausal matched-support replay. The AoI baseline is explicitly index-inspired, not
  Whittle. Artifact:
  `rl_agent/policy/experiments/task_c/20260814_220006`.
- `SCENESENSE_MONTHLY_CHECKLIST.md` is reconciled through 2026-08-17. **Phase-2 contract plumbing exists:**
  `phase2_map_sharing/` passes synthetic contract validation for recipient isolation, causal hazard-only selection,
  association, warning provenance, exact byte accounting, and production-header chunk reassembly. This is plumbing,
  not C2 evidence. Its adapter also passes the existing two-stream recordings (26 fresh accepted, 11 stale rejected),
  which lack synchronized hazard truth. The `phase2_paired_causal_v1` spec and v2 collector/replay/verifier path are
  now integrated offline: strict schema/unknown-field rejection, object+recipient uncertainty propagation,
  pre-capture causal audit, physically separate truth, source-local tracking, quota-bounded raw/logit retention,
  isolated three-arm replay, and the nine fail-fast gates. The reviewed legal-route two-trajectory pilot is complete
  at `data_collection/experiments/phase2_paired_causal_v1/20260817_181354_pilot`; authoritative
  `evaluation_v4` / `verification_v4` pass all nine structural/computability gates. The evaluation-only
  `hazard_adjudication_v2` also passes: positive hazard truth uses the matched benign no-yield recipient trajectory,
  while realized stopping remains explicitly non-policy-attributable. The v1 adjudication is superseded due to the
  intervention paradox. The deterministic `phase2_suite_ab_v1` candidate now fixes Suite A=designed and Suite
  B=naturalistic across 210 independent groups / 330 world trajectories with hashed 20/20/60 assignments,
  conditional power and tiered retention. The signalized-corner and parked-van-midblock pedestrian geometries are
  visually accepted and route-hash frozen. Full collection, OAI evaluation, and RL remain unauthorized until three
  pending geometries and two paired routes are visually accepted and the 15-trajectory calibration audit plus
  registered simulation-power gate pass. The renderer gate is now resolved operationally: every primary Phase-2
  design row locks explicit CARLA `Epic` (`-quality-level=Epic`). Low survives only as the already-captured labelled
  stress condition; no future Low collection is authorized. The frozen <=12 m dense weighted comparison was
  inconclusive because it had zero near-pedestrian support, so do not claim Epic statistically dominated every
  class or retroactively relabel the unrecorded M-prime training renderer.

> The `~/.claude` memory cache was wiped by a retention cleanup on 2026-08-03 (harness, not us). This
> repo-tracked file exists so project state is never lost that way again. Keep it updated as work advances.
