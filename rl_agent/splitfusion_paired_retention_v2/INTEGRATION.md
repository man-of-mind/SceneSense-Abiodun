# Paired-input retention v2: integration notes (NOT INSTALLED, NOT AUTHORISED FOR LIVE USE)

No launcher imports this package. `RetentionConfigV2()` is disabled, and `make_observer()` then returns a
no-op. Turning it on requires a new prospective registration and Abiodun's explicit authorisation.

## What a run with retention enabled is
A **quality diagnostic**. In POST_DRAIN mode about 100 x 9.63 MB (~0.96 GB) of prepared tensors stay referenced
in the UE child until drain. In BACKGROUND_SPILL mode a small RAM window is kept, and one extra writer thread
does about 9.6 MB of write, fsync and SHA work per sampled decision. Either way, allocation and I/O pressure
change. Do not quote such a run as an uninstrumented timing measurement. Its registration must say this.

## Seams (wrapped at runtime by `observer.install_retention`; no file is edited)
| v1 seam (`splitfusion_paired_quality_replay_v1/scope.py`) | current numeric-GT Run-5 pipeline equivalent |
|---|---|
| source-rewrites `return fused.unsqueeze(0).to(...)` in `splitfusion_live_parity_repair_v1/perception.py:187` (`prepare_live_input`) | That function is **no longer the live input path**. `entries.py:69-72` installs `install_bgr2rgb_candidate()` and then `install_fused_candidate()` (`splitfusion_ue_input_prep_v2/candidate.py:72-82`), which sets `live_pilot_runtime._prepare_live_input` (`splitfusion_live_dispatch_v1/live_pilot_runtime.py:682`) to `prepare_live_input_fused`. v2 rewrites that function's single final `return out.to(device=device, dtype=torch.float32)` (`candidate.py:60`). The rewritten function is installed as `B._prepare_live_input`. It is reached late-bound from `phase6_ue_runtime_v2.py:622` (`input_7ch=lambda: base._prepare_live_input(...)`). `out` is a fresh host tensor on every call, so retaining it by reference is safe. |
| wraps `Run4FramePipelineV2.materialize` (`phase6_ue_runtime_v2.py:213`; input at :232, SFD4 wire at :236) | It is the same method. Note that `install_ue_pose` already wraps it (`splitfusion_live_parity_runtime_v1/pose.py:391`, installed at :418). The retention scope must be the **innermost** context, entered after `install_fused_candidate()`, `install_ue_pose(...)` and `install_early_gt_ue(...)` in the with-stack of `splitfusion_run5_full_live_v1/entries.py:69-72`. It is exited first, and it restores only objects that are still its own. |
| `archive.flush(collectors=...)` after `worker`/`evaluation_worker` are dead (`ue_route_b_split_cell_adapter_v1.py:1247,1253`) | Call `observer.finalize(directory, workers=[collector.worker, collector.evaluation_worker])` after `CHILD.main` returns. The early-GT UE threads (`splitfusion_exact_gt_live_v2/ue.py:153,155`) never touch the observer. |
| (none) | `observer.notify_feedback(frame_id)` is optional. It could be called beside `engine.on_feedback` (`phase6_ue_runtime_v2.py:696`). Without it, spill eligibility falls back to `spill_lag_decisions` later decisions (default 2, about 400 ms at the 2-tensor decision cadence, which is more than the 170 ms deadline). |

GT is **not** captured in the UE. The host service already archives each scored decision at
`capture/cells/<cell>/numeric_gt/service/archive/frame_NNNN/{producer,scorer}/`
(`splitfusion_exact_gt_service_proto_v2/scorer_core.py:301` `archive_build`, made durable by
`splitfusion_live_gt_handoff_v1/archive.py`). Postflight, `gt_adapter.resolve_gt_references` binds every retained
record to that archive by path and SHA-256. It requires exact identity joins: received action identity, GT
identity, capture ns, (session, tensor, frame), and the wire FrameContext equal to the target `frame_context`. It
also requires exact rescoring (`numerical_score.json` rebuilt byte-for-byte, and the `evaluate_prepared`
measurement identical). A decision that was never scored, because GT or the prediction was missing, has no
archived native prediction. It is then `INCOMPLETE_GT_*` and refused at admission. Retaining unscored predictions
would need an edge-side hook beside `splitfusion_exact_gt_live_v2/edge.py` (the `EarlyGtEvaluatorV1` prediction
hand-off). That hook is not implemented.

## Pins that must be registered (SHA pin cascade)
* `observer.SEAM_PINS` checks `splitfusion_ue_input_prep_v2/candidate.py`, `phase6_ue_runtime_v2.py` and
  `live_pilot_runtime.py` at install. Any drift refuses the scope.
* Add `rl_agent/splitfusion_paired_retention_v2/{plan,observer,writer}.py` and the **pre-launch plan file digest**
  to the Run-5 registration's `exact_gt_sources`. These are verified in `entries.py:31-37` and
  `host.py:133-134` through `contract.exact_sources`. Because `contract.own_sources` only globs its own package, a
  versioned `splitfusion_run5_full_live_v2` contract (or an explicit support-source entry) is needed. Do not edit
  the v1 contract.
* The retained v2 pipeline proofs bind child sources by hash. They are reused today only through the registered
  substitutions `splitfusion_gt_archive_capacity_v1.contract.substitution_ok` (2 files) and
  `splitfusion_gt_mask_lookup_v1.contract.substitution_ok` (3 files). Adding the retention scope to the child
  changes its composition, so it needs a new registered substitution or a re-proof, plus a 100-frame diag first.
* The deep SFD1 layer (`edge_runtime.py`, `context_tail.py`, `frame_context_binding.json`,
  `runtime_binding.json`) and the campaign-config pins on `live_pilot_runtime.py` and
  `ue_route_b_split_cell_adapter_v1.py` are **not touched**. Retention only rebinds module attributes in-process.
* Seal at postflight: the retention `MANIFEST.json` digest, the plan digest, and the `GT_REFERENCES.json` digest in
  the run's postflight inputs. `validate.admit_for_paired_replay` takes all three as externally pinned values.

## Disk and memory budget
* Plan (1,000 frames): 500 planned decisions, stride 5, offset 2, 100 ordinals. About 493 real decisions give 99
  retained records and 1 `NOT_REACHED`.
* RAM: POST_DRAIN cap is 1.25 GiB (100 x 9,633,792 B = 0.963 GB plus wires). BACKGROUND_SPILL default cap is
  (queue 4 + lag 2 + 1) x (9.63 MB + 2 MiB), about 82 MB.
* Disk: about 0.97 GB of retention (`max_storage_bytes` default 1.25 GiB, hard ceiling 2 GiB), on top of about
  1.9 GB of normal capture. The writer refuses if writing would leave less than `min_free_disk_bytes` (default
  4 GiB). On 2026-10-06 the root volume had **13 GB free (100% used)**. Check before any launch.

## Pre-launch sequence (when authorised)
1. Use `plan.build_plan(plan_id=..., planned_frames=<transmitted_budget>)` and `plan.write_plan(path)`
   (create-only). Record the digest in the registration.
2. Start the service offline with the registration limits, as for any Run-5 launch.
3. Run a 100-frame diag with retention on. Then run `verify_archive_v2`, `resolve_gt_references` and
   `admit_for_paired_replay`. Check selected-branch byte/label/record parity with
   `splitfusion_paired_quality_replay_v1.replay` before using the 1,000-frame run.
