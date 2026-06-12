# Unattended Multimodal Fusion Training — Instructions To Paste Back To Claude

Save this file. When you want a fresh end-to-end run, open a Claude session in this
repo and paste **the prompt at the bottom of this file** (the section titled
"PROMPT TO PASTE TO CLAUDE"). Everything Claude needs to act autonomously is in
this document.

---

## 1. Goal

Train a traffic-light-pole **RGB + radar early-fusion split-inference model** in
CARLA Town10HD that:

1. **Segments** persons and vehicles from RGB pixels (LR-ASPP head).
2. **Detects and localizes** vehicles and persons by **direct neural regression**
   in a CenterFusion-style object head — outputs object center heatmap,
   sensor-relative XYZ, dimensions (l,w,h), yaw (sin/cos), parked/stopped state,
   and a radar-support confidence. Global x,y is recovered by transforming the
   predicted sensor-relative location through the saved camera calibration.
   **No classical post-processing localization** is used as the primary metric.
3. Performs **early fusion** of radar in the first half of the model: radar
   returns are spherically transformed, projected into the RGB image plane,
   rasterized into per-pixel channels (occupancy, inverse range, radial velocity,
   stationary-age), and concatenated with RGB before the backbone (or injected
   into early backbone features). The split point lives in the backbone so the
   first half is the fused RGB+radar feature extractor, matching the existing
   `carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py`
   transport convention.
4. Includes a **temporal stationary-age accumulator** and lightweight
   association tracker so a parked car (≥5 s stationary) and a stopped car
   (<5 s) are distinguished as a learned label, not a hand-rolled rule at
   inference time.
5. Is trained on data sampled from **all eligible traffic-light poles** in
   Town10HD at **6 m mounting height**, sweeping traffic density, pedestrian
   density, FoV, yaw and pitch. Weather and lighting stay fixed.
6. Uses **CARLA instance segmentation** as pixel ground truth and CARLA actor
   bounding boxes / velocities as object ground truth.

The pipeline must run **fully unattended** for 5–6+ hours: data collection,
training over a hyperparameter sweep, validation, test, and a final report —
with automatic CARLA restart on crash and resume from the last completed stage.

## 2. Success criteria

A run is **successful** when **all** of the following hold on the held-out test
split (read `metrics/test_fusion_evaluation_metrics.json`):

| Metric                                  | Target              |
| --------------------------------------- | ------------------- |
| `miou` (3-class average)                | ≥ 0.85              |
| `vehicle_iou`                           | ≥ 0.85              |
| `person_iou`                            | ≥ 0.55 (not NaN)    |
| `learned_object_recall` (vehicles)      | ≥ 0.60              |
| `learned_object_precision`              | ≥ 0.55              |
| `learned_object_f1`                     | ≥ 0.55              |
| `learned_global_xy_mae_m`               | ≤ **1.0**           |
| `learned_dimension_mae_m`               | ≤ 0.6               |
| `learned_yaw_mae_deg`                   | ≤ **10**            |
| `learned_parked_accuracy`               | ≥ 0.80              |
| `fusion_miou_delta_vs_rgb`              | > 0 (radar must add)|

The `xy_mae` and `yaw_mae` targets were tightened on 2026-05-07 after the
3-iteration sweep delivered xy_mae 2.43 m / yaw_mae 19.9° with the high-only
object head. Hitting these tighter targets requires (a) the `fuse_low_feature`
flag in `object_heads` so the head sees the 1/8-stride `low` feature alongside
1/16-stride `high`, (b) substantially more data than the prior 12k-sample
collections, and (c) higher epoch counts. The previous run-3 best.pt is
preserved at
`pole_lraspp_multimodal_fusion/preserved_checkpoints/run3_high_only_object_head_best.pt`
to revert to if a new training cycle regresses.

If any object-head metric is NaN or 0, the run is **not** successful, regardless
of how good segmentation looks. The original failure mode that this instruction
set was written to fix (`learned_object_f1 = 0` from a heatmap-target encoding
bug) is documented in `experiments/.../20260506_144909_*/final_summary.md`.

## 3. Existing scaffolding (do not rebuild)

- Workflow root: [pole_lraspp_multimodal_fusion/](.)
- Pipeline package: [pole_lraspp_multimodal_fusion/](pole_lraspp_multimodal_fusion/)
  - `run_pipeline.py` — supervisor: collection → splits → per-trial training → val → test → report. Handles CARLA restart and `--resume auto`.
  - `collect_dataset.py` — multi-pole sensor capture, instance-seg + actor-bbox ground truth, radar tensor rasterization, stationary-age tracker.
  - `radar_fusion.py` — spherical→world→image radar projection, per-channel rasterization, optional learned-injection helpers.
  - `model.py` — fused LR-ASPP backbone + segmentation head + object-detection head; split point inside backbone.
  - `object_targets.py` — center-heatmap / xyz / dims / yaw / parked target encoding and decoding.
  - `train_fusion.py` — multi-task loss (seg + center + location + dims + yaw + parked + radar-support), AMP, early stop.
  - `evaluate_fusion.py` — segmentation metrics, learned-object matching to GT, baseline RGB-only comparison.
  - `split_runtime.py` — split-inference packaging consistent with the existing UDP transport.
- Configs: [configs/fusion_full_run.yaml](configs/fusion_full_run.yaml), [configs/fusion_smoke.yaml](configs/fusion_smoke.yaml)
- Host-shell scripts:
  - [launch_unattended_fusion_training.sh](launch_unattended_fusion_training.sh)
  - [status_unattended_fusion_training.sh](status_unattended_fusion_training.sh)
  - [stop_unattended_fusion_training.sh](stop_unattended_fusion_training.sh)
- RGB-only seed checkpoint (used for warm init AND as the comparison baseline):
  `experiments/pole_lraspp_training/20260505_173329_pole_lraspp_training/checkpoints/adamw_640x360_lr1e-4_wd1e-4_aug_medium_bs6/best.pt`

Every experiment writes to:
`experiments/pole_lraspp_multimodal_fusion/<YYYYMMDD_HHMMSS>_pole_lraspp_multimodal_fusion_learned_localization/`

## 4. Why Claude cannot launch this itself

Claude Code's `Bash` tool runs in a sandboxed shell. CARLA + multi-hour GPU
training does not survive that sandbox cleanly (no display server, no long-lived
process, sandbox tear-down on session end). **The user must run the launcher
from a normal host shell** (`screen` or `nohup`). Claude prepares the config,
validates the launcher with a dry run, watches the experiment directory while
the user is away, diagnoses problems, edits configs, and re-launches — but the
user is the one who actually starts the long-running process.

This split is documented in the workflow's [README.md](README.md) and enforced
by `manifest.json: "execution_boundary"`.

## 5. Pre-flight: what Claude does before you launch

When you paste the prompt at the bottom of this file, Claude will, before asking
you to launch, **without your intervention**:

1. Confirm the launcher, configs, and pipeline package are present and the venv
   Python is executable.
2. Read the most recent experiment under
   `experiments/pole_lraspp_multimodal_fusion/` and summarize its state from
   `manifest.json` and `final_report.txt`.
3. If the previous run had `learned_object_f1 = 0` or any NaN object metric:
   - Read `pole_lraspp_multimodal_fusion/object_targets.py`,
     `model.py`, `train_fusion.py`, `evaluate_fusion.py`, and
     `radar_fusion.py`.
   - Identify the bug (likely candidates: heatmap target sigma vs decoder peak
     threshold mismatch, sensor-relative XYZ target normalization not inverted
     at decode time, NMS radius eating all peaks, `min_gt_area_px` filtering
     out everything, GT actor frame mismatch with camera calibration, AMP
     casting break in regression heads, focal-loss alpha/gamma making the
     gradient vanish, target tensor stride mismatch with prediction stride).
   - Apply a focused fix and add an assertion or unit-style sanity check in
     the affected module. **Do not refactor unrelated code.**
4. Refresh `configs/fusion_full_run.yaml` with the targeted updates listed in
   §6 below.
5. Run the launcher in `--dry-run` mode to validate config parsing and
   experiment-dir creation:
   `./launch_unattended_fusion_training.sh --dry-run`
6. Print the exact host-shell command you should run, the expected experiment
   directory path, and the monitoring commands.

Claude must not start the long run itself. If it tries to invoke the launcher
without `--dry-run`, the run will likely die when the Bash sandbox tears down.

## 6. Required config refinements vs the prior run

Apply these edits to `configs/fusion_full_run.yaml` before the next launch.
They address the specific failure modes of the 2026-05-06 13:11 run.

1. **Pedestrian coverage.** Change `collection.pedestrian_densities` to `[16, 32, 48]`
   (drop 0). The prior run got `person_iou = NaN` because too many scenes had
   no pedestrians. Optionally add `min_person_pixels_per_frame: 200` and
   resample frames that fall below.
2. **Object-head sanity.** In `object_heads`, lower `min_gt_area_px` from 24 to
   12 so distant cars are kept as positives. Set `heatmap_radius_px` to a
   gaussian sigma derived from object size (already supported in
   `object_targets.py`); if not, hard-set `heatmap_radius_px: 3`.
3. **Decoder consistency.** In `evaluation`, set
   `object_score_threshold: 0.15` (was 0.25 — too strict given the prior
   run's heatmap distribution), `object_nms_radius_px: 4`, and
   `topk_objects: 80`. Increase `match_distance_m` to `7.5` for the first
   re-run so a partially-trained head is not penalized by hard matching.
4. **Loss reweighting.** Under `training.loss_weights.object`, raise `center`
   to `2.0` and `location` to `0.20` (was 0.05 — too small to drive the head).
   Keep `dimensions: 0.2`, `yaw: 0.05`, `parked: 0.2`, `radar_support: 0.1`.
5. **Class balance.** Add `training.class_loss_weights: [0.5, 1.0, 4.0]` for
   background/vehicle/person to compensate for the rarity of pedestrian pixels.
6. **AMP for regression heads.** Ensure `train_fusion.py` keeps the regression
   head FP32 (the segmentation head can stay AMP). If the file currently casts
   regression head outputs in autocast, wrap them in
   `with torch.cuda.amp.autocast(enabled=False):` and cast targets to FP32.
   Add a comment only if the workaround is non-obvious.
7. **Trial sweep.** Replace the three trials with these three; the SGD trial in
   the prior run scored `-inf` so SGD is dropped:

   ```yaml
   trials:
     - name: fusion_v2_adamw_512x288_lr3e-4_radar4_aug_medium_bs6
       optimizer: adamw
       lr: 0.0003
       weight_decay: 0.0001
       augment_strength: medium
       input_size: [512, 288]
       batch_size: 6
     - name: fusion_v2_adamw_640x360_lr1.5e-4_radar4_aug_medium_bs4
       optimizer: adamw
       lr: 0.00015
       weight_decay: 0.0001
       augment_strength: medium
       input_size: [640, 360]
       batch_size: 4
     - name: fusion_v2_adamw_768x432_lr1e-4_radar4_aug_strong_bs2
       optimizer: adamw
       lr: 0.0001
       weight_decay: 0.0002
       augment_strength: strong
       input_size: [768, 432]
       batch_size: 2
   ```
8. **Budgets.** `runtime_budget_hours: 6.0`, `collection_budget_hours: 2.0`,
   `training_budget_hours: 3.5`. Leave 0.5 h for evaluation + report. If the
   GPU is faster than expected, the supervisor uses any saved time to re-run
   the best trial at higher epochs.
9. **Resume contract.** Always launch with `--resume auto` so a CARLA crash
   does not throw away completed scenarios.

## 7. The exact host-shell command you run

Paste this into a normal terminal on the remote host (not the Claude pane). It
detaches via `screen`, survives VS Code disconnect, and writes a screen log:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/pole_lraspp_multimodal_fusion

./launch_unattended_fusion_training.sh \
  --mode screen \
  --session-name pole_lraspp_fusion \
  --config configs/fusion_full_run.yaml \
  --resume auto
```

The launcher will print:

```
screen_session=pole_lraspp_fusion
attach_command=screen -r pole_lraspp_fusion
screen_log=.../background_logs/pole_lraspp_fusion.screen.log
experiment_dir=.../experiments/pole_lraspp_multimodal_fusion/<stamp>_pole_lraspp_multimodal_fusion_learned_localization
```

Send the printed `experiment_dir` line back to Claude so the monitoring loop in
§8 has the right run.

If `screen` is unavailable, use `--mode nohup` instead — same arguments. Both
modes survive SSH disconnect.

To stop the run cleanly:

```bash
./stop_unattended_fusion_training.sh
```

## 8. What Claude does while the run is going

Claude does not poll on its own. Each time you say "check on the run" (or
similar), Claude will:

1. Read `latest_run.txt` to get the current experiment directory.
2. Run `./status_unattended_fusion_training.sh` and parse the JSON.
3. Tail `supervisor.log`, the current stage's `logs/*.log`, and the latest
   line of `status.jsonl`.
4. Report (in <10 lines): stage, elapsed, samples collected so far,
   per-trial best mIoU, last CARLA restart count, and any error.
5. If the supervisor is dead but no `completed_at` is set, restart it with
   `--resume auto` (instructions printed back to you to paste — Claude will
   not run the long launcher itself).

If Claude detects a stage failure that is config-fixable (e.g. OOM on the
largest trial, target/decoder mismatch surfaced by a logged assertion), it will
edit the config, ask you to stop the run, and re-launch with `--resume auto`
from the next trial.

## 9. Post-run: how Claude decides to iterate

After `manifest.json` says `status: complete`, Claude reads
`metrics/test_fusion_evaluation_metrics.json` and compares to §2.

- **All targets met** → Claude writes a one-page summary, saves a memory
  entry pointing to the best checkpoint and the experiment directory, and
  stops.
- **Some targets missed** → Claude diagnoses (which metric, which trial,
  which loss curve) and proposes one focused config change. It does **not**
  trigger a sweep over many hyperparameters. It picks the single most likely
  cause, edits `fusion_full_run.yaml`, writes a short rationale into the
  experiment directory's `next_iteration_notes.md`, and asks you to launch
  the next run.
- **Any object metric is 0 or NaN again** → the bug is in the model/target
  code, not hyperparameters. Claude reads the relevant module, fixes the
  code, and asks you to re-launch.

The total iteration cap is **three** unattended runs in one session, ~6 h
each. If targets are still missed after three runs, Claude stops and writes a
diagnosis report rather than burning more GPU time blindly.

## 10. Crash recovery (already handled, do not re-implement)

`run_pipeline.py` already supervises CARLA. The contract:

- On CARLA crash, the supervisor SIGKILLs orphan `CarlaUnreal.sh`, waits
  `restart_cooldown_s`, relaunches via `carla.server_command`, polls until
  the RPC port is open or `startup_timeout_s` elapses, then resumes the
  current scenario. Bounded by `max_restarts: 12`.
- On supervisor SIGTERM (from `stop_unattended_fusion_training.sh`), it
  finishes the current frame, saves a checkpoint of the in-progress trial
  (if any), updates `manifest.json`, and exits 0.
- `--resume auto` re-reads `manifest.json` and the dataset manifest, skips
  any stage with a non-empty `*_finished_at`, and resumes the current stage
  from the last complete sub-step.

If a future run reveals that the supervisor itself is the failure mode (e.g.
cooldown too short, port-poll false-positive), Claude will fix
`run_pipeline.py` directly. Do not work around it in the launcher.

## 11. What Claude must not do

- **Do not** run the long launcher from the Claude Bash tool. Dry-run only.
- **Do not** delete prior experiment directories. Each run is a sibling.
- **Do not** edit checkpoints from prior experiments.
- **Do not** add backwards-compat shims for renamed config keys; just rename
  them and update both configs.
- **Do not** silently widen success thresholds in §2 to declare a run a
  success. If a target is too tight, propose changing it to the user first.
- **Do not** add classical-post-processing localization back into the
  evaluation as the primary metric. The diagnostic flag
  `evaluation.classical_radar_diagnostic` may be set to `true` only as a
  *secondary* sanity check in `evaluate_fusion.py` — it must not be used to
  decide success.
- **Do not** introduce new dependencies. Stick to PyTorch, torchvision,
  numpy, opencv-python, pyyaml, pandas — already in the venv.

---

## PROMPT TO PASTE TO CLAUDE

Paste **everything between the fences** as a single message to start a new run.

```
I want to run another unattended end-to-end multimodal-fusion training. Please
follow /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/pole_lraspp_multimodal_fusion/UNATTENDED_TRAINING_INSTRUCTIONS.md
exactly. Do the full pre-flight from §5: read the latest experiment under
experiments/pole_lraspp_multimodal_fusion/, summarize its state, fix any
bugs you find in object_targets.py / model.py / train_fusion.py /
evaluate_fusion.py / radar_fusion.py that explain the prior run's
learned_object_f1 = 0 or any NaN learned-object metric, apply the §6 config
refinements to configs/fusion_full_run.yaml, run the launcher with
--dry-run, then print the exact host-shell screen command from §7 that I
should run on this machine, plus the experiment directory I should send
back to you. Do NOT launch the long run yourself. After I confirm the run
is going, you will only check on it when I ask. After the run completes,
follow §9 to decide whether to iterate; up to three unattended runs total
this session.
```

End of file.
