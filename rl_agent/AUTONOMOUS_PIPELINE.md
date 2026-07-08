# Autonomous pipeline — drop-aware M′ → complete knob-relationship matrix → RL policy
*Long unattended operation. Goal: the full relationship of {quantization, ROI, AE} vs
{accuracy, latency, reliability, payload} on the ROI-robust model M′, to derive the RL policy.*

## Stage graph (dependencies + decision gates)
```
0. PREREQ (interactive): build + smoke-test objectness-drop hook in train_fusion.py   [must pass to launch]
        │
1. TRAIN M′  (GPU, multi-hour): 2-stage drop-aware fine-tune from 200k, reuse exact recipe
   + objectness feature-dropout q~U(0,0.8)                                             [heavy GPU]
        │
        ▼   ── GATE A: eval M′ at q=0 ──
            PASS  = mIoU≥0.837, veh-IoU≥0.934, obj-recall≥0.775, ped-loc≤1.38m, heatmap peaks alive
            FAIL  → STOP + flag "M′ regressed → retune". DO NOT build the rest on a bad model.
        │ (PASS)
        ▼
2. On M′ (GPU-sequenced — single GPU, so chained not literally co-GPU):
     2a. SWEEPS: quant {8,6,4}×{none,zlib,zstd} payload (needs CARLA) + accuracy (offline)
                 + ROI-fraction {0,.1,.3,.5,.7} accuracy (offline)
     2b. AE TRAIN on M′ (ROI-drop q~U(0,0.8) in loop), {128,64,32}          ── GATE B: recon+distill loss plateaued
        │
        ▼
3. AE EVAL on M′ (offline): each AE {128,64,32} accuracy + payload
        │
        ▼
4. AGGREGATE → COMPLETE_KNOB_MATRIX.md : rows = action profiles (quant×ROI×AE),
   cols = accuracy (mIoU/recall/loc) · latency (front/back) · reliability (delivery) · payload (KB)  + Pareto
        │
        ▼
5. RL: matrix = offline action-cost table → controller harness B (baselines + LinUCB) → policy + Month-2 report
```

## Autonomy & decision-making (how I handle it while it runs)
- **Orchestrated by a chained script** (`run_pipeline_m_prime.sh`, launched `setsid` so it survives the
  session), logging every step to `rl_agent/PIPELINE_LOG.md`.
- **Decision gates are enforced in the script:** GATE A halts the pipeline if M′ regressed (never build the
  sweeps/AE/RL on a bad model — accuracy is the whole point). GATE B waits for AE convergence.
- **I review at each gate on re-engagement** (M′ metrics, sweep results, AE eval) and decide proceed / retune.
- **Decisions that serve the goal:** keep plain-200k results as the pre-robustness baseline; if a stage
  fails, log it and continue with what's still possible rather than hard-stopping the whole run.

## Resilience (lessons from the weekend + CARLA flakiness)
- **Offline stages need NO CARLA** (M′ train, accuracy, ROI, AE train, AE eval, aggregate all run on the
  saved dataset) → they survive CARLA dying. **Only the payload sweep needs CARLA**; if it's down, the
  pipeline **skips payload (flags it) and proceeds** with the offline relationships, filling payload later.
- **Checkpoint-based / resumable:** M′ and AE checkpoints are saved to disk; `PIPELINE_LOG.md` records the
  last completed stage. If the machine reboots mid-run, on re-invocation I read the log and **resume from the
  last completed stage** rather than restart.
- **GPU:** heavy jobs are **sequenced** (M′ alone → sweeps → AE) to avoid OOM; "parallel" here means
  unattended chaining, not literal co-GPU (single GPU).

## Final deliverable
`COMPLETE_KNOB_MATRIX.md` + Pareto = the full {quant,ROI,AE} × {accuracy,latency,reliability,payload}
relationship on M′ → the offline action-cost model the RL policy is derived from, and the input to the
controller harness (B). Then: RL state/action/reward finalized (schema already drafted) + Month-2 report.
