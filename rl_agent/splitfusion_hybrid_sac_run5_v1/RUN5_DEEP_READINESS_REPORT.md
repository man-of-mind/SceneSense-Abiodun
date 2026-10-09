# Run-5 pre-deep-training readiness (2026-09-29)

**READINESS = PASS · DEEP_TRAINING_AUTHORIZATION = NOT_GRANTED**

This was a readiness and fix pass only. No training was run, no smoke was
retrained, no deep run was started, and CARLA, OAI, Docker and CUDA were not used.
All changes are inside `rl_agent/splitfusion_hybrid_sac_run5_v1/`; no Run-4 file
was touched. Nothing was pushed.

**Evidence.** `RUN5_DEEP_READINESS_AUDIT.json` holds passes 1–2, the profile
schedule and the disk projection; it was produced by a fresh process with 0
optimizer steps. `launch_rehearsal/REHEARSAL_seed_{17,29,43}.json` holds pass 3.

## 1. Contract and dataflow (15/15 checks pass)

The audit restored the committed update-500 smoke bundle in a fresh process with
`update_once` and `Adam.step` patched to fail.

- **Fresh native models.**
  - The update-0 actor, critics and targets are bit-equal to a fresh
    `build_run5_models` from the Run-5 seed plan, and both optimizers are empty.
  - The actor is 128×22 and the critics 128×35.
  - The first 21 columns differ from the Run-4 initialization, and the SNR column
    is non-zero in every bundle. No weights were padded or transferred.
- **Features 0–20.**
  - The order equals `run4_contract.POLICY_FEATURE_ORDER`.
  - Camera SI, radar P40, MCS and backlog equal the Run-4 definitions of the
    measured values (0 mismatches over 2,288 states).
  - All 11 monitored features vary.
- **Previous action and outcome.** The encoding equals the prior record's
  resolved outcome on every transition (0 mismatches). Genesis has
  prev_present = 0, and both SUCCESS and TIMEOUT predecessors occur.
- **Feature 21.**
  - Every value equals `(ACKed active leased target − 5.5)/19` for the command of
    that decision; the ACK and heartbeat precede state commit.
  - It is never equal to a later generated sample.
  - The channel's actor surface is only `{snr_db, mcs, tick}`; no profile, hidden
    state or trace field exists.
  - No zero-fill path exists: a modeled guard failure raises instead.
- **Reward.** All 2,288 rewards equal `Q_perc − 0.25·latency/170` or −1. SNR is
  absent from the reward, and the Run-4 reward-schema digest is unchanged.
- **Design unchanged.** The sealed config and design are byte-equal to the code.
  The profiles, seeds 17/29/43 and all Hybrid-SAC hyper-parameters equal Run 4's.

## 2. Checkpoint and recovery (11/11 checks pass)

- **Bundle contents.** Each of the 4 bundles in each smoke run holds exactly:
  - `event.json`: ledger, collector checkpoint, boundary, preflight, cadence;
  - `training_state.pt`: online and target critics, both Adam states, all 5
    generators;
  - `actor_state_dict.pt`;
  - `channel_state.json`: tick, hidden state, segment position, remaining profile
    block, all 3 RNG states;
  - `manifest.json` and `COMMITTED`.

  The decision count is 288 + 4u, and the schedule position follows from the
  update count plus the seeded warm-up.
- **250→500 evidence.** It remains valid: all bundle files are still byte-equal
  across the two runs. The training path has 0 changed lines since the evidence
  was produced (`run5_training/bundle/collector/channel/snr_v2/models`). It was
  not re-run.
- **Fixed (five operational gaps, amendment A2).**
  1. Intermediate evaluation-checkpoint actors could not be cold-loaded (the
     verifier refused non-final bundles). Completion now cold-loads the final
     actor and **every evaluation checkpoint ≤ target**
     (500/1500/2500/5000/7500/10000) in one fresh `weights_only` process and
     binds the results into `SEED_COMPLETE.json`.
  2. `CAMPAIGN_COMPLETE` had no entry point. `--finalize-campaign` now re-verifies
     all 23 bundles per seed plus all actors in a fresh process before writing it.
  3. The deep disk preflight reserved space for the invoking seed only. It now
     reserves for every incomplete seed, so parallel launches are safe.
  4. There was no durable-path guard. Deep campaign paths under `/tmp`,
     `/var/tmp`, `/dev/shm` or `/run`, paths with hidden or staging-like
     components, and paths on tmpfs, ramfs, overlay or squashfs are refused.
     Staging directories are `.staging-*`; final bundles use only the registered
     `checkpoint_/emergency_/final_actor_NNNNNN` names.
  5. `disk_preflight` crashed on a nested, not-yet-created campaign path. The
     rehearsal found this; the probe now walks to the nearest existing ancestor.

  These changes are recorded in the preregistration as amendment A2, with the
  training path and design unchanged. The preregistration was resealed to
  `2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf`.

## 3. Launch rehearsal

- **Deep evaluator: not required before launch.** The preregistration defines the
  deep-validation diagnostics as "diagnostics only". Actor selection is fixed a
  priori (the update-10,000 actor, with no validation-based selection), so
  nothing at launch depends on it. Everything needed afterwards is retained:
  - evaluation-checkpoint bundles (actors, critics, channel) that are never
    pruned;
  - fresh-process cold-load proof in `SEED_COMPLETE.json`;
  - registered validation seeds 9017/9029/9043 and their hashed profile
    schedules;
  - the sealed kernel and design.

  The future evaluator must still bind the held-out scene partition and the Run-4
  actor it compares against.
- **Profile schedule.** It was generated without training for 40,288 decisions per
  stream: 269 segments, with each profile 67–68 times and every complete 4-block
  containing each profile once. It is action-independent (a dedicated RNG stream
  and exactly 2 ticks per decision). The six streams are distinct, and the set
  SHA-256 is `68ad0277…08e9`.
- **Disk and inodes, measured from the smoke bundles.**
  - A bundle is 426 KB + 650 B per decision; the ledgers add 318 B per decision
    and 598 B per update.
  - Three seeds need **1.08 GB (2.16 GB with a 2× margin) and 561 inodes**.
  - The conservative launch gate requires 8.11 GB.
  - Free now: 19.3 GB and 121M inodes, on ext4 `/`.
- **Clean-process zero-update preflight.** Three separate processes, one per seed,
  all passed. Each ran the full deep launch preflight and the 288-decision
  no-gradient warm-up:
  - 0 optimizer steps, empty optimizer states, update 0;
  - no campaign or checkpoint directory created;
  - cold host confirmed;
  - about 4 s each.
- **Tests.** 88/88 non-training Run-5 tests pass, including the 7 new guard tests.
  The two training test classes were not re-run because the training path is
  unchanged; they passed at `602136b`/`d9e9eaf`.

## Proposed launch (not executed; needs authorization)

Authorization would be a file
`rl_agent/splitfusion_hybrid_sac_run5_v1/DEEP_TRAINING_AUTHORIZATION.json`
containing `{"preregistration_sha256": "2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf", ...}`.
It has not been created.

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun_run5_wt
CAMP=$PWD/rl_agent/splitfusion_hybrid_sac_run5_v1/campaign_runs/run5_three_seed_10000_v1
EVID=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
mkdir -p $CAMP/logs
for S in 17 29 43; do
  CUDA_VISIBLE_DEVICES="" nohup python3 -m rl_agent.splitfusion_hybrid_sac_run5_v1.run5_campaign \
    --mode deep --seed $S --campaign-dir $CAMP --evidence-root $EVID \
    > $CAMP/logs/seed_$S.log 2>&1 &
done; wait
python3 -m rl_agent.splitfusion_hybrid_sac_run5_v1.run5_campaign --finalize-campaign --campaign-dir $CAMP
# after any interruption: the same per-seed command plus --resume
```

| Item | Value |
|---|---|
| Output path | `$CAMP` (worktree, ext4 `/`) |
| Checkpoint path | `$CAMP/seed_{17,29,43}/checkpoints/checkpoint_NNNNNN/` + `LATEST` |
| Final actors | `$CAMP/seed_S/final_actor_010000/` |
| Completion | `seed_S/SEED_COMPLETE.json`, then `CAMPAIGN_COMPLETE.json` |
| Target | 10,000 updates per seed (40,288 decisions) |
| Cadence | 0, 100, 250, then every 500 to 10,000 (23 bundles per seed); evaluation checkpoints 500/1500/2500/5000/7500/10000 |
| Disk | Gate requires 8.11 GB; calibrated need 1.08 GB; 19.3 GB free |
| Duration | About 15–20 min per seed (smoke: 48.4 s for 500 updates incl. setup and completion); **about 20–30 min wall** for 3 parallel seeds (12 of 24 cores); a resume from update 10,000 re-executes about 40k actions in about 6–7 min |

READINESS = PASS
DEEP_TRAINING_AUTHORIZATION = NOT_GRANTED
