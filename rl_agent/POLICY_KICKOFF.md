# Policy formulation — START HERE (kickoff, 2026-08-04)

Goal: a **safety-constrained RL controller** that, each frame, observes lagged network state + current scene
urgency and picks **compression / FPS / send** actions to keep spatial-map staleness within the localization
error budget — without congesting the uplink.

## You can start immediately — NO OAI / CARLA needed
Policy formulation is **table-driven**. The environment is already measured; you build a fast **surrogate env**
from three tables and train in it. OAI/CARLA are only needed later, for the *live validation* step.

### The surrogate environment = these three inputs
1. **Transport surface** — `channel_condition_sweep/combined_surface.csv` (+ `CHANNEL_SWEEP_RESULTS.md`, plots):
   payload × SNR → delivery, capture→map latency, BSR backlog, app-offered vs scheduled-UL. Gives
   `capacity(SNR)` and the congestion knee.
2. **Accuracy ↔ knob ↔ payload** — `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md` (+ `density_knob/`). Transport-
   invariant (lossless codec), so it holds byte-for-byte over OAI. Gives per-frame accuracy for each action.
3. **Staleness model** — `staleness/STALENESS_RESULTS.md` (+ `uplink_only_latency_budget/`). The master
   inequality and the capture→map latency decomposition. Turns latency+delivery+speed → localization error.

End-to-end reward signal = compose( knob accuracy [2] ⊕ staleness from latency/delivery [1,3] ⊕ object speed ).
No fusion-model re-run required.

## Locked design (do not re-derive) — `AGENT_CONSTRAINTS.md §9`
- **§9.1 STATE:** channel state (SNR/CQI, MCS, BLER/HARQ, sched-UL rate, UE BSR/RLC — *observed with lag*);
  object speed (+σ); scene-emptiness/urgency gate (current frame, pre-transmit); **+ previous action+outcome**
  (last payload/FPS, last latency/delivery — needed because channel is lagged).
- **§9.2 ACTION** (cheap→costly): send/skip · quant u8→u4 (free) · AE bottleneck (main accuracy↔bytes dial) ·
  FPS · ROI/spatial-crop (accuracy-risky, LAST RESORT).
- **§9.3 REWARD:** + fresh-delivered map update + localization accuracy/low staleness error − network-resource
  cost (PRB-time, not raw bytes) − dropped/stale frames. Hard-constrained (see below).
- State/action/reward MDP diagram: **`rl_agent/state_diagram.md`** (Mermaid — render at mermaid.live / VS Code / GitHub).

### Constraints (safety) — enforce in the reward/policy
- **C1** payload ≤ `budget(SNR) = capacity(SNR)/fps` — fits the channel (violate → congestion collapse:
  BSR→48 MiB, latency→seconds, delivery cliff).
- **C2** `v × total_staleness ≤ sqrt(ε² − floor²)`, floor ≈ 1.1 m; total_staleness = sensor prep + front +
  uplink + edge/map.
- **C3** keep the seg-safe floor (≥ 90 KB, `ae32/u4/ROI0`) unless truly forced.
- **C4** object range ≤ 40 m (perception-valid region — M′ trained/eval'd with `max_gt_distance_m=40`).

## Build sequence
1. **Surrogate env** from the three tables (interpolate `capacity(SNR)`, staleness, accuracy).
2. **Bandit / lookup baseline** off `payload_budget(SNR)` — the number to beat (near-static feedforward).
3. **Reward + constraint check** — sanity vs reward-hacking (empty-scene skip good, real-object skip penalized;
   seg vs ROI).
4. **Constrained RL** in the surrogate: Lagrangian-PPO or small DQN/Rainbow (discrete actions); model-based/
   offline is natural since we have the transition+reward tables. Then **validate live over OAI**.
5. **Generalization eval:** replay channel traces (step + realistic SNR patterns) — proves state-conditioned
   policy, not a schedule.

## Open items to carry forward (none block starting)
- **Shaped-burst @ fixed 10 fps** (Mode A, no CARLA) to pin the *absolute* knee — current knee is empirical at
  ~6 fps; `capacity(SNR)` has ±~30 % uncertainty (delivered-ceiling estimate). Treat capacity as a band.
- **Add mid SNR rungs** (25–45 dB) — current ladder clusters low (50/19.5/15.6/8.2).
- **If you run anything live on the new box:** pin the back-half fusion to a separate GPU (or use the
  shaped-burst sender) so CARLA render contention can't silently throttle offered load (the bug that cost us a
  grid). And per `CHANNEL_SWEEP_PLAN.md` guardrail 0a, **never** characterize the knee on the closed-loop
  runner — uplink-only only.

## Pointers
`AGENT_CONSTRAINTS.md` (§9 design, §1–5 staleness bounds) · `channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md`
+ `plots/` · `PERMODEL_KNOB_MATRIX_ZSTD.md` · `staleness/STALENESS_RESULTS.md` · `CLAUDE.md` (project state).
