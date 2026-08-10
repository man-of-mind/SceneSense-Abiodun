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
   the environment's measured `capacity(SNR)` surface and congestion knee. The policy does not observe this
   true surface directly; it receives a lagged/noisy achievable-capacity estimate.
2. **Accuracy ↔ knob ↔ payload** — `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md` (+ `density_knob/`). Transport-
   invariant (lossless codec), so it holds byte-for-byte over OAI. Gives per-frame accuracy for each action.
3. **Staleness model** — `staleness/STALENESS_RESULTS.md` (+ `uplink_only_latency_budget/`). The master
   inequality and the capture→map latency decomposition. Turns latency+delivery+speed → localization error.

End-to-end reward signal = compose( knob accuracy [2] ⊕ AoI transition from latency/delivery [1,3] ⊕ object
speed ). Canonical AoI is per-object shared-map age:
`AoI_map,j = now − capture_timestamp(newest valid contribution for object j, any source)`. In phase 1,
delivery normally resets every included object's age to capture→map latency, while skip/drop continues each
age. Keep repeatable contribution provenance per `PHASE2_FORWARD_COMPAT.md`; a scalar is only a derived
single-UE summary. No fusion-model re-run required.

## Locked design (do not re-derive) — `AGENT_CONSTRAINTS.md §9`
- **§9.1 STATE:** lagged/noisy channel telemetry + achievable-capacity estimate/confidence; object speed (+σ);
  current scene-emptiness/urgency; **per-object shared-map AoI**; previous action+outcome; and the locally known
  **scheduler phase + observable in-flight summary**. Estimate capacity from either
  full-resource `TBS_per_grant × attainable_grant_rate` or MCS spectral efficiency × configured available UL
  resources/time, corroborated by backlogged BSR/RLC drain and outcomes. Raw scheduled throughput and actual
  light-load allocations are demand-censored lower bounds, not capacity.
- **§9.2 ACTION** (cheap→costly): send/skip · quant u8→u4 (free) · AE bottleneck (main accuracy↔bytes dial) ·
  FPS · ROI/spatial-crop (accuracy-risky, LAST RESORT).
- **§9.3 REWARD:** one AoI-composed localization term + segmentation/recall utility − MCS-scaled PRB-time
  cost. Explicit delivery/drop terms are light diagnostics only, so a delivery outcome is not counted again
  at full strength. C1 is action-masked; C2 is a soft target.
- State/action/reward MDP diagram: **`rl_agent/state_diagram.md`** (Mermaid — render at mermaid.live / VS Code / GitHub).

### Constraints (safety) — enforce in the mask/reward
- **C1 (hard vs observation):** action-mask any send with
  `payload × fps > pessimistic(lagged/noisy achievable-capacity estimate)`; skip remains admissible. Hidden
  true-capacity misses caused by lag are logged and fed back to the estimator, not oracle-prevented.
- **C2 (soft):** for each object `j`,
  `loc_error_j = sqrt(base_loc(knob)^2 + (speed_j × AoI_map,j)^2)`, with configurable ε (default 2.0 m) as the
  target; aggregate objects as specified in `REWARD_FORMULATION.md`. AoI already includes pipeline and
  inter-update age; do not add separate latency, `1/FPS`, or staleness penalties. Use the generic 1.1 m floor
  only for operating-envelope reporting.
- **C3:** prefer the 90 KB `ae32/u4/ROI0` segmentation-safe floor. Sub-90 KB ROI actions are last-resort and
  pay their measured mIoU loss plus a configurable escalation penalty.
- **C4:** score only objects in the measured M′ validity region (`in_camera_frustum` and within 40 m); headline
  localization remains ≤25 m pending advisor confirmation. Off-FOV actors are outside this single-view Track A
  metric, not shield false admissions.

Reward v4 is authoritative: localization is enforced structurally by the tail-risk shield plus the small
mandatory `−w_E·E_expected/ε` margin; do **not** retain the old `−0.50·loc_error/ε` utility term. Post-action map
quality uses `0.50·mIoU/mIoU_ref + 0.25·ped_recall/ped_ref + 0.25·obj_recall/obj_ref` in the Track A pilot.

**Phase-1 shield basis (2026-08-10, Step-A result — see REWARD_FORMULATION §5c):** in the *surrogate* the tail
shield is realized as the **hard C1 mask + the deterministic p95 localization tail**, with the UCB margin
**inert** (`ucb_k = 0`, `c1_pessimism_factor = 0.70`). The `B = E_hat_risk + k·sigma_hat` UCB stays in code as
the design. At the 0.70 C1 floor it is numerical zero for every C1-admitted/raw-safe/selected action; some
C1-rejected candidates can still have nonzero ensemble spread, so `sigma_hat` is not globally zero. The margin
therefore has no leverage on phase-1 safe sets or selections and activates only with a validated
residual/conformal model at **live validation**. Do not present `ucb_k` or the 0.70 floor as statistically
calibrated phase-1 knobs: 0.70 is a conservative engineering convention matching the modeled -30% capacity
floor. The completed 4×3 estimator grid **falsified** lag/noise as the driver of the ~42% full-GT false-reject
rate at this fixed point: the ideal `(lag=0, noise=0)` cell recovered 0.00 percentage points and all 12 cells
spanned only 0.10 points. Lag/noise changed raw-safe sets but almost never the selected action. Attribute the
remaining gap among observation mismatch/coverage, speed uncertainty, worst-object aggregation, and map-state
trajectory before changing the shield or reward; do not label it irreducible.
Delivery installs the selected profile quality; drop/SKIP retains prior valid map quality; a new unobserved
object has zero map quality. Define references from best-achievable measurements and run the pre-registered
one-at-a-time weight sensitivity before the 12-condition sweep.

Resource cost is
`airtime_cost ∝ payload_bits × fps × tx_attempt_factor / spectral_efficiency(MCS)`, with
`tx_attempt_factor = 1 + retransmission_ratio` (or measured mean transmissions per original block). Prefer
measured PRB-seconds when present. Use the environment's realized MCS/resource outcome for reward accounting;
it is not exposed as oracle current state before the action. Thus identical bytes cost more under low MCS.

## Build sequence
1. **Surrogate env** from the three tables (interpolate `capacity(SNR)`, staleness, accuracy).
2. **Bandit / lookup baseline** off `payload_budget = pessimistic_estimated_capacity/fps` — the number to beat
   (near-static feedforward; no oracle access to current true capacity).
3. **Reward + constraint check** — sanity vs reward-hacking (empty-scene skip good, real-object skip penalized;
   seg vs ROI).
4. **Constrained RL** in the surrogate: Lagrangian-PPO or small DQN/Rainbow (discrete actions); model-based/
   offline is natural since we have the transition+reward tables. Then **validate live over OAI**.
5. **Generalization eval:** replay channel traces (step + realistic SNR patterns) — proves state-conditioned
   policy, not a schedule.

### Frozen Track A temporal contract
The surrogate uses a fixed 20 Hz event clock, a target-FPS rate accumulator, in-flight publish events, and
newer-capture-wins ordering. Changing the selected schedule or selecting SKIP resets fractional send credit.
The policy observes scheduler phase and locally observable pending-send summaries, never hidden delivery truth.
Exact projection, reward, replay-split, preferred-core, and pilot defaults are frozen in
`rl_agent/policy/IMPLEMENTATION_CONTRACT.md` and `rl_agent/policy/configs/track_a_pilot.yaml`.

## Open items to carry forward (none block starting)
- **Shaped-burst @ fixed 10 fps** (Mode A, no CARLA) to pin the *absolute* knee — current knee is empirical at
  ~6 fps; `capacity(SNR)` has ±~30 % uncertainty (delivered-ceiling estimate). Treat capacity as a band.
- **Add mid SNR rungs** (25–45 dB) — current ladder clusters low (50/19.5/15.6/8.2).
- **If you run anything live on the new box:** pin the back-half fusion to a separate GPU (or use the
  shaped-burst sender) so CARLA render contention can't silently throttle offered load (the bug that cost us a
  grid). And per `CHANNEL_SWEEP_PLAN.md` guardrail 0a, **never** characterize the knee on the closed-loop
  runner — uplink-only only.

## Pointers
`AGENT_CONSTRAINTS.md` (§9 design, §1–5 staleness bounds) · `PHASE2_FORWARD_COMPAT.md` (per-object map-AoI +
repeatable provenance contract) · `channel_condition_sweep/CHANNEL_SWEEP_RESULTS.md`
+ `plots/` · `PERMODEL_KNOB_MATRIX_ZSTD.md` · `staleness/STALENESS_RESULTS.md` · `CLAUDE.md` (project state).
