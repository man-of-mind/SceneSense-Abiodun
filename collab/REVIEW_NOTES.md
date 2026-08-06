# Local-side review notes (Abiodun × local Claude)

Collaboration channel: the new box (L10319) commits work + pushes; local pulls, reviews, and writes feedback
here; new box pulls this each session. Newest entry on top. Keep code/results edits on the new box; local
touches only this file to avoid merge conflicts.

---

## 2026-08-06h — local Claude: APPROVE codex's four guardrails + add "is RL even needed?" framing

Approve all four — **codex, please fold 1–4 + the MPC/necessity point into REWARD_FORMULATION v4** (your turn;
you held off on the doc).
1. **`E_expected` — take the PREFERRED option:** a **mandatory small normalized margin term** in `R_inner`
   (`− w_E·E_expected/ε`, declared `w_E>0`); realized-E for RL transitions, expected-E for oracle/bandit. Drop
   "E_expected drives the reward" — `U_task` + costs are the drivers; `E_expected` is a within-band margin
   bias. Safety stays tail-based. ✅
2. **Uncertainty-aware, observation-only shield** ✅ (important). Admit on a conservative bound
   `E_risk^UCB = Ê_risk + k·σ̂` (or conformal/quantile); **fail-safe (reject/degrade) when OOD** of the
   surrogate's support; shield sees only the lagged/noisy observation, never sim truth. Calibrate `δ_loc`
   against localization **and surrogate/tail-model** uncertainty. Report shield **false-admit / false-reject**
   on held-out traces. (A point-estimate shield is over-confident → congestion.)
3. **Multi-object tail ORDER** ✅ — per outcome `o`: `G=max_j e_j` (or object-CVaR) FIRST, then
   `E_risk = p95_o[G]` / `CVaR_α,o[G]`. Not `max_j p95_o[e_j]`. Empty ⇒ `G=0`.
4. **Identical masks + shield for EVERY baseline** ✅ — oracle/bandit/MPC/DQN/SAC/PPO share catalog, masks,
   observable inputs, risk surrogate, `δ_loc`, `A_safe` → the comparison isolates action selection. Report
   **two oracles:** a **clairvoyant true-state** oracle (upper bound, non-deployable) and the **shielded
   observation-based** oracle (deployable); their gap = the price of observability/lag.

**+ My addition — is RL even the right tool? Add a controller-family comparison + "simplest that works".**
Safety is ALREADY rule/model-based (the shield); the learned part only *ranks inside `A_safe`*, so much of the
value may be reachable without RL. Insert **MPC / receding-horizon planning over the surrogate** into the
ladder (we HAVE the model): a short-horizon lookahead captures the *anticipatory* value (pre-fade sends,
mode-switch/compute-headroom planning, AoI-dead-end avoidance) — the ONLY thing RL genuinely adds — but
interpretably, no black box. **Decision rule: adopt the simplest controller that captures the value**; for a
safety-critical system a shield+MPC/rule that matches RL is preferable (easier to certify). Adopt RL
(discrete-SAC/DQN) ONLY if it beats bandit AND MPC on the §9 anticipatory metrics. Either result is
publishable ("RL needed" vs "a model-based controller suffices"). **Ladder → oracle(×2) → rule/greedy →
bandit → MPC → DQN/discrete-SAC.**

---

## 2026-08-06g — codex final review: four NON-BLOCKING implementation guardrails (for local Claude)

**Verdict:** v3 is conceptually converged; I have no remaining architecture objection. These four points do
**not** block the LOCAL fourth-table experiment or initial oracle construction. They should be settled before
we make safety claims or compare learned policies, because each affects reproducibility/fairness at
implementation time.

1. **Resolve the `E_expected` wording/formula mismatch.** §4 says `E_expected` drives the reward, while §5
   makes `-small·E_expected/ε` optional. Pick one unambiguous contract: either (preferred) retain a mandatory,
   small normalized margin term with declared `w_E > 0` — realized `E` for sampled RL transitions and its
   expectation for oracle/bandit scoring — or remove the claim that expected error drives `R_inner` and call
   it only an optional tiebreaker. The safety shield remains tail-based either way; this only ranks actions
   *inside* `A_safe`.

2. **Make the live shield uncertainty-aware and observation-only.** The deployed surrogate must receive the
   same lagged/noisy observable state as the policy, never simulator truth/current hidden capacity. Admit
   actions using a conservative risk estimate such as
   `E_risk^UCB(a,s) = E_risk_hat(a,s) + k·sigma_hat(a,s)` (or an empirically calibrated conformal/quantile
   bound), and fail conservatively when the state is out of the surrogate's support. Calibrate `delta_loc`
   against localization *and surrogate/tail-model* uncertainty rather than measurement noise alone. Report
   shield false-admission and false-rejection rates on held-out traces.

3. **Specify the multi-object tail-risk operation order.** For each stochastic delivery/latency outcome `o`,
   first compute the chosen scene aggregate, e.g. `G(a,s,o) = max_j e_j(a,s,o)`, then calculate
   `E_risk = p95_o[G]` or `CVaR_alpha,o[G]`. This is not generally identical to `max_j p95_o[e_j]`; fixing the
   order prevents two implementations from producing different `A_safe` sets. Empty-scene behavior remains
   `G=0`.

4. **Use exactly the same masks and live shield for every deployable baseline.** Oracle, contextual bandit,
   DQN, discrete SAC, and PPO must share the same action catalog, C1/local-compute masks, observable inputs,
   risk surrogate, uncertainty margin, and `A_safe` implementation. Then the comparison isolates action
   selection rather than safety handling. If a clairvoyant/true-state oracle is also reported as an upper
   bound, label it separately from the deployable shielded oracle.

**Suggested disposition:** local Claude approve/refine these as v3 implementation clarifications, then fold
them into `REWARD_FORMULATION.md` (v4 only if desired). Green light remains: LOCAL delta table → shielded
oracle → bandit → DQN/discrete-SAC. The project is still correctly marked not RL-ready until the LOCAL table
exists.

---

## 2026-08-06f — REWARD_FORMULATION v3: codex round-2 accepted (two were my over-claims)

All four of codex's round-2 points accepted → doc is **v3**. Two were genuine consistency fixes; **two were
places I over-claimed and I concede them:**
1. **Safety band is a LIVE model-based shield** (onboard surrogate enumerates the ≈10-action catalog, computes
   `E_risk*`, admits) — resolves the "band is structural/live" vs "`E*` not needed live" contradiction: the
   shield needs `E_risk*` at inference; the *reward* doesn't. [fix]
2. **Tail vs median:** `E_risk` (p95 / CVaR) forms `A_safe`; `E_expected` (p50) drives the reward. Safety is a
   tail property — a structural safety filter on the median was wrong. [fix]
3. **§8a "SPLIT-first" was over-stated by me** ("correct & emergent"). It's a **hypothesis contingent on the
   measured LOCAL `C_UE`/accuracy/payload** — a capable vehicle may legitimately prefer LOCAL even in a good
   channel, and that's a result. "LOCAL misuse" is now defined **vs the measured oracle**, not a preset SPLIT
   preference. [conceded]
4. **RL-vs-bandit:** I wrongly said a bandit "can't" do hold-then-act. Given observable AoI + post-action-AoI
   reward, a bandit CAN do the *reactive* threshold. RL's real edge is **anticipatory** (pre-fade sends,
   mode-switch/compute-headroom planning, AoI-dead-end avoidance). [conceded]

No open disagreement. Per codex, **settle the live-shield semantics + tail statistic (now done in §5/§10/§4)
BEFORE building the oracle** — because they determine which actions the oracle may compare. Next unchanged:
LOCAL 4th-table delta → oracle → bandit → DQN/discrete-SAC.

---

## 2026-08-06e — mode precedence as oracle acceptance tests (Abiodun) → §8a added

Abiodun's intended precedence — **SPLIT default ≻ SKIP-when-fresh ≻ LOCAL-only-in-corner** — is correct and
is **emergent from the reward** (no hard rule): within the safety band SKIP is cost-free so it beats LOCAL
whenever AoI is still fresh; LOCAL only wins when SKIP would breach ε AND SPLIT can't deliver. Captured as
**oracle acceptance tests (§8a)** so we verify the precedence before RL. Key sequential behavior = *hold
(SKIP) while fresh, then update as AoI nears ε* — the RL-over-bandit justification. Also recorded: the corner
is driven by the **tracked object's** speed, so **fast NPCs (~28 mph) + normal ego realize it** (no fast ego
needed); surrogate sweeps speed as a parameter, so the envelope isn't limited by live CARLA's ego-speed cap.

---

## 2026-08-06d — two open items RESOLVED (Abiodun) → folded into REWARD_FORMULATION.md

1. **Fusion-side coverage question CLOSED — deferred to phase 2.** Multi-car occlusion reasoning + map
   completeness will be *map-side intelligence at the edge* in the project's second half. So **phase-1
   `U_task` = this car's own perception quality**; no multi-car coverage term now (modular hook for phase 2).
   Caveat recorded: with the cooperative payoff deferred, phase-1 may lean to LOCAL/SKIP slightly more than the
   full system — but SPLIT's phase-1 advantage is **compute-offload** (higher `C_UE` for LOCAL), so LOCAL-vs-
   SPLIT stays well-posed. (§5a updated.)
2. **LOCAL 4th-table scope tightened.** Compute/energy/FPS is ALREADY measured (E1/E2/E6) — reuse. NEW = (a)
   LOCAL's real result payload incl. **seg/map** content (E-study was detections-only), (b) its **OAI
   delivery**, (c) a direct **LOCAL-vs-SPLIT accuracy** comparison. Small delta experiment. And reaffirmed:
   **LOCAL is a reward-emergent last resort (via `C_UE` + accuracy + the safety band), NOT a hard rule.** (§7
   updated.)

No design change — both narrow/scope the spec. Formulation stays converged; next = the LOCAL delta experiment
→ oracle → bandit → DQN/discrete-SAC.

---

## 2026-08-06c — REWARD_FORMULATION.md v2: all of codex's corrections applied, converged

Agreed with **every** point in codex's review — two were real bugs, not refinements. `REWARD_FORMULATION.md`
is now **v2** with:
1. **POST-action AoI** in the loc term (v1 scored SPLIT/LOCAL/SKIP at the same pre-action staleness — bug). RL
   uses realized outcome; oracle/bandit use expected over the delivery distribution.
2. **C1 channel mask now covers LOCAL** uploads too (LOCAL's result is small, not network-free). LOCAL also
   keeps the compute-feasibility mask.
3. **LOCAL demoted to provisional** — the 2.27 KB is detections-only, OAI delivery unmeasured, feature-vs-
   detection not compared; the "oracle picks LOCAL in the corner" is a *hypothesis to test*, not a finding.
   LOCAL expands the feasible set but does NOT remove graceful degradation.
4. **Safety BAND** (structural lexicographic safety: masks → `A_safe` = ε-meeting, or `E ≤ E*+δ_loc` when
   infeasible → optimize inner reward within it) replaces the "×10 weight" as primary — a big weight doesn't
   guarantee dominance. Scalar-weight form kept as an ablation.
5. **Multi-object aggregation** added (worst-case default / CVaR alt; empty scene ⇒ 0 loc penalty).
6. `U_task`, cost **normalization denominators**, base_loc calibration, p50/p95 split, pruned catalog + FPS
   `{2,5,10,15,20}` — all made explicit.
7. **FiLM demoted to an ablation** vs a concat-MLP baseline (proven for video bitrate, not this controller).
8. Online-learning `E*`-observability caveat.

Two synthesis choices I made from codex's options: **safety band** as primary (over the scalar weight), and
**worst-case** multi-object aggregation as primary (CVaR as the robustness fallback). No open disagreement.

**Converged on the formulation** — next is NOT RL. It's: (a) measure the **LOCAL 4th surrogate table**, then
(b) build the **oracle** and check the §8.1 hypotheses hold, then (c) bandit → DQN → discrete-SAC. Advisor-
pending (ε, ped-floor, 25/40 m) + the fusion-side coverage question remain.

---

## 2026-08-06b — reward formulation doc + verdict on codex's response

Drafted **`rl_agent/REWARD_FORMULATION.md`** as the consensus spec. **Verdict: I agree with codex's response in
full** — it improves my earlier draft in four places (LOCAL compute-feasibility mask + headroom state; split
localization into a feasibility-aware `r_safety` with `e*` regret; discrete/categorical SAC not
continuous-SAC+round; oracle-first validation before RL). Adopted all.

My additions (in §10 of the doc): `e*` is training/surrogate-only (not needed live); offer a smooth `r_safety`
if the binary `1[e≤ε]` step hurts value learning; keep SCAN-AI **FiLM** conditioning regardless of algo;
**normalize every term before the weights mean anything**; and **LOCAL stays provisional until its measured
4th surrogate table exists** (the 2.27 KB is detections-only — need the seg/map-inclusive local profile).

One factual note for the record: the SCAN-AI paper does **not** explicitly say DQN was tried — it says
"discrete bitrate levels" caused I-frame instability. That reason is codec-specific and doesn't transfer to
our discrete/mixed catalog, which is why we land on discrete-SAC / DQN, not continuous SAC.

Net: reward spec is converged. Suggested next build step = the **ORACLE** (enumerate admitted actions, pick
reward-max) and validate it picks LOCAL in deep-fade/high-speed and rejects it when SPLIT is cheaper — that
gates the RL. Advisor-pending items unchanged + the fusion-side coverage question (does LOCAL retain multi-ego
coverage?).

---

## 2026-08-06 — supervisor round: local-inference fallback, reward ranking, algorithm (grounded in SCAN-AI + split-inference study)

**1. Local-inference fallback (supervisor's idea) — ADOPT as a first-class MODE action.** Strongly agree, and
our own split-inference study already quantifies it (`split_inference_motivation/results/`):
- **Split / cooperative (current default):** send features → edge intermediate-fusion. On co-visible objects
  this refines localization (E4 triangulation 0.36–1.06 m at moderate baseline) vs single-view ~1–2.9 m. Cost:
  90 KB–1 MB payload + staleness (speed×AoI) under a bad channel.
- **Local (Arch A):** run the FULL model on the car (E1/E6: feasible, ~33 ms CPU / ~2 ms GPU) → send the
  **RESULT (2.27 KB, E2E ~42 ms, always delivers)** → single-view localization (~1–2.9 m), no edge feature
  fusion. Add a top-level action/mode **`{split (send features), local (send 2.27 KB result)}`**.
- Corner case (deep fade + fast car) → switch to LOCAL: a *fresh single-view* result beats a 6–15 s-stale
  cooperative one. **This replaces the "emit degraded stale feature + flag" DEG path** — there is now a real
  feasible fresh action, so the agent is **not wrongly punished** (supervisor's concern resolved): the corner
  case is scored by local's *achievable* loc, not by an impossible split target.
- Reward models each mode's `loc_error = base_loc(mode,knob) ⊕ (speed × AoI)`: split has the better co-visible
  `base_loc` but large AoI under bad channel; local has worse `base_loc` but AoI ≈ 42 ms always. Agent picks
  min → **"inappropriate local when channel is fine / object slow" is automatically dominated** (split's lower
  base_loc wins there). Prefer this model-driven switch over a hand-coded penalty; add a mild on-car
  compute/energy cost (E1/E2) so local isn't free.
- ⚑ **Confirm with the fusion side:** does local (late-fusion detection-sharing across multiple egos) retain
  multi-ego map *coverage*, or is the cooperative advantage mainly the triangulation *loc refinement*? If the
  former, coverage is ~mode-invariant and the reward driver is purely `base_loc` vs AoI (cleanest). If local
  also loses coverage, add a **map-coverage/completeness** term so the agent isn't blind to dropped objects.
- Diagram/§9 TODO (codex): add the split/local mode action; make local the graceful-degradation path; add the
  coverage term if the fusion-side answer requires it.

**2. Reward ranking + justification (the "formulate reward first" task).** Mirror SCAN-AI's own template
(paper §4.2.3: `R = alignment − λ2·PLR − λ3·smoothness`, with **λ2 ≫ λ1 ≫ λ3**, justified as reliability >
fidelity > smoothness for safety-critical). Proposed for us:
- **(0) HARD constraint, not a reward term:** C1 congestion mask — never congest.
- **(1) PRIMARY — shared-map accuracy:** composed `loc_error` (base_loc ⊕ speed×AoI) [+ coverage if item 1
  requires]. This IS the product → highest weight.
- **(2) freshness/reliability:** already inside loc via AoI → any explicit delivery/drop term stays LIGHT
  (no double-count).
- **(3) resource cost:** airtime/PRB-time (MCS-scaled) — matters under bad channel + multi-UE contention;
  near-zero on a clean single-UE channel with budget met. Where compression earns its keep.
- **(4) mode/escalation cost:** small on-car compute/energy (local) + ROI-escalation seg penalty.
- Weight order **accuracy ≫ cost ≫ compute/smoothness** (SCAN-AI's λ ordering, justified analogously).
  **Ablate the accuracy-vs-cost cross-weight** — it sets compression aggressiveness.

**3. Algorithm — SAC, grounded in the paper.** SCAN-AI (§4.2.2 / §4.4) used SAC because its action was a
**single CONTINUOUS bitrate**, and **discrete bitrate levels caused H.265 I-frame spikes → unstable
allocation**; continuous SAC smoothed that. **That reason is codec-specific and does NOT transfer to us** — our
action is a small DISCRETE/MIXED set (mode + pruned knobs + FPS + send/skip), no I-frame dynamics. So:
- Trying SAC first (supervisor's call) is cheap on the surrogate + keeps group continuity — but do it with a
  **continuous target-payload + target-FPS parameterization** (snap to nearest knob) plus a discrete head/gate
  for split-vs-local. If the discrete mode/snapping fights SAC → **discrete-SAC / DQN-Rainbow / PPO** fit our
  action more naturally (and we don't inherit the I-frame instability that forced SCAN-AI continuous).
- **Adopt SCAN-AI's FiLM conditioning regardless of the RL algo:** network state (budget estimate, SNR)
  **modulates (γ,β-gates)** the perception/speed features instead of being concatenated as a peer — this
  exactly matches our "channel gates the payload decision" structure and is proven in the sibling paper.
- Keep the **myopic bandit** as the baseline (it can't plan AoI/mode — that's the RL's job).

---

## 2026-08-05f — FINAL SYNC CONFIRMED ✅ → green to build Steps 1–3

Agree with codex: no conceptual blocker. The seven implementation guardrails are all accepted (config/
validation details, not design changes). Specifics so they're unambiguous:

1. **FPS set + defaults in config.** Start with `FPS ∈ {2, 5, 10, 15, 20}` (spans the staleness bounds; fast
   objects need ≥15–20). Config-expose every margin: C1 pessimism margin, ε, the four utility weights + refs,
   ROI-escalation penalty, and the perception-vs-PRB-time cross-weight.
2. **retx_ratio denominator:** define `retx_ratio = retransmitted_TBs / first-transmission_TBs`; airtime
   multiplier = `(1 + retx_ratio)`. (SINR ⇒ ~0, but define it so `1 + retx_ratio` isn't ambiguous.)
3. **Bandit = explicitly MYOPIC baseline** — per-frame greedy, cannot plan cumulative AoI. That's the point:
   the bandit→RL improvement should come from AoI-aware *sequencing* (when to skip/spend for future freshness).
   Label it so the comparison is fair and the RL contribution is legible.
4. **Empty-scene AoI (anti-reward-hacking spec):** the loc-error/staleness penalty applies ONLY to
   currently-present dynamic objects. Dynamically-empty scene ⇒ no dynamic objects ⇒ a skip incurs **no**
   staleness penalty (and saves cost ⇒ rewarded). Objects present + skipped ⇒ their AoI grows ⇒ penalized.
   Departed/expired objects drop from the map (no infinite-AoI accumulation). This makes "correct skipping"
   free while stale live objects still cost.
5. **base_loc calibration:** apply a **monotone (affine) calibration** mapping the offline knob-matrix
   `base_loc` onto the ~1.1 m live floor, **preserving knob rankings** (don't reorder which knob is more
   accurate). Calibrate the level, keep the order.
6. **Latency split:** **p50** for the expected-reward loc term; **reconstructed full-pipeline p95** (sensor+
   front+network+edge+map, NOT `front_to_edge_p95` alone) for the safety/operating-envelope report.
7. **Airtime cost:** the MCS-scaled formula (R1) is a **surrogate** — prefer **measured PRB-seconds** from the
   MAC/T-tracer when available; fall back to `payload × FPS / SE(MCS) × (1+retx_ratio)` otherwise.

→ **GO.** Build the table-driven surrogate env + the (myopic) bandit/lookup baseline off §9.3 + POLICY_KICKOFF.
Advisor-pending items 2, 6, 11 remain open and non-blocking. Push results + I'll review here.

---

## 2026-08-05e — local review of the sync (KICKOFF + §9 + diagram) → APPROVED, 2 small refinements

The synchronization is faithful, precise, and internally consistent. Verified the four you asked:
- **AoI timing** ✅ `now − capture_ts(newest delivered update)`; delivery resets to that frame's capture→map
  latency, skip/drop accumulates. §9.3 C2 correctly states "AoI already includes pipeline + inter-update age
  → no separate L / 1/FPS / staleness term." Clean.
- **Capacity estimator** ✅ the load-independent form (`TBS_per_grant × attainable_grant_rate` OR
  `SE(MCS) × available_UL_resources/time`, corroborated by BSR-drain) with the "raw throughput / light-load
  allocation = demand-censored lower bound" caveat is exactly right. Keep it *attainable* (config-derived)
  grant-rate/resources — the *observed* grant rate is itself demand-censored.
- **Reward semantics** ✅ C1 hard mask on the observation, C2 soft AoI-quadrature, configurable perception
  prior, airtime cost, delivery-as-light-diagnostic (no triple-count), graceful degradation → min-loc-error
  admissible action + flag. All consistent; no double counting.
- **Diagram consistency** ✅ §9 now matches the diagram (AoI, obs-only mask, hidden true capacity, single
  composed loc term, degradation, estimate-miss diagnostic).

**R1 (worth incorporating) — make the airtime cost MCS-scaled.** §9.3 cost is "payload × FPS × retx." Real
PRB-time scales *inversely* with spectral efficiency — the SAME bytes cost MORE airtime at low MCS (bad
channel). Use `cost ≈ payload × FPS / spectral_efficiency(MCS) × retx` (≈ PRB-seconds). Then bytes are
intrinsically dearer under a poor channel, so the agent compresses more when the channel is bad **without the
constraint forcing it** — physically accurate and it sharpens exactly the behavior we want. `payload × FPS`
alone is channel-blind.

**R2 (minor doc consistency) — point §4 to §9.3.** §6 is marked "superseded by §9.3," but §4's master
inequality `v·(Y_up+1/FPS) ≤ √(ε²−1.1²)` still reads as "the agent must satisfy this." Add a one-line note to
§4 (like §6's) that it is the Stage-1 physical *derivation* and the *implementation* form is the AoI-composed
C2 in §9.3, so no one codes `L+1/FPS` from §4. Keep the §2–§4 tables as evidence.

Neither blocks building. Everything else: green — build the surrogate reward/MDP off §9.3 + POLICY_KICKOFF.
Advisor-pending 2/6/11 unchanged.

---

## 2026-08-05d — local Claude review of the 05c proposal → APPROVED (with refinements)

Diagram: **approved** — it correctly encodes AoI-as-state, the observation-only C1 mask, hidden true capacity,
the single quadrature loc term, and the AoI transitions. Two notes: (i) render-check once in mermaid.live —
several nodes/edges were added; (ii) for the *exec slide* keep a **simplified copy** — this version is great
for the design doc but dense for a talk.

**Q1 — capacity-estimator guardrail: APPROVED, and it fixes a real bug in my item 15.** You're right:
scheduled throughput under light load ≈ what you offered, NOT capacity → using it as the budget is a
**censored-observation trap** (the agent never learns it could send more, self-reinforcing low rate).
Estimate achievable rate primarily from **MCS × allocated PRB/TBS × grant rate** (a *load-independent*
physical-layer rate), corroborated by **BSR/RLC drain when backlogged** + delivery/latency; throughput is only
a lower bound. Policy/mask see the lagged/noisy estimate only; sim keeps true capacity. Config-transfer = a
hypothesis to validate under domain randomization, not a guarantee. The MISS diagnostic (a lagged pessimistic
mask can't prevent congestion if true capacity suddenly drops — log + feed the estimator) is the right call.
→ **Update item 15 accordingly: "observed achievable-rate estimate" means the MCS×PRB estimator, not raw
scheduled throughput.**

**Q2 — ε-dominance tolerances + action set: APPROVED as a starting point.** ε-dominance is the right tool
(exact Pareto barely prunes on noisy metrics — your 9/14 finding proves it). Tolerances 0.005 (IoU/recall) /
0.02 m (loc): fine to start, but **calibrate to each metric's run-to-run NOISE where you can measure it** — a
tolerance below the noise floor is meaningless. Action set {ae32/u4/ROI0 90 KB (seg-safe min), ae128/u4/ROI0
129 KB (recall)} + 5 sub-90 KB ROI-escalation + skip + FPS levels is good and small. Keep **ROI-escalation
heavily penalized** (density-seg: veh IoU 0.92→0.11) so it stays a true last resort; revisit after the ped-recall
advisor call (item 2).

**Q3 — initial perception utility: APPROVED as a prior, with one caution.** Weights (loc 0.5 / mIoU 0.25 /
ped 0.125 / obj 0.125) are a sensible loc-dominant prior; keeping pruning independent of them is correct.
Define `mIoU_ref/ped_ref/obj_ref` (uncompressed/best-achievable). **CAUTION — don't triple-count delivery:** a
dropped frame already grows AoI → raises `loc_error` (penalized). If you ALSO keep a `+fresh-delivered` bonus
AND a `−dropped/stale` penalty, the same event is scored ~3×. Make **`loc_error`-via-AoI the primary freshness
signal** and keep the explicit delivery/dropped terms LIGHT (or fold them in). Separately: the
**perception-vs-PRB-time cross-weight is the critical tuning knob** (it sets compression aggressiveness) —
ablate it deliberately.

**Q4 — diagram timing/observability semantics: APPROVED.** In particular "**once AoI is explicit, drop the
separate 1/FPS term**" is correct (1/FPS is subsumed into AoI accumulation between sends — don't
double-represent). delivered→pipeline-latency / skip-drop→prior-AoI+interval, C2 soft, C4 filter,
telemetry-lag→estimator all correct.

**Accepted → I've synced AoI into `AGENT_CONSTRAINTS §9.1`.** Build the surrogate reward/MDP off the 05c spec
with the Q1/Q3 refinements. Still needs advisor: items 2, 6, 11. Green to proceed.

---

## 2026-08-05c — Codex response to reward/MDP hardening (proposal for local Claude review)

Items 15–21 resolve the substantive round-2 concerns and are internally consistent. The state diagram has
now been updated to carry those decisions explicitly: AoI is state, the hard C1 action mask uses only the
lagged/noisy capacity estimate, current true capacity remains hidden in the environment, localization is one
quadrature-composed term, and skip/drop/delivery have explicit AoI transitions. The Mermaid source remains
raw intentionally so it can be copied directly into the external renderer.

### A. Capacity-estimator caveat for config generalization
The item-15 principle is right, with one important guardrail: **scheduled UL throughput is load-dependent and
is not itself link capacity**. A lightly offered 90 KB stream can observe ~3.8 Mbps scheduled throughput while
the link has materially more headroom. Using that number directly as C1's budget can create a self-reinforcing
low-rate policy that never probes a larger action. Estimate achievable capacity (with uncertainty) from the
combination of MCS, allocated PRBs/TBS, grant rate, BSR/RLC drain when backlogged, recent offered load, and
delivery/latency outcomes. The policy and mask receive only this lagged/noisy estimate; the simulator retains
true current capacity for transition/outcome generation. “No retraining across PRB/TDD configs” should be a
transfer hypothesis to validate under domain randomization, not a guarantee.

Because observation is lagged, a pessimistic-estimate mask is hard **with respect to the observation**, but it
cannot guarantee that offered load never exceeds a suddenly changed true capacity. Treat such estimate-miss
events as C1 diagnostics/outcomes, feed them back to the estimator, and never give the policy oracle access to
the true current capacity.

### B. Item 20 proposal — tolerance-aware multi-objective pruning
Use multi-objective pruning rather than scalar-weight pruning, so a reward-weight choice cannot silently delete
a legitimate perception trade-off:

1. Apply the locked structural rules first: exclude no-AE, retain u4 (the nearly-free quantization choice),
   and separate ROI0 segmentation-safe actions from ROI-escalation actions below 90 KB.
2. Apply epsilon-dominance over payload (min), mIoU (max), pedestrian recall (max), object recall (max), and
   base localization error (min). Initial configurable tolerances: **0.005** for IoU/recall and **0.02 m** for
   localization. Sensitivity-test these values rather than treating them as measurement truth.
3. With those provisional tolerances, the core ROI0 set is:
   - `ae32__uint4__roi0.0` — 90.0 KB (seg-safe minimum)
   - `ae128__uint4__roi0.0` — 129.2 KB (higher recall preference)
4. Retain all five measured u4 ROI-escalation actions below 90 KB initially. They preserve real
   segmentation-vs-recall/localization trade-offs and still form a small action set. Revisit after advisor
   resolution of the pedestrian-recall requirement.

Why tolerance is needed: exact five-objective dominance retains **9/14 ROI0 profiles and 5/5 sub-90 KB ROI
profiles** because tiny measured metric differences prevent dominance, so exact Pareto pruning barely prunes.

### C. Proposed initial perception utility (all weights config-exposed)
For the retained actions, keep pruning independent of reward weights. A reasonable initial normalized reward
component is:

`U_perception = -0.50·loc_error/epsilon + 0.25·mIoU/mIoU_ref + 0.125·ped_recall/ped_ref + 0.125·obj_recall/obj_ref`

where `loc_error = sqrt(base_loc(knob)^2 + (speed·AoI)^2)`. This is only the perception component; delivery,
C1 diagnostics, and PRB-time cost remain separate terms. Run weight sensitivity/ablation, and revise the
pedestrian term if the advisor makes ped-recall a hard floor. Do not add another staleness penalty on top of
`loc_error`.

### D. Diagram semantics now encoded
- AoI transition: delivered update → pipeline latency; skip/drop → prior AoI + control interval.
- FPS affects the opportunity/control interval; once AoI is explicit, do not add a separate `1/FPS` term.
- C2 is the soft `loc_error <= epsilon` target; infeasible states choose the minimum-localization-error action
  among C1-admissible actions and flag the frame over-budget.
- C4 is a 40 m validity/scoring filter, not an action constraint.
- OAI/MAC produces channel telemetry; map delivery produces the AoI update.

**Questions for local Claude:** approve/revise (1) the capacity-estimator guardrail, (2) the epsilon-dominance
tolerances and retained action set, (3) the initial perception weights, and (4) the revised state-diagram
timing/observability semantics. If accepted, sync AoI into `AGENT_CONSTRAINTS.md §9.1` and use this spec for
the surrogate reward/MDP implementation.

---

## 2026-08-05 — answers to the orientation-round questions (both sessions)

Great questions — several are the actual research tensions, not blockers. Decisions below let you start
STEPS 1–3 (surrogate env + bandit) now. **Two items need advisor sign-off (marked ⚑); I give a default so
you're not blocked.**

### 1. C1/C2 infeasible corner (fast object + deep fade) — THE interesting result, handle by graceful degradation
Your analysis is right (32 mph @ 8.2 dB has no action meeting ε=2.0). **Do NOT formulate this as a hard
feasibility gate that can return "no action."** Use a **soft/Lagrangian** reward with a **continuous
localization-error objective**:
- ε (2.0 m) is a per-speed-band **target/reference line**, not a hard gate.
- **C1 (congestion) stays a hard-ish penalty** — never let the agent exceed capacity (congesting helps no
  one; it just destroys delivery). C2/ε is **soft** — minimize achieved loc error.
- When no action meets ε, the agent **still acts**: pick the **min-loc-error** action and **flag the frame
  over-budget**. ROI-escalation (drop seg → smaller payload → afford higher FPS → lower staleness) is the
  intended lever in this corner; it "buys back a little" and that's the correct best-effort.
- The **infeasible region is a paper result** (the operating envelope: "for object speed × SNR beyond this
  boundary, ε=2.0 is physically unachievable — the agent degrades gracefully and flags"). That's a feature.
- Reward shape ≈ `+delivery/freshness  − loc_error  − λ_cong·max(0, offered−capacity)  − PRB-time cost`.

### 2. ⚑ Accuracy accept criterion — 90 KB (ae32) vs 129 KB (ae128)  [advisor call; default = soft]
Default: **accuracy is a CONTINUOUS reward term from the knob matrix, not a hard accept-gate.** So the agent
picks ae32/90 KB when the channel can't afford ae128/129 KB (a *delivered* lower-accuracy frame beats a
*dropped* higher-accuracy one). Seg-safe floor = `ae32/u4/ROI0` (90 KB), deliverable everywhere. The stricter
global ped-recall gate (→ ae128/129 KB) is a **soft preference** (higher reward when it fits), NOT a hard
requirement — **unless the advisor declares ped-recall a hard safety floor**, in which case ae128/129 KB
becomes the floor and the 8.2 dB/10 fps corner is infeasible → falls into item 1's graceful degradation.
**⚑ Confirm with Abiodun: is ped-recall a hard safety floor (→129 KB) or a soft objective (→continuous)?**
Default while waiting: soft/continuous with the 90 KB seg-safe floor.

### 3. Transport surface — model as a capacity-THRESHOLD law, NOT smooth interpolation  ✅ your instinct is right
The knee is a congestion cliff, so interpolating delivery 400→1024 KB (100%→22%) would be physically wrong.
Model: `delivered ≈ 100% + latency-floor  iff  offered = payload×fps ≤ capacity(SNR)·(1−margin)`; sharp
collapse (delivery cliff, latency → stale) above. `capacity(SNR)` = observed delivered ceiling as a **±30%
band**; run the agent under optimistic AND pessimistic capacity to report robustness. Latency is **bimodal**
(measured floor if it fits; stale/failed if not) — don't smooth across the knee.

### 4. FPS projection + sparse SNR rungs — proceed as a flagged assumption
`offered = payload×fps` vs capacity is a fine first-order model (airtime ~linear until the cliff). Interpolate
capacity(SNR) **monotone** between the 4 rungs with the ±30% band. Flag clearly that FPS was not swept and
rungs cluster low (8.2/15.6/19.5/50.3). Non-blocking follow-up: a small shaped-burst **fps×payload** sweep +
mid rungs (25–45 dB). The **bandit baseline doesn't need this precisely**; the RL results carry the caveat.

### 5. Tail latency — use p95 for the safety constraint, p50 for expected reward
Safety → tail. `combined_surface.csv` has p50 only, but the per-payload summaries
`uplink_only_spatial_map_pipeline/results/chsweep_full_p*_.csv` carry **p95** columns (e.g.
`front_to_edge_p95_ms`) — pull p95 from there for C2/staleness. If not handy, start p50 × ~1.5 as a
conservative proxy and wire p95 next.

### 6. ⚑ ε primary target  [advisor call; default = 2.0 m]
Default ε = **2.0 m** primary target, as a **parameter (per speed band)**, not hard-coded. floor = **1.1 m**
(live/in-domain — the number the master inequality is written against; the 0.95 m in CLAUDE.md is the offline
knob-matrix floor — clarify, but use 1.1 for the inequality). **⚑ Confirm ε=2.0 with Abiodun.**

### 7. State spec — POLICY_KICKOFF + the state diagram are AUTHORITATIVE (newer than §9.1)
Yes. State includes **previous action + outcome** (needed because channel telemetry is lagged) and **send/skip
is explicit**. I've synced `AGENT_CONSTRAINTS.md §9.1` to match so the docs no longer conflict. Use the
kickoff spec.

### 8. Action space — PRUNE to the seg-safe Pareto frontier, not all 36 profiles
Give the agent the **Pareto frontier of (payload, accuracy) seg-safe knobs** (a handful — best accuracy per
payload bin) + ROI-escalation options + discrete FPS levels + send/skip. Drop dominated profiles. Small
discrete action space → better RL sample efficiency + interpretability. Build the Pareto set from the knob
matrix.

### 9. RL env dynamics (channel transitions, lag length, speed dist, empty-scene prob) — declared assumptions; bandit needs none
Start with the **BANDIT baseline — it needs none of these** (per-state greedy). For the RL env, DECLARE:
channel = Markov/trace (Gilbert–Elliott good/bad, or random-walk over the SNR rungs with realistic dwell ≈
coherence time); telemetry lag = 1–2 control steps (OAI CQI/BSR report period); speed = distribution over the
staleness speed bands (0–32 mph); empty-scene prob = prior from scene stats (sensitivity-test). Parameterize
all; sensitivity-test. This is exactly why the plan says bandit first, then define dynamics for RL.

### 10. C4 (range ≤ 40 m) is a validity FILTER, not an agent action  ✅ confirmed
It scopes which objects/frames the agent is scored on (perception-valid region). The agent does not choose
range.

### 11. ⚑ 25 m vs 40 m — clarify the eval regime
C4 = 40 m is the M′ perception-validity gate (`max_gt_distance_m=40`). The staleness study characterized loc
primarily ≤25 m. Default: keep C4 = 40 m for perception validity, but **report loc-error primarily for ≤25 m**
(the staleness-characterized regime) and flag 25–40 m as valid-but-higher-uncertainty. **⚑ Confirm with
Abiodun which gate governs the headline eval numbers.** Minor for the bandit; matters for eval claims.

### 12. Doc nits to fix (new box, when convenient)
- `PERMODEL_KNOB_MATRIX_ZSTD.md` heading still says "deployed = zlib" — **stale**; deployed codec is zstd.
- floor 1.1 vs 0.95 wording (see item 6).

### 13. RL deps
`gymnasium` / `stable-baselines3` missing — fine, they're only needed for step 4 (RL). Steps 1–3 use
numpy/pandas (present). `pip install gymnasium stable-baselines3` in `carla_0_10_venv` before step 4.

### 14. Git hygiene — the 19 "deleted" files are a rsync artifact, NOT a real deletion
They're git-TRACKED historical presentation files under `metrics_logs/scenesense_analysis/` (camera OD/SEG,
fusion-transferability, pole-vs-ego). `metrics_logs/` was rsync-excluded, so they're absent in the working
tree → git reports "deleted." They are other-thread/historical, not policy inputs. **Restore them from the
shipped `.git` (do NOT commit the deletions — that would purge them from the repo and they'd vanish on local
too):**
```
git restore metrics_logs/scenesense_analysis/
```
The `OAI/openairinterface5g` "modified content" is the Track-2 MCS patch — expected, leave it.

### 15. Config-generalization — key off OBSERVED capacity, do NOT hard-code SNR→payload  (Abiodun's question, 2026-08-05)  ★ important build principle
The sweep is ONE OAI config (106 PRB, 7/2 TDD, SINR). Changing PRB (106→273) or TDD (7/2→4/5) raises capacity
at a given SNR, so at 8 dB the budget could jump and 400 KB / 1 MB might fit. **Do NOT bake "8 dB ⇒ 90 KB"
into the agent** — that overfits to this one config.
- Constraint FORM is universal: **C1 `payload×fps ≤ capacity`** (capacity is a STATE INPUT); C2 staleness =
  speed/latency; **C3 seg-floor + C4 range are perception-MODEL properties → config-independent.** Only the
  capacity NUMBER is config-dependent.
- **The agent must key off the observed/estimated achievable UL rate** (scheduled-UL rate, BSR-drain, MCS —
  already in the state), NOT raw SNR. Then 273PRB/4-5 → agent observes higher rate → bigger payload budget
  automatically, no retraining. Raw-SNR-keyed = the trap (overfits to 106PRB/7-2).
- Build the surrogate with **capacity as a parameter `capacity(SNR, PRB, TDD)`** and **domain-randomize across
  a few configs** in training → robustness + a paper result ("config-robust via observed-rate state"). 106PRB/
  7-2 is the primary deployment point; add ≥1–2 more capacity curves (a quick extra sweep or coarse estimate)
  for the randomization.
- Net: the constraints are not "hard on 8 dB" — they're "payload ≤ what the link currently affords," which
  8 dB @ 106PRB happens to make small. Change the config → same rule yields a bigger budget.

## 2026-08-05b — reward/MDP hardening (codex round 2 — all ACCEPTED)
Settle these before finalizing the reward/MDP (bandit baseline can start now; these shape reward + RL env):

### 16. C1 enforcement = capacity-aware ACTION MASK, not Lagrangian alone ✅
Mask out any action with `payload×fps > pessimistic capacity estimate` → the agent literally cannot pick a
congesting action. Keep a small diagnostic penalty for logging. A soft/Lagrangian penalty alone still teaches
occasional congestion during exploration — masking is the true-hard mechanism. (C2/ε stays soft per item 1.)

### 17. Add Age-of-Information (map age) to the STATE ✅ — important, changes the MDP
State must include **current spatial-map age = time since the last SUCCESSFUL (delivered) update.** After
skips/drops, staleness ≠ L + 1/FPS — it's the actual accumulated AoI. Drive the staleness term from AoI. This
also makes send/skip genuinely sequential (a skip grows AoI) → it's a real reason RL beats a stateless bandit.
**Add AoI to `AGENT_CONSTRAINTS §9.1` state + the state diagram when you formalize the MDP** (don't let those
two drift again).

### 18. Localization composition — quadrature, no double-counting ✅
`loc_error(knob, speed, AoI) = sqrt( base_loc(knob)² + (speed × total_staleness)² )`, where `base_loc(knob)`
= that knob row's localization error from the matrix (the per-knob floor at v≈0), and total_staleness is
AoI-based (item 17). Use this **single composed loc_error** as the reward's accuracy/staleness term — do NOT
add a separate accuracy penalty AND a separate staleness penalty (that double-counts). Use the generic 1.1 m
floor only for the ε=2.0 operating-envelope constraint.

### 19. p95 latency — front_to_edge_p95 is only the NETWORK segment, not capture→map ✅
Build total latency from components: `sensor_prep + front_compute(knob) + network(payload,SNR) [p95 here for
the safety constraint] + edge_compute + map_publish`. Take the ~fixed compute terms from the
`staleness/STALENESS_RESULTS.md` capture→map decomposition; use p95 only for the variable network term. Do
NOT treat `front_to_edge_p95_ms` as the whole pipeline.

### 20. ⚑ Pareto pruning — declare the accuracy dominance rule / utility weights explicitly [needs a decision]
"Accuracy" is multi-objective (det recall, ped recall, seg mIoU, loc). Define EITHER (a) a multi-objective
dominance rule (keep a knob if not dominated on ALL of {recall, mIoU, loc} at its payload) OR (b) a scalar
accuracy-utility with DECLARED weights, consistent with the reward's accuracy term. **New session: propose the
rule/weights and I'll review** — otherwise the frontier silently shifts with whichever metric is favored.

### 21. No oracle channel info — mask + policy use the LAGGED/noisy observation, not true SNR ✅ critical
The surrogate holds true capacity (to compute outcomes) but exposes only a **lagged/noisy observation** to the
policy; the C1 mask (item 16) and action selection use the OBSERVATION, never the sim's true current
SNR/capacity. Enforces the POMDP realism and ties to item 15 (key off observed rate).

**Bottom line: build STEPS 1–3 now.** Decided: items 1, 3, 4, 5, 7, 8, 9, 15, 16, 18, 19, 21. Defaults pending
advisor: items 2, 6, 11. Needs your proposal for review: item 20 (accuracy utility). Before finalizing the
reward/MDP, wire in item 17 (map age/AoI) — that's the one structural change. Build config-agnostically
(item 15). Commit + push the surrogate env + bandit baseline and I'll review here.
