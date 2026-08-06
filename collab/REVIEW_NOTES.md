# Local-side review notes (Abiodun × local Claude)

Collaboration channel: the new box (L10319) commits work + pushes; local pulls, reviews, and writes feedback
here; new box pulls this each session. Newest entry on top. Keep code/results edits on the new box; local
touches only this file to avoid merge conflicts.

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
