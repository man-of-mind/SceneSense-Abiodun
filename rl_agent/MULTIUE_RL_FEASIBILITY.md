# Multi-UE RL feasibility — does contention create coordination/RL headroom? (2026-08-13)

> **Final status — toy GO retracted, measured NO-GO (2026-08-14).** The early +40–93 pp claim below came from the
> audited collapse abstraction and strawman C1 override; it is preserved only as a failure-analysis trail. The
> later DG-A OAI measurement found no N=2 application-layer admission gap because the MAC scheduler already
> operated at the measured capacity ceiling, and the corrected large-N screen left 0/216 cells. DG-B/campaign/RL
> stopped. This result is independent of the separate Phase-1 replay causality defect.

**Question:** the single-UE controller ladder showed greedy ≈ MPC (RL NO-GO). Does the *multi-UE* setting
(every UE running the policy, competing for shared uplink) create headroom that justifies learned/coordinated
(RL / multi-agent-RL) control? Tested with a fast standalone contention model (NO CARLA, ~minutes) before
committing to a full multi-UE build.

**Model (minimal, honest, grounded — not the full surrogate):** N UEs share one cell capacity drawn from the
measured 4-rung Markov process (clear 37 / mild 28 / mid 20 / strong 10 Mbps, measured transition matrix, ±30%).
Each UE must refresh its shared-map entry before its object goes stale (AoI budget = sqrt(ε²−base²)/speed).
Two policies: (a) **decentralized greedy+AIMD-backoff**, where freshness-critical UEs override backoff (they
cannot defer a stale-critical send); (b) **clairvoyant coordinator** that admits the neediest due UEs up to
capacity, never over-offering. Over-subscription applies a `collapse_frac` = throughput retained past the knee.
Script: `scratchpad/multiue_feasibility.py`.

## Historical toy result — superseded by DG-A
| collapse_frac (throughput retained when over-offered) | coord − greedy freshness (RL headroom ceiling) |
|---|---|
| 1.00 (graceful: serve K, rest retry) | ~0 pp (greedy fine) |
| 0.50 | +18 to +26 pp |
| 0.25 | +40 to +57 pp |
| 0.00 (total collapse) | +80 to +93 pp (greedy craters to ~0% fresh) |

**The measured OAI sweep is in the HARSH regime:** over-offered cells deliver only **5–30%**, BSR pins at
**47.7 MiB**, latency explodes to **6–15 s** (`combined_surface.csv`). So `collapse_frac ≈ 0.05–0.3` → the
coordination advantage is **large (~+40 pp or more)**.

## Historical toy mechanism — invalid abstraction
Decentralized greedy **death-spirals**: freshness-critical UEs override backoff → synchronized over-offer →
congestion collapse → nobody delivers → more UEs go stale/critical → worse over-offer. A coordinator that keeps
aggregate offered ≤ capacity avoids collapse entirely and holds 80–94% freshness. The C1 backoff cannot save
greedy because freshness-criticality forbids deferral at exactly the wrong moment.

## Retracted toy verdict: ~~GO~~
The single-UE NO-GO does NOT generalize. Single-UE is myopic (greedy≈optimal); multi-UE in the measured
hard-collapse regime is a real coordination problem with large headroom. This is where the RL / multi-agent-RL
contribution lives.

## Honest caveats
- The coordinator is a **clairvoyant oracle** = the *ceiling*. Real decentralized RL/MARL captures a *fraction*
  of it — but even a fraction of +40 pp dwarfs the single-UE ~1%.
- Minimal abstraction (throughput-retained collapse, AIMD+critical-override greedy, freshness metric). Directional.
- **The actual research question:** how much of the coordination ceiling can a *decentralized learned* policy
  recover (anticipate contention, send earlier to avoid the critical-rush, trade own staleness to prevent
  collective collapse)? Learnable only if the training env contains contention (N copies sharing the channel).

## Superseded next step
Build the proper multi-UE surrogate (N copies sharing the measured capacity surface + the measured collapse law)
and run the real ladder there: greedy-everywhere vs coordinated-oracle vs decentralized-learned (RL/MARL). Ground
`collapse_frac`/delivery from the over-capacity cells of `combined_surface.csv`.

## ⚠️ 2026-08-13 codex review — this toy OVERSTATES the headroom; corrected plan below (ACCEPTED)
The toy (`multiue_feasibility_toy.py`) is directional only and its +40–93 pp is an **overstated upper bound.**
Two accepted corrections:
1. **Wrong collapse abstraction.** `collapse_frac` treats over-subscription as *lost cell throughput*. The
   measured OAI reality: throughput stays ~at the service ceiling while **queues back up (BSR pins) and latency
   explodes (6–15 s)** — i.e. delivery is *late*, not *lost*. The freshness hit comes via latency/staleness, not
   destroyed throughput. A **queue-service model** is the correct abstraction, and it yields a smaller, subtler
   coordination advantage than throughput-destruction did.
2. **Strawman greedy.** The toy lets freshness-critical UEs OVERRIDE backoff and over-offer — which violates the
   LOCKED design (C1 capacity admission is HARD; C2 staleness is soft). A correct C1-respecting decentralized
   greedy *cannot* over-offer beyond its observed budget, so it does not death-spiral the way the toy showed →
   less headroom.
Both push the same way: real coordination headroom is **modest and hinges on observation quality + lag +
fairness**, not a death-spiral. **Verdict: GO investigate multi-UE contention; HOLD multi-UE RL.**

**Corrected sequencing:** (a) run a small **CARLA-free 1/2/4-UE OAI shaped-traffic measurement**; (b) **fit a
queue-service model** from it (throughput-ceiling + queue-latency) — do NOT train against the extrapolated
`collapse_frac`; (c) ladder = **decentralized C1-greedy → decentralized token/AIMD → observable centralized
EDF/max-weight admission → clairvoyant oracle**, all before any learned controller; (d) only if learning survives
that ladder, **masked categorical PPO** is the natural first POMDP baseline (DQN only if the action space is
small/factorized; continuous SAC inappropriate). Full review: `collab/REVIEW_NOTES.md` (2026-08-13, codex).
