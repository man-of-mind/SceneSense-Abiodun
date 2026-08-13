# Multi-UE RL feasibility — does contention create coordination/RL headroom? (2026-08-13)

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

## Result — headroom scales with collapse severity, and the measured channel is in the harsh regime
| collapse_frac (throughput retained when over-offered) | coord − greedy freshness (RL headroom ceiling) |
|---|---|
| 1.00 (graceful: serve K, rest retry) | ~0 pp (greedy fine) |
| 0.50 | +18 to +26 pp |
| 0.25 | +40 to +57 pp |
| 0.00 (total collapse) | +80 to +93 pp (greedy craters to ~0% fresh) |

**The measured OAI sweep is in the HARSH regime:** over-offered cells deliver only **5–30%**, BSR pins at
**47.7 MiB**, latency explodes to **6–15 s** (`combined_surface.csv`). So `collapse_frac ≈ 0.05–0.3` → the
coordination advantage is **large (~+40 pp or more)**.

## Mechanism
Decentralized greedy **death-spirals**: freshness-critical UEs override backoff → synchronized over-offer →
congestion collapse → nobody delivers → more UEs go stale/critical → worse over-offer. A coordinator that keeps
aggregate offered ≤ capacity avoids collapse entirely and holds 80–94% freshness. The C1 backoff cannot save
greedy because freshness-criticality forbids deferral at exactly the wrong moment.

## Verdict: **GO** — multi-UE contention genuinely motivates learned/coordinated control
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

## Next step
Build the proper multi-UE surrogate (N copies sharing the measured capacity surface + the measured collapse law)
and run the real ladder there: greedy-everywhere vs coordinated-oracle vs decentralized-learned (RL/MARL). Ground
`collapse_frac`/delivery from the over-capacity cells of `combined_surface.csv`.
