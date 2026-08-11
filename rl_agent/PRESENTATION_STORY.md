# Presentation story + north star — network-aware split-inference controller

Plain-language narrative for slides AND the shared "why are we doing this" reference for Abiodun + codex +
local Claude. Read top-to-bottom = the talk track. "We are here" marker is in §9.

---

## Slide 1 — The problem
A self-driving car sees the world with camera + radar. It wants to share what it sees with an edge server over
**5G**, so a **shared map** stays accurate for everyone. But the 5G uplink is **limited** — the car cannot send
everything, all the time.

## Slide 2 — The decision, many times per second
Every frame the car must choose one of:
- **SEND** the highest-fidelity supported split representation (most accurate available action, most network),
- **COMPRESS** then send (less accurate, less network),
- **SKIP** this frame (free, but the map goes stale).
Send too much → the link congests and everything arrives late. Send too little → the map's object positions
drift out of date. **There is a real trade-off, and the right choice changes with the situation.**

## Slide 3 — Why a *smart* (adaptive) decision-maker
A fixed rule can fail: "always send" can congest; "always compress" can lose accuracy. The right choice depends on
**how good the link is right now, how fast the objects are moving, and how stale the map already is.** We want a
decision-maker that **adapts to the situation**. RL is one candidate, to be compared against rules, a contextual
bandit, and model-predictive control rather than assumed to be the answer.
The research question in one line:
> *Does a network-aware adaptive policy keep the shared map safe while using the 5G link efficiently — and beat
> simple fixed strategies?*

## Slide 4 — Our design (this is what we agreed before the last meeting)
We defined the decision precisely (the **state diagram**):
- **State** = what the agent looks at before deciding (link quality, object speeds, map staleness, …).
- **Actions** = send / compress / skip (+ how much to compress, at what frame rate).
- **Reward** = the score: keep object locations within a safety budget **ε**, while spending little airtime.
This design is stable. Everything since has been about **testing it properly before committing.**

## Slide 5 — How we test without burning months: a fast "environment" from REAL data
We cannot train on live CARLA + live 5G (far too slow). So we built a fast **environment** (a.k.a. surrogate) —
a stand-in world — out of **measurements we already collected**:
1. **Real driving traces** — CARLA drives logging every object's true position each frame + what the detector saw.
2. **5G table** — measured OAI anchors plus explicitly labelled payload/FPS projections: send X bytes at a
   channel rung → delivered? how late?
3. **Accuracy table** — how much accuracy survives each compression level.
4. **Staleness model** — how location error grows with delay × object speed.
*(We inspected the driving traces to confirm they are physically real — correct object sizes, plausible
positions, the validated position convention. They are sound.)*

## Slide 6 — Safety first: the "shield"
Before the controller's preference matters, an **observation-based deterministic surrogate shield** blocks actions
whose estimated p95 localization bound exceeds ε. If none meets ε, it enters a flagged graceful-degradation band
and chooses the least-bad admissible action. This is a structural model gate, not a live safety guarantee.

## Slide 7 — The thought process: we tested 3 safety dials (don't hide this — it's the science)
We turned three plausible safety dials to learn which actually carries weight:

| Dial | What it controls | What we found | Lesson |
|---|---|---|---|
| **Safety margin** | Extra buffer on the worst-case estimate, to hedge uncertainty | Did nothing to selected actions at the admitted fixed point | Candidate uncertainty is not globally zero, but it is effectively inert on selected/raw-safe actions after the C1=0.70 gate. Residual/conformal calibration belongs in live validation. |
| **Network caution** | How much to under-trust the bandwidth estimate before allowing a send | Changed which sends were *allowed*, rarely which was *chosen* | The choice is driven by the accuracy-vs-staleness trade, not the bandwidth cushion → a conservative default is fine. |
| **Estimator quality** | How stale/noisy the car's view of the network is | A *perfect* estimator recovered **0.00 percentage points** of full-GT false rejection, although raw-safe sets moved | At this fixed point, estimator quality did not explain the headline gap; this is not a universal estimator result. |

**Takeaway slide line:** *we tried three plausible levers; each taught us something; none was the lever — which
pointed us at the real limiter.*

## Slide 8 — What we actually learned (the results)
- ✅ **Within the measured 25 m operating region, we observed no matched-object false admissions in the
  core90 fixed-point pilot** (0/15 admitted sends). The denominator is thin, and pooled advisor cells did include
  false admissions, so this is evidence to expand—not a proof or guarantee. Beyond 25 m is extrapolative.
- ✅ **Achievability frontier (a real result):** even with *perfect* information, the safety budget is only
  reachable about 61% of frames at ε = 2 m, core90, and 25 m—about 39% are infeasible—because sometimes the
  physics (speed + delay) simply will not allow it.
  **This is exactly why an adaptive, graceful-degradation controller is needed** — you must choose the least-bad
  action when the ideal is impossible.
- ⚠️ **The next limiter is the DATA:** the richer CARLA candidate fixed the missing-pedestrian-truth problem and
  has substantial controller-independent freshness pressure, but pedestrian detection is weak and sustained
  ≥10 m/s vehicle motion is split-thin. We therefore cannot yet make strong phase-1 pedestrian-freshness claims.

## Slide 9 — Where we are, and the plan  ← WE ARE HERE
1. ✅ Design (state/actions/reward) + shield + fast environment + validation that the pieces are sound.
2. ✅ The first richer CARLA candidate was collected end-to-end and added pedestrian truth. Its original
   `FAIL_QUARANTINED` report is preserved, but the shield-trajectory send-needed gate was later found to be the
   wrong test for phase-1 corpus motion.
3. 🔄 The corrected table-driven freshness re-score shows abundant slow cases, 47.59% GT-seeded skip-only
   pressure, and a real fast tail—but sustained ≥10 m/s motion is only in 2/1/1 train/validation/test runs, while
   pedestrian replay observation coverage is only 20.75%. Corpus disposition is **HUMAN_REVIEW_REQUIRED**.
4. ⏭ Abiodun + Claude decide salvage versus a small missing-regime supplement; then build the controller
   comparison and finally validate live over real CARLA + OAI 5G. Do not repeat the full 24-run collection.

## One-line summary for the talk
> We designed a network-aware decision-maker, built a fast environment from real 5G + driving measurements,
> found that the target is not always physically achievable, collected the missing pedestrian truth, and used a
> controller-independent freshness re-score to isolate the remaining data questions—split-thin fast motion and
> weak pedestrian detection—before controller training.
