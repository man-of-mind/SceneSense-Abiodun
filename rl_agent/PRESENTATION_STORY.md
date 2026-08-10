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
- **SEND** the full detail (most accurate, most network),
- **COMPRESS** then send (less accurate, less network),
- **SKIP** this frame (free, but the map goes stale).
Send too much → the link congests and everything arrives late. Send too little → the map's object positions
drift out of date. **There is a real trade-off, and the right choice changes with the situation.**

## Slide 3 — Why a *smart* (adaptive) decision-maker
A fixed rule fails: "always send" congests; "always compress" loses accuracy. The right choice depends on
**how good the link is right now, how fast the objects are moving, and how stale the map already is.** We want a
decision-maker that **adapts to the situation** — that is what a learned controller (an RL "agent") gives us.
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
2. **Real 5G table** — from the OAI sweep: send X bytes at signal quality Y → delivered? how late?
3. **Accuracy table** — how much accuracy survives each compression level.
4. **Staleness model** — how location error grows with delay × object speed.
*(We inspected the driving traces to confirm they are physically real — correct object sizes, plausible
positions, the validated position convention. They are sound.)*

## Slide 6 — Safety first: the "shield"
Before the agent's preference matters, a **shield** blocks any decision that would let an object's position get
more wrong than the safety budget ε. Safety is **structural** (a hard gate), not just a reward term you hope wins.

## Slide 7 — The thought process: we tested 3 safety dials (don't hide this — it's the science)
We turned three plausible safety dials to learn which actually carries weight:

| Dial | What it controls | What we found | Lesson |
|---|---|---|---|
| **Safety margin** | Extra buffer on the worst-case estimate, to hedge uncertainty | Did nothing in the simulator | The simulator's estimate is deterministic — no uncertainty to hedge. This buffer earns its keep only against *live* uncertainty → turn on at live validation. |
| **Network caution** | How much to under-trust the bandwidth estimate before allowing a send | Changed which sends were *allowed*, rarely which was *chosen* | The choice is driven by the accuracy-vs-staleness trade, not the bandwidth cushion → a conservative default is fine. |
| **Estimator quality** | How stale/noisy the car's view of the network is | A *perfect* estimator fixed **0%** of the over-caution | Our hypothesis was wrong (cleanly). The over-caution is **structural**, not an estimation error. |

**Takeaway slide line:** *we tried three plausible levers; each taught us something; none was the lever — which
pointed us at the real limiter.*

## Slide 8 — What we actually learned (the results)
- ✅ **The shield is sound where we measured it** (objects within 25 m): it does not wave through unsafe
  decisions. Beyond 25 m the perception model is extrapolated and the guarantee breaks — so **25 m is the honest
  operating region.**
- ✅ **Achievability frontier (a real result):** even with *perfect* information, the safety budget is only
  reachable ~half the time at ε = 2 m — because sometimes the physics (speed + delay) simply won't allow it.
  **This is exactly why an adaptive, graceful-degradation controller is needed** — you must choose the least-bad
  action when the ideal is impossible.
- ⚠️ **The real limiter is the DATA:** our traces contain **no pedestrians** (only vehicles), and the car rarely
  needs to send in them — so we cannot yet prove strong safety numbers, especially for the safety-critical class.

## Slide 9 — Where we are, and the plan  ← WE ARE HERE
1. ✅ Design (state/actions/reward) + shield + fast environment + validation that the pieces are sound.
2. ▶ **Collect proper CARLA data** — with **pedestrians** and denser/faster traffic — so the environment covers
   the safety-critical cases it is currently blind to. *(This is the immediate next step.)*
3. ⏭ Build the comparison: two simple baselines vs the adaptive agent, scored on safety + efficiency.
4. ⏭ **Validate live** over real CARLA + OAI 5G.
*(Internship extended +3 months → time to do steps 2–4 properly, no shortcuts.)*

## One-line summary for the talk
> We designed a network-aware decision-maker, built a fast environment from real 5G + driving measurements,
> proved the safety mechanism is sound within its validity range, found that hitting the target isn't always
> physically possible (which is *why* adaptation matters), and identified richer data (with pedestrians) as the
> next thing we need to build the environment cleanly.
