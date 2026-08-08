# Phase-2 forward-compatibility — keep the single-UE agent extensible

**Decision (advisor, 2026-08-06):** phase 1 = build the single-UE agent; **revisit the UE-agent design when the
map-sharing / cooperative-fusion layer is built (phase 2).** This note records the hooks to *preserve* now so
that revisit is **additive (new inputs + retrain), not a rebuild.**

## Two hooks to preserve in phase 1 (do NOT let these drift during implementation)
1. **AoI = the SHARED-MAP's freshness** — `now − capture_ts of the newest successfully published map update`,
   from **any** source — NOT this UE's own last transmit. In phase 1 (single UE) they coincide; in phase 2 a
   peer's fresh contribution lowers this UE's observed map-AoI, so **"a peer keeps the object fresh → I skip"
   emerges with no redesign.** (v4 §4 already defines it this way — keep it.)
2. **Modular urgency/state interface** — leave a clean slot to add a **per-object peer/map-feedback signal**
   later (e.g. "is my view valuable for this object?", "is a peer already covering it?"). The reward, shield,
   mode set, and safety structure should not need to change to accept it.

Corollary: **do not bake single-UE assumptions into the reward** (e.g. "this UE must cover every object").
The map-level-AoI framing already avoids that.

## Parked phase-2 questions (revisit at map-sharing — design NONE of these now)
- **Redundancy-aware deferral** — skip when a peer provides a fresher/more-accurate contribution (nearly free
  given map-level AoI; just needs the map feedback signal).
- **Per-object contribution value** — the fusion side decides whose view is best for which object; feeds the
  UE's urgency as a new input.
- **Multi-agent training / contention** — the tragedy-of-commons + non-stationarity when every UE runs the
  policy and competes for shared airtime. The real research meat of the multi-UE thread. Options to consider
  then: population/self-play training, a social/fairness term, or rely on the C1 mask + per-UE airtime cost.
- **Global-map vs per-car reward** — should the reward credit the *map's* quality (a peer's fresh contribution
  is good even if this UE did nothing) rather than this UE's own contribution.
- **Coverage / cooperative-fusion `U_task` term** — the deferred phase-2 map-completeness factor (v4 §5a).

## When phase 2 starts, the revisit is
add peer/map-feedback signals to the state → possibly switch the reward to global-map quality → add the
multi-agent training regime → re-validate. The v4 masks/shield/mode/safety-band carry over.
