# Phase-2 forward-compatibility — keep the single-UE agent extensible

**Decision (advisor, 2026-08-06):** phase 1 = build the single-UE agent; **revisit the UE-agent design when the
map-sharing / cooperative-fusion layer is built (phase 2).** This note records the hooks to *preserve* now so
that revisit is **additive (new inputs + retrain), not a rebuild.**

## Hooks to preserve in phase 1 (do NOT let these drift during implementation)
1. **AoI is PER-OBJECT (codex, 2026-08-06)** — `AoI_map,j = now − capture_ts(newest VALID map contribution for
   object j, from ANY source)`. **Not** this UE's own last transmit, and **not** a single global map-update age
   — a peer update for object A must not make object B look fresh. In phase 1 (single UE, per-frame send) it
   reduces to this UE's per-object update ages; in phase 2 a peer's fresh contribution to object *j* lowers
   *this UE's* `AoI_j`, so object *j* no longer drives the **frame-level** send decision. The current `SKIP`
   action remains whole-frame: it is safe only when the aggregate criterion over all relevant objects admits
   it. Matches v4 §4/§11 per-object semantics — keep AoI per-object; a map-level scalar is only a derived
   single-UE convenience, not the stored design.
2. **Modular urgency/state interface** — leave a clean slot to add a **per-object peer/map-feedback signal**
   later ("is my view valuable for object j?", "is a peer already covering it?"). The reward, shield, mode
   set, and safety structure should not need to change to accept it.
3. **Preserve repeatable per-object PROVENANCE now (codex)** — use one object record with a repeated
   contribution collection, not singular source/timestamp fields:
   ```text
   object_state:
     track_id, speed, speed_sigma, range_m, AoI_map_j
     contributions[]:
       source_ue_id, capture_timestamp, publish_timestamp, confidence
       profile_id, task_quality_snapshot
   ```
   `AoI_map_j` and the current post-action map utility are derived from the newest **valid** entry in
   `contributions[]`. The profile/quality snapshot prevents a dropped or skipped action from receiving credit
   for perception quality that never reached the map. Phase 1 writes a one-element
   collection; phase 2 can write several contributions without changing the data model. Keep the raw
   collection even when the phase-1 policy consumes only fixed-size summaries.

Corollary: **do not bake single-UE assumptions into the reward** (e.g. "this UE must cover every object").
The per-object-AoI framing already avoids that.

## Parked phase-2 questions (revisit at map-sharing — design NONE of these now)
- **Redundancy-aware deferral** — a fresher/more-accurate peer contribution prevents object *j* from forcing a
  send. Whole-frame `SKIP` is selected only if the aggregate safety criterion over all relevant objects
  admits it. Object-selective transmission would be a future action-space extension, not current behavior.
- **Per-object contribution value** — the fusion side decides whose view is best for which object; feeds the
  UE's urgency as a new input.
- **Multi-agent training / contention** — the tragedy-of-commons + non-stationarity when every UE runs the
  policy and competes for shared airtime. The real research meat of the multi-UE thread. Options to consider
  then: population/self-play training, a social/fairness term, or rely on the C1 mask + per-UE airtime cost.
- **Global-map vs per-car reward** — should the reward credit the *map's* quality (a peer's fresh contribution
  is good even if this UE did nothing) rather than this UE's own contribution.
- **Coverage / cooperative-fusion `U_task` term** — the deferred phase-2 map-completeness factor (v4 §5a).

## When phase 2 starts, the revisit is
add per-object peer/map-feedback signals to the state → possibly switch the reward to global-map quality → add
the multi-agent training regime → re-validate. The v4 **interfaces carry forward** (the safety structure +
C1 inequality survive); some **implementations may change** — `U_task` may become global-map utility, and a
**coordinated / per-UE capacity provider** may be needed under contention. Interfaces ≠ every implementation.
