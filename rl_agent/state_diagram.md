---
config:
  layout: dagre
  theme: redux
---
flowchart LR
 subgraph S["STATE  s(t) — observation"]
    direction TB
        ch["Channel state (observed WITH LAG + NOISE):<br>SNR/CQI, MCS, BLER/HARQ, PRB/TBS/grant telemetry,<br>scheduled UL rate, UE BSR/RLC buffer,<br>estimated sustainable UL budget + confidence"]
        sp["Object speed (+ uncertainty)"]
        em["Front-side urgency proxy<br>current frame objectness + radar/range cue<br>+ tracker prior, not lagged"]
        prev["Previous action + outcome<br>last payload/FPS, last latency/delivery"]
        age["Age-of-Information (AoI)<br>now − capture timestamp of newest<br>successfully published map update"]
  end
 subgraph A["ACTION  a(t) — cheap to costly"]
    direction TB
        gate["send / skip"]
        quant["quant u8 → u4<br>(cheap / nearly free)"]
        ae["AE bottleneck size<br>(main accuracy vs bytes dial)"]
        fps["FPS"]
        roi["ROI / spatial crop<br>accuracy-risky, LAST RESORT"]
  end
 subgraph ENV["ENVIRONMENT — 5G uplink + edge (transition)"]
    direction TB
        off["offered load = payload × FPS"]
        truth["HIDDEN true current UL service capacity<br>depends on channel + PRB/TDD config<br>+ scheduler/link adaptation<br>(never exposed directly to policy/mask)"]
        oai["OAI uplink:<br>delivery, latency, BSR backlog, MCS/BLER"]
        mp["edge fuses to shared map<br>(freshness / staleness)"]
  end
    src1(["gNB / UE MAC + T-tracer<br>MCS, BLER, BSR/RLC, PRB/TBS,<br>grant rate + scheduled UL rate"]) --> est["Lagged/noisy UL-budget estimator<br>full-resource TBS/grant × attainable grant rate OR<br>MCS efficiency × configured UL resources/time;<br>allocated throughput is a light-load lower bound<br>corroborate with backlog drain + prior outcomes"]
    est --> ch
    src2(["Front perception / tracker<br>object speed + uncertainty"]) --> sp
    src3(["UE front-side urgency proxy<br>front features + radar/range cue<br>+ tracker/map prior"]) --> em
    src4(["Agent memory (t-1)<br>last action + outcome"]) --> prev
    src5(["Map feedback / ACK clock<br>latest published frame_id<br>+ capture timestamp"]) --> age
    S --> MASK{{"C1 ACTION MASK (observation only)<br>admit payload × FPS ≤ pessimistic<br>estimated sustainable UL budget<br>skip is always admissible"}}
    MASK --> POL{{"Policy π(a|s) — RL, safety-constrained<br>chooses payload / FPS / send<br>from the C1-admissible action set"}}
    POL --> A
    A --> C{{"CONSTRAINTS / OPERATING RULES<br>C1 mask is HARD vs observed pessimistic UL budget;<br>&nbsp;&nbsp;&nbsp;&nbsp;true-budget estimate misses are logged outcomes<br>C2 composed loc_error ≤ epsilon (default 2.0 m) — SOFT / best-effort<br>C3 prefer perception-safe ROI0 floor = 90 KB (ae32/u4/ROI0);<br>&nbsp;&nbsp;&nbsp;&nbsp;allow sub-90 KB ROI only when forced<br>C4 object range ≤ 40 m — validity/scoring FILTER, not an action"}}
    off --> oai
    truth --> oai
    oai --> mp
    C -- admitted action --> off
    ENV --> AOI["AoI TRANSITION<br>delivered → now − delivered frame's capture timestamp<br>= capture→map pipeline latency<br>skip/drop → previous AoI + control interval"]
    A --> AOI
    AOI --> LOC["ONE composed localization term<br>loc_error = sqrt(base_loc(knob)² + (speed × AoI)²)<br>generic 1.1 m floor only for operating-envelope reporting"]
    sp --> LOC
    A --> LOC
    ENV --> S2["NEXT STATE  s(t+1)<br>updated AoI, lagged channel observation,<br>previous action/outcome, scene state"]
    LOC --> R["REWARD r(t)<br>PRIMARY freshness signal: − ONE composed loc_error<br>+ segmentation mIoU + pedestrian/object recall<br>- realized PRB-time ∝ payload × FPS × (1+retx) / SE(MCS)<br>- configurable last-resort ROI penalty<br>± LIGHT delivery/drop + C1 estimate-miss diagnostics only"]
    ENV --> R
    S2 --> MASK
    oai -. telemetry at t+lag plus noise .-> est
    mp -. delivered capture timestamp / publish time .-> AOI
    ENV -. "feeds back as (t-1) action + outcome" .-> prev
    oai -. "if admitted offered load > hidden true UL service capacity:<br>estimate-miss congestion (BSR → 48 MiB,<br>latency → seconds, delivery cliff)" .-> MISS["C1 ESTIMATE-MISS DIAGNOSTIC<br>not oracle-preventable; log + feed estimator"]
    MISS -.-> est
    C -. "no action meets C2 (fast object + deep fade)" .-> DEG["GRACEFUL DEGRADATION<br>emit min-localization-error C1-admissible action (ROI-escalate),<br>flag frame over-budget<br>= operating-envelope result, NOT a deadlock"]
    DEG -.-> off
    R -. learn .-> POL

     ch
     sp
     em
     prev
     age
     gate
     quant
     ae
     fps
     roi
     off
     truth
     oai
     mp
     src1
     src2
     src3
     src4
     src5
     est
     MASK
     POL
     C
     AOI
     LOC
     S2
     R
     MISS
