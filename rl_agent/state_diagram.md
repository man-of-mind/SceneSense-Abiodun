---
config:
  layout: dagre
  theme: redux
---
flowchart LR
 subgraph S["STATE  s(t) — observation"]
    direction TB
        ch["Channel state (observed WITH LAG):<br>SNR/CQI, MCS, BLER/HARQ,<br>scheduled UL rate, UE BSR/RLC buffer"]
        sp["Object speed (+ uncertainty)"]
        em["Scene-emptiness / urgency gate<br>(current frame, not lagged)"]
        prev["Previous action + outcome<br>last payload/FPS, last latency/delivery"]
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
        oai["OAI uplink:<br>delivery, latency, BSR backlog, MCS/BLER"]
        mp["edge fuses to shared map<br>(freshness / staleness)"]
  end
    src1(["gNB / UE MAC + T-tracer<br>MCS, BLER, BSR, RLC occupancy, UL sched rate"]) --> ch
    src2(["Front perception / tracker<br>object speed + uncertainty"]) --> sp
    src3(["UE front backbone, pre-transmit<br>frame objectness / urgency<br>(near object, occlusion risk)"]) --> em
    src4(["Agent memory (t-1)<br>last action + outcome"]) --> prev
    S --> POL{{"Policy π(a|s) — RL, safety-constrained<br>chooses payload / FPS / send under ESTIMATED UL budget<br><i>intuition: payload × FPS ≤ estimated UL budget</i>"}}
    POL --> A
    A --> C{{"CONSTRAINTS (safety)<br>C1 payload × FPS ≤ channel budget — HARD (never congest)<br>C2 v × total_staleness ≤ sqrt(eps² - floor²) — SOFT (best-effort)<br>&nbsp;&nbsp;&nbsp;&nbsp;total_staleness = sensor prep + front + uplink + edge/map; floor ≈ 1.1 m<br>C3 keep seg-safe floor ≥ 90 KB (ae32/u4/ROI0) unless forced<br>C4 object range ≤ ~40 m (perception-valid region)"}}
    off --> oai
    oai --> mp
    C -- payload & FPS admitted --> ENV
    ENV --> S2["NEXT STATE  s(t+1)"] & R["REWARD r(t)<br>+ fresh-delivered map update<br>+ localization accuracy / low staleness error<br>- network-resource cost (PRB-time / payload)<br>- dropped or stale frames"]
    S2 --> POL
    mp -. delivery / latency / BSR / MCS → next channel obs .-> ch
    mp -. staleness × speed → localization error .-> sp
    ENV -. "feeds back as (t-1) memory" .-> prev
    C -. if C1 violated: CONGESTION COLLAPSE<br>BSR → 48 MiB, latency → seconds, delivery cliff .-> oai
    C -. "no action meets C2 (fast object + deep fade)" .-> DEG["GRACEFUL DEGRADATION<br>emit min-localization-error action (ROI-escalate),<br>flag frame over-budget<br>= operating-envelope result, NOT a deadlock"]
    DEG -.-> ENV
    R -. learn .-> POL

     ch
     sp
     em
     prev
     gate
     quant
     ae
     fps
     roi
     off
     oai
     mp
     src1
     src2
     src3
     src4
     POL
     C
     S2
     R