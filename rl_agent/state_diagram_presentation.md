---
config:
  layout: dagre
  theme: redux
---
flowchart LR
  %% ===== data sources =====
  src1(["gNB / UE MAC + T-tracer<br/>MCS, BLER/HARQ, BSR/RLC,<br/>PRB/TBS, grant rate, scheduled UL rate"]):::src
  src2(["Sensor<br/>front perception / tracker<br/>object locations, speed + uncertainty"]):::src
  src3(["Model front<br/>UE front-side urgency proxy<br/>front features + radar/range cue<br/>+ tracker/map prior"]):::src
  src4(["Agent memory (t−1)<br/>last mode/profile/FPS<br/>last latency/delivery"]):::src
  src5(["Map feedback / ACK clock<br/>latest published frame_id<br/>+ capture timestamp"]):::src
  src6(["On-device compute monitor<br/>CPU/GPU load<br/>sustainable LOCAL FPS"]):::src

  %% ===== STATE s_obs =====
  subgraph S["STATE  s_obs(t) — lagged / noisy observation"]
    direction TB
      ch["Channel state + UL budget<br/>SNR/CQI, MCS, BLER/HARQ,<br/>scheduled UL rate, UE BSR/RLC buffer,<br/>estimated sustainable UL budget + confidence"]:::state
      sp["Dynamic objects ≤ 40 m<br/>track_id, speed + uncertainty, range<br/>+ contributions[] provenance"]:::state
      em["Front-side urgency<br/>objectness + radar/range cue<br/>+ tracker prior, current frame"]:::state
      prev["Previous action + outcome<br/>mode/profile/FPS, latency/delivery"]:::state
      age["Per-object shared-map AoI<br/>AoI_map,j = now − capture time of newest valid<br/>contribution for object j, any source"]:::state
      sched["20 Hz scheduler + pending summary<br/>active profile/FPS, send credit,<br/>in-flight count / expected arrival"]:::state
      comp["Local-compute headroom<br/>available CPU/GPU load<br/>sustainable full-local FPS"]:::state
  end

  src1 --> est["Lagged/noisy UL-budget estimator<br/>uses grants/TBS/MCS, backlog drain,<br/>and prior outcomes"]:::state
  est --> ch
  src2 --> sp
  src3 --> em
  src4 --> prev
  src5 --> age
  src4 --> sched
  src6 --> comp

  %% ===== ACTION catalog =====
  subgraph A["ACTION  a(t) — mode first, then knobs"]
    direction TB
      split["SPLIT<br/>front features → quant / AE / ROI / FPS<br/>send features for edge fusion"]:::act
      local["LOCAL<br/>full on-car inference → small result upload<br/>FPS limited by compute"]:::act
      skip["SKIP (whole frame)<br/>send nothing; map keeps prior updates"]:::act
  end

  %% ===== admission + controller =====
  S --> MASK{{"C1 HARD ACTION MASK<br/>SPLIT/LOCAL: payload × FPS ≤ pessimistic UL budget<br/>LOCAL: FPS ≤ compute headroom<br/>SKIP is C1-admissible"}}:::con
  A --> MASK

  MASK --> SHIELD{{"LIVE SAFETY SHIELD<br/>predict AoI_map,j,next and localization risk for each action<br/>G = max_j sqrt(base_loc(a)² + (v_j × AoI_map,j,next)²)<br/>safe if tail-risk bound B ≤ ε"}}:::con

  SHIELD --> POL{{"Shielded controller / policy<br/>rule, bandit, MPC, or masked RL<br/>selects only from A_safe"}}:::pol

  POL --> ACT["SELECTED ACTION<br/>SPLIT / LOCAL / SKIP + profile/FPS"]:::pol

  SHIELD -. "if no action meets ε:<br/>choose near-best C1-admissible action<br/>and flag over-budget" .-> DEG["GRACEFUL DEGRADATION<br/>operating-envelope miss,<br/>not a deadlock"]:::con
  DEG -.-> ACT

  %% ===== environment / transition =====
  subgraph ENV["ENVIRONMENT — vehicle compute + 5G uplink + edge map"]
    direction TB
      off["offered load = payload × FPS"]:::env
      truth["HIDDEN true UL service capacity<br/>channel + PRB/TDD config<br/>+ scheduler/link adaptation<br/>(not visible to policy/shield)"]:::env
      flight["rate accumulator + in-flight event queue<br/>newer capture wins"]:::env
      oai["5G / OAI uplink outcome<br/>delivery/drop, latency, PRB-time,<br/>BSR backlog, MCS/BLER"]:::env
      edge["edge/map processing<br/>SPLIT: edge feature fusion<br/>LOCAL: publish small result"]:::env
      mp["shared spatial map update<br/>freshness / staleness"]:::env
      off --> flight --> oai
      truth --> oai
      oai --> edge --> mp
  end

  ACT -- "SPLIT / LOCAL" --> off
  ACT -- "SKIP" --> AOI

  ENV --> AOI["PER-OBJECT AoI TRANSITION<br/>valid contribution for j → its capture→map latency<br/>otherwise → previous AoI_map,j + control interval"]:::state

  AOI --> LOC["REALIZED localization term<br/>G = max object error<br/>uses selected action's base_loc<br/>+ object speed × AoI"]:::state

  LOC --> R["POST-ACTION MAP UTILITY / TRAINING SIGNAL<br/>w_task E[U_task_post] − w_E E[G]/ε<br/>− C_UE − C_PRB<br/>− λ_ROI C_ROI − λ_switch C_switch<br/>drop/SKIP retain prior map quality"]:::rew
  oai --> R

  ENV --> FB["NEXT-STEP FEEDBACK<br/>lagged RAN telemetry + delivery/latency<br/>map ACK timestamp + previous action/outcome<br/>estimate-miss diagnostics update UL-budget estimator"]:::state
  AOI --> FB
  ACT -. "previous mode/profile/FPS" .-> FB
  FB --> S2["NEXT STATE  s_obs(t+1)<br/>updated per-object AoI + provenance summaries,<br/>lagged channel obs, previous outcome,<br/>scheduler/pending state, scene + compute state"]:::state
  S2 --> MASK

  %% ===== learning feedback =====
  R -. "learn / update policy" .-> POL

  classDef src fill:#eeeeee,stroke:#999999,color:#333333,stroke-width:1.5px;
  classDef state fill:#dbeafe,stroke:#2a78d6,color:#0b0b0b,stroke-width:2px;
  classDef act fill:#ffe3d3,stroke:#eb6834,color:#0b0b0b,stroke-width:2px;
  classDef con fill:#fde2e1,stroke:#e34948,color:#0b0b0b,stroke-width:2px;
  classDef env fill:#d7f2e6,stroke:#1baf7a,color:#0b0b0b,stroke-width:2px;
  classDef pol fill:#ece9fb,stroke:#4a3aa7,color:#0b0b0b,stroke-width:2px;
  classDef rew fill:#fff3cd,stroke:#eda100,color:#0b0b0b,stroke-width:2px;
