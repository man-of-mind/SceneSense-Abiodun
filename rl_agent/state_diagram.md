---
config:
  layout: dagre
  theme: redux
---
flowchart LR
 subgraph S["STATE  s_obs(t) — lagged / noisy observation only"]
    direction TB
        ch["Channel state:<br>SNR/CQI, MCS, BLER/HARQ, PRB/TBS/grant telemetry,<br>scheduled UL rate, UE BSR/RLC buffer,<br>estimated sustainable UL budget + confidence"]
        sp["Dynamic objects inside C4 validity range ≤ 40 m:<br>per-object speed + uncertainty<br>(headline loc regime ≤ 25 m pending advisor)"]
        em["Front-side urgency proxy:<br>current-frame objectness + radar/range cue<br>+ tracker prior, not lagged"]
        prev["Previous action + outcome:<br>mode/profile/FPS, latency/delivery"]
        age["Age-of-Information (AoI):<br>now − capture timestamp of newest<br>successfully published map update"]
        comp["Local-compute headroom:<br>available CPU/GPU load + confidence /<br>max sustainable full-local FPS"]
        obs(("s_obs"))
        ch --> obs
        sp --> obs
        em --> obs
        prev --> obs
        age --> obs
        comp --> obs
  end

 subgraph CAT["ACTION CATALOG  a(t) — mode first"]
    direction TB
        mode["mode ∈ {SPLIT, LOCAL, SKIP}"]
        split["SPLIT:<br>front backbone → quant u8→u4 → AE bottleneck<br>→ ROI / spatial crop LAST RESORT → FPS<br>send features for edge fusion"]
        local["LOCAL (provisional until fourth table):<br>run full model on-car → upload small result<br>FPS only; compute-bound; still uses uplink"]
        skip["SKIP:<br>send nothing<br>only network-free action"]
        cand["Flattened discrete candidate catalog"]
        mode --> split
        mode --> local
        mode --> skip
        split --> cand
        local --> cand
        skip --> cand
  end

 subgraph SAFE["SHARED v4 ADMISSION STACK — every deployable controller"]
    direction TB
        masks{{"HARD MASKS on s_obs<br>C1 for SPLIT + LOCAL: payload × FPS ≤ pessimistic UL budget<br>LOCAL: required FPS ≤ measured compute headroom<br>SKIP always admissible"}}
        am["A_m(s_obs) — hard-admitted actions"]
        pred["Per-action surrogate prediction on s_obs<br>post-action AoI for each delivery/drop outcome o<br>e_j = sqrt(base_loc(a)² + (v_j × AoI_next)²)<br>G(a,s,o) = max_j e_j FIRST; empty scene → G=0"]
        exp["E_expected = mean_o[G]<br>small mandatory within-band reward margin"]
        bound["Tail safety statistic + uncertainty<br>E_hat_risk = p95_o[G] or CVaR_alpha,o[G]<br>B = E_hat_risk + k × sigma_hat<br>(or calibrated conformal / quantile bound)"]
        shield{{"LIVE SHIELD<br>F_hat=1 when some action has B ≤ epsilon<br>B* = min over A_m of B"}}
        asafe["A_safe — controller-visible action set"]
        deg["GRACEFUL DEGRADATION  F_hat=0<br>near-best band {B ≤ B* + delta_loc}<br>optimize inside band + flag frame over-budget"]
        ood["shield_ood DEGRADED MODE<br>reject actions lacking a calibrated bound;<br>if none can be bounded, fixed worst-case-risk fallback over A_m<br>do NOT assume SKIP or LOCAL is safest; flag OOD"]
        masks --> am
        am --> pred
        pred --> exp
        pred --> bound
        am --> shield
        bound --> shield
        shield -- "F_hat=1: B ≤ epsilon" --> asafe
        shield -- "F_hat=0" --> deg
        deg --> asafe
        bound -. "outside calibrated support" .-> ood
  end

    src1(["gNB / UE MAC + T-tracer<br>MCS, BLER, BSR/RLC, PRB/TBS,<br>grant rate + scheduled UL rate"]) --> est["Lagged/noisy UL-budget estimator<br>full-resource TBS/grant × attainable grant rate OR<br>MCS efficiency × configured UL resources/time;<br>allocated throughput is a light-load lower bound<br>corroborate with backlog drain + prior outcomes"]
    est --> ch
    src2(["Front perception / tracker<br>object locations, speed + uncertainty"]) --> sp
    src3(["UE front-side urgency monitor<br>front features + radar/range cue<br>+ tracker/map prior"]) --> em
    src4(["Agent memory (t−1)<br>last action + outcome"]) --> prev
    src5(["Map feedback / ACK clock<br>latest published frame_id<br>+ capture timestamp"]) --> age
    src6(["On-device compute monitor<br>CPU/GPU availability, load,<br>sustainable full-local FPS"]) --> comp

    obs --> masks
    cand --> masks
    obs --> pred
    exp --> rin["INNER REWARD / RANKING<br>w_task × U_task − C_UE − C_PRB<br>− 0.5 × C_ROI − 0.1 × C_switch<br>− w_E × E_expected / epsilon, with w_E > 0<br>sampled RL transition substitutes realized G for E_expected"]
    asafe --> ctrl{{"SHARED-SHIELD CONTROLLER<br>shielded one-step oracle / rule / contextual bandit / MPC /<br>masked DQN / discrete SAC / PPO fallback<br>selects only from A_safe"}}
    rin -. "rank / train" .-> ctrl
    ctrl --> act["SELECTED ACTION  a(t)"]
    ood --> act

 subgraph ENV["ENVIRONMENT — vehicle compute + 5G uplink + edge / map transition"]
    direction TB
        splitx["SPLIT execution:<br>front features + selected compression profile"]
        localx["LOCAL execution:<br>full on-car inference + small result"]
        skipx["SKIP execution:<br>no upload; map keeps prior update"]
        offered["offered load = payload × FPS"]
        truth["HIDDEN true current UL service capacity<br>channel + PRB/TDD config + scheduler/link adaptation<br>NEVER exposed to policy, masks, or live shield"]
        oai["5G / OAI uplink outcome:<br>delivery/drop, latency, PRB-time,<br>BSR backlog, MCS/BLER"]
        edge["SPLIT delivered features:<br>edge intermediate fusion"]
        mp["publish result to shared spatial map"]
        splitx --> offered
        localx --> offered
        offered --> oai
        truth --> oai
        oai -- "delivered SPLIT features" --> edge
        edge --> mp
        oai -- "delivered LOCAL result" --> mp
  end

    act -- SPLIT --> splitx
    act -- LOCAL --> localx
    act -- SKIP --> skipx
    oai --> aoi2["AoI TRANSITION<br>delivered → capture→map pipeline latency<br>drop → previous AoI + control interval"]
    mp -. "delivered capture timestamp / publish time" .-> aoi2
    skipx --> aoi2
    aoi2 --> greal["REALIZED per-outcome localization<br>compute per-object e_j, then G = max_j e_j<br>(same normative object-first order)"]
    sp --> greal
    act --> greal
    greal -- "realized G" --> rin
    oai -- "realized PRB / delivery cost" --> rin

    act --> s2["NEXT STATE  s_obs(t+1)<br>updated AoI, lagged channel observation,<br>previous action/outcome, scene state,<br>local-compute headroom"]
    aoi2 --> s2
    oai --> s2
    s2 -. "next control step" .-> obs
    oai -. "telemetry at t+lag plus noise" .-> est
    mp -. "published frame metadata" .-> age
    act -. "mode/profile/FPS" .-> prev
    oai -. "latency/delivery outcome" .-> prev
    oai -. "admitted offered load exceeds hidden capacity:<br>congestion / estimate miss" .-> miss["C1 ESTIMATE-MISS DIAGNOSTIC<br>log + feed estimator; not oracle-preventable"]
    miss -.-> est
    truth -. "offline evaluation only" .-> clair["CLAIRVOYANT TRUE-STATE ORACLE<br>non-deployable upper bound;<br>never feeds policy / masks / live shield"]

    ch
    sp
    em
    prev
    age
    comp
    obs
    mode
    split
    local
    skip
    cand
    masks
    am
    pred
    exp
    bound
    shield
    asafe
    deg
    ood
    ctrl
    act
    splitx
    localx
    skipx
    offered
    truth
    oai
    edge
    mp
    aoi2
    greal
    rin
    s2
    miss
    clair
