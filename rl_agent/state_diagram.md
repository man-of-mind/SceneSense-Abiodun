---
config:
  layout: dagre
  theme: redux
---
flowchart LR
  %% LOCKED CURRENT MILESTONE: one UE -> one edge map.
  %% Sensors capture every frame independently of the controller action.
  %% The v1 action is selected before any model/front output exists.

  capture["ALWAYS-ON SENSOR CAPTURE + ALIGN<br/>RGB 3 channels + radar 4 channels<br/>new capture timestamp every sensor frame"]

  state["EXACT LEAN v1 STATE — 7 SCALARS<br/>1 freshness_slack_s<br/>2 radar_risk<br/>3 ego_speed_mps<br/>4 ul_capacity_lcb_bps<br/>5 in_flight_age_s<br/>6 local_compute_slack_ms<br/>7 time_since_last_processed_s"]

  guards{{"VALIDITY GUARDS — NOT LEARNED INPUTS<br/>RGB/radar alignment · ACK validity<br/>state support · timestamps available before decision"}}

  masks{{"U1/U2 HARD MASKS + SERVICE SHIELD<br/>physical network/compute feasibility<br/>SKIP allowed only under frozen freshness/radar/time rules<br/>minimum-debt fallback if no action meets service"}}

  controller{{"ONE PRE-MODEL DECISION<br/>rule / exact greedy / bandit / MPC first<br/>RL only after a registered temporal gap"}}

  action["ACTION<br/>SKIP_INFERENCE<br/>LOCAL_INFER<br/>SPLIT_FEATURE(profile)"]

  capture --> state --> guards --> masks --> controller --> action

  action -- SKIP_INFERENCE --> skip["DROP THIS CAPTURE FROM INFERENCE<br/>sensors remain on<br/>known edge-map freshness ages"]

  action -- LOCAL or SPLIT --> front["COMMON UE FRONT<br/>runs only after action selection<br/>front/object-head output is not v1 state"]

  front -- selected LOCAL --> local["LOCAL_INFER<br/>finish back half on UE<br/>record local_result_available_at"]
  front -- selected SPLIT --> split["SPLIT_FEATURE<br/>compress measured profile"]

  local --> compact["MEASURED COMPACT RESULT<br/>nominal ~2 KB is a hypothesis"]
  compact --> one_shot["ONE-SHOT DEADLINE-BOUNDED ENQUEUE<br/>LOCAL result or SPLIT feature<br/>no application queue · no old-frame retry"]
  split --> one_shot
  one_shot -- SPLIT feature accepted --> split_wire["OAI FEATURE TRANSPORT<br/>edge back-half inference"]

  one_shot -- LOCAL result accepted --> fixed_pub["FIXED PUBLISH-ALL PATH<br/>not a learned action"]
  split_wire --> fixed_pub
  fixed_pub --> validate{{"EDGE VALIDATION<br/>capture ordering · schema · provenance"}}
  validate -- accepted --> install["EDGE MAP INSTALL<br/>newer capture ID wins"]
  validate -- rejected / no install --> nack["REJECTED STATUS / NACK<br/>accepted=false · edge_install_at=null<br/>observable only at nack_received_at"]
  install --> ack["ACCEPTED INSTALL ACK<br/>capture_id · capture_timestamp<br/>edge_install_at · accepted<br/>UE logs ack_received_at"]

  one_shot -- immediate backpressure / drop --> retain["RETAIN PRIOR KNOWN MAP STATE<br/>no freshness credit"]
  split_wire -- delivery failure known --> retain
  skip --> retain
  nack -- received --> retain
  nack -- lost --> timeout["DECLARED ACK/NACK TIMEOUT<br/>clears in-flight state · no resend"]
  ack -- lost --> timeout
  timeout --> retain
  ack -- received and available --> edge_outcome["NEXT KNOWN EDGE STATE<br/>freshness advances only from accepted ACK<br/>quality · latency · delivery · PRB/bytes"]
  retain --> edge_outcome

  edge_outcome -. "next decision only" .-> state

  capture_next["NEXT SENSOR FRAME<br/>capture continues during SKIP,<br/>compute, transport, and ACK wait"]
  capture -. "independent sensor clock" .-> capture_next
  capture_next -. "next sensor tick" .-> capture

  visual["OPTIONAL OFFLINE v1+SI/TI<br/>counterfactual controller ablation<br/>complexity/activity, not density<br/>never silently expands base v1"]
  visual -. "paired decision-value evaluation" .-> metrics

  subgraph EVAL["SEPARATE EVALUATION PLANE — NEVER POLICY STATE"]
    direction TB
    truth["Hidden scene/channel truth<br/>GT actors · future motion · true capacity"]
    counter["Unchosen-action counterfactuals<br/>offline/table model only"]
    metrics["UE metrics<br/>deadline misses · p50/p90/p95+CI<br/>stage latency · edge freshness<br/>task utility · PRB/bytes · compute<br/>action and ACK/drop reasons"]
    truth --> metrics
    counter --> metrics
    edge_outcome --> metrics
  end

  leak{{"FORBIDDEN v1 INPUT<br/>front/tail/object-head result · objectness<br/>GT/future channel · route/scenario/time ID<br/>speed limit · junction/occluder/stopping geometry"}}
  front -. forbidden .-> leak
  local -. forbidden .-> leak
  split -. forbidden .-> leak
  truth -. forbidden .-> leak
  counter -. forbidden .-> leak

  parked["PHASE 2 PARKED<br/>helper/recipient sharing · warning/braking<br/>dynamic occluder/cooperative information"]
  metrics -. "later separate authorization" .-> parked

  classDef obs fill:#dbeafe,stroke:#2a78d6,color:#0b0b0b;
  classDef dec fill:#ece9fb,stroke:#4a3aa7,color:#0b0b0b;
  classDef act fill:#ffe3d3,stroke:#eb6834,color:#0b0b0b;
  classDef env fill:#d7f2e6,stroke:#1baf7a,color:#0b0b0b;
  classDef eval fill:#eeeeee,stroke:#888888,color:#222222;
  classDef bad fill:#fde2e1,stroke:#e34948,color:#0b0b0b;
  classDef hold fill:#fff3cd,stroke:#eda100,color:#0b0b0b;

  class capture,state obs;
  class guards,masks,controller dec;
  class action,skip,front,local,split act;
  class compact,one_shot,split_wire,fixed_pub,validate,install,nack,ack,timeout,retain,edge_outcome,capture_next env;
  class truth,counter,metrics,visual eval;
  class leak bad;
  class parked hold;
