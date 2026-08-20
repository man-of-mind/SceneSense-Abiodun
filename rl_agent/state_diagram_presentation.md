---
config:
  layout: dagre
  theme: redux
---
flowchart LR
  %% Compact UE-only presentation diagram.

  sensor["ALWAYS-ON SENSORS<br/>capture aligned RGB+radar<br/>every frame"]

  state["LEAN CAUSAL STATE — 7 SCALARS<br/>freshness · radar risk · ego speed<br/>pessimistic capacity · in-flight age<br/>local-compute slack · time since process"]

  sensor --> state --> valid{{"Validity + hard masks<br/>causal · physical · freshness"}}
  valid --> choose{{"ONE PRE-MODEL ACTION<br/>SKIP · LOCAL · SPLIT(profile)"}}

  choose -- SKIP --> drop["Drop current frame<br/>from inference<br/>sensors continue"]
  choose -- LOCAL / SPLIT --> front["Common UE front<br/>after action selection"]

  front -- LOCAL --> local["Finish locally<br/>compact result"]
  front -- SPLIT --> split["Compressed feature<br/>edge back half"]

  local --> send["One-shot outbound enqueue<br/>LOCAL result or SPLIT feature<br/>no application buffer/retry"]
  split --> send
  send -- LOCAL accepted --> fixed["Fixed publish-all<br/>to one edge map"]
  send -- SPLIT accepted --> edge_tail["OAI + edge back half"]
  edge_tail --> fixed
  fixed --> validate{{"Edge validation"}}
  validate -- accepted --> install["Edge-map install"]
  install --> ack["Accepted install ACK<br/>advances known freshness"]
  validate -- rejected --> nack["NACK received<br/>no install"]

  send -- immediate drop --> age["Prior known map ages"]
  edge_tail -- delivery failure known --> age
  drop --> age
  nack -- received --> age
  nack -- lost --> timeout["ACK/NACK timeout<br/>no resend"]
  ack -- lost --> timeout
  timeout --> age

  ack -- accepted --> outcome["EDGE OUTCOME<br/>freshness · quality · latency<br/>PRB/bytes · compute"]
  age --> outcome
  outcome -. "next decision only" .-> state

  next["NEXT SENSOR FRAME<br/>always arrives"]
  sensor -. "independent clock" .-> next
  next -. "sensor clock" .-> sensor

  optional["OPTIONAL OFFLINE v1+SI/TI<br/>paired counterfactual ablation<br/>never expands base v1 silently"]
  optional -. "decision-value evaluation" .-> outcome

  no["OUT OF SCOPE NOW<br/>helper/recipient sharing · warning/braking<br/>all geometry-derived policy features"]
  outcome -. "parked Phase 2" .-> no

  leak{{"REJECTED v1 INPUT<br/>front/tail objectness · GT/future<br/>route/scenario/time · geometry"}}
  choose -. "rejects" .-> leak

  classDef obs fill:#dbeafe,stroke:#2a78d6,color:#0b0b0b;
  classDef dec fill:#ece9fb,stroke:#4a3aa7,color:#0b0b0b;
  classDef act fill:#ffe3d3,stroke:#eb6834,color:#0b0b0b;
  classDef env fill:#d7f2e6,stroke:#1baf7a,color:#0b0b0b;
  classDef bad fill:#fde2e1,stroke:#e34948,color:#0b0b0b;
  classDef hold fill:#fff3cd,stroke:#eda100,color:#0b0b0b;

  class sensor,state obs;
  class valid,choose dec;
  class drop,front,local,split act;
  class send,edge_tail,fixed,validate,install,nack,ack,timeout,age,outcome,next,optional env;
  class leak bad;
  class no hold;
