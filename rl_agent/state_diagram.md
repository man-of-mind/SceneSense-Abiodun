---
config:
  layout: dagre
  theme: redux
---
flowchart LR
  %% Phase-2 causal design. Runtime state is available before the action.
  %% Current-frame inference outputs and CARLA truth never feed the placement decision.

  subgraph PRE["CAUSAL PRE-ACTION STATE  s_pre(t)"]
    direction TB
    net["Lagged network estimate<br/>capacity + uncertainty, prior MCS/BLER/BSR,<br/>previous delivery and latency"]
    map["Previously installed recipient map<br/>causal tracks, AoI, covariance,<br/>capture/install provenance"]
    trk["Prior completed source-local tracks<br/>detections available before decision only<br/>no GT actor IDs"]
    ego["Timestamped kinematics<br/>current helper-local state + most recent<br/>causally received recipient state"]
    sched["Local scheduler and compute state<br/>credit, in-flight summaries,<br/>available LOCAL headroom"]
    allow{{"CAUSAL ALLOWLIST<br/>source/observed/available timestamps<br/>+ consuming decision ID/stage<br/>assert available_at_s ≤ referenced decision_at_s"}}
    net --> allow
    map --> allow
    trk --> allow
    ego --> allow
    sched --> allow
  end

  subgraph PLACE["PRE-INFERENCE PLACEMENT DECISION"]
    direction TB
    pcand["Placement candidates<br/>SPLIT_FEATURE profile/FPS<br/>LOCAL_INFER profile/FPS<br/>SKIP_INFERENCE"]
    pmask{{"HARD ADMISSION<br/>SPLIT: estimated UL budget<br/>LOCAL: compute headroom<br/>all: declared uncertainty/deadline contract"}}
    pctrl{{"Simplest causal controller first<br/>exact enumerator / fixed / rule / greedy / MPC<br/>RL only after registered residual gap"}}
    pa["PLACEMENT ACTION<br/>at placement_decision_at_s"]
    pcand --> pmask
    allow --> pmask
    pmask --> pctrl --> pa
  end

  subgraph INFER["ACTION-CONDITIONED INFERENCE"]
    direction TB
    split["SPLIT_FEATURE<br/>capture → head → feature encode<br/>→ OAI/edge tail"]
    local["LOCAL_INFER<br/>capture → full local inference"]
    noinfer["SKIP_INFERENCE<br/>no new perception result"]
    result["POST-INFERENCE RESULT<br/>unfiltered detections + final detections,<br/>causal tracker IDs, covariance,<br/>capture/inference provenance"]
    split --> result
    local --> result
  end

  pa -- SPLIT_FEATURE --> split
  pa -- LOCAL_INFER --> local
  pa -- SKIP_INFERENCE --> noinfer

  subgraph PUB["POST-INFERENCE PUBLICATION DECISION"]
    direction TB
    pubcand["Publication candidates<br/>PUBLISH_ALL<br/>PUBLISH_HAZARD_SUBSET<br/>SKIP_PUBLICATION"]
    hsel{{"Recipient-conditioned causal selection<br/>current result + newest recipient state<br/>available at the logged decision locus<br/>no future truth / GT identity"}}
    pub["PUBLICATION ACTION<br/>at publication_decision_at_s<br/>scenesense.map_contribution.v2<br/>state + covariance + motion model<br/>source/recipient/time/byte provenance"]
    nopub["SKIP_PUBLICATION<br/>retain prior map"]
    result --> pubcand --> hsel
    hsel -- publish --> pub
    hsel -- skip --> nopub
  end

  subgraph TRANSPORT["TRANSPORT + RECIPIENT MAP"]
    direction TB
    wire["Local path or identical bytes over OAI RFsim<br/>enqueue, first/last byte, reassembly"]
    install["Recipient install<br/>ordering + recipient isolation + association"]
    predict["Uncertainty-aware propagation<br/>z(t+dt)=Fz<br/>P(t+dt)=F P Fᵀ + Q"]
    warn["Recipient warning<br/>first-warning time, TTC, closest approach,<br/>AoI/uncertainty + evidence provenance"]
    pub --> wire --> install --> predict --> warn
    nopub --> predict
    noinfer --> predict
  end

  subgraph ACTUATE["LATER FIXED WARNING-ACTUATION ADAPTER — NOT IN CURRENT PILOT"]
    direction TB
    brake["Identical braking/replanning rule in every arm<br/>warning time → actuation latency → vehicle control"]
    outcome["Physical outcomes<br/>collision / near miss · minimum surface clearance<br/>stop clearance band · deceleration/jerk · route progress"]
    warn -. "future override stage" .-> brake --> outcome
  end

  subgraph EVAL["SEPARATE EVALUATION PLANE — NEVER POLICY STATE"]
    direction TB
    gt["Synchronized CARLA truth<br/>actor IDs, future trajectory, hazard label"]
    cf["Matched no-yield counterfactual recipient trajectory<br/>future-hazard label only; prevents intervention paradox"]
    shadow["Shadow unchosen LOCAL/SPLIT outputs<br/>evaluation_only=true; offline/separate pass<br/>cannot perturb primary timing/resources"]
    metrics["Paired C2/C3 metrics<br/>warning lead, false/missed warning,<br/>bytes, latency, AoI/uncertainty, tracking<br/>later: clearance/collision/comfort with attribution"]
    gt --> metrics
    cf --> metrics
    shadow --> metrics
    warn --> metrics
    outcome --> metrics
  end

  wire -. "prior outcome only<br/>after availability lag" .-> net
  install -. "next decision only" .-> map
  result -. "next decision only" .-> trk

  leak{{"FORBIDDEN CAUSAL LEAKAGE<br/>same-frame result/confidence/track/map quality<br/>cannot feed the placement action that produced it"}}
  result -. forbidden .-> leak
  gt -. forbidden .-> leak
  shadow -. forbidden .-> leak

  gate["TWO-TRAJECTORY PILOT GATE<br/>positive occlusion + matched benign negative<br/>prove causality, raw recoverability, and C2 computability<br/>PASS still requires human review before full collection"]
  metrics --> gate
