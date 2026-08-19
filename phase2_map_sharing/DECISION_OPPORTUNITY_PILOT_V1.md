# Phase-2 decision-opportunity pilot v1

Status: **designed, not visually accepted, not collected**.

This is a bounded repair of the scenario timing, not a tracker sweep, a new
corpus, or C2 evidence.  The accepted curbside legal-opposing geometry remains
unchanged.  The only physical treatment change is that the registered
pedestrian begins moving at 2.0 s instead of 3.0 s.  The 1.3 m/s pedestrian,
both ego transforms and speeds, the grounded Sprinter, the two byte-frozen
routes, Epic rendering, 10 Hz sensor/world clock, and matched benign treatment
remain fixed.

## Why this single change

In the accepted 3.0 s run, the frozen v3 tracker contains five-frame helper
target chains at about 4.6--5.0 s, while the recipient first confirms the
target around 6.4 s.  The sensing asymmetry exists, but CARLA's safety route
controller begins yielding at about 4.8 s and the recipient is almost stopped
by 5.4 s.  Advancing pedestrian motion by one second is the smallest candidate
that should move the helper evidence to about 3.6--4.0 s, while the recipient
is still approaching near 4.4 m/s.  This is a preregistered hypothesis; only
the instrumented pilot can accept it.

The pilot-only retained raw window is `[3.0, 7.0] s` (40 frames).  Lightweight
detections, causal tracks, ego states, decisions, actor truth, and yield
telemetry cover all 120 frames.

## Three trajectories only

1. `controlled_positive_occlusion`: revised 2.0 s pedestrian timing.
2. `matched_benign_negative`: identical route, seeds, static context, cameras,
   and ego commands, with only the pedestrian absent.
3. `naturalistic_operation`: the already reviewed naturalistic denominator;
   no registered target is inserted.

No full collection, OAI replay, controller ladder, or RL follows
automatically.  Any failed gate stops this direction without a parameter
sweep.

## Manual geometry gate

Run the positive in an empty Town10HD_Opt world with the CARLA server in Epic
quality:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m data_collection.review_phase2_pair_geometry \
  --layout curbside_opposite \
  --scenario-role controlled_positive_occlusion \
  --duration-s 12 \
  --helper-speed-mps 4.5 \
  --recipient-speed-mps 5.0 \
  --pedestrian-start-delay-s 2.0 \
  --pedestrian-speed-mps 1.3
```

The operator must observe and report all of the following:

- helper follows road 17 lane `+1`; recipient follows road 10 lane `-2`;
- no U-turn, overlap, collision, or visually implausible vehicle motion;
- the Sprinter is fixed and grounded;
- pedestrian begins northbound at `2.1 +/- 0.1 s` and crosses at realistic
  walking speed;
- helper sees the pedestrian clearly before the recipient can see around the
  Sprinter;
- recipient is still visibly approaching during that helper-only interval;
- recipient later yields safely; and
- both camera views and occlusion geometry remain plausible.

Manual visibility is necessary but not sufficient.  It cannot authorize the
collection or establish a detection/warning lead.

## Instrumented stop/go gates

The positive must show, without GT entering policy state:

- at least five consecutive 10 Hz helper detections classified as pedestrian,
  score at least 0.05, and actor-origin error at most 5 m;
- a confirmed frozen v3 helper track;
- recipient v3 confirmation at least 1.0 s after helper confirmation;
- a helper-derived, truth-positive warning at least 0.5 s before ego-only;
- the first helper-derived warning occurs before the route controller's first
  hidden-actor safety yield and while recipient speed is at least 2.0 m/s; and
- zero registered-target miss.

The matched benign trajectory must keep false-warning active frames at or
below 10% for every arm and cooperative excess at or below +2 percentage
points versus ego-only.  The naturalistic trajectory is reported beside the
designed pair.  Episode/minute remains report-only because three short
trajectories cannot estimate a tail rate.

The positive future-hazard adjudicator uses the matched-benign recipient path
as the no-target counterfactual.  The CARLA route controller's hidden-actor
yield is a safety intervention, never an agent action or cooperative benefit.

## Static truth contract

Each future trajectory snapshots a create-only, hashed Town10 static
vehicle-object catalog before dynamic capture.  Dynamic actor-origin truth
remains a separate per-frame stream.  Warning adjudication matches dynamic
actors first and static environment objects second; a static match is not
automatically a hazard.  Static truth is evaluation-only and never enters the
causal policy observation.

