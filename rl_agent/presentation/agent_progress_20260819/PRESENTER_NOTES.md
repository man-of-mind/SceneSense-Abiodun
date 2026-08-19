# Presenter notes — SceneSense Agent progress review

## Slide 1 — SceneSense Agent
Lead with the refined objective: the project is not committed to RL. We are
building a causal cooperative-perception controller and will use measured
evidence to decide whether learning is necessary.

## Slide 2 — One scene defines the problem
Point to the pedestrian in the helper view and the van/occlusion in the
recipient view. The agent trades three scarce resources: uplink, local compute,
and map freshness.

## Slide 3 — Causal two-stage controller
Emphasize that placement and publication happen at different times. The red
band is the key validity safeguard: the current inference output cannot select
the inference action that produced it.

## Slide 4 — State
Walk through network, prior map, motion, and runtime fields. Each field carries
source and availability timestamps. CARLA truth is evaluation-only.

## Slide 5 — Action space
Profiles and semantic actions are discrete because they are measured. FPS may
be made continuous only after interpolation is validated. This is why
continuous SAC is not the current choice.

## Slide 6 — Worked scenario loop
This is an illustrative step, not a result. Describe how the same scene may
lead to SPLIT, LOCAL, or SKIP depending on lagged channel, map state, and local
headroom—and how that outcome changes the next state.

## Slide 7 — Transition model
Explain the three coupled dynamics: queue service, object state, and covariance.
The actionable deadline becomes physically meaningful only after reaction and
braking assumptions are frozen.

## Slide 8 — Reward v5
Pedestrian recall is highest, segmentation remains substantial, and vehicle
recall stays explicit. The masks happen before the inner objective. Reward
weights are not yet tuned from the tiny pilot.

## Slide 9 — Reward effects
Explain why there is no global SKIP penalty. Correct abstention is useful;
unserved hazard debt is the failure. Stopping distance enters only after a
common warning-actuation adapter makes it attributable.

## Slide 10 — Constraint ranking
This is a constrained/lexicographic problem. A large reward cannot compensate
for a causal leak or unsupported action.

## Slide 11 — Constraint relationships
Physical context sets the deadline. Network/compute determine feasibility.
Transport outcomes determine map age and uncertainty. Reward ranks the safe,
feasible survivors.

## Slide 12 — Physical constraints
Distinguish current report-only physical outcomes from later hard constraints.
Mention legal lanes, realistic traffic, matched futures, and actor cleanup as
experiment-validity requirements.

## Slide 13 — Network/compute constraints
Use the measured heatmap to show the payload-dependent cliff. LOCAL is not a
free fallback: it needs measured local latency and sustainable FPS.

## Slide 14 — Environment
Training is offline/replay and Gym-style, decoupled from CARLA. Paired arms see
the same immutable evidence; truth is attached only by evaluation.

## Slide 15 — Designed scenarios
The six families intentionally create decision opportunities across pedestrians
and vehicles. Every positive has a benign twin.

## Slide 16 — Naturalistic suite
Designed cases can flatter a controller. Suite B is the honest denominator,
reported with the same metrics and grouped confidence intervals.

## Slide 17 — Banked progress
The pilot proves the complete causal artifact chain and warning computability.
Do not quote its lead as performance evidence.

## Slide 18 — Next steps
Ask the advisor to freeze the physical response parameters and LOCAL hardware
target, and to approve only the next bounded calibration stage. End with the
simplest-controller-that-works principle.
