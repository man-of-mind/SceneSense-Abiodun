# Phase-2 constraint and outcome catalog

Status: 2026-08-17 constraint ranking retained; 2026-08-19 C2 endpoint
reconciliation **proposed and pending joint review**. This catalog does not
authorize CARLA/OAI collection, RL, or warning-triggered braking.

## 1. Scope and control boundary

The Phase-2 agent controls **perception placement and map publication**, not
steering or braking:

- pre-inference: `SPLIT_FEATURE`, `LOCAL_INFER`, or `SKIP_INFERENCE`, plus a
  measured profile and update rate where applicable;
- post-inference: `PUBLISH_ALL`, `PUBLISH_HAZARD_SUBSET`, or
  `SKIP_PUBLICATION`.

Vehicle stopping is therefore an attributable controller outcome only after a
fixed, identical warning-to-braking/replanning adapter is inserted downstream
of every sharing arm. Until then, stopping distance and collision are
evaluation diagnostics of the scenario controller, not reward earned by a
sharing policy.

The action catalog is **hybrid and measurement-supported**, not an unconstrained
continuous box. Payload/profile choices are the measured Pareto-frontier
points; FPS/update interval may become a bounded continuous parameter only
after interpolation is validated on held-out measurements. Continuous values
such as latency, clearance, AoI, uncertainty, energy, and PRB occupancy are
outcomes and constraints; their continuity does not make an unmeasured action
physically valid.

## 2. Priority order

1. **Causal and structural invariants** — never tradeable for reward.
2. **Physical safety constraints** — collision and minimum clearance once
   warning actuation exists; observed vulnerable-road-user protection applies
   only to objects the causal pipeline has actually observed.
3. **Deadline/service constraints** — make an accepted usable helper track
   available to the recipient consumer before its recipient-specific deadline,
   with declared uncertainty; warning/actuation safety remains a separate later
   layer.
4. **Network and compute feasibility** — do not knowingly enqueue work that
   exceeds conservative radio or local-compute capacity.
5. **Task utility and efficiency** — perception quality, recipient-available
   information gain, secondary warning lead, bytes, PRB-time, UE compute,
   switching, comfort, and progress.

This is a constrained/lexicographic design. A scalar reward may rank actions
*inside* the admitted set, but a tuned weight must not buy permission to violate
causality or a hard physical-safety invariant.

## 3. Catalog

| ID | Constraint or outcome | Causal measurement | Treatment now | Later acceptance/evaluation |
|---|---|---|---|---|
| S0 | No future truth, GT actor identity, same-action output, shadow result, scenario clock/label, authored hazard schedule, or driver-profile identity in policy state | timestamped allowlist, anti-memorization manifest, and consuming-decision audit | hard reject | zero violations |
| S1 | Action is supported by the measured profile/FPS/local table | catalog membership and config hash | hard mask | zero unsupported actions |
| S2 | Recipient/source isolation and monotone contribution ordering | recipient ID, sequence, capture/install times | hard reject | zero cross-recipient or time-order violations |
| N1 | Conservative uplink admission | lagged capacity estimate + uncertainty, payload, target rate, in-flight bytes | hard mask for known infeasibility; estimate misses logged | C1 false-admit/false-reject and overload rate |
| N2 | Usable helper evidence available at the recipient before its deadline | capture/enqueue/reassembly/install/consumer-available times and typed actionability slack | service constraint | deadline-stratified availability, tail latency, starvation, bytes/PRB-time |
| N3 | Queue stability | BSR/queue, replacement/drop state, service estimate | bounded backlog/drop-expired work | queue tail and recovery after overload |
| K1 | Local inference feasibility | measured local p50/p95, sustainable FPS, bounded queue/headroom | hard mask after LOCAL calibration | deadline misses, compute utilization |
| K2 | UE energy/thermal budget | measured energy and thermal state | deferred until measured | energy/update and thermal throttling |
| M1 | Map contribution validity | confidence, covariance, motion-model ID, process noise, validity horizon | hard schema/TTL checks | stale rejection and calibration |
| M2 | Recipient information gain | recipient-self availability versus accepted usable-helper consumer availability, AoI, covariance, track quality | proposed primary C2 information endpoint, not a raw send bonus | recipient-available track gain/censoring, misses, false/duplicate/fragmented tracks, benign map pollution |
| W1 | Warning service quality | causal warning, AoI, covariance, deadline debt | secondary; present v3 stack failed unchanged specificity gates | warning lead, misses, false-warning exposure |
| M3 | Perception utility | segmentation, pedestrian recall, vehicle recall from frozen measured tables/eval | soft utility | class-stratified quality and calibration |
| P1 | Collision | CARLA collision stream, separately from policy state | report-only until warning actuation; then hard/terminal | collision/near-miss rate |
| P2 | Minimum surface clearance | oriented ego/hazard boxes over synchronized truth | report-only now; later safety constraint | minimum and quantile clearance |
| P3 | Stopping placement | first sustained stop and surface clearance to hazard | report-only now; later soft cost inside an advisor-frozen safe band | too-close and unnecessarily-early stop rates |
| P4 | Ride comfort | longitudinal deceleration and jerk | not yet logged/calibrated | peak/p95 deceleration and jerk |
| P5 | Mobility | route progress, delay, unnecessary intervention | not yet attributable | completion, delay, unnecessary-stop rate |
| O1 | Runtime health | dropped frames, actor cleanup, storage quota, clock/sensor contract | hard experiment gate | zero leaks; declared loss/quota status |

## 4. Stopping outcome and future reward position

After warning actuation is wired, define a declared safe/comfortable surface
clearance band `[d_min, d_comfort_max]`. A candidate soft stopping cost is

`C_stop = ((d_min - d_stop)_+ / d_min)^2
        + lambda_early * ((d_stop - d_comfort_max)_+ / d_scale)^2`.

This term distinguishes stopping too close from stopping unnecessarily early,
but it is subordinate to a hard collision/minimum-clearance rule. It must be
reported alongside deceleration, jerk, route progress, and unnecessary
intervention; optimizing clearance alone can reward stopping far too early.
The values of `d_min`, `d_comfort_max`, and `lambda_early` are not frozen by the
pilot and require advisor/application input.

The current pilot can compute collision count, minimum surface clearance,
sustained-stop frame, and stop surface clearance. Its ego bounding box is only
a same-blueprint proxy. The full override corpus must log the recipient ego
bounding box directly.

## 5. SKIP semantics and anti-collapse rule

There is no global `-lambda * I[SKIP]` term. It would punish correct abstention
in empty scenes and can force useless traffic during overload. Instead:

- `SKIP_INFERENCE` is judged by the service debt it creates before inference;
- `SKIP_PUBLICATION` is judged after inference by missed usable hazard evidence,
  recipient-availability/deadline slack, map AoI/uncertainty, and secondary
  warning consequences;
- a hazard-conditioned consecutive-skip/service constraint may be admitted only
  from causal observed evidence, never future truth;
- all registered arrivals remain in deadline denominators, so skip cannot erase
  difficult work; and
- report skip rates stratified by `no demand`, `fresh/safe map`, `compute
  blocked`, `network blocked`, and `reward/service preferred`.

Thus frequent skip is acceptable only where there is no current service need.
The failure mode to prevent is **unserved hazard debt**, not a particular global
skip percentage.

## 6. Reward structure after the causal corpus

Do not tune weights from the completed bounded pilots. Once all required local and
transport tables exist, the candidate ordering is:

1. apply S0--S2 and measured N1/K1 masks;
2. enforce usable-track deadline/service constraints with explicit
   graceful-degradation flags when no action can satisfy them; apply physical
   safety only after a common causal warning-to-actuation adapter exists;
3. rank the remaining actions by task utility minus realized radio, compute,
   switching, service-debt, and—only with fixed warning actuation—comfort costs.

The simplest exact/rule/greedy/MPC controller is evaluated first. Learning is
authorized only if the causal paired corpus exposes a held-out sequential gap
that these controllers cannot close.
