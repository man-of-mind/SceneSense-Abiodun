# SplitFusion RL from first principles

Status: **Study chapter 1 — formulation before implementation**

## 1. Begin with the engineering decision

Every decision epoch, the UE must decide what to do with the newest causal
information available to it:

1. `SPLIT`: run the model front, encode a feature representation, send it over
   OAI, and let the edge run the tail.
2. `LOCAL`: run the complete model locally and send a compact object-map
   result.
3. `SKIP`: send nothing and let existing map tracks become older.

If it chooses `SPLIT`, it must also select one of the 72 registered profiles:

```text
4 model families × 3 quantizers × 6 q values = 72 actions
```

The decision is not "which action has the best model accuracy?"  A large,
accurate payload may arrive too late to be useful.  A smaller payload may have
lower instantaneous accuracy but produce a safer, fresher map.  Conversely,
extreme compression may destroy detection coverage and cannot be justified by
good matched-object XY error alone.

The true goal is therefore:

> Maintain the most useful and freshest object map permitted by the current
> communication and compute state, while controlling safety risk, airtime,
> energy and unnecessary switching.

## 2. Why this is a sequential decision problem

The consequences of an action persist:

- a successful update refreshes some map tracks;
- a dropped or stale result leaves those tracks older;
- a missed new pedestrian cannot be recovered by propagating an old track;
- a large payload can occupy buffers and affect later frames;
- switching execution mode may have a cost;
- the network profile evolves with temporal correlation;
- local inference consumes compute headroom that may be needed later.

A rule that maximizes only the current frame can therefore choose an action
that makes later states worse.  This is why reinforcement learning is a
candidate rather than ordinary multiclass classification.

## 3. Core definitions

### 3.1 Markov decision process

An MDP is commonly written

\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma).
\]

- \(s_t\in\mathcal{S}\): the environment state at decision time \(t\).
- \(a_t\in\mathcal{A}\): the selected action.
- \(P(s_{t+1}\mid s_t,a_t)\): how the environment evolves.
- \(R(s_t,a_t,s_{t+1})\): the immediate reward.
- \(\gamma\in[0,1)\): how strongly future reward matters.

The discounted return is

\[
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}.
\]

A larger \(\gamma\) makes the policy care more about future map freshness and
queue consequences.  It should be related to a physical time horizon, not
chosen only because `0.99` is common.  For a 100 ms decision interval, a
continuous-time half-life \(T_{1/2}\) corresponds to

\[
\gamma=2^{-\Delta t/T_{1/2}}.
\]

For example, a two-second half-life with \(\Delta t=0.1\) s gives
\(\gamma\approx0.966\).

### 3.2 Policy

A policy is a conditional distribution over actions:

\[
\pi_\theta(a_t\mid s_t).
\]

The parameters \(\theta\) are the neural-network weights.  During training the
policy samples actions so it can explore.  During evaluation it can select the
highest-probability feasible action or sample under a fixed, reported rule.

### 3.3 Value, action value and advantage

The value function is the expected future return from a state:

\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s].
\]

The action-value function also conditions on the first action:

\[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a].
\]

The advantage asks whether an action is better than the policy's normal choice
at that state:

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

PPO uses estimated advantages to increase the probability of actions that did
better than expected and decrease the probability of actions that did worse.

## 4. This system is better described as a POMDP

The UE never observes the full physical state.  It sees delayed/noisy SNR and
scheduler telemetry, previous outcomes, local utilization and a current causal
sensor/radar summary.  It does not know the next fade, the true unobserved
object set, or whether the next fragmented message will arrive intact.

A POMDP distinguishes latent state \(s_t\) from observation \(o_t\):

\[
o_t\sim O(o_t\mid s_t).
\]

The policy should act on causal observation history.  A GRU constructs a
learned summary

\[
h_t=f_\theta(h_{t-1},o_t),
\qquad
\pi(a_t\mid h_t).
\]

Why recurrence is useful here: an observed SNR of 12 dB can mean something
different while entering a fade than while recovering from one.  Queue trend,
recent delivery and map-age trend also require history.

Recurrence does not legalize future or privileged information.  The following
remain forbidden policy inputs:

- future SNR or authored network-profile identity;
- ground truth, future trajectory or scenario label;
- current-frame tail detections before choosing how to process that frame;
- absolute frame/sample identifiers that allow route memorization;
- the outcome of the action being selected.

## 5. State and observation design

The first observation vector should be compact and auditable.  Every feature
must be available before the action.

| Block | Candidate causal features | Purpose |
|---|---|---|
| Network | lagged SNR, MCS, delivered throughput, retransmission/drop summaries, RLC/BSR occupancy, recent ACK delay | Estimate whether a payload can arrive usefully |
| Map | maximum/mean/percentile object AoI, critical-track AoI, number of unconfirmed tracks, freshness slack | Express what information is becoming stale |
| Scene | ego speed, aligned radar activity/risk, causal object/track summary already present before this action | Express urgency without GT leakage |
| Compute | CPU/GPU utilization, memory headroom, local-inference availability, preparation queue state | Determine whether `LOCAL` or `SPLIT` can execute |
| History | prior mode/action, outcome, installed AoI, in-flight age/count, time since useful install | Capture hysteresis and delayed consequences |

Features should be normalized using training-split statistics frozen before
evaluation.  Missing telemetry requires an explicit availability bit; silently
turning missing values into zero confuses "not measured" with a real zero.

## 6. Hierarchical action formulation

Let the top-level mode be

\[
m_t\in\{\text{SPLIT},\text{LOCAL},\text{SKIP}\}.
\]

For `SPLIT`, let \(p_t\in\{0,\ldots,71\}\) be the registered profile.  The
policy factorizes as

\[
\pi(a_t\mid h_t)
=\pi_m(m_t\mid h_t)
\begin{cases}
\pi_p(p_t\mid h_t,m_t=\text{SPLIT}), & m_t=\text{SPLIT},\\
1, & m_t\in\{\text{LOCAL},\text{SKIP}\}.
\end{cases}
\]

For a split action, its log probability is

\[
\log\pi(a_t\mid h_t)
=\log\pi_m(\text{SPLIT}\mid h_t)
+\log\pi_p(p_t\mid h_t,\text{SPLIT}).
\]

This is more faithful than one opaque flat label because the first decision is
*where or whether to compute*, while the second is *how to encode a split
feature*.  A flat 74-action implementation is possible, but it loses that
structure and makes a future continuous parameter awkward.

`LOCAL` CPU/GPU placement should initially be a deterministic feasibility
rule: prefer a qualified GPU when available and fall back to a qualified CPU.
Making the device a learned third head is deferred until both paths have causal
latency, energy and quality measurements.

## 7. What “continuous ROI” means in this project

The project's q knob is not an arbitrary image crop or a learned rectangular
ROI.  The stable ranker orders \(N=112\times192=21{,}504\) feature cells.  The
codec drops

\[
D(q)=\operatorname{round}(qN)
\]

lowest-ranked cells and keeps

\[
K(q)=N-D(q).
\]

Examples:

| q | Kept cells |
|---:|---:|
| 0.00 | 21,504 |
| 0.30 | 15,053 |
| 0.50 | 10,752 |
| 0.70 | 6,451 |
| 0.90 | 2,150 |
| 0.98 | 430 |

Thus q can be represented by a continuous scalar, but the executed keep set is
piecewise constant: it changes only when rounding changes \(D(q)\).  The policy
should never directly output 21,504 cell decisions.  That would replace a
qualified stable ranker with a huge combinatorial action and require an
entirely new perception-training contract.

### 7.1 Why v1 remains discrete

The current scientific evidence measures six anchors.  A continuous policy
could select q=0.43 even though its perception and transport distribution was
not independently established.  A learned surrogate may make interpolation
errors, and the agent can exploit those errors during training.

Therefore v1 selects one of the 72 measured profiles.  This gives every action
an auditable checkpoint, payload distribution, quality result and live network
cell.

### 7.2 How the continuous extension would work

After a dense-q study demonstrates predictable interpolation, the split policy
can factor into

1. a categorical branch \(b\) for 4 families × 3 quantizers = 12 branches;
2. a bounded continuous q distribution conditioned on \(h_t\) and \(b\).

A Beta distribution is natural because its support is bounded:

\[
u_t\sim\operatorname{Beta}(\alpha_\theta,\beta_\theta),
\qquad
q_t=q_{\min}+(q_{\max}-q_{\min})u_t.
\]

The primary validated interval should be established prospectively.  The q=.90
and .98 settings are emergency/stress anchors and should not automatically
define the continuous training support.

This is a hybrid or parameterized action policy.  PPO can optimize the sum of
categorical and continuous log probabilities.  The exact project shorthand
would change, but the PPO foundation would not.

## 8. Invalid-action masking

Let \(M_t(a)\in\{0,1\}\) say whether an action is executable in the current
state.  If the network emits logit \(z_a\), the masked policy is

\[
\pi_M(a\mid h_t)=
\frac{M_t(a)e^{z_a}}
{\sum_b M_t(b)e^{z_b}}.
\]

In code, invalid logits are replaced with a large negative value before the
categorical distribution is constructed.  At least one mode must remain
available.

Mask actions for hard facts such as:

- missing/mismatched codec or decoder identity;
- unavailable required model or device;
- insufficient memory for a path whose allocation is known to fail;
- protocol integrity failure;
- a mutually exclusive in-flight rule that makes a second operation impossible.

Do **not** mask an action merely because it is dominated on average, misses a
quality gate, is labelled emergency, or is predicted to be slow.  Those are
performance outcomes the policy is supposed to learn.  Corrupt/non-finite
actions remain invalid, while poor but valid profiles remain agent-enabled.

## 9. PPO mathematics

PPO collects trajectories with an old policy \(\pi_{\theta_{old}}\), estimates
advantages, and updates a new policy without allowing one batch to change it
too aggressively.

The probability ratio is

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid h_t)}
{\pi_{\theta_{old}}(a_t\mid h_t)}.
\]

The clipped policy objective is

\[
L^{clip}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

If an update tries to make an advantageous action vastly more likely, or a bad
action vastly less likely, clipping limits how much that sample can drive the
update.

Generalized advantage estimation starts from temporal-difference residuals

\[
\delta_t=r_t+\gamma V(h_{t+1})-V(h_t)
\]

and forms

\[
\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
\]

The parameter \(\lambda\) trades lower variance against greater bias.  The
complete training loss normally combines the negative clipped objective, a
value-regression loss and an entropy bonus.  Recurrent training must preserve
sequence order, reset hidden state at episode boundaries and mask padded
timesteps.

## 10. Why constraints should not be hidden inside reward

Some preferences can trade against one another; others cannot.

- Airtime versus energy is a normal trade-off.
- A corrupt payload is not something the agent may choose for a sufficiently
  large reward.
- Critical-object freshness deserves explicit risk accounting rather than an
  arbitrary giant penalty.

Let \(J_R(\theta)\) be expected reward and \(J_{C_i}(\theta)\) expected cost for
constraint \(i\), with allowed budget \(d_i\):

\[
\max_\theta J_R(\theta)
\quad\text{subject to}\quad
J_{C_i}(\theta)\le d_i.
\]

A practical PPO implementation can use nonnegative Lagrange multipliers:

\[
\mathcal{L}(\theta,\boldsymbol\eta)
=J_R(\theta)
-\sum_i\eta_i\left(J_{C_i}(\theta)-d_i\right),
\qquad \eta_i\ge0.
\]

If a cost remains above its budget, its multiplier increases and makes that
violation more expensive.  This is a practical constrained-learning method,
not a formal runtime guarantee.  Deployment still needs hard protocol checks
and explicit fallback behavior.

## 11. Freshness and localization mathematics

For map object \(j\), define age of information

\[
A_{j,t}=t-t^{capture}_{j,newest}.
\]

A conservative position-error bound is

\[
E^{bound}_{j,t}
\le E^{model}_{j,a}+v^{relative}_{j,t}A_{j,t}.
\]

The square-root composition

\[
E^{rms}_{j,t}\approx
\sqrt{(E^{model}_{j,a})^2+(v^{relative}_{j,t}A_{j,t})^2}
\]

can be reported as a sensitivity model, but it should not be called the safety
bound.

Example: with 0.5 m model error, 10 m/s relative speed and 0.20 s AoI, the
conservative bound is 2.5 m.  At 0.05 s AoI it is 1.0 m.  This is why a smaller,
slightly less accurate payload can sometimes be safer.

Matched-object XY error must never stand alone.  A detector can obtain low XY
error on the few pedestrians it detects while missing many others.  Detection
coverage, person precision/recall/F1 and range-stratified recall must accompany
localization risk.

## 12. Reward: what should be optimized

The cleanest reward is based on the **post-outcome map**, not the label of the
selected action.

Let \(U_{map,t+1}\) summarize useful object coverage, confidence, localization
quality and a smaller segmentation term after delivery/drop/skip has been
resolved.  Then a provisional scalar utility is

\[
r_t =
w_U U_{map,t+1}
-w_A C_{airtime,t}
-w_E C_{energy,t}
-w_C C_{compute,t}
-w_S\mathbf{1}[m_t\ne m_{t-1}].
\]

Separate cost critics can represent:

\[
C_{risk,t}=\max_j
w_j\left[\frac{E^{bound}_{j,t+1}}{\epsilon_j}-1\right]_+,
\]

and an unobserved-critical-object cost that does not disappear merely because
matched XY error is good.  Here \([x]_+=\max(x,0)\), \(w_j\) increases with
object criticality, and \(\epsilon_j\) is the registered tolerance.

Important accounting rules:

1. Delivery changes map state and therefore AoI/utility.  Do not also attach a
   second full-strength delivery bonus; that double-counts one event.
2. `SKIP` has low immediate resource cost but ages existing tracks and cannot
   discover a new object.
3. A failed `SPLIT` does not receive the selected profile's validation quality.
   The old map remains installed and becomes older.
4. Segmentation utility is credited only when policy metadata permits the new
   segmentation layer to replace the previous one.
5. The 100 ms value is a service reference.  Actual transport, install AoI and
   deadline misses remain continuous measured outcomes.
6. The 500 ms ACK timeout is an accounting boundary, not proof that a 450 ms
   update met the 100 ms service reference.

No reward weights should be frozen until each term is normalized on training
data and one-at-a-time sensitivity shows that no term numerically overwhelms
the rest.

## 13. Initial neural architecture

A small auditable starting architecture is sufficient:

```text
causal normalized observation o_t
              │
        2-layer MLP encoder
              │
           GRU state h_t
       ┌──────┼─────────┬────────────┐
       │      │         │            │
   mode head  split   reward V   cost values
    (3 logits) head    (scalar)   (one/constraint)
              (72 logits)
```

The split head is evaluated only when `SPLIT` is chosen.  Both categorical
heads apply their own executable-action masks.  The value heads share the
causal encoder but have separate final layers.

This architecture is intentionally modest.  The research question concerns
the policy induced by measured communication/perception trade-offs, not whether
a very large neural network can memorize four traces.

## 14. Why PPO is selected over the immediate alternatives

### Dueling Double DQN

It is a strong later baseline for the fixed discrete catalog and can reuse
experience efficiently.  It is not the primary choice because the planned
action is conditional/hierarchical, a bounded continuous q extension is
plausible, and multiple explicit cost critics are easier to express in a
policy-gradient formulation.  A recurrent masked DQN is possible, but becomes
more bespoke as these requirements accumulate.

### LinUCB/contextual bandit

It is interpretable and data-efficient, but it optimizes immediate contextual
reward.  It cannot naturally value an action for its effect on later map AoI,
queues and compute headroom.  It remains an important later baseline.

### SAC/TD3

They are attractive for continuous actions, but the first action catalog is
categorical and hierarchical.  Applying them would require a mixed-action
construction before continuous q has even been justified.

### Why not claim PPO is automatically best

The selection is an engineering hypothesis.  PPO is the best *starting fit*
for the intended policy structure.  Its performance must later be compared
against exact lookup/rule, LinUCB and value-based baselines on held-out causal
conditions.  This study deliberately postpones that comparison while building
one primary agent correctly.

## 15. What the 288 measurements provide

The campaign measures each fixed split action under each of four evolving
network profiles.  It can provide conditional distributions for:

- payload and fragmentation;
- transport and stage latency;
- complete reassembly/delivery;
- install AoI and deadline outcomes;
- radio state and queue behavior;
- perception quality for installed results;
- preparation and lifecycle outcomes.

It does **not** directly produce trajectories from a switching policy.  Every
cell holds one action fixed, so counterfactual effects of switching and shared
queue state must be modeled explicitly and later validated live.

The first training environment should therefore be a trace-driven stochastic
emulator:

1. preserve each profile's temporal network evolution;
2. condition outcome models on causal telemetry and the selected action;
3. evolve per-object map AoI after delivery/drop/skip;
4. model bounded queues and in-flight work;
5. use block/episode splits that preserve autocorrelation;
6. evaluate on independent trace seeds or later live runs.

Training and evaluating on the identical 288 cells would measure memorization,
not generalization.

`LOCAL` and `SKIP` are not part of the 72×4 campaign.  `SKIP` has an analytical
state transition, but `LOCAL` requires separate CPU/GPU latency, energy,
quality and compact-upload measurements.  Until those exist, the architecture
may expose `LOCAL` while the training contract marks it unavailable; its
outcomes must not be invented.

## 16. Freeze points before implementation

The following must be reviewed before training code becomes authoritative:

1. exact causal observation schema and normalization;
2. decision interval and episode boundary;
3. `LOCAL` measurement contract;
4. map-state transition for delivery, stale result, drop and skip;
5. reward-term definitions before numerical weights;
6. cost constraints and allowed budgets;
7. train/validation/test trace separation;
8. recurrent hidden-state reset and burn-in rules;
9. deterministic feasibility-mask rules;
10. continuous-q promotion criteria.

Only after these are frozen should we implement the policy network and trainer.

## 17. Short answers for a supervisor

**Is “Recurrent Hierarchical Masked PPO” an existing named algorithm?**

No.  It is project shorthand for established components: PPO, a GRU/recurrent
policy, conditional action heads and invalid-action masking.  We should not
present the phrase as a novel published algorithm.

**Did we consider continuous ROI?**

Yes.  The actual spatial ROI is still chosen by the validated stable ranker.
The agent could later choose continuous q, the scalar fraction of ranked cells
to drop, using a bounded Beta head.  The first agent uses six discrete anchors
because those are the settings with full model and live-network evidence.

**Why not start continuous immediately?**

Because a policy trained through a surrogate can exploit interpolation errors
at unmeasured q values.  Dense-q validation must first show that payload,
detection, localization and segmentation change predictably within each of the
12 family/quantizer branches.

**Where is the project novelty?**

Not in inventing PPO.  It is in the integrated and measured decision problem:
task-aware split inference, live OAI protocol behavior, map freshness,
multi-modal perception, multiple execution modes and later cooperative
critical-information sharing.
