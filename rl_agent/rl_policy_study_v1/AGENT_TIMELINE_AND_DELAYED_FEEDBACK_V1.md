# Split-only PPO timeline and delayed-feedback contract v1

Status: **DESIGN CONTRACT — NO TRAINING OR DEPLOYMENT CLAIM**

The newer three-component reward discussion is in
[REWARD_FORMULATION_V2.md](REWARD_FORMULATION_V2.md). This document remains
authoritative for delayed, missing and out-of-order feedback behavior.

## The ambiguity we must avoid

If the UE chooses action $a_1$ for frame $f_1$ and no feedback has arrived when
frame $f_2$ becomes ready, the system does **not** know that $f_1$ was lost.
It knows only that the outcome is still pending. The policy may choose $a_2$
without waiting, but it must not fabricate a failure reward for $a_1$.

This means action selection is frame-driven while reward attribution is
event-driven and can arrive late or out of order. For the first experiment,
the registered mode is `tail_only_v1`: the service ticket ends at compact
model-tail feedback, while map outcome is retained as a separate audit stream.

## Frame-by-frame timeline

![Frame-driven SplitFusion agent timeline](AGENT_TIMELINE_V1.svg)

The same flow is retained below as Mermaid source so it can be edited directly
in Markdown.

```mermaid
sequenceDiagram
    autonumber
    participant S as Camera + radar
    participant UE as UE front + PPO-LSTM
    participant R as OAI uplink
    participant E as Edge latest-only worker
    participant M as Spatial map
    participant P as Pending-transition ledger

    S->>UE: frame f1 ready; causal SNR/MCS/BSR
    UE->>UE: encoder + LSTM(h0,c0)
    UE->>UE: policy samples action a1
    UE->>P: OPEN(session, UE, f1, a1, logprob, value, h0, c0)
    UE->>R: feature(f1, a1, capture time, deadline)
    R-->>E: complete reassembly or explicit failure

    E->>E: reconstruct feature + complete model tail
    E->>E: emit wire ACK before starting post-processing
    par independent map/audit path
        E->>E: post-process + filter compact object records
        E-->>M: direct object-map update
        M-->>E: MAP_OUTCOME(f1, install status/time)
        Note over E,M: map outcome is audited, not part of tail_only_v1 reward
    and frame-driven control path
        opt no terminal event for f1 has reached UE by 140 ms
            UE->>P: latch DEADLINE_MISSED_PENDING(f1); not radio loss
        end
        S->>UE: f2 sensor starts near 111.1 ms; action cutoff near 141.1 ms
        alt f1 ACK is visible at the cutoff
            E-->>UE: ACK already arrived<br/>edge event + quality provenance + actual costs
            UE->>UE: stamp receipt; derive feedback latency + 140-ms result
            UE->>P: drain and close f1 by exact identity
            P-->>UE: f1 reward eligible after horizon reconciliation
        else f1 ACK is not yet visible
            UE->>P: drain only events received by this cutoff
            Note over UE,P: f1 remains PENDING; it is not called lost
        end
        UE->>UE: observe tail lag + pending age/count + terminal history
        UE->>UE: LSTM(h1,c1), causal channel forecast influences a2
        UE->>P: OPEN(session, UE, f2, a2, logprob, value, h1, c1)
        UE->>R: feature(f2, a2, capture time, deadline)
    end

    opt f1 ACK arrives after the f2 cutoff
        E-->>UE: late wire ACK(f1,a1)
        UE->>UE: stamp receipt; derive late feedback latency
        UE->>P: close exact f1 ticket; keep latched miss
    end
    alt f2 replaces an older pending frame
        E-->>UE: SUPERSEDED_PENDING(old frame, replacement frame)
        UE->>P: close old ticket as intentional supersession
    else feature never reassembles
        E-->>UE: REASSEMBLY_FAILED(f2) or cumulative status later
        UE->>P: close f2 as transport failure
    end
```

The causal SNR forecast produced at epoch $t$ is trained against the mean SNR
over the next registered 100-ms radio-exposure window observed later, but its
predicted mean and uncertainty are used immediately when choosing $a_t$. This
is proactive prediction, not future-data leakage: the true future window is
unavailable until after the action.

At 9 FPS the nominal capture interval is 111.1 ms. With about 30 ms of sensor
compute, the following action is nominally ready at about 141.1 ms, motivating
a candidate 140-ms feedback deadline. This is a soft/chance deadline, not a
claim that all feedback reaches frame $f_{j+1}$: the present Favorable P50 is
already 141.7 ms before adding ACK return. A late event enters the next
observation cutoff after it is received, often $f_{j+2}$; the agent never
blocks the sensor pipeline merely to force one-step feedback.

## Pending-transition ledger

Each action creates exactly one ticket keyed by

$$
k=(\text{session\_id},\text{ue\_id},\text{frame\_id},\text{action\_id}).
$$

The immutable ticket retains:

- capture/action timestamps, payload and action identity;
- PPO log-probability, value estimate, and incoming LSTM state;
- bytes and compute already consumed; and
- the registered deadline and reconciliation horizon.

The edge's wire `TAIL_COMPLETED_ACK` contains exact identity, the edge
completion event, actual resource charges, separate segmentation/localization
quality anchors (or a qualified proxy), and quality
source/version/catalog hash. It cannot contain a latency measured at its
future UE receipt. The UE stamps receipt, derives the same-clock latency and
140-ms result, then uses that enriched record to close the service ticket in
`tail_only_v1`. The later map event uses a separate audit key. A service ticket
may close only with one terminal outcome:

- `TAIL_COMPLETED_ACK_RECEIVED`;
- `SUPERSEDED_PENDING`;
- `REASSEMBLY_FAILED`;
- `STALE_BEFORE_EDGE`;
- `ACTION_EXECUTION_FAILED`; or
- `FEEDBACK_HORIZON_EXPIRED_AFTER_RECONCILIATION`.

`PENDING` and `DEADLINE_MISSED_PENDING` are states, not terminal outcomes. The
latter is a real service miss but is not proof of radio loss. A missing UDP
feedback packet is also not proof of feature loss. Later cumulative feedback
must carry a terminal watermark and recent per-frame outcomes, or the UE must
query a durable edge ledger before an unresolved ticket may expire.

Feedback identity is validated before credit assignment. Duplicate identical
feedback is idempotent; conflicting feedback for the same key is an integrity
failure. Out-of-order feedback closes the matching ticket, never merely the
most recent action.

## What frame ID means to the policy

Raw frame ID is not fed to the neural network as an ever-growing number and is
not itself a reward. It provides exact attribution and causal tail/pending lag
features:

$$
\Delta f^{\mathrm{pending}}_t=f_t-f_{\mathrm{oldest\ pending}}.
$$

For tail feedback define similarly

$$
\Delta f_t^{\mathrm{tail}}
=f_t-f_{\mathrm{latest\ tail\ feedback}}.
$$

At nominal 9 FPS, a lag of three means roughly three 111.1-ms frame periods,
before considering irregular timing. The observation also includes elapsed
milliseconds, last feedback latency divided by 140 ms, deadline result,
quality/proxy availability and value, pending count/age, and rolling on-time
statistics so the agent does not infer time from frame number alone.

## Initial compact reward

For the ticket belonging to frame $j$:

$$
\begin{aligned}
r_j={}&\mathbf{1}_{\mathrm{tail\ completed}}
\left[w_S\bar S_{a_j}+w_G\bar G_{a_j}
-w_T\phi_T(T_j^{\mathrm{fb}})\right]\\
&-\beta_TD_j^{\mathrm{miss}}
-\beta_D\mathbf{1}_{\mathrm{proven\ transport\ failure}}
-\lambda_A\mathbf{1}[a_j\ne a_{j-1}].
\end{aligned}
$$

| Term | Meaning |
|---|---|
| $\bar S_{a_j},\bar G_{a_j}$ | Frozen segmentation and localization quality anchors; a live per-frame value requires a qualified proxy. |
| $T_j^{\mathrm{fb}}$ | Sensor-compute-start to compact tail-feedback receipt at the UE. |
| $\phi_T$ | Smooth latency cost for a successful tail completion. |
| $D_j^{\mathrm{miss}}$ | Latched once when no terminal service event has reached the UE by 140 ms, regardless of the eventual outcome. |
| $\Delta f_t^{\mathrm{tail}}$ | Tail-feedback frame lag used in the next available observation; raw frame ID is only the attribution key. |
| $\mathbf{1}_{\mathrm{transport\ failure}}$ | One only for a proven incomplete/failed feature delivery, not for intentional supersession. |
| $B_j/B_{\max}$ | Separate normalized byte-cost signal for the named byte cost critic. |
| $C_j/C_{\max}$ | Separate normalized compute-cost signal, including work before supersession, for the named compute cost critic. |

This version trains a tail-ready perception service. Tail completion means the
frame is ready to enter post-processing/map service; it does not claim that the
map installed it. Object-level map utility remains a later registered reward
version if the application requires different freshness tolerances for
pedestrians and vehicles.

`SUPERSEDED_PENDING` gets no fictitious quality credit and no transport-loss
penalty. It still pays bytes and compute already consumed. The explicit
replacement frame ID tells the agent why the old work vanished.

## Recurrent PPO bookkeeping

Each rollout has a fixed actor-policy version and a declared terminal decision
horizon. Collection stops at that horizon, and generalized-advantage estimation
or parameter updates wait until **every ticket through the horizon** is
terminally reconciled. A merely contiguous prefix does not authorize an update
while an already-collected unresolved suffix would be made stale. Each UE and
episode has its own ordered ledger. Each stored step retains its incoming LSTM state so
the old-policy log probability can be reproduced exactly. It also retains the
detached forecast features used by the sampled policy; PPO re-evaluation may
not silently recompute them after an auxiliary forecast-head update. Model
parameters are frozen during rollout, and measured KL is checked during PPO
epochs. Episode shutdown
drains or explicitly terminalizes every ticket; unresolved tickets cannot be
silently dropped from training. If a later CARLA quality correction is enabled,
the affected transition remains unreleased until that correction arrives;
standard on-policy PPO must never consume a provisional reward and mutate it
afterward.

These are necessary bookkeeping conditions for on-policy credit assignment
when feedback for $f_1$ arrives after actions for $f_2$ and $f_3$; the PPO
ratio/clip and KL checks remain necessary as well.
