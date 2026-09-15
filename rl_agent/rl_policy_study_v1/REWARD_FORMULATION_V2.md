# Split-only PPO reward formulation v2

Status: **ADVISOR-DISCUSSION DRAFT — DO NOT START TRAINING UNTIL THE LATENCY BOUNDARY AND WEIGHTS ARE LOCKED**

## Design decision

The first policy should optimize three interpretable outcomes for each selected
split action:

1. semantic-segmentation quality;
2. physical localization quality; and
3. action-to-map latency.

The returned frame ID is essential, but it is not itself a numerical reward.
It binds delayed feedback to the exact frame and action that caused it, and it
lets the next observation express how far the installed map is behind.

## Normalized quality terms

For a terminal result belonging to frame $j$, let

$$
S_j=mIoU_{\mathrm{seg},j}\in[0,1]
$$

be the aggregate semantic-segmentation quality. The retained evidence does not
provide separate vehicle/person semantic mIoU, so object-overlap metrics must
not be relabelled as class-specific segmentation IoU.

The current localization coordinate combines box/footprint overlap with
centroid-position accuracy:

$$
Q_{\mathrm{overlap},j}
=\sqrt{\mathrm{IoU}_{\mathrm{vehicle},j}
       \mathrm{IoU}_{\mathrm{person},j}},
$$

$$
e_{xy,j}
=\sqrt{\frac{e_{\mathrm{vehicle},j}^{2}
                   +e_{\mathrm{person},j}^{2}}{2}},
\qquad
Q_{xy,j}=\exp\!\left(-\frac{e_{xy,j}}{d_0}\right),
$$

$$
G_j=\sqrt{Q_{\mathrm{overlap},j}Q_{xy,j}}\in[0,1].
$$

Here $d_0=1\,\mathrm m$ is a presentation normalizer, not a correctness gate.
The geometric means are conservative: excellent vehicle performance cannot
hide failed person localization, and strong overlap cannot hide a large
centroid error.

## Latency penalty

Let $T_j$ denote the agreed causal interval from the action boundary to direct
spatial-map installation. For the current analysis that boundary is the start
of seven-channel concatenation; it must be updated here if the forthcoming
latency-operation review selects a different deployable boundary. Sensor wait,
visual rendering and the later reward-feedback trip are not silently folded
into $T_j$.

Use a bounded smooth penalty:

$$
\phi_T(T_j)=1-\exp\!\left(-\frac{T_j}{\tau_T}\right)\in[0,1].
$$

It is nearly linear for small latency, grows monotonically, and saturates so a
single extreme delay cannot dominate an entire PPO update. The scale $\tau_T$
is a registered service tolerance, not a value estimated after looking at a
training outcome.

## Recommended first reward

For the transition ticket created by frame $j$:

$$
\begin{aligned}
r_j={}&\mathbf{1}_{\mathrm{installed}}
\left[
w_S S_j+w_G G_j-w_T\phi_T(T_j)
\right]\\
&-\beta_D\mathbf{1}_{\mathrm{proven\ transport\ failure}}
-\lambda_B\frac{B_j}{B_{\max}}
-\lambda_C\frac{C_j}{C_{\max}}
-\lambda_A\mathbf{1}[a_j\ne a_{j-1}].
\end{aligned}
$$

Use $w_S,w_G,w_T\ge0$ with $w_S+w_G+w_T=1$ during the first registered weight
sweep. This makes the accuracy/latency trade-off readable. A safety-oriented
study may give localization more weight than aggregate segmentation, but the
actual weights must be agreed before training rather than selected from the
best-looking policy result.

| Symbol | Meaning |
|---|---|
| $S_j$ | Normalized semantic-segmentation mIoU. |
| $G_j$ | Normalized vehicle/person localization coordinate defined above. |
| $T_j$ | Action-boundary to direct-map-install latency for this exact frame. |
| $B_j/B_{\max}$ | Normalized bytes actually transmitted; $B_{\max}$ is a fixed catalog normalizer. |
| $C_j/C_{\max}$ | Normalized compute already consumed, including work spent before supersession. |
| $\mathbf{1}_{\mathrm{installed}}$ | One only when this frame produces a map installation. |
| $\mathbf{1}_{\mathrm{proven\ transport\ failure}}$ | One only after incomplete delivery is proven; silence at the next frame is still `PENDING`. |
| $\mathbf{1}[a_j\ne a_{j-1}]$ | Optional action-switch penalty. |

The first advisor decision is whether bytes, compute and switching remain in
the scalar reward or move to separate cost critics. They are shown explicitly
so their effect cannot be hidden inside the three primary weights.

## Frame identity and delayed feedback

Every action opens a ticket keyed by

$$
k_j=(\mathrm{session\_id},\mathrm{ue\_id},\mathrm{frame\_id},\mathrm{action\_id}).
$$

The terminal feedback record should carry at least

$$
(k_j,\;\mathrm{outcome},\;S_j,\;G_j,\;T_j,\;
\mathrm{installed\_frame\_id},\;\mathrm{replacement\_frame\_id}).
$$

If feedback for frame $j$ has not arrived when frame $j+1$ is ready, ticket
$j$ remains `PENDING`; the agent chooses again using pending age/count, BSR and
the latest installed-frame lag. It must not invent a loss reward. Out-of-order
feedback closes the exact ticket, not merely the most recent action.

Frame identities yield causal observation features such as

$$
\Delta f_t=f_t-f_{\mathrm{latest\ installed}},
\qquad
\Delta f^{\mathrm{pending}}_t
=f_t-f_{\mathrm{oldest\ pending}}.
$$

This is how map freshness first enters the controller: through installed-frame
lag and elapsed time since installation in the next state. Adding raw frame ID
to the scalar reward would be unbounded, episode-dependent and scientifically
meaningless; adding a second arbitrary freshness penalty would also double
count information already represented by $T_j$ and $\Delta f_t$.

## Terminal-outcome rules

- `MAP_INSTALLED`: use measured $S_j$, $G_j$ and $T_j$.
- `PENDING`: assign no reward yet.
- `SUPERSEDED_PENDING`: no fictitious quality and no transport-loss penalty;
  charge bytes/compute already consumed and report the replacement frame.
- `REASSEMBLY_FAILED`: apply the proven-transport-failure penalty.
- stale-before-edge/map or execution failure: close with its distinct terminal
  class; never relabel it as radio loss.
- missing feedback: reconcile against the durable edge ledger before expiry.

## Evidence and deployment boundary

During trace-driven training, $S_j$ and $G_j$ come from the frozen 72-action
validation surface joined to the simulated terminal outcome. CARLA ground
truth is not available to a real deployed vehicle, so live adaptation requires
a separately qualified quality proxy or offline learning followed by fixed
deployment weights. BSR, SNR and MCS are causal observations; they help predict
delivery and latency but are not themselves rewards.

Before training begins, lock: the latency timestamps, $\tau_T$, the weight
grid, terminal reconciliation horizon, whether resource terms are scalar or
constrained costs, and the treatment of metrics unavailable at deployment.
