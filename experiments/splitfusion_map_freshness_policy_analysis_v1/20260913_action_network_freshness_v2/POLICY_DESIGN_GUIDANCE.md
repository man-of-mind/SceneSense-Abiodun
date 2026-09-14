# Policy-design guidance from the 288-cell freshness analysis

## What the controller is actually optimizing

`installed / sent` is a throughput statistic. It does not say whether
the installed information was still fresh. The controller-facing
outcomes in this analysis are:

- **fresh-map fraction:** fraction of route time for which the newest
  installed contribution has capture-to-current-time AoI within the
  selected 150, 200 or 250 ms budget;
- **timely useful-update yield:** sent frames that install within the
  budget and advance the map to a newer capture;
- **quality-weighted freshness:** frozen validation F1 for the selected
  action multiplied by its network-conditioned fresh-map fraction.

The first metric is the primary map-service outcome. The second
explains how efficiently an action creates those fresh intervals. The
third exposes the quality/freshness trade rather than rewarding a tiny
but inaccurate payload solely for arriving quickly.

## Queue policy decision

FIFO reaches a worst observed counterfactual wait of 10.29 s and a backlog of 103 frames. Strict latest-only caps
the pending compute slot at one frame. When active work finishes, the
newest pending frame runs and all older pending frames receive an
explicit `SUPERSEDED_PENDING_COMPUTE` terminal.

This lowers median cell map AoI from 371.6
to 308.0 ms while intentionally reducing
the number of eventual installations. That reduction is not failure:
old work is being exchanged for newer map state. There is no 25 ms
expiry and no reason to discard the only pending frame merely because
it waited.

## Best observed split regions

The table reports the action with the largest *raw* fresh-map fraction
for each network and budget. It must not be read as the best final
policy action because it does not yet include person/vehicle quality.

| Budget | Network | Action | Payload | Fresh-map time |
|---:|---|---:|---:|---:|
| 150 ms | Favorable | 71 | 6.3 KiB | 8.9% |
| 150 ms | Mid-variable | 65 | 10.5 KiB | 6.9% |
| 150 ms | Fade/recovery | 65 | 10.5 KiB | 6.8% |
| 150 ms | Adverse | 65 | 10.5 KiB | 1.4% |
| 200 ms | Favorable | 71 | 6.3 KiB | 37.4% |
| 200 ms | Mid-variable | 65 | 10.5 KiB | 33.3% |
| 200 ms | Fade/recovery | 65 | 10.5 KiB | 33.9% |
| 200 ms | Adverse | 65 | 10.5 KiB | 10.1% |
| 250 ms | Favorable | 71 | 6.3 KiB | 68.9% |
| 250 ms | Mid-variable | 65 | 10.5 KiB | 66.2% |
| 250 ms | Fade/recovery | 65 | 10.5 KiB | 66.4% |
| 250 ms | Adverse | 65 | 10.5 KiB | 30.5% |

No split action keeps the map continuously inside any tested
budget. The 150 ms surface is an aggressive diagnostic; even the
best favorable result is below 10%. At 200 ms, the best favorable
result remains below 40%. At 250 ms, small-payload actions become
useful in favorable/mid/fade conditions, but the best adverse result
is only about 30%. This creates a real decision problem rather than
one globally dominant split action.

## Causal controller inputs

A recurrent policy should receive only values available before the
next choice: lagged SNR/MCS or capacity estimate, current installed
map AoI, time since last useful install, edge busy/pending flags,
recent service-time EWMA, previous action, and its terminal outcome.
The LSTM hidden state may learn channel trend and predict the next
condition implicitly; future profile labels and future ACKs remain
forbidden.

A compact starting reward is:

```math
r_t = w_p Q_p(a_t)F_B(\mathrm{AoI}_{t+1})
    + w_v Q_v(a_t)F_B(\mathrm{AoI}_{t+1})
    - \lambda_b \frac{b(a_t)}{B_{\max}}
    - \lambda_c C(a_t) - \lambda_s I[a_t \ne a_{t-1}]
```

where `Q_p` and `Q_v` are frozen person/vehicle quality, `F_B` is a
hard or soft freshness utility at budget `B`, `b(a)` is transmitted
feature bytes, and `C(a)` is measured compute cost. A superseded
frame earns no installation utility, but its already-spent bytes and
compute remain charged. It should not receive an extra arbitrary
discard penalty, because replacement is the scheduler's correct
map-first behavior.

## LOCAL decision boundary

The split results trigger—not satisfy—the LOCAL baseline step. The
older LR-ASPP local measurements are not interchangeable with the
current frozen FCOS service. Before `LOCAL_INFER` enters the action
set, measure the current FCOS full-local compute distribution and
sustainable rate, compact object-result bytes versus object count,
identical-input quality, and compact-result delivery/map-install
latency over all four OAI profiles. Record local completion, edge
installation and ACK as distinct times. Until then, train and report
the present surface as **split-only** and do not manufacture a local
transition from old measurements.

## What not to conclude

- Network profiles do not change intrinsic model quality; they change
  whether and when that quality reaches the map.
- A high install ratio does not establish freshness.
- Person localization degradation with age is not identified here;
  the retained campaign has no matched person rows for that check.
- The counterfactual predicts the qualified final implementation; it
  is not a second 288-cell live campaign.
