# Run-4 held-frame payload model

Status: **bounded structural implementation; not temporal evidence**.

`held_payload.py` supplies a modeled byte load for virtual held transmissions
inside one semi-Markov decision cycle, while the originating ticket remains
outstanding. It does not create another policy step or reward; its sole output
is the payload quantity that a caller may apply to its queue. The caller must
explicitly choose this marginal fallback. Every result is labelled:

`MODELED_EMPIRICAL_MARGINAL_HELD_PAYLOAD_NOT_TEMPORAL`

It does not claim that a selected Route-B scene is the missing next frame. It
does not synthesize temporal imagery, RGB/radar data, quality, latency, queue
state, feedback, policy decisions, or rewards. It performs no filesystem
access and does not load evidence at import or at evaluation time. Actual
10-Hz frame adjacency and cadence are deployment/live-validation questions;
this non-temporal marginal model does not test them.

## Input and binding contract

The caller supplies complete train-only curves for every scene and all 12
discrete modes. Each curve must contain the exact registered q grid:

`0, 1500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9400, 9800`.

Every endpoint carries its source-row SHA-256. Every scene carries its source
identity/digest, selection-manifest digest, database digest, inclusion
probability, and sampling weight. The provider rejects held, validation, test,
or generically named `fit` sources: the adapter constructing these records must
have already intersected the frozen fit evidence with the registered training
partition and relabelled it explicitly `train`.

The caller must pin the canonical inventory SHA-256. Mixed bindings, missing
q endpoints or modes, duplicate/conflicting identifiers, nonpositive payloads,
and non-monotone curves fail closed.

## Selection and exact action evaluation

The frozen scene selection was stratified rather than uniform. Its stored
`sampling_weight` is inverse inclusion probability, and the existing Run-3
environment selects contexts proportional to that weight. This provider keeps
the same rule. Uniform sampling would over-represent heavily sampled strata.

The provider accepts a caller-owned local RNG and an explicit
`(session_id, decision_seq, held_ordinal, rng_stream_id)` counter. It never uses
module/global RNG state. Reusing one physical counter for another action or RNG
stream is rejected; an identical retry is idempotent and consumes no second
draw.

After selecting a scene independently of the action, the provider evaluates
the exact executed `(mode_id, q_e4)`. A measured q node is returned directly.
For an interior q, payload is linearly interpolated only between the adjacent
q nodes of that same scene and mode, matching the existing empirical surface:

```text
alpha = (q - q_lower) / (q_upper - q_lower)
payload = payload_lower + alpha * (payload_upper - payload_lower)
```

There is no nearest-anchor substitution, cross-scene interpolation, or
extrapolation. Both endpoint row digests remain in the output.

## Read-only payload-variation audit

The completed frozen grid was independently read through SQLite
`mode=ro&immutable=1`. For each of the 132 `(mode_id, q_e4)` cells, the
coefficient of variation of `total_transmitted_bytes` across scenes was
computed with the stored inverse-inclusion-probability weights:

```text
weighted mean = sum(w_i * x_i) / sum(w_i)
weighted CV   = sqrt(sum(w_i * (x_i - mean)^2) / sum(w_i)) / mean
```

| source split | scenes/cell | fixed-action CV median | CV P95 | CV maximum |
|---|---:|---:|---:|---:|
| fit | 512 | 0.0087390 | 0.0472839 | 0.0773897 |
| held_scene | 256 | 0.00774784 | 0.0345331 | 0.0586556 |

Source bindings:

- quality database SHA-256:
  `8b2c0754f135873cc339dc62cc7e0e09e02e3a85426d0027c2cd38b6a2a68c88`
- selection-manifest file SHA-256:
  `a25b93d35a5bb70e062d275daf8767ce73a5ab7ba34079ab6a468bda5ffe4c4d`

These small fixed-action scene CVs support a bounded marginal payload
sensitivity model. They do **not** establish temporal autocorrelation, genuine
10-Hz held frames, action-conditioned queue transitions, or deployment
validity. `held_scene` was inspected only for this read-only diagnostic and is
categorically unavailable to the provider.
