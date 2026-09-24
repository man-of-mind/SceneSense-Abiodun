# Run-4 train-only fit-scene provider

Status: **implemented, hash-bound mechanics; caller must supply verified
evidence objects**.

`fit_scene_provider.py` is the only component that chooses a visual scene for
Run-4 training. It has no filesystem discovery and performs no import-time
evidence loading. A runner must first load the existing frozen quality surface,
intersect its 476 action-independent quality-valid `fit` scenes with the
registered temporal-block partition, and pass the resulting selections in a
verified envelope.

## Frozen population boundary

The registered partition contains:

- 391 training scenes;
- 85 development fit-validation scenes.

The provider independently re-derives the temporal-block split for every
selection and accepts only the 391 training identities. It also pins the
canonical identity/covariate/design-weight inventory:

`1d0d102fafb8797fd4b69e8524dd301229ddbc6f9bc76072048545c1d022ebbd`

Therefore a missing, duplicated, substituted or fit-validation scene changes
the digest and fails construction. The selection-manifest, quality-database,
surface, adapter, and held-provider digests must close exactly. The held
provider's complete 391-scene identity snapshot is compared against the same
inventory; matching a count alone is insufficient.

## Sampling and actor visibility

Scenes are selected proportional to their recorded inverse-inclusion
`sampling_weight`, matching the existing Run-3 design. The provider owns two
explicit local `random.Random` streams:

- one for reward-request scene selection;
- one for marginal held-frame payload selection.

Neither stream touches Python's global RNG. `state_dict()` and
`load_state_dict()` preserve both streams, their counters, and the active scene
so checkpoint continuation is exact.

The selected scene record remains environment-private. The only scene fields
returned for the 21-D policy state are:

```text
(camera_si, radar_p40)
```

Numeric zero is a valid measured SI/P40 value. Missing values are rejected and
are never converted to zero.

## Exact downstream joins

For the reward-request tensor, the provider passes the exact selected
`FitSceneSelectionV1` to `Run4QualityPayloadAdapterV1.reward_tensor()`. The
adapter re-queries that sample and exact executed `(mode_id, q_e4)` and returns
the same-frame `Q_perc` and payload.

For held tensors, the existing non-temporal marginal provider makes an
independent weighted draw from the same 391-scene population. Every returned
sample, episode/frame identity, inclusion probability, sampling weight, and
scene digest is checked back against the registered inventory. Held tensors
never request another reward.

This component does not create radio/queue outcomes, latency, temporal scene
adjacency, rewards, replay transitions, or training eligibility. Those remain
separate Run-4 gates.
