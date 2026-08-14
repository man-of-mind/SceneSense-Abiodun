# Expanded SPLIT+SKIP action gate v2 — validity correction frozen before rerun

Status: **pre-registered before the v2 outcome**. This document inherits every source, action, group, seed,
metric, threshold, and authorization boundary from v1. The v1 execution never wrote a manifest or completion
sentinel and is invalid, not an outcome.

## Why v1 is invalid

The v1 implementation selected oracle actions using *all* hidden ground-truth objects, while the deployable
environment correctly allowed transmitted frames to update only tracker-observed object keys. The registered
primary metric was matched-truth reward over those deployable keys. An oracle optimizing impossible hidden
updates is not an upper bound on that metric: it can spend capacity on objects its executed frame cannot install
and become arbitrarily worse than greedy. The first partial run exposed exactly this contradiction through
sentinel-scale negative oracle rewards. It then failed during manifest finalization and is marked `FAILED` and
`INVALID_NOT_ACCEPTED`.

This is a direction-independent validity repair. It does not change the action set, reward, decision threshold,
trajectory, seed, or result in response to whether the partial value favored RL.

## Single correction

The v2 oracle sees hidden true capacity and true kinematics **only for the currently deployable tracker keys**,
using the existing `matched_truth_observation()` contract. Those are the same keys that a selected SPLIT frame
may update and the same keys in the primary matched-truth reward. It remains non-deployable and optimistic, but
it now optimizes the metric it is claimed to upper-bound.

Everything else is inherited byte-for-byte from
`configs/expanded_action_gate_v1.yaml`, whose SHA-256 is pinned in the v2 config. In particular:

- all 35 measured SPLIT profile/FPS choices plus SKIP; LOCAL excluded;
- held-out groups and three paired channel seeds;
- reward v5, hard C1, safety/degradation semantics, exact joint multiple-choice selection;
- queue-free necessary frontier and the 1% aggregate true-capacity miss validity ceiling;
- the five registered headroom checks and the same stop/build-next verdicts;
- no OAI, CARLA, max-weight, MPC, or RL.

The v2 oracle is named a **matched-support joint true-state one-step upper bound**. It is not future-perfect and
it is not a shared-queue oracle. A positive result can authorize only the queue-aware non-learning max-weight
rung; a negative result closes only this expanded queue-free replay contract.

