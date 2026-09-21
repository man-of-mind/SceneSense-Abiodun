# Hybrid-SAC empirical mechanics smoke

Status: **PASS**

This bounded run verifies that the registered empirical contextual Hybrid-SAC
training path executes deterministically. It is a mechanics qualification over
the complete D1 fit sampler; it is not validation, convergence, generalization,
or deployment evidence.

## Registered schedule

- Seeds: 17, 29, 43
- Warm-up transitions per seed: 1,024
- Updates per seed: 500
- New transitions per update: 4
- Total transitions per seed: 3,024
- Batch size: 256
- Replay capacity: 8,192
- Device and threading: CPU, one Torch thread
- Config SHA-256: `d0f991545bf7d4bcf4cf513bd8882bba5c952274ef3a0a8f9d9b53e680645431`

## Results

| Seed | Updates | Transitions | Reward range | Actor delta | Critic-1 delta | Critic-2 delta | Acceptance |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 17 | 500 | 3,024 | -0.8626 to 0.7955 | 3.8767 | 3.7298 | 3.6793 | PASS |
| 29 | 500 | 3,024 | -0.8185 to 0.7693 | 3.8175 | 3.1767 | 3.2416 | PASS |
| 43 | 500 | 3,024 | -0.8256 to 0.7746 | 3.3651 | 2.8254 | 2.7257 | PASS |

All three seeds satisfied the following gates:

- all 500 updates completed with finite metrics;
- all 12 discrete modes were exercised during warm-up;
- every continuous action remained inside its registered mode-specific support;
- replay retained all 3,024 transitions with zero eviction;
- terminal critic targets were bit-identical to reward (`y = r`);
- actor and both online critics changed from initialization;
- module-global Python and Torch RNG states were unchanged;
- the runner did not initialize CUDA;
- checkpoint/resume equivalence passed in the focused integration test.

The machine-readable result is `report.json` with SHA-256
`f4ceb8dc180447f9c5fc0dd88ba3d85c35737e3728e288320415bb9351167cad`.

## Claim boundary

This run used the whole D1 fit distribution and therefore cannot measure held
performance or learning convergence. The next preliminary run uses the frozen
train-only partition; its checkpoints will be evaluated separately on the
reward-held fit-validation panel.
