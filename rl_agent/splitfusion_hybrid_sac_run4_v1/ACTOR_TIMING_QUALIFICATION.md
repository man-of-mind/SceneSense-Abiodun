# Run-4 actor timing qualification

The 170-ms agent-path clock starts when the guarded state and seven-channel
input are ready, immediately before the batch-1 actor is invoked. Therefore
actor inference, stochastic mode selection, exact q quantization and action
catalog resolution are part of the latency budget.

The direct action-50 quality-feedback probe used a fixed action and does not
measure this cost. `actor_timing_qualification.py` provides a create-only,
CPU-only engineering benchmark of the exact Run-4 selection path. Modeled
training must add `max(1 ms, measured P99)` to each fixed-action latency draw.
It must never report the augmented total as a direct live measurement.

The timed path begins with the guarded 21-feature tuple and includes its
float32 batch-1 tensor construction, matching `_actor_action()`.

A later live pilot should stamp action-open immediately before actor inference
and validate the same boundary end to end. Until then this qualification is
appropriate for the bounded offline modeled-composite smoke only.
