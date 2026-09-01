# Protocol v2 amendment 001: optional vehicle instance diagnostic

Registered on 2026-09-01 before traffic collection or model inference.

The first v2 controlled run produced 5,156 finite pixels in the vehicle's actor-only depth support but no component in the colocated isolated instance camera. It stopped before rendering the pedestrian or any clear/partial/full cases.

The instance camera is not part of the renderer z-buffer visibility equation and was already proven unusable for pedestrians. Requiring its vehicle output as a blocking cross-check incorrectly made the new method depend on the broken sensor it was designed to avoid.

Amendment 001 makes that one vehicle comparison optional and nonblocking. If no instance component is available, the implementation must preserve the raw evidence, record `instance_diagnostic_unavailable`, and continue the registered depth qualification.

Nothing scientific changes: the actor-only depth support, scene-to-actor per-pixel depth comparison, fixed `0.02 m` tolerances, clear/partial/full gates, five visibility thresholds, model locks, episodes and reporting plan are unchanged.
