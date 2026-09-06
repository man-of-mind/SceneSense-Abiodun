# SplitFusion 16-cell live-pilot implementation review

## 1. Scientific/runtime review — PASS

- The new versioned contract retains exactly actions 0, 71, 46, and 20 in the registered cell order, crossed with the four frozen profile traces.
- It hash-binds the locked 72-action catalog, Phase-12B/13A artifacts, corrected Phase-14B evidence, selected 100-MHz mapping, and current 100-MHz launcher.
- The UE uses the preloaded Phase-13 SFD1-v2 front/ranker/AE path; q=0 follows the public ranker-bypass route.  A separate `oai-perception-rx` service preloads the frozen decoder/tail/p025 path and validates packet identity plus frame context.
- Both legs use the production `!IHH` fragmentation header. Inner payloads retain mandatory zstd level 1; the only zlib use is the existing, separately documented spatial-map ingest protocol.
- The historical/default106 and LR-ASPP/M-prime runtime bindings are not used by the new contract or live start path.

## 2. Lifecycle/recovery review — PASS

- Each cell uses a create-only attempt, fresh qualified three-process gNB/UE lifecycle, clean `noise_power_dB=-50` proof, isolated edge readiness leaf, fresh CARLA process, and a fresh target-profile generator started by the first accepted capture.
- The target sequence has absolute 100-ms deadlines and its existing no-wrap/no-hold/no-reseed behavior.  The Phase-14B probe is explicitly forbidden.
- Radio, edge container, map process, UE sockets, CARLA process group, temporary diagnostics, and the edge readiness leaf are stopped/removed before the next cell.  Only compact registered per-cell records are retained.
- Resume skips only a revalidated, hash-bound passed cell; failed/interrupted attempts remain immutable and are never reused.
