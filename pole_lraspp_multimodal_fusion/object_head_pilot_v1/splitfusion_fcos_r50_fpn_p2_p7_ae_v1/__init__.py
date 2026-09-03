"""Channel-compressing feature autoencoder over the frozen SplitFusion-FCOS C2 split.

Phase 9A is implementation only: one shared `SplitFeatureAE` parameterized by
bottleneck B in {128, 64, 32}, the latent mask/composition helper, the
task-aware reconstruction loss interface and the AE-latent UINT8 + mandatory
zstd wire. Nothing here loads a checkpoint, touches CUDA, reads a dataset or
cache, trains, infers, validates, evaluates or launches CARLA.

The frozen hybrid-q package is imported, never modified: the ranker, exact q
semantics, spatial bitmask helpers, fail-closed guards, the validated noAE
UINT8 codec and the level-1 zstd compressor are all reused as-is.
"""
