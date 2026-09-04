# Phase 11C zstd level sweep

`ae_phase11c_zstd_level_sweep.py` is a bounded, non-production comparison of
zstd levels 1, 3, and 5 over the real 72-profile inner transport catalog. It
reuses the immutable Phase-7 128-frame fit sample and binds completed Phase-11B
evidence before CUDA use. It measures host compression/decompression only;
zstd remains lossless and this phase neither scores perception nor changes the
locked production level.
