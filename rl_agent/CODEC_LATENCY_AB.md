# Codec latency A/B — zstd vs zlib (ideal 8 MB loopback, 100% delivery both)

**Accuracy codec-invariant** (lossless). **Payload ~±5% codec-dependent** (compression ratio). zlib now measured on
all 36 profiles (`sweeps_loopback_ideal_zlib_full`); zstd on the 8 pure quant/ROI/AE points (`sweeps_loopback_ideal`);
overlap shown. `transport` incl. reassembly+**decompress**; `front` incl. **compress**.

| profile | payload KB | front z→z | transport zstd→zlib | ×penalty | RTT zstd→zlib |
|---|--:|--:|--:|--:|--:|
| uint8 roi0.0 ae64 | 259 | 24.9→29.3 | 2.0→8.5 | 4.2× | 13.0→18.9 |
| uint8 roi0.0 ae128 | 332 | 25.5→30.8 | 2.2→10.4 | 4.7× | 11.5→20.5 |
| uint4 roi0.0 | 386 | 27.6→35.2 | 4.6→14.6 | 3.2× | 13.9→24.2 |
| uint8 roi0.5 | 579 | 30.5→39.4 | 6.2→20.0 | 3.2× | 13.8→28.3 |
| uint8 roi0.3 | 745 | 31.0→43.6 | 6.8→23.8 | 3.5× | 14.6→31.9 |
| uint6 roi0.0 | 760 | 31.2→44.8 | 6.3→24.8 | 3.9× | 17.3→35.7 |
| uint8 roi0.0 | 1018 | 29.8→46.0 | 7.4→30.7 | 4.1× | 15.0→39.8 |

## Headline — no-AE u8 (~1 MB) capture→result floor
- **zstd:** 45 ms   **zlib (deployed):** 86 ms   → zlib→zstd cuts the floor ~48%, same accuracy.

## Takeaways
- Penalty grows with payload: ~1–1.5× at small AE payloads, ~4× at the ~1 MB no-AE payload.
- It is **compute** (zlib (de)compress), channel-independent — inflates OAI anchors identically.
- zstd is a **~free** latency lever (lossless, payload ~±5%). Deployed = zlib → use `PERMODEL_KNOB_MATRIX_ZLIB.md` (all 36 measured).
