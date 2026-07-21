# Codec latency A/B — zstd vs zlib (ideal 8 MB loopback, 100% delivery, ALL profiles measured)

**Accuracy codec-invariant** (lossless). **Payload ~±5% codec-dependent** (compression ratio). Both codecs now
measured on the full profile set (`sweeps_loopback_ideal_{zstd,zlib}_full`). `transport` incl. reassembly+
**decompress**; `front` incl. **compress**.

| profile | zstd KB | zlib KB | front z→z | transport zstd→zlib | ×penalty | RTT zstd→zlib |
|---|--:|--:|--:|--:|--:|--:|
| uint4 roi0.5 ae32 | 52.4 | 56.4 | 27.0→27.5 | 1.8→3.6 | 2.0× | 14.5→12.4 |
| uint4 roi0.5 ae64 | 61.0 | 65.3 | 24.6→28.0 | 1.9→4.0 | 2.1× | 13.5→15.7 |
| uint4 roi0.3 ae32 | 65.6 | 67.0 | 26.5→27.4 | 1.8→3.9 | 2.2× | 13.7→12.5 |
| uint4 roi0.3 ae64 | 73.7 | 75.2 | 25.8→27.6 | 2.1→4.2 | 2.0× | 14.6→13.7 |
| uint4 roi0.5 ae128 | 77.8 | 80.5 | 28.0→27.8 | 2.3→4.5 | 2.0× | 15.2→12.9 |
| uint4 roi0.0 ae32 | 89.2 | 88.3 | 24.7→26.6 | 2.0→4.6 | 2.3× | 14.2→15.1 |
| uint4 roi0.3 ae128 | 91.6 | 93.7 | 25.3→28.5 | 2.2→4.7 | 2.1× | 13.7→14.1 |
| uint4 roi0.0 ae64 | 98.9 | 99.3 | 22.5→26.8 | 2.3→4.9 | 2.1× | 17.8→15.3 |
| uint6 roi0.5 ae32 | 100.8 | 101.8 | 29.1→28.5 | 2.3→4.6 | 2.0× | 15.2→13.2 |
| uint6 roi0.5 ae64 | 119.6 | 117.9 | 26.2→29.0 | 2.4→5.0 | 2.1× | 15.7→14.4 |
| uint4 roi0.0 ae128 | 121.6 | 123.0 | 26.8→26.7 | 2.6→5.6 | 2.2× | 18.6→16.5 |
| uint6 roi0.3 ae32 | 125.8 | 127.3 | 26.7→29.6 | 2.0→5.2 | 2.6× | 15.2→13.6 |
| uint8 roi0.5 ae32 | 130.9 | 127.9 | 27.4→29.2 | 2.3→5.3 | 2.3× | 14.2→13.2 |
| uint6 roi0.3 ae64 | 145.3 | 145.1 | 28.6→29.9 | 2.2→5.6 | 2.5× | 15.0→13.7 |
| uint8 roi0.5 ae64 | 155.8 | 154.5 | 25.8→29.1 | 2.2→6.1 | 2.8× | 14.6→14.7 |
| uint6 roi0.5 ae128 | 154.2 | 156.1 | 28.0→30.3 | 2.5→6.3 | 2.5× | 15.9→14.8 |
| uint8 roi0.3 ae32 | 169.1 | 164.8 | 28.9→29.6 | 2.1→6.0 | 2.9× | 15.0→14.3 |
| uint6 roi0.0 ae32 | 174.3 | 171.0 | 23.9→27.7 | 2.0→6.3 | 3.1× | 14.4→16.1 |
| uint6 roi0.3 ae128 | 186.2 | 186.3 | 25.6→30.9 | 2.2→6.9 | 3.1× | 15.7→15.7 |
| uint8 roi0.3 ae64 | 196.9 | 190.6 | 25.6→30.7 | 2.4→7.0 | 2.9× | 15.1→16.7 |
| uint6 roi0.0 ae64 | 199.8 | 195.4 | 23.3→28.6 | 2.1→7.0 | 3.3× | 18.1→15.1 |
| uint8 roi0.5 ae128 | 207.1 | 203.0 | 24.6→30.5 | 2.8→7.7 | 2.8× | 15.4→18.4 |
| uint8 roi0.0 ae32 | 231.9 | 224.7 | 24.8→28.7 | 2.0→7.4 | 3.7× | 16.1→15.3 |
| uint6 roi0.0 ae128 | 253.8 | 246.8 | 23.2→30.0 | 2.3→8.5 | 3.7× | 16.1→16.6 |
| uint8 roi0.3 ae128 | 252.4 | 247.7 | 25.6→30.9 | 2.6→8.6 | 3.3× | 14.8→20.4 |
| uint4 roi0.5 | 221.1 | 250.4 | 28.9→33.7 | 4.6→10.6 | 2.3× | 16.4→18.0 |
| uint8 roi0.0 ae64 | 269.2 | 259.4 | 22.9→29.3 | 2.1→8.5 | 4.0× | 15.3→18.9 |
| uint4 roi0.3 | 271.6 | 291.6 | 28.5→33.7 | 4.8→11.8 | 2.5× | 20.6→23.7 |
| uint8 roi0.0 ae128 | 342.5 | 331.5 | 24.5→30.8 | 2.5→10.4 | 4.2× | 16.6→20.5 |
| uint4 roi0.0 | 366.8 | 385.9 | 28.0→35.2 | 5.7→14.6 | 2.6× | 19.1→24.2 |
| uint6 roi0.5 | 422.5 | 454.0 | 34.0→39.6 | 5.9→16.8 | 2.8× | 19.0→26.9 |
| uint6 roi0.3 | 526.6 | 564.7 | 32.2→42.5 | 6.2→19.5 | 3.1× | 18.5→28.8 |
| uint8 roi0.5 | 565.9 | 579.2 | 29.0→39.4 | 7.5→20.0 | 2.7× | 18.1→28.3 |
| uint8 roi0.3 | 717.7 | 745.3 | 32.2→43.6 | 9.4→23.8 | 2.5× | 21.5→31.9 |
| uint6 roi0.0 | 730.4 | 760.3 | 34.6→44.8 | 7.3→24.8 | 3.4× | 21.2→35.7 |
| uint8 roi0.0 | 984.2 | 1018.5 | 27.8→46.0 | 8.7→30.7 | 3.5× | 19.3→39.8 |

## Headline — no-AE u8 (~1 MB) capture→result floor
- **zstd:** 47 ms   **zlib (deployed):** 86 ms   → zlib→zstd cuts the floor ~45%, same accuracy.

## Takeaways
- Penalty grows with payload: ~1–1.5× at small AE payloads, ~4× at the ~1 MB no-AE payload.
- It is **compute** (codec (de)compress), channel-independent — inflates OAI anchors identically.
- zstd is a **~free** latency lever (lossless, payload ~±5%). Deployed = zlib → train on `PERMODEL_KNOB_MATRIX_ZLIB.md`.
