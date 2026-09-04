# Phase 11B — shared UINT6/UINT4 GPU qualification

Terminal: `SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED`

One registered fit-training frame was run through the public low-bit encode/receive paths. This is a structural GPU qualification only: no validation, test, accuracy, scoring, calibration, NMS, training, tuning or CARLA activity occurred.

| family | quantizer | q | keep | pre-zstd analytical/measured B | zstd B (diagnostic) | ranker | decomp | tail finite |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| noAE | UINT6 | 0.00 | 21,504 | 4,130,868/4,130,868 | 2,777,188 | 0 | 1 | yes |
| noAE | UINT6 | 0.50 | 10,752 | 2,069,172/2,069,172 | 1,397,349 | 1 | 1 | yes |
| noAE | UINT4 | 0.00 | 21,504 | 2,754,612/2,754,612 | 1,525,386 | 0 | 1 | yes |
| noAE | UINT4 | 0.50 | 10,752 | 1,381,044/1,381,044 | 800,510 | 1 | 1 | yes |
| AE128 | UINT6 | 0.00 | 21,504 | 2,065,460/2,065,460 | 1,893,307 | 0 | 1 | yes |
| AE128 | UINT6 | 0.50 | 10,752 | 1,035,956/1,035,956 | 969,682 | 1 | 1 | yes |
| AE128 | UINT4 | 0.00 | 21,504 | 1,377,332/1,377,332 | 940,366 | 0 | 1 | yes |
| AE128 | UINT4 | 0.50 | 10,752 | 691,892/691,892 | 504,573 | 1 | 1 | yes |
| AE64 | UINT6 | 0.00 | 21,504 | 1,032,756/1,032,756 | 967,647 | 0 | 1 | yes |
| AE64 | UINT6 | 0.50 | 10,752 | 519,348/519,348 | 491,590 | 1 | 1 | yes |
| AE64 | UINT4 | 0.00 | 21,504 | 688,692/688,692 | 498,997 | 0 | 1 | yes |
| AE64 | UINT4 | 0.50 | 10,752 | 347,316/347,316 | 263,302 | 1 | 1 | yes |
| AE32 | UINT6 | 0.00 | 21,504 | 516,404/516,404 | 477,037 | 0 | 1 | yes |
| AE32 | UINT6 | 0.50 | 10,752 | 261,044/261,044 | 244,204 | 1 | 1 | yes |
| AE32 | UINT4 | 0.00 | 21,504 | 344,372/344,372 | 242,454 | 0 | 1 | yes |
| AE32 | UINT4 | 0.50 | 10,752 | 175,028/175,028 | 129,143 | 1 | 1 | yes |

## Integrity

- settings: 16/16
- ranker invocations: 8 (required 8)
- zstd decompressions: 16 (required 16)
- q=0 ranker bypass: True
- q=0.50 masks/indices identical: True
- all frozen states unchanged: True
- peak allocated/reserved VRAM: 445,905,920/580,911,104 bytes
- wall time: 3.722 s
