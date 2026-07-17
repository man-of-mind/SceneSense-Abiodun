# Post-hoc natural-scene FOV localization split

Vehicle-only analysis at score >= 0.20, match gate <= 2 m, objects <= 40 m.
The point is to test FOV position on natural moving-ego data, not the artificial one-car Experiment-3 scene.

## Dataset counts

| source | opportunities | matches | availability |
|---|---:|---:|---:|
| live moving control | 1024 | 316 | 0.309 |
| live speed sweep | 7984 | 962 | 0.120 |
| offline eval | 1066 | 948 | 0.889 |

## Findings

- The natural offline 200k split gives a meaningful FOV signal, but not a simple monotonic `center is always best` curve.
- Near vehicles (0-15 m) are easy across the FOV; edge localization is not worse there. Offline center: n=42, avail=1.000, median=0.298 m, score=0.877; edge: n=193, avail=0.845, median=0.208 m, score=0.713.
- For 15-40 m vehicles, edge bins lose availability and usually have worse matched localization. Offline 15-25 m center: n=69, avail=1.000, median=0.986 m, score=0.458; edge: n=28, avail=0.786, median=1.251 m, score=0.495. Offline 25-40 m center: n=102, avail=0.922, median=0.882 m, score=0.483; edge: n=231, avail=0.792, median=1.174 m, score=0.406.
- The 200k live moving-control run is smaller/noisier but points the same way for availability: 25-40 m center: n=308, avail=0.114, median=1.394 m, score=0.366; edge: n=35, avail=0.000, median=n/a m, score=n/a.
- The live speed-sweep logs are included only as a secondary check because they are 5k-PPS runs; the offline split and the 200k moving-control run are the best references for the current 200k model.
- Practical takeaway for RL/FOV prioritization: use range-aware edge risk, not a blanket center prior. At close range, edge objects can be localized well; at medium/far range, edge objects are more likely to be missed and sometimes localize worse when matched.

## Absolute FOV bins

### offline eval — 0-15m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 42 | 42 | 1.000 | 0.298 | 0.398 | 0.803 | 0.877 |
| inner 10-25 | 37 | 35 | 0.946 | 0.401 | 0.576 | 1.033 | 0.711 |
| outer 25-40 | 91 | 89 | 0.978 | 0.267 | 0.378 | 0.719 | 0.681 |
| edge 40-60 | 193 | 163 | 0.845 | 0.208 | 0.366 | 0.686 | 0.713 |

### offline eval — 15-25m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 69 | 69 | 1.000 | 0.986 | 1.055 | 1.876 | 0.458 |
| inner 10-25 | 43 | 42 | 0.977 | 0.828 | 1.111 | 2.227 | 0.499 |
| outer 25-40 | 22 | 20 | 0.909 | 0.960 | 1.066 | 1.615 | 0.425 |
| edge 40-60 | 28 | 22 | 0.786 | 1.251 | 1.383 | 2.330 | 0.495 |

### offline eval — 25-40m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 102 | 94 | 0.922 | 0.882 | 1.160 | 2.257 | 0.483 |
| inner 10-25 | 110 | 104 | 0.945 | 0.506 | 0.732 | 1.607 | 0.487 |
| outer 25-40 | 98 | 85 | 0.867 | 0.995 | 1.150 | 2.253 | 0.435 |
| edge 40-60 | 231 | 183 | 0.792 | 1.174 | 1.406 | 2.782 | 0.406 |

### live speed sweep — 0-15m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 409 | 76 | 0.186 | 1.124 | 1.162 | 1.792 | 0.411 |
| inner 10-25 | 514 | 151 | 0.294 | 1.198 | 1.191 | 1.874 | 0.600 |
| outer 25-40 | 582 | 159 | 0.273 | 1.046 | 1.084 | 1.783 | 0.554 |
| edge 40-60 | 237 | 38 | 0.160 | 1.184 | 1.176 | 1.847 | 0.432 |

### live speed sweep — 15-25m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 990 | 169 | 0.171 | 1.430 | 1.300 | 1.845 | 0.383 |
| inner 10-25 | 927 | 125 | 0.135 | 1.228 | 1.159 | 1.834 | 0.451 |
| outer 25-40 | 258 | 29 | 0.112 | 1.357 | 1.226 | 1.784 | 0.439 |
| edge 40-60 | 136 | 4 | 0.029 | 0.430 | 0.767 | 1.533 | 0.298 |

### live speed sweep — 25-40m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 2101 | 137 | 0.065 | 1.517 | 1.396 | 1.893 | 0.388 |
| inner 10-25 | 918 | 48 | 0.052 | 1.418 | 1.267 | 1.935 | 0.348 |
| outer 25-40 | 431 | 15 | 0.035 | 1.504 | 1.544 | 1.943 | 0.335 |
| edge 40-60 | 481 | 11 | 0.023 | 1.549 | 1.322 | 1.803 | 0.341 |

### live moving control — 0-15m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 28 | 10 | 0.357 | 0.980 | 1.040 | 1.778 | 0.588 |
| inner 10-25 | 110 | 85 | 0.773 | 0.812 | 0.838 | 1.448 | 0.687 |
| outer 25-40 | 88 | 48 | 0.545 | 0.683 | 0.689 | 0.989 | 0.703 |
| edge 40-60 | 72 | 16 | 0.222 | 0.685 | 0.706 | 0.891 | 0.429 |

### live moving control — 15-25m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 155 | 70 | 0.452 | 1.341 | 1.291 | 1.794 | 0.404 |
| inner 10-25 | 78 | 37 | 0.474 | 1.302 | 1.279 | 1.694 | 0.490 |
| outer 25-40 | 32 | 3 | 0.094 | 1.415 | 1.321 | 1.611 | 0.526 |
| edge 40-60 | 7 | 0 | 0.000 | n/a | n/a | n/a | n/a |

### live moving control — 25-40m

| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| center 0-10 | 308 | 35 | 0.114 | 1.394 | 1.341 | 1.837 | 0.366 |
| inner 10-25 | 69 | 12 | 0.174 | 1.145 | 1.186 | 1.873 | 0.328 |
| outer 25-40 | 42 | 0 | 0.000 | n/a | n/a | n/a | n/a |
| edge 40-60 | 35 | 0 | 0.000 | n/a | n/a | n/a | n/a |

## Files

- `offline_live_vehicle_abs_fov_summary.csv` — absolute-angle bins by source and distance.
- `offline_live_vehicle_signed_fov_summary.csv` — signed left/right bins by source and distance.
- `*_fov_error_by_distance.{png,pdf}` — median localization error vs absolute FOV position.
- `*_fov_availability_by_distance.{png,pdf}` — match availability vs absolute FOV position.
