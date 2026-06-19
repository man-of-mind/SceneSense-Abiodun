# Moving vs Parked Fusion Transfer Summary

| Model | Test domain | mIoU | Vehicle IoU | Person IoU | XY MAE (m) | Obj F1 |
|---|---:|---:|---:|---:|---:|---:|
| Parked A model | View A | 0.788 | 0.918 | 0.472 | 1.157 | 0.423 |
| Parked B model | View B | 0.806 | 0.938 | 0.506 | 1.287 | 0.368 |
| Parked A+B model | View A | 0.787 | 0.913 | 0.472 | 1.184 | 0.424 |
| Parked A+B model | View B | 0.797 | 0.930 | 0.490 | 1.324 | 0.401 |
| Parked A+B model | A+B | 0.793 | 0.924 | 0.482 | 1.257 | 0.411 |
| Moving model | Moving | 0.825 | 0.874 | 0.630 | 1.430 | 0.287 |
| Moving model | View A | 0.534 | 0.449 | 0.227 | 1.843 | 0.064 |
| Moving model | View B | 0.534 | 0.506 | 0.193 | 1.985 | 0.039 |
| Moving model | A+B | 0.536 | 0.484 | 0.210 | 1.902 | 0.051 |
