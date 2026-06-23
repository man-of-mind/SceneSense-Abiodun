# Moving-Ego Fusion Training Objective Audit

## Readout

### 8-loop moving model

- Saved-selection epoch proxy: epoch `76`, selection score `0.5732`, vehicle IoU `0.8739`, mIoU `0.8215`.
- Best vehicle-IoU epoch: epoch `76`, vehicle IoU `0.8739`, mIoU `0.8215`, selection score `0.5732`.
- Best mIoU epoch: epoch `69`, mIoU `0.8216`, vehicle IoU `0.8731`.

### 12-loop moving model

- Saved-selection epoch proxy: epoch `56`, selection score `0.5966`, vehicle IoU `0.8355`, mIoU `0.8099`.
- Best vehicle-IoU epoch: epoch `60`, vehicle IoU `0.8418`, mIoU `0.8106`, selection score `0.5962`.
- Best mIoU epoch: epoch `60`, mIoU `0.8106`, vehicle IoU `0.8418`.

## Conclusion

- For the 8-loop model, the selection-score checkpoint is effectively aligned with the best vehicle-IoU epoch. Checkpoint selection is not the main reason vehicle IoU is below 0.90.
- The 12-loop run has lower vehicle IoU even at its best vehicle-IoU epoch, so repeated loops on the same route are not the right standalone fix.
- The next useful levers are objective weighting, lower object-head pressure during segmentation fine-tuning, and/or route/view diversity focused on low and medium density scenes.
