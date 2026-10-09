# Offline exact scorer component interface

This package consumes the coordinator-owned interface in
`rl_agent/splitfusion_numeric_gt_integration_v1/INTERFACE.json`. It defines no
wire format and installs no runtime hook. All code changes are additive.

```python
from rl_agent.splitfusion_minimal_quality_scorer_v1.scorer import (
    NumericalScorer, reconstruct_evidence, archive_payloads,
)
from rl_agent.splitfusion_minimal_quality_scorer_v1.shared_admission import (
    admit_prediction, join_admitted_inputs,
)

scorer = NumericalScorer()  # startup: verify and bind frozen numerical code
prediction = admit_prediction(
    scorer, identity=owned_prediction_identity, labels=actual_native_labels,
    records=complete_prediction_records, prediction_contract=prediction_contract,
)
frame = join_admitted_inputs(scorer, target=admitted_numerical_target,
                            prediction=prediction)
result = scorer.score(frame)
measurement = result.measurement  # unchanged dict/list representation
associations = result.associations

# After scoring/feedback, when the coordinator's artifact reference is ready:
evidence = reconstruct_evidence(result, legacy_reference=legacy_reference)
payloads = archive_payloads(result, legacy_reference=legacy_reference)
```

`admit_prediction` is an actual numerical trust boundary, not an unchecked
flag. It validates full capture/action identity, reward-ticket status, native
calibration/shape/dtype/labels, content checksum, the 512-record evaluator limit,
classes, frame binding, score bounds and every frozen required numerical field,
including fields not consumed by matching. It owns immutable label bytes,
records and identity. The prediction contract is the existing v2
`PredictionContract.document()`; production is exactly **720x1280**. The
original explicit CPU-fixture contract remains isolated to synthetic tests.

`join_admitted_inputs` requires the coordinator's private `NumericalTarget`
and this scorer's private prediction. It compares complete capture and policy
identities, retains the target's source/plan/history/semantic bindings, converts
original ordered row values with the unchanged CSV string-to-float semantics,
and returns the private immutable scoring input. It performs no file reads,
hashes, PNG operations, CSV parsing, JSON packaging or metadata thawing. Native
GT masks remain 432x768 uint8 bytes-backed views. Full input ownership stays
alive through the result. This component does not advance history.

`score` performs prediction nearest-neighbor resize when needed, IGNORE=255
confusion counts, one original detection match pass, and the three ordered
person stages: eligible, AVO-ignore, structural-ignore. No restoration, source
reads/hashing, PNG operation or evaluation-evidence construction occurs there.
There is no Q_perc/reward reimplementation: pass its measurement to the
unchanged caller. Undefined support remains undefined.

Association coordinates:

- `detection_distance_order`: `(original_prediction_index, full_object_table_index, error_m)`
  in the original matcher insertion order; both classes are retained.
- `vehicle_prediction_order`: the original detector's score/class stable
  prediction order, with original record indices and full object-table indices.
  This is the measurement error-list order; it need not equal raw record order.
- `person_prediction_order`: original raw prediction order, with target indices
  into the eligible AVO list. `eligible_indices` maps those back to the original
  qualified list. The same mapping is retained for AVO-ignore associations.
- `person_avo_ignored`, `person_structural_ignored`, `vehicle_neutral`: additional
  neutralization audit associations. The original evidence's public matched
  pairs are reconstructed without rerunning any matcher.

`reconstruct_evidence` requires a coordinator `LegacyArtifactReference` for
the shared path. It checks the semantic/source/full-table binding, then uses
the original prepared-target digest. That digest is never relabeled as the new
semantic digest. The artifact producer must prove the old digest from actual
artifact bytes; this scorer cannot infer it before artifacts exist. Missing or
foreign references and encoding failures raise explicitly. No disk writes or
archive scheduling occur in this component.

Coordinator-owned work still required before a live path:

- Authenticate and admit numerical target transport; bind source verification
  and this loaded code to the session and recheck pins at postflight.
- Reconcile the actual prediction FrameContext, calibration, exact camera pose,
  stream/sequence/capture time and execution contract against the target and
  planned job. The retained prediction archive lacks the entire live context;
  this component test does not replace existing pose guards.
- Join admitted prediction/target once, route measurements to the unchanged
  quality/feedback functions, and enforce accepted-history transactions.
- Keep immutable target, prediction and source attachment alive until the
  bounded archive confirms durability; preserve explicit late archive faults.
- Run full-path offline composition with the new producer, then serialize
  performance measurement. No runtime or deadline qualification is claimed.

`offline_admission.admit_oracle_fixture` is only a fixture derivation helper:
it deliberately restores the legacy target and must not be wired into a new
live path. The shared adapter above does not depend on that helper or a producer.
