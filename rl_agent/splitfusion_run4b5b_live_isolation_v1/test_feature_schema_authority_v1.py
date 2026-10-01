"""Refusal tests for exact B training feature-schema authorities."""

from __future__ import annotations

import copy
import unittest

from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as R4B
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import (
    run5b_state_contract as R5B,
)

from . import feature_schema_authority_v1 as S


class FeatureSchemaAuthorityTest(unittest.TestCase):
    def test_both_exact_training_authorities_pass(self) -> None:
        self.assertEqual(S.feature_schema_sha256(
            S.RUN4B_VARIANT, R4B.FEATURE_ORDER), R4B.FEATURE_SCHEMA_SHA256)
        self.assertEqual(S.feature_schema_sha256(
            S.RUN5B_VARIANT, R5B.FEATURE_ORDER), R5B.FEATURE_SCHEMA_SHA256)

    def test_live_order_and_authority_order_drift_refuse(self) -> None:
        changed_order = list(R4B.FEATURE_ORDER)
        changed_order[0], changed_order[1] = changed_order[1], changed_order[0]
        with self.assertRaisesRegex(
                S.FeatureSchemaAuthorityError, "order differs"):
            S.feature_schema_sha256(S.RUN4B_VARIANT, changed_order)
        changed_schema = copy.deepcopy(R4B.FEATURE_SCHEMA)
        changed_schema["feature_order"][0] = "foreign_feature"
        with self.assertRaisesRegex(
                S.FeatureSchemaAuthorityError, "order differs"):
            S.feature_schema_sha256(
                S.RUN4B_VARIANT, R4B.FEATURE_ORDER,
                schema_override=changed_schema)

    def test_count_scaling_prior_and_excluded_drift_refuse(self) -> None:
        cases = (
            ("feature_count", 19, "count differs"),
            ("scaling", {**R4B.FEATURE_SCHEMA["scaling"],
                         "camera_si_scaled": "changed"}, "scaling semantics"),
            ("prior_semantics", {**R4B.FEATURE_SCHEMA["prior_semantics"],
                                 "genesis": "changed"}, "prior_semantics"),
            ("excluded", [*R4B.FEATURE_SCHEMA["excluded"], "foreign"],
             "excluded semantics"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                changed = copy.deepcopy(R4B.FEATURE_SCHEMA)
                changed[field] = value
                with self.assertRaisesRegex(
                        S.FeatureSchemaAuthorityError, message):
                    S.feature_schema_sha256(
                        S.RUN4B_VARIANT, R4B.FEATURE_ORDER,
                        schema_override=changed)

    def test_run5_snr_scaling_and_exclusion_semantics_drift_refuse(self) -> None:
        changed = copy.deepcopy(R5B.FEATURE_SCHEMA)
        changed["position_20"]["scaling"] = "changed"
        with self.assertRaisesRegex(
                S.FeatureSchemaAuthorityError, "position_20 semantics"):
            S.feature_schema_sha256(
                S.RUN5B_VARIANT, R5B.FEATURE_ORDER,
                schema_override=changed)
        changed = copy.deepcopy(R5B.FEATURE_SCHEMA)
        changed["forbidden_feature_tokens"].append("foreign")
        with self.assertRaisesRegex(
                S.FeatureSchemaAuthorityError,
                "forbidden_feature_tokens semantics"):
            S.feature_schema_sha256(
                S.RUN5B_VARIANT, R5B.FEATURE_ORDER,
                schema_override=changed)


if __name__ == "__main__":
    unittest.main()
