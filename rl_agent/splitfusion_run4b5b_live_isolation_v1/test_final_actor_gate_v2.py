"""CPU-only tests for the paired final-actor gate."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    final_actor_gate_v2 as G,
)


class FinalActorManifestTest(unittest.TestCase):
    def test_bundled_manifests_are_exact_paired_authorities(self) -> None:
        run4, _ = G.load_manifest(G.bundled_manifest_path(G.RUN4B_VARIANT))
        run5, _ = G.load_manifest(G.bundled_manifest_path(G.RUN5B_VARIANT))
        self.assertEqual(run4.feature_order, run5.feature_order[:20])
        self.assertEqual(
            run4.scientific_channel_sha256,
            G.SCIENTIFIC_CHANNEL_SHA256,
        )
        self.assertEqual(
            run5.scientific_channel_sha256,
            G.SCIENTIFIC_CHANNEL_SHA256,
        )
        self.assertIsNone(run4.run5b_only_authority_sha256)
        self.assertEqual(
            run5.run5b_only_authority_sha256,
            G.RUN5B_ONLY_AUTHORITY_SHA256,
        )
        self.assertNotEqual(
            run4.actor_state_dict_sha256,
            G.PILOT_ACTOR_SHA256,
        )

    def _mutated_manifest(self, variant: str, mutation) -> Path:
        raw = json.loads(
            G.bundled_manifest_path(variant).read_text(encoding="utf-8")
        )
        mutation(raw)
        self._temp = tempfile.TemporaryDirectory()
        target = Path(self._temp.name) / "manifest.json"
        target.write_text(
            json.dumps(raw, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def tearDown(self) -> None:
        temp = getattr(self, "_temp", None)
        if temp is not None:
            temp.cleanup()

    def test_missing_extra_reordered_and_channel_drift_refused(self) -> None:
        cases = (
            lambda row: row.pop("model_binding_sha256"),
            lambda row: row.__setitem__("foreign", 1),
            lambda row: row["feature_order"].__setitem__(
                slice(0, 2), list(reversed(row["feature_order"][:2]))
            ),
            lambda row: row.__setitem__(
                "scientific_channel_sha256", "0" * 64
            ),
            lambda row: row.__setitem__(
                "run5b_only_authority_sha256",
                G.RUN5B_ONLY_AUTHORITY_SHA256,
            ),
        )
        for mutation in cases:
            with self.subTest(mutation=repr(mutation)):
                path = self._mutated_manifest(G.RUN4B_VARIANT, mutation)
                with self.assertRaises(G.FinalActorGateError):
                    G.load_manifest(path)
                self._temp.cleanup()
                del self._temp

    def test_run5b_missing_authority_and_selection_drift_refused(self) -> None:
        for key, value in (
            ("run5b_only_authority_sha256", None),
            ("selected_seed", 17),
            ("selected_update", 9500),
            ("preregistered_live_actor", False),
        ):
            with self.subTest(key=key):
                path = self._mutated_manifest(
                    G.RUN5B_VARIANT,
                    lambda row, k=key, v=value: row.__setitem__(k, v),
                )
                with self.assertRaises(G.FinalActorGateError):
                    G.load_manifest(path)
                self._temp.cleanup()
                del self._temp

    def test_pilot_refused_before_tensor_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            actor = Path(temp) / "actor_state_dict.pt"
            actor.write_bytes(b"pilot fixture")
            with (
                mock.patch.object(G, "_verify_evidence"),
                mock.patch.object(
                    G, "_sha_file", return_value=G.PILOT_ACTOR_SHA256
                ),
                mock.patch.object(
                    G,
                    "_load_tensor_state",
                    side_effect=AssertionError("must not deserialize pilot"),
                ) as tensor_load,
            ):
                with self.assertRaisesRegex(
                    G.FinalActorGateError, "quarantined"
                ):
                    G.verify_and_load_final_actor(
                        G.bundled_manifest_path(G.RUN4B_VARIANT),
                        actor,
                        evidence_root=Path(temp),
                    )
                tensor_load.assert_not_called()

    def test_cross_variant_file_hash_refused_before_tensor_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            actor = Path(temp) / "actor_state_dict.pt"
            actor.write_bytes(b"wrong variant")
            run4, _ = G.load_manifest(
                G.bundled_manifest_path(G.RUN4B_VARIANT)
            )
            with (
                mock.patch.object(G, "_verify_evidence"),
                mock.patch.object(
                    G,
                    "_sha_file",
                    return_value=run4.actor_state_dict_sha256,
                ),
                mock.patch.object(
                    G,
                    "_load_tensor_state",
                    side_effect=G.FinalActorGateError("actor file hash differs"),
                ),
            ):
                with self.assertRaises(G.FinalActorGateError):
                    G.verify_and_load_final_actor(
                        G.bundled_manifest_path(G.RUN5B_VARIANT),
                        actor,
                        evidence_root=Path(temp),
                    )


if __name__ == "__main__":
    unittest.main()
