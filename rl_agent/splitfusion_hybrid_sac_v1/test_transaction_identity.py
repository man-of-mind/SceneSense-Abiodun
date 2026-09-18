"""Comprehensive tests for the Hybrid-SAC transaction/action/feedback identity.

Three test methods:

1. ``test_executed_action_identity_exhausts_modes_and_q_range`` -- all 12 joint
   modes across every ``q_e4`` in ``[0, 9800]``: identity fields, keep/drop,
   schema/catalog bindings, exact-anchor IDs and null IDs for non-anchors.
2. ``test_multi_tensor_hold_feedback_and_canonical_bytes`` -- a multi-tensor
   hold whose earliest ``tensor_seq`` carries the reward request, the two-frame
   minimum, variable-duration holds, the derived feedback identity,
   permutation-invariant canonical bytes, and independently recomputed hashes.
3. ``test_rejects_malformed_identity_and_invalid_holds`` -- table-driven
   negative cases, including attestation-transfer forgeries built with
   :func:`dataclasses.replace`.

Hashes are recomputed here with a locally written canonicalizer and a locally
written container-flattener, rather than by calling the module's own helpers, so
the tests do not merely restate the implementation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
import hashlib
import json
import unittest
from itertools import permutations
from typing import Any, Mapping

from . import action_contract as ac
from . import transaction_identity as ti

SESSION = "3f263fce-cc44-476e-93b5-19d09d439471"
OTHER_SESSION = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _plain(value: Any) -> Any:
    """Locally written flattener for read-only mappings/tuples to JSON types."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _independent_canonical_bytes(payload: Any) -> bytes:
    """A locally written canonicalizer, independent of the module under test."""
    return json.dumps(
        _plain(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _independent_sha256(payload: Any) -> str:
    return hashlib.sha256(_independent_canonical_bytes(payload)).hexdigest()


class TransactionIdentityTest(unittest.TestCase):
    """Phase-2 identity records, canonical serialization and fail-closed rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ac.load_contract()
        cls.anchor_q_e4 = set(cls.contract.q_anchor_order)

    def _action_identity(self, mode_id: int, q_e4: int) -> ti.ExecutedActionIdentity:
        """Build an executed-action identity through the Phase-1 contract."""
        executable = self.contract.resolve(mode_id, q_e4 / ac.Q_E4_SCALE)
        return ti.ExecutedActionIdentity.from_executable_action(
            executable, self.contract
        )

    def _envelope(
        self,
        tensor_seq: int,
        carla_frame_id: int,
        reward_requested: bool,
        action: ti.ExecutedActionIdentity,
        *,
        session_uuid: str = SESSION,
        decision_seq: int = 7,
    ) -> ti.TensorTransmissionEnvelope:
        return ti.TensorTransmissionEnvelope(
            transaction=ti.TensorTransactionId(
                session_uuid=session_uuid,
                decision_seq=decision_seq,
                tensor_seq=tensor_seq,
                carla_frame_id=carla_frame_id,
            ),
            reward_requested=reward_requested,
            action=action,
        )

    # ------------------------------------------------------------------ #

    def test_executed_action_identity_exhausts_modes_and_q_range(self) -> None:
        contract = self.contract
        declared_anchors = contract.q_anchor_order
        self.assertEqual(contract.mode_count, 12)

        # schema constants are versioned, hex, and independently reproducible
        self.assertEqual(
            ti.SCHEMA_ID, "splitfusion_hybrid_sac_transaction_identity_v1"
        )
        self.assertEqual(ti.SCHEMA_VERSION, 2)
        self.assertEqual(
            ti.ACTION_IDENTITY_SCHEMA_ID,
            "splitfusion_hybrid_sac_executed_action_identity_v1",
        )
        self.assertEqual(
            ti.SCHEMA_SHA256, _independent_sha256(_plain(ti.SCHEMA_DESCRIPTOR))
        )
        self.assertEqual(
            ti.ACTION_IDENTITY_SCHEMA_SHA256,
            _independent_sha256(_plain(ti.ACTION_IDENTITY_DESCRIPTOR)),
        )
        for digest in (ti.SCHEMA_SHA256, ti.ACTION_IDENTITY_SCHEMA_SHA256):
            self.assertEqual(len(digest), 64)
            self.assertEqual(digest, digest.lower())
            int(digest, 16)
        # the descriptor embeds the frozen catalog binding and the action schema
        self.assertEqual(
            ti.SCHEMA_DESCRIPTOR["catalog_binding"],
            {"schema": ac.CATALOG_SCHEMA, "sha256": ac.CATALOG_SHA256},
        )
        self.assertEqual(
            ti.SCHEMA_DESCRIPTOR["executed_action_identity"],
            ti.ACTION_IDENTITY_DESCRIPTOR,
        )

        # --- exported descriptors are deep-frozen ----------------------- #
        manifest_spec = ti.SCHEMA_DESCRIPTOR["records"]["action_hold_manifest"]
        self.assertEqual(manifest_spec["minimum_tensors"], ti.MINIMUM_HOLD_TENSORS)
        self.assertEqual(ti.MINIMUM_HOLD_TENSORS, 2)
        self.assertIsNone(manifest_spec["maximum_tensors"])
        for target, key, value in (
            (ti.SCHEMA_DESCRIPTOR, "schema_id", "tampered"),
            (ti.ACTION_IDENTITY_DESCRIPTOR, "version", 99),
            (ti.SCHEMA_DESCRIPTOR["records"], "action_hold_manifest", {}),
            (manifest_spec, "minimum_tensors", 1),
            (manifest_spec["fields"], "tensors", "tampered"),
        ):
            with self.assertRaises(TypeError):
                target[key] = value  # type: ignore[index]
            with self.assertRaises(AttributeError):
                target.pop(key)  # type: ignore[union-attr]
        for sequence in (
            manifest_spec["invariants"],
            ti.ACTION_IDENTITY_DESCRIPTOR["q_e4_bounds"],
        ):
            self.assertIsInstance(sequence, tuple)
            with self.assertRaises(TypeError):
                sequence[0] = "tampered"  # type: ignore[index]
        # after every attempted mutation the descriptors still hash as published
        self.assertEqual(
            ti.SCHEMA_SHA256, _independent_sha256(_plain(ti.SCHEMA_DESCRIPTOR))
        )
        self.assertEqual(
            ti.ACTION_IDENTITY_SCHEMA_SHA256,
            _independent_sha256(_plain(ti.ACTION_IDENTITY_DESCRIPTOR)),
        )
        # the frozen chronology rule is stated in the descriptor
        self.assertIn("tensor_seq", ti.SCHEMA_DESCRIPTOR["tensor_seq_semantics"])
        self.assertIn(
            "carla_frame_id", ti.SCHEMA_DESCRIPTOR["tensor_seq_semantics"]
        )

        seen_anchor_ids = set()
        seen_anchor_profiles = set()
        anchor_rows = 0
        non_anchor_rows = 0

        for mode in contract.modes:
            for q_e4 in range(ac.Q_E4_MIN, ac.Q_E4_MAX + 1):
                identity = self._action_identity(mode.mode_id, q_e4)

                # --- identity fields ------------------------------------ #
                self.assertEqual(identity.execution_mode, ac.EXECUTION_MODE)
                self.assertEqual(identity.mode_id, mode.mode_id)
                self.assertEqual(identity.family, mode.family)
                self.assertEqual(identity.quantizer, mode.quantizer)
                self.assertEqual(identity.q_e4, q_e4)
                self.assertEqual(identity.canonical_mode, mode.canonical)

                # --- keep/drop, carried not recomputed ------------------ #
                expected_keep, expected_drop = ac.keep_drop_counts(q_e4)
                self.assertEqual(identity.keep_count, expected_keep)
                self.assertEqual(identity.drop_count, expected_drop)
                self.assertEqual(
                    identity.keep_count + identity.drop_count, ac.SPATIAL_CELLS
                )

                # --- schema / catalog bindings -------------------------- #
                self.assertEqual(identity.catalog_schema, ac.CATALOG_SCHEMA)
                self.assertEqual(identity.catalog_sha256, ac.CATALOG_SHA256)
                self.assertEqual(
                    identity.action_identity_schema, ti.ACTION_IDENTITY_SCHEMA_ID
                )
                self.assertEqual(
                    identity.action_identity_sha256,
                    ti.ACTION_IDENTITY_SCHEMA_SHA256,
                )

                # --- anchor IDs only for exact registered anchors ------- #
                payload = identity.to_canonical_dict()
                if q_e4 in self.anchor_q_e4:
                    anchor = contract.find_anchor(mode.family, mode.quantizer, q_e4)
                    self.assertIsNotNone(anchor)
                    self.assertTrue(identity.is_registered_anchor)
                    self.assertEqual(identity.action_id, anchor.action_id)
                    self.assertEqual(identity.profile_id, anchor.profile_id)
                    self.assertEqual(payload["action_id"], anchor.action_id)
                    self.assertEqual(payload["profile_id"], anchor.profile_id)
                    seen_anchor_ids.add(anchor.action_id)
                    seen_anchor_profiles.add(anchor.profile_id)
                    anchor_rows += 1
                else:
                    self.assertFalse(identity.is_registered_anchor)
                    self.assertIsNone(identity.action_id)
                    self.assertIsNone(identity.profile_id)
                    self.assertIsNone(payload["action_id"])
                    self.assertIsNone(payload["profile_id"])
                    non_anchor_rows += 1

                # catalog reconciliation succeeds, and the record built by
                # from_executable_action carries the attestation that makes it
                # serializable
                identity.verify_against_catalog(contract)
                self.assertTrue(identity.is_catalog_reconciled)
                identity.require_reconciled()

        # --- sweep accounting ------------------------------------------- #
        total = ac.Q_E4_MAX - ac.Q_E4_MIN + 1
        self.assertEqual(total, 9801)
        self.assertEqual(anchor_rows, 12 * len(declared_anchors))
        self.assertEqual(anchor_rows, 72)
        self.assertEqual(non_anchor_rows, 12 * (total - len(declared_anchors)))
        self.assertEqual(len(seen_anchor_ids), 72)
        self.assertEqual(len(seen_anchor_profiles), 72)
        self.assertEqual(seen_anchor_ids, {a.action_id for a in contract.anchors})
        self.assertEqual(
            seen_anchor_profiles, {a.profile_id for a in contract.anchors}
        )

        # --- canonical bytes at the boundaries and one interior point --- #
        for mode_id, q_e4 in ((0, ac.Q_E4_MIN), (11, ac.Q_E4_MAX), (5, 4237)):
            identity = self._action_identity(mode_id, q_e4)
            blob = identity.canonical_bytes()
            self.assertEqual(blob, _independent_canonical_bytes(
                identity.to_canonical_dict()
            ))
            self.assertEqual(
                identity.canonical_sha256(),
                _independent_sha256(identity.to_canonical_dict()),
            )
            text = blob.decode("ascii")
            self.assertNotIn(", ", text)
            self.assertNotIn(": ", text)
            if q_e4 == 4237:
                self.assertIn('"action_id":null', text)
                self.assertIn('"profile_id":null', text)

        # --- records are immutable -------------------------------------- #
        identity = self._action_identity(0, 3000)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identity.q_e4 = 0  # type: ignore[misc]

    # ------------------------------------------------------------------ #

    def test_multi_tensor_hold_feedback_and_canonical_bytes(self) -> None:
        contract = self.contract
        anchor_action = self._action_identity(9, 9000)
        non_anchor_action = self._action_identity(9, 4237)

        # Adversarial ordering, so that neither input position nor
        # carla_frame_id can be mistaken for chronology:
        #   * the earliest tensor_seq (101) is LAST in the input tuple;
        #   * the earliest tensor_seq does NOT carry the smallest frame id
        #     (5040, while tensor_seq 104 carries 5000);
        #   * tensor_seq and carla_frame_id are both non-consecutive, and
        #     carla_frame_id is not monotone in tensor_seq.
        tensors = (
            self._envelope(104, 5000, False, anchor_action),
            self._envelope(117, 5011, False, anchor_action),
            self._envelope(109, 5102, False, anchor_action),
            self._envelope(101, 5040, True, anchor_action),
        )
        manifest = ti.ActionHoldManifest.build(tensors)

        # --- hold structure --------------------------------------------- #
        self.assertEqual(manifest.tensor_count, 4)
        self.assertEqual(manifest.session_uuid, SESSION)
        self.assertEqual(manifest.decision_seq, 7)
        self.assertEqual(manifest.action, anchor_action)
        self.assertEqual(manifest.tensor_seqs, (101, 104, 109, 117))
        self.assertEqual(
            list(manifest.tensor_seqs), sorted(manifest.tensor_seqs)
        )
        # non-consecutive tensor_seq and non-monotone frame ids are preserved
        gaps = {
            b - a
            for a, b in zip(manifest.tensor_seqs, manifest.tensor_seqs[1:])
        }
        self.assertNotEqual(gaps, {1})
        frames = [m.transaction.carla_frame_id for m in manifest.tensors]
        self.assertNotEqual(frames, sorted(frames))

        # --- the two-frame minimum is the smallest completed hold -------- #
        self.assertEqual(ti.MINIMUM_HOLD_TENSORS, 2)
        minimum_hold = ti.ActionHoldManifest.build(
            [
                self._envelope(41, 6100, False, anchor_action),
                self._envelope(40, 6200, True, anchor_action),
            ]
        )
        self.assertEqual(minimum_hold.tensor_count, ti.MINIMUM_HOLD_TENSORS)
        self.assertEqual(minimum_hold.tensor_seqs, (40, 41))
        self.assertEqual(minimum_hold.reward_tensor_seq, 40)
        # a one-tensor hold is not a completed hold (negative case in test 3)
        with self.assertRaises(ti.ActionHoldError):
            ti.ActionHoldManifest.build(
                [self._envelope(500, 6000, True, anchor_action)]
            )

        # --- long variable-duration holds, no maximum -------------------- #
        for length in (2, 3, 7, 40, 137):
            variable = ti.ActionHoldManifest.build(
                [
                    self._envelope(3 * i, 9000 + 7 * i, i == 0, anchor_action)
                    for i in reversed(range(length))
                ]
            )
            self.assertEqual(variable.tensor_count, length)
            self.assertEqual(variable.reward_tensor_seq, 0)
            self.assertEqual(variable.tensor_seqs[0], 0)
            self.assertEqual(
                [m.reward_requested for m in variable.tensors],
                [True] + [False] * (length - 1),
            )
        long_hold = ti.ActionHoldManifest.build(
            [
                self._envelope(3 * i, 9000 + 7 * i, i == 0, anchor_action)
                for i in range(40)
            ]
        )
        self.assertEqual(long_hold.tensor_count, 40)
        self.assertEqual(long_hold.reward_tensor_seq, 0)

        # --- the earliest tensor_seq is the registered reward tensor ---- #
        self.assertEqual(manifest.reward_tensor_seq, 101)
        self.assertEqual(manifest.reward_tensor_seq, min(manifest.tensor_seqs))
        self.assertIs(manifest.reward_tensor, manifest.tensors[0])
        self.assertTrue(manifest.reward_tensor.reward_requested)
        self.assertEqual(
            [m.reward_requested for m in manifest.tensors],
            [True, False, False, False],
        )
        # chronology came from tensor_seq, not from input order or frame id
        self.assertIsNot(manifest.reward_tensor, tensors[0])
        self.assertEqual(manifest.reward_tensor.transaction.carla_frame_id, 5040)
        self.assertNotEqual(
            manifest.reward_tensor.transaction.carla_frame_id, min(frames)
        )
        self.assertEqual(
            sum(1 for m in manifest.tensors if m.reward_requested), 1
        )
        for member in manifest.tensors:
            self.assertEqual(member.action, anchor_action)
            self.assertEqual(member.transaction.decision_key, (SESSION, 7))

        # --- feedback identity derives from the reward tensor ----------- #
        feedback = manifest.reward_feedback_identity()
        self.assertEqual(feedback, ti.RewardFeedbackIdentity.from_manifest(manifest))
        self.assertEqual(feedback.session_uuid, SESSION)
        self.assertEqual(feedback.decision_seq, 7)
        self.assertEqual(feedback.reward_tensor_seq, 101)
        self.assertEqual(feedback.carla_frame_id, 5040)
        self.assertEqual(feedback.action, anchor_action)
        self.assertEqual(feedback.decision_key, (SESSION, 7))
        # it is the earliest tensor_seq's frame: neither the input-first
        # tensor's frame nor the smallest frame in the hold
        self.assertNotEqual(
            feedback.carla_frame_id, tensors[0].transaction.carla_frame_id
        )
        self.assertNotEqual(feedback.carla_frame_id, min(frames))
        self.assertEqual(feedback.action.action_id, 58)
        self.assertEqual(feedback.action.profile_id, "split_ae32_uint8_q9000")

        # --- permutation-invariant canonical bytes ---------------------- #
        reference_bytes = manifest.canonical_bytes()
        reference_sha = manifest.canonical_sha256()
        for order in permutations(tensors):
            permuted = ti.ActionHoldManifest.build(order)
            self.assertEqual(permuted.tensor_seqs, manifest.tensor_seqs)
            self.assertEqual(permuted.canonical_bytes(), reference_bytes)
            self.assertEqual(permuted.canonical_sha256(), reference_sha)
            self.assertEqual(
                permuted.reward_feedback_identity().canonical_bytes(),
                feedback.canonical_bytes(),
            )

        # --- canonical form properties ---------------------------------- #
        text = reference_bytes.decode("ascii")
        self.assertNotIn(", ", text)
        self.assertNotIn(": ", text)
        self.assertEqual(reference_bytes, text.encode("utf-8"))
        round_tripped = json.loads(text)
        self.assertEqual(round_tripped, manifest.to_canonical_dict())
        self.assertEqual(_independent_canonical_bytes(round_tripped), reference_bytes)
        # tensors keep their canonical ascending order inside the bytes
        self.assertEqual(
            [m["transaction"]["tensor_seq"] for m in round_tripped["tensors"]],
            [101, 104, 109, 117],
        )
        self.assertEqual(
            [m["reward_requested"] for m in round_tripped["tensors"]],
            [True, False, False, False],
        )
        # NaN/infinity can never appear in canonical bytes
        with self.assertRaises(ValueError):
            ti.canonical_json_bytes({"x": float("nan")})
        with self.assertRaises(ValueError):
            ti.canonical_json_bytes({"x": float("inf")})

        # --- schema/catalog bindings in every top-level record ---------- #
        for record in (manifest, feedback, manifest.reward_tensor):
            payload = record.to_canonical_dict()
            self.assertEqual(payload["schema_id"], ti.SCHEMA_ID)
            self.assertEqual(payload["schema_sha256"], ti.SCHEMA_SHA256)
            self.assertEqual(payload["schema_version"], ti.SCHEMA_VERSION)
            self.assertEqual(payload["catalog_schema"], ac.CATALOG_SCHEMA)
            self.assertEqual(payload["catalog_sha256"], ac.CATALOG_SHA256)
            self.assertEqual(
                payload["executed_action"], anchor_action.to_canonical_dict()
            )
            # independently recomputed hash
            self.assertEqual(record.canonical_sha256(), _independent_sha256(payload))
            self.assertEqual(
                record.canonical_bytes(), _independent_canonical_bytes(payload)
            )
        self.assertEqual(
            manifest.to_canonical_dict()["record"], "action_hold_manifest"
        )
        self.assertEqual(
            feedback.to_canonical_dict()["record"], "reward_feedback_identity"
        )
        self.assertEqual(
            manifest.reward_tensor.to_canonical_dict()["record"],
            "tensor_transmission_envelope",
        )

        # --- an independently rebuilt expected action block ------------- #
        expected_action_payload = {
            "action_id": 58,
            "action_identity_schema": (
                "splitfusion_hybrid_sac_executed_action_identity_v1"
            ),
            "action_identity_sha256": ti.ACTION_IDENTITY_SCHEMA_SHA256,
            "catalog_schema": "splitfusion_72_action_catalog_v1",
            "catalog_sha256": (
                "07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3"
            ),
            "drop_count": 19354,
            "execution_mode": "SPLIT",
            "family": "AE32",
            "keep_count": 2150,
            "mode_id": 9,
            "profile_id": "split_ae32_uint8_q9000",
            "q_e4": 9000,
            "quantizer": "UINT8",
        }
        self.assertEqual(anchor_action.to_canonical_dict(), expected_action_payload)
        self.assertEqual(
            anchor_action.canonical_sha256(),
            _independent_sha256(expected_action_payload),
        )
        self.assertEqual(
            sum(expected_action_payload[k] for k in ("keep_count", "drop_count")),
            ac.SPATIAL_CELLS,
        )

        # --- a hold on a non-anchor action serializes null IDs ---------- #
        non_anchor_manifest = ti.ActionHoldManifest.build(
            [
                self._envelope(203, 7013, False, non_anchor_action, decision_seq=8),
                self._envelope(200, 7000, True, non_anchor_action, decision_seq=8),
            ]
        )
        self.assertEqual(non_anchor_manifest.reward_tensor_seq, 200)
        non_anchor_payload = non_anchor_manifest.to_canonical_dict()
        self.assertIsNone(non_anchor_payload["executed_action"]["action_id"])
        self.assertIsNone(non_anchor_payload["executed_action"]["profile_id"])
        self.assertEqual(non_anchor_payload["executed_action"]["q_e4"], 4237)
        non_anchor_text = non_anchor_manifest.canonical_bytes().decode("ascii")
        self.assertIn('"action_id":null', non_anchor_text)
        self.assertIn('"profile_id":null', non_anchor_text)
        self.assertEqual(
            non_anchor_manifest.reward_feedback_identity().action, non_anchor_action
        )
        # different decisions produce different bytes
        self.assertNotEqual(
            non_anchor_manifest.canonical_bytes(), reference_bytes
        )

        # --- records are immutable and hashable value objects ----------- #
        with self.assertRaises(dataclasses.FrozenInstanceError):
            manifest.tensors = ()  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            feedback.reward_tensor_seq = 0  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            manifest.reward_tensor.reward_requested = False  # type: ignore[misc]
        self.assertEqual(
            ti.TensorTransactionId(SESSION, 7, 101, 5040),
            manifest.reward_tensor.transaction,
        )

    # ------------------------------------------------------------------ #

    def test_rejects_malformed_identity_and_invalid_holds(self) -> None:
        contract = self.contract
        action = self._action_identity(0, 3000)
        other_action = self._action_identity(0, 5000)
        other_mode_action = self._action_identity(4, 3000)
        keep_4237, drop_4237 = ac.keep_drop_counts(4237)
        keep_3000, drop_3000 = ac.keep_drop_counts(3000)
        anchor_3000 = contract.find_anchor("noAE", "UINT8", 3000)

        def txn(**overrides: Any) -> ti.TensorTransactionId:
            kwargs = {
                "session_uuid": SESSION,
                "decision_seq": 7,
                "tensor_seq": 101,
                "carla_frame_id": 5000,
            }
            kwargs.update(overrides)
            return ti.TensorTransactionId(**kwargs)  # type: ignore[arg-type]

        def act(**overrides: Any) -> ti.ExecutedActionIdentity:
            kwargs = {
                "execution_mode": "SPLIT",
                "mode_id": 0,
                "family": "noAE",
                "quantizer": "UINT8",
                "q_e4": 3000,
                "keep_count": keep_3000,
                "drop_count": drop_3000,
                "action_id": anchor_3000.action_id,
                "profile_id": anchor_3000.profile_id,
            }
            kwargs.update(overrides)
            return ti.ExecutedActionIdentity(**kwargs)  # type: ignore[arg-type]

        def hold(*members: ti.TensorTransmissionEnvelope) -> ti.ActionHoldManifest:
            return ti.ActionHoldManifest.build(members)

        fabricated_non_anchor = act(
            q_e4=4237,
            keep_count=keep_4237,
            drop_count=drop_4237,
            action_id=anchor_3000.action_id,
            profile_id=anchor_3000.profile_id,
        )
        # structurally honest, but never reconciled against the catalog
        unreconciled_honest = act()
        self.assertFalse(unreconciled_honest.is_catalog_reconciled)
        self.assertFalse(fabricated_non_anchor.is_catalog_reconciled)

        # Genuinely reconciled records, used as the *source* of the attestation
        # in the transfer forgeries below.  A reconciliation attestation is
        # bound to every serialized field, so it must not survive any mutation
        # of those fields and must not authenticate a different identity.
        reconciled_3000 = self._action_identity(0, 3000)
        reconciled_5000 = self._action_identity(0, 5000)
        reconciled_non_anchor = self._action_identity(0, 4237)
        anchor_5000 = contract.find_anchor("noAE", "UINT8", 5000)
        self.assertTrue(reconciled_3000.is_catalog_reconciled)
        self.assertTrue(reconciled_5000.is_catalog_reconciled)
        self.assertTrue(reconciled_non_anchor.is_catalog_reconciled)
        stolen_attestation = reconciled_3000._reconciliation
        wrong_anchor_id = act(action_id=anchor_3000.action_id + 1)
        missing_anchor_id = act(action_id=None, profile_id=None)
        contradictory_mode = act(family="AE128", quantizer="UINT8")

        cases = [
            # --- malformed session UUID ----------------------------------- #
            ("uuid uppercase", lambda: txn(session_uuid=SESSION.upper()),
             ti.IdentityFieldError),
            ("uuid braced", lambda: txn(session_uuid="{" + SESSION + "}"),
             ti.IdentityFieldError),
            ("uuid urn form", lambda: txn(session_uuid="urn:uuid:" + SESSION),
             ti.IdentityFieldError),
            ("uuid unhyphenated",
             lambda: txn(session_uuid=SESSION.replace("-", "")),
             ti.IdentityFieldError),
            ("uuid truncated", lambda: txn(session_uuid=SESSION[:-1]),
             ti.IdentityFieldError),
            ("uuid empty", lambda: txn(session_uuid=""), ti.IdentityFieldError),
            ("uuid garbage", lambda: txn(session_uuid="not-a-uuid"),
             ti.IdentityFieldError),
            ("uuid none", lambda: txn(session_uuid=None), ti.IdentityFieldError),
            ("uuid bytes", lambda: txn(session_uuid=SESSION.encode()),
             ti.IdentityFieldError),
            ("uuid object",
             lambda: txn(session_uuid=__import__("uuid").UUID(SESSION)),
             ti.IdentityFieldError),

            # --- malformed integer identifiers ---------------------------- #
            ("decision_seq negative", lambda: txn(decision_seq=-1),
             ti.IdentityFieldError),
            ("decision_seq bool", lambda: txn(decision_seq=True),
             ti.IdentityFieldError),
            ("decision_seq float", lambda: txn(decision_seq=7.0),
             ti.IdentityFieldError),
            ("decision_seq str", lambda: txn(decision_seq="7"),
             ti.IdentityFieldError),
            ("decision_seq none", lambda: txn(decision_seq=None),
             ti.IdentityFieldError),
            ("tensor_seq negative", lambda: txn(tensor_seq=-5),
             ti.IdentityFieldError),
            ("tensor_seq bool false", lambda: txn(tensor_seq=False),
             ti.IdentityFieldError),
            ("carla_frame_id negative", lambda: txn(carla_frame_id=-1),
             ti.IdentityFieldError),
            ("carla_frame_id bool", lambda: txn(carla_frame_id=True),
             ti.IdentityFieldError),
            ("carla_frame_id float", lambda: txn(carla_frame_id=5000.5),
             ti.IdentityFieldError),

            # --- malformed envelope --------------------------------------- #
            ("reward_requested int 1",
             lambda: ti.TensorTransmissionEnvelope(txn(), 1, action),
             ti.IdentityFieldError),
            ("reward_requested int 0",
             lambda: ti.TensorTransmissionEnvelope(txn(), 0, action),
             ti.IdentityFieldError),
            ("reward_requested str",
             lambda: ti.TensorTransmissionEnvelope(txn(), "true", action),
             ti.IdentityFieldError),
            ("reward_requested none",
             lambda: ti.TensorTransmissionEnvelope(txn(), None, action),
             ti.IdentityFieldError),
            ("envelope transaction wrong type",
             lambda: ti.TensorTransmissionEnvelope(
                 (SESSION, 7, 101, 5000), True, action
             ),
             ti.IdentityFieldError),
            ("envelope action wrong type",
             lambda: ti.TensorTransmissionEnvelope(txn(), True, "SPLIT/noAE/UINT8"),
             ti.ActionIdentityError),

            # --- malformed executed action identity ----------------------- #
            ("execution_mode LOCAL_GPU", lambda: act(execution_mode="LOCAL_GPU"),
             ti.ActionIdentityError),
            ("execution_mode SKIP", lambda: act(execution_mode="SKIP"),
             ti.ActionIdentityError),
            ("execution_mode lowercase", lambda: act(execution_mode="split"),
             ti.ActionIdentityError),
            ("mode_id too large", lambda: act(mode_id=12),
             ti.ActionIdentityError),
            ("mode_id negative", lambda: act(mode_id=-1),
             ti.IdentityFieldError),
            ("mode_id bool", lambda: act(mode_id=True), ti.IdentityFieldError),
            ("family empty", lambda: act(family=""), ti.IdentityFieldError),
            ("quantizer none", lambda: act(quantizer=None),
             ti.IdentityFieldError),
            ("q_e4 above max", lambda: act(q_e4=9801), ti.ActionIdentityError),
            ("q_e4 negative", lambda: act(q_e4=-1), ti.IdentityFieldError),
            ("q_e4 float", lambda: act(q_e4=3000.0), ti.IdentityFieldError),
            ("keep/drop swapped",
             lambda: act(keep_count=drop_3000, drop_count=keep_3000),
             ti.ActionIdentityError),
            ("keep off by one", lambda: act(keep_count=keep_3000 + 1),
             ti.ActionIdentityError),
            ("keep/drop from another q_e4",
             lambda: act(keep_count=keep_4237, drop_count=drop_4237),
             ti.ActionIdentityError),
            ("action_id without profile_id", lambda: act(profile_id=None),
             ti.ActionIdentityError),
            ("profile_id without action_id", lambda: act(action_id=None),
             ti.ActionIdentityError),
            ("action_id negative", lambda: act(action_id=-1),
             ti.IdentityFieldError),
            ("action_id bool", lambda: act(action_id=True),
             ti.IdentityFieldError),
            ("profile_id empty", lambda: act(profile_id=""),
             ti.IdentityFieldError),
            ("catalog schema wrong",
             lambda: act(catalog_schema="splitfusion_72_action_catalog_v2"),
             ti.ActionIdentityError),
            ("catalog sha wrong", lambda: act(catalog_sha256="0" * 64),
             ti.ActionIdentityError),
            ("action identity schema wrong",
             lambda: act(action_identity_schema="something_else_v1"),
             ti.ActionIdentityError),
            ("action identity sha wrong",
             lambda: act(action_identity_sha256="f" * 64),
             ti.ActionIdentityError),

            # --- a reconciliation attestation is not transferable ---------- #
            # The reported blocker: replace() a reconciled record's q_e4 and
            # keep/drop consistently, so nothing but the attestation can catch
            # it.  This previously serialized a q_e4=4237 record carrying the
            # q3000 anchor's action_id.
            ("replace() changes q_e4 and keep/drop consistently",
             lambda: replace(
                 reconciled_3000,
                 q_e4=4237,
                 keep_count=keep_4237,
                 drop_count=drop_4237,
             ),
             ti.UnreconciledActionIdentityError),
            ("replace() changes q_e4 to another registered anchor",
             lambda: replace(
                 reconciled_3000,
                 q_e4=5000,
                 keep_count=anchor_5000.keep_count,
                 drop_count=anchor_5000.drop_count,
             ),
             ti.UnreconciledActionIdentityError),
            # keep/drop alone is caught earlier, by the registered keep/drop
            # rule rather than by the attestation
            ("replace() changes keep_count only",
             lambda: replace(
                 reconciled_3000, keep_count=reconciled_3000.keep_count + 1
             ),
             ti.ActionIdentityError),
            ("replace() changes drop_count only",
             lambda: replace(
                 reconciled_3000, drop_count=reconciled_3000.drop_count - 1
             ),
             ti.ActionIdentityError),
            ("replace() changes mode_id, family and quantizer",
             lambda: replace(
                 reconciled_3000, mode_id=3, family="AE128", quantizer="UINT8"
             ),
             ti.UnreconciledActionIdentityError),
            ("replace() changes mode_id only",
             lambda: replace(reconciled_3000, mode_id=7),
             ti.UnreconciledActionIdentityError),
            ("replace() changes family only",
             lambda: replace(reconciled_3000, family="AE32"),
             ti.UnreconciledActionIdentityError),
            ("replace() changes quantizer only",
             lambda: replace(reconciled_3000, quantizer="UINT4"),
             ti.UnreconciledActionIdentityError),
            ("replace() changes action_id and profile_id",
             lambda: replace(
                 reconciled_3000,
                 action_id=anchor_5000.action_id,
                 profile_id=anchor_5000.profile_id,
             ),
             ti.UnreconciledActionIdentityError),
            ("replace() grafts anchor ids onto a reconciled non-anchor",
             lambda: replace(
                 reconciled_non_anchor,
                 action_id=anchor_5000.action_id,
                 profile_id=anchor_5000.profile_id,
             ),
             ti.UnreconciledActionIdentityError),
            ("replace() drops the anchor ids of a reconciled anchor",
             lambda: replace(reconciled_3000, action_id=None, profile_id=None),
             ti.UnreconciledActionIdentityError),
            ("attestation copied onto another valid identity",
             lambda: replace(
                 reconciled_5000, _reconciliation=stolen_attestation
             ),
             ti.UnreconciledActionIdentityError),
            ("attestation copied onto a reconciled non-anchor",
             lambda: replace(
                 reconciled_non_anchor, _reconciliation=stolen_attestation
             ),
             ti.UnreconciledActionIdentityError),
            ("attestation copied into a fresh direct construction",
             lambda: act(
                 q_e4=5000,
                 keep_count=anchor_5000.keep_count,
                 drop_count=anchor_5000.drop_count,
                 action_id=anchor_5000.action_id,
                 profile_id=anchor_5000.profile_id,
                 _reconciliation=stolen_attestation,
             ),
             ti.UnreconciledActionIdentityError),
            ("attestation copied onto a fabricated anchor identity",
             lambda: act(
                 q_e4=4237,
                 keep_count=keep_4237,
                 drop_count=drop_4237,
                 action_id=anchor_3000.action_id,
                 profile_id=anchor_3000.profile_id,
                 _reconciliation=stolen_attestation,
             ),
             ti.UnreconciledActionIdentityError),

            # --- unreconciled / fabricated identity blocked before any
            #     canonical serialization can happen -------------------------- #
            ("fabricated identity to_canonical_dict",
             lambda: fabricated_non_anchor.to_canonical_dict(),
             ti.UnreconciledActionIdentityError),
            ("fabricated identity canonical_bytes",
             lambda: fabricated_non_anchor.canonical_bytes(),
             ti.UnreconciledActionIdentityError),
            ("fabricated identity canonical_sha256",
             lambda: fabricated_non_anchor.canonical_sha256(),
             ti.UnreconciledActionIdentityError),
            ("fabricated identity in an envelope",
             lambda: ti.TensorTransmissionEnvelope(
                 txn(), True, fabricated_non_anchor
             ),
             ti.UnreconciledActionIdentityError),
            ("unreconciled honest identity to_canonical_dict",
             lambda: unreconciled_honest.to_canonical_dict(),
             ti.UnreconciledActionIdentityError),
            ("unreconciled honest identity in an envelope",
             lambda: ti.TensorTransmissionEnvelope(
                 txn(), True, unreconciled_honest
             ),
             ti.UnreconciledActionIdentityError),
            ("unreconciled honest identity in a feedback record",
             lambda: ti.RewardFeedbackIdentity(
                 session_uuid=SESSION,
                 decision_seq=7,
                 reward_tensor_seq=101,
                 carla_frame_id=5000,
                 action=unreconciled_honest,
             ),
             ti.UnreconciledActionIdentityError),
            ("require_reconciled on an unreconciled identity",
             lambda: unreconciled_honest.require_reconciled(),
             ti.UnreconciledActionIdentityError),
            ("forged reconciliation attestation",
             lambda: act(_reconciliation=("forged", ac.CATALOG_SHA256)),
             ti.UnreconciledActionIdentityError),
            ("attestation bound to a foreign catalog sha",
             lambda: act(_reconciliation=(object(), "0" * 64)),
             ti.UnreconciledActionIdentityError),
            ("reconciled_against a fabricated identity",
             lambda: fabricated_non_anchor.reconciled_against(contract),
             ti.ActionIdentityError),

            # --- fabricated anchor identity (needs the catalog) ----------- #
            ("anchor id fabricated on non-anchor q_e4",
             lambda: fabricated_non_anchor.verify_against_catalog(contract),
             ti.ActionIdentityError),
            ("wrong anchor id on a real anchor",
             lambda: wrong_anchor_id.verify_against_catalog(contract),
             ti.ActionIdentityError),
            ("missing anchor id on a real anchor",
             lambda: missing_anchor_id.verify_against_catalog(contract),
             ti.ActionIdentityError),
            ("mode_id contradicts family/quantizer",
             lambda: contradictory_mode.verify_against_catalog(contract),
             ti.ActionIdentityError),
            ("from_executable_action with a non-action",
             lambda: ti.ExecutedActionIdentity.from_executable_action(
                 "SPLIT/noAE/UINT8", contract
             ),
             ti.ActionIdentityError),
            ("from_executable_action with a non-contract",
             lambda: ti.ExecutedActionIdentity.from_executable_action(
                 contract.resolve(0, 0.3), object()
             ),
             ti.ActionIdentityError),

            # --- invalid action holds ------------------------------------- #
            ("hold with zero tensors", lambda: hold(), ti.ActionHoldError),
            ("hold from an empty list",
             lambda: ti.ActionHoldManifest.build([]), ti.ActionHoldError),
            ("one-tensor hold is not completed",
             lambda: hold(self._envelope(101, 5000, True, action)),
             ti.ActionHoldError),
            ("one-tensor hold without a reward request",
             lambda: hold(self._envelope(101, 5000, False, action)),
             ti.ActionHoldError),
            ("reward requested on the later tensor only",
             lambda: hold(
                 self._envelope(101, 5000, False, action),
                 self._envelope(104, 5010, True, action),
             ),
             ti.ActionHoldError),
            ("reward requested on the last of four",
             lambda: hold(
                 self._envelope(101, 5000, False, action),
                 self._envelope(104, 5010, False, action),
                 self._envelope(109, 5020, False, action),
                 self._envelope(117, 5030, True, action),
             ),
             ti.ActionHoldError),
            ("reward requested on a middle tensor",
             lambda: hold(
                 self._envelope(101, 5000, False, action),
                 self._envelope(104, 5010, True, action),
                 self._envelope(109, 5020, False, action),
             ),
             ti.ActionHoldError),
            ("reward on the smallest frame but not the earliest tensor_seq",
             lambda: hold(
                 self._envelope(101, 5900, False, action),
                 self._envelope(104, 5000, True, action),
             ),
             ti.ActionHoldError),
            ("earliest requests but a later tensor also requests",
             lambda: hold(
                 self._envelope(101, 5000, True, action),
                 self._envelope(104, 5010, False, action),
                 self._envelope(109, 5020, True, action),
             ),
             ti.ActionHoldError),
            ("hold with no reward request",
             lambda: hold(
                 self._envelope(101, 5000, False, action),
                 self._envelope(102, 5010, False, action),
             ),
             ti.ActionHoldError),
            ("hold with two reward requests",
             lambda: hold(
                 self._envelope(101, 5000, True, action),
                 self._envelope(102, 5010, True, action),
             ),
             ti.ActionHoldError),
            ("hold with three reward requests",
             lambda: hold(
                 self._envelope(101, 5000, True, action),
                 self._envelope(102, 5010, True, action),
                 self._envelope(103, 5020, True, action),
             ),
             ti.ActionHoldError),
            ("hold with duplicate tensor_seq",
             lambda: hold(
                 self._envelope(101, 5000, True, action),
                 self._envelope(101, 5010, False, action),
             ),
             ti.ActionHoldError),
            ("hold with mixed decision_seq",
             lambda: hold(
                 self._envelope(101, 5000, True, action),
                 self._envelope(102, 5010, False, action, decision_seq=8),
             ),
             ti.ActionHoldError),
            ("hold with mixed session_uuid",
             lambda: hold(
                 self._envelope(101, 5000, True, action),
                 self._envelope(
                     102, 5010, False, action, session_uuid=OTHER_SESSION
                 ),
             ),
             ti.ActionHoldError),
            ("hold with mixed q_e4",
             lambda: hold(
                 self._envelope(101, 5000, True, action),
                 self._envelope(102, 5010, False, other_action),
             ),
             ti.ActionHoldError),
            ("hold with mixed joint mode",
             lambda: hold(
                 self._envelope(101, 5000, True, action),
                 self._envelope(102, 5010, False, other_mode_action),
             ),
             ti.ActionHoldError),
            ("hold with a non-envelope member",
             lambda: hold(self._envelope(101, 5000, True, action), "tensor"),
             ti.ActionHoldError),
            ("hold from a non-iterable",
             lambda: ti.ActionHoldManifest(42), ti.ActionHoldError),
            ("hold from a string",
             lambda: ti.ActionHoldManifest("tensors"), ti.ActionHoldError),

            # --- invalid feedback identity -------------------------------- #
            ("feedback from a non-manifest",
             lambda: ti.RewardFeedbackIdentity.from_manifest(
                 self._envelope(101, 5000, True, action)
             ),
             ti.ActionHoldError),
            ("feedback with a malformed uuid",
             lambda: ti.RewardFeedbackIdentity(
                 session_uuid="nope",
                 decision_seq=7,
                 reward_tensor_seq=101,
                 carla_frame_id=5000,
                 action=action,
             ),
             ti.IdentityFieldError),
            ("feedback with a negative tensor_seq",
             lambda: ti.RewardFeedbackIdentity(
                 session_uuid=SESSION,
                 decision_seq=7,
                 reward_tensor_seq=-1,
                 carla_frame_id=5000,
                 action=action,
             ),
             ti.IdentityFieldError),
            ("feedback with a bool decision_seq",
             lambda: ti.RewardFeedbackIdentity(
                 session_uuid=SESSION,
                 decision_seq=False,
                 reward_tensor_seq=101,
                 carla_frame_id=5000,
                 action=action,
             ),
             ti.IdentityFieldError),
            ("feedback with a wrong-typed action",
             lambda: ti.RewardFeedbackIdentity(
                 session_uuid=SESSION,
                 decision_seq=7,
                 reward_tensor_seq=101,
                 carla_frame_id=5000,
                 action={"q_e4": 3000},
             ),
             ti.ActionIdentityError),
        ]

        # The negative-case count is derived from the table, never hard-coded:
        # labels must be unique and every row must actually be exercised.
        labels = [label for label, _, _ in cases]
        self.assertEqual(
            len(labels), len(set(labels)), "negative-case labels must be unique"
        )
        exercised = 0
        for label, thunk, expected in cases:
            with self.subTest(case=label):
                with self.assertRaises(expected) as caught:
                    thunk()
                # every failure is a specific, catchable contract error with a
                # useful message
                self.assertIsInstance(caught.exception, ti.TransactionIdentityError)
                self.assertIsInstance(caught.exception, ac.ActionContractError)
                self.assertTrue(str(caught.exception).strip())
            exercised += 1
        self.assertEqual(exercised, len(cases))
        # published for the report; the number is the table length itself
        type(self).negative_case_count = len(cases)

        # --- the fabricated record can never be serialized, while the
        #     honest record for the same q_e4 serializes normally ---------- #
        self.assertEqual(fabricated_non_anchor.q_e4, 4237)
        self.assertTrue(fabricated_non_anchor.is_registered_anchor)
        self.assertFalse(fabricated_non_anchor.is_catalog_reconciled)
        for blocked in (
            fabricated_non_anchor.to_canonical_dict,
            fabricated_non_anchor.canonical_bytes,
            fabricated_non_anchor.canonical_sha256,
        ):
            with self.assertRaises(ti.UnreconciledActionIdentityError):
                blocked()
        honest_non_anchor = self._action_identity(0, 4237)
        self.assertFalse(honest_non_anchor.is_registered_anchor)
        self.assertTrue(honest_non_anchor.is_catalog_reconciled)
        honest_non_anchor.verify_against_catalog(contract)
        honest_payload = honest_non_anchor.to_canonical_dict()
        self.assertIsNone(honest_payload["action_id"])
        self.assertIsNone(honest_payload["profile_id"])
        self.assertEqual(honest_payload["q_e4"], 4237)

        # --- an unchanged honest identity still serializes normally ------ #
        for honest in (reconciled_3000, reconciled_5000, reconciled_non_anchor):
            self.assertTrue(honest.is_catalog_reconciled)
            honest.require_reconciled()
            honest.verify_against_catalog(contract)
            self.assertEqual(
                honest.canonical_sha256(),
                _independent_sha256(honest.to_canonical_dict()),
            )
            # replace() with no field change is not a forgery: the binding still
            # matches, so the record stays reconciled and byte-identical
            untouched = replace(honest)
            self.assertTrue(untouched.is_catalog_reconciled)
            self.assertEqual(untouched.canonical_bytes(), honest.canonical_bytes())
            # and it is still usable in the records that embed it
            ti.TensorTransmissionEnvelope(txn(), True, untouched)
        # a mutated copy becomes serializable again only by re-reconciling it
        re_reconciled = replace(
            reconciled_3000,
            q_e4=5000,
            keep_count=anchor_5000.keep_count,
            drop_count=anchor_5000.drop_count,
            action_id=anchor_5000.action_id,
            profile_id=anchor_5000.profile_id,
            _reconciliation=None,
        ).reconciled_against(contract)
        self.assertTrue(re_reconciled.is_catalog_reconciled)
        self.assertEqual(re_reconciled.to_canonical_dict(),
                         reconciled_5000.to_canonical_dict())

        # --- mutation that bypasses __init__ is caught before serialization - #
        # object.__setattr__ evades every constructor check, so the binding is
        # recomputed on acceptance as well: this must fail no later than
        # canonical serialization.
        smuggled = self._action_identity(0, 3000)
        object.__setattr__(smuggled, "q_e4", 4237)
        object.__setattr__(smuggled, "keep_count", keep_4237)
        object.__setattr__(smuggled, "drop_count", drop_4237)
        self.assertFalse(smuggled.is_catalog_reconciled)
        for blocked in (
            smuggled.require_reconciled,
            smuggled.to_canonical_dict,
            smuggled.canonical_bytes,
            smuggled.canonical_sha256,
        ):
            with self.assertRaises(ti.UnreconciledActionIdentityError):
                blocked()
        with self.assertRaises(ti.UnreconciledActionIdentityError):
            ti.TensorTransmissionEnvelope(txn(), True, smuggled)
        with self.assertRaises(ti.UnreconciledActionIdentityError):
            ti.RewardFeedbackIdentity(
                session_uuid=SESSION,
                decision_seq=7,
                reward_tensor_seq=101,
                carla_frame_id=5000,
                action=smuggled,
            )

        # --- the reconciled counterpart of the same identity is accepted - #
        reconciled = unreconciled_honest.reconciled_against(contract)
        self.assertTrue(reconciled.is_catalog_reconciled)
        self.assertEqual(reconciled, unreconciled_honest)  # same serialized fields
        self.assertEqual(
            reconciled.to_canonical_dict(),
            self._action_identity(0, 3000).to_canonical_dict(),
        )
        ti.TensorTransmissionEnvelope(txn(), True, reconciled)

        # --- the minimum completed hold and its permutation are accepted - #
        accepted = ti.ActionHoldManifest.build(
            [
                self._envelope(104, 5000, False, action),
                self._envelope(101, 5090, True, action),
            ]
        )
        self.assertEqual(accepted.tensor_count, ti.MINIMUM_HOLD_TENSORS)
        self.assertEqual(accepted.reward_tensor_seq, 101)
        self.assertEqual(
            ti.ActionHoldManifest.build(
                list(reversed(accepted.tensors))
            ).canonical_bytes(),
            accepted.canonical_bytes(),
        )

        # --- valid boundary values are accepted ------------------------- #
        ti.TensorTransactionId(SESSION, 0, 0, 0)
        self._action_identity(0, ac.Q_E4_MIN).verify_against_catalog(contract)
        self._action_identity(11, ac.Q_E4_MAX).verify_against_catalog(contract)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
