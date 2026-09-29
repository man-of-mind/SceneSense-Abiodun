"""Phase-3 CPU tests: continuous (mode_id, q_e4) identity and protocol parity.

Fake front/ranker/codec/tail/AE objects only.  The ranker-driven cell
selection is the registered ``ProductionSplitCodec._selection`` (real
``continuous_q.select_cells``) on a CPU score map.  No CUDA, model inference,
OAI or network process is used.  This is identity/protocol parity, not a
model-output or performance claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
import uuid

import torch

from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as ac
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)
from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
from rl_agent.splitfusion_live_dispatch_v1 import envelope as sfd1
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    DecodedC2,
    InnerIdentity,
    InspectedInnerPayload,
    ProductionSplitCodec,
)

from . import continuous_execution_v2 as X

CONTRACT = dec.load_dynamic_execution_contract()
SESSION = str(uuid.UUID(int=43))
LINEAGE = hashlib.sha256(b"run4-seed43-update10000").hexdigest()


class FakeRanker:
    def __init__(self) -> None:
        self.calls = 0
        generator = torch.Generator().manual_seed(1234)
        self.scores = torch.rand((112, 192), generator=generator)

    def score_cells(self, c2):
        self.calls += 1
        return self.scores


class FakeAE:
    def __init__(self, family):
        self.family = family


class FakeCodec:
    """Real registered cell selection; JSON framing in place of the codecs."""

    def __init__(self, *, corrupt_q_e4: bool = False) -> None:
        self.corrupt_q_e4 = corrupt_q_e4

    def encode(self, view, c2, *, ranker, ae_encoder, timing) -> bytes:
        with timing.stage("ranker_selection"):
            plan, selection = ProductionSplitCodec._selection(view, c2, ranker)
        if view.family == "noAE":
            assert ae_encoder is None
        else:
            assert ae_encoder is not None and ae_encoder.family == view.family
        identity = dataclasses.asdict(X.expected_inner(view))
        if self.corrupt_q_e4:
            identity["q_e4"] = (identity["q_e4"] + 1) % 9801
        return json.dumps({
            "identity": identity,
            "keep": None if selection is None else selection.keep_indices.tolist(),
        }, sort_keys=True).encode()

    def inspect(self, payload, *, timing):
        document = json.loads(payload)
        return InspectedInnerPayload(
            identity=InnerIdentity(**document["identity"]),
            compressed_bytes=len(payload), uncompressed_bytes=len(payload),
            kind="fake", sparse_bytes=payload, parsed=document)

    def decode(self, inspected, *, decoder, tail_device, timing):
        return DecodedC2(c2=torch.zeros(1), finite=True, device=tail_device)


def runtimes(codec=None):
    ranker = FakeRanker()
    ue = X.ContinuousUERuntimeV2(
        CONTRACT, front=lambda frame: torch.zeros(1), ranker=ranker,
        ae_encoders={f: FakeAE(f) for f in X.AE_FAMILIES}, codec=codec or FakeCodec())
    edge = X.ContinuousEdgeRuntimeV2(
        CONTRACT, tail=lambda c2, _meta: "PERCEPTION",
        ae_decoders={f: FakeAE(f) for f in X.AE_FAMILIES}, codec=FakeCodec())
    return ue, edge, ranker


def frame(seq=0, *, reward=True):
    return X.FrameIdentityV2(
        session_uuid=SESSION, controller_lineage_sha256=LINEAGE,
        decision_seq=seq, ticket_seq=seq, frame_id=1000 + seq, tensor_seq=seq,
        capture_timestamp_ns=10_000_000_000 + seq, reward_requested=reward)


def off_anchor_values(mode_id: int) -> tuple[int, int, int]:
    lower, upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[mode_id]
    anchors = set(CONTRACT.action_contract.q_anchor_order)

    def off(value, step):
        while value in anchors:
            value += step
        return value
    return off(lower, 1), off((lower + upper) // 2, 1), off(upper, -1)


class IdentitySeamTest(unittest.TestCase):
    def test_all_72_anchors_keep_exact_registered_identity(self) -> None:
        count = 0
        for mode_id in range(12):
            for anchor in CONTRACT.action_contract.anchors_for_mode(mode_id):
                profile = CONTRACT.resolve_q_e4(mode_id, anchor.q_e4)
                identity = X.executed_identity_from_profile(profile, CONTRACT)
                self.assertEqual((identity.action_id, identity.profile_id),
                                 (anchor.action_id, anchor.profile_id))
                self.assertEqual(profile.measurement_status, dec.MEASURED_ANCHOR)
                identity.require_reconciled()
                count += 1
        self.assertEqual(count, 72)

    def test_seam_equals_training_identity_path_for_every_q(self) -> None:
        catalog = ac.load_contract()
        for mode_id in range(12):
            for q_e4 in range(0, 9801):
                profile = CONTRACT.resolve_q_e4(mode_id, q_e4)
                seam = X.executed_identity_from_profile(profile, CONTRACT)
                training = ExecutedActionIdentity.from_executable_action(
                    catalog.resolve(mode_id, q_e4 / float(ac.Q_E4_SCALE)), catalog)
                self.assertEqual(seam, training)
                self.assertEqual(seam.to_canonical_dict(), training.to_canonical_dict())
                view = X.codec_view(profile)
                self.assertEqual(view.q_e4, q_e4)

    def test_off_anchor_values_carry_no_anchor_and_no_snapping(self) -> None:
        for mode_id in range(12):
            for q_e4 in off_anchor_values(mode_id):
                profile = CONTRACT.resolve_q_e4(mode_id, q_e4)
                identity = X.executed_identity_from_profile(profile, CONTRACT)
                self.assertIsNone(identity.action_id)
                self.assertIsNone(identity.profile_id)
                self.assertEqual(profile.measurement_status, dec.UNMEASURED_OFF_ANCHOR)
                self.assertEqual(identity.q_e4, q_e4)
                self.assertEqual((identity.keep_count, identity.drop_count),
                                 ac.keep_drop_counts(q_e4))

    def test_tampered_profile_refused(self) -> None:
        profile = CONTRACT.resolve_q_e4(4, 8123)
        for change in ({"q_e4": 8124}, {"keep_count": profile.keep_count + 1},
                       {"action_id": 5, "profile_id": "fake"},
                       {"execution_bundle_sha256": "0" * 64}):
            with self.assertRaises(dec.DynamicExecutionContractError):
                X.executed_identity_from_profile(
                    dataclasses.replace(profile, **change), CONTRACT)


class RoundTripTest(unittest.TestCase):
    def _round_trip(self, mode_id, q_e4, *, reward=True):
        ue, edge, ranker = runtimes()
        profile = CONTRACT.resolve_q_e4(mode_id, q_e4)
        prepared = ue.prepare(profile, object(), frame(reward=reward))
        result = edge.process(prepared.wire_bytes)
        self.assertEqual(result.action, prepared.action)
        self.assertEqual(result.profile, profile)
        self.assertEqual(result.envelope.identity_dict(), prepared.envelope.identity_dict())
        self.assertEqual(result.perception, "PERCEPTION")
        return prepared, result, ranker

    def test_anchors_and_off_anchor_values_round_trip_for_every_mode(self) -> None:
        for mode_id in range(12):
            values = list(off_anchor_values(mode_id)) + list(
                CONTRACT.action_contract.q_anchor_order)
            for q_e4 in values:
                prepared, result, _ = self._round_trip(mode_id, q_e4)
                self.assertEqual(result.envelope.anchor_action_id,
                                 CONTRACT.resolve_q_e4(mode_id, q_e4).action_id)

    def test_selection_is_saliency_ranked_not_random(self) -> None:
        for q_e4 in (1, 3000, 4321, 9799):
            prepared, result, ranker = self._round_trip(9, q_e4)
            keep = json.loads(result.envelope.inner_payload)["keep"]
            expected = torch.topk(ranker.scores.reshape(-1),
                                  prepared.profile.keep_count).indices.sort().values
            self.assertEqual(keep, expected.tolist())
            self.assertEqual(len(keep), prepared.profile.keep_count)

    def test_q_zero_bypasses_ranker(self) -> None:
        prepared, result, ranker = self._round_trip(11, 0)
        self.assertEqual(ranker.calls, 0)
        self.assertIsNone(json.loads(result.envelope.inner_payload)["keep"])

    def test_reward_flag_and_serialization_are_deterministic(self) -> None:
        ue, _, _ = runtimes()
        profile = CONTRACT.resolve_q_e4(7, 6001)
        a = ue.prepare(profile, object(), frame(3, reward=False)).wire_bytes
        b = ue.prepare(profile, object(), frame(3, reward=False)).wire_bytes
        self.assertEqual(a, b)
        self.assertFalse(X.unpack_envelope_v3(a).reward_requested)
        self.assertTrue(X.unpack_envelope_v3(
            ue.prepare(profile, object(), frame(3)).wire_bytes).reward_requested)


class RefusalTest(unittest.TestCase):
    def _prepared(self, mode_id=6, q_e4=7777):
        ue, edge, _ = runtimes()
        return ue.prepare(CONTRACT.resolve_q_e4(mode_id, q_e4), object(), frame()), edge

    def _repack(self, prepared, **changes):
        return X.pack_envelope_v3(dataclasses.replace(prepared.envelope, **changes))

    def test_edge_refuses_identity_mismatch(self) -> None:
        prepared, edge = self._prepared()
        anchor = CONTRACT.resolve_q_e4(6, 7000)
        for changes in ({"mode_id": 7}, {"q_e4": 7778},
                        {"keep_count": prepared.envelope.keep_count - 1},
                        {"anchor_action_id": anchor.action_id},
                        {"execution_bundle_sha256": "ab" * 32}):
            with self.assertRaises(X.ContinuousExecutionError, msg=str(changes)):
                edge.process(self._repack(prepared, **changes))
        anchored, edge = self._prepared(q_e4=7000)
        with self.assertRaises(X.ContinuousExecutionError):
            edge.process(self._repack(anchored, anchor_action_id=None))

    def test_edge_refuses_codec_identity_mismatch(self) -> None:
        ue, edge, _ = runtimes(codec=FakeCodec(corrupt_q_e4=True))
        prepared = ue.prepare(CONTRACT.resolve_q_e4(2, 8000), object(), frame())
        with self.assertRaises(X.ContinuousExecutionError):
            edge.process(prepared.wire_bytes)

    def test_corrupt_envelopes_fail_closed(self) -> None:
        prepared, edge = self._prepared()
        wire = bytearray(prepared.wire_bytes)
        for index in (6, 20, X.HEADER_V3_BYTES - 1, X.HEADER_V3_BYTES + 3, len(wire) - 1):
            corrupt = bytearray(wire)
            corrupt[index] ^= 0x01
            with self.assertRaises(X.ContinuousExecutionError):
                edge.process(bytes(corrupt))
        for bad in (bytes(wire[:40]), bytes(wire[:-1]), b"SFD1" + bytes(wire[4:])):
            with self.assertRaises(X.ContinuousExecutionError):
                edge.process(bad)

    def test_old_envelope_versions_unchanged(self) -> None:
        old = sfd1.pack_envelope(b"inner", action_id=39, sequence_id=5,
                                 capture_timestamp_ns=7)
        self.assertEqual(sfd1.unpack_envelope(old).action_id, 39)
        with self.assertRaises(X.ContinuousExecutionError):
            X.unpack_envelope_v3(old)
        prepared, _ = self._prepared()
        with self.assertRaises(sfd1.EnvelopeError):
            sfd1.unpack_envelope(prepared.wire_bytes)


if __name__ == "__main__":
    unittest.main()
