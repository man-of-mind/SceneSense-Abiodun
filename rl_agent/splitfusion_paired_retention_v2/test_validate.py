"""verify_archive_v2: round trip, tamper, plan binding, incomplete and identity-mismatch refusals."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from . import fixtures as F, gt_adapter as G, observer as OB, plan as P, validate as V
from .test_observer import SESSION, decisions


def make_config(**kwargs):
    kwargs.setdefault("planned_frames", 24)
    kwargs.setdefault("target_records", 3)
    kwargs.setdefault("min_free_disk_bytes", 0)
    plan = P.build_plan(plan_id="validate", **kwargs)
    return P.RetentionConfigV2(enabled=True, plan=plan, plan_sha256=P.plan_sha256(plan))


def finalize(directory, events, cfg=None, fail=None):
    cfg = cfg or make_config()
    observer = OB.make_observer(cfg)
    tensors = {}
    for i, (kind, planned, prepared, _) in enumerate(events):
        token = observer.begin(planned)
        if fail is not None and fail(i):
            observer.end(token, None)
            continue
        tensors[i] = torch.full(P.INPUT_SHAPE, float(i))
        observer.observe_input(tensors[i], "cuda:0")
        observer.end(token, prepared)
    return cfg, observer.finalize(directory, workers=F.worker_done())


class ValidateTests(unittest.TestCase):
    def test_round_trip_and_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r"
            cfg, manifest = finalize(path, decisions(12))
            self.assertEqual(manifest["ordinal_status"], {"2": "RETAINED_COMPLETE", "6": "RETAINED_COMPLETE",
                                                          "10": "RETAINED_COMPLETE"})
            self.assertEqual(manifest["run_length"], "EQUAL_TO_PLANNED")
            verified = V.verify_archive_v2(path, expected_manifest_sha256=manifest["manifest_sha256"],
                                           expected_plan_sha256=cfg.plan_sha256)
            self.assertEqual(len(verified["records"]), 3)
            self.assertEqual(verified["records"][0]["input_bytes"], 9633792)
            self.assertEqual(verified["records"][0]["join_key"]["decision_ordinal"], 2)
            with self.assertRaises(FileExistsError):
                finalize(path, decisions(12))

    def test_plan_digest_mismatch_and_member_tamper_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r"
            cfg, manifest = finalize(path, decisions(12))
            other = P.plan_sha256(P.build_plan(plan_id="validate", planned_frames=26, target_records=3,
                                               min_free_disk_bytes=0))
            with self.assertRaisesRegex(V.RetentionAdmissionError, "plan digest"):
                V.verify_archive_v2(path, expected_manifest_sha256=manifest["manifest_sha256"], expected_plan_sha256=other)
            with (path / manifest["records"][0]["wire_file"]).open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(V.RetentionAdmissionError, "modified"):
                V.verify_archive_v2(path, expected_manifest_sha256=manifest["manifest_sha256"],
                                    expected_plan_sha256=cfg.plan_sha256)

    def test_rebound_semantic_tampering_refused(self):
        mutations = [
            lambda m: m["ordinal_status"].update({"2": "NOT_REACHED"}),
            lambda m: m["records"][0]["join_key"].update(frame_id=1),
            lambda m: m["records"][0]["gt_identity"].update(frame_id=1),
            lambda m: m["records"][0]["frame_context"].update(sequence_id=99),
            lambda m: m["records"][0]["action"].update(q_e4=1),
            lambda m: m["records"][0].update(transfer_device="cpu"),
            lambda m: m["counters"].update(skipped_off_sample=0),
            lambda m: m["drain"]["0"].update(alive=True),
            lambda m: m["plan"].update(ordinals=[0, 6, 10]),
            lambda m: m.update(records=m["records"][1:]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "r"
                cfg, manifest = finalize(path, decisions(12))
                document = json.loads((path / "MANIFEST.json").read_bytes())
                mutation(document)
                raw = P.canonical(document)
                (path / "MANIFEST.json").write_bytes(raw)
                with self.assertRaises(V.RetentionAdmissionError):
                    V.verify_archive_v2(path, expected_manifest_sha256=manifest["manifest_sha256"],
                                        expected_plan_sha256=cfg.plan_sha256)
                with self.assertRaises(V.RetentionAdmissionError):      # even when rebound to the new digest
                    V.verify_archive_v2(path, expected_manifest_sha256=P.sha256_bytes(raw),
                                        expected_plan_sha256=cfg.plan_sha256)

    def test_incomplete_record_fails_strict_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r"
            cfg, manifest = finalize(path, decisions(12), fail=lambda i: i == 6)
            self.assertEqual(manifest["ordinal_status"]["6"], "INCOMPLETE_PRODUCTION_EXCEPTION")
            with self.assertRaisesRegex(V.RetentionAdmissionError, "incomplete retention refused"):
                V.verify_archive_v2(path, expected_manifest_sha256=manifest["manifest_sha256"],
                                    expected_plan_sha256=cfg.plan_sha256)
            relaxed = V.verify_archive_v2(path, expected_manifest_sha256=manifest["manifest_sha256"],
                                          expected_plan_sha256=cfg.plan_sha256, strict=False)
            self.assertEqual(relaxed["status_counts"], {"RETAINED_COMPLETE": 2, "INCOMPLETE_PRODUCTION_EXCEPTION": 1})

    def test_shorter_run_not_reached_is_declared_and_admissible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r"
            cfg, manifest = finalize(path, decisions(8))
            self.assertEqual(manifest["run_length"], "SHORTER_THAN_PLANNED")
            self.assertEqual(manifest["ordinal_status"]["10"], "NOT_REACHED")
            V.verify_archive_v2(path, expected_manifest_sha256=manifest["manifest_sha256"],
                                expected_plan_sha256=cfg.plan_sha256)

    def test_join_identity_mismatches_never_become_retained(self):
        events = decisions(12)
        _, planned, prepared, _ = events[2]
        bad_gt = dict(prepared.gt_identity, frame_id=prepared.gt_identity["frame_id"] + 4)
        events[2] = ("decision", planned, type(prepared)(wire=prepared.wire, identity=prepared.identity,
                                                         gt_identity=bad_gt), None)
        _, planned6, prepared6, _ = events[6]
        planned6.q_e4 += 1                                               # plan/wire action drift
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r"
            cfg, manifest = finalize(path, events)
            self.assertEqual(manifest["ordinal_status"]["2"], "INCOMPLETE_GT_WIRE_IDENTITY_MISMATCH")
            self.assertEqual(manifest["ordinal_status"]["6"], "INCOMPLETE_PLAN_WIRE_IDENTITY_MISMATCH")
            with self.assertRaises(V.RetentionAdmissionError):
                V.verify_archive_v2(path, expected_manifest_sha256=manifest["manifest_sha256"],
                                    expected_plan_sha256=cfg.plan_sha256)

    def test_storage_cap_is_checked(self):
        cfg = make_config(max_storage_bytes=2 * P.INPUT_BYTES + 8192)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r"
            cfg, manifest = finalize(path, decisions(12), cfg=cfg)
            self.assertEqual(manifest["status_counts"], {"RETAINED_COMPLETE": 2, "INCOMPLETE_STORAGE_CAP": 1})
            self.assertLessEqual(manifest["storage_bytes"], cfg.plan["bounds"]["max_storage_bytes"])

    def test_edge_target_hash_domain_must_be_declared(self):
        packet = "1" * 64
        timing = {"target_bundle_sha256": packet, "scoring_packet_sha256": packet}
        with self.assertRaisesRegex(G.GtAdapterError, "UNDECLARED"):
            G.edge_target_hash_crosscheck(timing, declared_domain="GUESS")
        self.assertEqual(G.edge_target_hash_crosscheck(timing, declared_domain="V11_SCORING_PACKET_SHA256"),
                         "V11_SCORING_PACKET_SHA256")
        with self.assertRaisesRegex(G.GtAdapterError, "NOT_COMPLETE_BUNDLE"):
            G.edge_target_hash_crosscheck(timing, declared_domain="COMPLETE_TARGET_BUNDLE_MANIFEST_SHA256",
                                          complete_bundle_manifest_sha256="2" * 64)
        with self.assertRaisesRegex(G.GtAdapterError, "AMBIGUOUS"):
            G.edge_target_hash_crosscheck(timing, declared_domain="V11_SCORING_PACKET_SHA256",
                                          complete_bundle_manifest_sha256=packet)


if __name__ == "__main__":
    unittest.main()
