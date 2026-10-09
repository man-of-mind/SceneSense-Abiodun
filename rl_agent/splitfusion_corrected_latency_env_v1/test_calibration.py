"""Rejection of missing/incompatible calibration bindings, uncalibrated terms and the launch gate (CPU-light)."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import calibration as K
from . import chains as CH

BINDING = K.PACKAGE_DIR / "bindings" / "LIVE100_REGRESSION_BINDING_V2.json"


def _doc():
    return json.loads(BINDING.read_text())


def _relaxed(doc=None):
    """Test-only: mode-level strata and wide scene support, to reach the later checks deterministically."""
    doc = copy.deepcopy(doc or _doc())
    doc["policy"]["action_match"]["q_tolerance_e4"] = 9800
    doc["policy"]["scene_conditioning"]["support_margin_fraction"] = 50.0
    return doc


class BindingRejection(unittest.TestCase):
    def assertRejected(self, doc, fragment):
        with self.assertRaises(K.CalibrationBindingError) as raised:
            K.CalibrationSetV1(doc)
        self.assertIn(fragment, str(raised.exception))

    def test_registered_fixture_loads(self):
        cal = K.CalibrationSetV1(_doc())
        self.assertEqual(len(cal.chains), 46)          # 48 decisions minus the 2 cold ones

    def test_missing_binding_file_and_non_json(self):
        with self.assertRaises(K.CalibrationBindingError):
            K.load_binding(Path("/nonexistent/binding.json"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.json"
            path.write_text("{not json")
            with self.assertRaises(K.CalibrationBindingError):
                K.load_binding(path)

    def test_superseded_v1_and_missing_service_record(self):
        doc = _doc()
        doc["schema"] = "splitfusion.corrected_latency_env.calibration_binding.v1"
        self.assertRejected(doc, "superseded")
        doc = _doc()
        doc["composition_model"] = "RESIDUAL_V1"
        self.assertRejected(doc, "composition model")
        doc = _doc()
        del doc["sources"][0]["files"]["service_frames"]
        self.assertRejected(doc, "path+sha256")
        doc = _doc()
        doc["sources"][0]["files"]["service_frames"]["sha256"] = "0" * 64
        self.assertRejected(doc, "differs from bound")

    def test_schema_version_and_deadline(self):
        for key, value, fragment in (("schema", "other.v1", "schema"),
                                     ("candidate_version", "splitfusion_corrected_latency_env_v0", "candidate"),
                                     ("deadline_ms", 171, "deadline")):
            doc = _doc()
            doc[key] = value
            self.assertRejected(doc, fragment)

    def test_drifted_or_missing_evidence(self):
        doc = _doc()
        doc["sources"][0]["files"]["boundaries"]["sha256"] = "0" * 64
        self.assertRejected(doc, "differs from bound")
        doc = _doc()
        doc["sources"][0]["files"]["per_frame"]["path"] += ".missing"
        self.assertRejected(doc, "missing file")
        doc = _doc()
        del doc["sources"][0]["files"]["evidence"]
        self.assertRejected(doc, "path+sha256")

    def test_incompatible_pipeline_identity(self):
        doc = _doc()
        doc["sources"][0]["pipeline_identity"] = dict(doc["pipeline_identity"], edge_image_id="sha256:other")
        self.assertRejected(doc, "pipeline identity")

    def test_wrong_transport_artifact_and_unknown_kind(self):
        doc = _doc()
        doc["policy"]["uplink"]["transport_artifact_sha256"] = "1" * 64
        self.assertRejected(doc, "transport artifact")
        doc = _doc()
        doc["sources"][0]["kind"] = "LEGACY_CHAIN_CSV"
        self.assertRejected(doc, "unknown kind")

    def test_validation_scope_cannot_exceed_its_evidence(self):
        doc = _doc()
        doc["policy"]["uplink"]["validation"]["scope"]["mcs_range"] = [5, 28]
        self.assertRejected(doc, "MCS scope exceeds")
        doc = _doc()
        doc["policy"]["uplink"]["validation"]["scope"]["max_median_abs_error_ms"] = 1.0
        self.assertRejected(doc, "exceeds the declared tolerance")

    def test_calibrated_fallback_needs_evidence(self):
        doc = _doc()
        doc["policy"]["fallback"] = {"status": "CALIBRATED"}
        self.assertRejected(doc, "fallback")

    def test_policy_fields_required(self):
        doc = _doc()
        del doc["policy"]["scene_conditioning"]
        self.assertRejected(doc, "scene_conditioning")
        doc = _doc()
        doc["policy"]["action_match"]["rule"] = "NEAREST_ACTION"
        self.assertRejected(doc, "EXACT_MODE_Q")


class InlineSource(unittest.TestCase):
    """The format the fixed-action screen will deliver: context inline in each row, mapped explicitly."""

    def _write(self, tmp, drop=None):
        cal = K.CalibrationSetV1(_doc())
        rows = []
        for chain in cal.chains.values():
            row = {name: chain.instants[name] for name in CH.INSTANTS}
            c = chain.context
            row.update(frame_id=chain.frame_id, scoring_request_op=chain.scoring_request_op, q_perc=chain.q_perc,
                       controller_resolution=chain.controller_resolution,
                       executed={"mode_id": c.mode_id, "q_e4": c.q_e4, "family": c.family, "quantizer": c.quantizer},
                       payload_bytes=c.wire_bytes - 8 * 0, prior_ul_mcs=c.prior_ul_mcs, backlog=c.backlog_bytes,
                       si=c.camera_si, p40=c.radar_p40, profile=c.network_profile, cold=c.cold)
            rows.append(row)
        path = Path(tmp) / "rows.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        field_map = {"mode_id": "executed.mode_id", "q_e4": "executed.q_e4", "family": "executed.family",
                     "quantizer": "executed.quantizer", "payload_bytes": "payload_bytes",
                     "prior_ul_mcs": "prior_ul_mcs", "backlog_bytes": "backlog", "camera_si": "si",
                     "radar_p40": "p40", "network_profile": "profile", "cold": "cold"}
        if drop:
            field_map.pop(drop)
        doc = _doc()
        doc["sources"][0] = dict(doc["sources"][0], kind="BOUNDARIES_WITH_INLINE_CONTEXT_V1", field_map=field_map,
                                 files={"boundaries": {"path": str(path), "sha256": K.sha256_file(path)},
                                        "service_frames": doc["sources"][0]["files"]["service_frames"]})
        return doc

    def test_inline_rows_load_and_unmapped_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(K.CalibrationSetV1(self._write(tmp)).chains), 46)
            with self.assertRaises(K.CalibrationBindingError):
                K.CalibrationSetV1(self._write(tmp, drop="prior_ul_mcs"))


class UncalibratedTerms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strict = K.CalibrationSetV1(_doc())
        cls.relaxed = K.CalibrationSetV1(_relaxed())
        cls.chain = next(c for c in cls.strict.chains.values() if c.context.prior_ul_mcs == 26
                         and 96_627 <= c.context.wire_bytes <= 403_480)

    def _draw(self, cal, **overrides):
        c = self.chain.context
        kwargs = dict(mode_id=c.mode_id, q_e4=c.q_e4, profile="FAVORABLE_STABLE", camera_si=c.camera_si,
                      radar_p40=c.radar_p40, prior_ul_mcs=c.prior_ul_mcs, backlog_bytes=c.backlog_bytes,
                      wire_bytes=c.wire_bytes, rng=random.Random(1))
        kwargs.update(overrides)
        return cal.draw(**kwargs)

    def assertUncalibrated(self, term, cal, **overrides):
        with self.assertRaises(CH.UncalibratedTermError) as raised:
            self._draw(cal, **overrides)
        self.assertEqual(raised.exception.term, term)

    def test_measured_point_is_calibrated(self):
        draw = self._draw(self.strict)
        self.assertEqual(draw.chain_id, self.chain.chain_id)
        self.assertAlmostEqual(draw.uplink_delta_ms, 0.0, places=9)

    def test_unmeasured_action_profile_and_scene(self):
        self.assertUncalibrated("action_stratum", self.strict, mode_id=3)
        self.assertUncalibrated("action_stratum", self.strict, q_e4=self.chain.context.q_e4 + 1)  # 7 points != space
        self.assertUncalibrated("action_stratum", self.strict, profile="ADVERSE_STABLE")
        self.assertUncalibrated("scene_conditioning", self.strict, camera_si=self.chain.context.camera_si + 1.0)

    def test_unlabeled_channel_and_uplink_scope(self):
        self.assertUncalibrated("channel_profile", self.relaxed, profile=None, prior_ul_mcs=5)
        self.assertUncalibrated("E_uplink", self.relaxed, prior_ul_mcs=20)            # validated MCS is 24-28
        self.assertUncalibrated("E_uplink", self.relaxed, wire_bytes=500_000)
        doc = _relaxed()
        doc["policy"]["uplink"]["validation"]["scope"]["profiles"] = ["MID_VARIABLE"]
        doc["policy"]["uplink"]["validation"]["scope"]["mcs_range"] = [24, 28]
        self.assertUncalibrated("E_uplink", K.CalibrationSetV1(doc))
        doc = _relaxed()
        del doc["policy"]["uplink"]["validation"]
        self.assertUncalibrated("E_uplink", K.CalibrationSetV1(doc))

    def test_uplink_transfer_moves_only_the_uplink(self):
        base = self._draw(self.strict)
        moved = self._draw(self.strict, wire_bytes=self.chain.context.wire_bytes - 50_000)
        self.assertLess(moved.uplink_delta_ms, 0)
        self.assertAlmostEqual(moved.vector.E, base.vector.E + moved.uplink_delta_ms, places=9)
        for stage in ("A", "B", "C", "D", "F", "H", "G", "R"):
            self.assertEqual(getattr(moved.vector, stage), getattr(base.vector, stage))

    def test_draws_are_deterministic_per_rng(self):
        a = [self._draw(self.relaxed, rng=random.Random(5)).chain_id for _ in range(3)]
        b = [self._draw(self.relaxed, rng=random.Random(5)).chain_id for _ in range(3)]
        self.assertEqual(a, b)


class LaunchGate(unittest.TestCase):
    def test_launch_is_blocked_without_approval(self):
        cal = K.CalibrationSetV1(_doc())
        self.assertFalse(K.APPROVAL_PATH.exists())
        with self.assertRaises(K.LaunchBlockedError):
            K.require_launch_approval(cal)
        from .collector import CorrectedRun4CollectorV1
        with self.assertRaises(K.LaunchBlockedError):
            CorrectedRun4CollectorV1(calibration=cal, seed=1, mode="CALIBRATED", shared_sources={})

    def test_fixture_binding_lists_its_blockers(self):
        blockers = K.CalibrationSetV1(_doc()).launch_blockers()
        for fragment in ("PROFILES_UNMEASURED", "UPLINK_TRANSFER_UNVALIDATED", "CONTINUOUS_Q_EXACT_POINTS_ONLY",
                         "MODES_UNMEASURED", "FALLBACK", "BINDING_IS_A_REGRESSION_FIXTURE"):
            self.assertTrue(any(b.startswith(fragment) for b in blockers), fragment)

    def test_approval_must_name_binding_source_and_every_blocker(self):
        cal = K.CalibrationSetV1(_doc())
        good = {"calibration_binding_sha256": cal.binding_sha256, "candidate_source_digest": K.candidate_source_digest(),
                "approved_by": "test", "approved_at": "2026-10-05", "waived_blockers": cal.launch_blockers()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "APPROVAL.json"
            with mock.patch.object(K, "APPROVAL_PATH", path):
                for change in ({"calibration_binding_sha256": "0" * 64}, {"candidate_source_digest": "0" * 64},
                               {"approved_by": ""}, {"waived_blockers": cal.launch_blockers()[1:]}):
                    path.write_text(json.dumps(dict(good, **change)))
                    with self.assertRaises(K.LaunchBlockedError):
                        K.require_launch_approval(cal)
                path.write_text(json.dumps(good))
                K.require_launch_approval(cal)          # only an explicit, complete approval opens the gate


if __name__ == "__main__":
    unittest.main()
