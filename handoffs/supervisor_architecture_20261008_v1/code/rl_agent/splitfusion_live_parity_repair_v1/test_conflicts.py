"""Original-detector composition, retained geometry and causal-input refusal."""
import copy
import ast
import csv
import math
from collections import defaultdict
from dataclasses import replace
from statistics import median
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_live_target_parity_v1.fixtures import FIXTURES, load_retained_fixture
from . import conflicts as C


class ConservativeConflictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retained, _ = load_retained_fixture(next(iter(FIXTURES)))
        path = A.ROOT / cls.retained.provenance["evidence_paths"]["static_catalogue"]
        if A.sha256(path) != cls.retained.provenance["evidence_sha256"]["static_catalogue"]:
            raise RuntimeError("retained catalogue changed")
        with path.open(newline="") as stream:
            cls.catalogue = list(csv.DictReader(stream))

    def inputs(self):
        frame = dict(self.retained.frame, sample_id="prospective_frame_100", experiment_id="prospective_episode", frame_id=100, timestamp=100.)
        def rebound(rows):
            return [dict(r, **{k: frame[k] for k in ("sample_id", "experiment_id", "frame_id", "timestamp")}) for r in rows]
        cat = copy.deepcopy(self.catalogue)
        return dict(frame=frame, actor_boxes=rebound(self.retained.actor_boxes),
                    actor_visibility=rebound(self.retained.actor_visibility),
                    static_candidates=rebound(self.retained.static_candidates), static_catalogue=cat,
                    prior_static_candidates=[], provenance={"scope": C.SCOPE,
                        "capture_identity": {"frame_id": 100},
                        "complete": dict.fromkeys(C.COMPLETENESS, True),
                        "catalogue_content_sha256": C.catalogue_content_sha256(cat),
                        "history_window": {"start_timestamp": 100., "end_timestamp": 100.}})

    def controlled(self):
        args = self.inputs()
        base = dict(args["static_catalogue"][0], bbox_center_x_m=0., bbox_center_y_m=0.,
                    bbox_center_z_m=1., bbox_extent_x_m=2., bbox_extent_y_m=1.,
                    bbox_extent_z_m=1., bbox_rotation_yaw_deg=0.)
        cat = [dict(base, environment_object_id="current_static_1", carla_environment_object_id="1"),
               dict(base, environment_object_id="current_static_2", carla_environment_object_id="2")]
        frame = args["frame"]
        static = [dict(r, **{k: frame[k] for k in ("sample_id", "experiment_id", "frame_id", "timestamp")},
                       clipped_bbox_x=10., clipped_bbox_y=20., clipped_bbox_w=30., clipped_bbox_h=20.,
                       range_m=10., actor_near_depth_m=9.) for r in cat]
        args.update(actor_boxes=[], actor_visibility=[], static_catalogue=cat, static_candidates=static)
        args["provenance"]["catalogue_content_sha256"] = C.catalogue_content_sha256(cat)
        return args

    def test_retained_geometry_matches_whole_original_detector_and_full_builder(self):
        args = self.inputs()
        actual = C.derive_live_quarantine(**args)
        builder, _, _ = A._original_modules(A.ROOT)
        # Load the entire original function library without its historical
        # import-time scorer setup; call its untouched detectors independently.
        source = A.ROOT / A.SOURCES["reconciliation"][0]
        tree = ast.parse(source.read_text())
        nodes = ast.parse("from __future__ import annotations").body
        nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        namespace = dict(np=np, cv2=cv2, math=math, median=median, defaultdict=defaultdict)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        recon = SimpleNamespace(**namespace)
        rows, secondary, manual = recon.static_static_conflicts(args["static_catalogue"], args["static_candidates"])
        skeys, akeys, drows = builder.compute_train_dynamic_conflicts([args["frame"]], args["actor_boxes"],
            args["actor_visibility"], args["static_candidates"], recon)
        self.assertEqual(actual["evidence"]["static_static_adjudications"], rows)
        self.assertEqual(actual["evidence"]["static_dynamic_adjudications"], drows)
        self.assertEqual(actual["quarantine"]["confirmed_secondary"], secondary)
        self.assertEqual(actual["quarantine"]["manual_static_ids"], manual)
        self.assertEqual(actual["quarantine"]["dynamic_static_keys"], skeys)
        self.assertEqual(actual["quarantine"]["dynamic_actor_keys"], akeys)
        self.assertEqual(actual["quarantine"]["panel_static_keys"], set())
        self.assertFalse(actual["evidence"]["live_qualified"])
        provenance = dict(self.retained.provenance, scope=C.SCOPE, sample_id=args["frame"]["sample_id"],
            capture_identity=args["provenance"]["capture_identity"],
            evidence_sha256={"conflicts": actual["evidence_sha256"]})
        value = replace(self.retained, frame=args["frame"], actor_boxes=args["actor_boxes"],
            actor_visibility=args["actor_visibility"], static_candidates=args["static_candidates"],
            quarantine=actual["quarantine"], provenance=provenance)
        value = replace(value, provenance=dict(provenance, input_content_sha256=A.input_content_sha256(value)))
        target = A.build_complete_v010(value)
        self.assertEqual(target["segmentation"].shape, (432, 768))
        self.assertTrue(set(np.unique(target["segmentation"])) <= {0, 1, 2, 255})
        self.assertTrue(all(r["sample_id"] == args["frame"]["sample_id"] for r in target["tables"]["object_boxes.csv"]))

    def test_ambiguous_static_pair_ignored_until_three_causal_observations(self):
        args = self.controlled()
        first = C.derive_live_quarantine(**args)
        self.assertEqual(first["quarantine"]["manual_static_ids"], {"current_static_1", "current_static_2"})
        self.assertEqual(first["quarantine"]["confirmed_secondary"], set())
        args["prior_static_candidates"] = [dict(row, sample_id=f"prospective_frame_{i}", frame_id=i, timestamp=float(i))
            for i in (98, 99) for row in args["static_candidates"]]
        args["provenance"]["history_window"]["start_timestamp"] = 98.
        later = C.derive_live_quarantine(**args)
        self.assertEqual(later["quarantine"]["confirmed_secondary"], {"current_static_2"})
        self.assertEqual(later["quarantine"]["manual_static_ids"], set())
        self.assertEqual(later["quarantine"]["confirmed_pairs"][0]["common_projected_frames"], 3)

    def test_dynamic_same_physical_and_ambiguous_quarantined_but_distinct_not(self):
        args = self.controlled()
        args["static_catalogue"] = args["static_catalogue"][:1]
        args["static_candidates"] = args["static_candidates"][:1]
        args["provenance"]["catalogue_content_sha256"] = C.catalogue_content_sha256(args["static_catalogue"])
        actor = dict(args["frame"], label="vehicle", gt_actor_id="7", object_world_x=0., object_world_y=0.,
                     object_world_z=1., gt_size_x_m=4., gt_size_y_m=2., gt_size_z_m=2., object_yaw_deg=0.)
        vis = dict(actor, clipped_bbox_x=10., clipped_bbox_y=20., clipped_bbox_w=30., clipped_bbox_h=20.)
        args.update(actor_boxes=[actor], actor_visibility=[vis])
        for x, expected, quarantine in ((0., "CONFIRMED_SAME_PHYSICAL_OBJECT", True),
                                        (1.5, "MANUAL_REVIEW_REQUIRED", True),
                                        (10., "REJECTED_DISTINCT_OCCLUSION", False)):
            actor["object_world_x"] = x
            result = C.derive_live_quarantine(**args)
            self.assertEqual(result["evidence"]["static_dynamic_adjudications"][0]["adjudication"], expected)
            self.assertEqual(bool(result["quarantine"]["dynamic_actor_keys"]), quarantine)
            self.assertEqual(bool(result["quarantine"]["dynamic_static_keys"]), quarantine)

    def test_historical_missing_foreign_noncaual_and_tampered_inputs_refused(self):
        baseline = self.inputs()
        variations = []
        bad = copy.deepcopy(baseline); bad["provenance"]["scope"] = "HISTORICAL_REGISTERED_V010"; variations.append(bad)
        bad = copy.deepcopy(baseline); bad["frame"]["sample_id"] = "canonical_v3_old"; variations.append(bad)
        bad = copy.deepcopy(baseline); bad["provenance"]["complete"].pop("catalogue_complete"); variations.append(bad)
        bad = copy.deepcopy(baseline); bad["provenance"]["capture_identity"]["frame_id"] += 1; variations.append(bad)
        bad = copy.deepcopy(baseline); bad["static_catalogue"][0]["bbox_center_x_m"] = 1234.; variations.append(bad)
        bad = copy.deepcopy(baseline); bad["actor_visibility"] = []; variations.append(bad)
        bad = copy.deepcopy(baseline); bad["static_candidates"][0]["timestamp"] += 1.; variations.append(bad)
        bad = copy.deepcopy(baseline); bad["static_candidates"][0]["frame_id"] += 1; variations.append(bad)
        bad = copy.deepcopy(baseline); bad["prior_static_candidates"] = [dict(bad["static_candidates"][0], sample_id="future", frame_id=101, timestamp=101.)]; variations.append(bad)
        for index, args in enumerate(variations):
            with self.subTest(case=index), self.assertRaises(C.ConflictInputError):
                C.derive_live_quarantine(**args)

    def test_complete_empty_scene_explicitly_allowed_and_deterministic(self):
        args = self.inputs()
        args.update(actor_boxes=[], actor_visibility=[], static_candidates=[], static_catalogue=[])
        args["provenance"]["catalogue_content_sha256"] = C.catalogue_content_sha256([])
        first, second = C.derive_live_quarantine(**args), C.derive_live_quarantine(**args)
        self.assertEqual(first, second)
        self.assertEqual(set(first["quarantine"]), set(A.QUARANTINE_KEYS))
        self.assertFalse(any(first["quarantine"].values()))

    def _compare_prefixes(self, template, frames, *, perturb=None):
        cache = C.IncrementalConflictState(template["static_catalogue"], max_frames=frames)
        history, prior_chain = [], cache.chain
        for index in range(frames):
            args = copy.deepcopy(template)
            frame = args["frame"]
            frame.update(sample_id=f"new_frame_{index}", frame_id=index+1, timestamp=float(index+1))
            for name in ("actor_boxes", "actor_visibility", "static_candidates"):
                for row in args[name]:
                    row.update({k: frame[k] for k in ("sample_id", "frame_id", "timestamp")})
            if perturb is not None:
                perturb(index, args)
            args["provenance"]["capture_identity"]["frame_id"] = index+1
            args["provenance"]["history_window"] = {"start_timestamp": 1., "end_timestamp": float(index+1)}
            args["prior_static_candidates"] = history
            expected = C.derive_live_quarantine(**args)
            observed = cache.derive(**{k: v for k,v in args.items() if k not in ("static_catalogue", "prior_static_candidates")})
            with self.subTest(prefix=index+1):
                self.assertEqual(observed["quarantine"], expected["quarantine"])
                self.assertEqual(observed["evidence"]["static_static_adjudications"], expected["evidence"]["static_static_adjudications"])
                self.assertEqual(observed["evidence"]["static_dynamic_adjudications"], expected["evidence"]["static_dynamic_adjudications"])
                self.assertEqual(observed["evidence"]["causal_static_history"]["previous_sha256"], prior_chain)
                self.assertEqual(observed["evidence"]["causal_static_history"]["frames"], index+1)
                self.assertFalse(observed["evidence"]["causal_static_history"]["history_discarded"])
            prior_chain = cache.chain
            history.extend(args["static_candidates"])
        with self.assertRaisesRegex(C.ConflictInputError, "bound"):
            cache.derive(**{k: v for k,v in args.items() if k not in ("static_catalogue", "prior_static_candidates")})

    def test_incremental_retained_geometry_every_prefix_equals_full_original(self):
        def vary(index, args):
            # Original observations, varying projection/depth/range and
            # subsets while catalogue geometry remains the same physical map.
            rng = np.random.default_rng(index)
            rows = args["static_candidates"]
            args["static_candidates"] = [row for row in rows if rng.random() > .15]
            for row in args["static_candidates"]:
                row["clipped_bbox_x"] = float(row["clipped_bbox_x"]) + float(rng.integers(-5,6))
                row["range_m"] = float(row["range_m"]) + float(rng.random())
        self._compare_prefixes(self.inputs(), 24, perturb=vary)

    def test_incremental_duplicate_adjudication_matches_original_every_prefix(self):
        def vary(index,args):
            # Covers odd/even medians, manual→confirmed→manual, tie samples,
            # and previously observed pair absent from the current frame.
            if index in (4,5):
                args["static_candidates"] = []
            elif index % 5 in (1,2):
                args["static_candidates"][1]["range_m"] += 1.5
                args["static_candidates"][1]["clipped_bbox_x"] += 10.
        self._compare_prefixes(self.controlled(), 20, perturb=vary)

    def test_incremental_far_pair_later_admission_uses_full_history(self):
        args = self.controlled()
        args["static_catalogue"][1]["bbox_center_x_m"] = 20.
        args["static_candidates"][1]["bbox_center_x_m"] = 20.
        args["provenance"]["catalogue_content_sha256"] = C.catalogue_content_sha256(args["static_catalogue"])
        def vary(index,args):
            if index == 0:
                args["static_candidates"][1]["range_m"] += 3.
            else:
                args["static_candidates"][1]["clipped_bbox_x"] += 100.
                args["static_candidates"][1]["range_m"] += .5
        self._compare_prefixes(args, 6, perturb=vary)

    def test_incremental_rejection_does_not_advance_state(self):
        args = self.controlled()
        cache = C.IncrementalConflictState(args["static_catalogue"], max_frames=2)
        kwargs = {k:v for k,v in args.items() if k not in ("static_catalogue", "prior_static_candidates")}
        before = cache.chain
        bad = copy.deepcopy(kwargs); bad["static_candidates"][0]["timestamp"] += 1
        with self.assertRaises(C.ConflictInputError): cache.derive(**bad)
        self.assertEqual(cache.chain,before)
        self.assertEqual(cache.frame_count,0)
        result = cache.derive(**kwargs)
        with self.assertRaises(C.ConflictInputError): cache.derive(**kwargs)
        self.assertEqual(cache.chain,result["evidence"]["causal_static_history"]["chain_sha256"])
        self.assertEqual(cache.frame_count,1)


if __name__ == "__main__":
    unittest.main()
