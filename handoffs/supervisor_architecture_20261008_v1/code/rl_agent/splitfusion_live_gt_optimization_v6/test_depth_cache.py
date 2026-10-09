"""Independent numerical and adversarial CPU proofs for request-local depth.

These tests never create a CARLA client, worker process, or CUDA context. The
numerical oracles are the unchanged AVO and original visibility decoders.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from data_collection.route_b_perception_v3 import visibility_v1 as visibility
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as avo
from .depth_cache import DepthReuse, DepthReuseError


def fixture(*, frame=101, raw=None, buffer=None):
    if raw is None:
        raw = np.random.default_rng(193).integers(0, 256, (5, 7, 4), dtype=np.uint8)
    timestamp = frame / 20.0
    image = SimpleNamespace(raw_data=raw.tobytes() if buffer is None else buffer,
        width=raw.shape[1], height=raw.shape[0], frame=frame, timestamp=timestamp,
        transform=SimpleNamespace(get_matrix=lambda: np.eye(4).tolist(),
                                  get_inverse_matrix=lambda: np.eye(4).tolist()))
    identity = dict(run_id="depth-test", cell_id="cpu", stream_id="ue", frame_id=frame,
                    capture_timestamp_ns=round(timestamp * 1e9), action_id=None, profile_id=None)
    return identity, image, raw


class DecoderEquivalenceTests(unittest.TestCase):
    def test_every_24bit_encoded_value_matches_both_unchanged_decoders_bitwise(self):
        # 256 bounded chunks exhaust all 16,777,216 possible RGB depth codes.
        # Alpha varies across its full domain but remains irrelevant to depth.
        codes = np.arange(65536, dtype=np.uint32).reshape(256, 256)
        raw = np.empty((256, 256, 4), dtype=np.uint8)
        raw[..., 1] = codes >> 8
        raw[..., 2] = codes & 255
        for blue in range(256):
            raw[..., 0] = blue
            raw[..., 3] = (codes + blue) & 255
            image = SimpleNamespace(raw_data=raw.tobytes(), height=256, width=256)
            first, second = avo.decode_depth(image), visibility.decode_depth_bgra(raw)
            self.assertEqual(first.dtype, np.dtype(np.float32))
            self.assertEqual(second.dtype, np.dtype(np.float32))
            self.assertTrue(np.array_equal(first.view(np.uint32), second.view(np.uint32)),
                            f"float32 depth bits differ for blue={blue}")


class DepthReuseTests(unittest.TestCase):
    def controller(self):
        first = mock.Mock(wraps=avo.decode_depth)
        second = mock.Mock(wraps=visibility.decode_depth_bgra)
        return DepthReuse(first, second), first, second

    def assert_cleared(self, controller, *, completed):
        self.assertFalse(controller.active)
        self.assertFalse(controller.retained)
        self.assertIs(controller.last_stats["completed"], completed)

    def test_lazy_single_decode_and_three_exact_copy_reuses(self):
        identity, image, raw = fixture()
        controller, first, second = self.controller()
        self.assertFalse(controller.active)
        self.assertFalse(controller.retained)
        with controller.request_scope(identity=identity, depth_image=image):
            self.assertTrue(controller.active)
            first.assert_not_called()
            second.assert_not_called()
            depth = controller.decode_image(image)
            for candidate in (raw, raw.copy(), np.frombuffer(image.raw_data, np.uint8).reshape(raw.shape)):
                self.assertIs(controller.decode_bgra(candidate), depth)
            np.testing.assert_array_equal(depth.view(np.uint32),
                                          visibility.decode_depth_bgra(raw).view(np.uint32))
        self.assert_cleared(controller, completed=True)
        self.assertEqual(first.call_count, 1)
        second.assert_not_called()
        for key, expected in (("decodes", 1), ("reuses", 3), ("calls", 4), ("frame_id", 101)):
            self.assertEqual(controller.last_stats[key], expected, key)

    def test_decoded_output_cannot_be_written_or_made_writeable(self):
        identity, image, raw = fixture()
        controller, _, _ = self.controller()
        with controller.request_scope(identity=identity, depth_image=image):
            depth = controller.decode_image(image)
            self.assertEqual(depth.dtype, np.dtype(np.float32))
            self.assertEqual(depth.shape, raw.shape[:2])
            self.assertFalse(depth.flags.writeable)
            self.assertFalse(np.shares_memory(depth, raw))
            with self.assertRaises(ValueError):
                depth[0, 0] = 123.0
            with self.assertRaises(ValueError):
                depth.setflags(write=True)
            with self.assertRaises(ValueError):
                depth.view().setflags(write=True)
            base = depth.base
            while isinstance(base, np.ndarray):
                with self.assertRaises(ValueError):
                    base.setflags(write=True)
                base = base.base
            self.assertIs(controller.decode_bgra(raw), depth)
        self.assert_cleared(controller, completed=True)

    def test_edge_values_and_all_channel_boundaries_are_bit_exact(self):
        samples = [[0, 0, 0, 0], [0, 0, 1, 255], [0, 0, 255, 1],
                   [0, 1, 0, 127], [0, 255, 255, 128], [1, 0, 0, 254],
                   [127, 255, 255, 0], [128, 0, 0, 255], [255, 255, 254, 1],
                   [255, 255, 255, 255]]
        raw = np.array(samples, dtype=np.uint8).reshape(2, 5, 4)
        identity, image, raw = fixture(raw=raw)
        controller, _, _ = self.controller()
        with controller.request_scope(identity=identity, depth_image=image):
            actual = controller.decode_image(image)
            for expected in (avo.decode_depth(image), visibility.decode_depth_bgra(raw)):
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(float(actual[0, 0]), 0.)
            self.assertEqual(float(actual[-1, -1]), 1000.)

    def test_first_last_pixel_changes_in_every_channel_fail_closed(self):
        for pixel in ((0, 0), (4, 6)):
            for channel in range(4):
                with self.subTest(pixel=pixel, channel=channel):
                    identity, image, raw = fixture()
                    changed = raw.copy()
                    changed[pixel + (channel,)] ^= np.uint8(1)
                    controller, _, second = self.controller()
                    with self.assertRaises(DepthReuseError):
                        with controller.request_scope(identity=identity, depth_image=image):
                            controller.decode_image(image)
                            controller.decode_bgra(changed)
                    self.assert_cleared(controller, completed=False)
                    second.assert_not_called()

    def test_alpha_only_change_rejects_even_when_decoded_values_are_identical(self):
        identity, image, raw = fixture()
        changed = raw.copy()
        changed[-1, -1, 3] ^= np.uint8(255)
        np.testing.assert_array_equal(visibility.decode_depth_bgra(raw),
                                      visibility.decode_depth_bgra(changed))
        controller, _, second = self.controller()
        with self.assertRaises(DepthReuseError):
            with controller.request_scope(identity=identity, depth_image=image):
                controller.decode_image(image)
                controller.decode_bgra(changed)
        self.assert_cleared(controller, completed=False)
        second.assert_not_called()

    def test_same_previously_matching_array_is_rechecked_after_mutation(self):
        identity, image, raw = fixture()
        controller, _, second = self.controller()
        with self.assertRaises(DepthReuseError):
            with controller.request_scope(identity=identity, depth_image=image):
                controller.decode_image(image)
                controller.decode_bgra(raw)
                raw[-1, -1, 0] ^= np.uint8(1)
                controller.decode_bgra(raw)
        self.assert_cleared(controller, completed=False)
        second.assert_not_called()

    def test_source_replacement_and_readonly_mutable_backing_are_rechecked(self):
        for decoder in ("image", "bgra"):
            for readonly_view in (False, True):
                with self.subTest(decoder=decoder, readonly_view=readonly_view):
                    identity, image, raw = fixture()
                    owner = bytearray(image.raw_data)
                    if readonly_view:
                        image.raw_data = memoryview(owner).toreadonly()
                    controller, _, second = self.controller()
                    with self.assertRaises(DepthReuseError):
                        with controller.request_scope(identity=identity, depth_image=image):
                            controller.decode_image(image)
                            owner[-1] ^= 1
                            if not readonly_view:
                                image.raw_data = bytes(owner)
                            if decoder == "image":
                                controller.decode_image(image)
                            else:
                                controller.decode_bgra(raw)
                    self.assert_cleared(controller, completed=False)
                    second.assert_not_called()

    def test_source_metadata_mutation_fails_before_reuse(self):
        for name, value in (("frame", 102), ("timestamp", 999.), ("width", 8), ("height", 6)):
            with self.subTest(field=name):
                identity, image, raw = fixture()
                controller, _, second = self.controller()
                with self.assertRaises(DepthReuseError):
                    with controller.request_scope(identity=identity, depth_image=image):
                        controller.decode_image(image)
                        setattr(image, name, value)
                        controller.decode_bgra(raw)
                self.assert_cleared(controller, completed=False)
                second.assert_not_called()

    def test_request_identity_and_bounded_native_geometry_fail_closed(self):
        for changed in (dict(frame=102), dict(width=0), dict(height=-1),
                        dict(width=1281), dict(height=721)):
            with self.subTest(changed=changed):
                identity, image, _ = fixture()
                for name, value in changed.items():
                    setattr(image, name, value)
                controller, first, second = self.controller()
                with self.assertRaises(DepthReuseError):
                    with controller.request_scope(identity=identity, depth_image=image):
                        controller.decode_image(image)
                self.assertFalse(controller.active)
                self.assertFalse(controller.retained)
                first.assert_not_called()
                second.assert_not_called()

    def test_noncontiguous_and_subclass_arrays_use_original_without_reuse(self):
        class Untrusted(np.ndarray):
            def tobytes(self, *args, **kwargs):
                raise AssertionError("subclass byte override must not decide reuse")

        identity, image, raw = fixture()
        candidates = [np.asfortranarray(raw), raw[:, ::-1], raw[::-1], raw.view(Untrusted)]
        controller, first, second = self.controller()
        with controller.request_scope(identity=identity, depth_image=image):
            cached = controller.decode_image(image)
            for candidate in candidates:
                result = controller.decode_bgra(candidate)
                self.assertIsNot(result, cached)
                np.testing.assert_array_equal(result.view(np.uint32),
                    visibility.decode_depth_bgra(candidate).view(np.uint32))
            self.assertIs(controller.decode_bgra(raw.copy()), cached)
        self.assert_cleared(controller, completed=True)
        self.assertEqual(first.call_count, 1)
        self.assertEqual(second.call_count, len(candidates))
        self.assertEqual(controller.last_stats["fallbacks"], len(candidates))
        self.assertEqual(controller.last_stats["reuses"], 1)

    def test_invalid_decoder_signatures_preserve_original_rejection(self):
        for candidate in (np.zeros((5, 7, 4), np.float32), np.zeros((5, 7), np.uint8),
                          np.zeros((5, 7, 3), np.uint8)):
            with self.subTest(shape=candidate.shape, dtype=candidate.dtype):
                identity, image, _ = fixture()
                controller, _, second = self.controller()
                with self.assertRaises(ValueError):
                    visibility.decode_depth_bgra(candidate)
                with self.assertRaises(ValueError):
                    with controller.request_scope(identity=identity, depth_image=image):
                        controller.decode_image(image)
                        controller.decode_bgra(candidate)
                self.assert_cleared(controller, completed=False)
                self.assertEqual(second.call_count, 1)

    def test_valid_changed_shape_fails_closed_without_original_decode(self):
        identity, image, raw = fixture()
        controller, _, second = self.controller()
        with self.assertRaises(DepthReuseError):
            with controller.request_scope(identity=identity, depth_image=image):
                controller.decode_image(image)
                controller.decode_bgra(raw.reshape(7, 5, 4))
        self.assert_cleared(controller, completed=False)
        second.assert_not_called()

    def test_decode_requires_active_scope_and_initial_avo_decode(self):
        identity, image, raw = fixture()
        controller, first, second = self.controller()
        for action in (lambda: controller.decode_image(image), lambda: controller.decode_bgra(raw)):
            with self.assertRaises(DepthReuseError):
                action()
        with controller.request_scope(identity=identity, depth_image=image):
            with self.assertRaises(DepthReuseError):
                controller.decode_bgra(raw)
            first.assert_not_called()
            second.assert_not_called()
            controller.decode_image(image)
        for action in (lambda: controller.decode_image(image), lambda: controller.decode_bgra(raw)):
            with self.assertRaises(DepthReuseError):
                action()
        self.assert_cleared(controller, completed=True)

    def test_first_decode_cannot_bind_a_different_source_image(self):
        identity, image, raw = fixture()
        changed = raw.copy()
        changed[-1, -1, 0] ^= np.uint8(1)
        _, foreign, _ = fixture(raw=changed)
        controller, first, second = self.controller()
        with self.assertRaises(DepthReuseError):
            with controller.request_scope(identity=identity, depth_image=image):
                controller.decode_image(foreign)
        self.assert_cleared(controller, completed=False)
        first.assert_not_called()
        second.assert_not_called()

    def test_failed_scope_initialization_clears_partial_ownership(self):
        identity, image, _ = fixture()
        incomplete = SimpleNamespace(raw_data=image.raw_data, width=image.width, height=image.height)
        for request_identity, request_image in ((None, image), (identity, incomplete)):
            with self.subTest(identity=request_identity, image_has_frame=hasattr(request_image, "frame")):
                controller, first, second = self.controller()
                with controller.request_scope(identity=identity, depth_image=image):
                    controller.decode_image(image)
                self.assertTrue(controller.last_stats["completed"])
                first.reset_mock()
                with self.assertRaises((DepthReuseError, TypeError, AttributeError)):
                    with controller.request_scope(identity=request_identity, depth_image=request_image):
                        self.fail("incomplete source request admitted")
                self.assertFalse(controller.active)
                self.assertFalse(controller.retained)
                self.assertIsNone(controller.last_stats, "failed setup retained a preceding request's statistics")
                first.assert_not_called()
                second.assert_not_called()

    def test_source_mutation_during_original_decode_is_rejected(self):
        identity, image, _ = fixture()
        controller, first, second = self.controller()

        def mutating_decoder(source):
            decoded = avo.decode_depth(source)
            changed = bytearray(source.raw_data)
            changed[-1] ^= 1
            source.raw_data = bytes(changed)
            return decoded

        first.side_effect = mutating_decoder
        with self.assertRaises(DepthReuseError):
            with controller.request_scope(identity=identity, depth_image=image):
                controller.decode_image(image)
        self.assert_cleared(controller, completed=False)
        second.assert_not_called()

    def test_nested_scope_rejection_preserves_outer_scope(self):
        identity, image, raw = fixture()
        controller, _, _ = self.controller()
        with controller.request_scope(identity=identity, depth_image=image):
            depth = controller.decode_image(image)
            with self.assertRaises(DepthReuseError):
                with controller.request_scope(identity=identity, depth_image=image):
                    self.fail("nested request admitted")
            self.assertTrue(controller.active)
            self.assertIs(controller.decode_bgra(raw), depth)
        self.assert_cleared(controller, completed=True)

    def test_cross_thread_decoder_and_scope_calls_fail_without_disturbing_owner(self):
        identity, image, raw = fixture()
        controller, _, _ = self.controller()
        results = []

        def foreign_calls():
            actions = [lambda: controller.decode_image(image), lambda: controller.decode_bgra(raw)]
            for action in actions:
                try:
                    action()
                except BaseException as error:
                    results.append(error)
                else:
                    results.append(None)
            try:
                with controller.request_scope(identity=identity, depth_image=image):
                    pass
            except BaseException as error:
                results.append(error)
            else:
                results.append(None)

        with controller.request_scope(identity=identity, depth_image=image):
            depth = controller.decode_image(image)
            thread = threading.Thread(target=foreign_calls)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(results), 3)
            self.assertTrue(all(isinstance(error, DepthReuseError) for error in results), results)
            self.assertTrue(controller.active)
            self.assertIs(controller.decode_bgra(raw), depth)
        self.assert_cleared(controller, completed=True)

    def test_consecutive_frames_with_identical_bytes_decode_independently(self):
        controller, first, second = self.controller()
        depths = []
        for frame in (101, 102):
            identity, image, raw = fixture(frame=frame)
            with controller.request_scope(identity=identity, depth_image=image):
                depths.append(controller.decode_image(image))
                self.assertIs(controller.decode_bgra(raw), depths[-1])
            self.assert_cleared(controller, completed=True)
            self.assertEqual(controller.last_stats["frame_id"], frame)
            self.assertEqual(controller.last_stats["decodes"], 1)
            self.assertEqual(controller.last_stats["reuses"], 1)
        self.assertIsNot(depths[0], depths[1])
        np.testing.assert_array_equal(depths[0], depths[1])
        self.assertEqual(first.call_count, 2)
        second.assert_not_called()

    def test_body_and_decoder_exceptions_clear_scope_and_allow_fresh_request(self):
        for decoder_failure in (False, True):
            with self.subTest(decoder_failure=decoder_failure):
                identity, image, raw = fixture()
                controller, first, _ = self.controller()
                failure = RuntimeError("injected depth request failure")
                if decoder_failure:
                    first.side_effect = failure
                with self.assertRaises(RuntimeError) as caught:
                    with controller.request_scope(identity=identity, depth_image=image):
                        controller.decode_image(image)
                        raise failure
                self.assertIs(caught.exception, failure)
                self.assert_cleared(controller, completed=False)
                first.side_effect = None
                identity, image, raw = fixture(frame=102)
                with controller.request_scope(identity=identity, depth_image=image):
                    depth = controller.decode_image(image)
                    self.assertIs(controller.decode_bgra(raw), depth)
                self.assert_cleared(controller, completed=True)


if __name__ == "__main__":
    unittest.main()
