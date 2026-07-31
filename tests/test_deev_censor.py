from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "deev_censor.py"
sys.path.insert(0, str(MODULE_PATH.parent))


def load_module():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = tempfile.gettempdir()
    sys.modules["folder_paths"] = folder_paths
    spec = importlib.util.spec_from_file_location("deev_censor_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


censor = load_module()


class FakeSession:
    def __init__(
        self,
        prediction=None,
        prototype=None,
        error=None,
        responses=None,
    ):
        self.prediction = prediction
        self.prototype = prototype
        self.error = error
        self.responses = list(responses) if responses is not None else None
        self.call_count = 0
        self.feeds = []

    def run(self, output_names, feed):
        self.call_count += 1
        self.feeds.append(feed)
        if self.responses is not None:
            if not self.responses:
                raise AssertionError("unexpected inference call")
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return list(response)
        if self.error is not None:
            raise self.error
        return [self.prediction, self.prototype]


def runtime(session):
    return censor._Runtime(
        session=session,
        input_name="images",
        prediction_output_name="output0",
        prototype_output_name="output1",
        model_signature=(0, 0, 0, 0),
    )


def outputs(
    class_id=None,
    confidence=0.9,
    mask_logit=5.0,
    box=(640, 640, 320, 320),
):
    channels = 4 + len(censor.EXPECTED_CLASS_NAMES) + censor.MODEL_MASK_DIMENSIONS
    prediction = np.zeros((1, channels, 1), dtype=np.float32)
    prediction[0, 0:4, 0] = box
    if class_id is not None:
        prediction[0, 4 + class_id, 0] = confidence
    prediction[0, 4 + len(censor.EXPECTED_CLASS_NAMES), 0] = 1

    prototype = np.full(
        (1, censor.MODEL_MASK_DIMENSIONS, 32, 32),
        -mask_logit,
        dtype=np.float32,
    )
    prototype[0, 0, 12:20, 12:20] = mask_logit
    return prediction, prototype


class DeevCensorUnitTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((256, 256, 3), 0.25, dtype=np.float32)

    def test_nipple_detection_does_not_modify_image(self):
        prediction, prototype = outputs(class_id=1)
        result = censor._censor_numpy_image(
            self.image,
            runtime(FakeSession(prediction, prototype)),
        )
        np.testing.assert_array_equal(result, self.image)
        self.assertIsNot(result, self.image)

    def test_single_clean_target_uses_dilated_white_mask(self):
        prediction, prototype = outputs(class_id=0)
        session = FakeSession(prediction, prototype)
        result = censor._censor_numpy_image(
            self.image,
            runtime(session),
            enable_tiled_retry=True,
        )
        self.assertGreater(np.count_nonzero(np.all(result == 1, axis=2)), 0)
        np.testing.assert_array_equal(result[0, 0], self.image[0, 0])
        self.assertEqual(session.call_count, 1)

    def test_low_confidence_target_uses_expanded_mosaic(self):
        gradient = np.linspace(0, 1, 256, dtype=np.float32)
        image = np.repeat(gradient[None, :, None], 256, axis=0)
        image = np.repeat(image, 3, axis=2)
        prediction, prototype = outputs(
            class_id=2,
            confidence=censor.DETECTION_CONFIDENCE,
        )
        result = censor._censor_numpy_image(
            image,
            runtime(FakeSession(prediction, prototype)),
        )
        self.assertFalse(np.array_equal(result, image))
        self.assertEqual(
            np.count_nonzero(np.all(result == 1, axis=2)),
            np.count_nonzero(np.all(image == 1, axis=2)),
        )

    def test_configurable_lower_threshold_accepts_weak_target(self):
        gradient = np.linspace(0, 1, 256, dtype=np.float32)
        image = np.repeat(gradient[None, :, None], 256, axis=0)
        image = np.repeat(image, 3, axis=2)
        prediction, prototype = outputs(class_id=0, confidence=0.08)
        legacy_result = censor._censor_numpy_image(
            image,
            runtime(FakeSession(prediction, prototype)),
            detection_confidence=0.15,
        )
        conservative_result = censor._censor_numpy_image(
            image,
            runtime(FakeSession(prediction, prototype)),
            detection_confidence=0.05,
        )
        np.testing.assert_array_equal(legacy_result, image)
        self.assertFalse(np.array_equal(conservative_result, image))

    def test_node_rejects_detection_confidence_outside_safe_range(self):
        node = censor.DeevGenitalAnusCensor()
        with (
            mock.patch.object(censor, "_get_runtime"),
            self.assertRaisesRegex(
                censor.DeevCensorError,
                "detection confidence must be between",
            ),
        ):
            node.censor(
                censor.torch.zeros((1, 32, 32, 3), dtype=censor.torch.float32),
                detection_confidence=0.5,
            )

    def test_inference_error_is_fail_closed(self):
        with self.assertRaisesRegex(censor.DeevCensorError, "inference failed"):
            censor._censor_numpy_image(
                self.image,
                runtime(FakeSession(error=RuntimeError("broken provider"))),
            )

    def test_invalid_output_is_fail_closed(self):
        prediction, prototype = outputs(class_id=3)
        prediction[0, 0, 0] = np.nan
        with self.assertRaisesRegex(censor.DeevCensorError, "non-finite"):
            censor._censor_numpy_image(
                self.image,
                runtime(FakeSession(prediction, prototype)),
            )

    def test_wrong_model_hash_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / censor.MODEL_FILENAME
            path.write_bytes(b"not the pinned model")
            with (
                mock.patch.object(censor, "_model_path", return_value=path),
                self.assertRaisesRegex(censor.DeevCensorError, "SHA256 mismatch"),
            ):
                censor._load_runtime()

    def test_full_miss_without_tiled_retry_stays_single_pass(self):
        prediction, prototype = outputs()
        session = FakeSession(prediction, prototype)
        result = censor._censor_numpy_image(
            self.image,
            runtime(session),
            enable_tiled_retry=False,
        )
        np.testing.assert_array_equal(result, self.image)
        self.assertIsNot(result, self.image)
        self.assertEqual(session.call_count, 1)

    def test_full_miss_retries_all_tiles_and_maps_hit_to_source_slice(self):
        miss = outputs()
        hit = outputs(class_id=0, confidence=0.20)
        session = FakeSession(responses=[miss, hit, miss, miss, miss])
        gradient = np.linspace(0, 1, 256, dtype=np.float32)
        image = np.repeat(gradient[None, :, None], 256, axis=0)
        image = np.repeat(image, 3, axis=2)
        result = censor._censor_numpy_image(
            image,
            runtime(session),
            enable_tiled_retry=True,
        )
        self.assertEqual(session.call_count, 5)
        self.assertFalse(np.array_equal(result, image))
        np.testing.assert_array_equal(result[-16:, -16:], image[-16:, -16:])

    def test_tiled_retry_ignores_non_anus_target_detections(self):
        miss = outputs()
        penis = outputs(class_id=2, confidence=0.90)
        session = FakeSession(responses=[miss, penis, miss, miss, miss])
        result = censor._censor_numpy_image(
            self.image,
            runtime(session),
            enable_tiled_retry=True,
        )
        np.testing.assert_array_equal(result, self.image)
        self.assertEqual(session.call_count, 5)

    def test_tiled_retry_uses_stricter_anus_confidence(self):
        miss = outputs()
        weak_anus = outputs(class_id=0, confidence=0.08)
        session = FakeSession(
            responses=[miss, weak_anus, miss, miss, miss],
        )
        result = censor._censor_numpy_image(
            self.image,
            runtime(session),
            enable_tiled_retry=True,
        )
        np.testing.assert_array_equal(result, self.image)
        self.assertEqual(session.call_count, 5)

    def test_full_non_anus_target_is_kept_while_tiled_anus_is_added(self):
        full_penis = outputs(class_id=2, confidence=0.20)
        tile_anus = outputs(class_id=0, confidence=0.90)
        miss = outputs()
        session = FakeSession(
            responses=[full_penis, tile_anus, miss, miss, miss],
        )
        gradient = np.linspace(0, 1, 256, dtype=np.float32)
        image = np.repeat(gradient[None, :, None], 256, axis=0)
        image = np.repeat(image, 3, axis=2)
        result = censor._censor_numpy_image(
            image,
            runtime(session),
            enable_tiled_retry=True,
        )
        self.assertFalse(np.array_equal(result, image))
        self.assertEqual(session.call_count, 5)

    def test_all_tiled_retry_views_can_miss_without_recursive_calls(self):
        miss = outputs()
        session = FakeSession(responses=[miss, miss, miss, miss, miss])
        result = censor._censor_numpy_image(
            self.image,
            runtime(session),
            enable_tiled_retry=True,
        )
        np.testing.assert_array_equal(result, self.image)
        self.assertEqual(session.call_count, 5)

    def test_multiple_tile_hits_force_mosaic_instead_of_white(self):
        miss = outputs()
        hit = outputs(class_id=0, confidence=0.90)
        session = FakeSession(responses=[miss, hit, hit, miss, miss])
        gradient = np.linspace(0, 1, 256, dtype=np.float32)
        image = np.repeat(gradient[None, :, None], 256, axis=0)
        image = np.repeat(image, 3, axis=2)
        result = censor._censor_numpy_image(
            image,
            runtime(session),
            enable_tiled_retry=True,
        )
        self.assertFalse(np.array_equal(result, image))
        self.assertEqual(
            np.count_nonzero(np.all(result == 1, axis=2)),
            np.count_nonzero(np.all(image == 1, axis=2)),
        )

    def test_tile_edge_hit_expands_mosaic_in_global_coordinates(self):
        miss = outputs()
        edge_anus = outputs(
            class_id=0,
            confidence=0.90,
            # The raw box remains just inside the tile, but its minimum
            # 8-pixel policy expansion crosses the tile boundary.
            box=(1220, 640, 40, 320),
        )
        session = FakeSession(
            responses=[miss, edge_anus, miss, miss, miss],
        )
        gradient = np.linspace(0, 1, 256, dtype=np.float32)
        image = np.repeat(gradient[None, :, None], 256, axis=0)
        image = np.repeat(image, 3, axis=2)
        result = censor._censor_numpy_image(
            image,
            runtime(session),
            enable_tiled_retry=True,
        )
        first_tile_right = censor._tile_bounds(256, 256)[0][2]
        self.assertGreater(
            np.count_nonzero(
                result[:, first_tile_right:] != image[:, first_tile_right:],
            ),
            0,
        )

    def test_tile_bounds_cover_odd_non_square_image_with_overlap(self):
        bounds = censor._tile_bounds(width=503, height=301)
        self.assertEqual(len(bounds), 4)
        for left, top, right, bottom in bounds:
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, 503)
            self.assertLessEqual(bottom, 301)
            self.assertGreater(right, left)
            self.assertGreater(bottom, top)
        self.assertGreater(bounds[0][2] - bounds[1][0], 0)
        self.assertGreater(bounds[0][3] - bounds[2][1], 0)

    def test_tile_bounds_dedupe_extremely_short_axis(self):
        bounds = censor._tile_bounds(width=503, height=1)
        self.assertEqual(len(bounds), 2)
        self.assertTrue(
            all(
                top == 0 and bottom == 1
                for _, top, _, bottom in bounds
            ),
        )

    def test_tiled_retry_error_is_fail_closed_before_rendering(self):
        miss = outputs()
        session = FakeSession(
            responses=[
                miss,
                miss,
                RuntimeError("broken tiled provider"),
            ],
        )
        original = self.image.copy()
        with self.assertRaisesRegex(censor.DeevCensorError, "inference failed"):
            censor._censor_numpy_image(
                self.image,
                runtime(session),
                enable_tiled_retry=True,
            )
        np.testing.assert_array_equal(self.image, original)


if __name__ == "__main__":
    unittest.main()
