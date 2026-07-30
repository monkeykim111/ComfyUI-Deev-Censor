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
    def __init__(self, prediction=None, prototype=None, error=None):
        self.prediction = prediction
        self.prototype = prototype
        self.error = error

    def run(self, output_names, feed):
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


def outputs(class_id=None, confidence=0.9, mask_logit=5.0):
    channels = 4 + len(censor.EXPECTED_CLASS_NAMES) + censor.MODEL_MASK_DIMENSIONS
    prediction = np.zeros((1, channels, 1), dtype=np.float32)
    prediction[0, 0:4, 0] = [640, 640, 320, 320]
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
        result = censor._censor_numpy_image(
            self.image,
            runtime(FakeSession(prediction, prototype)),
        )
        self.assertGreater(np.count_nonzero(np.all(result == 1, axis=2)), 0)
        np.testing.assert_array_equal(result[0, 0], self.image[0, 0])

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


if __name__ == "__main__":
    unittest.main()
