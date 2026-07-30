"""Read-only validator for the pinned model and one known PoC image.

This is intentionally not part of unittest discovery. Pass an absolute model
and image path when validating a built image or the remote PoC environment:

    python tests/validate_real_model.py MODEL.onnx IMAGE.png
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def load_module(model_path: Path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(model_path.parent)
    sys.modules["folder_paths"] = folder_paths
    module_path = PACKAGE_ROOT / "deev_censor.py"
    spec = importlib.util.spec_from_file_location(
        "deev_censor_real_model_validator",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._model_path = lambda: model_path
    return module


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: validate_real_model.py MODEL.onnx IMAGE.png")
    model_path = Path(sys.argv[1]).resolve()
    image_path = Path(sys.argv[2]).resolve()
    module = load_module(model_path)
    runtime = module._load_runtime()

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    model_input, letterbox = module._letterbox(image)
    raw_prediction, raw_prototype = runtime.session.run(
        [runtime.prediction_output_name, runtime.prototype_output_name],
        {runtime.input_name: model_input},
    )
    prediction, prototype = module._normalize_outputs(
        raw_prediction,
        raw_prototype,
    )
    detections, face_count = module._detections(prediction)
    output = module._apply_policy(
        image,
        detections,
        face_count,
        prototype,
        letterbox,
    )
    if detections and np.array_equal(output, image):
        raise RuntimeError("target detections did not change the output")
    print(
        {
            "detections": [
                {
                    "class": module.EXPECTED_CLASS_NAMES[item.class_id],
                    "confidence": round(item.confidence, 6),
                    "box": [round(float(value), 2) for value in item.box_xyxy],
                }
                for item in detections
            ],
            "faceCount": face_count,
            "outputChanged": not np.array_equal(output, image),
        },
    )


if __name__ == "__main__":
    main()
