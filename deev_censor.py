"""Fail-closed genital/anus censorship for Deev ComfyUI workflows.

The model identity, target classes, and rendering policy are production
invariants. Detection confidence can be tightened within a bounded
safety-oriented range for validated Deev workflows.
"""

from __future__ import annotations

import ast
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import folder_paths
import numpy as np
import onnxruntime as ort
import torch

if __package__:
    from .model_downloader import (
        EXPECTED_MODEL_SHA256,
        MODEL_FILENAME,
        MODEL_SUBDIRECTORY,
        ModelDownloadError,
        ensure_model,
        sha256_file,
    )
else:
    # Support direct validation scripts outside normal ComfyUI package loading.
    from model_downloader import (
        EXPECTED_MODEL_SHA256,
        MODEL_FILENAME,
        MODEL_SUBDIRECTORY,
        ModelDownloadError,
        ensure_model,
        sha256_file,
    )
EXPECTED_CLASS_NAMES = {
    0: "anus",
    1: "nipple",
    2: "penis",
    3: "vagina",
    4: "female face",
    5: "male face",
    6: "pubic hair",
}
TARGET_CLASS_IDS = frozenset((0, 2, 3))
FACE_CLASS_IDS = frozenset((4, 5))

MODEL_IMAGE_SIZE = 1280
MODEL_MASK_DIMENSIONS = 32
DETECTION_CONFIDENCE = 0.05
MIN_DETECTION_CONFIDENCE = 0.01
MAX_DETECTION_CONFIDENCE = 0.15
STABLE_MASK_CONFIDENCE = 0.35
NMS_IOU_THRESHOLD = 0.70
MAX_DETECTIONS = 100
MAX_NMS_CANDIDATES_PER_CLASS = 1_000

WHITE_MASK_DILATION_PIXELS = 6
BOX_EXPANSION_RATIO_PER_SIDE = 0.20
BOX_EXPANSION_MINIMUM_PIXELS = 8
MOSAIC_SHORT_SIDE_CELLS = 10

_EXPECTED_METADATA = {
    "author": "Ultralytics",
    "batch": "1",
    "channels": "3",
    "end2end": "False",
    "imgsz": "[1280, 1280]",
    "stride": "32",
    "task": "segment",
    "version": "8.4.52",
}


class DeevCensorError(RuntimeError):
    """Raised when censorship cannot be completed safely."""


@dataclass(frozen=True)
class _Letterbox:
    scale: float
    left: int
    top: int
    resized_width: int
    resized_height: int
    source_width: int
    source_height: int


@dataclass(frozen=True)
class _Detection:
    class_id: int
    confidence: float
    box_xyxy: np.ndarray
    mask_coefficients: np.ndarray
    source_index: int


@dataclass(frozen=True)
class _Runtime:
    session: Any
    input_name: str
    prediction_output_name: str
    prototype_output_name: str
    model_signature: tuple[int, int, int, int]


_runtime_lock = threading.Lock()
_runtime: _Runtime | None = None


def _model_path() -> Path:
    return Path(folder_paths.models_dir) / MODEL_SUBDIRECTORY / MODEL_FILENAME


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        raise DeevCensorError(f"required Deev censor model is unavailable: {path}") from error
    if not path.is_file():
        raise DeevCensorError(f"required Deev censor model is not a file: {path}")
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _sha256(path: Path) -> str:
    try:
        return sha256_file(path)
    except ModelDownloadError as error:
        raise DeevCensorError(f"cannot read Deev censor model: {path}") from error


def _parse_literal_map(value: str, field: str) -> dict[Any, Any]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise DeevCensorError(f"invalid ONNX metadata field: {field}") from error
    if not isinstance(parsed, dict):
        raise DeevCensorError(f"ONNX metadata field is not a map: {field}")
    return parsed


def _verify_metadata(session: Any) -> None:
    try:
        metadata = dict(session.get_modelmeta().custom_metadata_map)
    except Exception as error:
        raise DeevCensorError("cannot read ONNX model metadata") from error

    for key, expected in _EXPECTED_METADATA.items():
        if metadata.get(key) != expected:
            raise DeevCensorError(
                f"unexpected ONNX metadata value for {key!r}: {metadata.get(key)!r}",
            )

    if metadata.get("license") != (
        "AGPL-3.0 License (https://ultralytics.com/license)"
    ):
        raise DeevCensorError("unexpected ONNX model license metadata")

    names = _parse_literal_map(metadata.get("names", ""), "names")
    try:
        normalized_names = {int(key): str(value) for key, value in names.items()}
    except (TypeError, ValueError) as error:
        raise DeevCensorError("invalid ONNX class names metadata") from error
    if normalized_names != EXPECTED_CLASS_NAMES:
        raise DeevCensorError("unexpected ONNX class names")

    args = _parse_literal_map(metadata.get("args", ""), "args")
    expected_args = {
        "batch": 1,
        "half": True,
        "dynamic": True,
        "simplify": True,
        "opset": None,
        "nms": False,
    }
    if args != expected_args:
        raise DeevCensorError("unexpected ONNX export arguments")


def _shape_rank(value: Any) -> int:
    shape = getattr(value, "shape", None)
    return len(shape) if isinstance(shape, Sequence) else -1


def _verify_io(session: Any) -> tuple[str, str, str]:
    try:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
    except Exception as error:
        raise DeevCensorError("cannot inspect ONNX model inputs and outputs") from error

    if len(inputs) != 1:
        raise DeevCensorError("Deev censor ONNX must have exactly one input")
    model_input = inputs[0]
    if (
        model_input.name != "images"
        or model_input.type != "tensor(float)"
        or _shape_rank(model_input) != 4
        or model_input.shape[1] != 3
    ):
        raise DeevCensorError("unexpected Deev censor ONNX input signature")

    outputs_by_name = {output.name: output for output in outputs}
    prediction = outputs_by_name.get("output0")
    prototype = outputs_by_name.get("output1")
    if (
        len(outputs) != 2
        or prediction is None
        or prototype is None
        or prediction.type != "tensor(float)"
        or prototype.type != "tensor(float)"
        or _shape_rank(prediction) != 3
        or _shape_rank(prototype) != 4
    ):
        raise DeevCensorError("unexpected Deev censor ONNX output signature")

    prediction_channels = prediction.shape[1]
    prototype_channels = prototype.shape[1]
    expected_channels = 4 + len(EXPECTED_CLASS_NAMES) + MODEL_MASK_DIMENSIONS
    if prediction_channels not in (expected_channels, "channels"):
        raise DeevCensorError("unexpected ONNX prediction channel count")
    if prototype_channels not in (MODEL_MASK_DIMENSIONS, "mask_channels"):
        raise DeevCensorError("unexpected ONNX prototype channel count")

    return model_input.name, prediction.name, prototype.name


def _new_session(path: Path) -> Any:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if hasattr(options, "use_deterministic_compute"):
        options.use_deterministic_compute = True

    try:
        available = set(ort.get_available_providers())
    except Exception as error:
        raise DeevCensorError("cannot enumerate ONNX Runtime providers") from error
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "CPUExecutionProvider" in available:
        providers = ["CPUExecutionProvider"]
    else:
        raise DeevCensorError("no supported ONNX Runtime execution provider")

    try:
        return ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=providers,
        )
    except Exception as error:
        raise DeevCensorError("cannot initialize Deev censor ONNX session") from error


def _load_runtime() -> _Runtime:
    path = _model_path()
    signature_before = _file_signature(path)
    actual_hash = _sha256(path)
    signature_after = _file_signature(path)
    if signature_before != signature_after:
        raise DeevCensorError("Deev censor model changed while it was being verified")
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise DeevCensorError(
            f"Deev censor model SHA256 mismatch: expected {EXPECTED_MODEL_SHA256}",
        )

    session = _new_session(path)
    _verify_metadata(session)
    input_name, prediction_name, prototype_name = _verify_io(session)
    return _Runtime(
        session=session,
        input_name=input_name,
        prediction_output_name=prediction_name,
        prototype_output_name=prototype_name,
        model_signature=signature_after,
    )


def _get_runtime() -> _Runtime:
    global _runtime

    path = _model_path()
    try:
        current_signature = _file_signature(path)
    except DeevCensorError:
        try:
            ensure_model(path)
        except ModelDownloadError as error:
            raise DeevCensorError(str(error)) from error
        current_signature = _file_signature(path)

    if _runtime is not None and _runtime.model_signature == current_signature:
        return _runtime

    with _runtime_lock:
        try:
            current_signature = _file_signature(path)
        except DeevCensorError:
            try:
                ensure_model(path)
            except ModelDownloadError as error:
                raise DeevCensorError(str(error)) from error
            current_signature = _file_signature(path)
        if _runtime is not None and _runtime.model_signature == current_signature:
            return _runtime
        _runtime = _load_runtime()
        return _runtime


def _letterbox(image: np.ndarray) -> tuple[np.ndarray, _Letterbox]:
    source_height, source_width = image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise DeevCensorError("cannot censor an empty image")

    scale = min(
        MODEL_IMAGE_SIZE / source_height,
        MODEL_IMAGE_SIZE / source_width,
    )
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    left = round((MODEL_IMAGE_SIZE - resized_width) / 2 - 0.1)
    top = round((MODEL_IMAGE_SIZE - resized_height) / 2 - 0.1)
    right = MODEL_IMAGE_SIZE - resized_width - left
    bottom = MODEL_IMAGE_SIZE - resized_height - top

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114 / 255, 114 / 255, 114 / 255),
    )
    if padded.shape != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3):
        raise DeevCensorError("letterbox preprocessing produced an invalid shape")

    tensor = np.ascontiguousarray(
        padded.transpose(2, 0, 1)[None],
        dtype=np.float32,
    )
    return tensor, _Letterbox(
        scale=scale,
        left=left,
        top=top,
        resized_width=resized_width,
        resized_height=resized_height,
        source_width=source_width,
        source_height=source_height,
    )


def _normalize_outputs(
    prediction: Any,
    prototype: Any,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(prediction)
    prototype_array = np.asarray(prototype)
    expected_channels = (
        4 + len(EXPECTED_CLASS_NAMES) + MODEL_MASK_DIMENSIONS
    )

    if (
        prediction_array.ndim != 3
        or prototype_array.ndim != 4
        or prediction_array.shape[0] != 1
        or prototype_array.shape[0] != 1
        or prototype_array.shape[1] != MODEL_MASK_DIMENSIONS
    ):
        raise DeevCensorError("ONNX inference returned invalid output dimensions")
    if prediction_array.shape[1] == expected_channels:
        prediction_array = prediction_array[0].T
    elif prediction_array.shape[2] == expected_channels:
        prediction_array = prediction_array[0]
    else:
        raise DeevCensorError("ONNX inference returned invalid prediction channels")

    prototype_array = prototype_array[0]
    if (
        prediction_array.shape[0] == 0
        or prototype_array.shape[1] == 0
        or prototype_array.shape[2] == 0
        or not np.isfinite(prediction_array).all()
        or not np.isfinite(prototype_array).all()
    ):
        raise DeevCensorError("ONNX inference returned empty or non-finite output")

    return (
        np.asarray(prediction_array, dtype=np.float32),
        np.asarray(prototype_array, dtype=np.float32),
    )


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(0, right_bottom - left_top)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0,
        float(box[3] - box[1]),
    )
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = box_area + boxes_area - intersection_area
    return np.divide(
        intersection_area,
        union,
        out=np.zeros_like(intersection_area),
        where=union > 0,
    )


def _nms(detections: Iterable[_Detection]) -> list[_Detection]:
    kept: list[_Detection] = []
    grouped: dict[int, list[_Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.class_id, []).append(detection)

    for class_id in sorted(grouped):
        candidates = sorted(
            grouped[class_id],
            key=lambda item: (-item.confidence, item.source_index),
        )[:MAX_NMS_CANDIDATES_PER_CLASS]
        while candidates and len(kept) < MAX_DETECTIONS:
            selected = candidates.pop(0)
            kept.append(selected)
            if not candidates:
                continue
            boxes = np.stack([item.box_xyxy for item in candidates])
            overlaps = _box_iou(selected.box_xyxy, boxes)
            candidates = [
                item
                for item, overlap in zip(candidates, overlaps, strict=True)
                if overlap <= NMS_IOU_THRESHOLD
            ]

    return sorted(
        kept,
        key=lambda item: (-item.confidence, item.class_id, item.source_index),
    )


def _detections(
    prediction: np.ndarray,
    detection_confidence: float = DETECTION_CONFIDENCE,
) -> tuple[list[_Detection], int]:
    boxes = _xywh_to_xyxy(prediction[:, :4])
    class_scores = prediction[:, 4 : 4 + len(EXPECTED_CLASS_NAMES)]
    mask_coefficients = prediction[:, 4 + len(EXPECTED_CLASS_NAMES) :]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[
        np.arange(class_scores.shape[0]),
        class_ids,
    ]

    candidates: list[_Detection] = []
    face_candidates: list[_Detection] = []
    for index in np.flatnonzero(confidences >= detection_confidence):
        class_id = int(class_ids[index])
        if class_id not in TARGET_CLASS_IDS and class_id not in FACE_CLASS_IDS:
            # In particular, class 1 (nipple) never affects censoring policy.
            continue
        box = np.clip(boxes[index], 0, MODEL_IMAGE_SIZE).astype(
            np.float32,
            copy=True,
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            if class_id in TARGET_CLASS_IDS:
                raise DeevCensorError("target detection has an invalid box")
            continue
        detection = _Detection(
            class_id=class_id,
            confidence=float(confidences[index]),
            box_xyxy=box,
            mask_coefficients=mask_coefficients[index].astype(
                np.float32,
                copy=True,
            ),
            source_index=int(index),
        )
        if class_id in TARGET_CLASS_IDS:
            candidates.append(detection)
        else:
            face_candidates.append(detection)

    return _nms(candidates), len(_nms(face_candidates))


def _restore_box(
    input_box: np.ndarray,
    letterbox: _Letterbox,
) -> tuple[int, int, int, int]:
    restored = input_box.astype(np.float64, copy=True)
    restored[[0, 2]] = (restored[[0, 2]] - letterbox.left) / letterbox.scale
    restored[[1, 3]] = (restored[[1, 3]] - letterbox.top) / letterbox.scale
    restored[[0, 2]] = np.clip(
        restored[[0, 2]],
        0,
        letterbox.source_width,
    )
    restored[[1, 3]] = np.clip(
        restored[[1, 3]],
        0,
        letterbox.source_height,
    )
    left = max(0, int(np.floor(restored[0])))
    top = max(0, int(np.floor(restored[1])))
    right = min(letterbox.source_width, int(np.ceil(restored[2])))
    bottom = min(letterbox.source_height, int(np.ceil(restored[3])))
    if right <= left or bottom <= top:
        raise DeevCensorError("target detection is outside the source image")
    return left, top, right, bottom


def _decode_mask(
    detection: _Detection,
    prototype: np.ndarray,
    letterbox: _Letterbox,
) -> np.ndarray:
    channels, mask_height, mask_width = prototype.shape
    if (
        channels != MODEL_MASK_DIMENSIONS
        or detection.mask_coefficients.shape != (MODEL_MASK_DIMENSIONS,)
    ):
        raise DeevCensorError("cannot decode ONNX segmentation coefficients")

    logits = detection.mask_coefficients @ prototype.reshape(channels, -1)
    logits = logits.reshape(mask_height, mask_width)
    if not np.isfinite(logits).all():
        raise DeevCensorError("segmentation mask contains non-finite values")

    box = detection.box_xyxy
    proto_box = np.array(
        [
            box[0] * mask_width / MODEL_IMAGE_SIZE,
            box[1] * mask_height / MODEL_IMAGE_SIZE,
            box[2] * mask_width / MODEL_IMAGE_SIZE,
            box[3] * mask_height / MODEL_IMAGE_SIZE,
        ],
    )
    crop_left = max(0, int(np.floor(proto_box[0])))
    crop_top = max(0, int(np.floor(proto_box[1])))
    crop_right = min(mask_width, int(np.ceil(proto_box[2])))
    crop_bottom = min(mask_height, int(np.ceil(proto_box[3])))
    # Keep the crop sentinel finite. Some OpenCV interpolation kernels can
    # otherwise produce NaN at the boundary between finite values and -inf.
    cropped_logits = np.full_like(logits, -10_000)
    if crop_right > crop_left and crop_bottom > crop_top:
        cropped_logits[crop_top:crop_bottom, crop_left:crop_right] = logits[
            crop_top:crop_bottom,
            crop_left:crop_right,
        ]

    input_logits = cv2.resize(
        cropped_logits,
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    content_logits = input_logits[
        letterbox.top : letterbox.top + letterbox.resized_height,
        letterbox.left : letterbox.left + letterbox.resized_width,
    ]
    if content_logits.shape != (
        letterbox.resized_height,
        letterbox.resized_width,
    ):
        raise DeevCensorError("cannot remove letterbox padding from mask")
    source_logits = cv2.resize(
        content_logits,
        (letterbox.source_width, letterbox.source_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return source_logits > 0


def _is_stable_mask(
    mask: np.ndarray,
    source_box: tuple[int, int, int, int],
) -> bool:
    mask_area = int(np.count_nonzero(mask))
    left, top, right, bottom = source_box
    box_area = (right - left) * (bottom - top)
    if mask_area < 16 or box_area <= 0:
        return False
    area_ratio = mask_area / box_area
    if not 0.03 <= area_ratio <= 0.95:
        return False

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    component_areas = sorted(
        (
            int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, component_count)
            if stats[index, cv2.CC_STAT_AREA] >= max(4, round(mask_area * 0.01))
        ),
        reverse=True,
    )
    if not component_areas or len(component_areas) > 2:
        return False
    return component_areas[0] / mask_area >= 0.80


def _dilate(mask: np.ndarray) -> np.ndarray:
    diameter = WHITE_MASK_DILATION_PIXELS * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (diameter, diameter),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _expanded_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    horizontal = max(
        BOX_EXPANSION_MINIMUM_PIXELS,
        round((right - left) * BOX_EXPANSION_RATIO_PER_SIDE),
    )
    vertical = max(
        BOX_EXPANSION_MINIMUM_PIXELS,
        round((bottom - top) * BOX_EXPANSION_RATIO_PER_SIDE),
    )
    return (
        max(0, left - horizontal),
        max(0, top - vertical),
        min(width, right + horizontal),
        min(height, bottom + vertical),
    )


def _mosaic(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise DeevCensorError("cannot mosaic an empty target region")
    crop = image[top:bottom, left:right, :3]
    block_size = max(4, round(min(width, height) / MOSAIC_SHORT_SIDE_CELLS))
    down_width = max(1, int(np.ceil(width / block_size)))
    down_height = max(1, int(np.ceil(height / block_size)))
    reduced = cv2.resize(
        crop,
        (down_width, down_height),
        interpolation=cv2.INTER_AREA,
    )
    image[top:bottom, left:right, :3] = cv2.resize(
        reduced,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def _apply_policy(
    image: np.ndarray,
    detections: list[_Detection],
    face_count: int,
    prototype: np.ndarray,
    letterbox: _Letterbox,
) -> np.ndarray:
    output = image.copy()
    if not detections:
        return output

    source_boxes = [
        _restore_box(detection.box_xyxy, letterbox)
        for detection in detections
    ]
    masks = [
        _decode_mask(detection, prototype, letterbox)
        for detection in detections
    ]

    stable = (
        len(detections) == 1
        and face_count <= 1
        and detections[0].confidence >= STABLE_MASK_CONFIDENCE
        and _is_stable_mask(masks[0], source_boxes[0])
    )
    if stable:
        output[_dilate(masks[0]), :3] = 1.0
        return output

    for source_box in source_boxes:
        _mosaic(
            output,
            _expanded_box(
                source_box,
                letterbox.source_width,
                letterbox.source_height,
            ),
        )
    return output


def _censor_numpy_image(
    image: np.ndarray,
    runtime: _Runtime,
    detection_confidence: float = DETECTION_CONFIDENCE,
) -> np.ndarray:
    if (
        image.ndim != 3
        or image.shape[2] < 3
        or not np.issubdtype(image.dtype, np.floating)
        or not np.isfinite(image).all()
    ):
        raise DeevCensorError("ComfyUI IMAGE input is invalid")
    if np.any(image < 0) or np.any(image > 1):
        raise DeevCensorError("ComfyUI IMAGE values must be within [0, 1]")

    model_input, letterbox = _letterbox(
        np.asarray(image[:, :, :3], dtype=np.float32),
    )
    try:
        raw_outputs = runtime.session.run(
            [
                runtime.prediction_output_name,
                runtime.prototype_output_name,
            ],
            {runtime.input_name: model_input},
        )
    except Exception as error:
        raise DeevCensorError("Deev censor ONNX inference failed") from error
    if not isinstance(raw_outputs, Sequence) or len(raw_outputs) != 2:
        raise DeevCensorError("Deev censor ONNX returned unexpected outputs")

    prediction, prototype = _normalize_outputs(
        raw_outputs[0],
        raw_outputs[1],
    )
    detections, face_count = _detections(
        prediction,
        detection_confidence,
    )
    return _apply_policy(
        image,
        detections,
        face_count,
        prototype,
        letterbox,
    )


class DeevGenitalAnusCensor:
    """Censor anus/penis/vagina while deliberately leaving nipples untouched."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "detection_confidence": (
                    "FLOAT",
                    {
                        "default": DETECTION_CONFIDENCE,
                        "min": MIN_DETECTION_CONFIDENCE,
                        "max": MAX_DETECTION_CONFIDENCE,
                        "step": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "censor"
    CATEGORY = "Toonsquare/Deev"

    def censor(
        self,
        images: torch.Tensor,
        detection_confidence: float = DETECTION_CONFIDENCE,
    ) -> tuple[torch.Tensor]:
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[-1] < 3
            or images.shape[0] <= 0
        ):
            raise DeevCensorError("expected a non-empty ComfyUI IMAGE batch")
        try:
            detection_confidence = float(detection_confidence)
        except (TypeError, ValueError) as error:
            raise DeevCensorError("detection confidence must be numeric") from error
        if (
            not np.isfinite(detection_confidence)
            or detection_confidence < MIN_DETECTION_CONFIDENCE
            or detection_confidence > MAX_DETECTION_CONFIDENCE
        ):
            raise DeevCensorError(
                "detection confidence must be between "
                f"{MIN_DETECTION_CONFIDENCE} and {MAX_DETECTION_CONFIDENCE}",
            )

        runtime = _get_runtime()
        try:
            source = images.detach().to(device="cpu", dtype=torch.float32).numpy()
        except Exception as error:
            raise DeevCensorError("cannot read ComfyUI IMAGE tensor") from error

        # Do not return partial output: every image must be censored successfully.
        processed = [
            _censor_numpy_image(
                image,
                runtime,
                detection_confidence,
            )
            for image in source
        ]
        try:
            output = torch.from_numpy(np.stack(processed)).to(
                device=images.device,
                dtype=images.dtype,
            )
        except Exception as error:
            raise DeevCensorError("cannot construct censored ComfyUI IMAGE batch") from error
        return (output,)
