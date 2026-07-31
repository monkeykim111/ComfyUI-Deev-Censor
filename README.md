# ComfyUI Deev Censor

A fail-closed ComfyUI image node for deterministic anime genital and anus
censorship.

The node detects only `anus`, `penis`, and `vagina`. Nipple detections are
intentionally ignored. A stable single segmentation mask is filled white;
ambiguous or multiple detections use an expanded mosaic fallback.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/monkeykim111/ComfyUI-Deev-Censor.git
```

The default requirements install CPU ONNX Runtime for portable ComfyUI Manager
installs. CUDA hosts can replace `onnxruntime` with a compatible
`onnxruntime-gpu` package. PyTorch is supplied by ComfyUI. All runtime
dependencies are already present in the Toonsquare async worker image, which
does not reinstall this requirements file.

The pinned ONNX model is downloaded lazily on the first node execution when it
is absent. To prefetch it before starting ComfyUI:

```bash
python ComfyUI-Deev-Censor/download_model.py
```

The model is installed at:

```text
ComfyUI/models/deev_censor/nsfw-anime-medium-x1280.onnx
```

An existing model with an unexpected size or SHA256 is rejected. It is replaced
only when `download_model.py --repair` is run explicitly.

## Node

```text
Deev Genital/Anus Censor (01miku)
```

Input and output are standard ComfyUI `IMAGE` batches.

`detection_confidence` defaults to the conservative value `0.05`. Lower values
accept more weak detections and can increase false positives. The node bounds
the setting to `0.01` through `0.15`; the upper bound is the previous production
threshold, so a workflow cannot configure a looser policy than before.

`enable_tiled_retry` defaults to off. When enabled, a full-image anus miss runs
up to four overlapping crop inferences at the same model input size. This makes
small rear-view targets larger to the detector. Only tile-level `anus`
detections at confidence `0.10` or higher are accepted; this stricter,
class-specific fallback avoids the large number of penis/vagina false positives
observed when tiled crops were evaluated at `0.05`. Tile detections are mapped
back by rendering through each crop's own letterbox and segmentation prototype.
Overlapping results are composed conservatively; multiple or crop-edge
detections use mosaic instead of a white segmentation fill.

The retry only runs when the full-image pass detects zero anus instances. A
zero result after all retry views is logged, but an IMAGE-only node cannot tell
whether an otherwise safe image truly contains no anus or the detector missed
one. A strict adult pipeline still needs a separate reject/regenerate policy if
zero-after-retry must be blocked.

## Pinned model

- Source: `01miku/anime-nsfw-segm-yolo26`
- Revision: `1697d5d1827b6a818b350b44bf3ec27f08837a2a`
- File: `nsfw-anime-medium-x1280.onnx`
- Size: `47,600,269` bytes
- SHA256: `a12ac5532e93be9dfeb96a77fc3f3647791335c9df0de9a18fcd503f7877a828`

See `THIRD_PARTY_NOTICES.md` before production or commercial use.
