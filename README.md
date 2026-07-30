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

## Pinned model

- Source: `01miku/anime-nsfw-segm-yolo26`
- Revision: `1697d5d1827b6a818b350b44bf3ec27f08837a2a`
- File: `nsfw-anime-medium-x1280.onnx`
- Size: `47,600,269` bytes
- SHA256: `a12ac5532e93be9dfeb96a77fc3f3647791335c9df0de9a18fcd503f7877a828`

See `THIRD_PARTY_NOTICES.md` before production or commercial use.
