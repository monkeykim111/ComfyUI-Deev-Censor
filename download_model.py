"""Prefetch the pinned Deev censor model before ComfyUI readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model_downloader import (
    MODEL_FILENAME,
    MODEL_SUBDIRECTORY,
    ensure_model,
)


def _default_models_directory() -> Path:
    comfy_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(comfy_root))
    try:
        import folder_paths
    except ImportError as error:
        raise RuntimeError(
            "cannot locate ComfyUI; pass --models-dir explicitly",
        ) from error
    return Path(folder_paths.models_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="ComfyUI models directory; detected automatically when omitted",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="replace an existing model that fails size or SHA256 validation",
    )
    args = parser.parse_args()

    models_directory = (
        args.models_dir.resolve()
        if args.models_dir is not None
        else _default_models_directory()
    )
    model_path = models_directory / MODEL_SUBDIRECTORY / MODEL_FILENAME
    installed = ensure_model(model_path, repair=args.repair)
    print(f"verified Deev censor model: {installed}")


if __name__ == "__main__":
    main()
