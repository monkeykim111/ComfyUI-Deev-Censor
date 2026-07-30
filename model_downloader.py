"""Pinned, atomic downloader for the Deev censor model."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL_SUBDIRECTORY = "deev_censor"
MODEL_FILENAME = "nsfw-anime-medium-x1280.onnx"
MODEL_REVISION = "1697d5d1827b6a818b350b44bf3ec27f08837a2a"
MODEL_URL = (
    "https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/"
    f"{MODEL_REVISION}/{MODEL_FILENAME}"
)
EXPECTED_MODEL_SIZE = 47_600_269
EXPECTED_MODEL_SHA256 = (
    "a12ac5532e93be9dfeb96a77fc3f3647791335c9df0de9a18fcd503f7877a828"
)

DOWNLOAD_TIMEOUT_SECONDS = 300
LOCK_TIMEOUT_SECONDS = 360
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class ModelDownloadError(RuntimeError):
    """Raised when the pinned model cannot be made available safely."""


def _try_lock(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        if lock_file.read(1) == b"":
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _model_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f"{path.name}.download.lock")
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    try:
        lock_file = lock_path.open("a+b")
    except OSError as error:
        raise ModelDownloadError(
            f"cannot open Deev censor model download lock: {lock_path}",
        ) from error

    locked = False
    try:
        while not locked:
            try:
                _try_lock(lock_file)
                locked = True
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise ModelDownloadError(
                        "timed out waiting for Deev censor model download "
                        f"lock: {path}",
                    ) from error
                time.sleep(0.1)
        yield
    finally:
        if locked:
            _unlock(lock_file)
        lock_file.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as model_file:
            while chunk := model_file.read(DOWNLOAD_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as error:
        raise ModelDownloadError(f"cannot read Deev censor model: {path}") from error
    return digest.hexdigest()


def verify_model(path: Path) -> None:
    try:
        stat = path.stat()
    except OSError as error:
        raise ModelDownloadError(f"Deev censor model is unavailable: {path}") from error
    if not path.is_file():
        raise ModelDownloadError(f"Deev censor model is not a file: {path}")
    if stat.st_size != EXPECTED_MODEL_SIZE:
        raise ModelDownloadError(
            "Deev censor model size mismatch: "
            f"expected {EXPECTED_MODEL_SIZE}, got {stat.st_size}",
        )
    if sha256_file(path) != EXPECTED_MODEL_SHA256:
        raise ModelDownloadError(
            f"Deev censor model SHA256 mismatch: expected {EXPECTED_MODEL_SHA256}",
        )


def _download_to_temporary_file(target: Path) -> Path:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    request = Request(
        MODEL_URL,
        headers={"User-Agent": "ComfyUI-Deev-Censor/1.0"},
    )
    digest = hashlib.sha256()
    downloaded = 0

    try:
        with (
            urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            temporary.open("xb") as output,
        ):
            content_length = response.headers.get("Content-Length")
            if (
                content_length is not None
                and int(content_length) != EXPECTED_MODEL_SIZE
            ):
                raise ModelDownloadError(
                    "Deev censor model download size header mismatch",
                )

            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                downloaded += len(chunk)
                if downloaded > EXPECTED_MODEL_SIZE:
                    raise ModelDownloadError(
                        "Deev censor model download exceeded expected size",
                    )
                digest.update(chunk)
                output.write(chunk)

            output.flush()
            os.fsync(output.fileno())
    except ModelDownloadError:
        temporary.unlink(missing_ok=True)
        raise
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"cannot download pinned Deev censor model from {MODEL_URL}",
        ) from error

    if downloaded != EXPECTED_MODEL_SIZE:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            "Deev censor model download size mismatch: "
            f"expected {EXPECTED_MODEL_SIZE}, got {downloaded}",
        )
    if digest.hexdigest() != EXPECTED_MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            "Deev censor model download SHA256 mismatch",
        )
    return temporary


def ensure_model(path: Path, *, repair: bool = False) -> Path:
    """Return a verified model, downloading only when it is absent."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ModelDownloadError(
            f"cannot create Deev censor model directory: {path.parent}",
        ) from error

    with _model_lock(path):
        if path.exists():
            try:
                verify_model(path)
                return path
            except ModelDownloadError:
                if not repair:
                    raise

        temporary = _download_to_temporary_file(path)
        try:
            os.replace(temporary, path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"cannot install Deev censor model at {path}",
            ) from error
        verify_model(path)
        return path


__all__ = [
    "EXPECTED_MODEL_SHA256",
    "EXPECTED_MODEL_SIZE",
    "MODEL_FILENAME",
    "MODEL_REVISION",
    "MODEL_SUBDIRECTORY",
    "MODEL_URL",
    "ModelDownloadError",
    "ensure_model",
    "sha256_file",
    "verify_model",
]
