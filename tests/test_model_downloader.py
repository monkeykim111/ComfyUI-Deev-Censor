from __future__ import annotations

import io
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1]))

import model_downloader


class ModelDownloaderUnitTests(unittest.TestCase):
    def test_missing_model_is_downloaded_and_installed_atomically(self):
        payload = b"pinned model"

        class FakeResponse(io.BytesIO):
            headers = {"Content-Length": str(len(payload))}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / model_downloader.MODEL_FILENAME
            with (
                mock.patch.object(
                    model_downloader,
                    "EXPECTED_MODEL_SIZE",
                    len(payload),
                ),
                mock.patch.object(
                    model_downloader,
                    "EXPECTED_MODEL_SHA256",
                    model_downloader.hashlib.sha256(payload).hexdigest(),
                ),
                mock.patch.object(
                    model_downloader,
                    "urlopen",
                    return_value=FakeResponse(payload),
                ),
            ):
                self.assertEqual(model_downloader.ensure_model(path), path)
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(list(path.parent.glob("*.part")), [])

    def test_valid_existing_model_is_reused_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / model_downloader.MODEL_FILENAME
            path.write_bytes(b"valid")
            with (
                mock.patch.object(
                    model_downloader,
                    "EXPECTED_MODEL_SIZE",
                    len(b"valid"),
                ),
                mock.patch.object(
                    model_downloader,
                    "EXPECTED_MODEL_SHA256",
                    model_downloader.hashlib.sha256(b"valid").hexdigest(),
                ),
                mock.patch.object(
                    model_downloader,
                    "_download_to_temporary_file",
                ) as download,
            ):
                self.assertEqual(model_downloader.ensure_model(path), path)
                download.assert_not_called()

    def test_invalid_existing_model_is_not_replaced_implicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / model_downloader.MODEL_FILENAME
            path.write_bytes(b"invalid")
            with self.assertRaisesRegex(
                model_downloader.ModelDownloadError,
                "size mismatch",
            ):
                model_downloader.ensure_model(path)
            self.assertEqual(path.read_bytes(), b"invalid")


if __name__ == "__main__":
    unittest.main()
