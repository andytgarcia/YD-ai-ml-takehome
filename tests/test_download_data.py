"""Downloader contracts; all network responses are simulated."""

from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

import download_data


def make_export(root: Path) -> None:
    root.mkdir()
    config = {
        "names": ["bottle", "mallet"], "nc": 2,
        "roboflow": {"workspace": "malletbottle2", "project": "sampled-yd-object-detection", "version": 2},
        "train": "../train/images", "val": "../valid/images", "test": "../test/images",
    }
    (root / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    for split, count in (("train", 800), ("valid", 199)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for index in range(count):
            # The downloader checks completeness; decoding and auditing are Step 2.
            (root / split / "images" / f"{index}.jpg").write_bytes(b"image fixture")
            (root / split / "labels" / f"{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "download"
        self.sdk = Mock()
        self.version = self.sdk.return_value.workspace.return_value.project.return_value.version.return_value
        for patcher in (
            patch.dict(os.environ, {}, clear=True),
            patch.object(download_data, "PROJECT_ROOT", self.root),
            patch.dict(sys.modules, {"roboflow": SimpleNamespace(Roboflow=self.sdk)}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def run_cli(self):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return download_data.main(["--output-dir", str(self.output)])

    def successful_download(self, model_format, *, location, overwrite):
        self.assertEqual(model_format, "yolov8")
        self.assertFalse(overwrite)
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(location).exists())
        make_export(Path(location))

    def test_existing_export_is_reused_without_credentials_or_changes(self):
        make_export(self.output)
        before = snapshot(self.output)
        self.assertEqual(self.run_cli(), 0)
        self.sdk.assert_not_called()
        self.assertEqual(snapshot(self.output), before)
        self.assertIn("Reusing existing dataset", self.stdout.getvalue())

    def test_existing_partial_export_is_never_overwritten(self):
        make_export(self.output)
        (self.output / "valid" / "labels" / "0.txt").unlink()
        before = snapshot(self.output)
        os.environ["ROBOFLOW_API_KEY"] = "fake-key"
        self.assertEqual(self.run_cli(), 1)
        self.sdk.assert_not_called()
        self.assertEqual(snapshot(self.output), before)

    def test_existing_file_and_empty_directory_are_preserved(self):
        self.output.write_text("keep me")
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.output.read_text(), "keep me")
        self.output.unlink()
        self.output.mkdir()
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(list(self.output.iterdir()), [])
        self.sdk.assert_not_called()

    def test_wrong_version_or_class_mapping_is_rejected(self):
        make_export(self.output)
        config_path = self.output / "data.yaml"
        original = yaml.safe_load(config_path.read_text())
        for field, value in (("names", ["mallet", "bottle"]), ("roboflow", {"version": 3})):
            with self.subTest(field=field):
                config_path.write_text(yaml.safe_dump({**original, field: value}))
                before = snapshot(self.output)
                self.assertEqual(self.run_cli(), 1)
                self.assertEqual(snapshot(self.output), before)
        self.sdk.assert_not_called()

    def test_missing_key_fails_without_creating_output(self):
        self.assertEqual(self.run_cli(), 1)
        self.assertIn("Set ROBOFLOW_API_KEY", self.stderr.getvalue())
        self.assertFalse(self.output.exists())
        self.sdk.assert_not_called()

    def test_download_uses_exact_project_and_environment_over_dotenv(self):
        (self.root / ".env").write_text("ROBOFLOW_API_KEY=dotenv-test-key\n")
        os.environ["ROBOFLOW_API_KEY"] = "environment-test-key"
        self.version.download.side_effect = self.successful_download
        self.assertEqual(self.run_cli(), 0)
        self.sdk.assert_called_once_with(api_key="environment-test-key")
        self.sdk.return_value.workspace.assert_called_once_with("malletbottle2")
        self.sdk.return_value.workspace.return_value.project.assert_called_once_with("sampled-yd-object-detection")
        self.sdk.return_value.workspace.return_value.project.return_value.version.assert_called_once_with(2)
        download_data.check_download(self.output)
        self.assertEqual(list(self.root.glob(".roboflow-download-*")), [])

    def test_dotenv_key_is_used_when_environment_is_unset(self):
        (self.root / ".env").write_text("ROBOFLOW_API_KEY=dotenv-test-key\n")
        self.version.download.side_effect = self.successful_download
        self.assertEqual(self.run_cli(), 0)
        self.sdk.assert_called_once_with(api_key="dotenv-test-key")

    def test_sdk_errors_and_output_never_expose_keys_and_allow_retry(self):
        secret = "synthetic-secret-for-test"
        os.environ["ROBOFLOW_API_KEY"] = secret
        # Bind a handler before stream redirection, as libraries sometimes do.
        handler = logging.StreamHandler(self.stderr)
        logger = logging.getLogger("download-test")
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        previous_logging_level = logging.root.manager.disable

        def fail(*args, **kwargs):
            Path(kwargs["location"]).mkdir()
            print(f"https://example.invalid/?api_key={secret}")
            print(secret, file=sys.stderr)
            logger.error(secret)
            raise RuntimeError(secret)

        self.version.download.side_effect = fail
        self.assertEqual(self.run_cli(), 1)
        self.assertNotIn(secret, self.stdout.getvalue() + self.stderr.getvalue())
        self.assertNotIn("Traceback", self.stderr.getvalue())
        self.assertEqual(logging.root.manager.disable, previous_logging_level)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".roboflow-download-*")), [])
        self.version.download.side_effect = self.successful_download
        self.assertEqual(self.run_cli(), 0)

    def test_incomplete_download_is_not_published(self):
        os.environ["ROBOFLOW_API_KEY"] = "test-key"
        self.version.download.side_effect = lambda *args, **kwargs: Path(kwargs["location"]).mkdir()
        self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".roboflow-download-*")), [])

    def test_destination_created_during_download_is_preserved(self):
        os.environ["ROBOFLOW_API_KEY"] = "test-key"

        def download(*args, **kwargs):
            self.successful_download(*args, **kwargs)
            self.output.mkdir()
            (self.output / "keep.txt").write_text("keep me")

        self.version.download.side_effect = download
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual((self.output / "keep.txt").read_text(), "keep me")
        self.assertEqual(list(self.root.glob(".roboflow-download-*")), [])


if __name__ == "__main__":
    unittest.main()
