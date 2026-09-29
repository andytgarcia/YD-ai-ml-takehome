"""Download the take-home dataset without changing an existing local copy."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Sampled-YD-Object-Detection-2"
WORKSPACE = "malletbottle2"
PROJECT = "sampled-yd-object-detection"
VERSION = 2
FORMAT = "yolov8"
SPLIT_COUNTS = {"train": 800, "valid": 199}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DownloadError(Exception):
    """A user-facing error whose message contains no SDK response or API key."""


def check_download(root: Path) -> None:
    """Check export identity and completeness; annotation auditing comes later."""
    try:
        config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        metadata = config["roboflow"]
        if (
            metadata["workspace"] != WORKSPACE
            or metadata["project"] != PROJECT
            or metadata["version"] != VERSION
            or config["names"] != ["bottle", "mallet"]
            or config["nc"] != 2
        ):
            raise ValueError
        for split, expected in SPLIT_COUNTS.items():
            images = [
                p for p in (root / split / "images").iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
            labels = [p for p in (root / split / "labels").glob("*.txt") if p.is_file()]
            image_stems = {p.stem for p in images}
            if (
                len(images) != expected
                or len(labels) != expected
                or len(image_stems) != expected
                or image_stems != {p.stem for p in labels}
            ):
                raise ValueError
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError, ValueError):
        raise DownloadError(
            "The dataset folder is incomplete or does not match the required version 2 "
            "export (800 train / 199 valid image-label pairs; bottle=0, mallet=1). "
            "No existing files were changed. Choose a new --output-dir to download again."
        ) from None


def _download_export(api_key: str, destination: Path) -> None:
    # The SDK prints signed download URLs on failures. Discard both streams and
    # logging, including during import/authentication, and never echo exceptions.
    previous_logging_level = logging.root.manager.disable
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            logging.disable(sys.maxsize)
            from roboflow import Roboflow

            version = Roboflow(api_key=api_key).workspace(WORKSPACE).project(PROJECT).version(VERSION)
            version.download(FORMAT, location=str(destination), overwrite=False)
    except (Exception, SystemExit):
        raise DownloadError(
            "Roboflow download failed. Check ROBOFLOW_API_KEY, dataset access, network "
            "connectivity, TLS certificates, and the installed requirements. "
            "SDK output was suppressed to protect credentials."
        ) from None
    finally:
        logging.disable(previous_logging_level)


def ensure_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Reuse a complete export, or download to a temporary sibling first."""
    output_dir = Path(output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        check_download(output_dir)
        print(f"Reusing existing dataset: {output_dir}")
        return output_dir

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise DownloadError(
            "Set ROBOFLOW_API_KEY in the environment or the project's git-ignored .env "
            "file before downloading (see .env.example)."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {WORKSPACE}/{PROJECT}, version {VERSION} ({FORMAT})...")
    with TemporaryDirectory(prefix=".roboflow-download-", dir=output_dir.parent) as temporary:
        # Pass a nonexistent child: the SDK skips downloads if the location exists.
        staged = Path(temporary) / "dataset"
        _download_export(api_key, staged)
        check_download(staged)
        if output_dir.exists() or output_dir.is_symlink():
            raise DownloadError("The output path appeared during download; refusing to replace it.")
        staged.rename(output_dir)
    print(f"Dataset ready: {output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Dataset destination (default: Sampled-YD-Object-Detection-2 beside this script).",
    )
    args = parser.parse_args(argv)
    try:
        ensure_dataset(args.output_dir)
    except DownloadError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("Error: could not read or write the dataset destination. Check permissions and disk space.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Download cancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
