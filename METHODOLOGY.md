# Methodology

Step 1 (dependencies and dataset download) is implemented. Training, data auditing,
evaluation, inference, and the video experiment are still pending; see `plan.md`.

## 1. How to run it

Tested on macOS 27.0, Apple Silicon (arm64), with Python 3.12.0. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

For a new download, set `ROBOFLOW_API_KEY` in your shell environment, or copy
`.env.example` to `.env` and fill in that variable. An exported environment value
takes precedence over `.env`; the file is loaded from beside `download_data.py`.
Keep credentials local. Then run:

```bash
python download_data.py
```

This downloads workspace `malletbottle2`, project `sampled-yd-object-detection`,
version **2**, in **YOLOv8** format, into `Sampled-YD-Object-Detection-2/` beside
the script. It checks the export metadata, class order (`0=bottle`, `1=mallet`),
and 800 training / 199 validation image-label pairs before accepting it.
The existing dataset is reused without requiring a key or contacting Roboflow.
An incomplete or unrelated existing folder causes an error and is left untouched.
To download into a different, git-ignored location:

```bash
python download_data.py --output-dir data/raw/Sampled-YD-Object-Detection-2
```

New downloads are staged in a temporary sibling directory, checked, and moved to
the destination. Failed downloads are cleaned up so they can be retried. Roboflow
output and exception details are suppressed because they can contain credentials.
The original `data.yaml`, including its unused `test` path, is not corrected here;
Step 2 will write a separate prepared dataset.

Run the downloader's offline checks with:

```bash
python -m unittest discover -s tests -v
```

Dependency validation also included imports of every pinned library, Pillow and
OpenCV image operations, NumPy/Torch conversion, Torchvision non-maximum suppression,
and CPU prediction using `YOLO("yolov8n.yaml")` on a synthetic 64×64 image. That model
uses random initialization and verifies compatibility, not trained performance.

## 2. Thought process

- Pin all direct dependencies in `requirements.txt`, including the matching
  [PyTorch/Torchvision pair](https://pytorch.org/get-started/previous-versions/).
  Transitive dependencies are resolved by pip; this is not a complete environment lockfile.
- Use the [official headless Ultralytics distribution](https://docs.ultralytics.com/quickstart/).
  It still imports as `ultralytics` and supports YOLOv8, but shares
  `opencv-python-headless` with Roboflow instead of installing two packages that
  both provide `cv2`. Image saving works; OpenCV GUI windows are not included.
- Reuse and preserve the source download. Completeness checks here do not replace
  the annotation, image-readability, or duplicate audit planned for Step 2.
- Keep the dataset, `.env`, and temporary download directories out of commits.

## 3. Known limitations

- The live authenticated Roboflow download was not exercised: no API key was
  configured. Download success and failure paths were tested with a simulated SDK;
  reuse was tested against the actual 999-image download.
- All 10 downloader tests and `pip check` passed. File hashes and modification
  times confirmed that all 2,003 files in `sampling/` and the downloaded dataset
  were unchanged.
- Dependency installation and the CPU smoke check were tested in the project's
  existing virtual environment, not yet from a fresh clone. Other OS/GPU builds
  and full training are untested. The submission workflow still needs the final
  fresh-clone verification described in `plan.md`.
