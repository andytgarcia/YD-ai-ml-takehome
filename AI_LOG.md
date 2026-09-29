# AI Usage Log

## 1. Step 1 setup and dataset downloader — 2026-09-29

**What I asked:** Read the supplied `plan.md` and implement Step 1: reproducible
dependencies and a credential-safe dataset downloader.

**What was retained or changed, and why:** Retained the plan's YOLOv8 direction,
fixed Roboflow version, environment-variable credentials, and read-only treatment
of the source dataset. Added `download_data.py`, pinned the direct dependencies,
documented setup, and added downloader contract tests. Used the official headless
Ultralytics distribution to share Roboflow's OpenCV dependency. Added temporary
download staging and basic export-completeness checks so failed downloads are not
mistaken for reusable copies. Later plan steps remain unimplemented.

**Problems caught during AI-assisted work:** The installed Roboflow SDK can print
credential-bearing URLs before raising an exception, so suppressing the final
exception alone would be insufficient. Its download method also skips an existing
destination; the staging directory therefore contains a nonexistent child for the
SDK to create. These behaviors were checked against the installed SDK source.

**Verification performed by the coding agent:** Installed `requirements.txt` in
the existing Python 3.12.0 environment on macOS 27.0 arm64; `pip check` passed.
All 10 offline downloader tests passed, covering reuse, incomplete/wrong exports,
credential loading and precedence, SDK error/output suppression, failed-download
cleanup/retry, and a destination created during download. Ran the script against
the real dataset without a key. Compared SHA-256 hashes and modification times for
all 2,003 source/sampling files before and after; none changed. Tested library
imports, image operations, NumPy/Torch conversion, Torchvision NMS, and synthetic
CPU prediction with a randomly initialized YOLOv8n. No trained accuracy was measured.
No live authenticated download or fresh-clone workflow was tested. These are agent
checks, not a claim of independent review by the project author.
