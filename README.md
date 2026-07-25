# Smart Livestock Gate

Smart Livestock Gate is a computer-vision project for detecting, classifying,
tracking, and counting farm animals as they pass through a virtual gate. The
current working milestone is a HOG and colour-histogram KNN species classifier
with a command-line interface, Flask API, and React client.

![High-level system architecture](docs/architecture.png)

## Repository layout

```text
.
|-- .github/workflows/       Continuous integration
|-- artifacts/               Generated models, predictions, and reports
|-- configs/                 Version-controlled experiment configuration
|-- data/
|   |-- raw/                 Immutable source datasets
|   |-- interim/             Partially processed data
|   `-- processed/           Model-ready data and annotations
|-- docs/                    Architecture and design documentation
|-- frontend/                React and Vite web client
|-- notebooks/               Exploratory analysis and evaluation
|-- scripts/                 Reproducible workflow automation
|-- src/smart_livestock_gate/
|   |-- api/                 Flask application
|   |-- baseline/            HOG, colour features, and KNN classifier
|   |-- counting/            Counting utilities
|   `-- data/                Data loading and preprocessing code
|-- tests/                   Automated Python tests
`-- pyproject.toml           Python package and tool configuration
```

Keeping importable application logic under `src/` prevents accidental imports
from the repository root. Datasets and generated outputs are separated from
code so future detection, tracking, line-crossing, and evaluation components
can be added without mixing concerns.

## Current dataset

The supplied Animal Counting Dataset is located at
`data/raw/animal_counting_dataset/`. It contains 2,000 images for each of five
classes: `chicken`, `cow`, `goat`, `horse`, and `sheep`. The proposed `pig` and
`turkey` classes are not currently included.

Large datasets should be fetched reproducibly and kept out of future commits.
See `data/README.md` and `scripts/README.md` for the intended conventions.

### CattleEyeView integration

The extracted CattleEyeView release is stored inside the project at
`data/raw/cattle_eye_view/`. It contains 14 videos, 30,703 frames, 753 tracked
cattle, body/head detections, 24-keypoint poses, segmentation labels, and
sequence-level crossing counts. The directory is ignored by Git because it is
too large for source control.

Validate the full release:

```powershell
livestock-gate dataset validate
```

The normal validation checks every file association and samples annotation
contents. Use `--deep-labels` to parse every coordinate in every segmentation
polygon, or `--skip-labels` for a quick structural check.

Export a corrected tracking manifest and count metadata:

```powershell
livestock-gate dataset export
```

This writes generated files to `data/processed/cattle_eye_view/`. The loader
normalizes a known inconsistency in the source COCO `image_id` values by using
the annotation `file_name`, which matches the actual frames.

Prepare an Ultralytics-compatible dataset without copying the large images:

```powershell
livestock-gate dataset prepare-yolo detect
livestock-gate dataset prepare-yolo detect_head
livestock-gate dataset prepare-yolo pose
livestock-gate dataset prepare-yolo segment
```

Each command creates hardlinks, a manifest CSV, and a local `dataset.yaml`
under `data/raw/cattle_eye_view/prepared/<task>/`.

### Train the cattle detector

Install the optional vision dependencies and prepare the detection layout:

```powershell
python -m pip install -e ".[dev,vision]"
livestock-gate dataset prepare-yolo detect
```

On an NVIDIA system, install the appropriate CUDA-enabled PyTorch wheel from
the official PyTorch selector. Confirm `torch.cuda.is_available()` is true
before starting a long run.

Run the reproducible detector experiment:

```powershell
livestock-gate train-detector
```

The default experiment fine-tunes a nano detector for 20 epochs at 512 px,
evaluates the best checkpoint on the official test split, and writes local
weights, plots, and a machine-readable metrics report under `artifacts/`.
Those generated files are intentionally ignored by Git.

## Backend setup

Python 3.10 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The optional deep-learning and tracking dependencies from the full project
specification can be installed with:

```powershell
python -m pip install -e ".[dev,vision]"
```

## Train and predict

The default training run samples 500 images per class so the KNN experiment is
reasonable on a laptop:

```powershell
livestock-gate train
```

Use all supplied images by setting the per-class limit to zero:

```powershell
livestock-gate train --max-per-class 0
```

Classify an image or every supported image in a directory:

```powershell
livestock-gate predict path\to\image.png
livestock-gate predict path\to\folder
```

The trained baseline is saved to `artifacts/models/animal_knn.joblib`. You can
also run the CLI without installing its console command:

```powershell
python -m smart_livestock_gate train
```

## API and web client

Start the backend from an activated Python environment:

```powershell
flask --app smart_livestock_gate.api run --debug --port 5000
```

In a second terminal, start the web client:

```powershell
cd frontend
npm ci
npm run dev
```

Copy `frontend/.env.example` to `frontend/.env` only when the API is hosted at a
different address.

The dashboard uses two read-only integration endpoints:

- `GET /api/health` checks backend connectivity.
- `GET /api/status` reports dataset, annotation, and model readiness without
  exposing machine-specific paths.
- `GET /api/detection/examples` lists allowlisted held-out frames.
- `GET /api/detection/examples/<id>/image` serves an allowlisted local frame.
- `POST /api/detection/examples/<id>/predict` returns normalized cattle boxes,
  confidence scores, counts, and inference time.

The frontend includes a detector lab for comparing AI predictions with human
annotations on held-out examples. Tracking, virtual-gate counting,
asynchronous video jobs, and result playback are laid out with acceptance
criteria in [the implementation roadmap](docs/roadmap.md).

## Quality checks

```powershell
ruff check src tests
pytest

cd frontend
npm run lint
npm run build
```

The same checks run in GitHub Actions for pushes and pull requests.

## Scope

The current prototype assigns one label to each input image. The full project
will add multi-animal detection, persistent tracking IDs, directional
line-crossing counts, annotated video and structured logs, error decomposition,
and domain-shift evaluation. Those features are not represented as complete in
the current codebase.
