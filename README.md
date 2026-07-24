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
