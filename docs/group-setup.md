# Group setup guide

This guide explains how to set up Smart Livestock Gate after cloning the
repository. The large CattleEyeView files are not stored in Git, so every group
member must place the downloaded data in the same repository-relative location.

## 1. Clone the repository

```powershell
git clone <repository-url>
cd cmpt310
```

All commands below should be run from this `cmpt310` directory unless stated
otherwise.

## 2. Download the required files

Obtain these files from the shared group storage:
https://drive.google.com/file/d/1tJbLjV9MK3XWcQXfS_XiQCZTxDHCDxC7/view?usp=sharing

- `images.tar.gz`
- `drive-download-20260725T000131Z-1-001.zip`
- `cattle_detector.pt` if the trained checkpoint is not included in Git

The two dataset archives are large and should remain outside the repository.
Only their extracted contents belong under `data/raw/`.

## 3. Extract CattleEyeView

Create the dataset directory:

```powershell
New-Item -ItemType Directory -Force data\raw\cattle_eye_view
```

Extract the image archive into that directory. Replace the example archive
path with its location on your computer:

```powershell
tar -xzf "C:\path\to\images.tar.gz" `
    -C "data\raw\cattle_eye_view"
```

Extract the annotation and video archive into the same directory:

```powershell
Expand-Archive `
    -Path "C:\path\to\drive-download-20260725T000131Z-1-001.zip" `
    -DestinationPath "data\raw\cattle_eye_view" `
    -Force
```

After extraction, the important directories must be arranged exactly as shown
below:

```text
cmpt310/
|-- data/
|   `-- raw/
|       `-- cattle_eye_view/
|           |-- annotation/
|           |   |-- detect/
|           |   |-- detect_head/
|           |   |-- pose/
|           |   |-- pose_COCO/
|           |   |-- segment/
|           |   `-- metadata and count.xlsx
|           |-- images/
|           |   |-- 01.mp4/
|           |   |-- 02.mp4/
|           |   `-- ...
|           `-- videos/
|               |-- 01.mp4
|               |-- 02.mp4
|               `-- ...
`-- artifacts/
    `-- models/
        `-- cattle_detector.pt
```

Do not leave an extra nested directory such as:

```text
data/raw/cattle_eye_view/cattle_eye_view/images/
```

The `annotation`, `images`, and `videos` directories must be directly inside
`data/raw/cattle_eye_view/`.

## 4. Add the trained checkpoint

To run the existing cattle detector without training it again, place the shared
checkpoint here:

```text
artifacts/models/cattle_detector.pt
```

Create the directory first if necessary:

```powershell
New-Item -ItemType Directory -Force artifacts\models
Copy-Item "C:\path\to\cattle_detector.pt" artifacts\models\cattle_detector.pt
```

If the checkpoint is unavailable, the detector can be trained after the data
has been validated and prepared. Training is optional for group members who
only need to run the existing demonstration.

## 5. Create the Python environment

Python 3.10 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vision]"
```

If PowerShell prevents activation, run this once in the current terminal and
then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 6. Validate and prepare the dataset

```powershell
livestock-gate dataset validate
livestock-gate dataset prepare-yolo detect
```

The second command generates the local detection layout used by training and
the frontend examples. Do not manually copy files into `data/processed/` or a
`prepared/` directory; the project creates those outputs automatically.

## 7. Start the backend

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m flask --app smart_livestock_gate.api run --port 5000
```

The API should become available at <http://localhost:5000>.

## 8. Start the frontend

Open a second PowerShell terminal and run:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173> in a browser. Keep the backend terminal running
while using the detection demonstration.

## 9. Quick verification

The following should all be true before testing the detector:

- `data/raw/cattle_eye_view/images/01.mp4/` contains JPG frames.
- `data/raw/cattle_eye_view/annotation/` contains the supplied annotations.
- `data/raw/cattle_eye_view/videos/01.mp4` exists.
- `data/raw/cattle_eye_view/prepared/detect/dataset.yaml` was generated.
- `artifacts/models/cattle_detector.pt` exists.
- <http://localhost:5000/api/health> responds successfully.
- <http://localhost:5173> loads the frontend.

The repository uses paths relative to its own root. It can therefore be cloned
anywhere as long as each member keeps the directory structure above.
