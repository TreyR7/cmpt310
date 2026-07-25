# Data directory

Data is separated by processing stage:

- `raw/`: immutable source datasets as downloaded.
- `interim/`: temporary or partially transformed data.
- `processed/`: model-ready splits, annotations, and features.

Large datasets should be obtained through reproducible scripts and should not be
added to Git. The currently supplied Animal Counting Dataset lives under
`raw/animal_counting_dataset/`.

## CattleEyeView

CattleEyeView is stored locally at `raw/cattle_eye_view/`. This keeps the
project self-contained while Git ignores the large source images, videos, and
generated training layouts.

The integration keeps the source release immutable and writes normalized
tracking/count manifests under `processed/cattle_eye_view/`. Ultralytics-ready
datasets are generated under `raw/cattle_eye_view/prepared/<task>/` using
hardlinks, so the 30,703 source images are not duplicated.
