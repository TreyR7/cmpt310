# Data directory

Data is separated by processing stage:

- `raw/`: immutable source datasets as downloaded.
- `interim/`: temporary or partially transformed data.
- `processed/`: model-ready splits, annotations, and features.

Large datasets should be obtained through reproducible scripts and should not be
added to Git. The currently supplied Animal Counting Dataset lives under
`raw/animal_counting_dataset/`.

