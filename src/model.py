from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_extraction import extract_features
from src.preprocess import image_paths, preprocess_image


def load_dataset(dataset_dir, max_per_class=500, seed=42):
    """Load labelled class folders and extract OpenCV features."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("The dataset must contain at least two class folders")

    rng = np.random.default_rng(seed)
    features, labels = [], []
    for class_dir in class_dirs:
        paths = image_paths(class_dir)
        if max_per_class is not None and len(paths) > max_per_class:
            indices = np.sort(
                rng.choice(len(paths), size=max_per_class, replace=False)
            )
            paths = [paths[index] for index in indices]

        loaded = 0
        for path in paths:
            try:
                features.append(extract_features(preprocess_image(path)))
                labels.append(class_dir.name)
                loaded += 1
            except ValueError as error:
                print(f"Skipping {path}: {error}")
        print(f"Loaded {loaded:>4} {class_dir.name} images")

    if not features:
        raise ValueError(f"No readable images found in {dataset_dir}")
    return np.vstack(features), np.asarray(labels)


def build_model(neighbors=5):
    """Create a scaled, distance-weighted KNN pipeline."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=neighbors,
                    weights="distance",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(features, labels, neighbors=5, test_size=0.2, seed=42):
    """Fit on a stratified training split and print holdout metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    model = build_model(neighbors)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    class_names = sorted(np.unique(labels))

    print("\nClassification report")
    print(classification_report(y_test, predictions, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted)")
    print("Labels:", ", ".join(class_names))
    print(confusion_matrix(y_test, predictions, labels=class_names))
    return model


def save_model(model, path):
    """Persist a trained scikit-learn pipeline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load a previously trained pipeline."""
    return joblib.load(path)


def predict_paths(model, paths):
    """Return (path, predicted_label) pairs for readable images."""
    valid_paths, features = [], []
    for path in paths:
        try:
            features.append(extract_features(preprocess_image(path)))
            valid_paths.append(Path(path))
        except ValueError as error:
            print(f"Skipping {path}: {error}")
    if not features:
        raise ValueError("No readable images were supplied")
    predictions = model.predict(np.vstack(features))
    return list(zip(valid_paths, predictions))
