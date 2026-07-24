"""OpenCV image loading and preprocessing."""

from pathlib import Path

import cv2 as cv

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(path, size=(64, 64)):
    """Load an image as BGR and resize it to a consistent size."""
    path = Path(path)
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image at {path}")
    return cv.resize(image, size, interpolation=cv.INTER_AREA)


def image_paths(directory):
    """Return supported image paths in a directory in deterministic order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def preprocess_directory(directory, size=(64, 64)):
    """Preprocess every readable image and return (image, filename) pairs."""
    results = []
    for path in image_paths(directory):
        try:
            results.append((preprocess_image(path, size), path.name))
        except ValueError as error:
            print(f"Skipping {path.name}: {error}")
    return results
