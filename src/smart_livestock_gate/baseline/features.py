import cv2 as cv
import numpy as np

_HOG = cv.HOGDescriptor(
    (64, 64),
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)


def extract_hog(image):
    """Extract Histogram of Oriented Gradients shape features."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return _HOG.compute(gray).reshape(-1).astype(np.float32)


def extract_color_histogram(image, bins=16):
    """Extract normalized hue, saturation, and value histograms."""
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    histograms = []
    ranges = ((0, 180), (0, 256), (0, 256))
    for channel, value_range in enumerate(ranges):
        histogram = cv.calcHist(
            [hsv], [channel], None, [bins], list(value_range)
        ).reshape(-1)
        histogram /= histogram.sum() + 1e-7
        histograms.append(histogram)
    return np.concatenate(histograms).astype(np.float32)


def combine_features(hog_vector, color_vector):
    """Combine shape and colour into one feature vector."""
    return np.concatenate((hog_vector, color_vector)).astype(np.float32)


def extract_features(image):
    """Extract the complete feature vector for one preprocessed image."""
    return combine_features(extract_hog(image), extract_color_histogram(image))
