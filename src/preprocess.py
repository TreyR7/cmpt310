import cv2 as cv
import os

# preprocess a single image
def preprocess_image(path, size=(128, 128)):
  # load image from path
  image = cv.imread(path)
  # raise error of image is not found
  if image is None:
    raise ValueError(f"Could not load image at {path}")
  # return resized image
  return cv.resize(image, size)

def preprocess_directory(directory, size=(128, 128)):
  # preprocess all images in folder, return list of (image, filename) pairs
  results = []
  for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    try:
      image = preprocess_image(path, size)
      results.append((image, filename))
    except ValueError as e:
      print(f"Skipping {filename}: {e}")
  return results
