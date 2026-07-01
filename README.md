# CMPT 310 Group Project - Farm Animal Classification System
Our project idea is to train and develop an AI system that can correctly identify some common farm animals. This is technology that could potentially be useful to farmers who need automated livestock inventory and transit tracking. Picture this: you’re an owner of a large commercial farm in British Columbia. Every time a truckload of animals arrive, you need to count and sort the animals. Normally, a human worker would have to count and sort the animals manually. This method of counting is subject to human error and takes a long time as well. Instead of a farmworker manually counting every individual animal, we can install an overhead security camera over the unloading ramps to capture live feeds of incoming animals, feeding the frames into our AI system. Then, the system can automatically increment the number of animals for each type. Obviously, machine error will still exist, but we hope to minimize them. Implementing this system on an actual security camera is out of the scope of our project and this course. We will only be designing and training our system to identify farm animals based on images. Nonetheless, this scenario demonstrates how our system could theoretically be expanded upon to work in more practical applications such as the aforementioned scenario. 

## Inputs
Images of animals (cows, pigs, chickens, sheep, goats, turkeys, etc). 

## Outputs
Animal label (what kind of animal is depicted in this image?).
Tally of each animal type (ex. given 1000 images of animals, how many are cows? pigs? chickens?)

## Minimum Viable System 
The simplest working version of our system is to be able to interpret images of farm animals and to classify them through the K-Nearest Neighbours algorithm. We will start by limiting the classifications to just cows, pigs, chickens, sheep, goats, and turkeys. Our AI system in its minimal form will still be able to classify a smaller subset of farm animals. This will still accomplish our goals set out in the problem statement.

![alt text](diagram.png)

## Current prototype

The first working baseline uses:

1. OpenCV to resize every image to 64x64 pixels.
2. HOG features for edges and shape.
3. HSV colour histograms for colour information.
4. A scaled, distance-weighted K-Nearest Neighbours classifier.
5. A stratified train/test split, classification report, confusion matrix, and
   prediction tally.

The included dataset currently contains 2,000 images for each of five classes:
`chicken`, `cow`, `goat`, `horse`, and `sheep`. It does not currently contain
the proposed `pig` or `turkey` classes.

## Setup and run

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py train
```

The default run samples 500 images per class so that the KNN experiment is
reasonable on a laptop. Use the full 10,000-image dataset with:

```powershell
python main.py train --max-per-class 0
```

After training, classify one image or every image in a directory:

```powershell
python main.py predict path\to\image.png
python main.py predict path\to\folder
```

The trained model is written to `models/animal_knn.joblib`.

## Scope note

This prototype classifies one label per image. Counting several animals inside
one camera frame requires object detection (finding each animal's bounding
box), which is a useful later extension but is separate from this KNN image
classification milestone.
