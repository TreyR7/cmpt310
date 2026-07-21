from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.model import evaluate_model, load_dataset, load_model, predict_paths, save_model
from src.preprocess import IMAGE_EXTENSIONS
from src.tally import tally_predictions

app = Flask(__name__)
CORS(app)

DATASET_PATH = Path("dataset/animal_counting_dataset")
MODEL_PATH = Path("models/animal_knn.joblib")


@app.route("/api/train", methods=["POST"])
def train():
    features, labels = load_dataset(DATASET_PATH, max_per_class=500, seed=42)
    model = evaluate_model(features, labels, neighbors=5, test_size=0.2, seed=42)
    model.fit(features, labels)  # refit on everything, same as main.py
    save_model(model, MODEL_PATH)
    return jsonify({"status": "trained", "num_images": len(labels)})


@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODEL_PATH.exists():
        return jsonify({"error": "Model not trained yet"}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    model = load_model(MODEL_PATH)  # loaded fresh each call so it reflects latest training

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        saved_paths = []
        for file in files:
            if Path(file.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dest = tmp_path / file.filename
            file.save(dest)
            saved_paths.append(dest)

        if not saved_paths:
            return jsonify({"error": "No supported image files found"}), 400

        results = predict_paths(model, saved_paths)
        tally = tally_predictions(label for _, label in results)

        return jsonify({
            "predictions": [{"filename": p.name, "label": l} for p, l in results],
            "tally": tally,
        })


if __name__ == "__main__":
    app.run(debug=True, port=5000)