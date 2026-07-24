from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from smart_livestock_gate.baseline.classifier import (
    evaluate_model,
    load_dataset,
    load_model,
    predict_paths,
    save_model,
)
from smart_livestock_gate.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from smart_livestock_gate.counting.tally import tally_predictions
from smart_livestock_gate.data.preprocessing import IMAGE_EXTENSIONS


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    @app.post("/api/train")
    def train():
        features, labels = load_dataset(
            DEFAULT_DATASET_PATH,
            max_per_class=500,
            seed=42,
        )
        model = evaluate_model(
            features,
            labels,
            neighbors=5,
            test_size=0.2,
            seed=42,
        )
        model.fit(features, labels)
        save_model(model, DEFAULT_MODEL_PATH)
        return jsonify({"status": "trained", "num_images": len(labels)})

    @app.post("/api/predict")
    def predict():
        if not DEFAULT_MODEL_PATH.exists():
            return jsonify({"error": "Model not trained yet"}), 400

        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No images uploaded"}), 400

        model = load_model(DEFAULT_MODEL_PATH)

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            saved_paths = []
            for file in files:
                filename = secure_filename(file.filename)
                is_supported = (
                    filename
                    and Path(filename).suffix.lower() in IMAGE_EXTENSIONS
                )
                if not is_supported:
                    continue
                destination = tmp_path / filename
                file.save(destination)
                saved_paths.append(destination)

            if not saved_paths:
                return jsonify({"error": "No supported image files found"}), 400

            results = predict_paths(model, saved_paths)
            tally = tally_predictions(label for _, label in results)

            return jsonify(
                {
                    "predictions": [
                        {"filename": path.name, "label": label}
                        for path, label in results
                    ],
                    "tally": tally,
                }
            )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
