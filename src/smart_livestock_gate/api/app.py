from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, abort, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from smart_livestock_gate import __version__
from smart_livestock_gate.api.status import build_project_status
from smart_livestock_gate.api.tracking import (
    build_tracking_overview,
    render_sequence_frame,
)
from smart_livestock_gate.baseline.classifier import (
    evaluate_model,
    load_dataset,
    load_model,
    predict_paths,
    save_model,
)
from smart_livestock_gate.config import (
    CATTLE_DETECTOR_MODEL_PATH,
    CATTLE_DETECTOR_REPORT_PATH,
    CATTLE_EYE_VIEW_PATH,
    CATTLE_EYE_VIEW_REPORT_PATH,
    CATTLE_TRACKING_DEMO_PATH,
    CATTLE_TRACKING_DEMO_VIDEO_PATH,
    CATTLE_TRACKING_METRICS_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
)
from smart_livestock_gate.counting.tally import tally_predictions
from smart_livestock_gate.data.preprocessing import IMAGE_EXTENSIONS
from smart_livestock_gate.detection.detector import predict_image
from smart_livestock_gate.detection.examples import (
    find_detection_example,
    list_detection_examples,
)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "smart-livestock-gate",
                "version": __version__,
            }
        )

    @app.get("/api/status")
    def project_status():
        return jsonify(
            build_project_status(
                CATTLE_EYE_VIEW_PATH,
                CATTLE_EYE_VIEW_REPORT_PATH,
                CATTLE_DETECTOR_MODEL_PATH,
                CATTLE_DETECTOR_REPORT_PATH,
                CATTLE_TRACKING_METRICS_PATH,
            )
        )

    @app.get("/api/detection/examples")
    def detection_examples():
        examples = list_detection_examples(CATTLE_EYE_VIEW_PATH)
        return jsonify(
            {
                "examples": [example.to_dict() for example in examples],
                "model_ready": CATTLE_DETECTOR_MODEL_PATH.is_file(),
            }
        )

    @app.get("/api/detection/examples/<example_id>/image")
    def detection_example_image(example_id: str):
        example = find_detection_example(CATTLE_EYE_VIEW_PATH, example_id)
        if example is None:
            abort(404)
        return send_file(example.image_path, conditional=True, max_age=3600)

    @app.post("/api/detection/examples/<example_id>/predict")
    def predict_detection_example(example_id: str):
        example = find_detection_example(CATTLE_EYE_VIEW_PATH, example_id)
        if example is None:
            return jsonify({"error": "Unknown detection example"}), 404
        if not CATTLE_DETECTOR_MODEL_PATH.is_file():
            return jsonify({"error": "Cattle detector has not been trained"}), 409
        try:
            prediction = predict_image(
                CATTLE_DETECTOR_MODEL_PATH,
                example.image_path,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            app.logger.exception("Detection example inference failed")
            return jsonify({"error": str(error)}), 500
        return jsonify(prediction)

    @app.get("/api/tracking/overview")
    def tracking_overview():
        return jsonify(
            build_tracking_overview(
                CATTLE_TRACKING_METRICS_PATH,
                CATTLE_TRACKING_DEMO_PATH,
                CATTLE_TRACKING_DEMO_VIDEO_PATH,
            )
        )

    @app.get("/api/tracking/demo")
    def tracking_demo():
        if not CATTLE_TRACKING_DEMO_PATH.is_file():
            return jsonify({"error": "Tracking demo has not been generated"}), 404
        return send_file(CATTLE_TRACKING_DEMO_PATH, mimetype="application/json")

    @app.get("/api/tracking/frames/<sequence>/<int:frame_index>")
    def tracking_frame(sequence: str, frame_index: int):
        jpeg = render_sequence_frame(
            CATTLE_EYE_VIEW_PATH,
            sequence,
            frame_index,
            width=request.args.get("width", default=640, type=int),
        )
        if jpeg is None:
            return jsonify({"error": "Frame not available"}), 404
        response = app.response_class(jpeg, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    @app.get("/api/tracking/video")
    def tracking_video():
        if not CATTLE_TRACKING_DEMO_VIDEO_PATH.is_file():
            return jsonify({"error": "Annotated video has not been generated"}), 404
        # Force an attachment: the HTML "download" attribute is ignored on
        # cross-origin links (the client runs on a different port), and this
        # OpenCV mp4v file cannot be decoded by browsers for inline playback.
        return send_file(
            CATTLE_TRACKING_DEMO_VIDEO_PATH,
            mimetype="video/mp4",
            conditional=True,
            max_age=3600,
            as_attachment=True,
            download_name="cattle_tracking_annotated.mp4",
        )

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
