import argparse
from pathlib import Path

from src.model import (
    evaluate_model,
    load_dataset,
    load_model,
    predict_paths,
    save_model,
)
from src.preprocess import IMAGE_EXTENSIONS, image_paths
from src.tally import tally_predictions

DEFAULT_DATASET = Path("dataset/animal_counting_dataset")
DEFAULT_MODEL = Path("models/animal_knn.joblib")


def train(args):
    max_per_class = None if args.max_per_class == 0 else args.max_per_class
    features, labels = load_dataset(
        args.dataset, max_per_class=max_per_class, seed=args.seed
    )
    print(f"\nFeature matrix: {features.shape[0]} images x {features.shape[1]} features")
    model = evaluate_model(
        features,
        labels,
        neighbors=args.neighbors,
        test_size=args.test_size,
        seed=args.seed,
    )
    # The metrics above stay honest because they came from unseen holdout data.
    # Refit afterward so the saved model can learn from every available sample.
    model.fit(features, labels)
    save_model(model, args.model)
    print(f"\nRefit on all {len(labels)} images and saved model to {args.model}")


def predict(args):
    input_path = Path(args.input)
    if input_path.is_dir():
        paths = image_paths(input_path)
    elif input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        paths = [input_path]
    else:
        raise ValueError(f"Input must be a supported image or directory: {input_path}")

    results = predict_paths(load_model(args.model), paths)
    for path, label in results:
        print(f"{path.name}: {label}")

    print("\nTally")
    for label, count in tally_predictions(label for _, label in results).items():
        print(f"{label}: {count}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Classify farm animal images using OpenCV features and KNN."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and evaluate a model")
    train_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--neighbors", type=int, default=5)
    train_parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="images sampled per class; use 0 for the full dataset",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser(
        "predict", help="classify an image or directory and print a tally"
    )
    predict_parser.add_argument("input")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict_parser.set_defaults(func=predict)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
