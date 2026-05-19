from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .class_names import load_class_names
from .config import DEFAULT_CLASS_NAMES_PATH, DEFAULT_MODEL_PATH
from .inference import load_model, predict_frame, top_k_predictions
from .model_utils import validate_model_class_alignment


def parse_args():
    parser = argparse.ArgumentParser(description="Test static image inference")
    parser.add_argument("--image", type=Path, required=True, help="Path to test image")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.40)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    class_names = load_class_names(args.class_names)
    model = load_model(args.model_path)
    validate_model_class_alignment(model, class_names)

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise RuntimeError(f"Could not read image file: {args.image}")

    label, confidence, probs = predict_frame(model, frame, class_names)
    topk = top_k_predictions(probs, class_names, k=args.top_k)

    print("=== ASL Static Inference ===")
    print(f"image: {args.image}")
    print(f"model: {args.model_path}")
    print(f"predicted label: {label}")
    print(f"confidence: {confidence:.2%}")

    if confidence < args.threshold:
        print(f"status: uncertain (below threshold {args.threshold:.0%})")
    else:
        print("status: confident")

    print("top-k predictions:")
    for i, (name, conf) in enumerate(topk, start=1):
        print(f"  {i}. {name}: {conf:.2%}")


if __name__ == "__main__":
    main()
