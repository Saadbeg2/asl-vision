from __future__ import annotations

import argparse
from pathlib import Path

from src.class_names import load_class_names
from src.config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CLASS_NAMES_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
)
from src.inference import load_model
from src.model_utils import validate_model_class_alignment
from src.webcam_inference import run_webcam_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASL webcam inference")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hide-fps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Train first via run_train.py"
        )

    class_names = load_class_names(args.class_names)
    model = load_model(args.model_path)
    validate_model_class_alignment(model, class_names)

    try:
        run_webcam_inference(
            model,
            class_names=class_names,
            camera_index=args.camera_index,
            confidence_threshold=args.threshold,
            show_fps=not args.hide_fps,
            top_k=args.top_k,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        print("macOS tips:")
        print("1. Grant camera access to Terminal/VS Code in System Settings.")
        print("2. Close other apps that may lock the webcam.")
        print("3. Try a different --camera-index (e.g., 0 or 1).")
        print("4. Run from a local GUI session (not headless/remote).")
        print("5. Reinstall OpenCV wheel inside your virtualenv if GUI still fails.")


if __name__ == "__main__":
    main()
