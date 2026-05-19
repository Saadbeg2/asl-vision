from __future__ import annotations

import platform
import sys
from pathlib import Path

from .config import DEFAULT_CAMERA_INDEX, DEFAULT_CLASS_NAMES_PATH, DEFAULT_MODEL_PATH


def _check_tensorflow():
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        return True, f"TensorFlow {tf.__version__} (GPUs detected: {len(gpus)})"
    except Exception as e:
        return False, f"TensorFlow check failed: {e}"


def _check_opencv():
    try:
        import cv2

        return True, f"OpenCV {cv2.__version__}"
    except Exception as e:
        return False, f"OpenCV check failed: {e}"


def _check_webcam(camera_index: int = DEFAULT_CAMERA_INDEX):
    try:
        import cv2

        cap = cv2.VideoCapture(camera_index)
        opened = cap.isOpened()
        cap.release()
        if opened:
            return True, f"Webcam index {camera_index} opened successfully"
        return False, (
            f"Webcam index {camera_index} could not be opened "
            "(check permissions/device index)"
        )
    except Exception as e:
        return False, f"Webcam check failed: {e}"


def _check_model_file(model_path: Path = DEFAULT_MODEL_PATH):
    if model_path.exists():
        return True, f"Model found at {model_path}"
    return False, f"Model not found at {model_path}"




def _check_class_names_file(class_path: Path = DEFAULT_CLASS_NAMES_PATH):
    if class_path.exists():
        return True, f"Class names found at {class_path}"
    return False, f"Class names not found at {class_path}"
def run_system_check(camera_index: int = DEFAULT_CAMERA_INDEX):
    print("=== ASL Vision System Check ===")
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Interpreter: {sys.executable}")

    checks = [
        ("TensorFlow", _check_tensorflow),
        ("OpenCV", _check_opencv),
        ("Webcam", lambda: _check_webcam(camera_index=camera_index)),
        ("Model file", _check_model_file),
        ("Class names file", _check_class_names_file),
    ]

    all_ok = True
    for name, fn in checks:
        ok, msg = fn()
        status = "OK" if ok else "WARN"
        print(f"[{status}] {name}: {msg}")
        all_ok = all_ok and ok

    return all_ok
