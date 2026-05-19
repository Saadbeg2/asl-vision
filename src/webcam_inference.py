from __future__ import annotations

import platform
import time
from typing import List

import cv2

from .config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DISPLAY_WINDOW_NAME,
)
from .inference import predict_frame, top_k_predictions


def _open_capture(camera_index: int):
    # macOS generally uses AVFoundation backend.
    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(camera_index)
    return cap


def run_webcam_inference(
    model,
    class_names: List[str],
    camera_index: int = DEFAULT_CAMERA_INDEX,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    show_fps: bool = True,
    top_k: int = 3,
):
    cap = _open_capture(camera_index)

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Camera index: {camera_index}")

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions and camera index."
        )

    print("Webcam started. Press 'q' to quit.")
    last_ts = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read frame from webcam.")
                continue

            label, confidence, probs = predict_frame(model, frame, class_names)
            if confidence < confidence_threshold:
                label_text = f"uncertain ({confidence:.1%})"
                color = (0, 165, 255)
            else:
                label_text = f"{label} ({confidence:.1%})"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                label_text,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

            preds = top_k_predictions(probs, class_names, k=top_k)
            for i, (name, conf) in enumerate(preds):
                row = 62 + i * 24
                cv2.putText(
                    frame,
                    f"{i+1}. {name}: {conf:.1%}",
                    (16, row),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if show_fps:
                now = time.time()
                dt = max(now - last_ts, 1e-6)
                fps = 1.0 / dt
                last_ts = now
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (16, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(DISPLAY_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except cv2.error as e:
        raise RuntimeError(
            "OpenCV display error while rendering webcam window. "
            "On macOS, this is often caused by missing camera/screen permissions, "
            "a non-GUI session, or backend/runtime issues."
        ) from e
    finally:
        cap.release()
        cv2.destroyAllWindows()
