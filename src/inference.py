from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf

from .config import IMAGE_SIZE


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_frame_for_model(frame_bgr, image_size: Tuple[int, int] = IMAGE_SIZE):
    """Resize BGR frame and build a model-ready float32 batch tensor.

    Training model includes MobileNetV2 preprocess_input as a model layer,
    so inference should pass raw pixel space (float32) images.
    """
    resized = tf.image.resize(frame_bgr, image_size)
    batch = tf.expand_dims(tf.cast(resized, tf.float32), axis=0)
    return batch


def predict_frame(
    model: tf.keras.Model,
    frame_bgr,
    class_names: List[str],
    image_size: Tuple[int, int] = IMAGE_SIZE,
):
    batch = preprocess_frame_for_model(frame_bgr, image_size=image_size)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label = class_names[pred_idx]
    return label, confidence, probs


def top_k_predictions(probs, class_names: List[str], k: int = 3):
    k = max(1, min(k, len(class_names)))
    idxs = np.argsort(probs)[::-1][:k]
    return [(class_names[int(i)], float(probs[int(i)])) for i in idxs]
