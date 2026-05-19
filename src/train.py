from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .config import DEFAULT_LEARNING_RATE, INPUT_SHAPE


def build_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    freeze_base: bool = True,
    dropout_rate: float = 0.2,
    augmentation: tf.keras.Model | None = None,
) -> tf.keras.Model:
    """Build MobileNetV2 transfer-learning classifier."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    model_layers = []
    if augmentation is not None:
        model_layers.append(augmentation)

    model_layers.extend(
        [
            layers.Lambda(preprocess_input, name="mobilenet_preprocess"),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return models.Sequential(model_layers, name="asl_mobilenetv2")


def compile_model(model: tf.keras.Model, learning_rate: float = DEFAULT_LEARNING_RATE) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_model(
    model: tf.keras.Model,
    train_dataset,
    validation_dataset,
    epochs: int,
):
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
    )


def save_model(model: tf.keras.Model, model_path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
