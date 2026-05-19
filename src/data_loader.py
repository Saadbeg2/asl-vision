from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from .config import BATCH_SIZE, DATASET_DIR, IMAGE_SIZE, SEED, VAL_SPLIT


def build_data_augmentation() -> tf.keras.Sequential:
    """Build lightweight augmentation for hand-sign classification.

    Note: We intentionally avoid horizontal flips because mirrored hands can
    change ASL semantics and may hurt label fidelity.
    """
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomBrightness(0.10),
        ],
        name="data_augmentation",
    )


def load_datasets(
    dataset_dir=DATASET_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    val_split: float = VAL_SPLIT,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
):
    """Load train/validation datasets from directory using a stable split."""
    dataset_dir = str(dataset_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle_val,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names
