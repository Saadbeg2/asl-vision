from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class EvaluationResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion_matrix: np.ndarray
    classification_report_text: str


def collect_predictions_and_labels(model: tf.keras.Model, dataset):
    """Collect aligned labels/predictions batch-by-batch.

    This avoids ordering mismatch bugs that can happen when labels and
    predictions are gathered through separate dataset traversals.
    """
    y_true_batches = []
    y_pred_batches = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true_batches.append(labels.numpy())
        y_pred_batches.append(preds)

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    return y_true, y_pred


def evaluate_model(model: tf.keras.Model, dataset, class_names: List[str]) -> EvaluationResult:
    y_true, y_pred = collect_predictions_and_labels(model, dataset)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    return EvaluationResult(
        y_true=y_true,
        y_pred=y_pred,
        confusion_matrix=cm,
        classification_report_text=report,
    )


def save_evaluation_artifacts(
    eval_result: EvaluationResult,
    class_names: List[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "evaluation_report.txt").write_text(eval_result.classification_report_text)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        eval_result.confusion_matrix,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("ASL Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def save_training_curves(history, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    if acc and val_acc:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, acc, label="train")
        plt.plot(epochs, val_acc, label="val")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_accuracy_curve.png", dpi=180)
        plt.close()

    if loss and val_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, label="train")
        plt.plot(epochs, val_loss, label="val")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_loss_curve.png", dpi=180)
        plt.close()
