from __future__ import annotations

import argparse
from pathlib import Path

from src.class_names import save_class_names
from src.config import (
    DATASET_DIR,
    DEFAULT_CLASS_NAMES_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
)
from src.data_loader import build_data_augmentation, load_datasets
from src.evaluate import evaluate_model, save_evaluation_artifacts, save_training_curves
from src.train import build_model, compile_model, save_model, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASL MobileNetV2 model")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--class-names-out", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "assets" / "images")
    return parser.parse_args()


def main():
    args = parse_args()

    train_ds, val_ds, class_names = load_datasets(dataset_dir=args.dataset_dir)

    augmentation = build_data_augmentation()
    model = build_model(num_classes=len(class_names), augmentation=augmentation)
    compile_model(model)

    history = train_model(
        model,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=args.epochs,
    )

    eval_result = evaluate_model(model, val_ds, class_names)

    print("\n=== Classification Report ===")
    print(eval_result.classification_report_text)

    print("\n=== Confusion Matrix Shape ===")
    print(eval_result.confusion_matrix.shape)

    save_model(model, args.model_out)
    class_path = save_class_names(class_names, path=args.class_names_out)

    if args.save_artifacts:
        save_evaluation_artifacts(eval_result, class_names, out_dir=args.artifacts_dir)
        save_training_curves(history, out_dir=args.artifacts_dir)
        print(f"Artifacts saved to: {args.artifacts_dir}")

    print(f"\nModel saved to: {args.model_out}")
    print(f"Class names saved to: {class_path}")

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final val accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
