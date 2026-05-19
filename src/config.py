from pathlib import Path

# Project root is one level above src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "asl_dataset"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "asl_mobilenetv2.keras"
DEFAULT_CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.2

DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3

# Webcam defaults
DEFAULT_CAMERA_INDEX = 0
DISPLAY_WINDOW_NAME = "ASL Vision - Webcam Inference"
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
