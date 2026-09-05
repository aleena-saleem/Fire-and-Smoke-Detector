import os
import torch

# Root of the dataset (expects train/valid/test subfolders, each with images/ and labels/)
DATASET_PATH = os.environ.get("DATASET_PATH", "./Dataset")

TRAIN_IMG_PATH = os.path.join(DATASET_PATH, "train", "images")
TRAIN_LBL_PATH = os.path.join(DATASET_PATH, "train", "labels")
VAL_IMG_PATH = os.path.join(DATASET_PATH, "valid", "images")
VAL_LBL_PATH = os.path.join(DATASET_PATH, "valid", "labels")
TEST_IMG_PATH = os.path.join(DATASET_PATH, "test", "images")
TEST_LBL_PATH = os.path.join(DATASET_PATH, "test", "labels")

CLASS_NAMES = ["Fire", "Smoke"]
DATA_YAML_PATH = "./data.yaml"

# Training configuration
MODEL_SIZE = "m"
EPOCHS = 150
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WORKERS = 4
PATIENCE = 45
PROJECT_NAME = "fire_smoke_detector"
EXPERIMENT_NAME = "experiment_v2"

RUN_DIR = os.path.join("runs", "detect", PROJECT_NAME, EXPERIMENT_NAME)
BEST_MODEL_PATH = os.path.join(RUN_DIR, "weights", "best.pt")
