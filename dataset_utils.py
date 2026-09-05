import os
import cv2
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config


def print_dataset_stats():
    print(f"Training images: {len(os.listdir(config.TRAIN_IMG_PATH))}")
    print(f"Training labels: {len(os.listdir(config.TRAIN_LBL_PATH))}")
    print(f"Validation images: {len(os.listdir(config.VAL_IMG_PATH))}")
    print(f"Validation labels: {len(os.listdir(config.VAL_LBL_PATH))}")
    print(f"Test images: {len(os.listdir(config.TEST_IMG_PATH))}")
    print(f"Test labels: {len(os.listdir(config.TEST_LBL_PATH))}")

    sample_img = os.listdir(config.TRAIN_IMG_PATH)[0]
    sample_img_path = os.path.join(config.TRAIN_IMG_PATH, sample_img)
    img = cv2.imread(sample_img_path)

    print(f"Sample image: {sample_img}")
    print(f"Sample image shape: {img.shape}")
    print(f"Class names: {config.CLASS_NAMES}")


def create_data_yaml():
    data_yaml = {
        "train": os.path.abspath(config.TRAIN_IMG_PATH),
        "val": os.path.abspath(config.VAL_IMG_PATH),
        "test": os.path.abspath(config.TEST_IMG_PATH),
        "nc": len(config.CLASS_NAMES),
        "names": config.CLASS_NAMES,
    }

    with open(config.DATA_YAML_PATH, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"data.yaml created at {config.DATA_YAML_PATH}")
    return config.DATA_YAML_PATH


def visualize_annotations(img_path, label_path, class_names=None):
    class_names = class_names or config.CLASS_NAMES
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center, y_center, box_w, box_h = (float(p) for p in parts[1:5])
            x_center, box_w = x_center * w, box_w * w
            y_center, box_h = y_center * h, box_h * h
            x = x_center - box_w / 2
            y = y_center - box_h / 2

            color = "red" if cls_id == 0 else "orange"
            rect = Rectangle((x, y), box_w, box_h, linewidth=3, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x, max(y - 5, 5), class_names[cls_id], color=color, fontsize=12,
                    weight="bold", backgroundcolor="white")

    ax.set_title("Sample Image with Annotations")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print_dataset_stats()
    create_data_yaml()
