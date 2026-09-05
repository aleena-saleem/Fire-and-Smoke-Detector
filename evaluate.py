from ultralytics import YOLO

import config


def evaluate(model_path=None):
    model_path = model_path or config.BEST_MODEL_PATH
    model = YOLO(model_path)
    print(f"Model loaded from: {model_path}")

    metrics = model.val(data=config.DATA_YAML_PATH, device=config.DEVICE)

    print("\nEvaluation metrics")
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")

    if metrics.box.mp > 0 and metrics.box.mr > 0:
        f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6)
        print(f"F1 Score:      {f1:.4f}")

    print("\nClass-wise mAP@0.5:0.95")
    if hasattr(metrics.box, "maps"):
        for i, class_name in enumerate(config.CLASS_NAMES):
            if i < len(metrics.box.maps):
                print(f"{class_name}: {metrics.box.maps[i]:.4f}")

    print("\nClass-wise mAP@0.5")
    if hasattr(metrics.box, "ap50"):
        for i, class_name in enumerate(config.CLASS_NAMES):
            if i < len(metrics.box.ap50):
                print(f"{class_name}: {metrics.box.ap50[i]:.4f}")
    else:
        print("Class-wise AP50 not available.")

    return metrics


if __name__ == "__main__":
    evaluate()
