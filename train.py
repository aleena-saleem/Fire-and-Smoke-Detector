from ultralytics import YOLO

import config


def train():
    model = YOLO(f"yolov8{config.MODEL_SIZE}.pt")

    results = model.train(
        data=config.DATA_YAML_PATH,
        epochs=config.EPOCHS,
        batch=config.BATCH_SIZE,
        imgsz=config.IMG_SIZE,
        device=config.DEVICE,
        workers=config.WORKERS,
        patience=config.PATIENCE,
        project=config.PROJECT_NAME,
        name=config.EXPERIMENT_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.7,           # more scale variation helps diffuse/varying-size smoke
        shear=0.0,
        perspective=0.0005,  # smoke has no fixed viewing angle
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,           # helps model learn soft/translucent smoke boundaries
        copy_paste=0.0,
    )

    print("Training completed successfully.")
    return results


if __name__ == "__main__":
    train()
