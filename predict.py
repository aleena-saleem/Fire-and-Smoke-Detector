import argparse
import glob
import os
import random

from ultralytics import YOLO

import config


def predict_on_test_set(model_path, num_images=11, conf=0.5, iou=0.3, output_dir="predictions"):
    model = YOLO(model_path)
    image_files = glob.glob(os.path.join(config.TEST_IMG_PATH, "*"))

    random.seed(42)
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    print(f"Running inference on {len(selected_images)} test images...")

    results = model.predict(
        source=selected_images,
        conf=conf,
        iou=iou,
        agnostic_nms=True,
        imgsz=config.IMG_SIZE,
        save=True,
        project=output_dir,
        name="test_predictions",
        exist_ok=True,
    )

    print(f"Results saved to: {os.path.join(output_dir, 'test_predictions')}")
    return results


def predict_on_image(model_path, image_path, conf=0.25, output_dir="predictions"):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        imgsz=config.IMG_SIZE,
        save=True,
        project=output_dir,
        name="single_predictions",
        exist_ok=True,
    )

    print(f"Result saved to: {results[0].save_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.BEST_MODEL_PATH)
    parser.add_argument("--image", default=None, help="Path to a single image. If omitted, samples the test set.")
    parser.add_argument("--num-images", type=int, default=11)
    parser.add_argument("--output-dir", default="predictions")
    args = parser.parse_args()

    if args.image:
        predict_on_image(args.model, args.image, output_dir=args.output_dir)
    else:
        predict_on_test_set(args.model, num_images=args.num_images, output_dir=args.output_dir)
