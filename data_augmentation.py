import os
import cv2
import numpy as np
from PIL import Image

import config


def count_day_night_images(img_dir):
    night_count = 0
    day_count = 0

    for fname in os.listdir(img_dir):
        img_path = os.path.join(img_dir, fname)
        try:
            img = Image.open(img_path).convert("L")
            avg_brightness = sum(img.getdata()) / (img.width * img.height)
            if avg_brightness < 60:
                night_count += 1
            else:
                day_count += 1
        except Exception:
            continue

    print(f"Likely night/dark images: {night_count}")
    print(f"Likely day/bright images: {day_count}")
    return night_count, day_count


def darken_image(img, factor=0.35):
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def generate_night_augmented_images(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    count = 0
    for fname in os.listdir(src_img_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(src_img_dir, fname)
        label_path = os.path.join(src_lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        dark_img = darken_image(img, factor=np.random.uniform(0.25, 0.45))

        new_name = f"dark_{fname}"
        cv2.imwrite(os.path.join(dst_img_dir, new_name), dark_img)

        # bounding boxes are unaffected by brightness changes, so the label is just copied
        with open(label_path, "r") as f:
            label_content = f.read()
        with open(os.path.join(dst_lbl_dir, f"dark_{os.path.splitext(fname)[0]}.txt"), "w") as f:
            f.write(label_content)

        count += 1

    print(f"Created {count} synthetic night-augmented images")
    return count


if __name__ == "__main__":
    count_day_night_images(config.TRAIN_IMG_PATH)
    generate_night_augmented_images(
        config.TRAIN_IMG_PATH,
        config.TRAIN_LBL_PATH,
        config.TRAIN_IMG_PATH,
        config.TRAIN_LBL_PATH,
    )
