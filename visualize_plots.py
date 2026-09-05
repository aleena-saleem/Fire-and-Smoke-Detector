import os

import config

PLOT_FILES = [
    "confusion_matrix.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "PR_curve.png",
    "labels.jpg",
]


def show_available_plots(results_path=None):
    results_path = results_path or config.RUN_DIR

    print("Available plots:")
    for plot in PLOT_FILES:
        full_path = os.path.join(results_path, plot)
        status = "found" if os.path.exists(full_path) else "missing"
        print(f"{plot}: {status}")

    from PIL import Image
    import matplotlib.pyplot as plt

    for plot in PLOT_FILES:
        full_path = os.path.join(results_path, plot)
        if not os.path.exists(full_path):
            continue
        img = Image.open(full_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(plot)
        plt.show()


if __name__ == "__main__":
    show_available_plots()
