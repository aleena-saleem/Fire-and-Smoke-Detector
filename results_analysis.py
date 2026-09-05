import os
import pandas as pd
import matplotlib.pyplot as plt

import config


def summarize_results(results_file=None):
    results_file = results_file or os.path.join(config.RUN_DIR, "results.csv")
    df = pd.read_csv(results_file)

    print("Number of epochs:", len(df))
    print("Last epoch:", df["epoch"].iloc[-1])

    best_idx = df["metrics/mAP50(B)"].idxmax()
    print("Best epoch:", int(df.loc[best_idx, "epoch"]))
    print("Best mAP@0.5:", df.loc[best_idx, "metrics/mAP50(B)"])

    return df


def plot_training_metrics(results_file=None):
    results_file = results_file or os.path.join(config.RUN_DIR, "results.csv")

    if not os.path.exists(results_file):
        print("Results file not found!")
        return

    df = pd.read_csv(results_file)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics_to_plot = [
        ("train/box_loss", "Box Loss", "Training Box Loss"),
        ("train/cls_loss", "Class Loss", "Training Class Loss"),
        ("train/dfl_loss", "DFL Loss", "Training DFL Loss"),
        ("metrics/precision(B)", "Precision", "Precision"),
        ("metrics/recall(B)", "Recall", "Recall"),
        ("metrics/mAP50(B)", "mAP@0.5", "mAP@0.5"),
    ]

    for i, (metric, ylabel, title) in enumerate(metrics_to_plot):
        if metric in df.columns:
            axes[i].plot(df["epoch"], df[metric], linewidth=2)
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(title)
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=150)
    plt.show()

    best_map = df["metrics/mAP50(B)"].max()
    best_map_epoch = df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"]
    best_precision = df["metrics/precision(B)"].max()
    best_precision_epoch = df.loc[df["metrics/precision(B)"].idxmax(), "epoch"]
    best_recall = df["metrics/recall(B)"].max()
    best_recall_epoch = df.loc[df["metrics/recall(B)"].idxmax(), "epoch"]

    print("Best training results")
    print(f"mAP@0.5:   {best_map:.4f} (epoch {int(best_map_epoch)})")
    print(f"Precision: {best_precision:.4f} (epoch {int(best_precision_epoch)})")
    print(f"Recall:    {best_recall:.4f} (epoch {int(best_recall_epoch)})")


if __name__ == "__main__":
    summarize_results()
    plot_training_metrics()
