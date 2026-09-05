# Fire & Smoke Detection using Deep Learning

A YOLOv8-based object detection system for identifying fire and smoke in images, developed as a computer vision research project.

## Overview

This project implements an object detector trained to localize fire and smoke regions in RGB images using YOLOv8. The system is evaluated quantitatively on a held-out test set and deployed as an interactive web application for inference on arbitrary user-uploaded images.


## Dataset

- **Classes:** Fire, Smoke
- **Format:** YOLO-format bounding box annotations (`train` / `valid` / `test` splits, each with `images/` and `labels/`)
- **Augmentation:** A custom night-condition augmentation pipeline that synthesizes darkened variants of training images to improve robustness to low-light fire/smoke scenes.

## Model

- **Architecture:** YOLOv8-m
- **Training:** 150 epochs, image size 640, AdamW optimizer, early-stopping patience of 45 epochs
- **Hardware:** Trained on GPU (Google Colab)

Full training configuration is in [`src/config.py`](src/config.py) and [`src/train.py`](src/train.py).

![Streamlit Demo](Results/Streamlit%20demo.png)

### Per-class breakdown

| Class | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Fire  | 83 | 64 | 41 | 0.565 | 0.669 |
| Smoke | 74 | 68 | 48 | 0.521 | 0.607 |

*(TP/FP/FN counts derived from the validation confusion matrix — see `results/confusion_matrix.png`.)*


## Error Analysis

The confusion matrix reveals a clear and consistent pattern:

- **Fire and Smoke are almost never confused with each other.** Cross-class misclassification is negligible (1 instance in each direction). When the model detects an object, it correctly identifies which of the two classes it belongs to.
- **The dominant source of error is confusion with the background class, in both directions:**
  - *False negatives:* 41 real Fire instances and 48 real Smoke instances were missed entirely and predicted as background.
  - *False positives:* 64 background regions were incorrectly predicted as Fire, and 68 as Smoke.


**Likely contributing factors:** visually ambiguous regions such as haze, backlighting, bright sky, and low-contrast smoke plumes that resemble background texture.

In other words, the model's class-discrimination ability (Fire vs. Smoke) is solid, but its object-vs-background boundary is comparatively weak.


## Repository Structure

```
├── app.py                      
├── requirements.txt             
├── src/
│   ├── config.py                - Paths and hyperparameters
│   ├── dataset_utils.py         -Dataset stats, data.yaml, annotation visualization
│   ├── data_augmentation.py     -Synthetic low-light augmentation
│   ├── train.py                 
│   ├── evaluate.py              
│   ├── results_analysis.py      
│   ├── visualize_plots.py       
│   ├── export_model.py          -Copy best weights to an output folder
│   └── predict.py               
└── results/
    ├── training_metrics.png
    ├── confusion_matrix.png
    └── README.md                 # Results summary and error analysis
```

## Setup

```bash
pip install -r requirements.txt
export DATASET_PATH=/path/to/your/Dataset   # expects train/valid/test subfolders
python src/dataset_utils.py
python src/train.py
python src/evaluate.py
python src/results_analysis.py
```

## Limitations

- Single-modality (RGB) detection only; no thermal, IR, or multi-sensor fusion.
- Precision on background false positives (~0.52–0.57 per class) means the current model is not suitable for unsupervised autonomous deployment without a human-in-the-loop review step.


## Author
Aleena Saleem
