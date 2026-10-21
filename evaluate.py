"""
Phase 5 - Evaluation and analysis for the report.

    python evaluate.py --run runs/e5_convnext_reg

Produces in figures/:
    results_table.csv        train/val/test metrics of every experiment in runs/
    model_comparison.png     bar chart comparing all experiments on the test set
    per_class_accuracy.png   distribution of per-class accuracy + the 15 hardest species
    confusion_matrix.png     200x200 confusion matrix of the chosen model
    confused_pairs.csv       the species pairs the model mixes up most often
    confused_examples.png    a mistake next to a training photo of the species it was mistaken for
    gradcam.png              Grad-CAM heatmaps: which pixels drove each prediction
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import confusion_matrix

from dataset import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD, get_dataloaders, get_transforms, load_annotations
from metrics import per_class_accuracy
from models import NUM_CLASSES, get_cam_target_layer, load_trained_model
from train import evaluate

FIG_DIR = "figures"

# Short labels for the charts, in experiment order
RUN_LABELS = {
    "e1_simple_cnn": "E1 Scratch CNN",
    "e2_resnet50_frozen": "E2 ResNet-50 frozen",
    "e3_resnet50_finetune": "E3 ResNet-50 fine-tuned",
    "e4_resnet50_reg": "E4 ResNet-50 + reg.",
    "e5_convnext_reg": "E5 ConvNeXt-T + reg.",
    "e6_convnext_384": "E6 ConvNeXt-T 384px",
}


# ---------------------------------------------------------------------------
# 1. Compare every experiment
# ---------------------------------------------------------------------------
def compare_experiments():
    """Collect runs/*/results.json into one table and one chart."""
    rows = []
    for path in sorted(glob.glob("runs/*/results.json")):
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "experiment": RUN_LABELS.get(r["name"], r["name"]),
            "params_M": round(r["params_total"] / 1e6, 2),
            "trainable_M": round(r["params_trainable"] / 1e6, 2),
            "best_epoch": r["best_epoch"],
            "train_top1": r["train"]["top1"],
            "val_top1": r["val"]["top1"],
            "test_top1": r["test"]["top1"],
            "test_per_class": r["test"]["per_class_avg"],
            "test_top5": r["test"]["top5"],
            # clean train accuracy minus validation accuracy = the overfitting gap
            "gap_train_val": r["train"]["top1"] - r["val"]["top1"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIG_DIR, "results_table.csv"), index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Grouped bar chart: three bars (Top-1, per-class, Top-5) per experiment
    metrics = [("test_top1", "Top-1"), ("test_per_class", "Avg per-class"), ("test_top5", "Top-5")]
    x = np.arange(len(df))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (col, label) in enumerate(metrics):
        bars = ax.bar(x + (i - 1) * width, df[col] * 100, width, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)  # print the value on each bar
    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment"], fontsize=9)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Test-set performance of each experiment")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Per-class accuracy
# ---------------------------------------------------------------------------
def analyse_per_class(logits, labels, class_names):
    """Histogram of per-class accuracy, and the hardest species."""
    acc = per_class_accuracy(logits, labels, NUM_CLASSES).numpy()  # T_i for every class
    df = pd.DataFrame({"label": range(NUM_CLASSES), "species": class_names, "accuracy": acc,
                       "test_images": np.bincount(labels.numpy(), minlength=NUM_CLASSES)})
    df.sort_values("accuracy").to_csv(os.path.join(FIG_DIR, "per_class_accuracy.csv"), index=False)

    perfect = (df["accuracy"] == 1.0).sum()
    print(f"\nClasses with 100% test accuracy: {perfect}/{NUM_CLASSES}")
    print("Hardest 10 classes:")
    print(df.sort_values("accuracy").head(10).to_string(index=False))

    worst = df.sort_values("accuracy").head(15).iloc[::-1]  # reversed so the worst is at the top
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.hist(df["accuracy"] * 100, bins=np.arange(0, 105, 10), edgecolor="white")
    ax1.set_xlabel("Per-class test accuracy (%)")
    ax1.set_ylabel("Number of classes")
    ax1.set_title("Distribution of per-class accuracy")
    ax2.barh(worst["species"], worst["accuracy"] * 100)
    ax2.set_xlabel("Test accuracy (%)")
    ax2.set_title("15 hardest species")
    ax2.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Confusion matrix and most-confused pairs
# ---------------------------------------------------------------------------
def analyse_confusions(preds, labels, class_names, top_n=10):
    """cm[i, j] = number of test images of true class i predicted as class j."""
    cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))

    # Normalise each row by the number of images of that class, so every row sums to 1
    # and classes with 4 or 8 test images are shown on the same scale
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalised confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Off-diagonal entries are mistakes. Zero the diagonal, then take the largest entries.
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    flat = np.argsort(errors, axis=None)[::-1][:top_n]       # indices of the largest values
    true_idx, pred_idx = np.unravel_index(flat, errors.shape)  # flat index -> (row, column)
    pairs = pd.DataFrame({
        "true_species": [class_names[i] for i in true_idx],
        "predicted_as": [class_names[j] for j in pred_idx],
        "count": errors[true_idx, pred_idx],
        "test_images_of_true": cm.sum(axis=1)[true_idx],
        "true_label": true_idx,
        "pred_label": pred_idx,
    })
    pairs.to_csv(os.path.join(FIG_DIR, "confused_pairs.csv"), index=False)
    print("\nMost confused pairs:")
    print(pairs.drop(columns=["true_label", "pred_label"]).to_string(index=False))
    return pairs


def plot_confused_examples(pairs, preds, labels, n=5):
    """For each top confused pair: the misclassified test photo next to a training photo
    of the species it was mistaken for. Shows *why* the mistakes are understandable."""
    test_df = load_annotations("test")
    train_df = load_annotations("train")
    preds, labels = preds.numpy(), labels.numpy()

    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    for row, (_, p) in zip(axes, pairs.head(n).iterrows()):
        # first test image of the true class that was predicted as the confused class
        idx = np.where((labels == p["true_label"]) & (preds == p["pred_label"]))[0][0]
        wrong_img = Image.open(os.path.join(DATA_DIR, "Test", test_df["file"][idx])).convert("RGB")
        # any training image of the predicted class, for comparison
        ref_file = train_df[train_df["label"] == p["pred_label"]]["file"].iloc[0]
        ref_img = Image.open(os.path.join(DATA_DIR, "Train", ref_file)).convert("RGB")

        row[0].imshow(wrong_img)
        row[0].set_title(f"Test: {p['true_species']}\npredicted: {p['predicted_as']}", fontsize=9, color="firebrick")
        row[1].imshow(ref_img)
        row[1].set_title(f"Training example of\n{p['predicted_as']}", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confused_examples.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Grad-CAM
# ---------------------------------------------------------------------------
def plot_gradcam(model, class_names, preds, labels, img_size, device, n_correct=6, n_wrong=6):
    """Grad-CAM: which regions of the image pushed the model towards its predicted class.

    It takes the gradient of the predicted class score with respect to the feature maps of the
    last conv stage, averages it per channel to get each channel's importance, and sums the
    feature maps weighted by that importance. ReLU keeps only regions that *increase* the
    score. The small map (7x7) is upsampled to the image size and drawn as a heatmap.
    """
    test_df = load_annotations("test")
    _, eval_tf = get_transforms(img_size)
    preds, labels = preds.numpy(), labels.numpy()

    rng = np.random.default_rng(0)  # fixed seed -> same images every run
    correct = rng.choice(np.where(preds == labels)[0], n_correct, replace=False)
    wrong = rng.choice(np.where(preds != labels)[0], n_wrong, replace=False)
    indices = list(correct) + list(wrong)

    cam = GradCAM(model=model, target_layers=[get_cam_target_layer(model)])
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    cols = 6
    rows = int(np.ceil(len(indices) / cols)) * 2  # each image takes two rows: photo, then heatmap
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows))
    for k, idx in enumerate(indices):
        img = Image.open(os.path.join(DATA_DIR, "Test", test_df["file"][idx])).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(device)  # same preprocessing as evaluation, batch of 1
        # Explain the class the model actually predicted
        heatmap = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(preds[idx]))])[0]
        rgb = (x[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)  # red = most important

        r, c = 2 * (k // cols), k % cols
        is_correct = preds[idx] == labels[idx]
        axes[r, c].imshow(rgb)
        axes[r, c].set_title(f"True: {class_names[labels[idx]]}", fontsize=7)
        axes[r + 1, c].imshow(overlay)
        axes[r + 1, c].set_title(f"Pred: {class_names[preds[idx]]}", fontsize=7,
                                 color="darkgreen" if is_correct else "firebrick")
        axes[r, c].axis("off")
        axes[r + 1, c].axis("off")
    fig.suptitle("Grad-CAM on test images (top two rows: correct, bottom two rows: mistakes)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gradcam.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/e6_convnext_384", help="run folder of the model to analyse")
    args = p.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== All experiments ===")
    compare_experiments()

    print(f"\n=== Detailed analysis of {args.run} ===")
    model, class_names, run_args = load_trained_model(os.path.join(args.run, "best.pth"), device)
    _, _, test_loader = get_dataloaders(batch_size=32, img_size=run_args["img_size"])
    _, test_m, logits, labels = evaluate(model, test_loader, device)
    print(f"Test top1 {test_m['top1']:.4f}  per-class {test_m['per_class_avg']:.4f}  top5 {test_m['top5']:.4f}")
    preds = logits.argmax(1)

    analyse_per_class(logits, labels, class_names)
    pairs = analyse_confusions(preds, labels, class_names)
    plot_confused_examples(pairs, preds, labels)
    plot_gradcam(model, class_names, preds, labels, run_args["img_size"], device)
    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
